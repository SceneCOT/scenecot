import os
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger

from data.data_utils import pad_tensor
from model.build import MODULE_REGISTRY
from model.vision3d import VoxelQFormer
from model.transformers import CrossAttentionLayer, TransformerEncoderLayer, TransformerSpatialEncoderLayer, \
                               ObjectQFormerLayer, VoxelQFormerLayer, VoxelQFormerLLaVALayer
from model.utils import _init_weights_bert, calc_pairwise_locs, generate_fourier_features, get_mlp_head, layer_repeat

logger = get_logger(__name__)

# @MODULE_REGISTRY.register()
# class ObjVoxelQFormer(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.hidden_dim = cfg.hidden_dim
#         self.voxel_size = cfg.voxel_size

#         # indicator of target region
#         self.tgt_vox_embed = nn.Embedding(1, self.hidden_dim)

#         self.proj_3d = nn.Linear(cfg.feat_dim_3d[cfg.feat_type_3d], self.hidden_dim)
#         self.proj_2d = nn.Linear(cfg.feat_dim_2d[cfg.feat_type_2d], self.hidden_dim)
#         self.proj_txt = nn.Linear(cfg.feat_dim_txt, self.hidden_dim)

#         self.init_pos_embed(self.hidden_dim, cfg.pos_embed_type)

#         self.dropout_3d = cfg.dropout_3d
#         self.dropout_2d = cfg.dropout_2d

#         q_former_layer = VoxelQFormerLayer(
#             d_model=self.hidden_dim,
#             nhead=cfg.q_former.num_attention_heads,
#             dim_feedforward=cfg.q_former.dim_feedforward,
#             dropout=cfg.q_former.dropout,
#             activation=cfg.q_former.activation,
#         )
#         self.q_former = layer_repeat(q_former_layer, cfg.q_former.num_layers)

#         self.query_pos = nn.Embedding(cfg.num_queries, self.hidden_dim)

#         logger.info(f"Build 3D module VoxelQFormer: voxel_size={self.voxel_size}, "
#                     f"pos_embed_type={self.pos_embed_type}, num_queries={cfg.num_queries}")

#     @staticmethod
#     def fnv_hash_vec(arr):
#         """
#         FNV64-1A
#         """
#         assert arr.ndim == 2
#         # Floor first for negative coordinates
#         arr = arr.copy()
#         arr = arr.astype(np.uint64, copy=False)
#         hashed_arr = np.uint64(14695981039346656037) * np.ones(
#             arr.shape[0], dtype=np.uint64
#         )
#         for j in range(arr.shape[1]):
#             hashed_arr *= np.uint64(1099511628211)
#             hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
#         return hashed_arr

#     def voxelize(self, pcd_pos, feats=[]):
#         coord_continuous = pcd_pos / self.voxel_size
#         coord_discrete = torch.floor(coord_continuous).int()
#         coord_min = coord_discrete.min(0)[0]
#         coord_discrete -= coord_min

#         key = self.fnv_hash_vec(coord_discrete.detach().cpu().numpy())
#         idx_sort = np.argsort(key)
#         key_sort = key[idx_sort]
#         _, inverse, counts = np.unique(key_sort, return_inverse=True, return_counts=True)

#         # Compute voxel-wise positions and features in parallel
#         num_voxels = inverse.max() + 1  # Number of unique voxels
#         voxel_pos = torch.zeros(num_voxels, pcd_pos.shape[1], device=pcd_pos.device)
#         voxel_feats = [torch.zeros(num_voxels, feat.shape[1], device=feat.device) for feat in feats]

#         # Scatter the mean values for positions
#         inverse = torch.from_numpy(inverse).to(pcd_pos.device)
#         counts = torch.from_numpy(counts).unsqueeze(-1).to(pcd_pos.device)
#         pcd_pos = pcd_pos[idx_sort]
#         voxel_pos.index_add_(0, inverse, pcd_pos)
#         voxel_pos = voxel_pos / counts

#         # Scatter the mean values for features
#         for i, feat in enumerate(feats):
#             temp = torch.zeros_like(voxel_feats[i])
#             feat = feat[idx_sort]
#             temp.index_add_(0, inverse, feat)
#             voxel_feats[i] = temp / counts

#         return voxel_pos, voxel_feats

#     @staticmethod
#     def get_relat_pos(pos, agent_pos, agent_ori):
#         """
#         pos: tensor, (B, N, 3)
#         agent_pos: tensor, (B, 3)
#         agent_ori: tensor, (B, 2)
#         """
#         if pos.ndim == 2:
#             pos = pos[None, :, :]
#         if agent_pos.ndim == 1:
#             agent_pos = agent_pos[None, :]
#         if agent_ori.ndim == 1:
#             agent_ori = agent_ori[None, :]

#         pos_relat = pos - agent_pos[:, None, :]
#         ori_normalized = F.normalize(agent_ori, dim=-1)
#         x = ori_normalized[:, 0]
#         y = ori_normalized[:, 1]
#         rot_mat = torch.zeros(pos_relat.shape[0], 3, 3, device=pos_relat.device)
#         rot_mat[:, 0, 0] = y
#         rot_mat[:, 0, 1] = x
#         rot_mat[:, 1, 0] = -x
#         rot_mat[:, 1, 1] = y
#         rot_mat[:, 2, 2] = 1

#         pos_relat = torch.bmm(pos_relat, rot_mat)
#         return pos_relat

#     def init_pos_embed(self, dim, pos_embed_type='rope'):
#         assert pos_embed_type in ['rope', 'fourier'], f"VoxelQFormer: {pos_embed_type} not supported for pos_embed_type"
#         self.pos_embed_type = pos_embed_type
#         if self.pos_embed_type == 'rope':
#             # 3D position, ensure dim % 3 == 0
#             pos_embed_dim_per_axis = dim // 3
#             inv_freq = 1.0 / (10000 ** (torch.arange(0, pos_embed_dim_per_axis, 2) / pos_embed_dim_per_axis))   # [dim//6,]
#             inv_freq = torch.repeat_interleave(inv_freq, repeats=2)   # [dim//3,]
#             inv_freq = torch.stack([inv_freq, inv_freq, inv_freq], dim=0)   # [3, dim//3]
#             self.register_buffer('inv_freq', inv_freq, persistent=False)
#         elif self.pos_embed_type == 'fourier':
#             self.fourier_proj = nn.Linear(3, dim//2)
#             nn.init.normal_(self.fourier_proj.weight)
#             nn.init.constant_(self.fourier_proj.bias, 0)
#             self.fourier_mlp = get_mlp_head(dim, dim, dim, activation='gelu')
#         else:
#             raise NotImplementedError()

#     def add_pos_embed(self, voxel_pos, voxel_feat):
#         """
#         voxel_pos: tensor (B, N, 3)
#         voxel_feat: tensor (B, N, D)
#         """
#         if voxel_pos.ndim == 2:
#             voxel_pos = voxel_pos[None, ...]
#         if voxel_feat.ndim == 2:
#             voxel_feat = voxel_feat[None, ...]

#         batch_size, num_voxels = voxel_pos.shape[:2]
#         if self.pos_embed_type == 'rope':
#             voxel_pos = voxel_pos.unsqueeze(-1)   # [B, N, 3, 1]
#             rotary_factor = voxel_pos * self.inv_freq   # [B, N, 3, D//3]
#             rotary_cos = rotary_factor.cos().reshape(batch_size, num_voxels, -1)   # [B, N, D]
#             rotary_sin = rotary_factor.sin().reshape(batch_size, num_voxels, -1)   # [B, N, D]
#             voxel_feat_1 = torch.stack([-voxel_feat[:, :, 1::2], voxel_feat[:, :, ::2]], dim=-1)   # [B, N, D//2, 2]
#             voxel_feat_1 = voxel_feat_1.reshape(batch_size, num_voxels, -1)   # [B, N, D]
#             return voxel_feat * rotary_cos + voxel_feat_1 * rotary_sin
#         elif self.pos_embed_type == 'fourier':
#             latent = self.fourier_proj(voxel_pos)   # [B, N, D//2]
#             cos = latent.cos()
#             sin = latent.sin()
#             fourier_feat = torch.cat([cos, sin], dim=-1)   # [B, N, D]
#             fourier_feat /= sqrt(fourier_feat.shape[-1])
#             fourier_feat = self.fourier_mlp(fourier_feat)   # [B, N, D]
#             return voxel_feat + fourier_feat
#         else:
#             raise NotImplementedError()

#     def forward(self, data_dict):
#         """
#         data_dict requires keys:
#             scene_voxs: tensor, (B, N, 6)
#             vox_masks: (B, N)
#             vox_3d_fts: tensor, (B, N, 768)
#             vox_2d_fts: tensor, (B, N, 768)
#             vox_2d_fts_mask: tensor, (B,)
#             tgt_obj_mask: tensor, (B, N)
#             pos: tensor, (B, 3)
#             ori: tensor, (B, 2)
#             input_txt_embed: tensor, (B, T, D_2)
#             input_txt_mask: tensor, (B, T)
#         """
#         scene_voxs = data_dict['scene_voxs']
#         vox_3d_fts = data_dict['vox_3d_fts']
#         vox_2d_fts = data_dict['vox_2d_fts']
#         vox_2d_fts_mask = data_dict['vox_2d_fts_mask']
#         tgt_obj_mask = data_dict['tgt_obj_mask']
#         agent_pos = data_dict['pos']
#         agent_ori = data_dict['ori']
#         batch_size, num_voxels = scene_voxs.shape[:2]
#         # voxel_feats = []
#         # # per-scene process
#         # for i in range(batch_size):
#             # # 1. highlight target regions
#             # num_points = scene_pcds[i].shape[0]
#             # tgt_pcd_embed = torch.zeros(num_points, self.hidden_dim, device=self.tgt_pcd_embed.weight.device)
#             # tgt_pcd_embed[tgt_pcd_mask[i]] = self.tgt_pcd_embed.weight

#             # # 2. voxelization and pool features
#             # this_scene_voxel_pos, this_scene_voxel_feats = self.voxelize(
#             #     pcd_pos=scene_pcds[i][:, :3],
#             #     feats=[scene_3d_fts[i], scene_2d_fts[i], tgt_pcd_embed]
#             # )

#             # # 3. feature fuse
#             # this_scene_voxel_3d_fts = self.proj_3d(this_scene_voxel_feats[0])
#             # this_scene_voxel_2d_fts = self.proj_2d(this_scene_voxel_feats[1])
#             # this_scene_voxel_feats = this_scene_voxel_3d_fts + this_scene_voxel_2d_fts \
#             #                          + this_scene_voxel_feats[2]

#             # # 4. get relative positions
#             # this_scene_voxel_pos_relat = self.get_relat_pos(
#             #     this_scene_voxel_pos, agent_pos[i], agent_ori[i]
#             # )

#             # # 5. add pos embed
#             # voxel_feats.append(self.add_pos_embed(this_scene_voxel_feats, this_scene_voxel_pos_relat))

#         # # 6. padding
#         # num_voxels = [f.shape[0] for f in voxel_feats]
#         # max_voxels = max(num_voxels)
#         # voxel_feats = [pad_tensor(f, dim=0, max_len=max_voxels, pad=0) for f in voxel_feats]
#         # voxel_feats = torch.stack(voxel_feats, dim=0)   # [B, N, D]
#         # voxel_mask = [(torch.arange(max_voxels) >= n) for n in num_voxels]
#         # # nn.Transformer convention: 0 for valid and 1 for masked
#         # voxel_mask = torch.stack(voxel_mask, dim=0).to(voxel_feats.device)   # [B, N]

#         # 1. highlight target regions
#         tgt_vox_embed = torch.zeros(batch_size, num_voxels, self.hidden_dim, device=self.tgt_vox_embed.weight.device)
#         tgt_vox_embed[tgt_obj_mask] = self.tgt_vox_embed.weight

#         # 2. feature fuse
#         vox_3d_fts = self.proj_3d(vox_3d_fts) + tgt_vox_embed
#         vox_2d_fts = self.proj_2d(vox_2d_fts) + tgt_vox_embed

#         # 3. get relative positions
#         vox_pos_relat = self.get_relat_pos(
#             pos=scene_voxs[:, :, :3], agent_pos=agent_pos, agent_ori=agent_ori
#         )

#         # 4. add pos embed
#         vox_3d_fts = self.add_pos_embed(voxel_pos=vox_pos_relat, voxel_feat=vox_3d_fts)
#         vox_2d_fts = self.add_pos_embed(voxel_pos=vox_pos_relat, voxel_feat=vox_2d_fts)

#         # 5. generate 2D/3D dropout masks
#         if self.training:
#             vox_2d_fts_dropout_mask = torch.rand(batch_size, device=vox_2d_fts.device) > self.dropout_2d
#             vox_2d_fts_dropout_mask = vox_2d_fts_dropout_mask & vox_2d_fts_mask.bool()
#             vox_3d_fts_dropout_mask = torch.rand(batch_size, device=vox_3d_fts.device) > self.dropout_3d
#             vox_3d_fts_dropout_mask = vox_3d_fts_dropout_mask | ~vox_2d_fts_dropout_mask
#             assert (vox_3d_fts_dropout_mask | vox_2d_fts_dropout_mask).all()
#         else:
#             vox_2d_fts_dropout_mask = vox_2d_fts_mask
#             vox_3d_fts_dropout_mask = torch.ones(batch_size, dtype=torch.bool, device=vox_3d_fts.device)

#         # 6. Q-Former style querying
#         query = torch.zeros_like(self.query_pos.weight)
#         query = query.unsqueeze(0).repeat(batch_size, 1, 1)
#         txt_feat = self.proj_txt(data_dict['input_txt_embed'].float())
#         # prepare masks according to nn.Transformer convention
#         txt_mask = ~(data_dict['input_txt_mask'].bool())
#         vox_masks = ~(data_dict['vox_masks'].bool())
#         for layer in self.q_former:
#             query = layer(
#                 tgt=query, query_pos=self.query_pos.weight,
#                 memory_txt=txt_feat, memory_txt_key_padding_mask=txt_mask,
#                 memory_vox_3d=vox_3d_fts, memory_vox_2d=vox_2d_fts,
#                 memory_vox_key_padding_mask=vox_masks,
#                 memory_vox_3d_dropout_mask=vox_3d_fts_dropout_mask,
#                 memory_vox_2d_dropout_mask=vox_2d_fts_dropout_mask,
#             )[0]

#         data_dict['scene_tokens'] = query
#         data_dict['scene_masks'] = torch.ones(batch_size, query.shape[1]).bool()
#         return data_dict

@MODULE_REGISTRY.register()
class LEO2_ObjectCentric(VoxelQFormer):
    def __init__(self, cfg):
        super(VoxelQFormer, self).__init__()
        self.hidden_dim = cfg.hidden_dim
        self.obj_num = cfg.obj_num

        self.init_pos_embed(self.hidden_dim, cfg.pos_embed_type)

        logger.info(f"Build 3D module LEO2_ObjectCentric: obj_num={self.obj_num}, "
                    f"pos_embed_type={self.pos_embed_type}")
        
    def forward(self, data_dict):
        """
        data_dict requires keys:
            obj_locs: tensor, (B, N, 3)
            obj_2d_fts: tensor, (B, N, D)
            obj_masks: (B, N)
            pos: tensor, (B, 3)
            ori: tensor, (B, 2)
        """
        obj_locs = data_dict['obj_locs']
        obj_2d_fts = data_dict['obj_2d_fts']
        obj_masks = data_dict['obj_masks'].bool()
        agent_pos = data_dict['pos']
        agent_ori = data_dict['ori']

        obj_pos_relat = self.get_relat_pos(
            pos=obj_locs[:, :,:3], agent_pos=agent_pos, agent_ori=agent_ori
        )

        obj_2d_fts = self.add_pos_embed(voxel_pos=obj_pos_relat, voxel_feat=obj_2d_fts)

        # 6. truncate objects, naive implementation
        data_dict['obj_tokens'] = obj_2d_fts[:, :self.obj_num]
        data_dict['obj_masks'] = obj_masks[:, :self.obj_num]

        return data_dict