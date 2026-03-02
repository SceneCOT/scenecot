import os
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger

from data.data_utils import pad_tensor
from model.build import MODULE_REGISTRY
from model.pcd_backbone import PointcloudBackbone
from model.pq3d import Query3DUnified
from model.transformers import CrossAttentionLayer, TransformerEncoderLayer, TransformerSpatialEncoderLayer, \
                               ObjectQFormerLayer, VoxelQFormerLayer, VoxelQFormerLLaVALayer
from model.utils import _init_weights_bert, calc_pairwise_locs, generate_fourier_features, get_mlp_head, layer_repeat
from torch_scatter import scatter_mean
from data.data_utils import pad_tensors

logger = get_logger(__name__)


@MODULE_REGISTRY.register()
class ObjectQFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # pcd backbone
        self.obj_encoder = PointcloudBackbone(cfg.backbone)
        self.pcd_proj = nn.Linear(self.obj_encoder.out_dim, cfg.hidden_dim)

        self.use_img_feat = cfg.use_img_feat
        if self.use_img_feat:
            self.img_proj = nn.Linear(cfg.img_feat_dim, cfg.hidden_dim)

        self.use_vox_feat = cfg.use_vox_feat
        if self.use_vox_feat:
            self.vox_proj = nn.Linear(cfg.vox_feat_dim, cfg.hidden_dim)

        # 3D positional embedding
        self.pos_embed_type = cfg.pos_embed_type
        # self.init_pos_embed(cfg.hidden_dim, cfg.pos_embed_type)

        # spatial encoder
        self.use_spatial_attn = cfg.use_spatial_attn
        if self.use_spatial_attn:
            self.init_spatial_encoder(cfg)

        # indicator of target object
        self.tgt_obj_embed = nn.Embedding(1, cfg.hidden_dim)

        self.txt_proj = nn.Linear(cfg.txt_feat_dim, cfg.hidden_dim)
        q_former_layer = ObjectQFormerLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.q_former.num_attention_heads,
            dim_feedforward=cfg.q_former.dim_feedforward,
            dropout=cfg.q_former.dropout,
            activation=cfg.q_former.activation,
            use_img_feat=self.use_img_feat,
            use_vox_feat=self.use_vox_feat,
        )
        self.q_former = layer_repeat(q_former_layer, cfg.q_former.num_layers)

        self.query_pos = nn.Embedding(cfg.num_queries, cfg.hidden_dim)

        logger.info(f"Build 3D module: ObjectQFormer, use_img_feat={self.use_img_feat}, use_vox_feat={self.use_vox_feat}, "
                    f"pos_embed_type={self.pos_embed_type}, use_spatial_attn={self.use_spatial_attn}")

    def init_pos_embed(self, dim, pos_embed_type='rope'):
        assert pos_embed_type in ['rope', 'fourier'], f"ObjectQFormer: {pos_embed_type} not supported for pos_embed_type"
        self.pos_embed_type = pos_embed_type
        if self.pos_embed_type == 'rope':
            # 3D position, ensure input_feat_dim % 3 == 0
            pos_embed_dim_per_axis = dim // 3
            inv_freq = 1.0 / (10000 ** (torch.arange(0, pos_embed_dim_per_axis, 2) / pos_embed_dim_per_axis))   # [input_feat_dim//6,]
            inv_freq = torch.repeat_interleave(inv_freq, repeats=2)   # [input_feat_dim//3,]
            inv_freq = torch.stack([inv_freq, inv_freq, inv_freq], dim=0)   # [3, input_feat_dim//3]
            self.register_buffer('inv_freq', inv_freq, persistent=False)
        elif self.pos_embed_type == 'fourier':
            self.fourier_proj = nn.Linear(3, dim//2)
            nn.init.normal_(self.fourier_proj.weight)
            nn.init.constant_(self.fourier_proj.bias, 0)
            self.fourier_mlp = get_mlp_head(dim, dim, dim, activation='gelu')
        else:
            raise NotImplementedError()

    def build_spatial_encoder(self, cfg):
        spatial_encoder_layer = TransformerSpatialEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.spatial_encoder.num_attention_heads,
            dim_feedforward=cfg.spatial_encoder.dim_feedforward,
            dropout=cfg.spatial_encoder.dropout,
            activation=cfg.spatial_encoder.activation,
            spatial_dim=cfg.spatial_encoder.spatial_dim,
            spatial_multihead=cfg.spatial_encoder.spatial_multihead,
            spatial_attn_fusion=cfg.spatial_encoder.spatial_attn_fusion,
        )
        return layer_repeat(spatial_encoder_layer, cfg.spatial_encoder.num_layers)

    def init_spatial_encoder(self, cfg):
        self.spatial_encoder_pcd = self.build_spatial_encoder(cfg)
        self.spatial_encoder_pcd.apply(_init_weights_bert)
        if self.use_img_feat:
            self.spatial_encoder_img = self.build_spatial_encoder(cfg)
            self.spatial_encoder_img.apply(_init_weights_bert)
        if self.use_vox_feat:
            self.spatial_encoder_vox = self.build_spatial_encoder(cfg)
            self.spatial_encoder_vox.apply(_init_weights_bert)

        self.pairwise_rel_type = cfg.spatial_encoder.pairwise_rel_type
        self.spatial_dist_norm = cfg.spatial_encoder.spatial_dist_norm
        self.spatial_dim = cfg.spatial_encoder.spatial_dim
        self.obj_loc_encoding = cfg.spatial_encoder.obj_loc_encoding

        # location encoding
        if self.obj_loc_encoding in ['same_0', 'same_all']:
            num_loc_layers = 1
        elif self.obj_loc_encoding == 'diff_all':
            num_loc_layers = cfg.spatial_encoder.num_layers

        loc_layer = nn.Sequential(
            nn.Linear(cfg.spatial_encoder.dim_loc, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.loc_layers = layer_repeat(loc_layer, num_loc_layers)
        self.loc_layers.apply(_init_weights_bert)

    @staticmethod
    def get_relat_pos(obj_pos, position, orientation):
        raise NotImplementedError

    def add_pos_embed(self, obj_feat, position):
        """
        obj_feat: tensor (N, D)
        position: tensor (N, 3)
        """
        raise NotImplementedError()

    def forward(self, data_dict):
        """
        data_dict requires keys:
            obj_fts: (B, N, P, 6), xyz + rgb
            obj_fts_img: (B, N, D)
            obj_fts_vox: (B, N, D)
            obj_masks: (B, N), 1 valid and 0 masked
            obj_locs: (B, N, 6), xyz + whd
            anchor_loc: (B, 3)
            anchor_orientation: (B, C)
            tgt_obj_idx: (B,)
        """
        obj_fts_pcd = data_dict['obj_fts']
        obj_fts_img = data_dict['obj_fts_img'] if self.use_img_feat else None
        obj_fts_vox = data_dict['obj_fts_vox'] if self.use_vox_feat else None
        obj_masks = ~data_dict['obj_masks']   # flipped due to different convention of nn.Transformer
        obj_locs = data_dict['obj_locs']

        B = obj_fts_pcd.shape[0]
        obj_fts_pcd = self.obj_encoder(obj_fts_pcd)
        obj_fts_pcd = self.pcd_proj(obj_fts_pcd)
        if self.use_img_feat:
            obj_fts_img = self.img_proj(obj_fts_img)
        if self.use_vox_feat:
            obj_fts_vox = self.vox_proj(obj_fts_vox)

        # TODO: add pos_embed for each object
        # obj_pos = obj_locs[..., :3]
        # obj_pos_relat = self.get_relat_pos(obj_pos, data_dict['pos'], data_dict['ori'])
        # obj_fts_pcd = self.add_pos_embed(obj_fts_pcd, obj_pos_relat)
        # if self.use_img_feat:
        #     obj_fts_img = self.add_pos_embed(obj_fts_img, obj_pos_relat)
        # if self.use_vox_feat:
        #     obj_fts_vox = self.add_pos_embed(obj_fts_vox, obj_pos_relat)

        # spatial transformer, obj_fts go separately (pcd, img, vox)
        if self.use_spatial_attn:
            pairwise_locs = calc_pairwise_locs(
                obj_locs[:, :, :3],
                obj_locs[:, :, 3:],
                pairwise_rel_type=self.pairwise_rel_type,
                spatial_dist_norm=self.spatial_dist_norm,
                spatial_dim=self.spatial_dim,
            )

            for i, layer in enumerate(self.spatial_encoder_pcd):
                if self.obj_loc_encoding == 'diff_all':
                    query_pos = self.loc_layers[i](obj_locs)
                else:
                    query_pos = self.loc_layers[0](obj_locs)

                if not (self.obj_loc_encoding == 'same_0' and i > 0):
                    obj_fts_pcd = obj_fts_pcd + query_pos
                    if self.use_img_feat:
                        obj_fts_img = obj_fts_img + query_pos
                    if self.use_vox_feat:
                        obj_fts_vox = obj_fts_vox + query_pos

                obj_fts_pcd = layer(obj_fts_pcd, pairwise_locs, tgt_key_padding_mask=obj_masks)[0]
                if self.use_img_feat:
                    obj_fts_img = self.spatial_encoder_img[i](
                        obj_fts_img, pairwise_locs, tgt_key_padding_mask=obj_masks
                    )[0]
                if self.use_vox_feat:
                    obj_fts_vox = self.spatial_encoder_vox[i](
                        obj_fts_vox, pairwise_locs, tgt_key_padding_mask=obj_masks
                    )[0]

        # object captioning
        scatter_idx_batch = []
        scatter_idx_obj = []
        for i, tgt_obj_idx in enumerate(data_dict['tgt_obj_idx']):
            if tgt_obj_idx >= 0:
                scatter_idx_batch.append(i)
                scatter_idx_obj.append(tgt_obj_idx)

        obj_fts_pcd[scatter_idx_batch, scatter_idx_obj] += self.tgt_obj_embed.weight
        if self.use_img_feat:
            obj_fts_img[scatter_idx_batch, scatter_idx_obj] += self.tgt_obj_embed.weight
        if self.use_vox_feat:
            obj_fts_vox[scatter_idx_batch, scatter_idx_obj] += self.tgt_obj_embed.weight

        txt_feat = self.txt_proj(data_dict['input_txt_embed'].float())
        txt_mask = ~data_dict['input_txt_mask'].bool()   # nn.Transformer convention
        query = torch.zeros_like(self.query_pos.weight)
        query = query.unsqueeze(0).repeat(B, 1, 1)
        for layer in self.q_former:
            query = layer(
                tgt=query, memory_txt=txt_feat, memory_pcd=obj_fts_pcd,
                memory_img=obj_fts_img, memory_vox=obj_fts_vox,
                memory_txt_key_padding_mask=txt_mask,
                memory_pcd_key_padding_mask=obj_masks,
                memory_img_key_padding_mask=obj_masks,
                memory_vox_key_padding_mask=obj_masks,
                query_pos=self.query_pos.weight
            )[0]

        data_dict['scene_tokens'] = query
        data_dict['obj_masks'] = torch.ones(B, query.shape[1]).bool()
        return data_dict


@MODULE_REGISTRY.register()
class PQ3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_embodied_token = cfg.use_embodied_token   # embodied token
        hidden_dim = cfg.hidden_dim

        # backbone
        self.query_cross_encoder = Query3DUnified(cfg.query3d_cfg)

        # embodied token
        if self.use_embodied_token:
            self.anchor_feat = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.anchor_size = nn.Parameter(torch.ones(1, 1, 3))
            self.orient_encoder = nn.Linear(cfg.fourier_size, hidden_dim)
            nn.init.normal_(self.anchor_feat, std=0.02)

        self.obj_type_embed = nn.Embedding(2, hidden_dim)

        logger.info("Build 3D module: PQ3D")
        if cfg.query3d_pretrain_path is not None and os.path.exists(cfg.query3d_pretrain_path):
            info = self.query_cross_encoder.load_state_dict(torch.load(cfg.query3d_pretrain_path, weights_only=True), strict=False)
            logger.info(f"Load pretrained weights from {cfg.query3d_pretrain_path}: {info}")

    @property
    def device(self):
        return list(self.parameters())[0].device

    def adapt_data_dict(self, data_dict):
        data_dict['mv_seg_fts'] = data_dict['obj_fts_img'].clone()
        data_dict['mv_seg_pad_masks'] = data_dict['obj_masks'].clone()
        data_dict['voxel_seg_fts'] = data_dict['obj_fts_vox'].clone()
        data_dict['voxel_seg_pad_masks'] = data_dict['obj_masks'].clone()
        data_dict['pc_seg_fts'] = data_dict['obj_fts'].clone()
        data_dict['pc_seg_pad_masks'] = data_dict['obj_masks'].clone()
        data_dict['query_locs'] = data_dict['obj_locs'].clone()
        data_dict['query_pad_masks'] = data_dict['obj_masks'].clone()
        data_dict['coord_min'] = data_dict['obj_locs'][..., :3].min(1)[0]   # (B, N, 3)
        data_dict['coord_max'] = data_dict['obj_locs'][..., :3].max(1)[0]   # (B, N, 3)
        data_dict['seg_center'] = data_dict['obj_locs'].clone()
        data_dict['seg_pad_masks'] = data_dict['obj_masks'].clone()
        return data_dict

    def forward(self, data_dict):
        data_dict = self.adapt_data_dict(data_dict)
        obj_feats = self.query_cross_encoder(data_dict)
        obj_masks = ~data_dict['obj_masks']   # flipped due to different convention of TransformerEncoder

        B, N = obj_feats.shape[:2]
        device = obj_feats.device

        obj_type_ids = torch.zeros((B, N), dtype=torch.long, device=device)
        obj_type_embeds = self.obj_type_embed(obj_type_ids)

        if self.use_embodied_token:
            # anchor feature
            anchor_orient = data_dict['anchor_orientation'].unsqueeze(1)
            anchor_orient_feat = self.orient_encoder(generate_fourier_features(anchor_orient))
            anchor_feat = self.anchor_feat + anchor_orient_feat
            anchor_mask = torch.zeros((B, 1), dtype=bool, device=device)

            # anchor type
            anchor_type_id = torch.ones((B, 1), dtype=torch.long, device=device)
            anchor_type_embed = self.obj_type_embed(anchor_type_id)

            # fuse anchor and objs
            all_obj_feats = torch.cat([anchor_feat, obj_feats], dim=1)
            all_obj_masks = torch.cat((anchor_mask, obj_masks), dim=1)

            all_obj_type_embeds = torch.cat((anchor_type_embed, obj_type_embeds), dim=1)

        else:
            all_obj_feats = obj_feats
            all_obj_masks = obj_masks

            all_obj_type_embeds = obj_type_embeds

        all_obj_feats = all_obj_feats + all_obj_type_embeds

        data_dict['scene_tokens'] = all_obj_feats
        data_dict['obj_masks'] = ~all_obj_masks

        return data_dict


@MODULE_REGISTRY.register()
class VoxelQFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.voxel_size = cfg.voxel_size

        # indicator of target region
        self.tgt_vox_embed = nn.Embedding(1, self.hidden_dim)

        self.proj_3d = nn.Linear(cfg.feat_dim_3d[cfg.feat_type_3d], self.hidden_dim)
        self.proj_2d = nn.Linear(cfg.feat_dim_2d[cfg.feat_type_2d], self.hidden_dim)
        self.proj_txt = nn.Linear(cfg.feat_dim_txt, self.hidden_dim)

        self.init_pos_embed(self.hidden_dim, cfg.pos_embed_type)

        self.dropout_3d = cfg.dropout_3d
        self.dropout_2d = cfg.dropout_2d

        q_former_layer = VoxelQFormerLayer(
            d_model=self.hidden_dim,
            nhead=cfg.q_former.num_attention_heads,
            dim_feedforward=cfg.q_former.dim_feedforward,
            dropout=cfg.q_former.dropout,
            activation=cfg.q_former.activation,
        )
        self.q_former = layer_repeat(q_former_layer, cfg.q_former.num_layers)

        self.query_pos = nn.Embedding(cfg.num_queries, self.hidden_dim)

        logger.info(f"Build 3D module VoxelQFormer: voxel_size={self.voxel_size}, "
                    f"pos_embed_type={self.pos_embed_type}, num_queries={cfg.num_queries}")

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    def voxelize(self, pcd_pos, feats=[]):
        coord_continuous = pcd_pos / self.voxel_size
        coord_discrete = torch.floor(coord_continuous).int()
        coord_min = coord_discrete.min(0)[0]
        coord_discrete -= coord_min

        key = self.fnv_hash_vec(coord_discrete.detach().cpu().numpy())
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, counts = np.unique(key_sort, return_inverse=True, return_counts=True)

        # Compute voxel-wise positions and features in parallel
        num_voxels = inverse.max() + 1  # Number of unique voxels
        voxel_pos = torch.zeros(num_voxels, pcd_pos.shape[1], device=pcd_pos.device)
        voxel_feats = [torch.zeros(num_voxels, feat.shape[1], device=feat.device) for feat in feats]

        # Scatter the mean values for positions
        inverse = torch.from_numpy(inverse).to(pcd_pos.device)
        counts = torch.from_numpy(counts).unsqueeze(-1).to(pcd_pos.device)
        pcd_pos = pcd_pos[idx_sort]
        voxel_pos.index_add_(0, inverse, pcd_pos)
        voxel_pos = voxel_pos / counts

        # Scatter the mean values for features
        for i, feat in enumerate(feats):
            temp = torch.zeros_like(voxel_feats[i])
            feat = feat[idx_sort]
            temp.index_add_(0, inverse, feat)
            voxel_feats[i] = temp / counts

        return voxel_pos, voxel_feats

    @staticmethod
    def get_relat_pos(pos, agent_pos, agent_ori):
        """
        pos: tensor, (B, N, 3)
        agent_pos: tensor, (B, 3)
        agent_ori: tensor, (B, 2)
        """
        if pos.ndim == 2:
            pos = pos[None, :, :]
        if agent_pos.ndim == 1:
            agent_pos = agent_pos[None, :]
        if agent_ori.ndim == 1:
            agent_ori = agent_ori[None, :]

        pos_relat = pos - agent_pos[:, None, :]
        ori_normalized = F.normalize(agent_ori, dim=-1)
        x = ori_normalized[:, 0]
        y = ori_normalized[:, 1]
        rot_mat = torch.zeros(pos_relat.shape[0], 3, 3, device=pos_relat.device)
        rot_mat[:, 0, 0] = y
        rot_mat[:, 0, 1] = x
        rot_mat[:, 1, 0] = -x
        rot_mat[:, 1, 1] = y
        rot_mat[:, 2, 2] = 1

        pos_relat = torch.bmm(pos_relat, rot_mat)
        return pos_relat

    def init_pos_embed(self, dim, pos_embed_type='rope'):
        assert pos_embed_type in ['rope', 'fourier'], f"VoxelQFormer: {pos_embed_type} not supported for pos_embed_type"
        self.pos_embed_type = pos_embed_type
        if self.pos_embed_type == 'rope':
            # 3D position, ensure dim % 3 == 0
            pos_embed_dim_per_axis = dim // 3
            inv_freq = 1.0 / (10000 ** (torch.arange(0, pos_embed_dim_per_axis, 2) / pos_embed_dim_per_axis))   # [dim//6,]
            inv_freq = torch.repeat_interleave(inv_freq, repeats=2)   # [dim//3,]
            inv_freq = torch.stack([inv_freq, inv_freq, inv_freq], dim=0)   # [3, dim//3]
            self.register_buffer('inv_freq', inv_freq, persistent=False)
        elif self.pos_embed_type == 'fourier':
            self.fourier_proj = nn.Linear(3, dim//2)
            nn.init.normal_(self.fourier_proj.weight)
            nn.init.constant_(self.fourier_proj.bias, 0)
            self.fourier_mlp = get_mlp_head(dim, dim, dim, activation='gelu')
        else:
            raise NotImplementedError()

    def add_pos_embed(self, voxel_pos, voxel_feat):
        """
        voxel_pos: tensor (B, N, 3)
        voxel_feat: tensor (B, N, D)
        """
        if voxel_pos.ndim == 2:
            voxel_pos = voxel_pos[None, ...]
        if voxel_feat.ndim == 2:
            voxel_feat = voxel_feat[None, ...]

        batch_size, num_voxels = voxel_pos.shape[:2]
        if self.pos_embed_type == 'rope':
            voxel_pos = voxel_pos.unsqueeze(-1)   # [B, N, 3, 1]
            rotary_factor = voxel_pos * self.inv_freq   # [B, N, 3, D//3]
            rotary_cos = rotary_factor.cos().reshape(batch_size, num_voxels, -1)   # [B, N, D]
            rotary_sin = rotary_factor.sin().reshape(batch_size, num_voxels, -1)   # [B, N, D]
            voxel_feat_1 = torch.stack([-voxel_feat[:, :, 1::2], voxel_feat[:, :, ::2]], dim=-1)   # [B, N, D//2, 2]
            voxel_feat_1 = voxel_feat_1.reshape(batch_size, num_voxels, -1)   # [B, N, D]
            return voxel_feat * rotary_cos + voxel_feat_1 * rotary_sin
        elif self.pos_embed_type == 'fourier':
            latent = self.fourier_proj(voxel_pos)   # [B, N, D//2]
            cos = latent.cos()
            sin = latent.sin()
            fourier_feat = torch.cat([cos, sin], dim=-1)   # [B, N, D]
            fourier_feat /= sqrt(fourier_feat.shape[-1])
            fourier_feat = self.fourier_mlp(fourier_feat)   # [B, N, D]
            return voxel_feat + fourier_feat
        else:
            raise NotImplementedError()

    def forward(self, data_dict):
        """
        data_dict requires keys:
            scene_voxs: tensor, (B, N, 6)
            vox_masks: (B, N)
            vox_3d_fts: tensor, (B, N, 768)
            vox_2d_fts: tensor, (B, N, 768)
            vox_2d_fts_mask: tensor, (B,)
            tgt_obj_mask: tensor, (B, N)
            pos: tensor, (B, 3)
            ori: tensor, (B, 2)
            input_txt_embed: tensor, (B, T, D_2)
            input_txt_mask: tensor, (B, T)
        """
        scene_voxs = data_dict['scene_voxs']
        vox_3d_fts = data_dict['vox_3d_fts']
        vox_2d_fts = data_dict['vox_2d_fts']
        vox_2d_fts_mask = data_dict['vox_2d_fts_mask']
        tgt_obj_mask = data_dict['tgt_obj_mask']
        agent_pos = data_dict['pos']
        agent_ori = data_dict['ori']
        batch_size, num_voxels = scene_voxs.shape[:2]
        # voxel_feats = []
        # # per-scene process
        # for i in range(batch_size):
            # # 1. highlight target regions
            # num_points = scene_pcds[i].shape[0]
            # tgt_pcd_embed = torch.zeros(num_points, self.hidden_dim, device=self.tgt_pcd_embed.weight.device)
            # tgt_pcd_embed[tgt_pcd_mask[i]] = self.tgt_pcd_embed.weight

            # # 2. voxelization and pool features
            # this_scene_voxel_pos, this_scene_voxel_feats = self.voxelize(
            #     pcd_pos=scene_pcds[i][:, :3],
            #     feats=[scene_3d_fts[i], scene_2d_fts[i], tgt_pcd_embed]
            # )

            # # 3. feature fuse
            # this_scene_voxel_3d_fts = self.proj_3d(this_scene_voxel_feats[0])
            # this_scene_voxel_2d_fts = self.proj_2d(this_scene_voxel_feats[1])
            # this_scene_voxel_feats = this_scene_voxel_3d_fts + this_scene_voxel_2d_fts \
            #                          + this_scene_voxel_feats[2]

            # # 4. get relative positions
            # this_scene_voxel_pos_relat = self.get_relat_pos(
            #     this_scene_voxel_pos, agent_pos[i], agent_ori[i]
            # )

            # # 5. add pos embed
            # voxel_feats.append(self.add_pos_embed(this_scene_voxel_feats, this_scene_voxel_pos_relat))

        # # 6. padding
        # num_voxels = [f.shape[0] for f in voxel_feats]
        # max_voxels = max(num_voxels)
        # voxel_feats = [pad_tensor(f, dim=0, max_len=max_voxels, pad=0) for f in voxel_feats]
        # voxel_feats = torch.stack(voxel_feats, dim=0)   # [B, N, D]
        # voxel_mask = [(torch.arange(max_voxels) >= n) for n in num_voxels]
        # # nn.Transformer convention: 0 for valid and 1 for masked
        # voxel_mask = torch.stack(voxel_mask, dim=0).to(voxel_feats.device)   # [B, N]

        # 1. highlight target regions
        tgt_vox_embed = torch.zeros(batch_size, num_voxels, self.hidden_dim, device=self.tgt_vox_embed.weight.device)
        tgt_vox_embed[tgt_obj_mask] = self.tgt_vox_embed.weight

        # 2. feature fuse
        vox_3d_fts = self.proj_3d(vox_3d_fts) + tgt_vox_embed
        vox_2d_fts = self.proj_2d(vox_2d_fts) + tgt_vox_embed

        # 3. get relative positions
        vox_pos_relat = self.get_relat_pos(
            pos=scene_voxs[:, :, :3], agent_pos=agent_pos, agent_ori=agent_ori
        )

        # 4. add pos embed
        vox_3d_fts = self.add_pos_embed(voxel_pos=vox_pos_relat, voxel_feat=vox_3d_fts)
        vox_2d_fts = self.add_pos_embed(voxel_pos=vox_pos_relat, voxel_feat=vox_2d_fts)

        # 5. generate 2D/3D dropout masks
        if self.training:
            vox_2d_fts_dropout_mask = torch.rand(batch_size, device=vox_2d_fts.device) > self.dropout_2d
            vox_2d_fts_dropout_mask = vox_2d_fts_dropout_mask & vox_2d_fts_mask.bool()
            vox_3d_fts_dropout_mask = torch.rand(batch_size, device=vox_3d_fts.device) > self.dropout_3d
            vox_3d_fts_dropout_mask = vox_3d_fts_dropout_mask | ~vox_2d_fts_dropout_mask
            assert (vox_3d_fts_dropout_mask | vox_2d_fts_dropout_mask).all()
        else:
            vox_2d_fts_dropout_mask = vox_2d_fts_mask
            vox_3d_fts_dropout_mask = torch.ones(batch_size, dtype=torch.bool, device=vox_3d_fts.device)

        # 6. Q-Former style querying
        query = torch.zeros_like(self.query_pos.weight)
        query = query.unsqueeze(0).repeat(batch_size, 1, 1)
        txt_feat = self.proj_txt(data_dict['input_txt_embed'].float())
        # prepare masks according to nn.Transformer convention
        txt_mask = ~(data_dict['input_txt_mask'].bool())
        vox_masks = ~(data_dict['vox_masks'].bool())
        for layer in self.q_former:
            query = layer(
                tgt=query, query_pos=self.query_pos.weight,
                memory_txt=txt_feat, memory_txt_key_padding_mask=txt_mask,
                memory_vox_3d=vox_3d_fts, memory_vox_2d=vox_2d_fts,
                memory_vox_key_padding_mask=vox_masks,
                memory_vox_3d_dropout_mask=vox_3d_fts_dropout_mask,
                memory_vox_2d_dropout_mask=vox_2d_fts_dropout_mask,
            )[0]

        data_dict['scene_tokens'] = query
        data_dict['scene_masks'] = torch.ones(batch_size, query.shape[1]).bool()
        return data_dict


@MODULE_REGISTRY.register()
class VoxelQFormerLLaVA(VoxelQFormer):
    def __init__(self, cfg):
        super(VoxelQFormer, self).__init__()
        self.hidden_dim = cfg.feat_dim_2d[cfg.feat_type_2d]
        self.voxel_size = cfg.voxel_size

        # indicator of target region
        self.tgt_vox_embed = nn.Embedding(1, self.hidden_dim)

        self.init_pos_embed(self.hidden_dim, cfg.pos_embed_type)

        self.resample = cfg.resample
        if self.resample:
            q_former_layer = VoxelQFormerLLaVALayer(
                d_model=self.hidden_dim,
                nhead=cfg.q_former.num_attention_heads,
                dim_feedforward=cfg.q_former.dim_feedforward,
                dropout=cfg.q_former.dropout,
                activation=cfg.q_former.activation,
                feat_dim_txt=cfg.feat_dim_txt,
            )
            self.q_former = layer_repeat(q_former_layer, cfg.q_former.num_layers)
            self.query_pos = nn.Embedding(cfg.num_queries, self.hidden_dim)
        else:
            self.max_num_voxels = cfg.max_num_voxels

        logger.info(f"Build 3D module VoxelQFormerLLaVA: voxel_size={self.voxel_size}, "
                    f"pos_embed_type={self.pos_embed_type}, resample={cfg.resample}")

    def forward(self, data_dict):
        """
        data_dict requires keys:
            scene_voxs: tensor, (B, N, 6)
            vox_masks: (B, N)
            vox_3d_fts: tensor, (B, N, 768)
            vox_2d_fts: tensor, (B, N, 768)
            vox_2d_fts_mask: tensor, (B,)
            tgt_obj_mask: tensor, (B, N)
            pos: tensor, (B, 3)
            ori: tensor, (B, 2)
            input_txt_embed: tensor, (B, T, D_2)
            input_txt_mask: tensor, (B, T)
        """
        scene_voxs = data_dict['scene_voxs']
        vox_masks = data_dict['vox_masks'].bool()
        vox_2d_fts = data_dict['vox_2d_fts']
        tgt_obj_mask = data_dict['tgt_obj_mask']
        agent_pos = data_dict['pos']
        agent_ori = data_dict['ori']
        batch_size, num_voxels = scene_voxs.shape[:2]

        # 1. highlight target regions
        tgt_vox_embed = torch.zeros(batch_size, num_voxels, self.hidden_dim, device=self.tgt_vox_embed.weight.device)
        tgt_vox_embed[tgt_obj_mask] = self.tgt_vox_embed.weight

        # 2. feature fuse
        vox_2d_fts += tgt_vox_embed

        # 3. get relative positions
        vox_pos_relat = self.get_relat_pos(
            pos=scene_voxs[:, :, :3], agent_pos=agent_pos, agent_ori=agent_ori
        )

        # 4. add pos embed
        vox_2d_fts = self.add_pos_embed(voxel_pos=vox_pos_relat, voxel_feat=vox_2d_fts)

        if self.resample:
            # 6. Q-Former style querying: TODO
            query = torch.zeros_like(self.query_pos.weight)
            query = query.unsqueeze(0).repeat(batch_size, 1, 1)
            # prepare masks according to nn.Transformer convention
            txt_mask = ~(data_dict['input_txt_mask'].bool())
            vox_masks = ~vox_masks
            for layer in self.q_former:
                query = layer(
                    tgt=query, query_pos=self.query_pos.weight,
                    memory_txt=data_dict['input_txt_embed'].float(), memory_txt_key_padding_mask=txt_mask,
                    memory_vox_2d=vox_2d_fts, memory_vox_key_padding_mask=vox_masks,
                )[0]
            data_dict['scene_tokens'] = query
            data_dict['scene_masks'] = torch.ones(batch_size, query.shape[1]).bool()
        else:
            # 6. truncate voxels, naive implementation
            data_dict['scene_tokens'] = vox_2d_fts[:, :self.max_num_voxels]
            data_dict['scene_masks'] = vox_masks[:, :self.max_num_voxels]

        return data_dict

@MODULE_REGISTRY.register()
class BEVQFormerLLaVA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.feat_dim_2d[cfg.feat_type_2d]
        self.voxel_size = cfg.voxel_size
        # indicator of target region
        self.tgt_vox_embed = nn.Embedding(1, self.hidden_dim)
        self.init_pos_embed(self.hidden_dim, cfg.pos_embed_type)
        self.resample = cfg.resample
        if self.resample:
            q_former_layer = VoxelQFormerLLaVALayer(
                d_model=self.hidden_dim,
                nhead=cfg.q_former.num_attention_heads,
                dim_feedforward=cfg.q_former.dim_feedforward,
                dropout=cfg.q_former.dropout,
                activation=cfg.q_former.activation,
                feat_dim_txt=cfg.feat_dim_txt,
            )
            self.q_former = layer_repeat(q_former_layer, cfg.q_former.num_layers)
            self.query_pos = nn.Embedding(cfg.num_queries, self.hidden_dim)
        else:
            self.max_num_voxels = cfg.max_num_voxels
        logger.info(f"Build 3D module BEVQFormerLLaVA: voxel_size={self.voxel_size}, "
                    f"pos_embed_type={self.pos_embed_type}, resample={cfg.resample}")
    def init_pos_embed(self, dim, pos_embed_type='rope'):
        assert pos_embed_type in ['rope', 'fourier'], f"BEVQFormerLLaVA: {pos_embed_type} not supported for pos_embed_type"
        self.pos_embed_type = pos_embed_type
        if self.pos_embed_type == 'rope':
            pos_embed_dim_xy = dim // 2
            inv_freq_xy = 1.0 / (10000 ** (torch.arange(0, pos_embed_dim_xy, 2) / pos_embed_dim_xy))   # [dim//4,]
            inv_freq_xy = torch.repeat_interleave(inv_freq_xy, repeats=2)   # [dim//2,]
            inv_freq_xy = torch.stack([inv_freq_xy, inv_freq_xy], dim=0)   # [2, dim//2]
            self.register_buffer('inv_freq_xy', inv_freq_xy, persistent=False)
            pos_embed_dim_z = dim
            inv_freq_z = 1.0 / (10000 ** (torch.arange(0, pos_embed_dim_z, 2) / pos_embed_dim_z))   # [dim//2,]
            inv_freq_z = torch.repeat_interleave(inv_freq_z, repeats=2)   # [dim,]
            inv_freq_z = inv_freq_z.unsqueeze(0)   # [1, dim]
            self.register_buffer('inv_freq_z', inv_freq_z, persistent=False)
        elif self.pos_embed_type == 'fourier':
            self.fourier_proj_xy = nn.Linear(2, dim//2)
            nn.init.normal_(self.fourier_proj_xy.weight)
            nn.init.constant_(self.fourier_proj_xy.bias, 0)
            self.fourier_mlp_xy = get_mlp_head(dim, dim, dim, activation='gelu')
            self.fourier_proj_z = nn.Linear(1, dim//2)
            nn.init.normal_(self.fourier_proj_z.weight)
            nn.init.constant_(self.fourier_proj_z.bias, 0)
            self.fourier_mlp_z = get_mlp_head(dim, dim, dim, activation='gelu')
        else:
            raise NotImplementedError()
    @staticmethod
    def get_relat_pos_xy(pos, agent_pos, agent_ori):
        """
        pos: tensor, (B, N, 3)
        agent_pos: tensor, (B, 3)
        agent_ori: tensor, (B, 2)
        """
        if pos.ndim == 2:
            pos = pos[None, :, :]
        if agent_pos.ndim == 1:
            agent_pos = agent_pos[None, :]
        if agent_ori.ndim == 1:
            agent_ori = agent_ori[None, :]
        pos_relat = pos[:, :, :2] - agent_pos[:, None, :2]
        ori_normalized = F.normalize(agent_ori, dim=-1)
        x = ori_normalized[:, 0]
        y = ori_normalized[:, 1]
        rot_mat = torch.zeros(pos_relat.shape[0], 2, 2, device=pos_relat.device)
        rot_mat[:, 0, 0] = y
        rot_mat[:, 0, 1] = x
        rot_mat[:, 1, 0] = -x
        rot_mat[:, 1, 1] = y
        pos_relat = torch.bmm(pos_relat, rot_mat)
        pos_relat = torch.cat([pos_relat, pos[:, :, 2:3]], dim=-1)
        return pos_relat
    def add_pos_embed(self, voxel_pos, voxel_feat, dim='xy'):
        """
        voxel_pos: tensor (B, N, 3)
        voxel_feat: tensor (B, N, D)
        """
        if voxel_pos.ndim == 2:
            voxel_pos = voxel_pos[None, ...]
        if voxel_feat.ndim == 2:
            voxel_feat = voxel_feat[None, ...]
        batch_size, num_voxels = voxel_pos.shape[:2]
        if self.pos_embed_type == 'rope':
            if dim == 'xy':
                voxel_pos = voxel_pos[:, :, :2].unsqueeze(-1)   # [B, N, 2, 1]
                rotary_factor = voxel_pos * self.inv_freq_xy   # [B, N, 2, D//2]
            elif dim == 'z':
                voxel_pos = voxel_pos[:, :, 2:3].unsqueeze(-1)   # [B, N, 1, 1]
                rotary_factor = voxel_pos * self.inv_freq_z   # [B, N, 1, D]
            rotary_cos = rotary_factor.cos().reshape(batch_size, num_voxels, -1)   # [B, N, D]
            rotary_sin = rotary_factor.sin().reshape(batch_size, num_voxels, -1)   # [B, N, D]
            voxel_feat_1 = torch.stack([-voxel_feat[:, :, 1::2], voxel_feat[:, :, ::2]], dim=-1)   # [B, N, D//2, 2]
            voxel_feat_1 = voxel_feat_1.reshape(batch_size, num_voxels, -1)   # [B, N, D]
            return voxel_feat * rotary_cos + voxel_feat_1 * rotary_sin
        elif self.pos_embed_type == 'fourier':
            if dim == 'xy':
                latent = self.fourier_proj_xy(voxel_pos[:, :, :2])   # [B, N, D//2]
            elif dim == 'z':
                latent = self.fourier_proj_z(voxel_pos[:, :, 2:3])   # [B, N, D//2]
            cos = latent.cos()
            sin = latent.sin()
            fourier_feat = torch.cat([cos, sin], dim=-1)   # [B, N, D]
            fourier_feat /= sqrt(fourier_feat.shape[-1])
            if dim == 'xy':
                fourier_feat = self.fourier_mlp_xy(fourier_feat)   # [B, N, D]
            elif dim == 'z':
                fourier_feat = self.fourier_mlp_z(fourier_feat)   # [B, N, D]
            return voxel_feat + fourier_feat
        else:
            raise NotImplementedError()
    def forward(self, data_dict):
        """
        data_dict requires keys:
            scene_voxs: tensor, (B, N, 6)
            vox_masks: (B, N)
            vox_3d_fts: tensor, (B, N, 768)
            vox_2d_fts: tensor, (B, N, 768)
            vox_2d_fts_mask: tensor, (B,)
            tgt_obj_mask: tensor, (B, N)
            pos: tensor, (B, 3)
            ori: tensor, (B, 2)
            input_txt_embed: tensor, (B, T, D_2)
            input_txt_mask: tensor, (B, T)
        """
        scene_voxs = data_dict['scene_voxs']
        vox_masks = data_dict['vox_masks'].bool()
        vox_2d_fts = data_dict['vox_2d_fts']
        tgt_obj_mask = data_dict['tgt_obj_mask']
        agent_pos = data_dict['pos']
        agent_ori = data_dict['ori']
        batch_size, num_voxels = scene_voxs.shape[:2]
        # 1. highlight target regions
        tgt_vox_embed = torch.zeros(batch_size, num_voxels, self.hidden_dim, device=self.tgt_vox_embed.weight.device)
        tgt_vox_embed[tgt_obj_mask] = self.tgt_vox_embed.weight
        # 2. feature fuse
        vox_2d_fts += tgt_vox_embed
        # 3. get relative positions
        vox_pos_relat = self.get_relat_pos_xy(
            pos=scene_voxs[:, :, :3], agent_pos=agent_pos, agent_ori=agent_ori
        )
        # 4. z-axis pooling with height encoding
        vox_2d_fts = self.add_pos_embed(voxel_pos=vox_pos_relat, voxel_feat=vox_2d_fts, dim='z')
        grid_xy = (vox_pos_relat[:, :, :2] // self.voxel_size).long()
        grid_xy_pooled = []
        grid_2d_fts = []
        for i in range(batch_size):
            _, vox_grid_id, grid_count = torch.unique(grid_xy[i, vox_masks[i]], dim=0, return_inverse=True, return_counts=True)
            # sort by counts
            sorted_indices = torch.argsort(grid_count, descending=True)
            new_mapping = torch.empty_like(sorted_indices)
            new_mapping[sorted_indices] = torch.arange(len(sorted_indices), device=new_mapping.device)
            new_vox_grid_id = new_mapping[vox_grid_id]
            grid_xy_pooled.append(scatter_mean(vox_pos_relat[i, vox_masks[i], :2], new_vox_grid_id, dim=0))
            grid_2d_fts.append(scatter_mean(vox_2d_fts[i, vox_masks[i], :], new_vox_grid_id, dim=0))
        grid_xy_pooled, grid_masks = pad_tensors(grid_xy_pooled, return_mask=True)
        grid_2d_fts = pad_tensors(grid_2d_fts)
        # 5. grid xy pos encoding
        grid_2d_fts = self.add_pos_embed(voxel_pos=grid_xy_pooled, voxel_feat=grid_2d_fts, dim='xy')
        if self.resample:
            # 6. Q-Former style querying
            query = torch.zeros_like(self.query_pos.weight)
            query = query.unsqueeze(0).repeat(batch_size, 1, 1)
            # prepare masks according to nn.Transformer convention
            txt_mask = ~(data_dict['input_txt_mask'].bool())
            grid_masks = ~grid_masks
            for layer in self.q_former:
                query = layer(
                    tgt=query, query_pos=self.query_pos.weight,
                    memory_txt=data_dict['input_txt_embed'].float(), memory_txt_key_padding_mask=txt_mask,
                    memory_vox_2d=grid_2d_fts, memory_vox_key_padding_mask=grid_masks,
                )[0]
            data_dict['scene_tokens'] = query
            data_dict['scene_masks'] = torch.ones(batch_size, query.shape[1]).bool()
        else:
            # 6. truncate voxels, naive implementation
            data_dict['scene_tokens'] = grid_2d_fts[:, :self.max_num_voxels]
            data_dict['scene_masks'] = grid_masks[:, :self.max_num_voxels]
        return data_dict

# LEO-1 object-centric
class OSE3D(nn.Module):
    # Open-vocabulary, Spatial-attention, Embodied-token, 3D-agent
    def __init__(self, cfg):
        super().__init__()
        self.use_spatial_attn = cfg.use_spatial_attn   # spatial attention
        self.use_embodied_token = cfg.use_embodied_token   # embodied token
        hidden_dim = cfg.hidden_dim

        # pcd backbone
        self.obj_encoder = PointcloudBackbone(cfg.backbone)
        self.use_img_feat = cfg.use_img_feat
        self.obj_proj = nn.Linear(self.obj_encoder.out_dim, hidden_dim)
        self.img_proj = nn.Linear(cfg.img_feat_dim, hidden_dim)

        # embodied token
        if self.use_embodied_token:
            self.anchor_feat = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.anchor_img = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.anchor_size = nn.Parameter(torch.ones(1, 1, 3))
            self.orient_encoder = nn.Linear(cfg.fourier_size, hidden_dim)
        self.obj_type_embed = nn.Embedding(2, hidden_dim)

        # spatial encoder
        if self.use_spatial_attn:
            spatial_encoder_layer = TransformerSpatialEncoderLayer(
                d_model=hidden_dim,
                nhead=cfg.spatial_encoder.num_attention_heads,
                dim_feedforward=cfg.spatial_encoder.dim_feedforward,
                dropout=cfg.spatial_encoder.dropout,
                activation=cfg.spatial_encoder.activation,
                spatial_dim=cfg.spatial_encoder.spatial_dim,
                spatial_multihead=cfg.spatial_encoder.spatial_multihead,
                spatial_attn_fusion=cfg.spatial_encoder.spatial_attn_fusion,
            )
        else:
            spatial_encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=cfg.spatial_encoder.num_attention_heads,
                dim_feedforward=cfg.spatial_encoder.dim_feedforward,
                dropout=cfg.spatial_encoder.dropout,
                activation=cfg.spatial_encoder.activation,
            )

        self.spatial_encoder = layer_repeat(
            spatial_encoder_layer,
            cfg.spatial_encoder.num_layers,
        )

        self.pairwise_rel_type = cfg.spatial_encoder.pairwise_rel_type
        self.spatial_dist_norm = cfg.spatial_encoder.spatial_dist_norm
        self.spatial_dim = cfg.spatial_encoder.spatial_dim
        self.obj_loc_encoding = cfg.spatial_encoder.obj_loc_encoding

        # location encoding
        if self.obj_loc_encoding in ['same_0', 'same_all']:
            num_loc_layers = 1
        elif self.obj_loc_encoding == 'diff_all':
            num_loc_layers = cfg.spatial_encoder.num_layers

        loc_layer = nn.Sequential(
            nn.Linear(cfg.spatial_encoder.dim_loc, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.loc_layers = layer_repeat(loc_layer, num_loc_layers)

        # only initialize spatial encoder and loc layers
        self.spatial_encoder.apply(_init_weights_bert)
        self.loc_layers.apply(_init_weights_bert)

        if self.use_embodied_token:
            nn.init.normal_(self.anchor_feat, std=0.02)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward_transformer(self, all_obj_feats, all_img_feats, all_obj_locs, all_obj_masks, pairwise_locs):
        raise NotImplementedError

    def forward(self, data_dict):
        """
        data_dict requires keys:
            obj_fts: (B, N, P, 6), xyz + rgb
            obj_fts_img: (B, N, D)
            obj_masks: (B, N), 1 valid and 0 masked
            obj_locs: (B, N, 6), xyz + whd
            anchor_loc: (B, 3)
            anchor_orientation: (B, C)
        """
        obj_feats = self.obj_encoder(data_dict['obj_fts'])
        obj_feats = self.obj_proj(obj_feats)
        img_feats = self.img_proj(data_dict['obj_fts_img']) if self.use_img_feat else None
        obj_masks = ~data_dict['obj_masks']   # flipped due to different convention of TransformerEncoder

        B, N = obj_feats.shape[:2]
        device = obj_feats.device

        obj_type_ids = torch.zeros((B, N), dtype=torch.long, device=device)
        obj_type_embeds = self.obj_type_embed(obj_type_ids)

        if self.use_embodied_token:
            # anchor feature
            anchor_orient = data_dict['anchor_orientation'].unsqueeze(1)
            anchor_orient_feat = self.orient_encoder(generate_fourier_features(anchor_orient))
            anchor_feat = self.anchor_feat + anchor_orient_feat
            anchor_img = self.anchor_img + anchor_orient_feat
            anchor_mask = torch.zeros((B, 1), dtype=bool, device=device)

            # anchor loc (3) + size (3)
            anchor_loc = torch.cat(
                [data_dict['anchor_loc'].unsqueeze(1), self.anchor_size.expand(B, -1, -1).to(device)], dim=-1
            )

            # anchor type
            anchor_type_id = torch.ones((B, 1), dtype=torch.long, device=device)
            anchor_type_embed = self.obj_type_embed(anchor_type_id)

            # fuse anchor and objs
            all_obj_feats = torch.cat([anchor_feat, obj_feats], dim=1)
            all_img_feats = torch.cat([anchor_img, img_feats], dim=1) if self.use_img_feat else None
            all_obj_masks = torch.cat((anchor_mask, obj_masks), dim=1)

            all_obj_locs = torch.cat([anchor_loc, data_dict['obj_locs']], dim=1)
            all_obj_type_embeds = torch.cat((anchor_type_embed, obj_type_embeds), dim=1)

        else:
            all_obj_feats = obj_feats
            all_img_feats = img_feats if self.use_img_feat else None
            all_obj_masks = obj_masks

            all_obj_locs = data_dict['obj_locs']
            all_obj_type_embeds = obj_type_embeds

        all_obj_feats = all_obj_feats + all_obj_type_embeds
        all_img_feats = all_img_feats + all_obj_type_embeds if self.use_img_feat else None

        # call spatial encoder
        if self.use_spatial_attn:
            pairwise_locs = calc_pairwise_locs(
                all_obj_locs[:, :, :3],
                all_obj_locs[:, :, 3:],
                pairwise_rel_type=self.pairwise_rel_type,
                spatial_dist_norm=self.spatial_dist_norm,
                spatial_dim=self.spatial_dim,
            )

        data_dict['scene_tokens'] = self.forward_transformer(
            all_obj_feats, all_img_feats, all_obj_locs, all_obj_masks, pairwise_locs
        )
        data_dict['obj_masks'] = ~all_obj_masks

        return data_dict


@MODULE_REGISTRY.register()
class OSE3DSelf(OSE3D):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"Build 3D module: OSE3DSelf, use_img_feat={cfg.use_img_feat}")

    def forward_transformer(self, all_obj_feats, all_img_feats, all_obj_locs, all_obj_masks, pairwise_locs):
        if all_img_feats is not None:
            all_obj_feats = (all_obj_feats + all_img_feats) / 2

        for i, pc_layer in enumerate(self.spatial_encoder):
            if self.obj_loc_encoding == 'diff_all':
                query_pos = self.loc_layers[i](all_obj_locs)
            else:
                query_pos = self.loc_layers[0](all_obj_locs)

            if not (self.obj_loc_encoding == 'same_0' and i > 0):
               all_obj_feats = all_obj_feats + query_pos

            if self.use_spatial_attn:
                all_obj_feats, _ = pc_layer(
                    all_obj_feats, pairwise_locs,
                    tgt_key_padding_mask=all_obj_masks
                )
            else:
                all_obj_feats, _ = pc_layer(
                    all_obj_feats,
                    tgt_key_padding_mask=all_obj_masks
                )

        return all_obj_feats


@MODULE_REGISTRY.register()
class OSE3DCross(OSE3D):
    # only for fixed query and #query == #object, i.e., #query == max_obj_len, due to spatial transformer
    def __init__(self, cfg):
        super().__init__(cfg)

        self.query_learnable = cfg.query_learnable
        if cfg.query_learnable:
            if cfg.use_embodied_token:
                self.query_pos = nn.Embedding(cfg.num_queries+1, cfg.hidden_dim)
            else:
                self.query_pos = nn.Embedding(cfg.num_queries, cfg.hidden_dim)

        self.query_cross_encoder = layer_repeat(
            CrossAttentionLayer(
                d_model=cfg.hidden_dim, nhead=cfg.spatial_encoder.num_attention_heads,
                dropout=cfg.spatial_encoder.dropout, activation=cfg.spatial_encoder.activation,
                normalize_before=False, batch_first=True
            ), cfg.spatial_encoder.num_layers
        )
        if cfg.use_img_feat:
            self.query_cross_encoder_img = layer_repeat(
            CrossAttentionLayer(
                d_model=cfg.hidden_dim, nhead=cfg.spatial_encoder.num_attention_heads,
                dropout=cfg.spatial_encoder.dropout, activation=cfg.spatial_encoder.activation,
                normalize_before=False, batch_first=True
            ), cfg.spatial_encoder.num_layers
        )

        logger.info(f"Build 3D module: OSE3DCross, use_img_feat={cfg.use_img_feat}, query_learnable={cfg.query_learnable}")

    def forward_transformer(self, all_obj_feats, all_img_feats, all_obj_locs, all_obj_masks, pairwise_locs):
        if self.query_learnable:
            query_feat = torch.zeros_like(self.query_pos.weight)
            query_feat = query_feat.unsqueeze(0).repeat(all_obj_feats.shape[0], 1, 1)
        else:
            query_feat = torch.zeros_like(all_obj_feats)

        for i, pc_layer in enumerate(self.spatial_encoder):
            if self.query_learnable:
                query_pos = self.query_pos.weight
            else:
                if self.obj_loc_encoding == 'diff_all':
                    query_pos = self.loc_layers[i](all_obj_locs)
                else:
                    query_pos = self.loc_layers[0](all_obj_locs)

            query_feat = self.query_cross_encoder[i](
                tgt=query_feat, memory=all_obj_feats,
                memory_key_padding_mask=all_obj_masks,
                query_pos=query_pos, pos=query_pos
            )
            if all_img_feats is not None:
                query_feat_img = self.query_cross_encoder_img[i](
                    tgt=query_feat, memory=all_img_feats,
                    memory_key_padding_mask=all_obj_masks,
                    query_pos=query_pos, pos=query_pos
                )
                query_feat = (query_feat + query_feat_img) / 2

            if self.use_spatial_attn:
                query_feat, _ = pc_layer(
                    query_feat, pairwise_locs,
                    tgt_key_padding_mask=all_obj_masks
                )
            else:
                query_feat, _ = pc_layer(
                    query_feat,
                    tgt_key_padding_mask=all_obj_masks
                )

        return query_feat
