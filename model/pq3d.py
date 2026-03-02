import math
from contextlib import nullcontext
from copy import copy, deepcopy
from functools import partial

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from transformers import CLIPTextModelWithProjection

from data.data_utils import PromptType
from model.pointnetpp.pointnetpp import PointNetPP
from model.utils import _init_weights_bert, calc_pairwise_locs

logger = get_logger(__name__)


def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros(
                (src_range[0].shape[0], 3), device=src_range[0].device
            ),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.d_pos = d_pos
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        num_channels = self.d_pos
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(
                    cdim, dtype=torch.float32, device=xyz.device
                )
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                out = self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                out = self.get_fourier_embeddings(
                    xyz, num_channels, input_range
                )
        else:
            raise ValueError(f"Unknown {self.pos_type}")

        return out

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
        return st

class CoordinateEncoder(nn.Module):
    def __init__(self, hidden_size, use_projection=True):
        super().__init__()
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier", d_pos=hidden_size, gauss_scale=1.0, normalize=True)
        if use_projection:
            self.feat_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
    
    def forward(self, coords, input_range):
        with autocast(enabled=False):
            pos = self.pos_enc(coords, input_range=input_range).permute(0, 2, 1)
        if hasattr(self, 'feat_proj'):
            pos = self.feat_proj(pos)
        return pos

def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(*[
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.LayerNorm(hidden_size, eps=1e-12),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, output_size)
    ])
    
def layer_repeat(module, N, share_layer=False):
    if share_layer:
        return nn.ModuleList([module] * N)
    else:
        return nn.ModuleList([deepcopy(module) for _ in range(N - 1)] + [module])

    
class CLIPLanguageEncoder(nn.Module):
    def __init__(self, cfg, weights="openai/clip-vit-large-patch14", output_dim=768, freeze_backbone=True, use_projection=False, projection_type='mlp', num_projection_layers=1, dropout=0.1):
        super().__init__()
        self.context = torch.no_grad if freeze_backbone else nullcontext
        self.model = CLIPTextModelWithProjection.from_pretrained(weights)
        self.use_projection = use_projection
        self.projection_type = projection_type
        if use_projection:
            if projection_type == 'mlp':
                self.projection = get_mlp_head(self.model.config.hidden_size, output_dim, output_dim, dropout=dropout)
            else:
                raise NotImplementedError
        #self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
        
    def forward(self, txt_ids, txt_masks):
        with self.context():
            txt = self.model(txt_ids, txt_masks).last_hidden_state
            txt = self.model.text_projection(txt)
            txt = torch.nn.functional.normalize(txt, p=2, dim=2)
        #txt = self.attention(txt, txt, txt, key_padding_mask=txt_masks.logical_not())[0]
        if self.use_projection:
            if self.projection_type == 'mlp':
                txt = self.projection(txt)
            elif self.projection_type == 'attention':
                for attention_layer in self.projection:
                    txt = attention_layer(txt, tgt_key_padding_mask = txt_masks.logical_not())
            else:
                raise NotImplementedError
        return txt

class ObjectEncoder(nn.Module):
    def __init__(self, cfg, backbone='none', input_feat_size=768, hidden_size=768, freeze_backbone=False, use_projection=False,
                 tgt_cls_num=607, pretrained=None, dropout=0.1, use_cls_head=True):
        super().__init__()
        self.cfg = cfg
        self.freeze_backbone = freeze_backbone
        self.context = torch.no_grad if freeze_backbone else nullcontext
        if backbone == 'pointnet++':
            self.backbone = PointNetPP(
                sa_n_points=[32, 16, None],
                sa_n_samples=[32, 32, None],
                sa_radii=[0.2, 0.4, None],
                sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
            )
        if use_cls_head:
            self.cls_head = get_mlp_head(input_feat_size, input_feat_size // 2, tgt_cls_num, dropout=0.3)

        self.use_projection = use_projection
        if use_projection:
            self.input_feat_proj = nn.Sequential(nn.Linear(input_feat_size, hidden_size), nn.LayerNorm(hidden_size))
        else:
            assert input_feat_size == hidden_size, "input_feat_size should be equal to hidden_size!"
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        # load weights
        self.apply(_init_weights_bert)
        if pretrained:
            logger.info(f"PQ3D ObjectEncoder: load pretrained weights from {pretrained}")
            pre_state_dict = torch.load(pretrained, weights_only=True)
            state_dict = {}
            for k, v in pre_state_dict.items():
                if k[0] in ['0', '2', '4']: # key mapping for voxel
                    k = 'cls_head.' + k
                k = k.replace('vision_encoder.vis_cls_head.', 'cls_head.') # key mapping for mv
                k = k.replace('point_cls_head.', 'cls_head.') # key mapping for pc 
                k = k.replace('point_feature_extractor.', 'backbone.')
                state_dict[k] = v
            warning = self.load_state_dict(state_dict, strict=False)
            logger.info(warning)
        # freeze backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad=False

    def freeze_bn(self, m):
        for layer in m.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, obj_feats, **kwargs):
        if self.freeze_backbone and hasattr(self, 'backbone'):
            self.freeze_bn(self.backbone)

        batch_size, num_objs  = obj_feats.shape[:2]
        with self.context():
            if hasattr(self, 'backbone'):
                obj_feats = self.backbone(einops.rearrange(obj_feats, 'b o p d -> (b o) p d'))
                obj_feats = einops.rearrange(obj_feats, '(b o) d -> b o d', b=batch_size)

        obj_embeds = self.input_feat_proj(obj_feats) if self.use_projection else obj_feats
        if hasattr(self, 'dropout'):
            obj_embeds = self.dropout(obj_embeds)

        if hasattr(self, 'cls_head'):
            obj_cls_logits = self.cls_head(obj_feats)
            return obj_embeds, obj_cls_logits
        else:
            return obj_embeds

def cfg2dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True)

class QueryEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, memories, dim_feedforward=2048, dropout=0.1, activation="relu", prenorm=False, spatial_selfattn=False, structure='mixed', memory_dropout=0, drop_memories_test=[]):
        super().__init__()
        if spatial_selfattn:
            self.self_attn = SpatialSelfAttentionLayer(d_model, nhead, dropout=dropout, activation=activation, normalize_before=prenorm, batch_first=True)
        else:
            self.self_attn = SelfAttentionLayer(d_model, nhead, dropout=dropout, activation=activation, normalize_before=prenorm, batch_first=True)
        cross_attn_layer = CrossAttentionLayer(d_model, nhead, dropout=dropout, activation=activation, normalize_before=prenorm, batch_first=True) 
        self.cross_attn_list = layer_repeat(cross_attn_layer, len(memories))
        self.memory2ca = {memory:ca for memory, ca in zip(memories, self.cross_attn_list)}
        self.ffn = FFNLayer(d_model, dim_feedforward, dropout=dropout, activation=activation, normalize_before=prenorm)
        self.structure = structure
        self.memories = memories
        self.memory_dropout = memory_dropout
        self.drop_memories_test = drop_memories_test
        if structure == 'gate':
            self.gate_proj = nn.Linear(d_model, d_model)

    def forward(self, query, input_dict, pairwise_locs=None):
        _, query_masks, query_pos = input_dict['query']

        def sequential_ca(query, memories):
            for memory in memories:
                cross_attn = self.memory2ca[memory]
                feat, mask, pos = input_dict[memory] 
                if mask.ndim == 2:
                    memory_key_padding_mask = mask
                    attn_mask = None
                else:
                    memory_key_padding_mask = None
                    attn_mask = mask
                query = cross_attn(tgt=query, memory=feat, attn_mask=attn_mask, memory_key_padding_mask = memory_key_padding_mask, query_pos = query_pos, pos = pos)
            return query

        def parallel_ca(query, memories):
            assert 'prompt' not in memories
            query_list = []
            for memory in memories:
                cross_attn = self.memory2ca[memory]
                feat, mask, pos = input_dict[memory] 
                if mask.ndim == 2:
                    memory_key_padding_mask = mask
                    attn_mask = None
                else:
                    memory_key_padding_mask = None
                    attn_mask = mask
                update = cross_attn(tgt=query, memory=feat, attn_mask=attn_mask, memory_key_padding_mask = memory_key_padding_mask, query_pos = query_pos, pos = pos)
                query_list.append(update)
            # training time memory dropout
            if self.training and self.memory_dropout > 0.0:
                dropout_mask = torch.rand(query.shape[0], len(memories), device=query.device) > self.memory_dropout
                num_remained_memories = dropout_mask.sum(dim=1)
                dropout_mask = torch.logical_or(dropout_mask, num_remained_memories.unsqueeze(-1) == 0)
                num_remained_memories = dropout_mask.sum(dim=1)
                query_tensor = torch.stack(query_list, dim=1)
                query = (query_tensor * dropout_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / num_remained_memories.unsqueeze(-1).unsqueeze(-1).float()
            else:
                query = torch.stack(query_list, dim=1).mean(dim=1)
            return query
        
        memories = self.memories if self.training else [m for m in self.memories if m not in self.drop_memories_test]
        
        if self.structure == 'sequential':
            query = sequential_ca(query, memories)
        elif self.structure == 'parallel':
            query = parallel_ca(query, memories)
        elif self.structure == 'mixed':
            # [mv,pc,vx] + prompt
            query = parallel_ca(query, [m for m in memories if m != 'prompt'])
            query = sequential_ca(query, ['prompt'])
        elif self.structure == 'gate':
            prompt = sequential_ca(query, ['prompt'])
            gate = torch.sigmoid(self.gate_proj(prompt))
            update = parallel_ca(query, [m for m in self.memories if m != 'prompt'])
            query = (1. - gate) * query + gate * update
        else:
            raise NotImplementedError(f"Unknow structure type: {self.structure}")

        if isinstance(self.self_attn, SpatialSelfAttentionLayer):
            query = self.self_attn(query, tgt_key_padding_mask = query_masks, query_pos = query_pos, 
                                   pairwise_locs = pairwise_locs)
        else:
            query = self.self_attn(query, tgt_key_padding_mask = query_masks, query_pos = query_pos)
        query = self.ffn(query)

        return query

def get_activation_fn(activation_type):
    if activation_type not in ["relu", "gelu", "glu"]:
        raise RuntimeError(f"activation function currently support relu/gelu, not {activation_type}")
    return getattr(F, activation_type)


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        batch_first=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=attn_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=attn_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, attn_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, attn_mask, tgt_key_padding_mask, query_pos
        )


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        batch_first=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, add_zero_attn=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        attn_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=attn_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        attn_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=attn_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        attn_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                attn_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt, memory, attn_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

class MultiHeadAttentionSpatial(nn.Module):
    def __init__(
            self, d_model, n_head, dropout=0.1, spatial_multihead=True, spatial_dim=5,
            spatial_attn_fusion='mul',
    ):
        super().__init__()
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' % (d_model, n_head)

        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.spatial_multihead = spatial_multihead
        self.spatial_dim = spatial_dim
        self.spatial_attn_fusion = spatial_attn_fusion

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

        self.spatial_n_head = n_head if spatial_multihead else 1
        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            self.pairwise_loc_fc = nn.Linear(spatial_dim, self.spatial_n_head)
        elif self.spatial_attn_fusion == 'ctx':
            self.pairwise_loc_fc = nn.Linear(spatial_dim, d_model)
        elif self.spatial_attn_fusion == 'cond':
            self.lang_cond_fc = nn.Linear(d_model, self.spatial_n_head * (spatial_dim + 1))
        else:
            raise NotImplementedError('unsupported spatial_attn_fusion %s' % (self.spatial_attn_fusion))

    def forward(self, q, k, v, pairwise_locs, key_padding_mask=None, txt_embeds=None):
        residual = q
        q = einops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = einops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        v = einops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])

        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, 'b l t h -> h b l t')
            if self.spatial_attn_fusion == 'mul':
                loc_attn = F.relu(loc_attn)
            if not self.spatial_multihead:
                loc_attn = einops.repeat(loc_attn, 'h b l t -> (h nh) b l t', nh=self.n_head)
        elif self.spatial_attn_fusion == 'ctx':
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, 'b l t (h k) -> h b l t k', h=self.n_head)
            loc_attn = torch.einsum('hblk,hbltk->hblt', q, loc_attn) / np.sqrt(q.shape[-1])
        elif self.spatial_attn_fusion == 'cond':
            spatial_weights = self.lang_cond_fc(residual)
            spatial_weights = einops.rearrange(spatial_weights, 'b l (h d) -> h b l d', h=self.spatial_n_head,
                                               d=self.spatial_dim + 1)
            if self.spatial_n_head == 1:
                spatial_weights = einops.repeat(spatial_weights, '1 b l d -> h b l d', h=self.n_head)
            spatial_bias = spatial_weights[..., :1]
            spatial_weights = spatial_weights[..., 1:]
            loc_attn = torch.einsum('hbld,bltd->hblt', spatial_weights, pairwise_locs) + spatial_bias
            loc_attn = torch.sigmoid(loc_attn)

        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask, 'b t -> h b l t', h=self.n_head, l=q.size(2))
            attn = attn.masked_fill(mask, -np.inf)
            if self.spatial_attn_fusion in ['mul', 'cond']:
                loc_attn = loc_attn.masked_fill(mask, 0)
            else:
                loc_attn = loc_attn.masked_fill(mask, -np.inf)

        if self.spatial_attn_fusion == 'add':
            fused_attn = (torch.softmax(attn, 3) + torch.softmax(loc_attn, 3)) / 2
        else:
            if self.spatial_attn_fusion in ['mul', 'cond']:
                fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
            else:
                fused_attn = loc_attn + attn
            fused_attn = torch.softmax(fused_attn, 3)

        assert torch.sum(torch.isnan(fused_attn) == 0), logger.info(fused_attn)

        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        output = self.fc(output)
        return output, fused_attn
      
class SpatialSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        batch_first=False,
        spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul'
    ):
        super().__init__()
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout,
            spatial_multihead=spatial_multihead,
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None,
        pairwise_locs=None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            tgt,
            key_padding_mask=tgt_key_padding_mask,
            pairwise_locs=pairwise_locs,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None,
        pairwise_locs=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            tgt,
            key_padding_mask=tgt_key_padding_mask,
            pairwise_locs=pairwise_locs,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None,
        pairwise_locs=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, attn_mask, tgt_key_padding_mask, query_pos,
                pairwise_locs
            )
        return self.forward_post(
            tgt, attn_mask, tgt_key_padding_mask, query_pos,
            pairwise_locs
        )

class QueryMaskEncoder(nn.Module):
    def __init__(self, cfg, memories=[], memory_dropout=0.0, hidden_size=768, num_attention_heads=12, num_layers=4,
                share_layer=False, spatial_selfattn=False, structure='sequential', drop_memories_test=[], use_self_mask=False, num_blocks=1):
        super().__init__()

        self.spatial_selfattn = spatial_selfattn
        query_encoder_layer = QueryEncoderLayer(hidden_size, num_attention_heads, memories, spatial_selfattn=spatial_selfattn, structure=structure, memory_dropout=memory_dropout, drop_memories_test=drop_memories_test)
        self.unified_encoder = layer_repeat(query_encoder_layer, num_layers, share_layer)

        self.apply(_init_weights_bert)
        self.memory_dropout = memory_dropout
        self.scene_meomories = [x for x in memories if x != 'prompt']
        self.drop_memories_test = drop_memories_test
        self.use_self_mask = use_self_mask
        self.num_heads = num_attention_heads
        self.num_blocks = num_blocks

    def forward(self, input_dict, pairwise_locs, mask_head=None):
            
        predictions_class, predictions_mask = [], []
        
        query = input_dict['query'][0]
        voxel_feat = input_dict['voxel'][0] if 'voxel' in input_dict.keys() else None

        for block_counter in range(self.num_blocks):
            for i, layer in enumerate(self.unified_encoder):
                if mask_head is not None:
                    output_class, outputs_mask, attn_mask = mask_head(query)
                    predictions_class.append(output_class)
                    predictions_mask.append(outputs_mask)  
                if self.use_self_mask:
                    attn_mask[attn_mask.all(-1)] = False # prevent query to attend to no point
                    attn_mask = attn_mask.repeat_interleave(self.num_heads, 0)
                    for memory in input_dict.keys():
                        if memory in ['query', 'prompt']:
                            continue
                        input_dict[memory][1] = attn_mask
                        
                if isinstance(voxel_feat, list):
                    input_dict['voxel'][0] = voxel_feat[i]  # select voxel features from multi-scale
                query = layer(query, input_dict, pairwise_locs)

        return query, predictions_class, predictions_mask

class Query3DUnified(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # record parameters
        self.cfg = cfg
        self.memories = cfg.model.memories
        self.use_offline_voxel_fts = cfg.model.get('use_offline_voxel_fts', False)
        self.use_offline_attn_mask = cfg.model.get('use_offline_attn_mask', False)
        self.inputs = self.memories[:]
        self.pairwise_rel_type = self.cfg.model.obj_loc.pairwise_rel_type
        self.spatial_dim = self.cfg.model.obj_loc.spatial_dim
        self.num_heads = self.cfg.model.unified_encoder.args.num_attention_heads
        self.skip_query_encoder_mask_pred = cfg.model.get('skip_query_encoder_mask_pred', False)
        # build prompt type
        self.prompt_types = ['txt', 'loc']
        # build feature encoder
        for input in self.inputs:
            if input == 'prompt':
                for prompt_type in self.prompt_types: # only text prompt for now
                    if prompt_type == 'loc':
                        continue
                    encoder = prompt_type + '_encoder'
                    setattr(self, encoder, CLIPLanguageEncoder(cfg, **cfg2dict(cfg.model.get(encoder).args)))
            else:
                encoder = input + '_encoder'
                setattr(self, encoder, ObjectEncoder(cfg, **cfg2dict(cfg.model.get(encoder).args)))
        # build location encoder
        dim_loc = self.cfg.model.obj_loc.dim_loc
        hidden_size = self.cfg.model.hidden_size
        self.dim_loc = dim_loc
        self.hidden_size = hidden_size
        if self.dim_loc > 3:
            self.coord_encoder = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            self.box_encoder = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        else:
            self.coord_encoder = CoordinateEncoder(hidden_size)
        # build unified encoder    
        self.unified_encoder = QueryMaskEncoder(cfg, **cfg2dict(self.cfg.model.unified_encoder.args))
        
        
    def prompt_encoder(self, data_dict):
        prompt = data_dict['prompt']
        prompt_pad_masks = data_dict['prompt_pad_masks']
        prompt_type = data_dict['prompt_type']
        prompt_feat = torch.zeros(prompt.shape + (self.hidden_size,), device=prompt.device)
        for type in self.prompt_types:
            # get idx
            idx = prompt_type == getattr(PromptType, type.upper())
            if idx.sum() == 0:
                continue
            input = prompt[idx]
            mask = prompt_pad_masks[idx]
            # encode
            if type == 'txt':
                encoder = self.txt_encoder
                feat = encoder(input.long(), mask)
                feat = feat.detach()   # avoid allreducing undefined grad
            elif type == 'loc':
                loc_prompts = input[:, :self.dim_loc]
                if self.dim_loc > 3:
                    feat = self.coord_encoder(loc_prompts[:, :3]).unsqueeze(1) + self.box_encoder(loc_prompts[:, 3:6]).unsqueeze(1)
                else:
                    feat = self.coord_encoder(loc_prompts[:, :3].unsqueeze(1), input_range=[data_dict['coord_min'][idx], data_dict['coord_max'][idx]])
                mask[:, 1:] = False
            else:
                raise NotImplementedError(f'{type} is not implemented')
            # put back to orignal prompt
            prompt_feat[idx] = feat
            prompt_pad_masks[idx] = mask
        return prompt_feat, prompt_pad_masks.logical_not()
        
    def forward(self, data_dict):
        input_dict = {}
        # build query
        mask = data_dict['query_pad_masks'].logical_not()
        query_locs = data_dict['query_locs'][:, :, :self.dim_loc]
        coord_min = data_dict['coord_min']
        coord_max = data_dict['coord_max']
        if self.dim_loc > 3:
            query_pos  = self.coord_encoder(query_locs[:, :, :3]) + self.box_encoder(query_locs[:, :, 3:6])
        else:
            query_pos = self.coord_encoder(query_locs[:, :, :3], input_range=[coord_min, coord_max])
        feat = torch.zeros_like(query_pos)
        pos = query_pos
        input_dict['query'] = (feat, mask, pos)
        # encode fts including point, voxel, image, and prompt
        # the semantics of the attention mask in pytorch (True as masked) is the opposite as Huggingface Transformers (False as masked)  
        fts_locs = data_dict['seg_center']
        if self.dim_loc > 3:
            fts_pos = self.coord_encoder(fts_locs[:, :, :3]) + self.box_encoder(fts_locs[:, :,  3:6])
        else:
            fts_pos = self.coord_encoder(fts_locs[:, :, :3], input_range=[coord_min, coord_max])
        if self.dim_loc > 3:
            fts_pos += self.box_encoder(fts_locs[:, :, 3:6])
        for input in self.inputs:
            feat, mask, pos = None, None, None
            if input == 'prompt':
                feat, mask = self.prompt_encoder(data_dict)
            elif input == 'mv':
                feat = self.mv_encoder(obj_feats = data_dict['mv_seg_fts'])
                mask = data_dict['mv_seg_pad_masks'].logical_not()
                pos = fts_pos
            elif input == 'pc':
                feat = self.pc_encoder(obj_feats = data_dict['pc_seg_fts'])
                mask = data_dict['pc_seg_pad_masks'].logical_not()
                pos = fts_pos
            elif input == 'voxel':
                if self.use_offline_voxel_fts:
                    feat = self.voxel_encoder(data_dict['voxel_seg_fts'])
                    mask = data_dict['voxel_seg_pad_masks'].logical_not()
                else:
                    raise NotImplementedError(input)
                pos = fts_pos
            else:
                raise NotImplementedError(f"Unknow input type: {input}")
            input_dict[input] = [feat, mask, pos]
        # build offline attention mask for guided mask training
        if self.use_offline_attn_mask:
            offline_attn_masks = data_dict['offline_attn_mask']
        else:
            offline_attn_masks = None
        # generate features for mask head
        seg_fts_for_match = []
        for input in self.inputs:
            if input in ['voxel', 'mv', 'pc']:
                feats = copy(input_dict[input][:])
                if isinstance(feats[0], list):
                    assert input == 'voxel'
                    feats[0] = feats[0][-1] # use the last scale of voxel features for segment matching
                seg_fts_for_match.append(feats)                 
        # build mask head
        if hasattr(self, 'mask_head'):
            mask_head_partial = partial(self.mask_head, seg_fts_for_match=seg_fts_for_match, seg_masks=data_dict['seg_pad_masks'].logical_not(),
                                        offline_attn_masks=offline_attn_masks, skip_prediction=self.skip_query_encoder_mask_pred)
        else:
            mask_head_partial = None
        # generate features for spatial attention
        if self.unified_encoder.spatial_selfattn:
            pairwise_locs = calc_pairwise_locs(query_locs[:, :, :3], None, 
                                           pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True,
                                           spatial_dim=self.spatial_dim)
        else:
            pairwise_locs = None
            
        # unified encoding                           
        query, predictions_class, predictions_mask = self.unified_encoder(input_dict, pairwise_locs, mask_head_partial)
               
        return query
