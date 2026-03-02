import math
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from torch import Tensor, nn

from model.utils import get_activation_fn

logger = get_logger(__name__)


# attention-based pooling, adapted from CLIP
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim=None):
        super().__init__()
        if embed_dim % num_heads != 0:
            for n in range(num_heads, 0, -1):
                if embed_dim % n == 0:
                    num_heads = n
                    break
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, output_dim if output_dim else embed_dim)

    def forward(self, x):
        # x: [B, T, D]
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # [B, 1+T, D]
        x = x.transpose(1, 0)   # [1+T, B, D]
        # nn multi-head attention assumes batch_first=False
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.o_proj.weight,
            out_proj_bias=self.o_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ObjectQFormerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 use_img_feat=True, use_vox_feat=True):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.multihead_attn_txt = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.multihead_attn_pcd = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.use_img_feat = use_img_feat
        if self.use_img_feat:
            self.multihead_attn_img = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.norm4 = nn.LayerNorm(d_model)
            self.dropout4 = nn.Dropout(dropout)

        self.use_vox_feat = use_vox_feat
        if self.use_vox_feat:
            self.multihead_attn_vox = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.norm5 = nn.LayerNorm(d_model)
            self.dropout5 = nn.Dropout(dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm6 = nn.LayerNorm(d_model)
        self.dropout6 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory_txt, memory_pcd, memory_img, memory_vox,
                tgt_mask: Optional[Tensor] = None,
                memory_txt_mask: Optional[Tensor] = None,
                memory_pcd_mask: Optional[Tensor] = None,
                memory_img_mask: Optional[Tensor] = None,
                memory_vox_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_txt_key_padding_mask: Optional[Tensor] = None,
                memory_pcd_key_padding_mask: Optional[Tensor] = None,
                memory_img_key_padding_mask: Optional[Tensor] = None,
                memory_vox_key_padding_mask: Optional[Tensor] = None,
                txt_pos: Optional[Tensor] = None,
                pcd_pos: Optional[Tensor] = None,
                img_pos: Optional[Tensor] = None,
                vox_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2, self_attn_matrices = self.self_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(tgt, query_pos),
            value=tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, txt_cross_attn_matrices = self.multihead_attn_txt(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory_txt, txt_pos),
            value=memory_txt, attn_mask=memory_txt_mask,
            key_padding_mask=memory_txt_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt_pcd_2, pcd_cross_attn_matrices = self.multihead_attn_pcd(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory_pcd, pcd_pos),
            value=memory_pcd, attn_mask=memory_pcd_mask,
            key_padding_mask=memory_pcd_key_padding_mask
        )
        tgt_pcd = tgt + self.dropout3(tgt_pcd_2)
        tgt_pcd = self.norm3(tgt_pcd)
        all_fts = [tgt_pcd]
        all_attn = [pcd_cross_attn_matrices]

        if memory_img is not None:
            assert hasattr(self, 'multihead_attn_img'), "ObjectQFormer: provide img feat but have no cross-attn layer"
            tgt_img_2, img_cross_attn_matrices = self.multihead_attn_img(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory_img, img_pos),
                value=memory_img, attn_mask=memory_img_mask,
                key_padding_mask=memory_img_key_padding_mask
            )
            tgt_img = tgt + self.dropout4(tgt_img_2)
            tgt_img = self.norm4(tgt_img)
            all_fts.append(tgt_img)
            all_attn.append(img_cross_attn_matrices)

        if memory_vox is not None:
            assert hasattr(self, 'multihead_attn_vox'), "ObjectQFormer: provide vox feat but have no cross-attn layer"
            tgt_vox_2, vox_cross_attn_matrices = self.multihead_attn_vox(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory_vox, vox_pos),
                value=memory_vox, attn_mask=memory_vox_mask,
                key_padding_mask=memory_vox_key_padding_mask
            )
            tgt_vox = tgt + self.dropout5(tgt_vox_2)
            tgt_vox = self.norm5(tgt_vox)
            all_fts.append(tgt_vox)
            all_attn.append(vox_cross_attn_matrices)

        tgt = torch.stack(all_fts, dim=0).mean(0)   # average over pcd, img, vox

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout6(tgt2)
        tgt = self.norm6(tgt)
        return tgt, self_attn_matrices, txt_cross_attn_matrices, all_attn


class VoxelQFormerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.multihead_attn_txt = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn_vox_3d = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn_vox_2d = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory_txt, memory_vox_3d, memory_vox_2d,
                query_pos: Optional[Tensor] = None,
                memory_txt_key_padding_mask: Optional[Tensor] = None,
                memory_vox_key_padding_mask: Optional[Tensor] = None,
                memory_vox_3d_dropout_mask: Optional[Tensor] = None,
                memory_vox_2d_dropout_mask: Optional[Tensor] = None):
        tgt2, txt_cross_attn_matrices = self.multihead_attn_txt(
            query=self.with_pos_embed(tgt, query_pos),
            key=memory_txt, value=memory_txt,
            key_padding_mask=memory_txt_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt_3d, vox_3d_cross_attn_matrices = self.multihead_attn_vox_3d(
            query=self.with_pos_embed(tgt, query_pos),
            key=memory_vox_3d, value=memory_vox_3d,
            key_padding_mask=memory_vox_key_padding_mask,
        )
        tgt_3d = tgt + self.dropout2(tgt_3d)
        tgt_3d = self.norm2(tgt_3d)

        tgt_2d, vox_2d_cross_attn_matrices = self.multihead_attn_vox_2d(
            query=self.with_pos_embed(tgt, query_pos),
            key=memory_vox_2d, value=memory_vox_2d,
            key_padding_mask=memory_vox_key_padding_mask,
        )
        tgt_2d = tgt + self.dropout3(tgt_2d)
        tgt_2d = self.norm3(tgt_2d)

        w_3d = memory_vox_3d_dropout_mask.int().view(-1, 1, 1)
        w_2d = memory_vox_2d_dropout_mask.int().view(-1, 1, 1)
        tgt = (tgt_3d * w_3d + tgt_2d * w_2d) / (w_3d + w_2d)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, txt_cross_attn_matrices, vox_3d_cross_attn_matrices, vox_2d_cross_attn_matrices


class VoxelQFormerLLaVALayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", feat_dim_txt=4096):
        super().__init__()
        self.multihead_attn_txt = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, kdim=feat_dim_txt, vdim=feat_dim_txt, batch_first=True
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory_txt, memory_vox_2d,
                query_pos: Optional[Tensor] = None,
                memory_txt_key_padding_mask: Optional[Tensor] = None,
                memory_vox_key_padding_mask: Optional[Tensor] = None):
        tgt2, txt_cross_attn_matrices = self.multihead_attn_txt(
            query=self.with_pos_embed(tgt, query_pos),
            key=memory_txt, value=memory_txt,
            key_padding_mask=memory_txt_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # tgt: (B, 60, 1024)
        # memory_vox_2d: (B, N, 1024)
        mask_float = torch.zeros_like(memory_vox_key_padding_mask, dtype=tgt.dtype)
        mask_float.masked_fill_(memory_vox_key_padding_mask, float('-inf'))
        mask_float = mask_float.unsqueeze(1)   # (B, N) -> (B, 1, N)
        tgt = tgt / math.sqrt(tgt.shape[-1])
        memory_vox_2d_T = einops.rearrange(memory_vox_2d, 'b n d -> b d n')
        attn_scores = torch.baddbmm(mask_float, tgt, memory_vox_2d_T)
        attn_weights = torch.softmax(attn_scores, dim=-1)   # (B, 60, N)
        attn_output = torch.bmm(attn_weights, memory_vox_2d)   # (B, 60, 1024)

        return attn_output, txt_cross_attn_matrices, attn_weights


# cross attention layer from PQ3D
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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, batch_first=True, dropout=0.1, activation="relu", prenorm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward modules
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.prenorm = prenorm

    def forward(
            self, tgt, tgt_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = tgt
        if self.prenorm:
            tgt2 = self.norm1(tgt2)
        tgt2, self_attn_matrices = self.self_attn(
            query=tgt2, key=tgt2, value=tgt2, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        if not self.prenorm:
            tgt = self.norm1(tgt)
        if self.prenorm:
            tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        if not self.prenorm:
            tgt = self.norm2(tgt)
        return tgt, self_attn_matrices


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
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

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
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, fused_attn


class TransformerSpatialEncoderLayer(TransformerEncoderLayer):
    def __init__(
            self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
            spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul'
    ):
        super().__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        del self.self_attn
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout,
            spatial_multihead=spatial_multihead,
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

    def forward(
            self, tgt, tgt_pairwise_locs,
            tgt_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = tgt
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, tgt2, tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt, self_attn_matrices
