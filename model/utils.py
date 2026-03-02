import contextlib
import copy

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

special_to_unused = {
    "<think_type>": "[unused1]",
    "[/think_type]": "[unused2]",
    "[think_grd]": "[unused3]",
    "[/think_grd]": "[unused4]",
    "[obj_prob]": "[unused5]",
    "[/obj_prob]": "[unused6]",
    "[think_task]": "[unused7]",
    "[/think_task]": "[unused8]",
    "[think_sum]": "[unused9]",
    "[/think_sum]": "[unused10]",
    "[answer]": "[unused11]",
    "[/answer]": "[unused12]",
    "[OBJ]": "[unused13]"
}

def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


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


def scene_pcd_to_voxel_tokens(pcd_pos, pcd_rgb, voxel_reso=0.2, hash_func='fnv', pcd_feat=None):
    # voxelization range: ..., [-voxel_reso, 0), [0, voxel_reso), ...
    if pcd_feat is None:
        pcd_feat = pcd_rgb.copy()
    coord_continuous = pcd_pos / voxel_reso
    coord_discrete = np.floor(coord_continuous).astype(int)
    coord_min = coord_discrete.min(0)
    coord_discrete -= coord_min
    # now xyz indices range from 0 to num_tokens per dim

    if hash_func.lower() == 'fnv':
        key = fnv_hash_vec(coord_discrete)
    elif hash_func.lower() == 'ravel':
        key = ravel_hash_vec(coord_discrete)
    else:
        raise NotImplementedError(f"Not supported hash func: {hash_func}")

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, inverse = np.unique(key_sort, return_inverse=True)

    voxel_pos = []
    voxel_rgb = []
    voxel_feat = []
    for i in range(inverse.max()+1):
        pcd_idx_for_this_voxel = idx_sort[inverse == i]
        voxel_pos.append(coord_continuous[pcd_idx_for_this_voxel].reshape(-1, 3).mean(0))
        voxel_rgb.append(pcd_rgb[pcd_idx_for_this_voxel].reshape(-1, 3).mean(0))
        voxel_feat.append(pcd_feat[pcd_idx_for_this_voxel].reshape(-1, pcd_feat.shape[-1]).mean(0))
    voxel_tokens = {
        'pos': torch.from_numpy(np.stack(voxel_pos, axis=0)).float(),
        'rgb': torch.from_numpy(np.stack(voxel_rgb, axis=0)).float(),
        'feat': torch.stack(voxel_feat, axis=0),
    }
    return voxel_tokens


def disabled_train(self, mode=True):
    """
    Overwrite model.train with this function to make sure train/eval mode does not change anymore
    """
    return self


def maybe_autocast(model, dtype='bf16', enabled=True):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    enable_autocast = model.device != torch.device('cpu')

    if dtype == 'bf16':
        dtype = torch.bfloat16
    elif dtype == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    if enable_autocast:
        return torch.amp.autocast('cuda', dtype=dtype, enabled=enabled)
    else:
        return contextlib.nullcontext()


def _init_weights_bert(module, std=0.02):
    """
        Huggingface transformer weight initialization,
        most commonly for bert initialization
    """
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


#########################################################
# General modules helpers
#########################################################
def get_activation_fn(activation_type):
    if activation_type not in ["relu", "gelu", "glu"]:
        raise RuntimeError(f"activation function currently support relu/gelu, not {activation_type}")
    return getattr(F, activation_type)


def get_activation_layer(activation_type):
    if activation_type == 'relu':
        return nn.ReLU()
    elif activation_type == 'gelu':
        return nn.GELU()
    elif activation_type == 'glu':
        return nn.GLU()
    else:
        raise RuntimeError(f"activation function currently support relu/gelu, not {activation_type}")


def get_mlp_head(input_size, hidden_size, output_size, dropout=0, activation='relu'):
    return nn.Sequential(*[
        nn.Linear(input_size, hidden_size),
        get_activation_layer(activation),
        nn.LayerNorm(hidden_size, eps=1e-12),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, output_size)
    ])


def layer_repeat(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N - 1)] + [module])


#########################################################
# Specific modules helpers
#########################################################
def generate_fourier_features(pos, num_bands=10, max_freq=15, concat_pos=True, sine_only=False):
    # Input: B, N, C
    # Output: B, N, C'
    batch_size = pos.shape[0]
    device = pos.device

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.linspace(start=min_freq, end=max_freq, steps=num_bands, device=device)

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos.unsqueeze(-1).repeat(1, 1, 1, num_bands) * freq_bands
    per_pos_features = torch.reshape(
        per_pos_features, [batch_size, -1, np.prod(per_pos_features.shape[2:])])
    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat(
            [pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)
    return per_pos_features


def calc_pairwise_locs(obj_centers, obj_whls, eps=1e-10, pairwise_rel_type='center', spatial_dist_norm=True,
                       spatial_dim=5):
    if pairwise_rel_type == 'mlp':
        obj_locs = torch.cat([obj_centers, obj_whls], 2)
        pairwise_locs = torch.cat(
            [einops.repeat(obj_locs, 'b l d -> b l x d', x=obj_locs.size(1)),
             einops.repeat(obj_locs, 'b l d -> b x l d', x=obj_locs.size(1))],
            dim=3
        )
        return pairwise_locs

    pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
                    - einops.repeat(obj_centers, 'b l d -> b 1 l d')
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 3) + eps)  # (b, l, l)
    if spatial_dist_norm:
        max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
        norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, 'b -> b 1 1')
    else:
        norm_pairwise_dists = pairwise_dists

    if spatial_dim == 1:
        return norm_pairwise_dists.unsqueeze(3)

    pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)
    if pairwise_rel_type == 'center':
        pairwise_locs = torch.stack(
            [norm_pairwise_dists, pairwise_locs[..., 2] / pairwise_dists,
             pairwise_dists_2d / pairwise_dists, pairwise_locs[..., 1] / pairwise_dists_2d,
             pairwise_locs[..., 0] / pairwise_dists_2d],
            dim=3
        )
    elif pairwise_rel_type == 'vertical_bottom':
        bottom_centers = torch.clone(obj_centers)
        bottom_centers[:, :, 2] -= obj_whls[:, :, 2]
        bottom_pairwise_locs = einops.repeat(bottom_centers, 'b l d -> b l 1 d') \
                               - einops.repeat(bottom_centers, 'b l d -> b 1 l d')
        bottom_pairwise_dists = torch.sqrt(torch.sum(bottom_pairwise_locs ** 2, 3) + eps)  # (b, l, l)
        bottom_pairwise_dists_2d = torch.sqrt(torch.sum(bottom_pairwise_locs[..., :2] ** 2, 3) + eps)
        pairwise_locs = torch.stack(
            [norm_pairwise_dists,
             bottom_pairwise_locs[..., 2] / bottom_pairwise_dists,
             bottom_pairwise_dists_2d / bottom_pairwise_dists,
             pairwise_locs[..., 1] / pairwise_dists_2d,
             pairwise_locs[..., 0] / pairwise_dists_2d],
            dim=3
        )

    if spatial_dim == 4:
        pairwise_locs = pairwise_locs[..., 1:]
    return pairwise_locs

def predict_with_beam_search(model, hidden_states, tokenizer, num_beams=5, 
                             batch_size=4, device='cuda', repetition_penalty=3.0):
    """
    Predict with beam search decoding
    """
    predicted_tokens = [[] for _ in range(batch_size*num_beams)]
    predicted_indices = [[] for _ in range(batch_size*num_beams)]
    beam_scores = torch.zeros(batch_size, num_beams, device=device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)

    for i in range(len(hidden_states)):
        last_logits = model.lm_head(hidden_states[i][-1][:, -1, :])
        last_scores = F.log_softmax(last_logits, dim=-1)

        # logits processor (repetition penalty)
        input_ids = torch.LongTensor(predicted_tokens).to(device)
        score = torch.gather(last_scores, 1, input_ids)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        last_scores = last_scores.scatter(1, input_ids, score)

        beam_scores_new = last_scores + beam_scores[:, None].expand_as(last_scores)
        vocab_size = last_scores.shape[-1]
        beam_scores_new = beam_scores_new.view(batch_size, -1)
        next_token_scores, next_tokens = torch.topk(beam_scores_new, num_beams, dim=1, largest=True, sorted=True)
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor").view(-1)
        next_tokens = (next_tokens % vocab_size).view(-1)

        next_predicted_tokens = []
        next_predicted_indices = []
        for j in range(batch_size):
            for k in range(num_beams):
                idx = j * num_beams + k
                beam_idx = j * num_beams + next_indices[idx].item()
                next_predicted_tokens.append(predicted_tokens[beam_idx] + [next_tokens[idx].item()])
                next_predicted_indices.append(predicted_indices[beam_idx] + [beam_idx])

        predicted_tokens = next_predicted_tokens
        predicted_indices = next_predicted_indices
        beam_scores = next_token_scores.view(-1)

    predicted_tokens = torch.LongTensor(predicted_tokens).view(batch_size, num_beams, -1)
    predicted_indices = torch.LongTensor(predicted_indices).view(batch_size, num_beams, -1)

    predicted_tokens = predicted_tokens[:, 0, :]
    predicted_indices = predicted_indices[:, 0, :]

    predicted_sequences = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)

    return predicted_tokens, predicted_sequences
