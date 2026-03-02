import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from transformers import LlamaTokenizer, LlamaForCausalLM
from data.cot_utils import GRD_TXT_LEFT_TOKEN_ID, GRD_TXT_RIGHT_TOKEN_ID, OBJ_LOC_PLR_START_TOKEN_ID, OBJ_LOC_PLR_END_TOKEN_ID

def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(*[
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.LayerNorm(hidden_size, eps=1e-12),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, output_size)
    ])

class OBJMaskDecoder(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        cfg = SimpleNamespace(**cfg)
        self.obj_token_head = nn.Linear(4096, cfg.hidden_dim)
        self.ground_head_obj = nn.Sequential(
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(cfg.hidden_dim),
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                )
        self.ground_head_query = nn.Sequential(
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(cfg.hidden_dim),
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                )
        try:
            self.ground_head_temperature = cfg.ground_head_temperature
        except:
            self.ground_head_temperature = 0.07
        
        self.ground_head_zero_target = torch.nn.Parameter(torch.randn(cfg.hidden_dim))
        
        if cfg.grounding_module_type == 'cross_attention':
            self.attention = nn.MultiheadAttention(embed_dim=cfg.hidden_dim, num_heads=cfg.num_heads, batch_first=True)
        elif cfg.grounding_module_type == 'concat':
            self.linear_head = get_mlp_head(cfg.hidden_dim * 2, cfg.hidden_dim // 2, 1, dropout=0.1 )
        self.bce_loss = nn.BCELoss() 
        self.obj_mask_token_idx = cfg.obj_mask_token_idx
        self.cfg = cfg
        
    def pred_object_logits(self, obj_tokens, hidden_states, obj_masks, device):
        
        B, N, C = obj_tokens.shape  
        
        if obj_tokens.shape[-1] != hidden_states.shape[-1]:
            obj_tokens_projected = self.obj_token_head(obj_tokens)  
        else:
            obj_tokens_projected = obj_tokens

        if self.cfg.loss_type == 'infonce':
            obj_masks = torch.cat([obj_masks, torch.ones(1,1).to(device)], dim=1)
            object_features = torch.cat([obj_tokens.squeeze(0), self.ground_head_zero_target.unsqueeze(0).to(device)], dim=0)
            self.ground_head_obj = self.ground_head_obj.to(device)
            obj_feat = self.ground_head_obj(object_features.to(hidden_states.dtype))
            self.ground_head_query = self.ground_head_query.to(device)
            query_feat = self.ground_head_query(hidden_states)
            obj_feat = F.normalize(obj_feat)
            query_feat = F.normalize(query_feat)
            scores = (obj_feat * query_feat).sum(dim=-1)
            scores = scores.masked_fill(obj_masks.logical_not(), -1e8)
            return scores
        # Add dummy target for other grounding types
        # dummy_target = self.ground_head_zero_target.unsqueeze(0)
        # obj_tokens_projected = torch.cat([obj_tokens_projected, dummy_target.unsqueeze(1)], dim=1)

        if self.cfg.grounding_module_type == 'cos_sim':
            obj_tokens_projected = obj_tokens_projected / obj_tokens_projected.norm(dim=-1, keepdim=True)  
            obj_feat = self.ground_head_obj(obj_tokens_projected)
            query_feat = self.ground_head_query(hidden_states)
            query_feat_broadcast = hidden_states.unsqueeze(1).expand(B, N, -1)  
            query_feat_broadcast = query_feat_broadcast[:, 0, :].unsqueeze(2)
            obj_logits = torch.bmm(obj_feat, query_feat_broadcast).squeeze(-1)  

        elif self.cfg.grounding_module_type == 'concat':
            query_feat = self.ground_head_query(hidden_states)
            query_feat_broadcast = query_feat.unsqueeze(1).expand(B, N, -1)  
            obj_feat = self.ground_head_obj(obj_tokens_projected)
            obj_logits = self.linear_head(torch.cat([obj_feat, query_feat_broadcast], dim=2))

        else:
            raise ValueError("Invalid grounding_module_type")

        if len(obj_logits.shape) > len(obj_masks.shape):
            if len(obj_logits.shape) == len(obj_masks.shape) + 1 and  obj_logits.shape[-1] == 1:
                obj_logits = obj_logits.squeeze(-1)
        assert len(obj_logits.shape) == len(obj_masks.shape)
        if obj_logits.shape[0] > obj_masks.shape[0]:
            # HACK: This is a temporary fix for the issue where obj_logits has more tokens than obj_masks. During inference time, the obj_logits with wrong shape will be discarded.
            obj_logits = obj_logits[:obj_masks.shape[0]]
        elif obj_logits.shape[0] < obj_masks.shape[0]:
            obj_logits = torch.cat([obj_logits, torch.zeros(obj_masks.shape[0] - obj_logits.shape[0], obj_logits.shape[1])], dim=0)
        obj_logits = obj_logits.masked_fill_(obj_masks.logical_not(), -1e8)  # Mask out the invalid tokens
        if torch.isnan(obj_logits).any():
            print("obj_logits has nan")
            print(f'max obj_logits: {torch.max(obj_logits)}')
            print(f'min obj_logits: {torch.min(obj_logits)}')
            print("obj_logits: ", obj_logits)
            print("obj_tokens_projected: ", obj_tokens_projected)
            if torch.isnan(obj_tokens_projected).any():
                print("obj_tokens_projected has nan")
            if torch.isnan(hidden_states).any():
                print("hidden_states has nan")
            if torch.isnan(obj_feat).any():
                print("obj_feat has nan")

        return obj_logits

    def obj_mask_loss(self, pred_obj_logits, ground_obj_mask, device):
        if len(pred_obj_logits.shape) > len(ground_obj_mask.shape):
            pred_obj_logits = pred_obj_logits.squeeze(-1)
        # ground_obj_mask = torch.cat([ground_obj_mask, torch.ones(1,1).to(device)], dim=1)
        if self.cfg.loss_type == 'ce':
            pred_obj_logits = torch.clamp(pred_obj_logits, min=-100, max=100)
            if ground_obj_mask.sum() == 0:
                ground_obj_mask = torch.zeros_like(pred_obj_logits)
            else:
                ground_obj_mask = ground_obj_mask/ground_obj_mask.sum()
            loss = F.cross_entropy(pred_obj_logits, ground_obj_mask)
            if torch.isnan(loss):
                print("loss is nan")
                print(f'max pred_obj_logits: {torch.max(pred_obj_logits)}')
                print(f'min pred_obj_logits: {torch.min(pred_obj_logits)}')
                print("pred_obj_logits: ", pred_obj_logits)
                print("ground_obj_mask: ", ground_obj_mask)
                print(f'sum of ground_obj_mask: {ground_obj_mask.sum()}')
        elif self.cfg.loss_type == 'bce':
            pred_obj_logits = torch.clamp(pred_obj_logits, min=-100, max=100)
            pred_obj_mask = torch.sigmoid(pred_obj_logits)
            loss = self.bce_loss(pred_obj_mask, ground_obj_mask)
        else:
            ValueError("Invalid loss_type")
        return loss
    
    def infonce_loss(self, pred_scores, ground_obj_mask, device=None):
        # gt_labels = torch.where(ground_obj_mask.squeeze(0) == 1)[0]
        # if len(gt_labels) == 0: # zero-target
        #     gt_labels.append(-1)
        # logits = torch.exp(pred_scores / self.ground_head_temperature)
        # print(f'shape of gt_labels: {gt_labels.shape}')
        # loss = - torch.log( logits[gt_labels].sum() / logits.sum())
        # print(f'grd infonce loss: {loss}')
        gt_labels = torch.where(ground_obj_mask.squeeze(0) == 1)[0]
        ground_obj_mask = torch.cat([ground_obj_mask, torch.ones(1,1).to(device)], dim=1)
        if len(gt_labels) == 0:
            gt_labels = torch.tensor([-1]).to(device)
        if torch.isnan(pred_scores).any():
            print("pred_scores has nan")
            print(f'max pred_scores: {torch.max(pred_scores)}')
            print(f'min pred_scores: {torch.min(pred_scores)}')
        logits = torch.exp((pred_scores / self.ground_head_temperature).clamp(max=100))
        if len(logits.shape) > len(gt_labels.shape):
            logits = logits.squeeze(0)     # (, Num_obj_tokens)
        positive_logits = logits[gt_labels]
        negative_logits_sum = logits.sum() - positive_logits.sum()
        eps = 1e-8  # Small constant
        loss = -torch.log(positive_logits / (negative_logits_sum + positive_logits + eps)).mean()
        print(f'grd infonce loss: {loss}')
        return loss

    def forward(self, input_ids,  hidden_states, obj_tokens, obj_mask_token_idx, obj_masks,gt_input_ids=None, gt_grounding_obj_mask=None, llm_model=None, device=None):
        
        all_pred_obj_mask = []
        all_obj_mask_loss = []
        bs = obj_tokens.shape[0]                                                                 
        if len(gt_grounding_obj_mask.shape) == 2:
            gt_grounding_obj_mask = gt_grounding_obj_mask.unsqueeze(1)
        if gt_input_ids is None:
            for i in range(bs):
                pred_obj_tokens_indices = torch.where(input_ids[i] == obj_mask_token_idx)[0]
                pred_grd_left_indices = torch.where(input_ids[i] == GRD_TXT_LEFT_TOKEN_ID)[0]
                pred_grd_right_indices = torch.where(input_ids[i] == GRD_TXT_RIGHT_TOKEN_ID)[0]
                if len(pred_obj_tokens_indices) == 0 or len(gt_grounding_obj_mask[i]) != len(pred_obj_tokens_indices) \
                or len(pred_grd_left_indices) == 0 or len(pred_grd_right_indices) == 0:
                    if self.cfg.loss_type == 'infonce':
                        pseduo_ground_obj_mask = torch.cat([gt_grounding_obj_mask[i], torch.ones(1,1).to(device)], dim=1)
                        all_pred_obj_mask.append(torch.zeros_like(pseduo_ground_obj_mask).to(device))
                    else:
                        all_pred_obj_mask.append(torch.zeros_like(gt_grounding_obj_mask[i]).to(device))
                    continue
                valid_obj_tokens = []
                valid_hidden_states = []
                valid_obj_masks = []
                for j in range(len(pred_obj_tokens_indices)):
                    valid_obj_tokens.append(obj_tokens[i])
                    if self.cfg.grd_text_hidden_states == 'special_token':
                        valid_hidden_states.append(hidden_states[i][pred_obj_tokens_indices[j]].squeeze(0))
                    elif self.cfg.grd_text_hidden_states == 'average_embedding':
                        obj_start_idx = pred_grd_left_indices[0]
                        obj_end_idx = pred_grd_right_indices[0]
                        obj_cap_hidden_states = llm_model.get_input_embeddings()(input_ids[i][obj_start_idx+1:obj_end_idx]).float()
                        valid_hidden_states.append(obj_cap_hidden_states.mean(dim=0))
                    valid_obj_masks.append(obj_masks[i])
                valid_obj_tokens = torch.stack(valid_obj_tokens)
                valid_hidden_states = torch.stack(valid_hidden_states)
                valid_obj_masks = torch.stack(valid_obj_masks)
                pred_obj_logits = self.pred_object_logits(valid_obj_tokens, valid_hidden_states, valid_obj_masks, device)
                pred_obj_mask = torch.sigmoid(pred_obj_logits)
                all_pred_obj_mask.append(pred_obj_mask)
            all_pred_obj_mask = torch.stack(all_pred_obj_mask)
            if all_pred_obj_mask.shape[-1] > obj_masks.shape[-1]:
                all_pred_obj_mask = all_pred_obj_mask[:, :,:-1]  # remove the dummy target
            return all_pred_obj_mask
        
        for i in range(bs):
            gt_obj_tokens_indices = torch.where(gt_input_ids[i] == obj_mask_token_idx)[0]
            pred_obj_tokens_indices = torch.where(input_ids[i] == obj_mask_token_idx)[0]
            pred_grd_left_indices = torch.where(input_ids[i] == GRD_TXT_LEFT_TOKEN_ID)[0]
            pred_grd_right_indices = torch.where(input_ids[i] == GRD_TXT_RIGHT_TOKEN_ID)[0]
            gt_grd_left_indices = torch.where(gt_input_ids[i] == GRD_TXT_LEFT_TOKEN_ID)[0]
            gt_grd_right_indices = torch.where(gt_input_ids[i] == GRD_TXT_RIGHT_TOKEN_ID)[0]
            # if len(pred_obj_tokens_indices) == 0 or len(gt_grounding_obj_mask[i]) != len(pred_obj_tokens_indices) \
            #     or len(pred_grd_left_indices) == 0 or len(pred_grd_right_indices) == 0 or pred_grd_right_indices[0] - pred_grd_left_indices[0] < 2:
            #     if self.cfg.loss_type == 'infonce':
            #         pseduo_ground_obj_mask = torch.cat([gt_grounding_obj_mask[i], torch.ones(1,1).to(device)], dim=1)
            #         all_pred_obj_mask.append(torch.zeros_like(pseduo_ground_obj_mask).to(device))
            #     else:
            #         all_pred_obj_mask.append(torch.zeros_like(gt_grounding_obj_mask[i]).to(device))
            #     all_obj_mask_loss.append(torch.zeros(1).squeeze(0).to(device))
            #     continue
            valid_obj_tokens = []
            valid_hidden_states = []
            valid_obj_masks = []
            valid_gt_grounding_obj_mask = []
            obj_seq_idx = 0
            for j in range(len(gt_obj_tokens_indices)):
                valid_obj_tokens.append(obj_tokens[i])
                if self.cfg.grd_text_hidden_states == 'special_token':
                    valid_hidden_states.append(hidden_states[i][gt_obj_tokens_indices[j]].squeeze(0))
                elif self.cfg.grd_text_hidden_states == 'average_embedding':
                    obj_start_idx = gt_grd_left_indices[0]     # NOTE: only support one grounding text for now
                    obj_end_idx = gt_grd_right_indices[0]
                    try:
                        obj_cap_hidden_states = llm_model.get_input_embeddings()(gt_input_ids[i][obj_start_idx+1:obj_end_idx]).float()
                    except:
                        print(f'gt_input_ids[i]: {gt_input_ids[i].cpu().tolist()}')
                        print(f'gt_input_ids[i] shape: {gt_input_ids[i].shape}')
                        print(f'obj_start_idx: {obj_start_idx}, obj_end_idx: {obj_end_idx}')
                    valid_hidden_states.append(obj_cap_hidden_states.mean(dim=0))
                valid_obj_masks.append(obj_masks[i])
                valid_gt_grounding_obj_mask.append(gt_grounding_obj_mask[i][obj_seq_idx].squeeze(0))
                obj_seq_idx += 1
            valid_obj_tokens = torch.stack(valid_obj_tokens)
            assert valid_obj_tokens.shape[0] == len(valid_hidden_states)
            valid_hidden_states = torch.stack(valid_hidden_states)
            valid_obj_masks = torch.stack(valid_obj_masks)
            valid_gt_grounding_obj_mask = torch.stack(valid_gt_grounding_obj_mask)  # 
            pred_obj_logits = self.pred_object_logits(valid_obj_tokens, valid_hidden_states, valid_obj_masks, device)
            if torch.isnan(pred_obj_logits).any():
                print(f"valid_hidden_states: {valid_hidden_states}")
                print(f"gt_input_ids: {gt_input_ids[i]}")
            if self.cfg.loss_type == 'infonce':
                obj_mask_loss = self.infonce_loss(pred_obj_logits, valid_gt_grounding_obj_mask, device)
            else:
                obj_mask_loss = self.obj_mask_loss(pred_obj_logits, valid_gt_grounding_obj_mask, valid_obj_masks)
            pred_obj_mask = torch.sigmoid(pred_obj_logits)
            all_pred_obj_mask.append(pred_obj_mask)
            all_obj_mask_loss.append(obj_mask_loss)
        obj_mask_loss = torch.stack(all_obj_mask_loss)
        all_pred_obj_mask = torch.stack(all_pred_obj_mask)
        print(f'in grd module: obj_mask_loss: {obj_mask_loss}')

        if all_pred_obj_mask.shape[-1] > obj_masks.shape[-1]:
            all_pred_obj_mask = all_pred_obj_mask[:, :,:-1]  # remove the dummy target

        return obj_mask_loss, all_pred_obj_mask