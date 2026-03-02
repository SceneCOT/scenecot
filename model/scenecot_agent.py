import copy
import math
from contextlib import nullcontext

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from einops import rearrange
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
                         T5ForConditionalGeneration, T5Tokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor

from model.build import build_module
from model.llava.llava_llama import LlavaLlamaForCausalLM
from model.transformers import AttentionPooling
from model.utils import disabled_train, maybe_autocast, predict_with_beam_search
from model.grounding_module import OBJMaskDecoder
from data.cot_utils import *
from model.llava.llava_llama import load_pretrained_llava

logger = get_logger(__name__)

torch_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

class SceneCOTAgent(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # LLM
        self.llm_is_decoder_only = True
        if 'vicuna' in cfg.llm.name.lower():
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.llm_model = LlamaForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.float16)
            self.llm_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        elif 'opt' in cfg.llm.name.lower():
            self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.llm_model = AutoModelForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.float16)
        elif 'llama3' in cfg.llm.name.lower():
            self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.llm_model = AutoModelForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.bfloat16)
            self.llm_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        elif 'gemma' in cfg.llm.name.lower():
            self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.llm_model = AutoModelForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.bfloat16)
        elif 't5' in cfg.llm.name.lower():
            self.llm_is_decoder_only = False
            self.llm_tokenizer = T5Tokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.llm_model = T5ForConditionalGeneration.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.float16)
        elif 'llava' in cfg.llm.name.lower():
            self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, use_fast=False, truncation_side=cfg.llm.truncation_side)
            self.llm_model = LlavaLlamaForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.float16)
        elif 'qwen' in cfg.llm.name.lower():
            model_kwargs = {'torch_dtype': torch_dtype_map[cfg.llm.torch_dtype]}
            if getattr(cfg.llm, 'attn_implementation', None):
                model_kwargs['attn_implementation'] = cfg.llm.attn_implementation
            self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.llm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.llm.cfg_path, **model_kwargs)
            self.use_chat_template = True
        else:
            raise NotImplementedError

        logger.info(f"Build {cfg.llm.name} from {cfg.llm.cfg_path}")

        for param in self.llm_model.parameters():
            param.requires_grad = False
        # self.llm_model.eval()
        # self.llm_model.train = disabled_train
        self.blind = cfg.blind
        logger.info(f"Freeze LLM, blind={self.blind}")

        # 2D vision
        # self.img_encoder = build_module(cfg.vision2d)
        # self.img_proj = nn.Linear(
        #     self.img_encoder.out_channels, self.llm_model.config.hidden_size
        # )

        # 2D image encoder
        if 'llava' in cfg.llm.name.lower():
            model_path = cfg.llm.cfg_path
            vision_model_path = cfg.llm.vision_model_path
            tokenizer, mm_model, self.img_transform = load_pretrained_llava(model_path=model_path, vision_tower_path=vision_model_path)
            self.vision_tower = copy.deepcopy(mm_model.get_model().get_vision_tower())
            self.mm_projector = copy.deepcopy(mm_model.get_model().mm_projector)
            self.mm_projector = self.mm_projector.float()
            print(f"vision_tower built from llava")
            print(f"mm_projector built from llava")
            if not cfg.cot.freeze_mm_projector:
                for p in self.mm_projector.parameters():
                    p.requires_grad = True
                print("enable mm_projector training")
            else:
                for p in self.mm_projector.parameters():
                    p.requires_grad = False
                print("freeze mm_projector")
        elif 'qwen' in cfg.llm.name.lower():
            self.vision_tower = copy.deepcopy(self.llm_model.model.vision_encoder)
            self.mm_projector = copy.deepcopy(self.llm_model.model.vision_projector)
            self.mm_projector = self.mm_projector.float()
            print(f"vision_tower built from qwen2.5-vl")
            print(f"mm_projector built from qwen2.5-vl")
            if not cfg.cot.freeze_mm_projector:
                for p in self.mm_projector.parameters():
                    p.requires_grad = True
                print("enable mm_projector training")
            else:
                for p in self.mm_projector.parameters():
                    p.requires_grad = False
                print("freeze mm_projector")

        # 3D vision
        self.pcd_encoder = build_module(cfg.vision3d)
        if 'llava' in cfg.llm.name.lower() and not hasattr(self, 'vision_tower'):
            self.pcd_proj = copy.deepcopy(self.llm_model.get_model().mm_projector)
            self.pcd_proj.float()
            for p in self.pcd_proj.parameters():
                p.requires_grad = True
        
        else:
            self.pcd_proj = nn.Linear(
                cfg.vision3d.hidden_dim, self.llm_model.config.hidden_size
            )
        self.vision3d_context = nullcontext
        if hasattr(cfg.vision3d, 'freeze') and cfg.vision3d.freeze:
            for p in self.pcd_encoder.parameters():
                p.requires_grad = False
            self.pcd_encoder.eval()
            self.pcd_encoder.train = disabled_train
            for p in self.pcd_proj.parameters():
                p.requires_grad = False
            self.pcd_proj.eval()
            self.pcd_proj.train = disabled_train
            self.vision3d_context = torch.no_grad
            logger.info(f"Freeze 3D encoder")

        # 3D vision object centric
        self.obj_encoder = build_module(cfg.vision3d_obj)

        # txt projection for grounding
        # self.txt_pool = AttentionPooling(self.llm_model.config.hidden_size, self.llm_model.config.num_attention_heads)

        # type embedding
        # self.img_type_embed = nn.Parameter(torch.zeros(self.llm_model.config.hidden_size), requires_grad=True)
        # self.pcd_type_embed = nn.Parameter(torch.zeros(self.llm_model.config.hidden_size), requires_grad=True)

        # LoRA
        if cfg.llm.lora.flag:
            logger.info(f"Apply LoRA with configs: {cfg.llm.lora}")
            lora_config = LoraConfig(
                r=cfg.llm.lora.rank,
                lora_alpha=cfg.llm.lora.alpha,
                target_modules=cfg.llm.lora.target_modules,
                lora_dropout=cfg.llm.lora.dropout,
                bias='none',
                modules_to_save=[],
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config=lora_config)
        else:
            self.llm_model.eval()
            self.llm_model.train = disabled_train

        self.max_context_len = cfg.llm.max_context_len
        self.max_out_len = cfg.llm.max_out_len

        # additional text x multi-modal tokens fusion
        self.clip_txt_guidance = cfg.clip_txt_guidance.flag
        if self.clip_txt_guidance:
            logger.info("Add CLIP semantics guidance")
            self.clip_model = clip.load('RN50')[0]
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
            self.clip_model.train = disabled_train
            self.clip_proj = nn.Linear(cfg.clip_txt_guidance.clip_out_dim, self.llm_model.config.hidden_size)

        # grounding module
        if hasattr(cfg, 'grounding') and cfg.grounding.enable:
            self.grounding_head = OBJMaskDecoder(cfg.grounding)
            logger.info(f"Build grounding module: {cfg.grounding.loss_type}")
            self.grd_token_id =self.llm_tokenizer(GRD_TOKEN_TXT, add_special_tokens=False).input_ids[-1]
            print('found grd_token_id:', self.grd_token_id)

            # Unfreeze the last N layers of the LLM
            if cfg.grounding.unfreeze_llm_layer.flag:
                num_unfreeze = cfg.grounding.unfreeze_llm_layer.unfreeze_layers_num  # Specify the number of layers to unfreeze

                for layer in self.llm_model.model.model.layers[-num_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                
                print(f"Unfreezing the last {num_unfreeze} layers of the LLM")

            if cfg.vision3d.name.lower()== "pq3d":
                self.grd_use_scene_tokens = True
            else:
                self.grd_use_scene_tokens = False

        if hasattr(cfg, 'cot'):
            self.cot_mask_obj_prob_loc_token = cfg.cot.mask_obj_prob_loc_token  # True or False
            if hasattr(cfg.cot, 'cot_answer_training'):
                self.cot_answer_training = cfg.cot.cot_answer_training
            self.max_cot_obj_num = cfg.cot.max_cot_obj_num
            self.use_oracle_obj_content = cfg.cot.use_oracle_obj_content

            print(f"cot_mask_obj_prob_loc_token: {self.cot_mask_obj_prob_loc_token}")

            if hasattr(cfg.cot, 'cot_no_scene_tokens'):
                self.cot_no_scene_tokens = cfg.cot.cot_no_scene_tokens

            print(f"cot_no_scene_tokens: {self.cot_no_scene_tokens}")

        self.cfg = cfg

        # NOTE: MOE configs,
        # load the predicted object mask by the best counting/existence expert
        print(f"cfg.grounding:")
        print(cfg.grounding)
        if hasattr(cfg.grounding, "moe_flag") and cfg.grounding.moe_flag:
            self.moe_flag = cfg.grounding.moe_flag
            self.moe_type_list = cfg.grounding.moe_type_list
            self.obj_prob_dict = torch.load(cfg.grounding.obj_prob_dict_path)

    def encode_images(self, images):
        if hasattr(self, 'vision_tower') and self.cot_no_scene_tokens:
            image_features = self.vision_tower(images)
            image_features = self.mm_projector(image_features)
            return image_features
        else:
            ValueError("No vision tower found!")

    @property
    def device(self):
        return list(self.parameters())[0].device

    def count_params(self, parameters):
        tot = sum([math.prod(p.shape) for p in parameters])
        return tot

    def show_params_size(self, tot):
        if tot >= 1e9:
            return '{:.1f}B'.format(tot / 1e9)
        elif tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}k'.format(tot / 1e3)

    def get_learnable_named_params(self):
        learnable_named_params = {}
        frozen_named_params = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                learnable_named_params.update({n: p})
            else:
                frozen_named_params.update({n: p})
        learnable_params_size = self.count_params(learnable_named_params.values())
        frozen_params_size = self.count_params(frozen_named_params.values())
        logger.info(
            f"Build SceneCOT with {self.show_params_size(learnable_params_size+frozen_params_size)} parameters, " +
            f"{self.show_params_size(learnable_params_size)} learnable and " +
            f"{self.show_params_size(frozen_params_size)} frozen"
        )
        # logger.info(f"🧊 Frozen parameters: {list(frozen_named_params.keys())}")
        # logger.info(f"🔥 Tuned parameters: {list(learnable_named_params.keys())}")

        return learnable_named_params

    def build_input_sequence(self, data_dict):
        """
        Return input sequence for causal LM: <scene>, <img>, <pad>, <instruction>.
        - Vicuna sequence:
            `scene_prompt_tokens`, `scene_tokens`, `egoview_prompt_tokens`, `img_tokens`, `input_txt_tokens`.
        - LLaMA3 sequence:
            <|start_header_id|>system<|end_header_id|>\n\n3D scene: [obj]. Ego-view image: img.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>\n\n
        """
        device = self.device
        bs = len(data_dict['input_txt'])

        input_txt_tokens = data_dict['input_txt_ids']
        input_txt_masks = data_dict['input_txt_mask']

        pad_length = torch.logical_not(input_txt_masks).sum(-1)
        input_txt_tokens_right_justified = input_txt_tokens.clone()
        input_txt_masks_right_justified = input_txt_masks.clone()
        for i in range(bs):
            this_pad_len = pad_length[i]
            if this_pad_len > 0:
                input_txt_tokens_right_justified[i, :this_pad_len] = input_txt_tokens[i, -this_pad_len:]
                input_txt_masks_right_justified[i, :this_pad_len] = 0
                input_txt_tokens_right_justified[i, this_pad_len:] = input_txt_tokens[i, :-this_pad_len]
                input_txt_masks_right_justified[i, this_pad_len:] = 1

        input_txt_embeds = self.llm_model.get_input_embeddings()(input_txt_tokens_right_justified)

        input_embeds = torch.cat([input_txt_embeds], dim=1)
        attn_masks = torch.cat([input_txt_masks_right_justified], dim=1)

        return input_embeds, attn_masks
       
    def build_new_inputs_embeds_targets(self, inputs_embeds, attention_mask, text_output_embeds, text_output_mask, text_output_ids, data_dict):

        inputs_embeds = torch.cat([inputs_embeds, text_output_embeds], dim=1)   
        attention_mask = torch.cat([attention_mask, text_output_mask], dim=1)  

        targets = torch.zeros_like(attention_mask).long().fill_(-100)   
        targets_idx = text_output_mask.bool()
        targets[:, -targets_idx.shape[1]:][targets_idx] = text_output_ids[targets_idx]
    
        new_targets_list = []
        new_inputs_embeds_list = [] 
        new_attention_mask_list = []
        for i in range(inputs_embeds.shape[0]):
            img_indicator_idx = torch.where(targets[i] == HIGHLIGHT_OBJ_TOKEN_ID)[0]
            if len(img_indicator_idx) == 0:
                new_targets_list.append(targets[i])
                new_inputs_embeds_list.append(inputs_embeds[i])
                new_attention_mask_list.append(attention_mask[i])
                continue
            img = data_dict['gt_img'][i]
            img = img.unsqueeze(0)   # (1, 3, H, W)
            self.mm_projector=self.mm_projector.float()
            img_embeds = self.mm_projector(self.vision_tower(img.to(self.device))) #(1, 3, H, W) -> (1, 576, D)
            img_start_embeds = self.llm_model.get_input_embeddings()(torch.tensor(IMG_START_TOKEN_ID).to(self.device)).unsqueeze(0).unsqueeze(0)  
            img_end_embeds = self.llm_model.get_input_embeddings()(torch.tensor(IMG_END_TOKEN_ID).to(self.device)).unsqueeze(0).unsqueeze(0)   
            img_embeds_with_pre_suffix = torch.cat([img_start_embeds, img_embeds, img_end_embeds], dim=1)   # (1, 578, D)
            img_masks = torch.ones(img_embeds_with_pre_suffix.shape[:2], dtype=torch.int64, device=self.device) 
            inputs_embeds_with_img_embeds_prefix = torch.cat([inputs_embeds[i:i+1, :img_indicator_idx+1, :], img_embeds_with_pre_suffix], dim=1)
            attention_mask_with_img_embeds_prefix = torch.cat([attention_mask[i:i+1, :img_indicator_idx+1], img_masks], dim=1)
            targets_suffix = targets[i:i+1, img_indicator_idx+1:]  # after <highlight_obj>, do not include <highlight_obj>
            inputs_embeds_suffix = inputs_embeds[i:i+1, img_indicator_idx+1:, :]
            attention_mask_suffix = attention_mask[i:i+1, img_indicator_idx+1:]
            inputs_embeds_with_img = torch.cat([inputs_embeds_with_img_embeds_prefix, inputs_embeds_suffix], dim=1)   
            attention_mask_with_img = torch.cat([attention_mask_with_img_embeds_prefix, attention_mask_suffix], dim=1)   
            targets_img_mask = torch.zeros_like(img_masks).long().fill_(-100)   
            new_targets = torch.cat([targets[i:i+1, :img_indicator_idx+1], targets_img_mask, targets_suffix], dim=1)      
            assert new_targets.shape[1] == inputs_embeds_with_img.shape[1]
            new_targets_list.append(new_targets.squeeze(0))   
            new_inputs_embeds_list.append(inputs_embeds_with_img.squeeze(0))
            new_attention_mask_list.append(attention_mask_with_img.squeeze(0))    
        
        max_tensor_len = max([t.shape[0] for t in new_targets_list])
        new_targets = torch.zeros(inputs_embeds.shape[0], max_tensor_len).long().fill_(-100).to(self.device)
        new_inputs_embeds = torch.zeros(inputs_embeds.shape[0], max_tensor_len, inputs_embeds.shape[-1]).to(self.device)
        new_attention_mask = torch.zeros(inputs_embeds.shape[0], max_tensor_len).long().fill_(0).to(self.device)
        for i in range(len(new_targets_list)):
            new_targets[i, :new_targets_list[i].shape[0]] = new_targets_list[i]
            new_inputs_embeds[i, :new_inputs_embeds_list[i].shape[0], :] = new_inputs_embeds_list[i]
            new_attention_mask[i, :new_attention_mask_list[i].shape[0]] = new_attention_mask_list[i]

        assert new_targets.shape[1] == new_inputs_embeds.shape[1] == new_attention_mask.shape[1]
        
        return new_inputs_embeds, new_attention_mask, new_targets

    def compute_grounding_logits(self, obj_embeds, txt_embeds, obj_masks):
        """
        obj_embeds: [B, N, D]
        txt_embeds: [B, T, D]
        obj_masks: [B, N], 1 for valid and 0 for pad
        """
        obj_masks = obj_masks.bool()
        txt_feat = self.txt_pool(txt_embeds)   # [B, D]
        txt_feat = F.normalize(txt_feat, dim=-1)   # [B, D]
        obj_feat = F.normalize(obj_embeds, dim=-1)   # [B, N, D]
        sim = torch.einsum('bnd,bd->bn', obj_feat, txt_feat)   # [B, N]
        return sim.masked_fill_(~obj_masks, -torch.inf)

    def forward(self, data_dict):
        """
        data_dict requires keys:
        # input
        scene_prompt_tokens: (B, L1)
        egoview_prompt_tokens: (B, L2)
        input_txt: list of str, (B,)
        obj_fts: (B, N, P, 6), xyz + rgb
        obj_masks: (B, N), 1 valid and 0 masked
        obj_locs: (B, N, 6), xyz + whd
        tgt_obj_idx: [B], int between [0, N-1] or -100
        anchor_loc: (B, 3)
        img_fts: (B, 3, H, W), rgb
        img_masks: (B, 1), 1 valid and 0 masked
        # output
        output_gt: list of str, (B,)
        """
        device = self.device
        bs = len(data_dict['input_txt'])

        # get embedding of input_txt
        self.llm_tokenizer.padding_side = 'right'
        self.llm_tokenizer.truncation_side = 'left'
        input_txt_tokenized = self.llm_tokenizer(
            data_dict['input_txt'],
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=False,
        ).to(device)   # [BOS, tokens, PAD]
        data_dict['input_txt_ids'] = input_txt_tokenized.input_ids
        data_dict['input_txt_embed'] = self.llm_model.get_input_embeddings()(input_txt_tokenized.input_ids)
        data_dict['input_txt_mask'] = input_txt_tokenized.attention_mask

        with self.vision3d_context():
            if self.blind:
                data_dict['scene_tokens'] = torch.zeros(bs, 1, data_dict['input_txt_embed'].shape[-1]).to(device)
                data_dict['scene_masks'] = torch.zeros(bs, 1, dtype=torch.bool).to(device)
            else:
                try:
                    data_dict = self.pcd_encoder(data_dict)
                except:
                    print(f'in leo_cot_agent.py line 215: error in pcd_encoder')
                    print(f'shape of obj_fts: {data_dict["obj_fts"].shape}')
                    print(f'shape of obj_masks: {data_dict["obj_masks"].shape}')
                    print(f'shape of obj_locs: {data_dict["obj_locs"].shape}')
                    print(f'shape of anchor_loc: {data_dict["anchor_loc"].shape}')
                data_dict['scene_tokens'] = self.pcd_proj(data_dict['scene_tokens'].to(device))

        inputs_embeds, attention_mask = self.build_input_sequence(data_dict=data_dict)
        # (B, T1+O+T2, D), (B, T1+O+T2)

        self.llm_tokenizer.padding_side = 'right'
        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in data_dict['output_gt']],
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_out_len,
            add_special_tokens=False,
        ).to(device)
        # deprecated, due to add_special_tokens=False
        # text_output_ids = text_output_tokens.input_ids[:, 1:]
        # text_output_mask = text_output_tokens.attention_mask[:, 1:]
        text_output_ids = text_output_tokens.input_ids
        text_output_mask = text_output_tokens.attention_mask
        text_output_embeds = self.llm_model.get_input_embeddings()(text_output_ids)   # (B, T3, D)

        if not self.llm_is_decoder_only:
            labels = text_output_ids.clone()
            labels[labels == self.llm_tokenizer.pad_token_id] = -100
            with maybe_autocast(self):
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            logits = outputs.logits.float()
            # empirical: encoder-decoder sequence does not need to shift
            num_tokens_for_loss = (labels >= 0).int().sum(1)   # (B,)
            logits = rearrange(logits, 'b t v -> (b t) v')
            labels = rearrange(labels, 'b t -> (b t)')
            loss = F.cross_entropy(logits, labels, reduction='none')
            loss = rearrange(loss, '(b t) -> b t', b=bs)
            loss = loss.sum(1) / num_tokens_for_loss   # (B,)
        else:
            
            inputs_embeds, attention_mask, targets = self.build_new_inputs_embeds_targets(inputs_embeds, attention_mask, text_output_embeds, text_output_mask, text_output_ids, data_dict)   
            # build new inputs_embeds and targets; 
            # insert image tokens; 
            # # mask the tokens before <highlight_obj> and after </highlight_obj>
            
            # mask token loss between <obj_prob>,</obj_prob; <obj_loc>,</obj_loc>; <obj_loc_plr>,</obj_loc_plr>   
            if hasattr(self, 'cot_mask_obj_prob_loc_token') and self.cot_mask_obj_prob_loc_token:
                assert self.max_out_len > 500    # shorter than 500 will miss the tokens after </obj_prob>

                for i in range(bs):
                    if len(torch.where(targets[i] == OBJ_PROB_START_TOKEN_ID)[0]) > 0 and len(torch.where(targets[i] == OBJ_PROB_END_TOKEN_ID)[0]) > 0:
                        start_idx = torch.where(targets[i] == OBJ_PROB_START_TOKEN_ID)[0]
                        end_idx = torch.where(targets[i] == OBJ_PROB_END_TOKEN_ID)[0]
                        targets[i, start_idx:end_idx+1] = -100    # including the start and end token
                    if len(torch.where(targets[i] == OBJ_LOC_START_TOKEN_ID)[0]) > 0 and len(torch.where(targets[i] == OBJ_LOC_END_TOKEN_ID)[0]) > 0:
                        start_idx = torch.where(targets[i] == OBJ_LOC_START_TOKEN_ID)[0]
                        end_idx = torch.where(targets[i] == OBJ_LOC_END_TOKEN_ID)[0]
                        targets[i, start_idx:end_idx+1] = -100
                    if len(torch.where(targets[i] == OBJ_LOC_PLR_START_TOKEN_ID)[0]) > 0 and len(torch.where(targets[i] == OBJ_LOC_PLR_END_TOKEN_ID)[0]) > 0:
                        start_idx = torch.where(targets[i] == OBJ_LOC_PLR_START_TOKEN_ID)[0]
                        end_idx = torch.where(targets[i] == OBJ_LOC_PLR_END_TOKEN_ID)[0]
                        targets[i, start_idx:end_idx+1] = -100
                    
            with maybe_autocast(self):
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )

            logits = outputs.logits.float()

            # different from the loss inside `llm_model.forward`, here we take mean of each sequence instead of sum
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            num_tokens_for_loss = (shift_labels >= 0).int().sum(1)   # (B,)

            shift_logits = rearrange(shift_logits, 'b t v -> (b t) v')
            shift_labels = rearrange(shift_labels, 'b t -> (b t)')

            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
            loss = rearrange(loss, '(b t) -> b t', b=bs)
            loss = loss.sum(1) / num_tokens_for_loss   # (B,)

        if hasattr(self, 'grounding_head'):
            modified_pred_input_ids = logits.contiguous().argmax(-1).clone()
            for batch_idx in range(bs):
                # bos and eos token index, mask the tokens before bos and after eos, to avoid unnecessary loss calculation
                if len(torch.nonzero(modified_pred_input_ids[batch_idx] == 2)) == 0 or len(torch.nonzero(text_output_tokens.input_ids[batch_idx] == 2)) == 0:
                    print(f"no eos token in batch {batch_idx}")
                    continue
                valid_len = text_output_tokens.input_ids[batch_idx].shape[0]
                end_idx = torch.nonzero(text_output_tokens.input_ids[batch_idx] == 2)[0][0]
                modified_pred_input_ids[batch_idx, :-valid_len] = -100
            last_hidden_states = outputs.hidden_states[-1]
            last_hidden_states = last_hidden_states.float()
            
            if self.grd_use_scene_tokens:
                # pq3d setting
                obj_tokens = data_dict['scene_tokens']
            else:
                data_dict = self.obj_encoder(data_dict)
                obj_tokens = self.pcd_proj(data_dict['obj_tokens'])

            obj_masks = data_dict['obj_masks']
            self.grounding_head = self.grounding_head.to(self.device)
            obj_mask_loss, all_pred_obj_mask = self.grounding_head(input_ids=modified_pred_input_ids, gt_input_ids=targets, hidden_states=last_hidden_states, obj_tokens=obj_tokens, obj_mask_token_idx=self.grd_token_id, obj_masks=obj_masks, gt_grounding_obj_mask=data_dict['grounding_obj_mask_gt'], llm_model=self.llm_model, device=self.device)
            data_dict['loss_grd'] = obj_mask_loss    # (B,)
            data_dict['pred_obj_prob'] = all_pred_obj_mask
            loss = loss + obj_mask_loss*self.cfg.grounding.grd_loss_weight

        data_dict['loss_gen'] = loss
        return data_dict

    def build_obj_prob_loc_txt(self, pred_obj_prob, obj_labels, obj_masks, obj_locs, build_choice='prob', obj_max_num=20, agent_pos_ori=[[0,0,0], [0,0,0]], device='cuda',coord_type='rectangle'):   # prob or loc
        '''
            pred_obj_prob: (N,)
        '''
        pred_obj_prob = pred_obj_prob[obj_masks]   # avoid indices out of range
        sorted_indices = torch.argsort(pred_obj_prob, descending=True)
        if hasattr(self, 'max_cot_obj_num'):
            obj_max_num = self.max_cot_obj_num                                                
        top_k_indices = sorted_indices[:obj_max_num]
        top_k_labels = [obj_labels[i] for i in top_k_indices]
        top_k_probs = [pred_obj_prob[i].item() for i in top_k_indices]
        
        if build_choice == 'prob':
            output_string = " ".join(f"{label} {prob:.2f}" for label, prob in zip(top_k_labels, top_k_probs))
        elif build_choice == 'loc':
            agent_pos = agent_pos_ori[0]
            agent_ori = agent_pos_ori[1]
            orientation_vec = agent_ori[:2]
            norm = torch.linalg.norm(orientation_vec)
            orientation_vec = orientation_vec / norm
            theta = math.atan2(orientation_vec[1], orientation_vec[0])  # Convert to angle
            cos_theta, sin_theta = math.cos(theta), math.sin(theta)
            rotation_matrix = torch.tensor([[cos_theta, sin_theta], [-sin_theta, cos_theta]]).to(device)
            top_k_relative_loc_str = []
            for i in top_k_indices:
                obj_loc = obj_locs[i]
                relative_loc = obj_loc[:2] - agent_pos[:2]
                relative_loc = torch.matmul(rotation_matrix, relative_loc[:2])
                relative_loc = relative_loc.tolist()
                relative_loc.append(obj_loc[2] - agent_pos[2])
                # relative_loc_str = ','.join([f"{coord:.1f}" for coord in relative_loc])
                if coord_type == 'rectangle':
                    obj_sizes = obj_loc[3:6]
                    relative_loc_str = ",".join([f"{coord:.1f}" for coord in relative_loc])
                    relative_loc_str += ","+",".join([f"{size:.1f}" for size in obj_sizes])
                elif coord_type == 'polar':
                    theta_obj = math.atan2(relative_loc[1], relative_loc[0])*180/math.pi
                    distance_obj = math.sqrt(relative_loc[0] ** 2 + relative_loc[1] ** 2)
                    relative_loc_str = f"{theta_obj:.1f}, {distance_obj:.1f}"
                top_k_relative_loc_str.append(relative_loc_str)
            output_string = " ".join(f"{label}: {loc_str}; prob: {prob:.2f}" for label, prob, loc_str in zip(top_k_labels, top_k_probs, top_k_relative_loc_str))
        else:
            ValueError("build_choice should be either 'prob' or 'loc'")
        
        return output_string

    @torch.no_grad()
    def generate_based_on_visual_cues(self, inputs_embeds, final_sequences, 
                                      attention_mask, pred_obj_prob, obj_masks,
                                      obj_labels=None, obj_locs=None,
                                      oracle_obj_prob=None, oracle_obj_loc=None,
                                      agent_pos_ori=[[0,0,0], [0,0,0]],
                                      num_beams=5, use_nucleus_sampling=False, top_p=0.9, 
                                      repetition_penalty=3.0, length_penalty=1, num_captions=1, temperature=1,device='cuda',data_dict=None):
        bs = inputs_embeds.shape[0]
        if pred_obj_prob.shape[1] == 1:
            pred_obj_prob = pred_obj_prob.squeeze(1)
        device = self.device
        model = self.llm_model
        tokenizer = self.llm_tokenizer
        output_after_visual_cues = []
        output_with_visual_cues = []
        for i in range(bs):
            sequence = final_sequences[i]
            list_obj_prob_idx = (sequence == LIST_OBJ_PROB_TOKEN_ID).nonzero(as_tuple=True)[0]
            list_obj_loc_idx = (sequence == LIST_OBJ_LOC_TOKEN_ID).nonzero(as_tuple=True)[0]
            list_obj_loc_plr_idx = (sequence == LIST_OBJ_LOC_PLR_TOKEN_ID).nonzero(as_tuple=True)[0]
            list_highlight_obj_idx = (sequence == HIGHLIGHT_OBJ_TOKEN_ID).nonzero(as_tuple=True)[0]
            print(f"LIST_OBJ_PROB_TOKEN_ID: {LIST_OBJ_PROB_TOKEN_ID}")
            print(f"sequence: {sequence}")
            if list_obj_prob_idx.numel() > 0 and list_obj_loc_idx.numel() == 0 and list_obj_loc_plr_idx.numel() == 0 and list_highlight_obj_idx.numel() == 0:
                list_obj_prob_idx = list_obj_prob_idx[0]
                if self.use_oracle_obj_content:
                    obj_prob_content = oracle_obj_prob[i]
                    print(f'obj_prob_oracle_content: {obj_prob_content}')
                else:
                    obj_prob_content = self.build_obj_prob_loc_txt(pred_obj_prob[i], obj_labels=obj_labels[i], obj_masks=obj_masks[i], obj_locs=obj_locs[i], build_choice='prob', device=device)
                obj_prob = f"\n{COT_INDICATORS_TOKENIZE['<obj_prob>']}{obj_prob_content}{COT_INDICATORS_TOKENIZE['</obj_prob>']}"
                obj_prob_tokenized = tokenizer(obj_prob, return_tensors='pt')
                obj_prob_tokens = obj_prob_tokenized.input_ids[0][2:].to(device)   # skip tokens: bos_token, 29871 (space)
                obj_prob_inputs_embeds = model.get_input_embeddings()(obj_prob_tokens)
                obj_prob_attention_mask = obj_prob_tokenized.attention_mask[0][2:].to(device)
                sequence_inputs_embeds = model.get_input_embeddings()(sequence)   
                sequence_attention_mask = torch.ones_like(sequence, dtype=torch.bool).to(device)
                new_inputs_embeds = torch.cat([inputs_embeds[i, :], sequence_inputs_embeds[:list_obj_prob_idx+1], obj_prob_inputs_embeds] , dim=0)
                new_attention_mask = torch.cat([attention_mask[i, :], sequence_attention_mask[:list_obj_prob_idx+1], obj_prob_attention_mask], dim=0)
                new_seqeunce = torch.cat([sequence[:list_obj_prob_idx+1], obj_prob_tokens], dim=0)    # do not contain the inputs embeds
                # print(f"new_seqeunce: {new_seqeunce}")
                visual_cues_content = obj_prob
            elif list_obj_loc_idx.numel() > 0 and list_obj_prob_idx.numel() == 0 and list_obj_loc_plr_idx.numel() == 0 and list_highlight_obj_idx.numel() == 0:
                list_obj_loc_idx = list_obj_loc_idx[0]
                if self.use_oracle_obj_content:
                    obj_loc_content = oracle_obj_loc[i]
                else:
                    obj_loc_content = self.build_obj_prob_loc_txt(pred_obj_prob[i], obj_labels=obj_labels[i], obj_masks=obj_masks[i], obj_locs=obj_locs[i], build_choice='loc', agent_pos_ori=agent_pos_ori[i], device=device, coord_type='rectangle')
                obj_loc = f"\n{COT_INDICATORS_TOKENIZE['<obj_loc_prob>']}{obj_loc_content}{COT_INDICATORS_TOKENIZE['</obj_loc_prob>']}" 
                obj_loc_tokenized = tokenizer(obj_loc, return_tensors='pt')
                obj_loc_tokens = obj_loc_tokenized.input_ids[0][2:].to(device)
                obj_loc_inputs_embeds = model.get_input_embeddings()(obj_loc_tokens)
                obj_loc_attention_mask = obj_loc_tokenized.attention_mask[0][2:].to(device) 
                sequence_inputs_embeds = model.get_input_embeddings()(sequence)   
                sequence_attention_mask = torch.ones_like(sequence, dtype=torch.bool).to(device)
                new_inputs_embeds = torch.cat([inputs_embeds[i, :], sequence_inputs_embeds[:list_obj_loc_idx+1], obj_loc_inputs_embeds] , dim=0)
                new_attention_mask = torch.cat([attention_mask[i, :], sequence_attention_mask[:list_obj_loc_idx+1], obj_loc_attention_mask], dim=0)
                new_seqeunce = torch.cat([sequence[:list_obj_loc_idx+1], obj_loc_tokens], dim=0)  # do not contain the inputs embeds
                visual_cues_content = obj_loc
            elif list_obj_loc_plr_idx.numel() > 0 and list_obj_prob_idx.numel() == 0 and list_obj_loc_idx.numel() == 0 and list_highlight_obj_idx.numel() == 0:
                list_obj_loc_plr_idx = list_obj_loc_plr_idx[0]
                if self.use_oracle_obj_content:
                    obj_loc_content = oracle_obj_loc[i]
                else:
                    obj_loc_content = self.build_obj_prob_loc_txt(pred_obj_prob[i], obj_labels=obj_labels[i], obj_masks=obj_masks[i], obj_locs=obj_locs[i], build_choice='loc', agent_pos_ori=agent_pos_ori[i], device=device, coord_type='polar')
                obj_loc = f"\n{COT_INDICATORS_TOKENIZE['<obj_loc_plr_prob>']}{obj_loc_content}{COT_INDICATORS_TOKENIZE['</obj_loc_plr_prob>']}"
                obj_loc_tokenized = tokenizer(obj_loc, return_tensors='pt')
                obj_loc_tokens = obj_loc_tokenized.input_ids[0][2:].to(device)
                obj_loc_inputs_embeds = model.get_input_embeddings()(obj_loc_tokens)
                obj_loc_attention_mask = obj_loc_tokenized.attention_mask[0][2:].to(device)
                sequence_inputs_embeds = model.get_input_embeddings()(sequence)
                sequence_attention_mask = torch.ones_like(sequence, dtype=torch.bool).to(device)
                new_inputs_embeds = torch.cat([inputs_embeds[i, :], sequence_inputs_embeds[:list_obj_loc_plr_idx+1], obj_loc_inputs_embeds] , dim=0)
                new_attention_mask = torch.cat([attention_mask[i, :], sequence_attention_mask[:list_obj_loc_plr_idx+1], obj_loc_attention_mask], dim=0)
                new_seqeunce = torch.cat([sequence[:list_obj_loc_plr_idx+1], obj_loc_tokens], dim=0)  # do not contain the inputs embeds
                visual_cues_content = obj_loc
            elif list_highlight_obj_idx.numel() > 0 and list_obj_prob_idx.numel() == 0 and list_obj_loc_idx.numel() == 0 and list_obj_loc_plr_idx.numel() == 0:
                list_highlight_obj_idx = list_highlight_obj_idx[0]
                # obj_id = data_dict['obj_ids'][0]  # attribute/description only has one object
                # img = data_dict['obj_imgs'][obj_id]
                sorted_indices = torch.argsort(pred_obj_prob[i][obj_masks[i]], descending=True)
                if hasattr(self, 'max_cot_obj_num'):
                    obj_max_num = self.max_cot_obj_num                                                
                top_k_indices = sorted_indices[:obj_max_num]
                img = data_dict['all_obj_imgs'][i][top_k_indices[0]]   # use the first object
                img = img.unsqueeze(0)   # (1, 3, H, W)
                img_embeds = self.mm_projector(self.vision_tower(img.to(device))) #(1, 3, H, W) -> (1, 576, D)
                img_start_embeds = model.get_input_embeddings()(torch.tensor(IMG_START_TOKEN_ID).to(device)).unsqueeze(0).unsqueeze(0)   # (1, 1, D)
                img_end_embeds = model.get_input_embeddings()(torch.tensor(IMG_END_TOKEN_ID).to(device)).unsqueeze(0).unsqueeze(0)   # (1, 1, D)
                img_embeds_with_pre_suffix = torch.cat([img_start_embeds, img_embeds, img_end_embeds], dim=1)   # (1, 578, D)
                img_masks = torch.ones(img_embeds_with_pre_suffix.shape[:2], dtype=torch.int64, device=device) # (1, 576)
                img_embeds_with_pre_suffix = img_embeds_with_pre_suffix.squeeze(0)
                img_masks = img_masks.squeeze(0)
                sequence_inputs_embeds = model.get_input_embeddings()(sequence)
                sequence_attention_mask = torch.ones_like(sequence, dtype=torch.bool).to(device)
                new_inputs_embeds = torch.cat([inputs_embeds[i, :], sequence_inputs_embeds[:list_highlight_obj_idx+1], img_embeds_with_pre_suffix] , dim=0)
                new_attention_mask = torch.cat([attention_mask[i, :], sequence_attention_mask[:list_highlight_obj_idx+1], img_masks], dim=0)
                new_seqeunce = sequence[:list_highlight_obj_idx+1]   # remove the content after highlight_obj
            else:
                new_inputs_embeds = inputs_embeds[i]
                new_attention_mask = attention_mask[i]
                visual_cues_content = ''
                new_seqeunce = sequence
            new_outputs = model.generate(
                    inputs_embeds=new_inputs_embeds.unsqueeze(0).to(self.llm_model.dtype),
                    attention_mask=new_attention_mask.unsqueeze(0),
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=128,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    )   
            new_txt_output = tokenizer.decode(new_outputs.sequences[0], skip_special_tokens=True)   # extactly the same output with batch decode
            new_output_with_visual_cues = tokenizer.decode(new_seqeunce, skip_special_tokens=True)
            print(f'output with visual cues: {new_output_with_visual_cues}')
            print(f'output after visual cues: {new_txt_output}')
            output_after_visual_cues.append(new_txt_output)
            output_with_visual_cues.append(new_output_with_visual_cues)

        return output_with_visual_cues, output_after_visual_cues

    @torch.no_grad()
    def generate(
        self,
        data_dict,
        use_nucleus_sampling=False,
        num_beams=5,
        top_p=0.9,
        repetition_penalty=3.0,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        """
        data_dict requires the same keys as forward() except output_gt
        """
        device = self.device
        bs = len(data_dict['input_txt'])

        # get embedding of input_txt
        self.llm_tokenizer.padding_side = 'right'
        self.llm_tokenizer.truncation_side = 'left'
        input_txt_tokenized = self.llm_tokenizer(
            data_dict['input_txt'],
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=False,
        ).to(device)   # [BOS, tokens, PAD]
        data_dict['input_txt_ids'] = input_txt_tokenized.input_ids
        data_dict['input_txt_embed'] = self.llm_model.get_input_embeddings()(input_txt_tokenized.input_ids)
        data_dict['input_txt_mask'] = input_txt_tokenized.attention_mask

        if 'scene_tokens' not in data_dict:
            if self.blind:
                data_dict['scene_tokens'] = torch.zeros(bs, 1, data_dict['input_txt_embed'].shape[-1]).to(device)
                data_dict['scene_masks'] = torch.zeros(bs, 1, dtype=torch.bool).to(device)
            else:
                data_dict = self.pcd_encoder(data_dict)
                data_dict['scene_tokens'] = self.pcd_proj(data_dict['scene_tokens'].to(device))

        inputs_embeds, attention_mask = self.build_input_sequence(data_dict=data_dict)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds.to(self.llm_model.dtype),
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=self.max_out_len,
            pad_token_id=self.llm_tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

        final_sequences = outputs.sequences
        # grounding
        if hasattr(self, 'grounding_head'):
            hidden_states = outputs.hidden_states  # step x layer x (bs x beam) x token_num x hidden_dim
            decoded_sequences = []
            for seq in final_sequences:
                decoded_sequences.append(self.llm_tokenizer.decode(seq, skip_special_tokens=True))

            batch_size = final_sequences.size(0)  
            hidden_size = hidden_states[0][-1].size(-1)
            predicted_tokens, predicted_sequences = predict_with_beam_search(self.llm_model, hidden_states, self.llm_tokenizer, num_beams=num_beams, batch_size=batch_size, device=device, repetition_penalty=repetition_penalty)
            for i in range(batch_size):
                if len(torch.nonzero(predicted_tokens[i] == 2)) == 0:
                    end_idx = predicted_tokens[i].shape[0]
                else:
                    end_idx = torch.nonzero(predicted_tokens[i] == 2)[0][0]
                predicted_tokens[i, end_idx+1:] = -100
            beam_indices = outputs.beam_indices if hasattr(outputs, 'beam_indices') else None
            hidden_states = outputs.hidden_states
            last_hidden_states = []
            for i in range(min(len(hidden_states), beam_indices.shape[1])):
                last_hidden_states.append(hidden_states[i][-1][beam_indices[:, i], -1, :])
            last_hidden_states = torch.stack(last_hidden_states, dim=1)
            predicted_tokens = predicted_tokens.to(device)
            predicted_tokens = predicted_tokens[:, :beam_indices.shape[1]] 
            
            if self.grd_use_scene_tokens:
                # pq3d
                obj_tokens = data_dict['scene_tokens']
            else:
                data_dict = self.obj_encoder(data_dict)
                obj_tokens = self.pcd_proj(data_dict['obj_tokens'])
            obj_masks = data_dict['obj_masks']
            self.grounding_head = self.grounding_head.to(self.device)
            last_hidden_states = last_hidden_states.float() 
            predicted_obj_prob = self.grounding_head(input_ids=predicted_tokens, hidden_states=last_hidden_states, obj_tokens=obj_tokens, obj_mask_token_idx=self.grd_token_id, obj_masks=obj_masks, gt_grounding_obj_mask=data_dict['grounding_obj_mask_gt'], llm_model=self.llm_model,device=self.device)
            data_dict['pred_obj_prob'] = predicted_obj_prob
            # print(f'predicted_obj_prob: {predicted_obj_prob}')
            if hasattr(self, 'moe_flag') and self.moe_flag:
                question_type = data_dict['type']
                if question_type in self.moe_type_list:
                    question_index = str(data_dict['index'])
                    predicted_obj_prob = self.obj_prob_dict[question_index]
                    print(f"$$$$$$$$$$$$$$$$$$$$$$$$")
                    print("Successfully load the predicted_obj_prob from the dictionary")

        final_sequences[final_sequences == self.llm_tokenizer.unk_token_id] = self.llm_tokenizer.eos_token_id
        # data_dict['output_tokens'] = outputs   # unable to gather variable-length tensors
        if hasattr(self, 'grounding_head') and hasattr(self, 'cot_mask_obj_prob_loc_token') and self.cot_mask_obj_prob_loc_token:
            output_with_visual_cues, output_after_visual_cues = self.generate_based_on_visual_cues(inputs_embeds, 
            final_sequences, attention_mask, predicted_obj_prob, data_dict['obj_masks'],
            obj_labels=data_dict['gt_labels'], obj_locs=data_dict['obj_locs'], 
            oracle_obj_prob=data_dict['oracle_obj_prob'], oracle_obj_loc=data_dict['oracle_obj_loc'],
            agent_pos_ori=data_dict['agent_pos_ori'],
            num_beams=num_beams, use_nucleus_sampling=use_nucleus_sampling, 
            top_p=top_p, repetition_penalty=repetition_penalty, 
            length_penalty=length_penalty, num_captions=num_captions, 
            temperature=temperature,data_dict=data_dict)
            output_txt_list = []
            for i in range(bs):
                output_txt = replace_cot_tokens_with_indicators(output_with_visual_cues[i]+output_after_visual_cues[i])   # replace the cot tokens with indicators
                output_txt_list.append(output_txt.strip())
            data_dict['output_txt'] = output_txt_list
        elif hasattr(self, 'grounding_head') and hasattr(self, 'cot_mask_obj_prob_loc_token') and not self.cot_mask_obj_prob_loc_token:
            output_txt_raw = self.llm_tokenizer.batch_decode(final_sequences, skip_special_tokens=True)
            output_txt = []
            for i in range(bs):
                output_txt.append(replace_cot_tokens_with_indicators(output_txt_raw[i]))
            data_dict['output_txt'] = output_txt
        else:
            output_txt = self.llm_tokenizer.batch_decode(final_sequences, skip_special_tokens=True)
            output_txt = [txt.strip() for txt in output_txt]
            data_dict['output_txt'] = output_txt
            # if hasattr(self, 'cot_answer_training') and self.cot_answer_training:
            #     for i in range(bs):
            #         output_txt = replace_cot_tokens_with_indicators(output_txt[i], COT_INDICATORS_TOKENIZE)

        return data_dict
