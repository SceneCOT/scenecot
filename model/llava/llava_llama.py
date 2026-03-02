# Adapted from LLaVA

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         CLIPImageProcessor, CLIPVisionModel, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from .llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)


def load_pretrained_llava(model_path, vision_tower_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    if vision_tower_path is not None:
        model.config.mm_vision_tower = vision_tower_path
        model.init_vision_tower()
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_path)
            vision_tower.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_path)
            vision_tower.vision_tower.requires_grad_(False)
            vision_tower.is_loaded = True
        image_processor = vision_tower.image_processor
    else:
        image_processor = None

    return tokenizer, model, image_processor
