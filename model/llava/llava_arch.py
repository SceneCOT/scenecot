# Adapted from LLaVA

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .vision_tower import build_vision_tower
from .vision_projector import build_vision_projector


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        self.config = config
        self.mm_projector = build_vision_projector(config)

    def init_vision_tower(self):
        if hasattr(self.config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(self.config, delay_load=True)
            if 'unpad' in getattr(self.config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(self.config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def init_vision_tower(self):
        return self.get_model().init_vision_tower()

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
