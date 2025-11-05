import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import copy
import numpy as np
import math
from detectron2.config import configurable

from .frame_decoder import build_frame_decoder
from .transformer_decoder import build_transformer_decoder
from .backbone import build_backbone

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class Network(nn.Module):
    @configurable
    def __init__(self, backbone, frame_decoder, transformer_predictor):
        super( ).__init__()
        self.backbone = backbone
        self.frame_decoder = frame_decoder
        self.predictor = transformer_predictor

    @classmethod
    def from_config(cls, cfg):
        return {"backbone": None,
                "frame_decoder": build_frame_decoder(cfg),
                "transformer_predictor": build_transformer_decoder(cfg)}

    def forward(self, x):
        mask = torch.ones_like(x).to(x.device)
        
        frame_out = self.frame_decoder(x, mask)

        # multi_scale_frame_decoder
        outputs = self.predictor(frame_out['multi_features'], frame_out['mask_features'], mask=None)

        return outputs