import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import copy
import numpy as np
import math
from detectron2.config import configurable
from detectron2.utils.registry import Registry

from ..backbone.asformer_model import AttModule

from . import FRAME_DECODER_REGISTRY

from ..backbone.mstcn_model import DilatedResidualLayer

@FRAME_DECODER_REGISTRY.register()
class SSTCNEncoder(nn.Module):
    '''just use asformer encoder as the frame-wise module'''
    @configurable()
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super( ).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

        self.mask_features = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=3, stride=1, padding=1)

    @classmethod
    def from_config(cls, cfg):
        if cfg.dataset.name == 'gtea':
            channel_masking_rate = 0.5
        else:
            channel_masking_rate = 0.3

        return {"num_layers": 10,
                "num_f_maps": cfg.model.action_seg.frame_decoder.embed_dim,
                "dim": cfg.model.action_seg.frame_decoder.input_dim,
                "num_classes": cfg.dataset.n_classes}

    def forward(self, x, mask):
        feature = self.conv_1x1(x)
        outputs = {}
        outputs["transformer_encoder_features"] = feature

        multi_features = [ ]

        for layer in self.layers:
            feature = layer(feature, mask)
            multi_features.append(feature)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        outputs["class_logits"] = out
        outputs["feature"] = feature
        outputs["mask_features"] = self.mask_features(feature) # for transformer decoder dim
        outputs["multi_features"] = multi_features
        return outputs
