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


@FRAME_DECODER_REGISTRY.register()
class ASFormerEncoder(nn.Module):
    '''just use asformer encoder as the frame-wise module'''
    @configurable()
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super( ).__init__()
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList(
                                    [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in  # 2**i
                                     range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

        self.mask_features = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=3, stride=1, padding=1)


    @classmethod
    def from_config(cls, cfg):
        if cfg.dataset.name == 'gtea':
            channel_masking_rate = 0.5
        else:
            channel_masking_rate = 0.3

        return {"num_layers": 10,
                "r1": 2,
                "r2": 2,
                "num_f_maps": cfg.model.action_seg.frame_decoder.embed_dim,
                "input_dim": cfg.model.action_seg.frame_decoder.input_dim,
                "num_classes": cfg.dataset.n_classes,
                "channel_masking_rate": channel_masking_rate,
                "att_type": 'sliding_att',
                "alpha": 1}

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''


        # ## before feed to the query emebd, maskformer have the following process
        # transformer = self.input_proj(x)
        # pos = self.pe_layer(x)  # where add the position embedding
        # transformer = self.transformer(transformer, None, pos)

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        outputs = {}
        outputs["transformer_encoder_features"] = feature

        multi_features = [ ]

        for layer in self.layers:
            feature = layer(feature, None, mask)
            multi_features.append(feature)

        out = self.conv_out(feature) * mask[:, 0:1, :] # for num_class

        outputs["class_logits"] = out
        outputs["feature"] = feature
        outputs["mask_features"] = self.mask_features(feature) # for transformer decoder dim
        outputs["multi_features"] = multi_features
        return outputs