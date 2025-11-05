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
class ASFormerFrameEnDecoder(nn.Module):
    '''just use asformer encoder as the frame-wise module'''
    @configurable()

    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate,
                               att_type='sliding_att', alpha=1)
        self.decoders = nn.ModuleList([copy.deepcopy(
            Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='sliding_att',
                    alpha=exponential_descrease(s))) for s in range(num_decoders)])  # num_decoders

    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)

        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs



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

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        outputs = {}
        outputs["transformer_encoder_features"] = feature

        multi_scale_features = [ ]

        for layer in self.layers:
            feature = layer(feature, None, mask)
            multi_scale_features.append(feature)

        out = self.conv_out(feature) * mask[:, 0:1, :] # for num_class

        outputs["class_logits"] = out
        outputs["feature"] = feature
        outputs["mask_features"] = self.mask_features(feature) # for transformer decoder dim
        outputs["multi_scale_features"] = multi_scale_features
        return outputs