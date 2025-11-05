import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import copy
import numpy as np
import math

from .backbone.asformer_model import Encoder, Decoder

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class Network(nn.Module):
    def __init__(self, config):
        super( ).__init__()
        if config.dataset.name == "gtea":
            channel_mask_rate = 0.5
        else:
            channel_mask_rate = 0.3

        num_layers = 10
        r1 = 2
        r2 = 2
        num_f_maps = 64
        input_dim = 2048
        num_classes = config.dataset.n_classes
        num_decoders = 3
        self.encoder = Encoder(num_layers= num_layers,
                               r1=r1,
                               r2=r2,
                               num_f_maps=num_f_maps,
                               input_dim=input_dim,
                               num_classes= num_classes,
                               channel_masking_rate = channel_mask_rate,
                               att_type='sliding_att', alpha=1)
        self.decoders = nn.ModuleList([copy.deepcopy(
            Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes,
                    att_type='sliding_att',
                    alpha=exponential_descrease(s))) for s in range(num_decoders)])  # num_decoders

    def forward(self, x):
        mask = torch.ones_like(x).to(x.device)
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)

        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs