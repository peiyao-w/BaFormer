# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from torch.nn import Conv1d
from detectron2.utils.registry import Registry
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .position_encoding import  PositionalEncoding
from .transformer import Transformer

from . import TRANSFORMER_DECODER_REGISTRY


@TRANSFORMER_DECODER_REGISTRY.register()
class TransformerDecoderMask_Boundary_MulKV(nn.Module):
    '''soft attn mask from sigmoid'''
    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_decode,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dropout: float,
            deep_supervision: bool,
            mask_dim: int,
            enforce_input_project: bool,
            layer_in_decode_block:int,
            threshold: float
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        self.num_decode = num_decode
        self.num_heads = nheads
        self.layer_in_decode_block = layer_in_decode_block
        self.threshold = threshold

        self.pe = PositionalEncoding(d_hid=in_channels, n_position=num_queries)
        self.query = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_decode):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv1d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1))
            else:
                self.input_proj.append(nn.Sequential())

        self.trans_decode = nn.ModuleList([])
        for i in range(self.num_decode):                         #in_dim, head_dim, heads, hidden_dim
            self.trans_decode.append(Transformer_decoder_layer(in_channels, in_channels, nheads, hidden_dim, dropout))

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.aux_loss = deep_supervision
        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3) 


        ## mlpmixer
        self.vid_embed = nn.Linear(num_queries, 1)
        self.vid_embed_before = MLP(hidden_dim, hidden_dim, hidden_dim, 3) 


    @classmethod
    def from_config(cls, cfg,):
        ret = {}
        if cfg.model.action_seg.backbone.name is not None:
            ret["in_channels"] = cfg.model.action_seg.backbone.embed_dim
        else:
            ret["in_channels"] = cfg.model.action_seg.frame_decoder.embed_dim
        ret["mask_classification"] = True

        ret["num_classes"] = cfg.dataset.n_classes
        ret["num_decode"] = cfg.model.action_seg.transformer_decoder.dec_layers
        ret["hidden_dim"] = cfg.model.action_seg.transformer_decoder.hidden_dim
        ret["num_queries"] = cfg.dataset.num_query # cfg.model.action_seg.transformer_decoder.num_queries
        # Transformer parameters:
        ret["nheads"] = cfg.model.action_seg.transformer_decoder.nheads
        ret["dropout"] = cfg.model.action_seg.transformer_decoder.dropout
        ret["enforce_input_project"] = True # cfg.model.action_seg.transformer_decoder.enforce_input_project 
        ret["mask_dim"] = cfg.model.action_seg.transformer_decoder.mask_dim
        ret["deep_supervision"] = cfg.model.action_seg.transformer_decoder.deep_supervision
        ret["threshold"] = cfg.model.action_seg.transformer_decoder.threshold
        ret["layer_in_decode_block"] = cfg.model.action_seg.transformer_decoder.layer_in_decode_block
        return ret

    def forward(self, x, mask_features, mask=None):
        assert self.num_decode == len(x)
        src = []
        for i in range(self.num_decode):
            src.append(self.input_proj[i](x[i]).transpose(-1,-2))

        output = self.pe(self.query) + self.query #blc

        predictions_class = []
        predictions_mask = []
        predictions_boundary = []

        outputs_class, outputs_mask, attn_mask, outputs_boundary = self.forward_prediction_heads(output, mask_features)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_boundary.append(outputs_boundary)

        for i in range(self.num_decode): #sepearate the decode to decode_level splits
            output = self.trans_decode[i](output, src[i], attn_mask)
            outputs_class, outputs_mask, attn_mask, outputs_boundary = self.forward_prediction_heads(output, mask_features)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_boundary.append(outputs_boundary)


        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_boundarys': predictions_boundary[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_boundary
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output) #b,l,c
        outputs_class = self.class_embed(decoder_output) #just use one linear for different layer
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bcl->bql", mask_embed, mask_features)

        ##boundary
        # ## mixer---------------
        output = self.vid_embed_before(output)

        vid_embed = self.vid_embed(output.transpose(-1, -2)).transpose(-1, -2)

        outputs_boundary = torch.einsum("bqc,bcl->bql", vid_embed, mask_features)

        attn_mask = outputs_mask #[b q l]

        attn_mask = attn_mask.sigmoid().unsqueeze(1).repeat(1, self.num_heads, 1, 1)


        return outputs_class, outputs_mask, attn_mask, outputs_boundary

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_boundary):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boundarys": c, }
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_boundary[:-1], )
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

class Sample_pe(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_position, d_hid))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        return self.pos_embed[:, :x.size(1)].clone().detach().to(x.device)

class Transformer_layer(nn.Module):
    def __init__(self, in_dim, head_dim, heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadAttention(dim=in_dim, head_dim=head_dim, heads=heads) # same as pre_norm+ attention
        self.ffn = Feedforward(dim=in_dim, hidden_dim=hidden_dim)# same as pre_norm+ ffn
    def forward(self, x_q, x_kv):
        out_attn = self.attn(x_q, x_kv)
        out = self.ffn(out_attn)
        return out

class Transformer_decoder_layer(nn.Module):
    def __init__(self, in_dim, head_dim, heads, hidden_dim, dropout = 0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(dim=in_dim, head_dim=head_dim, heads=heads, dropout = dropout)
        self.self_attn = MultiHeadAttention(dim=in_dim, head_dim=head_dim, heads=heads, dropout = dropout) # same as pre_norm+ attention
        self.ffn = Feedforward(dim=in_dim, hidden_dim=hidden_dim)# same as pre_norm+ ffn

    def forward(self, x_q, x_kv, mask):
        out_attn = self.cross_attn(x_q, x_kv, mask) 
        out_attn = self.self_attn(out_attn, out_attn) 
        out = self.ffn(out_attn)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head_dim, heads, dropout=0.1):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5

        self.pre_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, inner_dim)
        self.key_value = nn.Linear(dim, inner_dim * 2)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(inner_dim, dim)

    def forward(self, x_q, x_kv, mask=None): # b l c
        residual = x_q
        # residual
        x_q = self.pre_norm(x_q)

        # MHA
        query = self.query(x_q)
        key_value = self.key_value(x_kv)
        kvq = list(torch.chunk(key_value, 2, dim=-1))
        kvq.append(query)
        k, v, q = map(lambda t: rearrange(t, 'b l (h c) -> b h l c', h=self.heads), kvq)#separate to: b h n d

        # attention
        energy = torch.einsum('bhqc, bhkc -> bhqk', q, k)
        if mask is None:
            attn = F.softmax(energy * self.scale, dim=-1)
        else:
            attn = F.softmax(energy * mask * self.scale, dim=-1)
        attn = self.att_drop(attn)  # b n head q_win k_win
        out_att = torch.einsum('bhqk, bhkc -> bhqc', attn, v)
        out_att = rearrange(out_att, 'b h q c -> b q (h c)')
        out = self.projection(out_att)

        # residual
        out += residual # prenorm
        return out


class Feedforward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim))

    def forward(self, x):
        residual = x
        #pre_norm
        x = self.pre_norm(x)
        #ffn
        out = self.mlp(x)
        # residual
        out += residual # prenorm
        return out

class Pre_Post_Rearrange(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.rearrange1 = Rearrange('b l c -> b c l')# good for nn.averagepool
        self.rearrange2 = Rearrange('b c l -> b l c')

    def forward(self, x): # b,l,c -> b,c,l -> b, l', c
        out = self.rearrange1(x)
        out = self.fn(out)
        out = self.rearrange2(out)
        return out

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Mlp2(nn.Module):
    """from groupvit"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerMlp(Mlp2):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)