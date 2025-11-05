# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
# from detectron2.layers import Conv2d
from torch.nn import Conv1d

# from .position_encoding import PositionEmbeddingSine
from .position_encoding import PositionEmbeddingSine, PositionalEncoding
from .transformer import Transformer


class TransformerPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        deep_supervision: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dropout: dropout in Transformer
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            deep_supervision: whether to add supervision to every decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        self.mask_classification = mask_classification

        # positional encoding
        self.pe_layer = PositionalEncoding(hidden_dim, n_position=20000)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)#qc, hidden_dim is the transformer dim, need/nhead

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv1d(in_channels, hidden_dim, kernel_size=1)
            # weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = 64 # input of the whole transformer_predictor
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.dataset.n_classes
        ret["hidden_dim"] = 64 * 3 #the transformer hidden dim, need /nhead
        ret["num_queries"] = cfg.dataset.num_query
        # Transformer parameters:
        ret["nheads"] = 3
        ret["dropout"] = 0.1
        ret["dim_feedforward"] = 64
        ret["enc_layers"] = 6
        ret["dec_layers"] = 6
        ret["pre_norm"] = False
        ret["deep_supervision"] = True
        ret["enforce_input_project"] = False

        ret["mask_dim"] = 64

        return ret

    def forward(self, x, mask_features): # x_embedding(image_feature), mask_embedding
        x = x[0] # bcl
        pos = self.pe_layer(x.transpose(-1, -2)).transpose(-1,-2) #blc

        src = x
        mask = None
                                     #(src, mask, query_embed, pos_embed)
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)
        # memory: transformer_encoder (self-attn, the first layer of transformer decoder)
        # hs: from each transformer_decoder layer (multi-cross-attn, the rest layer of transforemr decoder)

        if self.mask_classification:
            outputs_class = self.class_embed(hs) # [6,1,100,20]each decoder layer have a hs
            out = {"pred_logits": outputs_class[-1]} # query log_pred
        else:
            out = {}

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(hs) #[6,1,100,64]
            outputs_seg_masks = torch.einsum("lbqc,bcn->lbqn", mask_embed, mask_features) #why l, what's the meaning of l
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class if self.mask_classification else None, outputs_seg_masks  # just get all the element as dict
            )
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(hs[-1]) # MLP, get N mask embeddings
            outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


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
