from typing import Callable, Tuple

import torch
import torch.nn as nn
import yacs.config

from .cutmix import CutMixLoss
from .mixup import MixupLoss
from .ricap import RICAPLoss
from .dual_cutout import DualCutoutLoss
from .label_smoothing import LabelSmoothingLoss

def softmax_focal_loss(x, target, gamma=2., alpha=0.25):
    n = x.shape[0]
    device = target.device
    range_n = torch.arange(0, n, dtype=torch.int64, device=device)

    pos_num = float(x.shape[1])
    p = torch.softmax(x, dim=1)
    p = p[range_n, target]
    loss = -(1-p)**gamma*alpha*torch.log(p)
    return torch.sum(loss) / pos_num

def create_loss(config: yacs.config.CfgNode) -> Tuple[Callable, Callable]:
    # if config.augmentation.use_mixup:
    #     train_loss = MixupLoss(reduction='mean')
    # elif config.augmentation.use_ricap:
    #     train_loss = RICAPLoss(reduction='mean')
    # elif config.augmentation.use_cutmix:
    #     train_loss = CutMixLoss(reduction='mean')
    # elif config.augmentation.use_label_smoothing:
    #     train_loss = LabelSmoothingLoss(config, reduction='mean')
    # elif config.augmentation.use_dual_cutout:
    #     train_loss = DualCutoutLoss(config, reduction='mean')
    # else:
    #     train_loss = nn.CrossEntropyLoss(reduction='mean')
    # if config.model.mult_stage :

    if config.augmentation.use_label_smoothing:
        train_loss = [LabelSmoothingLoss(config, reduction='mean')]
    else:
        train_loss = [nn.CrossEntropyLoss(reduction='mean')]
    train_loss.append(nn.MSELoss(reduction='none'))
    val_loss = [nn.CrossEntropyLoss(reduction='mean')]
    val_loss.append(nn.MSELoss(reduction='none'))

    return train_loss, val_loss
