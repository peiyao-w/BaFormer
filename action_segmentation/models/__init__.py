import importlib

import torch
import torch.nn as nn
import torch.distributed as dist
import yacs.config
from .criterion import SetCriterion,  sigmoid_focal_loss
from .criterion_box import SetCriterion_box
from .criterion_bd import SetCriterion_bd
from .criterion_maskw import SetCriterion_maskw
from .criterion_cat_bd import SetCriterion_cat_bd
from .criterion_trans_bd import SetCriterion_trans_bd

def create_model(config: yacs.config.CfgNode) -> nn.Module:
    module = importlib.import_module(
        'action_segmentation.models'
        f'.{config.model.name}')
    model = getattr(module, 'Network')(config) #get model class
    device = torch.device(config.device)
    model.to(device)
    return model


def apply_data_parallel_wrapper(config: yacs.config.CfgNode,
                                model: nn.Module) -> nn.Module:
    local_rank = config.train.dist.local_rank
    if dist.is_available() and dist.is_initialized():
        if config.train.dist.use_sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank,
                                                    find_unused_parameters=True)
    else:
        model.to(config.device)
    return model