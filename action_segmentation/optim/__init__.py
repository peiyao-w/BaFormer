import torch

from .adabound import AdaBound, AdaBoundW
from .lars import LARSOptimizer


def get_param_list(config, model):
    if config.train.no_weight_decay_on_bn:
        param_list = []
        for name, params in model.named_parameters():
            if 'conv.weight' in name: # construct param_list according to the param name
                param_list.append({
                    'params': params,
                    'weight_decay': config.train.weight_decay,
                })
            else:
                param_list.append({
                    'params': params,
                    'weight_decay': 0,
                })
    else:
        ##-------test gthe results of getting part of the parameters
        # param_list = []
        # names_list = []
        # for name, params in model.named_parameters():
        #     if 'patch_merg' in name:
        #         param_list.append({
        #             'params': params,
        #             'weight_decay': config.train.weight_decay,
        #         })
        #         names_list.append(name)
        #     if 'individual' in name:
        #         param_list.append({
        #             'params': params,
        #             'weight_decay': config.train.weight_decay,
        #         })
        #         names_list.append(name)
        # a = 1
        param_list = [{
            'params': list(model.parameters()),
            'weight_decay': config.train.weight_decay,
        }]
    return param_list

def get_part_param_list(config, model):
    param_list = []
    for name, params in model.named_parameters():
        if 'patch_merg' in name:
            param_list.append({
                'params': params,
                'weight_decay': config.train.weight_decay,
            })
        if 'individual' in name:
            param_list.append({
                'params': params,
                'weight_decay': config.train.weight_decay,
            })
    return param_list

def get_set_param_list(config, model):
    param_list = []
    for name, params in model.named_parameters():
        if 'cross_to_transcript' in name:
            # print('-------set_pram-------', name)
            param_list.append({
                'params': params,
                'weight_decay': config.train.weight_decay,
            })
    return param_list

def get_regression_param_list(config, model):
    param_list = []
    for name, params in model.named_parameters():
        if 'temporal_reason' in name: # can add: if param.required_grad ==True:
            param_list.append({
                'params': params,
                'weight_decay': config.train.weight_decay,
            })
    return param_list


def create_optimizer(config, model):
    params = get_param_list(config, model)

    if config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=config.train.base_lr,
                                    momentum=config.train.momentum,
                                    nesterov=config.train.nesterov)
    elif config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=config.train.base_lr,
                                     weight_decay=config.train.weight_decay, #
                                     betas=config.optim.adam.betas,
                                     eps=config.optim.adamw.eps) #
    elif config.train.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params,
                                      lr=config.train.base_lr,
                                      weight_decay=config.train.weight_decay,
                                      betas=config.optim.adamw.betas,
                                      eps=config.optim.adamw.eps,)
    elif config.train.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(params,
                                     lr=config.train.base_lr,
                                     betas=config.optim.adam.betas,
                                     amsgrad=True)
    elif config.train.optimizer == 'adabound':
        optimizer = AdaBound(params,
                             lr=config.train.base_lr,
                             betas=config.optim.adabound.betas,
                             final_lr=config.optim.adabound.final_lr,
                             gamma=config.optim.adabound.gamma)
    elif config.train.optimizer == 'adaboundw':
        optimizer = AdaBoundW(params,
                              lr=config.train.base_lr,
                              betas=config.optim.adabound.betas,
                              final_lr=config.optim.adabound.final_lr,
                              gamma=config.optim.adabound.gamma)
    elif config.train.optimizer == 'lars':
        optimizer = LARSOptimizer(params,
                                  lr=config.train.base_lr,
                                  momentum=config.train.momentum,
                                  eps=config.optim.lars.eps,
                                  thresh=config.optim.lars.threshold)
    else:
        raise ValueError()
    return optimizer

def create_part_optimizer(config, model):
    params = get_part_param_list(config, model)
    optimizer = torch.optim.Adam(params,
                                 lr=config.train.base_lr,
                                 betas=config.optim.adam.betas)
    return optimizer

def create_set_optimizer(config, model):
    params = get_set_param_list(config, model)
    optimizer = torch.optim.Adam(params,
                                 lr=config.train.base_lr_set,
                                 betas=config.optim.adam.betas)
    return optimizer

def create_regression_optimizer(config, model):
    params = get_regression_param_list(config, model)
    optimizer = torch.optim.Adam(params,
                                 lr=config.train.base_lr,
                                 betas=config.optim.adam.betas)
    return optimizer