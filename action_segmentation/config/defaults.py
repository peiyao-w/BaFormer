from .config_node import ConfigNode

import torch
import numpy as np


config = ConfigNode()

config.device = 'cuda' #cuda
# config.device_id = '0'

# cuDNN
config.cudnn = ConfigNode()
config.cudnn.benchmark = False #True
config.cudnn.deterministic = True #False


config.dataset = ConfigNode()
config.dataset.all_to_memory = False
config.dataset.split = 1
config.dataset.name = '50salads' # [gtea breakfast  50salads]
config.dataset.dataset_dir = '/data/peiyao/data/' 
config.dataset.classes_list = [11, 48, 19]
config.dataset.data_list = ['gtea', 'breakfast', '50salads']
config.dataset.n_channels = 2048

config.model = ConfigNode()
config.model.type = 'act_seg'
config.model.name = 'bk_fde_tde'
config.model.pose_weight_single = 0.7
config.model.ce_weight = 1.0 
config.model.mask_weight = 5.0 
config.model.dice_weight = 1.0 
config.model.bd_weight = 1.0
config.model.eos_coef = 0.01

config.model.bbox_weight = 1 
config.model.giou_weight = 0

if config.dataset.name == 'gtea':
    config.dataset.n_classes = 11
    config.dataset.sample_rate = 1
    config.dataset.roll = 4
    config.dataset.noise_weight = None
    config.dataset.num_query = 60 
    config.dataset.guassian_sigma = 0.5 
    config.dataset.pos_weight = 20
    config.dataset.threshold = 0.2
    config.model.note = 'final'  #

elif config.dataset.name == '50salads':
    config.dataset.n_classes = 19
    config.dataset.sample_rate = 2
    config.dataset.roll = None 
    config.dataset.noise_weight = None
    config.dataset.num_query = 100
    config.dataset.guassian_sigma = None 
    config.dataset.pos_weight = 300 
    config.dataset.threshold = 0.3
    config.model.note = 'final'  


elif config.dataset.name == 'breakfast':
    config.dataset.n_classes = 48
    config.dataset.sample_rate = 1
    config.dataset.roll = None
    config.dataset.noise_weight = None
    config.dataset.num_query  = 100 
    config.dataset.guassian_sigma = None 
    config.dataset.pos_weight = 80
    config.dataset.threshold = 0.3
    config.model.note = 'final'  #



#####--------whole model config----------

config.model.name == 'bk_fde_tde'  
config.model.action_seg = ConfigNode()
config.model.action_seg.num_stage = 3
config.model.action_seg.backbone = ConfigNode()
config.model.action_seg.backbone.name = None
config.model.action_seg.backbone.input_dim = 2048
config.model.action_seg.backbone.embed_dim = 64
config.model.action_seg.frame_decoder = ConfigNode()
config.model.action_seg.frame_decoder.name = 'ASFormerEncoder'
config.model.action_seg.frame_decoder.input_dim = 64
config.model.action_seg.frame_decoder.embed_dim = 64
config.model.action_seg.transformer_decoder = ConfigNode()
config.model.action_seg.transformer_decoder.name = 'MaskedTransformerDecoder'
config.model.action_seg.transformer_decoder.hidden_dim = 64 * 3
config.model.action_seg.transformer_decoder.num_queries = 100
config.model.action_seg.transformer_decoder.nheads = 3
config.model.action_seg.transformer_decoder.dim_feedforward = 128
config.model.action_seg.transformer_decoder.dropout = 0.0 
config.model.action_seg.transformer_decoder.dec_layers = 11 
config.model.action_seg.transformer_decoder.pre_norm = False
config.model.action_seg.transformer_decoder.enforce_input_project = False
config.model.action_seg.transformer_decoder.mask_dim = 64
config.model.action_seg.transformer_decoder.deep_supervision = False
config.model.action_seg.transformer_decoder.num_patch = 6
config.model.action_seg.transformer_decoder.threshold = 0.3  #if <, True, not atttend to
config.model.action_seg.transformer_decoder.layer_in_decode_block = 3

config.train = ConfigNode()
config.train.checkpoint = ''
config.train.resume = False
config.train.batch_size = 1
config.train.optimizer = 'adam' # adam
if config.dataset.name == 'gtea':
    config.train.base_lr = 0.0005 
elif config.dataset.name == 'breakfast':
    config.train.base_lr = 0.0001 
elif config.dataset.name == '50salads':
    config.train.base_lr = 0.0005 

config.train.momentum = 0.9
config.train.nesterov = True
if config.dataset.name == 'gtea':
    config.train.weight_decay = 0.0 
elif config.dataset.name == 'breakfast':
    config.train.weight_decay = 0.00 
elif config.dataset.name == '50salads':
    config.train.weight_decay = 0.000 

config.train.clip_grad = 5.0
config.train.no_weight_decay_on_bn = False
config.train.gradient_clip = 0 
config.train.start_epoch = 0
config.train.seed = 42 
config.train.val_first = True
config.train.val_period = 1
config.train.val_ratio = 0.0
config.train.use_test_as_val = True

config.train.output_dir = 'experiments'
config.train.log_period = 10000
config.train.checkpoint_period = 1
config.train.use_tensorboard = True

# optimizer
config.optim = ConfigNode()
# Adam
config.optim.adam = ConfigNode()
config.optim.adam.betas = (0.9, 0.999)
config.optim.adam.eps = 1e-8
#Adamw
config.optim.adamw = ConfigNode()
config.optim.adamw.betas = (0.9, 0.999)
config.optim.adamw.eps = 1e-8


# scheduler
config.scheduler = ConfigNode()
config.scheduler.epochs = 400 

config.scheduler.type = 'multistep'
if config.dataset.name == 'gtea':
    config.scheduler.milestones = [200, 400]
elif config.dataset.name == '50salads':
    config.scheduler.milestones = [200, 350]
elif config.dataset.name == 'breakfast':
    config.scheduler.milestones = [100, 200]
config.scheduler.lr_decay = 0.5 #0.5
config.scheduler.lr_min_factor = 0.001
config.scheduler.T0 = 10
config.scheduler.T_mul = 1.


# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.pin_memory = False
config.train.dataloader.non_blocking = False

# validation data loader
config.validation = ConfigNode()
config.validation.is_use = True
config.validation.dataloader = ConfigNode()
config.validation.dataloader.pin_memory = False
config.validation.dataloader.non_blocking = False


# distributed
config.train.distributed = True
config.train.dist = ConfigNode()
config.train.dist.backend = 'nccl' # gloo nccl
config.train.dist.init_method = 'env://'
config.train.dist.world_size = -1
config.train.dist.node_rank = -1
config.train.dist.local_rank = 0
config.train.dist.use_sync_bn = False

# augmentation
config.augmentation = ConfigNode()
config.augmentation.use_label_smoothing = True
if config.augmentation.use_label_smoothing:
    config.augmentation.label_smoothing_epsilon = 0.1
else:
    config.augmentation.label_smoothing_epsilon = 0.

config.augmentation.frame_ls = 0.

config.augmentation.label_smoothing = ConfigNode()
config.augmentation.label_smoothing.set_epsilon = 0.4

config.augmentation.is_use = False
config.augmentation.use_dual_cutout = False
config.augmentation.use_mixup = False
config.augmentation.use_ricap = False
config.augmentation.use_cutmix = False

# test config
config.test = ConfigNode()
config.test.checkpoint = ''
config.test.output_dir = ''
config.test.batch_size = 1
config.test.train_data = False
# test data loader
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2 
config.test.dataloader.pin_memory = False


def get_default_config():
    return config.clone()


