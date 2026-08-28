import numpy as np
import os
import random
import torch
import yaml
from logger import LogLevel
from yacs.config import CfgNode as CN


_C = CN()

# Base config files
_C.BASE = ['']


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = None
_C.MODEL.IN_CHS = None
_C.MODEL.N_CLASSES = None
_C.MODEL.CHS = [64, 128, 320, 512]
_C.MODEL.N_HEADS = [2, 4, 10, 16]
_C.MODEL.QKV_BIAS = True
_C.MODEL.HIDDEN_RATIO = 4.0
_C.MODEL.NORM_EPS = 1e-5
_C.MODEL.US_KS = 3
_C.MODEL.DROP_PROB = 0.0
_C.MODEL.DEEPSUPERVISION = False

""" LA """
_C.MODEL.LA = CN()
_C.MODEL.LA.USE_ROPE = False
_C.MODEL.LA.ROPE_BASE = 10000.0
_C.MODEL.LA.IMG_SIZE = [224, 224]

""" Diff Attn """
_C.MODEL.DIFF_ATTN = CN()
_C.MODEL.DIFF_ATTN.USE_ROPE = False
_C.MODEL.DIFF_ATTN.ROPE_BASE = 10000.0
_C.MODEL.DIFF_ATTN.IMG_SIZE = [224, 224]

""" Self Attn """
_C.MODEL.SELF_ATTN = CN()
_C.MODEL.SELF_ATTN.USE_ROPE = False
_C.MODEL.SELF_ATTN.ROPE_BASE = 10000.0
_C.MODEL.SELF_ATTN.IMG_SIZE = [224, 224]


# -----------------------------------------------------------------------------
# Checkpoint
# -----------------------------------------------------------------------------
_C.CKPT = CN()

""" Model """
_C.CKPT.DIR = "saved_models"
_C.CKPT.BEST = _C.CKPT.DIR + "/best.ckpt"
_C.CKPT.LAST = _C.CKPT.DIR + "/last.ckpt"


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
_C.OPTIM = CN()
_C.OPTIM.NAME = None
_C.OPTIM.BASE_LR = 5e-4
_C.OPTIM.WARMUP_LR = 1e-7
_C.OPTIM.MIN_LR = 5e-6

""" SGD """
_C.OPTIM.SGD = CN()
_C.OPTIM.SGD.MOMENTUM = 0.90
_C.OPTIM.SGD.WEIGHT_DECAY = 1e-4
_C.OPTIM.SGD.NESTEROV = True

""" AdamW """
_C.OPTIM.ADAMW = CN()
_C.OPTIM.ADAMW.BETAS = (0.9, 0.999)
_C.OPTIM.ADAMW.EPS = 1e-8
_C.OPTIM.ADAMW.WEIGHT_DECAY = 0.01


# -----------------------------------------------------------------------------
# Learning Rate Scheduler
# -----------------------------------------------------------------------------
_C.LRS = CN()
_C.LRS.NAME = None

""" CosineLRScheduler """
# set learning rate scheduler parameters in training
""" PolyLRScheduler """
_C.LRS.POLY = CN()
_C.LRS.POLY.POWER = 0.9
""" LinearLRScheduler """
# set learning rate scheduler parameters in training
""" StepLRScheduler """
_C.LRS.STEP_LR = CN()
_C.LRS.STEP_LR.DECAY_RATE = 0.1


# -----------------------------------------------------------------------------
# Criterion
# -----------------------------------------------------------------------------
_C.CRITERION = CN()
_C.CRITERION.NAME = None
_C.CRITERION.SOFTMAX = None
_C.CRITERION.SIGMOID = None
_C.CRITERION.EPS = 1e-5

""" DiceCELoss """
_C.CRITERION.DICECE = CN()
_C.CRITERION.DICECE.LAMBDA_CE = 0.5
_C.CRITERION.DICECE.LAMBDA_DICE = 0.5


# -----------------------------------------------------------------------------
# Metric
# -----------------------------------------------------------------------------
_C.METRIC = CN ()

""" DiceMetric """
_C.METRIC.DICE = CN()
_C.METRIC.DICE.INCL_BG = None
_C.METRIC.DICE.REDUCTION = "mean_batch"
_C.METRIC.DICE.GET_NOT_NANS = True


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()

""" Data """
_C.DATA.NAME = None
_C.DATA.ROOT_DIR = None
_C.DATA.TRAIN_DIR = None
_C.DATA.VAL_DIR = None
_C.DATA.TEST_DIR = None
_C.DATA.IMG_SIZE = None

""" Transform """
_C.DATA.TRANSFORM = CN()
_C.DATA.TRANSFORM.AUGMENT = True
_C.DATA.TRANSFORM.P = 0.5
_C.DATA.TRANSFORM.DEGREE = [0, 360]


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.LOADER = CN()

""" Train DataLoader """
_C.LOADER.TRAIN = CN()
_C.LOADER.TRAIN.BATCH_SIZE = 32
_C.LOADER.TRAIN.SHUFFLE = True
_C.LOADER.TRAIN.NUM_WORKERS = 1
_C.LOADER.TRAIN.PIN_MEMORY = True

""" Validation DataLoader """
_C.LOADER.VAL = CN()
_C.LOADER.VAL.BATCH_SIZE = 128
_C.LOADER.VAL.SHUFFLE = True
_C.LOADER.VAL.NUM_WORKERS = 1
_C.LOADER.VAL.PIN_MEMORY = True

""" Test DataLoader """
_C.LOADER.TEST = CN()
_C.LOADER.TEST.BATCH_SIZE = 128
_C.LOADER.TEST.SHUFFLE = True
_C.LOADER.TEST.NUM_WORKERS = 1
_C.LOADER.TEST.PIN_MEMORY = True


# -----------------------------------------------------------------------------
# Hyperparams
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(seed=SEED)
random.seed(a=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed_all(seed=SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
LOG_LEVEL = LogLevel.INFO


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

""" Training """
# epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.DECAY_EPOCHS = 30
_C.TRAIN.MAX_NORM = 1.0
_C.TRAIN.N_ITER_PER_EPOCH = 163
_C.TRAIN.WARMUP_EPOCHS = 2
_C.TRAIN.N_EPOCHS = 100
_C.TRAIN.STATS_FILEPATH = "stats.json"


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()

""" Validation """
_C.VAL.VAL_EVERY_N_EPOCH = 1


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print(f'[INFO] => merge config from `{cfg_file}`')
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)
    _update_config_from_file(config, args.dataset)

    opts = getattr(args, 'opts', None)
    if opts:
        config.defrost()
        config.merge_from_list(opts)
        config.freeze()


def get_config(args):
    """
    Get a yacs CfgNode object with default values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg = _C.clone()
    update_config(cfg, args)

    return cfg
