# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()


# -----------------------------------------------------------------------------
# SEARCH
# -----------------------------------------------------------------------------
_C.SEARCH = CN()
_C.SEARCH.ARCH_START_EPOCH = 20
_C.SEARCH.VAL_PORTION = 0.02
_C.SEARCH.PORTION = 0.5
_C.SEARCH.SEARCH_ON = False
_C.SEARCH.TIE_CELL = True


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "AutoDeepLab"
_C.MODEL.FILTER_MULTIPLIER = 20
_C.MODEL.NUM_LAYERS = 12
_C.MODEL.NUM_BLOCKS = 5
_C.MODEL.NUM_STRIDES = 4
_C.MODEL.IN_CHANNEL = 3
_C.MODEL.AFFINE = True
_C.MODEL.WEIGHT = ""  # Init weights
_C.MODEL.PRIMITIVES = "AutoDeepLab"
# _C.MODEL.ASPP_RATES = (6, 12, 18)
_C.MODEL.ASPP_RATES = (2, 4, 6)
_C.MODEL.META_MODE = "Scale"


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = -1
# Crop size of the side of the image during training
_C.INPUT.CROP_SIZE_TRAIN = 128
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1024
_C.INPUT.MIN_SIZE_TEST = -1
_C.INPUT.MAX_SIZE_TEST = 1024


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
# _C.DATASET.DATA_ROOT = "/home/hkzhang/Documents/sdb_a/nas_data"
_C.DATASET.DATA_ROOT = "/home/lidi/Documents/BSR/BSD500/data"
_C.DATASET.DATA_NAME = "BSD500_300"
_C.DATASET.TRAIN_DATASETS = []
_C.DATASET.TRAIN_DATASETS_WEIGHT = []
_C.DATASET.TEST_DATASETS = []
_C.DATASET.CROP_SIZE = 128
_C.DATASET.TASK = "denoise"
_C.DATASET.LOAD_ALL = False
_C.DATASET.TO_GRAY = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.BATCH_SIZE_TRAIN = 2
_C.DATALOADER.BATCH_SIZE_TEST = 2
_C.DATALOADER.SIGMA = []
_C.DATALOADER.DATA_LIST_DIR = "../preprocess/dataset_json"
_C.DATALOADER.DATA_AUG = 1

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 60
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.WEIGHT_DECAY = 0.00004
_C.SOLVER.MOMENTUM = 0.9
# cosine learning rate
_C.SOLVER.SEARCH = CN()
_C.SOLVER.SEARCH.LR_START = 0.025
_C.SOLVER.SEARCH.LR_END = 0.001
_C.SOLVER.SEARCH.MOMENTUM = 0.9
_C.SOLVER.SEARCH.WEIGHT_DECAY = 0.0003
# architecture encoding Adam params
_C.SOLVER.SEARCH.LR_A = 0.001  # learning rate
_C.SOLVER.SEARCH.WD_A = 0.001  # weight decay
_C.SOLVER.SEARCH.T_MAX = 10  # cosine lr time

_C.SOLVER.TRAIN = CN()
_C.SOLVER.TRAIN.INIT_LR = 0.05
# _C.SOLVER.TRAIN.INIT_LR = 0.001
_C.SOLVER.TRAIN.POWER = 0.9
_C.SOLVER.TRAIN.MAX_ITER = 500000
_C.SOLVER.TRAIN.VAL_PORTION = 0.01
_C.SOLVER.SCHEDULER = 'poly'  # poly lr
_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.VALIDATE_PERIOD = 1

_C.SOLVER.LOSS = ['l1', 'neg_ssim', 'grad']
_C.SOLVER.LOSS_WEIGHT = [1, 1, 1]


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.RESULT_DIR = "."

