from .auto_multitask import AutoMultiTask
from .amt_discrete import AMTDiscrete, AMTDiscrete_geno


ARCHITECTURES = {
    "AutoMultiTask": AutoMultiTask,
    "AMTDiscrete": AMTDiscrete
}


def build_model(cfg):
    meta_arch = ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
