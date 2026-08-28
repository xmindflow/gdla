import torch.nn as nn
from registry import CRITERION


def build_criterion(cfg) -> nn.Module:
    name = cfg.CRITERION.NAME
    if name not in CRITERION.registry:
        raise KeyError(f"Criterion `{name}` not found in registry. "
                       f"Available: {list(CRITERION.registry.keys())}")
    return CRITERION.registry[name](cfg=cfg)
