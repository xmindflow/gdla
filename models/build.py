import torch.nn as nn
from registry import MODEL


def build_model(cfg) -> nn.Module:
    name = cfg.MODEL.NAME
    if name not in MODEL.registry:
        raise KeyError(f"Model `{name}` not found in registry. "
                       f"Available: {list(MODEL.registry.keys())}")
    return MODEL.registry[name](cfg=cfg)
