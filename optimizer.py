import torch.nn as nn
import torch.optim as optim


def build_optimizer(cfg, model: nn.Module) -> optim.Optimizer:
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    
    if hasattr(model, 'lower_lr_kvs'):
        lower_lr_kvs = model.lower_lr_kvs
    else:
        lower_lr_kvs = {}

    parameters = set_weight_decay_and_lr(
        model=model,
        skip_list=skip,
        skip_keywords=skip_keywords,
        lower_lr_kvs=lower_lr_kvs,
        base_lr=cfg.OPTIM.BASE_LR,
    )

    name = cfg.OPTIM.NAME.lower()

    optimizer = None
    if name == 'sgd':
        optimizer = optim.SGD(
            params=parameters,
            lr=cfg.OPTIM.BASE_LR,
            momentum=cfg.OPTIM.SGD.MOMENTUM,
            weight_decay=cfg.OPTIM.SGD.WEIGHT_DECAY,
            nesterov=cfg.OPTIM.SGD.NESTEROV,
        )
    elif name == 'adamw':
        optimizer = optim.AdamW(
            params=parameters,
            lr=cfg.OPTIM.BASE_LR,
            betas=cfg.OPTIM.ADAMW.BETAS,
            eps=cfg.OPTIM.ADAMW.EPS,
            weight_decay=cfg.OPTIM.ADAMW.WEIGHT_DECAY,
        )
    else:
        raise ValueError("Invalid optimizer. Choose from {{'sgd', 'adamw'}}.")

    return optimizer


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def set_weight_decay_and_lr(
    model: nn.Module, 
    skip_list=(),
    skip_keywords=(), 
    lower_lr_kvs={},
    base_lr=5e-3,
) -> list[dict]:
    assert len(lower_lr_kvs) == 1 or len(lower_lr_kvs) == 0
    has_lower_lr = len(lower_lr_kvs) == 1
    if has_lower_lr:
        for k,v in lower_lr_kvs.items():
            lower_lr_key = k
            lower_lr = v * base_lr

    has_decay = []
    has_decay_low = []
    no_decay = []
    no_decay_low = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):

            if has_lower_lr and check_keywords_in_name(name, (lower_lr_key,)):
                no_decay_low.append(param)
            else:
                no_decay.append(param)
            
        else:

            if has_lower_lr and check_keywords_in_name(name, (lower_lr_key,)):
                has_decay_low.append(param)
            else:
                has_decay.append(param)

    if has_lower_lr:
        result = [
            {'params': has_decay},
            {'params': has_decay_low, 'lr': lower_lr},
            {'params': no_decay, 'weight_decay': 0.},
            {'params': no_decay_low, 'weight_decay': 0., 'lr': lower_lr}
        ]
    else:
        result = [
            {'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}
        ]

    return result
