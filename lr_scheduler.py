import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.poly_lr import PolyLRScheduler
from timm.scheduler.scheduler import Scheduler
from timm.scheduler.step_lr import StepLRScheduler


class LinearLRScheduler(Scheduler):
    def __init__(
            self,
            optimizer: optim.Optimizer,
            t_initial: int,
            lr_min_rate: float,
            warmup_t=0,
            warmup_lr_init=0.,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t for v in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [
                v - ((v - v * self.lr_min_rate) * (t / total_t))
                for v in self.base_values
            ]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


def build_scheduler(cfg, optimizer: optim.Optimizer) -> Scheduler:
    n_steps = int(cfg.TRAIN.N_ITER_PER_EPOCH * cfg.TRAIN.N_EPOCHS)
    warmup_steps = int(cfg.TRAIN.N_ITER_PER_EPOCH * cfg.TRAIN.WARMUP_EPOCHS)
    decay_steps = int(cfg.TRAIN.N_ITER_PER_EPOCH * cfg.TRAIN.DECAY_EPOCHS)

    if cfg.LRS.NAME is not None:
        name = cfg.LRS.NAME.lower()
    else:
        name = None

    lr_scheduler = None
    if name == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=n_steps,
            lr_min=cfg.OPTIM.MIN_LR,
            warmup_lr_init=cfg.OPTIM.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif name == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            t_initial=n_steps,
            lr_min=cfg.OPTIM.MIN_LR,
            power=cfg.LRS.POLY.POWER,
            warmup_lr_init=cfg.OPTIM.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif name == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=n_steps,
            lr_min_rate=cfg.OPTIM.MIN_LR,
            warmup_lr_init=cfg.OPTIM.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif name == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=cfg.LRS.STEP_LR.DECAY_RATE,
            warmup_lr_init=cfg.OPTIM.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler
