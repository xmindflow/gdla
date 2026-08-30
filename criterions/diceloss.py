from torch import Tensor
from typing import Optional

import torch
import torch.nn as nn
from registry import CRITERION
from utils import one_hot


class DiceLoss(nn.Module):
    def __init__(self, softmax: bool, sigmoid: bool, eps: float = 1e-5) -> None:
        super().__init__()

        if softmax and sigmoid:
            raise ValueError("Only one of `softmax` or `sigmoid` can be True.")

        self.softmax = softmax
        self.sigmoid = sigmoid
        self.eps = eps

    def forward(
            self,
            input: Tensor,
            target: Tensor,
            class_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if self.softmax:
            input = torch.softmax(input=input, dim=1)
        elif self.sigmoid:
            input = torch.sigmoid(input=input)

        target = one_hot(
            labels=target.unsqueeze(dim=1), n_classes=input.shape[1], dim=1
        )

        if input.size() != target.size():
            raise ValueError(
                f"Input and target must have the same shape, got "
                f"input shape {input.shape} vs target shape {target.shape}."
            )

        if class_weight is not None and class_weight.numel() != input.shape[1]:
            raise ValueError(
                f"Class weights length {class_weight.numel()} must match "
                f"number of channels {input.shape[1]}."
            )

        dim = [0] + list(range(2, target.dim()))

        intersect = torch.sum(input * target, dim=dim)
        y_sum = torch.sum(target * target, dim=dim)
        z_sum = torch.sum(input * input, dim=dim)
        loss = 1 - (2 * intersect + self.eps) / (z_sum + y_sum + self.eps)

        if class_weight is not None:
            class_weight = class_weight.to(dtype=loss.dtype)
            loss = loss * class_weight

        return loss.mean(dim=0)


@CRITERION.register(name="dice")
def build_criterion(cfg) -> nn.Module:
    return DiceLoss(
        softmax=cfg.CRITERION.SOFTMAX,
        sigmoid=cfg.CRITERION.SIGMOID,
        eps=cfg.CRITERION.EPS,
    )
