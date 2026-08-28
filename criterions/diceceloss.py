from torch import Tensor
from typing import Optional

import torch.nn as nn
from .diceloss import DiceLoss
from registry import CRITERION


class DiceCELoss(nn.Module):
    def __init__(self,
            softmax: bool,
            sigmoid: bool,
            eps: float = 1e-5,
            lambda_dice: float = 1.0,
            lambda_ce: float = 1.0,
    ) -> None:
        super().__init__()

        if softmax and sigmoid:
            raise ValueError("Only one of `softmax` or `sigmoid` can be True.")

        if lambda_dice < 0.0 or lambda_ce < 0.0:
            raise ValueError("`lambda_dice` and `lambda_ce` should be >= 0.0.")

        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

        self.ce = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(softmax=softmax, sigmoid=sigmoid, eps=eps)

    def forward(
            self,
            input: Tensor,
            target: Tensor,
            class_weight: Optional[Tensor] = None,
    ) -> Tensor:
        ce_loss = self.ce(input=input, target=target)
        dice_loss = self.dice_loss(
            input=input, target=target, class_weight=class_weight
        )
        loss = self.lambda_ce * ce_loss + self.lambda_dice * dice_loss

        return loss


@CRITERION.register(name="dicece")
def build_model(cfg) -> nn.Module:
    return DiceCELoss(
        softmax=cfg.CRITERION.SOFTMAX,
        sigmoid=cfg.CRITERION.SIGMOID,
        eps=cfg.CRITERION.EPS,
        lambda_ce=cfg.CRITERION.DICECE.LAMBDA_CE,
        lambda_dice=cfg.CRITERION.DICECE.LAMBDA_DICE,
    )
