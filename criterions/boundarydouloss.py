from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from registry import CRITERION
from utils import one_hot


class BoundaryDoULoss(nn.Module):
    def __init__(self, softmax: bool, sigmoid: bool, eps: float = 1e-5) -> None:
        super().__init__()

        if softmax and sigmoid:
            raise ValueError("Only one of `softmax` or `sigmoid` can be True.")

        self.softmax = softmax
        self.sigmoid = sigmoid
        self.eps = eps

        kernel = torch.tensor([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer(name='kernel', tensor=kernel)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.softmax:
            input = torch.softmax(input=input, dim=1)
        elif self.sigmoid:
            input = torch.sigmoid(input=input)

        target = one_hot(
            labels=target.unsqueeze(dim=1), n_classes=input.shape[1], dim=1
        )

        kernel = self.kernel.to(dtype=target.dtype, device=target.device)
        kernel = kernel.repeat(target.size(dim=1), 1, 1, 1)

        Y = F.conv2d(
            input=target,
            weight=kernel,
            bias=None,
            stride=1,
            padding=1,
            groups=target.size(dim=1),
        )
        Y = Y * target
        Y[Y == 5] = 0

        dim = [0] + list(range(2, target.dim()))
        C = torch.count_nonzero(input=Y, dim=dim)
        S = torch.count_nonzero(input=target, dim=dim)

        alpha = 1 - (C + self.eps) / (S + self.eps)
        alpha = 2 * alpha - 1
        alpha = torch.clamp(input=alpha, max=0.8)

        intersect = torch.sum(input * target, dim=[0, 2, 3])
        y_sum = torch.sum(target * target, dim=[0, 2, 3])
        z_sum = torch.sum(input * input, dim=[0, 2, 3])
        loss = (z_sum + y_sum - 2 * intersect + self.eps) / \
            (z_sum + y_sum - (1 + alpha) * intersect + self.eps)

        return loss.mean(dim=0)


@CRITERION.register(name="boundary")
def build_criterion(cfg) -> nn.Module:
    return BoundaryDoULoss(
        softmax=cfg.CRITERION.SOFTMAX,
        sigmoid=cfg.CRITERION.SIGMOID,
        eps=cfg.CRITERION.EPS,
    )
