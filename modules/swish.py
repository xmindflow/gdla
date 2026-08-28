from torch import Tensor

import torch.nn as nn


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.act(x)
