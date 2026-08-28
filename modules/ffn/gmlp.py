from torch import Tensor
from typing import Optional, Tuple

import torch
import torch.nn as nn


class GMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: Tuple[float, float] | float = 0.0,
        bias: Tuple[bool, bool] | bool = True,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or (in_features * 4)
        out_features = out_features or in_features

        if isinstance(drop, (int, float)):
            drop = (drop, drop)
        if isinstance(bias, bool):
            bias = (bias, bias)

        self.pwc1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_features * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.act = nn.SiLU()

        self.dwc = nn.Conv2d(
            in_channels=hidden_features * 2,
            out_channels=hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=True,
        )

        self.pwc2 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        H, W = int(N ** 0.5), int(N ** 0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.pwc1(x)
        x = self.act(x)
        x = self.dwc(x)
        x, gate = x.chunk(chunks=2, dim=1)
        gate = self.act(gate)
        x = x * gate
        x = self.pwc2(x)

        x = x.permute(0, 2, 3, 1).reshape(B, N, -1)

        return x
    