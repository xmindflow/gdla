from torch import Tensor

import torch.nn as nn


class ResBlk(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.proj = in_chs != out_chs
        self.conv1 = nn.Conv2d(
            in_channels=in_chs,
            out_channels=in_chs,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(num_features=in_chs, eps=norm_eps)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=in_chs,
            out_channels=out_chs,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(num_features=out_chs, eps=norm_eps)
        if self.proj:
            self.conv3 = nn.Conv2d(
                in_channels=in_chs,
                out_channels=out_chs,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.norm3 = nn.BatchNorm2d(num_features=out_chs, eps=norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        _x = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.proj:
            _x = self.conv3(_x)
            _x = self.norm3(_x)
        x = _x + x
        x = self.act(x)

        return x
