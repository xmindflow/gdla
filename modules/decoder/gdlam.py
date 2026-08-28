from torch import Tensor

import torch.nn as nn
from ..attn.gdla import GatedDiffLinearAttn
from ..ffn.gmlp import GMLP
from timm.layers import DropPath


class GDLAM(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            qkv_bias: bool,
            norm_eps: float = 1e-5,
            hidden_ratio: float = 4.0,
            drop_prob: float = 0.,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=dim, eps=norm_eps)
        self.dwc = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.act = nn.SiLU()
        self.attn = GatedDiffLinearAttn(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(normalized_shape=dim, eps=norm_eps)
        self.mlp = GMLP(
            in_features=dim,
            hidden_features=int(hidden_ratio * dim),
            out_features=dim,
            drop=drop_prob,
        )
        self.drop_path = DropPath(drop_prob=drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(dim0=-2, dim1=-1)

        _x = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.dwc(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        x = self.attn(x)
        x = _x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)
