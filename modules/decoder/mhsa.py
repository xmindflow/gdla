from torch import Tensor
from typing import Tuple

import torch.nn as nn
from ..attn.mhsa import MultiHeadSelfAttn
from ..ffn.mlp import MLP
from timm.layers import DropPath


class MHSA(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            qkv_bias: bool,
            use_rope: bool = False,
            rope_base: float = 10000.0,
            img_size: Tuple[int, int] | None = None,
            norm_eps: float = 1e-5,
            hidden_ratio: float = 4.0,
            drop_prob: float = 0.,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=dim, eps=norm_eps)
        self.attn = MultiHeadSelfAttn(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_base=rope_base,
            img_size=img_size,
        )
        self.norm2 = nn.LayerNorm(normalized_shape=dim, eps=norm_eps)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(hidden_ratio * dim),
            out_features=dim,
            drop=drop_prob,
        )
        self.drop_path = DropPath(drop_prob=drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(dim0=-2, dim1=-1)

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)
