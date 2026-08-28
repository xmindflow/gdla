from torch import Tensor
from typing import Tuple

import torch.nn as nn
from modules.rope import RoPE


class MultiHeadLinearAttn(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            qkv_bias: bool,
            use_rope: bool = False,
            rope_base: float = 10000.0,
            img_size: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.kernel = nn.ELU(inplace=False)
        self.qkv = nn.Linear(
            in_features=dim, out_features=3 * dim, bias=qkv_bias
        )

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE(
                dim=self.head_dim,
                max_seq_len=img_size[0] * img_size[1],
                base=rope_base,
            )

        self.out_proj = nn.Linear(in_features=dim, out_features=dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)

        q = q.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.kernel(q) + 1.0
        k = self.kernel(k) + 1.0

        if self.use_rope:
            q, k = self.rope.apply_rotary(q, k)

        z = 1.0 / (q @ k.sum(dim=-2, keepdim=True).transpose(dim0=-2, dim1=-1) + 1e-6)
        kv = (k.transpose(dim0=-2, dim1=-1)) @ v

        x = q @ kv * z

        x = x.transpose(dim0=1, dim1=2).reshape(B, N, C)
        x = self.out_proj(x)

        return x
