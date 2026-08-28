from torch import Tensor
from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from modules.rope import RoPE


class MultiHeadSelfAttn(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            qkv_bias: bool = True,
            use_rope: bool = False,
            rope_base: float = 10000.0,
            img_size: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

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

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B,N,C]
        returns: [B,N,C]
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)

        q = q.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.use_rope:
            q, k = self.rope.apply_rotary(q, k)

        # attention scores
        attn = q @ k.transpose(-2, -1) * self.scale  # (B, H, N, N)

        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # [B,H,N,D]
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        out = self.out_proj(out)

        return out
