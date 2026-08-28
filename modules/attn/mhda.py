from torch import Tensor
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..rmsnorm import RMSNorm
from ..rope import RoPE


def lambda_init_fn(depth: Tensor) -> Tensor:
    return 0.8 - 0.6 * torch.exp(-0.3 * depth)


class MultiHeadDiffAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        depth: int,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        img_size: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.d = self.head_dim // 2
        self.scaling = self.d ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE(
                dim=self.d,
                max_seq_len=img_size[0] * img_size[1],
                base=rope_base,
            )

        self.lambda_init = lambda_init_fn(torch.tensor(depth))
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.d, eps=1e-5, elementwise_affine=True)

        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        q = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = q.unbind(2)

        q = q.view(B, N, 2 * self.n_heads, self.d).permute(0, 2, 1, 3)
        k = k.view(B, N, 2 * self.n_heads, self.d).permute(0, 2, 1, 3)
        v = v.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.use_rope:
            q, k = self.rope.apply_rotary(q, k)

        q *= self.scaling
        attn_scores = q @ k.transpose(dim0=-2, dim1=-1)
        attn_scores = F.softmax(attn_scores, dim=-1, dtype=torch.float32)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_scores = attn_scores.view(B, self.n_heads, 2, N, N)
        attn_scores = attn_scores[:, :, 0] - lambda_full * attn_scores[:, :, 1]

        attn = attn_scores @ v
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(dim0=1, dim1=2).reshape(B, N, C)

        attn = self.out_proj(attn)

        return attn
