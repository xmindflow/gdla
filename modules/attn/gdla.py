from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..rmsnorm import RMSNorm


class GatedDiffLinearAttn(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            qkv_bias: bool,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.d = self.head_dim // 2
        self.kernel = nn.ELU(inplace=False)
        self.qkvg = nn.Linear(
            in_features=dim, out_features=4 * dim, bias=qkv_bias,
        )
        self.ln = RMSNorm(2 * self.d, eps=1e-5, elementwise_affine=True)
        self.dwc = nn.Conv2d(
            in_channels=4 * dim,
            out_channels=4 * dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=4 * dim,
            bias=True,
        )
        self.pwc = nn.Conv2d(
            in_channels=4 * dim,
            out_channels=4 * dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.lambda_scale_main = nn.Parameter(
            torch.zeros(self.n_heads, 2 * self.d).normal_(mean=0.0, std=0.1)
        )
        self.lambda_scale_aux = nn.Parameter(
            torch.zeros(self.n_heads, 2 * self.d).normal_(mean=0.0, std=0.1)
        )
        self.out_proj = nn.Linear(
            in_features=2 * dim, out_features=dim, bias=True
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'lambda_scale_main', 'lambda_scale_aux'}

    def la(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        q: [B,H,2d,N]
        k: [B,H,2d,N]
        v: [B,H,2d,N]
        y: [B,H,2d,N]
        """
        q = self.kernel(q) + 1.0
        k = self.kernel(k) + 1.0
        # [B,H,2d,N] -> [B,H,2d+1,N]
        v = F.pad(input=v, pad=(0, 0, 0, 1), mode='constant', value=1)
        # [B,H,2d,N] -> [B,H,N,2d]
        k = k.transpose(dim0=-2, dim1=-1)  
        # [B,H,2d+1,N] @ [B,H,N,2d] -> [B,H,2d+1,2d]
        vk = v @ k
        # [B,H,2d+1,2d] @ [B,H,2d,N] -> [B,H,2d+1,N]
        y = vk @ q
        # [B,H,2d,N]/[B,H,2d,1] -> [B,H,2d,N]
        y = y[:, :, :-1] / (y[:, :, -1:] + 1e-6)

        return y

    def diff_attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g: Tensor,
        lambda_scale: Tensor,
    ) -> Tensor:
        """
        q: [B,H,2d,N]
        k: [B,H,2d,N]
        v: [B,H,2d,N]
        y: [B,H,2d,N]
        """
        q1, q2 = torch.split(q, self.d, dim=-2)     # [B,H,d,N]
        k1, k2 = torch.split(k, self.d, dim=-2)     # [B,H,d,N]
        y1 = self.la(q=q1, k=k1, v=v)               # [B,H,2d,N]
        y2 = self.la(q=q2, k=k2, v=v)               # [B,H,2d,N]

        lambda_ = lambda_scale.view(1, self.n_heads, 2 * self.d, 1)

        y = y1 - lambda_ * y2                       # [B,H,2d,N]
        y = self.ln(y.transpose(dim0=-2, dim1=-1)).transpose(dim0=-2, dim1=-1) # [B,H,2d,N]
        # data-dependent gate (Eq. 20)
        # [B,H,2d,N] * [B,H,2d,N] -> [B,H,2d,N]
        y = y * torch.sigmoid(g)                    # [B,H,2d,N]

        return y

    def apply_diff_attn(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            g: Tensor,
            lambda_scale: Tensor,
    ) -> Tensor:
        """
        q: [B,N,C]
        k: [B,N,C]
        v: [B,N,C]
        g: [B,N,C]
        y: [B,N,C]
        """
        B, N, C = q.shape

        # [B,N,C] -> [B,C,N] -> [B,H,2d,N]
        q = q.transpose(dim0=-2, dim1=-1).reshape(B, self.n_heads, self.head_dim, N)
        k = k.transpose(dim0=-2, dim1=-1).reshape(B, self.n_heads, self.head_dim, N)
        v = v.transpose(dim0=-2, dim1=-1).reshape(B, self.n_heads, self.head_dim, N)
        g = g.transpose(dim0=-2, dim1=-1).reshape(B, self.n_heads, self.head_dim, N)

        # [B,H,2d,N]
        y = self.diff_attn(
            q=q, k=k, v=v, g=g,
            lambda_scale=lambda_scale,
        )
        # [B,H,2d,N] -> [B,C,N] -> [B,N,C]
        y = y.reshape(B, C, N).transpose(dim0=-2, dim1=-1)

        return y

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        
        # [B,N,4C]
        qkvg = self.qkvg(x)
        # [B,N,4C] -> [B,4C,N] -> [B,4C,H,W]
        qkvg_ = qkvg.transpose(dim0=-2, dim1=-1)\
            .reshape(B, 4 * C, int(N ** 0.5), int(N ** 0.5))
        qkvg_ = self.dwc(qkvg_)                                         # [B,4C,H,W]
        qkvg_ = self.pwc(qkvg_)                                         # [B,4C,H,W]
        qkvg_ = qkvg_.reshape(B, 4 * C, N).transpose(dim0=-2, dim1=-1)  # [B,N,4C]

        qkvg = qkvg.reshape(B, N, 4, C)     # [B,N,4C] -> [B,N,4,C]
        q, k, v, g = qkvg.unbind(dim=2)     # each [B,N,C]

        y1 = self.apply_diff_attn(
            q=q, k=k, v=v, g=g,
            lambda_scale=self.lambda_scale_main,
        )

        qkvg = qkvg_.reshape(B, N, 4, C)    # [B,N,4C] -> [B,N,4,C]
        q, k, v, g = qkvg.unbind(dim=2)     # each [B,N,C]

        y2 = self.apply_diff_attn(
            q=q, k=k, v=v, g=g,
            lambda_scale=self.lambda_scale_aux,
        )

        y = torch.cat([y1, y2], dim=-1)     # [B,N,2C]
        y = self.out_proj(y)                # [B,N,C]

        return y
