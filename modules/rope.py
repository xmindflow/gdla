import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Standard 1-D rotary positional embedding (GPT-NeoX/LLaMA style)."""

    def __init__(
            self, dim: int,
            max_seq_len: int,
            base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even.")
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len, device=torch.device("cpu"))

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, self.inv_freq.to(device))  # [seq_len, dim/2]
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)       # [seq_len, dim]
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        self.cos_cached = cos
        self.sin_cached = sin

    def _prepare_cache(self, seq_len: int, device: torch.device) -> None:
        if self.cos_cached.size(0) >= seq_len and self.cos_cached.device == device:
            return
        seq_len = max(seq_len, self.cos_cached.size(0) if self.cos_cached.numel() else 0)
        seq_len = max(seq_len, 1)
        self._build_cache(seq_len, device=device)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, H, N, D]
        B, H, N, D = q.shape
        self._prepare_cache(N, q.device)
        cos = self.cos_cached[:N].view(1, 1, N, D)
        sin = self.sin_cached[:N].view(1, 1, N, D)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
