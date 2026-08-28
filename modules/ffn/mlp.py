from torch import Tensor
from typing import Optional, Tuple

import torch.nn as nn


class MLP(nn.Module):
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

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])

        self.act = nn.GELU()

        self.drop1 = nn.Dropout(drop[0]) if drop[0] > 0 else nn.Identity()
        self.drop2 = nn.Dropout(drop[1]) if drop[1] > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if getattr(self.fc1, "bias", None) is not None:
            nn.init.zeros_(self.fc1.bias)
        if getattr(self.fc2, "bias", None) is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x
