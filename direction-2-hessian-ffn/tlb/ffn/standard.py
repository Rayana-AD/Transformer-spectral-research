"""Standard two-layer FFN."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * dim
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        elif activation == "silu":
            self.act = F.silu
        else:
            raise ValueError(f"unknown activation: {activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.act(self.linear1(x))))


class GatedFFN(nn.Module):
    """SwiGLU or GeGLU at the same intermediate width as a two-layer MLP.

    Not a lambda translation. The extra up-projection can be initialized to 1
    so SwiGLU reproduces a SiLU MLP at step 0.
    """

    def __init__(
        self,
        dim: int,
        dim_feedforward: int,
        *,
        gate: str = "swiglu",
    ) -> None:
        super().__init__()
        if gate not in ("swiglu", "geglu"):
            raise ValueError(f"unknown gate: {gate!r}")
        # Full intermediate width so a pretrained SiLU MLP can be copied
        # exactly (gate ← W1, up ← 1). Param-matched 2/3 width cannot.
        hidden = int(dim_feedforward)
        self.gate_kind = gate
        self.up = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)
        self._act = F.silu if gate == "swiglu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self._act(self.gate(x)) * self.up(x))
