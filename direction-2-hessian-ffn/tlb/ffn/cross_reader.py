"""Decoder FFN: Standard 4D backbone + zero-init cross-attention reader branch."""
from __future__ import annotations

import torch
import torch.nn as nn

from tlb.ffn.standard import StandardFFN


class CrossAttnReaderFFN(nn.Module):
    """Standard MLP plus a low-rank reader that consumes cross-attention output.

    At init the reader branch is zero (``gate=0``, ``out_proj`` zero-init), so the
    block matches Standard 4D FFN — useful for warm-starting from a Standard decoder
    while learning to use cross-attention context incrementally.
    """

    def __init__(
        self,
        dim: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        *,
        reader_rank: int = 32,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.reader_rank = int(reader_rank)
        self.base = StandardFFN(
            dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        rank = max(4, self.reader_rank)
        self.u_proj = nn.Linear(dim, rank, bias=False)
        self.v_proj = nn.Linear(dim, rank, bias=False)
        self.out_proj = nn.Linear(rank, dim, bias=True)
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cross_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.base(x)
        if cross_ctx is None:
            return out
        mid = self.u_proj(x) * self.v_proj(cross_ctx)
        read = self.out_proj(mid)
        return out + self.gate.tanh() * read
