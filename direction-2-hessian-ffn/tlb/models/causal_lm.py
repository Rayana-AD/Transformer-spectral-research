"""Causal decoder-only language model (GPT-style stack)."""
from __future__ import annotations

import torch
import torch.nn as nn

from tlb.attention.angular import AttentionType, NormalizationMode
from tlb.blocks.masks import build_causal_mask
from tlb.blocks.transformer_layer import TransformerEncoderLayer
from tlb.ffn.hessian_core import HessianVariant
from tlb.models.seq_model import sinusoidal_position_encoding


class CausalLM(nn.Module):
    """Decoder-only LM: causal self-attn + FFN (standard or Hessian) + lm_head."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        nhead: int,
        *,
        max_seq_len: int,
        dim_feedforward: int | None = None,
        pos_enc: str = "learned",
        dropout: float = 0.0,
        attention_type: AttentionType = "angular",
        normalize_mode: NormalizationMode = "row",
        alpha: float = 2.0,
        cottention_sigma: float = 0.5,
        ffn_type: str = "standard",
        ffn_activation: str = "gelu",
        hessian_variant: HessianVariant = "h1",
        hessian_num_units: int | None = None,
        pos_sin_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.attention_type = attention_type
        self.pos_enc = pos_enc
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim) if pos_enc == "learned" else None
        if pos_enc == "sinusoidal":
            if pos_sin_buffer is None:
                pos_sin_buffer = sinusoidal_position_encoding(max_seq_len, dim)
            self.register_buffer("pos_sin", pos_sin_buffer, persistent=False)
        else:
            self.pos_sin = None  # type: ignore[assignment]

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim,
                    nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_type=attention_type,
                    normalize_mode=normalize_mode,
                    alpha=alpha,
                    cottention_sigma=cottention_sigma,
                    ffn_type=ffn_type,
                    ffn_activation=ffn_activation,
                    hessian_variant=hessian_variant,
                    hessian_num_units=hessian_num_units,
                )
                for _ in range(depth)
            ]
        )
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

    def _add_pos(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if self.pos_embed is not None:
            pos = torch.arange(seq_len, device=x.device, dtype=torch.long)
            return x + self.pos_embed(pos)
        if self.pos_enc == "sinusoidal" and self.pos_sin is not None:
            return x + self.pos_sin[:seq_len].to(device=x.device, dtype=x.dtype)
        return x

    def _causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return build_causal_mask(
            seq_len,
            device,
            attention_type=self.attention_type,
            dtype=dtype,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits (B, T, V); position t predicts token t+1."""
        x = self.embed(token_ids)
        x = self._add_pos(x)
        attn_mask = self._causal_mask(token_ids.size(1), token_ids.device, x.dtype)
        for layer in self.layers:
            x = layer(
                x,
                src_key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )  # type: ignore[assignment]
        return self.lm_head(x)

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
