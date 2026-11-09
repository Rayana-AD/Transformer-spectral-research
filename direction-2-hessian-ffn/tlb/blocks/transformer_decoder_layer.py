"""Transformer decoder layer: causal self-attn + cross-attn + FFN."""
from __future__ import annotations

import torch
import torch.nn as nn

from tlb.attention.angular import AttentionType, NormalizationMode
from tlb.blocks.cross_attention import MultiheadCrossAttention
from tlb.blocks.transformer_layer import MultiheadSelfAttention
from tlb.blocks.hidden_adapter import HiddenStateAdapter
from tlb.ffn.cross_reader import CrossAttnReaderFFN
from tlb.ffn.hessian_core import DecoupleMinLambda, HessianFFN, HessianVariant
from tlb.ffn.standard import StandardFFN


class TransformerDecoderLayer(nn.Module):
    """Post-LN decoder: self-attn → cross-attn → FFN with residuals."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        *,
        attention_type: AttentionType = "vanilla",
        normalize_mode: NormalizationMode = "row",
        alpha: float = 2.0,
        cottention_sigma: float = 0.5,
        ffn_type: str = "standard",
        ffn_activation: str = "gelu",
        hessian_variant: HessianVariant = "h1",
        hessian_lambda_geometry: DecoupleMinLambda | None = None,
        hessian_tanh2_temperature: float | None = None,
        hessian_tanh2_amplitude: str | None = None,
        hessian_tanh2_amplitude_scale: float | None = None,
        hessian_tanh2_peak_centers: tuple[float, ...] | list[float] | None = None,
        hessian_tanh2_peak_shift: float | None = None,
        cross_reader_rank: int = 32,
        ffn_adapter: HiddenStateAdapter | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadSelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            attention_type=attention_type,
            normalize_mode=normalize_mode,
            alpha=alpha,
            cottention_sigma=cottention_sigma,
        )
        self.cross_attn = MultiheadCrossAttention(
            d_model,
            nhead,
            dropout=dropout,
            attention_type=attention_type,
            normalize_mode=normalize_mode,
            alpha=alpha,
            cottention_sigma=cottention_sigma,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self._ffn_needs_cross_ctx = False
        if ffn_type == "standard":
            self.ffn: nn.Module = StandardFFN(
                d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=ffn_activation,
            )
        elif ffn_type == "cross_reader":
            self.ffn = CrossAttnReaderFFN(
                d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=ffn_activation,
                reader_rank=cross_reader_rank,
            )
            self._ffn_needs_cross_ctx = True
        elif ffn_type == "hessian":
            module_variant: HessianVariant = (
                "h3"
                if hessian_variant in ("h3_nores", "h3_gated", "h3_mix")
                else hessian_variant
            )
            self.ffn = HessianFFN(
                d_model,
                variant=module_variant,
                lambda_geometry=hessian_lambda_geometry,
                tanh2_temperature=hessian_tanh2_temperature,
                tanh2_amplitude=hessian_tanh2_amplitude,
                tanh2_amplitude_scale=hessian_tanh2_amplitude_scale,
                tanh2_peak_centers=hessian_tanh2_peak_centers,
                tanh2_peak_shift=hessian_tanh2_peak_shift,
            )
        elif ffn_type == "none":
            self.ffn = nn.Identity()
        else:
            raise ValueError(f"unknown ffn_type: {ffn_type!r}")

        self.ffn_adapter = ffn_adapter

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        h = self.norm1(tgt)
        sa_out, sa_w = self.self_attn(
            h,
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_mask,
            need_weights=need_weights,
        )
        tgt = tgt + self.dropout1(sa_out)

        h = self.norm2(tgt)
        ca_out, ca_w = self.cross_attn(
            h,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
            need_weights=need_weights,
        )
        tgt = tgt + self.dropout2(ca_out)

        h = self.norm3(tgt)
        if self.ffn_adapter is not None:
            h = self.ffn_adapter(h)
        if self._ffn_needs_cross_ctx:
            ff_out = self.ffn(h, cross_ctx=ca_out)
        else:
            ff_out = self.ffn(h)
        tgt = tgt + self.dropout3(ff_out)

        if need_weights:
            return tgt, sa_w, ca_w
        return tgt
