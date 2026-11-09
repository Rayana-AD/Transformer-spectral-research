"""Transformer encoder layer with configurable attention and FFN."""
from __future__ import annotations

import torch
import torch.nn as nn

from tlb.attention.angular import (
    AttentionType,
    NormalizationMode,
    compute_attention,
)
from tlb.ffn.hessian_core import DecoupleMinLambda, HessianFFN, HessianVariant
from tlb.ffn.standard import StandardFFN


class MultiheadSelfAttention(nn.Module):
    """Self-attention with vanilla / cosine / angular scoring."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        attention_type: AttentionType = "vanilla",
        normalize_mode: NormalizationMode = "row",
        alpha: float = 2.0,
        cottention_sigma: float = 0.5,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_type = attention_type
        self.normalize_mode = normalize_mode
        self.alpha = alpha
        self.cottention_sigma = cottention_sigma
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        out, attn = compute_attention(
            q,
            k,
            v,
            attention_type=self.attention_type,
            normalize_mode=self.normalize_mode,
            alpha=self.alpha,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            cottention_sigma=self.cottention_sigma,
            dropout=self.dropout_p,
            training=self.training,
        )
        out = out.transpose(1, 2).reshape(b, t, self.embed_dim)
        out = self.out_proj(out)
        return out, attn if need_weights else None


class TransformerEncoderLayer(nn.Module):
    """Post-LN encoder: norm → attn → residual → norm → FFN/Hessian → residual."""

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
        hessian_num_units: int | None = None,
        hessian_lambda_geometry: DecoupleMinLambda | None = None,
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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn_type = ffn_type
        self.ffn_no_residual = ffn_type == "hessian" and hessian_variant == "h3_nores"
        self.ffn_gated_residual = ffn_type == "hessian" and hessian_variant == "h3_gated"
        self.ffn_mix_residual = ffn_type == "hessian" and hessian_variant == "h3_mix"
        self.ffn_residual_gate = (
            nn.Parameter(torch.tensor(1.0)) if self.ffn_gated_residual else None
        )
        self.ffn_src_scale = None
        self.ffn_out_scale = (
            nn.Parameter(torch.ones(d_model)) if self.ffn_mix_residual else None
        )

        if ffn_type == "standard":
            self.ffn: nn.Module = StandardFFN(
                d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=ffn_activation,
            )
        elif ffn_type == "hessian":
            module_variant: HessianVariant = (
                "h3"
                if hessian_variant in ("h3_nores", "h3_gated", "h3_mix")
                else hessian_variant
            )
            self.ffn = HessianFFN(
                d_model,
                variant=module_variant,
                num_units=hessian_num_units,
                lambda_geometry=hessian_lambda_geometry,
            )
        elif ffn_type == "none":
            self.ffn = nn.Identity()
        else:
            raise ValueError(f"unknown ffn_type: {ffn_type!r}")

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(src)
        attn_out, w = self.self_attn(
            h,
            key_padding_mask=src_key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )
        src = src + self.dropout1(attn_out)
        h2 = self.norm2(src)
        ff_out = self.ffn(h2)
        if self.ffn_no_residual:
            src = self.dropout2(ff_out)
        elif self.ffn_gated_residual:
            assert self.ffn_residual_gate is not None
            src = src + self.ffn_residual_gate * self.dropout2(ff_out)
        elif self.ffn_mix_residual:
            assert self.ffn_out_scale is not None
            src = src + self.ffn_out_scale * self.dropout2(ff_out)
        else:
            src = src + self.dropout2(ff_out)
        if need_weights:
            return src, w
        return src
