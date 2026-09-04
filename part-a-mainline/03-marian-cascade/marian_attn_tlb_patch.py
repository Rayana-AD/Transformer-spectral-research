"""Monkey-patch HuggingFace ``MarianAttention`` / ``eager_attention_forward`` (``attn_implementation='eager'``).

Install sets ``module._tlb_attn_input = hidden_states`` inside a wrapped ``MarianAttention.forward`` so
``nuc_inter`` can read layer input for ``||·||_*`` ratios. Uninstall restores originals.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from tlb_attention_core import _matrix_nuclear_norm_bh, _matrix_nuclear_norm_bh_keep, layer_hidden_nuclear_mean
from tlb_value_smooth import (
    SpatialMode,
    TemporalMode,
    apply_value_cascade,
)


def _get_marian_mod():
    import transformers.models.marian.modeling_marian as m

    return m


def _ensure_marian_orig_eager() -> None:
    m = _get_marian_mod()
    if getattr(m, "_tlb_marian_true_original_eager", None) is None:
        m._tlb_marian_true_original_eager = m.eager_attention_forward


def uninstall_marian_tlb() -> None:
    m = _get_marian_mod()
    orig = getattr(m, "_tlb_marian_true_original_eager", None)
    if orig is not None:
        m.eager_attention_forward = orig
    ma = getattr(m, "MarianAttention", None)
    if ma is not None:
        orig_fw = getattr(ma, "_tlb_marian_true_forward", None)
        if orig_fw is not None:
            ma.forward = orig_fw  # type: ignore[assignment]
            delattr(ma, "_tlb_marian_true_forward")


def _mha_nuc_mean_qkv(t: torch.Tensor, num_heads: int, eps: float) -> torch.Tensor:
    """Scalar mean ||·||_* over batch+heads for (B, H, T, Dh)."""
    b, h, tlen, d = t.shape
    assert h == num_heads
    n = _matrix_nuclear_norm_bh(t)
    return n.mean().clamp_min(eps)


def _make_marian_kv_inner_eager(eps: float) -> Callable:
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)
        kv = key * value
        nuc_v = _matrix_nuclear_norm_bh_keep(value)
        nuc_kv = _matrix_nuclear_norm_bh_keep(kv)
        c_l = nuc_v / (nuc_kv + eps)
        value_mix = c_l * kv

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_mix)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_nuc_inter_eager(eps: float) -> Callable:
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)
        input_hidden = kwargs.get("hidden_states")
        if input_hidden is None:
            input_hidden = getattr(module, "_tlb_attn_input", None)
        num_heads = int(module.num_heads)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        if input_hidden is not None:
            nuc_in = layer_hidden_nuclear_mean(input_hidden, num_heads, eps)
            nuc_o = layer_hidden_nuclear_mean(attn_output, num_heads, eps)
            attn_output = (nuc_in / (nuc_o + eps)) * attn_output

        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_nuc_inter_q_eager(eps: float) -> Callable:
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)
        input_hidden = kwargs.get("hidden_states") or getattr(module, "_tlb_attn_input", None)
        num_heads = int(module.num_heads)
        if input_hidden is not None:
            nuc_in = layer_hidden_nuclear_mean(input_hidden, num_heads, eps)
            nuc_q = _mha_nuc_mean_qkv(query, num_heads, eps)
            query = query * (nuc_in / nuc_q)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_v_jitter_eager(_eps: float, jitter_std: float) -> Callable:
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)
        eff = jitter_std if jitter_std > 0 else 1e-3
        v_mix = value + eff * torch.randn_like(value) if module.training else value

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, v_mix)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_v_smooth_eager(_eps: float, smooth_alpha: float = 0.1) -> Callable:
    """V variance reduction: smooth V towards sequence mean (temporal, pre-attention)."""
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)
        # value shape: (batch, num_heads, seq_len, head_dim)
        alpha = smooth_alpha
        v_mean = value.mean(dim=-2, keepdim=True)  # mean across seq_len
        v_smooth = (1 - alpha) * value + alpha * v_mean
        v_mix = v_smooth if module.training else value

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, v_mix)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_v_smooth_mean_eager(_eps: float, smooth_alpha: float = 0.1) -> Callable:
    """V smooth towards sequence mean (explicit naming, same as v_smooth)."""
    # This is functionally identical to v_smooth but with explicit naming for clarity
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)
        # value shape: (batch, num_heads, seq_len, head_dim)
        alpha = smooth_alpha
        v_mean = value.mean(dim=-2, keepdim=True)  # mean across seq_len
        v_smooth = (1 - alpha) * value + alpha * v_mean
        v_mix = v_smooth if module.training else value

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, v_mix)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_v_smooth_roll_eager(_eps: float, smooth_alpha: float = 0.1) -> Callable:
    """V smooth by rolling/shifting sequence (temporal neighbor smoothing).
    
    v_smooth = (1-alpha) * v + alpha * v.roll(1, -2)
    This mixes each position with its neighbor, creating local temporal smoothness.
    """
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)
        # value shape: (batch, num_heads, seq_len, head_dim)
        alpha = smooth_alpha
        # Roll along seq_len dimension (dim=-2), shift by 1
        v_rolled = value.roll(shifts=1, dims=-2)
        v_smooth = (1 - alpha) * value + alpha * v_rolled
        v_mix = v_smooth if module.training else value

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, v_mix)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_v_smooth_hybrid_eager(_eps: float, alpha_temporal: float = 0.1, alpha_global: float = 0.1) -> Callable:
    """V hybrid smoothing: combines temporal (roll) and global (mean) smoothing.
    
    v_final = (1 - alpha_temporal - alpha_global) * v 
            + alpha_temporal * v.roll(1, -2) 
            + alpha_global * v.mean(dim=-2, keepdim=True)
    
    Both alpha_temporal and alpha_global control the strength of each component.
    Constraint: alpha_temporal + alpha_global < 1.0
    """
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)
        
        # value shape: (batch, num_heads, seq_len, head_dim)
        alpha_t = alpha_temporal
        alpha_g = alpha_global
        
        # Ensure constraint
        total_alpha = alpha_t + alpha_g
        if total_alpha >= 1.0:
            alpha_t = alpha_t / total_alpha * 0.99
            alpha_g = alpha_g / total_alpha * 0.99
        
        v_rolled = value.roll(shifts=1, dims=-2)  # temporal
        v_global = value.mean(dim=-2, keepdim=True)  # global
        
        # Hybrid: weighted combination
        v_smooth = (1 - alpha_t - alpha_g) * value + alpha_t * v_rolled + alpha_g * v_global
        v_mix = v_smooth if module.training else value

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, v_mix)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_v_smooth_cascade_eager(
    _eps: float,
    alpha_spatial: float = 0.1,
    alpha_temporal: float = 0.1,
    *,
    temporal_mode: TemporalMode = "circular_roll",
    spatial_mode: SpatialMode = "spatial_global",
) -> Callable:
    """V cascade smoothing: spatial then temporal (see ``tlb_value_smooth``)."""

    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)

        _, _, v_smooth = apply_value_cascade(
            value,
            alpha_spatial,
            alpha_temporal,
            order="spatial_first",
            temporal_mode=temporal_mode,
            spatial_mode=spatial_mode,
        )
        v_mix = v_smooth if module.training else value

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, v_mix)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_v_smooth_cascade_rev_eager(
    _eps: float,
    alpha_spatial: float = 0.1,
    alpha_temporal: float = 0.1,
    *,
    temporal_mode: TemporalMode = "circular_roll",
    spatial_mode: SpatialMode = "spatial_global",
) -> Callable:
    """V cascade smoothing: temporal then spatial (reverse order)."""

    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)

        _, _, v_smooth = apply_value_cascade(
            value,
            alpha_spatial,
            alpha_temporal,
            order="temporal_first",
            temporal_mode=temporal_mode,
            spatial_mode=spatial_mode,
        )
        v_mix = v_smooth if module.training else value

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, v_mix)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_v_smooth_post_eager(_eps: float, smooth_alpha: float = 0.1) -> Callable:
    """V smooth post-attention: smooth the attention output (temporal smoothing after attn)."""
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Post-attention smoothing along sequence dimension
        if module.training:
            alpha = smooth_alpha
            # attn_output shape: (batch, seq_len, num_heads, head_dim)
            out_mean = attn_output.mean(dim=1, keepdim=True)  # mean across seq_len
            attn_output = (1 - alpha) * attn_output + alpha * out_mean

        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_v_smooth_feature_eager(_eps: float, smooth_alpha: float = 0.1) -> Callable:
    """V smooth along feature dimension (head_dim) instead of temporal."""
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)

        # Feature-wise smoothing: along head_dim (-1)
        alpha = smooth_alpha
        v_mean = value.mean(dim=-1, keepdim=True)  # mean across head_dim
        v_smooth = (1 - alpha) * value + alpha * v_mean
        v_mix = v_smooth if module.training else value

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, v_mix)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_v_smooth_both_eager(_eps: float, smooth_alpha: float = 0.1) -> Callable:
    """V smooth both pre and post attention."""
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)

        # Pre-attention smoothing
        alpha = smooth_alpha
        v_mean = value.mean(dim=-2, keepdim=True)
        v_smooth = (1 - alpha) * value + alpha * v_mean
        v_mix = v_smooth if module.training else value

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, v_mix)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Post-attention smoothing
        if module.training:
            out_mean = attn_output.mean(dim=1, keepdim=True)
            attn_output = (1 - alpha) * attn_output + alpha * out_mean

        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_nuc_inter_bias_eager(eps: float, bias: float = 0.9) -> Callable:
    """Nuc_inter with intentional bias (e.g., 0.9) instead of forcing ratio to 1."""
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)
        input_hidden = kwargs.get("hidden_states")
        if input_hidden is None:
            input_hidden = getattr(module, "_tlb_attn_input", None)
        num_heads = int(module.num_heads)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        if input_hidden is not None:
            # input_hidden: (B, T, D) where D = num_heads * head_dim
            # attn_output: (B, T, H, d) - need to reshape to (B, T, D)
            b, t, h, d = attn_output.shape
            attn_output_3d = attn_output.reshape(b, t, h * d)
            nuc_in = layer_hidden_nuclear_mean(input_hidden, num_heads, eps)
            nuc_o = layer_hidden_nuclear_mean(attn_output_3d, num_heads, eps)
            # Apply ratio AND the intentional bias (e.g., 0.9)
            attn_output = bias * (nuc_in / (nuc_o + eps)) * attn_output

        return attn_output, attn_weights

    return eager_attention_forward


def _make_marian_nuc_inter_q_learned_eager(eps: float) -> Callable:
    """Learnable per-layer Q scaling: query *= learned_scale * (nuc_in / nuc_q).
    
    Instead of forcing ratio to 1 or using fixed bias, learn a per-layer scalar
    that multiplies the nuclear norm ratio. Initialized to 1.0 (vanilla behavior).
    The learned parameter is stored on each MarianAttention module as _tlb_q_scale.
    """
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)
        input_hidden = kwargs.get("hidden_states")
        if input_hidden is None:
            input_hidden = getattr(module, "_tlb_attn_input", None)
        num_heads = int(module.num_heads)

        # Initialize learnable scale if not present (init to 1.0 = neutral)
        if not hasattr(module, "_tlb_q_scale"):
            module._tlb_q_scale = nn.Parameter(torch.tensor(1.0))

        # Compute nuclear norms
        if input_hidden is not None:
            nuc_in = layer_hidden_nuclear_mean(input_hidden, num_heads, eps)
            nuc_q = _mha_nuc_mean_qkv(query, num_heads, eps)
            # Apply learnable scale times the ratio
            scale_factor = module._tlb_q_scale * (nuc_in / nuc_q)
            query = query * scale_factor

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _wrap_marian_attention_forward() -> None:
    """Store hidden_states on module for nuc_inter / nuc_inter_q (Marian does not pass it to eager)."""
    m = _get_marian_mod()
    Ma = m.MarianAttention
    if getattr(Ma, "_tlb_marian_true_forward", None) is not None:
        return

    Ma._tlb_marian_true_forward = Ma.forward  # type: ignore[attr-defined]

    def forward_with_hidden(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        self._tlb_attn_input = hidden_states
        out = Ma._tlb_marian_true_forward(  # type: ignore[union-attr]
            self, hidden_states, key_value_states, *args, **kwargs
        )
        if hasattr(self, "_tlb_attn_input"):
            del self._tlb_attn_input
        return out

    Ma.forward = forward_with_hidden  # type: ignore[assignment]


def apply_marian_attention(
    mode: str,
    *,
    tlb_eps: float = 1e-8,
    jitter_std: float = 0.0,
    smooth_alpha: float = 0.1,
    alpha_temporal: float = 0.1,
    alpha_global: float = 0.1,
    alpha_spatial: float = 0.1,
    nuc_bias: float = 0.9,
    temporal_mode: TemporalMode = "circular_roll",
    spatial_mode: SpatialMode = "spatial_global",
) -> None:
    """``vanilla`` = restore HF; else replace ``eager_attention_forward`` (model must use eager)."""
    uninstall_marian_tlb()
    m = _get_marian_mod()
    _ensure_marian_orig_eager()
    if mode == "vanilla":
        m.eager_attention_forward = m._tlb_marian_true_original_eager
        return
    _wrap_marian_attention_forward()
    if mode == "kv_inner" or mode == "line_c":
        m.eager_attention_forward = _make_marian_kv_inner_eager(tlb_eps)
    elif mode == "nuc_inter":
        m.eager_attention_forward = _make_marian_nuc_inter_eager(tlb_eps)
    elif mode == "nuc_inter_q":
        m.eager_attention_forward = _make_marian_nuc_inter_q_eager(tlb_eps)
    elif mode == "v_jitter":
        m.eager_attention_forward = _make_marian_v_jitter_eager(tlb_eps, jitter_std)
    elif mode == "v_smooth":
        m.eager_attention_forward = _make_marian_v_smooth_eager(tlb_eps, smooth_alpha)
    elif mode == "v_smooth_mean":
        m.eager_attention_forward = _make_marian_v_smooth_mean_eager(tlb_eps, smooth_alpha)
    elif mode == "v_smooth_roll":
        m.eager_attention_forward = _make_marian_v_smooth_roll_eager(tlb_eps, smooth_alpha)
    elif mode == "v_smooth_hybrid":
        m.eager_attention_forward = _make_marian_v_smooth_hybrid_eager(tlb_eps, alpha_temporal, alpha_global)
    elif mode == "v_smooth_cascade":
        m.eager_attention_forward = _make_marian_v_smooth_cascade_eager(
            tlb_eps,
            alpha_spatial,
            alpha_temporal,
            temporal_mode=temporal_mode,
            spatial_mode=spatial_mode,
        )
    elif mode == "v_smooth_cascade_rev":
        m.eager_attention_forward = _make_marian_v_smooth_cascade_rev_eager(
            tlb_eps,
            alpha_spatial,
            alpha_temporal,
            temporal_mode=temporal_mode,
            spatial_mode=spatial_mode,
        )
    elif mode == "v_smooth_post":
        m.eager_attention_forward = _make_marian_v_smooth_post_eager(tlb_eps, smooth_alpha)
    elif mode == "v_smooth_feature":
        m.eager_attention_forward = _make_marian_v_smooth_feature_eager(tlb_eps, smooth_alpha)
    elif mode == "v_smooth_both":
        m.eager_attention_forward = _make_marian_v_smooth_both_eager(tlb_eps, smooth_alpha)
    elif mode == "nuc_inter_bias":
        m.eager_attention_forward = _make_marian_nuc_inter_bias_eager(tlb_eps, nuc_bias)
    elif mode == "nuc_inter_q_learned":
        m.eager_attention_forward = _make_marian_nuc_inter_q_learned_eager(tlb_eps)
    else:
        raise ValueError(f"unknown marian attention mode: {mode!r}")
