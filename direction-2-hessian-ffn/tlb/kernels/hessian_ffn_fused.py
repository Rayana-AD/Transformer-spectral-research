"""Fused Hessian FFN paths for MPS/CPU (Triton-style fusion without CUDA Triton).

Reduces kernel launches and HBM traffic by:
- computing global λ from ``x.mean(dim=1)`` without ``expand_as`` materialization
- using ``F.linear`` (no per-forward ``weight.transpose``)
- skipping multwin pools when ``scale_logits`` is global-only (analytic init path)
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from tlb.ffn.hessian_core import (
    _global_seq_mean,
    _symmetric_window_mean,
    _symmetric_window_mean_cumsum,
    _tanh2_invcombo_ladder_second_deriv,
    _tanh2_second_deriv,
)

DecoupleMinLambda = Literal[
    "global_tanh2",
    "proj_oracle",
    "proj_oracle_fast",
    "global_oracle",
    "proj_gelu_tanh2_taylor",
    "proj_oracle_x_native_norm0",
    "proj_oracle_div_native_norm0",
    "proj_oracle_geom_native_norm0",
    "proj_oracle_dot_norm_native",
    "proj_gelu_x_native_over_s",
    "proj_oracle_mod_native_ortho",
    "proj_oracle_mod_native_simple",
    "proj_hessian_gelu_tanh_gate",
    "proj_hessian_native_blend25",
    "proj_hessian_native_blend50",
    "proj_hessian_native_temp2",
    "proj_hessian_softplus_gate",
]

_FACTOR_LAMBDA_GEOMETRIES = frozenset(
    {
        "proj_oracle_x_native_norm0",
        "proj_oracle_div_native_norm0",
        "proj_oracle_geom_native_norm0",
        "proj_oracle_dot_norm_native",
        "proj_gelu_x_native_over_s",
        "proj_oracle_mod_native_ortho",
        "proj_oracle_mod_native_simple",
        "proj_gelu_tanh2_taylor",
        "proj_hessian_gelu_tanh_gate",
        "proj_hessian_native_blend25",
        "proj_hessian_native_blend50",
        "proj_hessian_native_temp2",
        "proj_hessian_softplus_gate",
    }
)


def _lambda_oracle_gelu_fused(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Vectorized GELU(u)/u; no boolean indexing (MPS-friendly fused path)."""
    abs_u = u.abs()
    safe = torch.where(abs_u < eps, u.new_tensor(eps), u)
    ratio = F.gelu(u) / safe
    return torch.where(abs_u < eps, torch.full_like(u, 0.5), ratio)


def _lambda_gate(s_z: torch.Tensor, *, activ: str = "tanh2") -> torch.Tensor:
    """λ(s_z) with broadcast shape (B, 1, M) for gating (B, T, M)."""
    if activ == "invcombo_ladder":
        lam = _tanh2_invcombo_ladder_second_deriv(s_z)
    else:
        lam = _tanh2_second_deriv(s_z)
    if lam.ndim == 2:
        return lam.unsqueeze(1)
    return lam


def _is_global_only_scale(scale_logits: torch.Tensor, threshold: float = 0.99) -> bool:
    with torch.no_grad():
        w = F.softmax(scale_logits, dim=0)
        return float(w[-1].item()) >= threshold


def decouple_min_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_weight: torch.Tensor,
    *,
    out_bias: torch.Tensor | None = None,
    lambda_geometry: DecoupleMinLambda = "global_tanh2",
) -> torch.Tensor:
    """Fused global-decouple-min: U/V decouple; λ geometry selectable."""
    if lambda_geometry == "proj_oracle":
        # Reproducible slow path (T2e @54.2% baseline — do not change).
        from tlb.ffn.mlp_hessian_analytic import lambda_oracle_gelu

        proj_x = F.linear(x, weight, bias)
        lam = lambda_oracle_gelu(proj_x)
    elif lambda_geometry == "proj_oracle_fast":
        proj_x = F.linear(x, weight, bias)
        lam = _lambda_oracle_gelu_fused(proj_x)
    elif lambda_geometry in _FACTOR_LAMBDA_GEOMETRIES:
        from tlb.ffn.mlp_hessian_analytic import lambda_decouple_geometry

        proj_x = F.linear(x, weight, bias)
        lam = lambda_decouple_geometry(proj_x, lambda_geometry)
    elif lambda_geometry == "global_oracle":
        proj_x = F.linear(x, weight, bias)
        s_z = F.linear(x.mean(dim=1), weight, bias)
        lam = _lambda_oracle_gelu_fused(s_z)
        if lam.ndim == 2:
            lam = lam.unsqueeze(1)
    else:
        proj_x = F.linear(x, weight)
        s_z = F.linear(x.mean(dim=1), weight, bias)
        lam = _lambda_gate(s_z)
    out = torch.matmul(proj_x * lam, out_weight)
    if out_bias is not None:
        out = out + out_bias
    return out


def global_light_decouple_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_weight: torch.Tensor,
    hp_logit: torch.Tensor,
    *,
    local_window: int = 4,
) -> torch.Tensor:
    """Fused light decouple: one local window for high-pass + global λ."""
    z_win = _symmetric_window_mean_cumsum(x, local_window)
    gamma = torch.sigmoid(hp_logit)
    proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
    proj_x = F.linear(proj_in, weight)
    s_z = F.linear(x.mean(dim=1), weight, bias)
    lam = _lambda_gate(s_z)
    return torch.matmul(proj_x * lam, out_weight)


def global_ms_tied_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    hp_logit: torch.Tensor,
    *,
    window_size: int = 64,
    z_win: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused tied global_ms with global-only z (no multwin pools)."""
    if z_win is None:
        z_win = _symmetric_window_mean(x, window_size)
    gamma = torch.sigmoid(hp_logit)
    proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
    proj_x = F.linear(proj_in, weight)
    s_z = F.linear(x.mean(dim=1), weight, bias)
    lam = _lambda_gate(s_z)
    return torch.matmul(proj_x * lam, weight)


def global_ms_decouple_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_weight: torch.Tensor,
    hp_logit: torch.Tensor,
    scale_logits: torch.Tensor,
    window_sizes: tuple[int, ...],
    window_size: int,
    *,
    lambda_activ: str = "tanh2",
    global_only_threshold: float = 0.99,
) -> torch.Tensor | None:
    """Fused ms-decouple when z is global-only; else returns None (caller uses legacy)."""
    if not _is_global_only_scale(scale_logits, global_only_threshold):
        return None

    z_win = None
    for w in window_sizes:
        if w == window_size:
            z_win = _symmetric_window_mean(x, w)
            break
    if z_win is None:
        z_win = _symmetric_window_mean(x, window_size)

    gamma = torch.sigmoid(hp_logit)
    proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
    proj_x = F.linear(proj_in, weight)
    s_z = F.linear(x.mean(dim=1), weight, bias)
    lam = _lambda_gate(s_z, activ=lambda_activ)
    return torch.matmul(proj_x * lam, out_weight)


def decouple_min_forward_legacy(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_weight: torch.Tensor,
    *,
    out_bias: torch.Tensor | None = None,
    lambda_geometry: DecoupleMinLambda = "global_tanh2",
) -> torch.Tensor:
    """Reference legacy path (matches old ``_PhiGlobalDecoupleMinCore``)."""
    w_in = weight.transpose(-2, -1)
    if lambda_geometry == "proj_oracle":
        from tlb.ffn.mlp_hessian_analytic import lambda_oracle_gelu

        proj_x = x @ w_in + bias
        lam = lambda_oracle_gelu(proj_x)
    elif lambda_geometry == "proj_oracle_fast":
        proj_x = x @ w_in + bias
        lam = _lambda_oracle_gelu_fused(proj_x)
    elif lambda_geometry in _FACTOR_LAMBDA_GEOMETRIES:
        from tlb.ffn.mlp_hessian_analytic import lambda_decouple_geometry

        proj_x = x @ w_in + bias
        lam = lambda_decouple_geometry(proj_x, lambda_geometry)
    elif lambda_geometry == "global_oracle":
        proj_x = x @ w_in + bias
        z = _global_seq_mean(x)
        s_z = z @ w_in + bias
        lam = _lambda_oracle_gelu_fused(s_z)
    else:
        proj_x = x @ w_in
        z = _global_seq_mean(x)
        s_z = z @ w_in + bias
        lam = _tanh2_second_deriv(s_z)
    out = (lam * proj_x) @ out_weight
    if out_bias is not None:
        out = out + out_bias
    return out
