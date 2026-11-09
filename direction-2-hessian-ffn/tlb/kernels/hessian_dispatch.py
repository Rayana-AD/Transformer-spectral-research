"""Backend dispatch for Hessian FFN (fused MPS path vs legacy eager)."""
from __future__ import annotations

import os
from typing import Literal

import torch

from tlb.kernels.hessian_ffn_fused import (
    decouple_min_forward,
    global_light_decouple_forward,
    global_ms_decouple_forward,
    global_ms_tied_forward,
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

HessianFfnBackend = Literal["legacy", "fused", "auto"]

_ENV_KEY = "TLB_HESSIAN_FFN_BACKEND"


def get_hessian_ffn_backend() -> HessianFfnBackend:
    raw = os.environ.get(_ENV_KEY, "auto").strip().lower()
    if raw in ("legacy", "fused", "auto"):
        return raw  # type: ignore[return-value]
    return "auto"


def resolve_hessian_ffn_backend(device: torch.device | None = None) -> Literal["legacy", "fused"]:
    choice = get_hessian_ffn_backend()
    if choice == "legacy":
        return "legacy"
    if choice == "fused":
        return "fused"
    # auto: prefer fused on MPS/CPU; CUDA may use either (fused still valid)
    if device is not None and device.type == "cuda":
        return "fused"
    return "fused"


def decouple_min(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_weight: torch.Tensor,
    *,
    out_bias: torch.Tensor | None = None,
    lambda_geometry: DecoupleMinLambda = "global_tanh2",
    backend: HessianFfnBackend | None = None,
) -> torch.Tensor:
    use = backend or resolve_hessian_ffn_backend(x.device)
    if use == "fused":
        return decouple_min_forward(
            x,
            weight,
            bias,
            out_weight,
            out_bias=out_bias,
            lambda_geometry=lambda_geometry,
        )
    from tlb.kernels.hessian_ffn_fused import decouple_min_forward_legacy

    return decouple_min_forward_legacy(
        x,
        weight,
        bias,
        out_weight,
        out_bias=out_bias,
        lambda_geometry=lambda_geometry,
    )


def global_light_decouple(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_weight: torch.Tensor,
    hp_logit: torch.Tensor,
    *,
    local_window: int = 4,
    backend: HessianFfnBackend | None = None,
) -> torch.Tensor:
    use = backend or resolve_hessian_ffn_backend(x.device)
    if use == "fused":
        return global_light_decouple_forward(
            x, weight, bias, out_weight, hp_logit, local_window=local_window
        )
    w_in = weight.transpose(-2, -1)
    z = _legacy_global_seq_mean(x)
    z_win = _legacy_symmetric_window_mean_cumsum(x, local_window)
    gamma = torch.sigmoid(hp_logit)
    proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
    proj_x = proj_in @ w_in
    s_z = z @ w_in + bias
    from tlb.ffn.hessian_core import _tanh2_second_deriv

    lam = _tanh2_second_deriv(s_z)
    return (lam * proj_x) @ out_weight


def global_ms_tied(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    hp_logit: torch.Tensor,
    *,
    window_size: int = 64,
    z_win: torch.Tensor | None = None,
    backend: HessianFfnBackend | None = None,
) -> torch.Tensor:
    use = backend or resolve_hessian_ffn_backend(x.device)
    if use == "fused":
        return global_ms_tied_forward(
            x, weight, bias, hp_logit, window_size=window_size, z_win=z_win
        )
    from tlb.ffn.hessian_core import _symmetric_window_mean, _tanh2_second_deriv

    if z_win is None:
        z_win = _symmetric_window_mean(x, window_size)
    gamma = torch.sigmoid(hp_logit)
    proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
    w = weight.transpose(-2, -1)
    proj_x = proj_in @ w
    s_z = _legacy_global_seq_mean(x) @ w + bias
    lam = _tanh2_second_deriv(s_z)
    return (lam * proj_x) @ weight


def global_ms_decouple(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_weight: torch.Tensor,
    hp_logit: torch.Tensor,
    scale_logits: torch.Tensor,
    window_sizes: tuple[int, ...],
    window_size: int,
    *,
    lambda_activ: str = "invcombo_ladder",
    legacy_forward,
    backend: HessianFfnBackend | None = None,
) -> torch.Tensor:
    """Try fused global-only path; fall back to ``legacy_forward()``."""
    use = backend or resolve_hessian_ffn_backend(x.device)
    if use == "fused":
        activ = "invcombo_ladder" if lambda_activ == "invcombo_ladder" else "tanh2"
        out = global_ms_decouple_forward(
            x,
            weight,
            bias,
            out_weight,
            hp_logit,
            scale_logits,
            window_sizes,
            window_size,
            lambda_activ=activ,
        )
        if out is not None:
            return out
    return legacy_forward()


def _legacy_global_seq_mean(x: torch.Tensor) -> torch.Tensor:
    from tlb.ffn.hessian_core import _global_seq_mean

    return _global_seq_mean(x)


def _legacy_symmetric_window_mean_cumsum(x: torch.Tensor, window_size: int) -> torch.Tensor:
    from tlb.ffn.hessian_core import _symmetric_window_mean_cumsum

    return _symmetric_window_mean_cumsum(x, window_size)
