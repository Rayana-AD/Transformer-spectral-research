"""Analytic MLP → Hessian FFN translation and arithmetic probes.

Maps standard ``Linear → GELU → Linear`` weights into decoupled Hessian units:

    MLP:  y = W2 @ GELU(W1 x + b1) + b2
    Hess: y ≈ sum_j  λ_j(s_j) * (w_j^T x + b_j) * v_j

where ``w_j`` comes from ``W1[j,:]``, ``v_j`` from ``W2[:,j]``, and
``λ_j(s) * u`` is calibrated to match ``GELU(u)`` on a fixed grid.

For cores with ``out_weight`` (decouple / global_decouple_min), ``W2`` maps
directly to ``out_weight`` rows.  For tied cores (``h4_phi768``), a joint
``w_j`` is used for both legs and ``readout`` absorbs the output mixing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tlb.ffn.hessian_core import (
    HessianFFN,
    _global_seq_mean,
    _symmetric_window_mean,
    _tanh2_first_deriv,
    _tanh2_second_deriv,
)
from tlb.ffn.standard import StandardFFN

LambdaMode = Literal["raw", "calibrated", "oracle_gelu"]


@dataclass
class JointUnit:
    """One coupled Hessian unit derived from MLP row/col pair."""

    w_in: torch.Tensor  # (D,)
    w_out: torch.Tensor  # (D,)
    b: float
    in_scale: float
    out_scale: float
    lam_scale: float
    lam_shift: float
    alignment: float


TieMode = Literal["w1", "out", "joint"]
MsDegenerateMode = Literal["decouple_min", "global_ms"]


@dataclass
class AnalyticTranslation:
    """Translated parameters for one FFN block."""

    variant: str
    weight: torch.Tensor  # (M, D)
    bias: torch.Tensor  # (M,)
    out_weight: torch.Tensor | None  # (M, D) when decoupled
    out_bias: torch.Tensor | None  # (D,) MLP b2
    readout_weight: torch.Tensor | None
    readout_bias: torch.Tensor | None
    hp_logit: float | None
    scale_logits: torch.Tensor | None  # multwin + global blend (ms variants)
    window_sizes: tuple[int, ...] | None
    units: list[JointUnit]
    lambda_mode: LambdaMode


def gelu_scalar(u: torch.Tensor) -> torch.Tensor:
    return F.gelu(u)


def lambda_raw(u: torch.Tensor) -> torch.Tensor:
    return _tanh2_second_deriv(u)


def lambda_oracle_gelu(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Oracle gate: λ(u) = GELU(u)/u (limit 0.5 at 0)."""
    small = u.abs() < eps
    safe = torch.where(small, torch.ones_like(u), u)
    ratio = gelu_scalar(safe) / safe
    return torch.where(small, torch.full_like(u, 0.5), ratio)


_PSI2_AT_ZERO = 2.0
_GELU_S_RATIO_AT_ZERO = 0.5
_DOT_NORM_SCALE_AT_ZERO = _GELU_S_RATIO_AT_ZERO / (
    _GELU_S_RATIO_AT_ZERO / math.sqrt(_GELU_S_RATIO_AT_ZERO**2 + 1.0)
)


def native_psi2_norm0(s: torch.Tensor) -> torch.Tensor:
    """Native gate ψ''(s) normalized to 1 at s=0."""
    return _tanh2_second_deriv(s) / _PSI2_AT_ZERO


def _silu_second_deriv(s: torch.Tensor) -> torch.Tensor:
    """``d²/ds² [s σ(s)]`` for Smish/SiLU (DreamTour ``activation_extras.Smish``)."""
    sig = torch.sigmoid(s)
    sig1 = sig * (1.0 - sig)
    return sig1 * (2.0 + s * (1.0 - 2.0 * sig))


def _lightdepth_second_deriv(s: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """``ψ(s)=s σ(s/t)``; ``t>1`` spreads curvature (DreamTour ``LightDepthAct``)."""
    t = max(float(temperature), 1e-6)
    return _silu_second_deriv(s / t) / t


def hessian_factor_norm0(s: torch.Tensor, kind: str) -> torch.Tensor:
    """Hessian-side factor ``f`` with ``f(0)=1``. Φ stays the other factor.

    ``kind``:
      * ``tanh2`` — paper ψ=tanh² (peaked, signed)
      * ``silu`` — Smish/SiLU second derivative
      * ``lightdepth2`` / ``lightdepth4`` — temperature-softened SiLU family
    """
    if kind == "tanh2":
        return native_psi2_norm0(s)
    if kind == "silu":
        return _silu_second_deriv(s) / 0.5
    if kind == "lightdepth2":
        psi0 = 0.5 / 2.0
        return _lightdepth_second_deriv(s, temperature=2.0) / psi0
    if kind == "lightdepth4":
        psi0 = 0.5 / 4.0
        return _lightdepth_second_deriv(s, temperature=4.0) / psi0
    raise ValueError(f"unknown hessian factor kind: {kind!r}")


def lambda_two_factor(
    s: torch.Tensor,
    *,
    kind: str = "tanh2",
    mix: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``λ = Φ(s) · (1 + mix · (f(s) − 1))``.

    ``mix=0`` is exact ``Φ=GELU/s`` (correct step-0 swap). Only ``f`` changes.
    """
    phi = lambda_oracle_gelu(s, eps=eps)
    if float(mix) == 0.0:
        return phi
    factor = hessian_factor_norm0(s, kind)
    return phi * (1.0 + float(mix) * (factor - 1.0))


# --- Paper Hessian soft-λ family (L1 / L3 / L4) — design before train ---
_GELU_TANH_INNER_COEFF = math.sqrt(2.0 / math.pi)


def lambda_hessian_gelu_tanh_gate(s: torch.Tensor) -> torch.Tensor:
    """L1: GELU-native tanh gate ``λ ≈ ½(1+tanh g(s))`` (eq.5 gate on content ``s``).

    Same swap anchor as ``proj_oracle_fast`` at ``s=0`` but explicit GELU–tanh
    narrative; avoids ratio ``GELU/s`` division.
    """
    inner = _GELU_TANH_INNER_COEFF * (s + 0.044715 * s.pow(3))
    return 0.5 * (1.0 + torch.tanh(inner))


def lambda_hessian_native_blend(
    s: torch.Tensor,
    *,
    beta: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L4: ``λ = Φ(s) · ((1-β) + β·ψ''/ψ''(0))`` — oracle × fixed native mix.

    ``β=0`` → oracle; ``β>0`` softens toward native curvature while ``λ(0)=Φ(0)=0.5``.
    """
    phi = lambda_oracle_gelu(s, eps=eps)
    mix = (1.0 - float(beta)) + float(beta) * native_psi2_norm0(s)
    return phi * mix


def lambda_hessian_native_temp(
    s: torch.Tensor,
    *,
    tau: float = 2.0,
) -> torch.Tensor:
    """Temperature-softened native ``ψ''(s/τ)``, scaled so ``λ(0)=GELU'(0)=0.5``."""
    tau = max(float(tau), 1e-6)
    psi2 = _tanh2_second_deriv(s / tau)
    return psi2 * (_GELU_S_RATIO_AT_ZERO / _PSI2_AT_ZERO)


def lambda_hessian_softplus_gate(s: torch.Tensor) -> torch.Tensor:
    """``ψ=softplus`` gate: ``λ ∝ softplus''(s)``, non-negative, smoother than tanh²''."""
    from tlb.ffn.hessian_core import _softplus_second_deriv

    sp2 = _softplus_second_deriv(s)
    sp2_0 = 0.25  # sigmoid(0)*(1-sigmoid(0))
    return sp2 * (_GELU_S_RATIO_AT_ZERO / sp2_0)


SOFT_LAMBDA_GEOMETRY_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "proj_hessian_gelu_tanh_gate": lambda_hessian_gelu_tanh_gate,
    "proj_hessian_native_blend25": lambda s: lambda_hessian_native_blend(s, beta=0.25),
    "proj_hessian_native_blend50": lambda s: lambda_hessian_native_blend(s, beta=0.50),
    "proj_hessian_native_temp2": lambda s: lambda_hessian_native_temp(s, tau=2.0),
    "proj_hessian_softplus_gate": lambda_hessian_softplus_gate,
}


def lambda_oracle_x_native_norm0(s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """λ = (GELU/s) · (ψ''/ψ''(0)); multiplicative, both factors anchored at s=0."""
    return lambda_oracle_gelu(s, eps=eps) * native_psi2_norm0(s)


def lambda_oracle_div_native_norm0(s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """λ = (GELU/s) / (ψ''/ψ''(0)); divides by native curvature (singular at ψ''=0)."""
    oracle = lambda_oracle_gelu(s, eps=eps)
    native = native_psi2_norm0(s)
    safe = torch.where(native.abs() < eps, native.new_tensor(eps), native)
    return oracle / safe


def lambda_oracle_geom_native_norm0(s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """λ = sign(oracle) · sqrt(|oracle · (ψ''/ψ''(0))|); geometric mean of the two gates."""
    oracle = lambda_oracle_gelu(s, eps=eps)
    native = native_psi2_norm0(s)
    return torch.sign(oracle) * torch.sqrt((oracle * native).abs().clamp(min=0.0))


def lambda_oracle_dot_norm_native(s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """λ = (oracle · native/2) / hypot(oracle, native/2), scaled so λ(0)=GELU'(0)."""
    oracle = lambda_oracle_gelu(s, eps=eps)
    native = native_psi2_norm0(s)
    numer = oracle * native
    denom = torch.sqrt(oracle * oracle + native * native).clamp(min=eps)
    return numer / denom * _DOT_NORM_SCALE_AT_ZERO


def lambda_gelu_x_native_over_s(s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """λ = GELU(s) · (ψ''/ψ''(0)) / s; same unit map as the product gate."""
    abs_s = s.abs()
    num = gelu_scalar(s) * native_psi2_norm0(s)
    safe = torch.where(abs_s < eps, s.new_tensor(1.0), s)
    ratio = num / safe
    return torch.where(
        abs_s < eps,
        torch.full_like(s, _GELU_S_RATIO_AT_ZERO),
        ratio,
    )


_NATIVE_DELTA_ORTHO_SCALE_CACHE: dict[float, float] = {}
_NATIVE_DELTA_FADE_VAR = 0.25


def _factor_temperature(temperature: float) -> float:
    return max(float(temperature), 1e-6)


def native_psi2_temp_norm0(s: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """``ψ''(s/τ) / ψ''(0)``. ``f(0)=1`` for every ``τ``; ``τ=1`` is native ψ''."""
    tau = _factor_temperature(temperature)
    return _tanh2_second_deriv(s / tau) / _PSI2_AT_ZERO


def _native_delta_fade(s: torch.Tensor) -> torch.Tensor:
    """0 at ``s=0``, →1 for ``|s|≫0.5``; keeps ``m(0)=1`` when ``δ(0)=0``."""
    return 1.0 - torch.exp(-(s * s) / _NATIVE_DELTA_FADE_VAR)


def native_delta_ortho_scale(
    *,
    grid: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> float:
    """Scalar ``c`` in ``δ=g-c·fade(s)`` with ``⟨δ,1⟩_{Φ²s²}=0`` and ``δ(0)=0``."""
    tau = _factor_temperature(temperature)
    if grid is None and tau in _NATIVE_DELTA_ORTHO_SCALE_CACHE:
        return _NATIVE_DELTA_ORTHO_SCALE_CACHE[tau]
    if grid is None:
        grid = torch.linspace(-6.0, 6.0, 2401)
    s = grid
    phi = lambda_oracle_gelu(s)
    raw = native_psi2_temp_norm0(s, tau) - 1.0
    fade = _native_delta_fade(s)
    weight = (phi * phi) * (s * s)
    num = float((weight * raw).sum().item())
    den = float((weight * fade).sum().clamp(min=1e-12).item())
    scale = num / den
    if grid.numel() == 2401:
        _NATIVE_DELTA_ORTHO_SCALE_CACHE[tau] = scale
    return scale


def native_delta_ortho_center(
    *,
    grid: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> float:
    """Alias: scale ``c`` used by the faded native modulation."""
    return native_delta_ortho_scale(grid=grid, temperature=temperature)


def native_delta_ortho(s: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Orthogonalized native modulation ``δ`` with ``δ(0)=0`` and ``⟨δ,1⟩_{Φ²s²}≈0``.

    ``temperature=1`` is paper ψ''. Higher ``τ`` widens ``ψ''(s/τ)`` on the Hessian leg only.
    """
    raw = native_psi2_temp_norm0(s, temperature) - 1.0
    return raw - native_delta_ortho_scale(temperature=temperature) * _native_delta_fade(s)


def lambda_oracle_mod_native_ortho(
    s: torch.Tensor,
    *,
    mix: float | torch.Tensor = 0.0,
    gate: torch.Tensor | None = None,
    eps: float = 1e-6,
    temperature: float = 1.0,
) -> torch.Tensor:
    """λ = Φ(s) · (1 + mix·δ(g)); ``mix=0`` is exact oracle (step-0 swap).

    Oracle anchors on content ``s`` (U leg); native ``δ`` acts on gate coordinate
    ``g`` (A leg, default ``g=s``).  ``δ`` is centered so ``⟨Φ·δ·s, Φ·s⟩_{Φ²s²}≈0``.
    ``temperature`` softens only ``δ`` (Hessian factor). It does not move ``Φ``.
    """
    oracle = lambda_oracle_gelu(s, eps=eps)
    g_coord = s if gate is None else gate
    if isinstance(mix, torch.Tensor):
        if mix.numel() == 1 and float(mix.detach().cpu().item()) == 0.0:
            return oracle
        return oracle * (1.0 + mix * native_delta_ortho(g_coord, temperature))
    if float(mix) == 0.0:
        return oracle
    return oracle * (1.0 + float(mix) * native_delta_ortho(g_coord, temperature))


def native_delta_ortho_unit_inner(*, grid: torch.Tensor | None = None) -> float:
    """Discrete ``⟨Φ·δ·s, Φ·s⟩ = Σ Φ² s² δ`` on ``grid`` (should be ~0)."""
    if grid is None:
        grid = torch.linspace(-6.0, 6.0, 2401)
    s = grid
    phi = lambda_oracle_gelu(s)
    delta = native_delta_ortho(s)
    return float((phi * phi * s * s * delta).sum().item())


def native_modulation_g(s: torch.Tensor) -> torch.Tensor:
    """Simple native factor ``g(s)=ψ''(s)/2-1`` with ``g(0)=0``."""
    return native_psi2_norm0(s) - 1.0


def lambda_oracle_mod_native_simple(
    s: torch.Tensor,
    *,
    mix: float | torch.Tensor = 0.0,
    gate: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """λ = Φ(s) · (1 + mix·g_native(g)); oracle on U-leg ``s``, native on A-leg ``g``."""
    oracle = lambda_oracle_gelu(s, eps=eps)
    g_coord = s if gate is None else gate
    if isinstance(mix, torch.Tensor):
        if mix.numel() == 1 and float(mix.detach().cpu().item()) == 0.0:
            return oracle
        return oracle * (1.0 + mix * native_modulation_g(g_coord))
    if float(mix) == 0.0:
        return oracle
    return oracle * (1.0 + float(mix) * native_modulation_g(g_coord))


_GELU_TANH2_TAYLOR_ORDER_DEFAULT = 3
_GELU_TANH2_TAYLOR_AMPLITUDE_CACHE: dict[int, float] = {}
# ``F.gelu`` derivatives at 0 (float64 autograd probe, cached once).
_GELU_DERIVATIVES_AT_ZERO: tuple[float, ...] = (
    0.5,
    0.7978845608028654,
    0.0,
    -1.5957691216057308,
)


def _gelu_nth_derivative_at_zero(n: int) -> float:
    if 1 <= n <= len(_GELU_DERIVATIVES_AT_ZERO):
        return _GELU_DERIVATIVES_AT_ZERO[n - 1]
    return 0.0


def gelu_taylor_coeffs(*, order: int = _GELU_TANH2_TAYLOR_ORDER_DEFAULT) -> tuple[float, ...]:
    """``c_n = GELU^(n)(0)/n!`` for ``ψ(t)=Σ c_n t^n``, ``t=tanh²(s)``."""
    return tuple(
        _gelu_nth_derivative_at_zero(n) / math.factorial(n) for n in range(1, order + 1)
    )


def _gelu_tanh2_taylor_psi_second_at_zero(*, order: int) -> float:
    s = torch.zeros(1)
    lam = torch.zeros((), dtype=s.dtype)
    for n, coeff in enumerate(gelu_taylor_coeffs(order=order), start=1):
        lam = lam + coeff * _tanh2_pow_n_second_deriv(s, n)
    return float(lam.item())


def _tanh2_pow_n_second_deriv(s: torch.Tensor, n: int) -> torch.Tensor:
    """``d²/ds² tanh²(s)^n``."""
    if n == 1:
        return _tanh2_second_deriv(s)
    t = torch.tanh(s).pow(2)
    tp = _tanh2_first_deriv(s)
    tpp = _tanh2_second_deriv(s)
    return n * (n - 1) * t.pow(n - 2) * tp * tp + n * t.pow(n - 1) * tpp


def lambda_gelu_tanh2_taylor(
    s: torch.Tensor,
    *,
    order: int = _GELU_TANH2_TAYLOR_ORDER_DEFAULT,
    amplitude: float | None = None,
) -> torch.Tensor:
    """Native Hessian gate: ``λ = ψ''(s)`` with ``ψ(s)=GELU(tanh²(s))`` Taylor-truncated.

    Fixed coefficients from ``F.gelu`` series at 0; single scalar amplitude fits
    ``λ(s)·s ≈ GELU(s)`` on ``[-4,4]`` (not learnable).
    """
    if amplitude is None:
        amplitude = default_gelu_tanh2_taylor_amplitude(order=order)
    lam = s.new_zeros(s.shape)
    for n, coeff in enumerate(gelu_taylor_coeffs(order=order), start=1):
        lam = lam + coeff * _tanh2_pow_n_second_deriv(s, n)
    return lam * amplitude


def default_gelu_tanh2_taylor_amplitude(
    *,
    order: int = _GELU_TANH2_TAYLOR_ORDER_DEFAULT,
    grid: torch.Tensor | None = None,
) -> float:
    """Scale ``λ`` so ``λ(0)=GELU'(0)=0.5`` (linear GELU match at the origin).

    Native ``ψ''`` from ``GELU(tanh²(s))`` Taylor differs from ``GELU(s)/s`` globally;
    this fixed scalar avoids learnable calib knobs while anchoring swap slope at 0.
    """
    del grid  # kept for API symmetry with grid-fit helpers
    if order in _GELU_TANH2_TAYLOR_AMPLITUDE_CACHE:
        return _GELU_TANH2_TAYLOR_AMPLITUDE_CACHE[order]
    psi2_0 = _gelu_tanh2_taylor_psi_second_at_zero(order=order)
    amp = 0.5 / psi2_0 if abs(psi2_0) > 1e-8 else 0.5
    _GELU_TANH2_TAYLOR_AMPLITUDE_CACHE[order] = amp
    return amp


def gelu_tanh2_taylor_gelu_match_rmse(
    *,
    order: int = _GELU_TANH2_TAYLOR_ORDER_DEFAULT,
    grid: torch.Tensor | None = None,
) -> float:
    """RMSE of ``λ(s)·s`` vs ``GELU(s)`` after amplitude fit."""
    if grid is None:
        grid = torch.linspace(-4.0, 4.0, 801)
    lam = lambda_gelu_tanh2_taylor(grid, order=order)
    pred = lam * grid
    target = gelu_scalar(grid)
    return float(((pred - target) ** 2).mean().sqrt().item())


def lambda_decouple_geometry(
    gate_coord: torch.Tensor,
    lambda_geometry: str,
) -> torch.Tensor:
    """Resolve ``λ`` for ``global_decouple_min`` family cores / fused paths."""
    if lambda_geometry in SOFT_LAMBDA_GEOMETRY_FNS:
        return SOFT_LAMBDA_GEOMETRY_FNS[lambda_geometry](gate_coord)
    if lambda_geometry == "proj_gelu_tanh2_taylor":
        return lambda_gelu_tanh2_taylor(gate_coord)
    if lambda_geometry == "proj_oracle_x_native_norm0":
        return lambda_oracle_x_native_norm0(gate_coord)
    if lambda_geometry == "proj_oracle_div_native_norm0":
        return lambda_oracle_div_native_norm0(gate_coord)
    if lambda_geometry == "proj_oracle_geom_native_norm0":
        return lambda_oracle_geom_native_norm0(gate_coord)
    if lambda_geometry == "proj_oracle_dot_norm_native":
        return lambda_oracle_dot_norm_native(gate_coord)
    if lambda_geometry == "proj_gelu_x_native_over_s":
        return lambda_gelu_x_native_over_s(gate_coord)
    if lambda_geometry == "proj_oracle_mod_native_ortho":
        return lambda_oracle_mod_native_ortho(gate_coord, mix=0.0)
    if lambda_geometry == "proj_oracle_mod_native_simple":
        return lambda_oracle_mod_native_simple(gate_coord, mix=0.0)
    if lambda_geometry == "proj_oracle_fast":
        from tlb.kernels.hessian_ffn_fused import _lambda_oracle_gelu_fused

        return _lambda_oracle_gelu_fused(gate_coord)
    if lambda_geometry in ("proj_oracle", "global_oracle"):
        return lambda_oracle_gelu(gate_coord)
    lam = _tanh2_second_deriv(gate_coord)
    if lam.ndim == 2 and gate_coord.ndim == 3:
        lam = lam.unsqueeze(1)
    return lam


def fit_lambda_affine_to_gelu(
    *,
    bias: float = 0.0,
    grid: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """Find ``scale, shift`` minimizing ``(scale * u * λ(shift + scale_u * u) - GELU(u))^2``.

    Returns ``(lam_scale, lam_shift, rmse)`` on the grid.
    """
    if grid is None:
        grid = torch.linspace(-4.0, 4.0, 801)
    target = gelu_scalar(grid)
    best = (1.0, float(bias), float("inf"))
    for shift in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
        for scale in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            s = scale * grid + shift
            pred = scale * grid * lambda_raw(s)
            rmse = float(((pred - target) ** 2).mean().sqrt().item())
            if rmse < best[2]:
                best = (scale, shift, rmse)
    return best[0], best[1], best[2]


def default_native_calib_init(*, bias: float = 0.0) -> tuple[float, float, float]:
    """Grid-fit ``(amplitude, inner_scale, shift)`` for ``λ·u ≈ GELU(u)`` via native ψ''."""
    amp, shift, _ = fit_lambda_affine_to_gelu(bias=bias)
    return float(amp), float(amp), float(shift)


def lambda_native_calibrated(
    u: torch.Tensor,
    *,
    amplitude: torch.Tensor,
    inner_scale: torch.Tensor,
    shift: torch.Tensor,
    mode: Literal["grid_fit", "affine_gate"] = "grid_fit",
) -> torch.Tensor:
    """Native tanh²'' gate calibrated toward GELU.

    ``grid_fit`` (default): ``λ = A · ψ''(shift + α·u)`` with ``A·u·λ ≈ GELU(u)``.
    ``affine_gate``: ``λ = ψ''(α·u + β)`` (user Scheme-B; no outer amplitude).
    """
    if mode == "affine_gate":
        gate = inner_scale * u + shift
        return lambda_raw(gate)
    gate = shift + inner_scale * u
    return amplitude * lambda_raw(gate)


def joint_unit_from_mlp_row_col(
    w1_row: torch.Tensor,
    b1: float,
    w2_col: torch.Tensor,
    *,
    lambda_mode: LambdaMode = "calibrated",
    joint_mode: Literal["sum", "w1_only"] = "sum",
) -> JointUnit:
    """Couple ``W1[j,:]`` and ``W2[:,j]`` into one Hessian unit."""
    eps = 1e-8
    w_in = w1_row.detach().float()
    w_out = w2_col.detach().float()
    n_in = float(w_in.norm().item())
    n_out = float(w_out.norm().item())
    alignment = float((w_in @ w_out).item() / (n_in * n_out + eps))

    if joint_mode == "w1_only":
        w_joint = w_in.clone()
    else:
        w_joint = w_in + w_out
        if float(w_joint.norm().item()) < eps:
            w_joint = w_in if n_in > eps else w_out
        w_joint = w_joint / (float(w_joint.norm().item()) + eps)
        w_joint = w_joint * math.sqrt(max(n_in, eps) * max(n_out, eps))

    in_scale = float((w_in @ w_joint).item() / (float(w_joint.pow(2).sum().item()) + eps))
    out_scale = float((w_out @ w_joint).item() / (float(w_joint.pow(2).sum().item()) + eps))

    lam_scale, lam_shift, _ = fit_lambda_affine_to_gelu(bias=b1)
    if lambda_mode == "raw":
        lam_scale, lam_shift = 1.0, b1
    elif lambda_mode == "oracle_gelu":
        lam_scale, lam_shift = 1.0, b1

    return JointUnit(
        w_in=w_joint,
        w_out=w_out,
        b=float(b1),
        in_scale=in_scale,
        out_scale=out_scale,
        lam_scale=lam_scale,
        lam_shift=lam_shift,
        alignment=alignment,
    )


def _pick_num_units(hidden: int, hessian_hidden: int | None) -> int:
    return int(hessian_hidden if hessian_hidden is not None else hidden)


def translate_standard_ffn(
    mlp: StandardFFN,
    *,
    hessian_variant: str = "h4_phi768_global_decouple_min",
    hessian_hidden: int | None = None,
    lambda_mode: LambdaMode = "calibrated",
    joint_mode: Literal["sum", "w1_only"] = "sum",
) -> AnalyticTranslation:
    """Parse ``StandardFFN`` into analytic Hessian parameter tensors."""
    w1 = mlp.linear1.weight.detach().float()  # (H, D)
    b1 = mlp.linear1.bias.detach().float()  # (H,)
    w2 = mlp.linear2.weight.detach().float()  # (D, H)
    b2 = mlp.linear2.bias.detach().float()  # (D,)
    dim = int(w1.shape[1])
    hidden = int(w1.shape[0])
    num_units = _pick_num_units(hidden, hessian_hidden)

    units: list[JointUnit] = []
    for j in range(min(hidden, num_units)):
        units.append(
            joint_unit_from_mlp_row_col(
                w1[j],
                float(b1[j].item()),
                w2[:, j],
                lambda_mode=lambda_mode,
                joint_mode=joint_mode,
            )
        )

    weight = torch.zeros(num_units, dim)
    bias = torch.zeros(num_units)
    out_weight = torch.zeros(num_units, dim)
    for j, unit in enumerate(units):
        if joint_mode == "w1_only":
            weight[j] = w1[j]
            out_weight[j] = w2[:, j]
        else:
            weight[j] = unit.w_in
            out_weight[j] = unit.w_out
        bias[j] = unit.b

    readout_w: torch.Tensor | None = None
    readout_b: torch.Tensor | None = None
    out_w: torch.Tensor | None = out_weight
    if hessian_variant in ("h4_phi768",):
        # Tied core: output uses the same unit directions; readout stays identity.
        out_w = None
        readout_w = torch.eye(dim)
        readout_b = b2.clone()
        for j, unit in enumerate(units):
            weight[j] = unit.w_in

    scale_logits = None
    window_sizes = None
    if hessian_variant in ("h4_phi768_global_ms", "h4_phi768_global_decouple"):
        window_sizes = (4, 16, 64)
        scale_logits = _ms_scale_logits_global_only(len(window_sizes))

    return AnalyticTranslation(
        variant=hessian_variant,
        weight=weight,
        bias=bias,
        out_weight=out_w,
        out_bias=b2.clone(),
        readout_weight=readout_w,
        readout_bias=readout_b,
        hp_logit=10.0 if hessian_variant.endswith("global_ms") else 2.0,
        scale_logits=scale_logits,
        window_sizes=window_sizes,
        units=units,
        lambda_mode=lambda_mode,
    )


def _ms_scale_logits_global_only(num_windows: int) -> torch.Tensor:
    """Softmax → 1 on the appended global-mean slot (last index)."""
    logits = torch.full((num_windows + 1,), -20.0)
    logits[-1] = 20.0
    return logits


def promote_decouple_min_to_global_ms_decouple(
    tr: AnalyticTranslation,
    *,
    variant: str = "h4_phi768_global_decouple",
) -> AnalyticTranslation:
    """Lift decouple_min translation to ms-decouple (same U/V, degenerate ms knobs)."""
    if tr.out_weight is None:
        raise ValueError("promote_decouple_min_to_global_ms_decouple requires out_weight")
    window_sizes = (4, 16, 64)
    return AnalyticTranslation(
        variant=variant,
        weight=tr.weight.clone(),
        bias=tr.bias.clone(),
        out_weight=tr.out_weight.clone(),
        out_bias=tr.out_bias.clone() if tr.out_bias is not None else None,
        readout_weight=tr.readout_weight.clone() if tr.readout_weight is not None else None,
        readout_bias=tr.readout_bias.clone() if tr.readout_bias is not None else None,
        hp_logit=10.0,
        scale_logits=_ms_scale_logits_global_only(len(window_sizes)),
        window_sizes=window_sizes,
        units=list(tr.units),
        lambda_mode=tr.lambda_mode,
    )


def tie_out_weight_to_ms(
    tr: AnalyticTranslation,
    *,
    tie_mode: TieMode = "w1",
    variant: str = "h4_phi768_global_ms",
) -> AnalyticTranslation:
    """Collapse ms-decouple U/V into tied global_ms weight."""
    if tr.out_weight is None:
        raise ValueError("tie_out_weight_to_ms requires out_weight")
    weight = tr.weight.clone()
    out_w = tr.out_weight
    m, _d = weight.shape
    for j in range(m):
        w_in = weight[j]
        w_out = out_w[j]
        if tie_mode == "out":
            weight[j] = w_out
        elif tie_mode == "joint":
            joint = w_in + w_out
            n = float(joint.norm().item())
            if n > 1e-8:
                weight[j] = joint / n * math.sqrt(
                    max(float(w_in.norm().item()), 1e-8)
                    * max(float(w_out.norm().item()), 1e-8)
                )
        # tie_mode == "w1": keep weight[j] as input row
    return AnalyticTranslation(
        variant=variant,
        weight=weight,
        bias=tr.bias.clone(),
        out_weight=None,
        out_bias=tr.out_bias.clone() if tr.out_bias is not None else None,
        readout_weight=tr.readout_weight.clone() if tr.readout_weight is not None else None,
        readout_bias=tr.readout_bias.clone() if tr.readout_bias is not None else None,
        hp_logit=tr.hp_logit,
        scale_logits=tr.scale_logits.clone() if tr.scale_logits is not None else None,
        window_sizes=tr.window_sizes,
        units=list(tr.units),
        lambda_mode=tr.lambda_mode,
    )


def promote_decouple_min_to_global_ms(
    tr: AnalyticTranslation,
    *,
    tie_mode: TieMode = "w1",
) -> AnalyticTranslation:
    """decouple_min → global_ms_decouple → tied global_ms."""
    ms_dec = promote_decouple_min_to_global_ms_decouple(tr)
    return tie_out_weight_to_ms(ms_dec, tie_mode=tie_mode)


def apply_analytic_translation(module: HessianFFN, tr: AnalyticTranslation) -> None:
    """Write ``AnalyticTranslation`` into an existing ``HessianFFN``.

    Init follows the forward compute order (U → A → V), not a simultaneous tie:

    1. ``weight`` (U) ← MLP ``W1`` rows  (content projection)
    2. ``context_weight`` (A) ← same ``W1`` basis when present (λ routing)
    3. ``out_weight`` (V) ← MLP ``W2`` cols (readout; always after U/A)

    Scratch ctxsplit uses ``A←U, V←U`` in ``__init__`` only to match the decouple
    parent at step~0.  Pretrain / fine-tune warm-start must use this ordered path so
    V inherits the contraction leg, not a copy of the expansion leg.
    """
    core = module.phi_core
    if core is None or not hasattr(core, "weight"):
        raise TypeError(f"variant {module.variant!r} has no phi_core.weight for analytic init")
    with torch.no_grad():
        rows = min(int(core.weight.shape[0]), int(tr.weight.shape[0]))
        dim = int(core.weight.shape[1])
        core.weight[:rows, :dim].copy_(tr.weight[:rows, :dim])
        if hasattr(core, "content_bias"):
            core.content_bias[:rows].copy_(tr.bias[:rows])
        elif hasattr(core, "bias"):
            core.bias[:rows].copy_(tr.bias[:rows])
        if hasattr(core, "gate_bias"):
            core.gate_bias[:rows].copy_(tr.bias[:rows])
        if hasattr(core, "context_weight"):
            if getattr(core, "context_weight_zero_init", False):
                core.context_weight.zero_()
            else:
                core.context_weight[:rows, :dim].copy_(tr.weight[:rows, :dim])
        if tr.out_weight is not None and hasattr(core, "out_weight"):
            core.out_weight[:rows, :dim].copy_(tr.out_weight[:rows, :dim])
        if tr.out_bias is not None and hasattr(core, "out_bias"):
            core.out_bias.copy_(tr.out_bias)
        total = int(core.weight.shape[0])
        if rows < total:
            core.weight[rows:].zero_()
            if hasattr(core, "content_bias"):
                core.content_bias[rows:].zero_()
            elif hasattr(core, "bias"):
                core.bias[rows:].zero_()
            if hasattr(core, "out_weight"):
                core.out_weight[rows:].zero_()
        if hasattr(core, "hp_logit") and tr.hp_logit is not None:
            core.hp_logit.fill_(float(tr.hp_logit))
        if hasattr(core, "scale_logits") and tr.scale_logits is not None:
            if int(core.scale_logits.numel()) == int(tr.scale_logits.numel()):
                core.scale_logits.copy_(tr.scale_logits)
        elif hasattr(core, "scale_logits") and hasattr(core, "window_sizes"):
            core.scale_logits.fill_(-1.0)
            core.scale_logits[-1] = 2.0
    if tr.readout_weight is not None and module.readout is not None and isinstance(
        module.readout, nn.Linear
    ):
        with torch.no_grad():
            module.readout.weight.copy_(tr.readout_weight)
            if tr.readout_bias is not None and module.readout.bias is not None:
                module.readout.bias.copy_(tr.readout_bias)


class AnalyticHessianFFN(nn.Module):
    """Explicit Hessian map for arithmetic tests (no training)."""

    def __init__(
        self,
        tr: AnalyticTranslation,
        *,
        use_outer_gelu: bool = False,
        lambda_on: Literal["proj", "global_z"] = "proj",
    ) -> None:
        super().__init__()
        self.tr = tr
        self.use_outer_gelu = use_outer_gelu
        self.lambda_on = lambda_on
        self.register_buffer("weight", tr.weight.clone())
        self.register_buffer("bias", tr.bias.clone())
        if tr.out_weight is not None:
            self.register_buffer("out_weight", tr.out_weight.clone())
        else:
            self.out_weight = None
        if tr.out_bias is not None:
            self.register_buffer("out_bias", tr.out_bias.clone())
        else:
            self.out_bias = None
        if tr.readout_weight is not None:
            self.register_buffer("readout_weight", tr.readout_weight.clone())
            self.register_buffer("readout_bias", tr.readout_bias.clone())
        else:
            self.readout_weight = None
            self.readout_bias = None

    def _lambda_from_u(self, u: torch.Tensor) -> torch.Tensor:
        if self.tr.lambda_mode == "oracle_gelu":
            return lambda_oracle_gelu(u)
        if self.tr.lambda_mode == "raw":
            return lambda_raw(u)
        out = torch.empty_like(u)
        for j, unit in enumerate(self.tr.units):
            if j >= u.shape[-1]:
                break
            u_j = u[..., j]
            s_j = unit.lam_scale * u_j + unit.lam_shift
            lam_j = lambda_raw(s_j)
            gelu_u = gelu_scalar(u_j)
            denom = (u_j * lam_j).abs().clamp(min=1e-6)
            out[..., j] = lam_j * (gelu_u / denom).clamp(-10.0, 10.0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)
        w_in = self.weight.transpose(-2, -1)
        proj = x @ w_in + self.bias
        if self.lambda_on == "global_z":
            u = x.mean(dim=1, keepdim=True) @ w_in + self.bias
            u = u.expand_as(proj)
        else:
            u = proj
        lam = self._lambda_from_u(u)
        if self.out_weight is not None:
            h = (lam * proj) @ self.out_weight
            if self.out_bias is not None:
                h = h + self.out_bias
        else:
            h = torch.einsum("...m,md->...d", lam * proj, self.weight)
            if self.readout_weight is not None:
                h = h @ self.readout_weight.T
                if self.readout_bias is not None:
                    h = h + self.readout_bias
        if self.use_outer_gelu:
            h = F.gelu(h)
        return h.squeeze(1) if squeeze else h


class AnalyticGlobalMsFFN(nn.Module):
    """Explicit global_ms / global_ms_decouple map for promotion tests."""

    def __init__(self, tr: AnalyticTranslation) -> None:
        super().__init__()
        if tr.window_sizes is None or tr.scale_logits is None or tr.hp_logit is None:
            raise ValueError("AnalyticGlobalMsFFN requires ms scale_logits/window_sizes/hp_logit")
        self.tr = tr
        self.window_sizes = tr.window_sizes
        self.register_buffer("weight", tr.weight.clone())
        self.register_buffer("bias", tr.bias.clone())
        self.register_buffer("scale_logits", tr.scale_logits.clone())
        self.register_buffer("hp_logit", torch.tensor(float(tr.hp_logit)))
        if tr.out_weight is not None:
            self.register_buffer("out_weight", tr.out_weight.clone())
        else:
            self.out_weight = None
        if tr.out_bias is not None:
            self.register_buffer("out_bias", tr.out_bias.clone())
        else:
            self.out_bias = None

    def _fused_z(self, x: torch.Tensor) -> torch.Tensor:
        z_parts = [_symmetric_window_mean(x, w) for w in self.window_sizes]
        z_parts.append(_global_seq_mean(x))
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = sum(sw * zp for sw, zp in zip(scale_w, z_parts))
        return z

    def _lambda_from_u(self, u: torch.Tensor) -> torch.Tensor:
        if self.tr.lambda_mode == "oracle_gelu":
            return lambda_oracle_gelu(u)
        if self.tr.lambda_mode == "raw":
            return lambda_raw(u)
        out = torch.empty_like(u)
        for j, unit in enumerate(self.tr.units):
            if j >= u.shape[-1]:
                break
            u_j = u[..., j]
            s_j = unit.lam_scale * u_j + unit.lam_shift
            lam_j = lambda_raw(s_j)
            gelu_u = gelu_scalar(u_j)
            denom = (u_j * lam_j).abs().clamp(min=1e-6)
            out[..., j] = lam_j * (gelu_u / denom).clamp(-10.0, 10.0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)
        z = self._fused_z(x)
        z_win = _symmetric_window_mean(x, self.window_sizes[-1])
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w_in = self.weight.transpose(-2, -1)
        proj = proj_in @ w_in + self.bias
        s_z = z @ w_in + self.bias
        lam = self._lambda_from_u(s_z)
        if self.out_weight is not None:
            h = (lam * proj) @ self.out_weight
        else:
            h = (lam * proj) @ self.weight
        if self.out_bias is not None:
            h = h + self.out_bias
        return h.squeeze(1) if squeeze else h


def build_analytic_ffn(tr: AnalyticTranslation) -> nn.Module:
    """Pick explicit analytic engine from translation variant."""
    if tr.window_sizes is not None and tr.scale_logits is not None:
        return AnalyticGlobalMsFFN(tr)
    return AnalyticHessianFFN(tr, lambda_on="proj")


@torch.no_grad()
def compare_ffn_outputs(
    mlp: StandardFFN,
    hessian: nn.Module,
    x: torch.Tensor,
) -> dict[str, float]:
    was_mlp = mlp.training
    was_h = hessian.training
    mlp.eval()
    hessian.eval()
    if x.ndim == 2:
        x_mlp = x
        x_h = x.unsqueeze(1)
    else:
        x_mlp = x
        x_h = x
    y_ref = mlp(x_mlp if x_mlp.ndim == 3 else x_mlp)
    y_hat = hessian(x_h if x_h.ndim == 3 else x_h)
    if y_hat.ndim == 2 and y_ref.ndim == 3:
        y_hat = y_hat.unsqueeze(1)
    y_ref_f = y_ref.reshape(-1, y_ref.shape[-1]).float()
    y_hat_f = y_hat.reshape(-1, y_hat.shape[-1]).float()
    mse = float(F.mse_loss(y_hat_f, y_ref_f).item())
    rel = mse / float(y_ref_f.pow(2).mean().item() + 1e-12)
    cos = float(F.cosine_similarity(y_hat_f, y_ref_f, dim=-1).mean().item())
    ref_norm = float(y_ref_f.norm(dim=-1).mean().item())
    hat_norm = float(y_hat_f.norm(dim=-1).mean().item())
    mlp.train(was_mlp)
    hessian.train(was_h)
    return {
        "mse": mse,
        "rel_mse": rel,
        "cosine": cos,
        "ref_norm": ref_norm,
        "hat_norm": hat_norm,
        "norm_ratio": hat_norm / (ref_norm + 1e-12),
    }


def gelu_lambda_curve_table(
    *,
    grid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if grid is None:
        grid = torch.linspace(-4.0, 4.0, 801)
    gelu = gelu_scalar(grid)
    lam = lambda_raw(grid)
    prod = grid * lam
    oracle = lambda_oracle_gelu(grid)
    scale, shift, rmse = fit_lambda_affine_to_gelu()
    calibrated = scale * grid * lambda_raw(scale * grid + shift)
    return {
        "grid": grid,
        "gelu": gelu,
        "u_times_lambda_raw": prod,
        "lambda_oracle_gelu": oracle,
        "u_times_lambda_calibrated": calibrated,
        "calibrated_rmse": torch.tensor(rmse),
        "calibrated_scale": torch.tensor(scale),
        "calibrated_shift": torch.tensor(shift),
    }
