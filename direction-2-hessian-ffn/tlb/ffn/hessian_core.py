"""Hessian FFN variants built on H1/H3 main path.

H3: GeLU(x @ W) → Linear(d,d)
H3_a: H3 with wide readout (d → 4d → d) — path A capacity
H3_b: H3 + learnable Φ top-k spectral branch at pooled z — path B
H3_c: H3 + learnable L(z)L(z)^T low-rank branch — path C
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

DecoupleMinLambda = Literal[
    "global_tanh2",
    "proj_oracle",
    "proj_oracle_fast",
    "global_oracle",
    "proj_native_calib",
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

HessianVariant = Literal[
    "h1",
    "h2",
    "h3",
    "h3_topk",
    "h3_stack",
    "h3_a",
    "h3_b",
    "h3_c",
    "h3_nores",
    "h3_gated",
    "h3_mix",
    "h4_rank",
    "h4_phi",
    "h4_phi768",
    "h4_phi768_prefix",
    "h4_phi768_prefix_ema",
    "h4_phi768_prefix_win",
    "h4_phi768_prefix_attn",
    "h4_phi768_prefix_attn_eps",
    "h4_phi768_prefix_ms",
    "h4_phi768_prefix_ms_decouple",
    "h4_phi768_prefix_ms_decouple_ctxsplit",
    "h4_phi768_prefix_ms_decouple_ctxsplit_x_tanh",
    "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_agg",
    "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z",
    "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z_active_rank",
    "h4_phi768_prefix_ms_decouple_ctxsplit_mod_native_gelu_z",
    "h4_phi768_prefix_ms_decouple_dual",
    "h4_phi768_prefix_ms_decouple_seq",
    "h4_phi768_prefix_ms_d1",
    "h4_phi768_prefix_ms_reg",
    "h4_phi768_prefix_ms_pt",
    "h4_phi768_prefix_ms_glu",
    "h4_phi768_prefix_ms_pt_glu",
    "h4_phi768_st",
    "h4_phi768_block_diag",
    "h4_phi768_prefix_moe",
    "h4_phi768_prefix_lam_global",
    "h4_phi768_prefix_ms_stats",
    "h4_phi768_prefix_ms_multwin",
    "h4_phi768_prefix_ms_multwin_deep",
    "h4_phi768_prefix_ms_multwin_dseg",
    "h4_phi768_prefix_ms_multwin_wide",
    "h4_phi768_prefix_ms_multwin_resread",
    "h4_phi1536_prefix_ms_multwin",
    "h4_phi2304_prefix_ms_multwin",
    "h4_phi3072_prefix_ms_multwin",
    "h3_form_a",
    "h4_phi768_prefix_ms_multwin_softplus",
    "h4_phi768_prefix_ms_multwin_temp",
    "h4_phi768_prefix_ms_multwin_rational",
    "h4_phi768_prefix_ms_multwin_shiftedsp",
    "h4_phi768_prefix_ms_multwin_mish",
    "h4_phi768_prefix_ms_multwin_swish",
    "h4_phi768_prefix_ms_multwin_signedfloor",
    "h4_phi768_prefix_ms_multwin_tanhsp",
    "h4_phi768_prefix_ms_multwin_invcombo",
    "h4_phi768_prefix_ms_multwin_invcombo_strong",
    "h4_phi768_prefix_ms_multwin_invcombo_deep",
    "h4_phi768_prefix_ms_multwin_invcombo_wide",
    "h4_phi768_prefix_ms_multwin_invcombo_strong_wide",
    "h4_phi768_prefix_ms_multwin_invcombo_deep_wide",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_small",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_mid",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_learn",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m10",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m20",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m35",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder8_odd_m20",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_m50",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_hot_m50",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_amp2_hot_m50",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_gain",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_gain_lammix",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_addodd_t10",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_addodd_delta_t10",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_mix_addodd_delta",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_cross_r64",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_bilinear_r64",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_decouple",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_decouple_ctxsplit",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_wide",
    "h4_phi768_prefix_ms_multwin_invcombo_ladder_resread",
    # Bidirectional (vision / non-causal seq): global + symmetric windows, no causal prefix.
    "h4_phi768_global_ms",
    "h4_phi768_global_decouple",
    "h4_phi768_global_light_decouple",
    "h4_phi768_global_decouple_min",
    "h4_phi768_global_decouple_min_side",
    "h4_phi768_global_decouple_min_side_up",
    "h4_phi768_global_decouple_min_side_curv",
    "h4_phi768_global_decouple_min_side_units",
    "h4_phi768_global_decouple_min_abroadcast",
    "h4_phi768_global_decouple_min_resonance",
    "h4_phi768_global_decouple_min_offaxis",
    "h4_phi768_global_decouple_min_active_rank",
    "h4_phi768_global_decouple_min_native_calib",
    "h4_phi768_global_decouple_min_uav",
    "h4_phi768_global_decouple_min_2d",
]

# phi_core output is the full FFN map (no outer GELU); used for vision MLP swap variants.
_PHI_DIRECT_OUTPUT = frozenset(
    {
        "h4_phi768_global_ms",
        "h4_phi768_global_decouple",
        "h4_phi768_global_light_decouple",
        "h4_phi768_global_decouple_min",
        "h4_phi768_global_decouple_min_side",
        "h4_phi768_global_decouple_min_side_up",
        "h4_phi768_global_decouple_min_side_curv",
        "h4_phi768_global_decouple_min_side_units",
        "h4_phi768_global_decouple_min_abroadcast",
        "h4_phi768_global_decouple_min_resonance",
        "h4_phi768_global_decouple_min_offaxis",
        "h4_phi768_global_decouple_min_active_rank",
        "h4_phi768_global_decouple_min_native_calib",
        "h4_phi768_global_decouple_min_uav",
        "h4_phi768_global_decouple_min_2d",
        "h4_phi768_prefix_ms_decouple_ctxsplit_x_tanh",
        "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_agg",
        "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z",
        "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z_active_rank",
    }
)

_SWISH_SPECTRAL = frozenset({"h4_phi768_prefix_attn_eps"})

_H3_SPECTRAL = frozenset(
    {"h3", "h3_topk", "h3_a", "h3_b", "h3_c", "h3_nores", "h3_gated", "h3_mix"}
)
_HAS_READOUT = frozenset(
    {
        "h3",
        "h3_topk",
        "h3_stack",
        "h3_a",
        "h3_b",
        "h3_c",
        "h3_nores",
        "h3_gated",
        "h3_mix",
        "h4_rank",
        "h4_phi",
        "h4_phi768",
        "h4_phi768_prefix",
        "h4_phi768_prefix_ema",
        "h4_phi768_prefix_win",
        "h4_phi768_prefix_attn",
        "h4_phi768_prefix_attn_eps",
        "h4_phi768_prefix_ms",
    "h4_phi768_prefix_ms_decouple",
    "h4_phi768_prefix_ms_decouple_ctxsplit",
    "h4_phi768_prefix_ms_decouple_dual",
    "h4_phi768_prefix_ms_decouple_seq",
        "h4_phi768_prefix_ms_d1",
        "h4_phi768_prefix_ms_reg",
        "h4_phi768_prefix_ms_pt",
        "h4_phi768_prefix_ms_glu",
        "h4_phi768_prefix_ms_pt_glu",
        "h4_phi768_st",
        "h4_phi768_block_diag",
        "h4_phi768_prefix_moe",
        "h4_phi768_prefix_lam_global",
        "h4_phi768_prefix_ms_stats",
        "h4_phi768_prefix_ms_multwin",
        "h4_phi768_prefix_ms_multwin_deep",
        "h4_phi768_prefix_ms_multwin_dseg",
        "h4_phi768_prefix_ms_multwin_wide",
        "h4_phi768_prefix_ms_multwin_resread",
        "h4_phi1536_prefix_ms_multwin",
        "h4_phi1536_prefix_ms_multwin_wide",
        "h4_phi768_prefix_ms_multwin_hybrid",
        "h3_form_a",
        "h4_phi768_prefix_ms_multwin_softplus",
        "h4_phi768_prefix_ms_multwin_temp",
        "h4_phi768_prefix_ms_multwin_rational",
        "h4_phi768_prefix_ms_multwin_shiftedsp",
        "h4_phi768_prefix_ms_multwin_mish",
        "h4_phi768_prefix_ms_multwin_swish",
        "h4_phi768_prefix_ms_multwin_signedfloor",
        "h4_phi768_prefix_ms_multwin_tanhsp",
        "h4_phi768_prefix_ms_multwin_invcombo",
        "h4_phi768_prefix_ms_multwin_invcombo_strong",
        "h4_phi768_prefix_ms_multwin_invcombo_deep",
        "h4_phi768_prefix_ms_multwin_invcombo_wide",
        "h4_phi768_prefix_ms_multwin_invcombo_strong_wide",
        "h4_phi768_prefix_ms_multwin_invcombo_deep_wide",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_small",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_mid",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_learn",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m10",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m20",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m35",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder8_odd_m20",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_m50",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_hot_m50",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_amp2_hot_m50",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_gain",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_gain_lammix",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_addodd_t10",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_addodd_delta_t10",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_mix_addodd_delta",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_cross_r64",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_bilinear_r64",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_decouple",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_decouple_ctxsplit",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_wide",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_resread",
    }
)


def _sum_tanh2_potential(z: torch.Tensor) -> torch.Tensor:
    return torch.tanh(z).pow(2).sum()


def _tanh2_hessian_diag(z: torch.Tensor) -> torch.Tensor:
    """Diagonal of Hessian of sum(tanh(x_i)^2). z: (..., D)."""
    t = torch.tanh(z)
    s2 = (1.0 - t * t).clamp(min=1e-8)
    return 2.0 * s2 * s2 - 4.0 * t * t * s2


def _learnable_sep_hessian_diag(
    z: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Diagonal of Hessian of sum_i alpha_i tanh(beta_i z_i)^2."""
    t = torch.tanh(beta * z)
    s2 = (1.0 - t * t).clamp(min=1e-8)
    return (
        2.0 * alpha * s2 * s2 - 4.0 * alpha * (beta * beta) * t * t * s2
    )


def _topk_spectral_apply(x_bt: torch.Tensor, h: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-|λ| of symmetric H to each row of x_bt. x_bt: (T,D), H: (D,D)."""
    d = x_bt.size(-1)
    k = min(int(k), d)
    evals, evecs = torch.linalg.eigh(h.float().cpu())
    evals = evals.to(device=x_bt.device, dtype=x_bt.dtype)
    evecs = evecs.to(device=x_bt.device, dtype=x_bt.dtype)
    idx = torch.topk(evals.abs(), k, largest=True).indices
    lk = evals[idx]
    vk = evecs[:, idx]
    coef = x_bt @ vk
    return (coef * lk.unsqueeze(0)) @ vk.transpose(0, 1)


def _topk_fixed_tanh2_branch(x: torch.Tensor, k: int) -> torch.Tensor:
    b, _t, _d = x.shape
    out = torch.empty_like(x)
    for i in range(b):
        z = x[i].mean(dim=0)
        h = torch.autograd.functional.hessian(_sum_tanh2_potential, z)
        out[i] = _topk_spectral_apply(x[i], h, k)
    return out


class _SymmetricPotential(nn.Module):
    """Learnable Φ(x) = 0.5 x^T W x, W = L L^T positive-semidefinite."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.l_param = nn.Parameter(torch.randn(dim, dim) * 0.02)

    def weight_matrix(self) -> torch.Tensor:
        l = torch.tril(self.l_param)
        return l @ l.transpose(-2, -1)


class _LearnableSepPotential(nn.Module):
    """Φ(z) = 0.5 z^T W z + sum_i alpha_i tanh(beta_i z_i)^2 with closed-form H(z)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.quad = _SymmetricPotential(dim)
        self.alpha = nn.Parameter(torch.full((dim,), 0.1))
        self.beta = nn.Parameter(torch.ones(dim))

    def hessian(self, z: torch.Tensor) -> torch.Tensor:
        w = self.quad.weight_matrix()
        diag = _learnable_sep_hessian_diag(z, self.alpha, self.beta)
        return w + torch.diag(diag)


class _LearnableTopKBranch(nn.Module):
    """Path B: top-k of learnable input-dependent H(z) at sequence mean."""

    def __init__(self, dim: int, k: int = 16) -> None:
        super().__init__()
        self.k = k
        self.potential = _LearnableSepPotential(dim)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _t, _d = x.shape
        out = torch.empty_like(x)
        for i in range(b):
            z = x[i].mean(dim=0)
            h = self.potential.hessian(z)
            out[i] = _topk_spectral_apply(x[i], h, self.k)
        return self.gate * out


class _DynamicLowRankBranch(nn.Module):
    """Path C: sum_r s_r(z) * (x·u_r) u_r with learnable directions u_r."""

    def __init__(self, dim: int, rank: int = 32) -> None:
        super().__init__()
        self.rank = rank
        self.directions = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.scale = nn.Linear(dim, rank)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.mean(dim=1)
        weights = self.scale(z)
        mid = torch.einsum("btd,rd->btr", x, self.directions)
        mid = mid * weights.unsqueeze(1)
        branch = torch.einsum("btr,rd->btd", mid, self.directions)
        return self.gate * branch


def _tanh2_second_deriv(z: torch.Tensor) -> torch.Tensor:
    """Second derivative of tanh(z)^2, elementwise."""
    t = torch.tanh(z)
    s2 = (1.0 - t * t).clamp(min=1e-8)
    return 2.0 * s2 * s2 - 4.0 * t * t * s2


_MULTPEAK_ORIGIN_SCALE: dict[tuple[float, tuple[float, ...]], float] = {}


def _tanh2_paper_lobe(
    z: torch.Tensor,
    temperature: float,
    centers: tuple[float, ...] | None,
    shift: float = 0.0,
) -> torch.Tensor:
    """``ψ''((s-μ)/τ)``, or a sum of lobes at ``centers`` rescaled to ``λ(0)=2``.

    A nonzero ``shift`` moves the single well off the origin and does not
    rescale. The peak stays 2 at ``s=μ``.
    """
    tau = max(float(temperature), 1e-6)
    mu = float(shift)
    if not centers or centers == (0.0,):
        arg = z - mu
        if abs(tau - 1.0) >= 1e-8:
            arg = arg / tau
        return _tanh2_second_deriv(arg)
    acc: torch.Tensor | None = None
    for mu in centers:
        piece = _tanh2_second_deriv((z - mu) / tau)
        acc = piece if acc is None else acc + piece
    assert acc is not None
    key = (round(tau, 6), centers)
    scale = _MULTPEAK_ORIGIN_SCALE.get(key)
    if scale is None:
        origin = 0.0
        for mu in centers:
            origin += float(_tanh2_second_deriv(z.new_tensor([-mu / tau])).item())
        scale = 2.0 / origin if abs(origin) > 1e-8 else 1.0
        _MULTPEAK_ORIGIN_SCALE[key] = scale
    return acc * scale


def tanh2_softened(
    z: torch.Tensor,
    temperature: float = 1.0,
    *,
    amplitude: str = "paper",
    amplitude_scale: float = 1.0,
    peak_centers: tuple[float, ...] | None = None,
    peak_shift: float = 0.0,
) -> torch.Tensor:
    """Paper gate ``ψ''(s/τ)``, optional peak match to ``GELU(s)/s``.

    ``τ=1`` is exact ``tanh²''``. Temperature widens the lobe; it does not
    change ``λ(0)`` by itself.

    ``amplitude='paper'`` keeps the signed tanh²'' shape. ``amplitude_scale``
    multiplies it, so ``λ(0)=2·scale`` (scale 1 is the paper peak).
    ``amplitude='gelu_band'`` affinely maps the tanh²'' range ``[-2/3, 2]``
    onto ``[0, 1]``, the same value strip as ``GELU(s)/s``. Peak stays at the
    top of that strip (``λ(0)=1``); the former negative lobe sits on 0.
    Temperature still only widens the lobe. ``amplitude_scale`` does not apply
    to ``gelu_band``.

    ``peak_centers`` sums the same signed lobe at each center and rescales so
    ``λ(0)=2`` before ``amplitude_scale``. ``(0,)`` is the single paper peak.
    ``peak_shift`` moves that single well to ``s=μ`` without rescaling.
    """
    lam = _tanh2_paper_lobe(z, temperature, peak_centers, shift=peak_shift)
    if amplitude == "paper":
        return lam * float(amplitude_scale)
    if amplitude == "gelu_band":
        # tanh²'' ∈ [-2/3, 2]; (λ - min) / (max - min) ∈ [0, 1].
        return (lam + (2.0 / 3.0)) / (8.0 / 3.0)
    raise ValueError(f"unknown tanh2 amplitude: {amplitude!r}")


def _softplus_second_deriv(z: torch.Tensor) -> torch.Tensor:
    """Second derivative of softplus(z), elementwise.

    softplus(x) = log(1 + exp(x))
    softplus'(x) = sigmoid(x)
    softplus''(x) = sigmoid(x) * (1 - sigmoid(x))

    Always positive, smooth, no hard saturation.
    """
    s = torch.sigmoid(z)
    return s * (1.0 - s)


def _rational_second_deriv(z: torch.Tensor, eps: float = 0.5) -> torch.Tensor:
    """Rational function approximation: ψ(s) = ε·s²/(ε+s²) = s²/(1+s²/ε).

    ψ'(s) = 2εs/(ε+s²)
    ψ''(s) = [2ε(ε+s²) - 2εs·2s] / (ε+s²)² = 2ε(ε-s²) / (ε+s²)²

    Characteristics:
    - Peak ~2.0 at s=0 (similar to tanh²'')
    - Negative ratio ~85% (similar to tanh²'')
    - Slower decay in |s|>2 region (0.083 vs tanh²'' 0.045)
    """
    z2 = z * z
    eps_tensor = torch.tensor(eps, dtype=z.dtype, device=z.device)
    denom = eps_tensor + z2
    # Numerical stability: clamp denominator
    denom = denom.clamp(min=1e-6)
    return 2.0 * eps_tensor * (eps_tensor - z2) / (denom * denom)


def _rational_second_deriv_v2(z: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Vectorized version with learnable eps (tensor input for gradient flow).

    eps: (1,) or scalar tensor that requires grad
    """
    z2 = z * z
    # Expand eps to match z's shape for broadcasting
    eps_expanded = eps.view(1, 1, 1) if eps.dim() == 0 else eps
    denom = eps_expanded + z2
    denom = denom.clamp(min=1e-6)
    return 2.0 * eps_expanded * (eps_expanded - z2) / (denom * denom)


def _shifted_softplus_second_deriv(z: torch.Tensor, shift: float = 2.0) -> torch.Tensor:
    """Shifted softplus: ψ(s) = softplus(s+shift) - softplus(s-shift).

    Creates bipolar response with positive center and negative tails.

    Characteristics:
    - Peak ~0.23 (lower than tanh²'')
    - 50% negative values
    - Very slow decay (0.145 in |s|>2 region)
    """
    s_pos = torch.sigmoid(z + shift)
    s_neg = torch.sigmoid(z - shift)
    return s_pos * (1.0 - s_pos) - s_neg * (1.0 - s_neg)


def _shifted_softplus_second_deriv_v2(z: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Vectorized version with learnable shift (tensor input for gradient flow)."""
    shift_expanded = shift.view(1, 1, 1) if shift.dim() == 0 else shift
    s_pos = torch.sigmoid(z + shift_expanded)
    s_neg = torch.sigmoid(z - shift_expanded)
    return s_pos * (1.0 - s_pos) - s_neg * (1.0 - s_neg)


def _swish_second_deriv(z: torch.Tensor) -> torch.Tensor:
    """Swish(x) = x * sigmoid(x) second derivative.

    Characteristics:
    - Peak ~0.50
    - 52% negative values
    - Smooth, self-gated behavior
    """
    s = torch.sigmoid(z)
    # Swish'' = s*(1-s) * [2 + z*(1-2s)]
    return s * (1.0 - s) * (2.0 + z * (1.0 - 2.0 * s))


def _mish_second_deriv(z: torch.Tensor) -> torch.Tensor:
    """Mish(x) = x * tanh(softplus(x)) second derivative.

    Characteristics:
    - Peak ~0.64
    - 62.6% negative values
    - Smooth, self-gated behavior
    """
    sp = F.softplus(z)
    t = torch.tanh(sp)
    sp_deriv = torch.sigmoid(z)
    t_deriv = 1.0 - t * t

    # First derivative components
    term1 = t_deriv * sp_deriv  # tanh'(softplus) * sigmoid(z)

    # Second derivative components
    t_second = -2.0 * t * t_deriv  # tanh''(softplus)
    sp_second = sp_deriv * (1.0 - sp_deriv)  # sigmoid'(z)

    # Mish'' = 2 * term1 + z * (t_second * sp_deriv^2 + t_deriv * sp_second)
    term2 = z * (t_second * sp_deriv * sp_deriv + t_deriv * sp_second)
    return term1 + term1 + term2


def _tanh2_signed_floor_second_deriv(
    z: torch.Tensor, floor_scale: float = 0.25, floor_temp: float = 2.0
) -> torch.Tensor:
    """tanh²'' with a broad signed curvature floor.

    softplus'' supplies non-saturating magnitude while sign(tanh²'') preserves
    the positive/negative curvature structure that plain softplus lost.
    """
    base = _tanh2_second_deriv(z)
    floor = 4.0 * _softplus_second_deriv(z / floor_temp)
    return base + floor_scale * torch.sign(base) * floor


def _tanh2_softplus_blend_second_deriv(
    z: torch.Tensor, blend: float = 0.25
) -> torch.Tensor:
    """Direct tanh²'' + softplus'' blend.

    This tests whether a small always-positive curvature floor helps without
    fully discarding tanh²'' signed structure.
    """
    return _tanh2_second_deriv(z) + blend * 4.0 * _softplus_second_deriv(z)


def _tanh2_invcombo_second_deriv(
    z: torch.Tensor, inv_scale: float = 0.25, temp: float = 2.0
) -> torch.Tensor:
    """tanh²'' plus a damped inverse-saturation proxy.

    ``sech²`` itself is just ``1 - tanh²`` so its second derivative is
    ``-tanh²''``. Instead use a broader signed proxy based on tanh(s/temp),
    which behaves like a softened inverse/complement path without singularities.
    """
    base = _tanh2_second_deriv(z)
    broad = _tanh2_second_deriv(z / temp)
    return base + inv_scale * broad


def _tanh2_invcombo_strong_second_deriv(z: torch.Tensor) -> torch.Tensor:
    """A stronger two-scale signed curvature: tanh²''(s) + 0.5*tanh²''(s/2)."""
    return _tanh2_second_deriv(z) + 0.5 * _tanh2_second_deriv(z / 2.0)


def _tanh2_invcombo_deep_second_deriv(z: torch.Tensor) -> torch.Tensor:
    """Three-scale signed curvature ladder with diminishing weights."""
    return (
        _tanh2_second_deriv(z)
        + 0.25 * _tanh2_second_deriv(z / 2.0)
        + 0.125 * _tanh2_second_deriv(z / 4.0)
    )


def _tanh2_invcombo_ladder_second_deriv(z: torch.Tensor) -> torch.Tensor:
    """Stronger four-scale signed curvature ladder with geometric weights."""
    return (
        _tanh2_second_deriv(z)
        + 0.5 * _tanh2_second_deriv(z / 2.0)
        + 0.25 * _tanh2_second_deriv(z / 4.0)
        + 0.125 * _tanh2_second_deriv(z / 8.0)
    )


def _tanh2_third_deriv(z: torch.Tensor) -> torch.Tensor:
    """Third derivative of tanh(s)^2; odd companion to tanh²''."""
    t = torch.tanh(z)
    u = 1.0 - t * t
    return 8.0 * t * u * (3.0 * t * t - 2.0)


def _tanh2_invcombo_ladder_third_deriv(z: torch.Tensor) -> torch.Tensor:
    """Chain-rule matched odd companion to the four-scale curvature ladder."""
    return (
        _tanh2_third_deriv(z)
        + 0.5 * 0.5 * _tanh2_third_deriv(z / 2.0)
        + 0.25 * 0.25 * _tanh2_third_deriv(z / 4.0)
        + 0.125 * 0.125 * _tanh2_third_deriv(z / 8.0)
    )


def _tanh2_invcombo_ladder_odd_second_deriv(
    z: torch.Tensor, odd_scale: torch.Tensor | float
) -> torch.Tensor:
    """Even signed-curvature ladder plus a small同源 odd third-derivative branch."""
    return _tanh2_invcombo_ladder_second_deriv(z) + odd_scale * _tanh2_invcombo_ladder_third_deriv(z)


def _tanh2_invcombo_ladder_oddgate_second_deriv(
    z: torch.Tensor, gate_scale: float = 0.10
) -> torch.Tensor:
    """Cheap odd/even mix: strong even ladder modulated by a small odd tanh gate."""
    t = torch.tanh(z)
    s2 = (1.0 - t * t).clamp(min=1e-8)
    base = 2.0 * s2 * s2 - 4.0 * t * t * s2
    ladder = (
        base
        + 0.5 * _tanh2_second_deriv(z / 2.0)
        + 0.25 * _tanh2_second_deriv(z / 4.0)
        + 0.125 * _tanh2_second_deriv(z / 8.0)
    )
    return ladder * (1.0 + gate_scale * t)


def _tanh2_invcombo_ladder_hot_oddgate_second_deriv(
    z: torch.Tensor,
    *,
    gate_scale: float,
    gate_gain: float,
    amp: float = 1.0,
) -> torch.Tensor:
    """Four-scale ladder with a stronger odd gate near the small-s_z regime."""
    return amp * _tanh2_invcombo_ladder_second_deriv(z) * (
        1.0 + gate_scale * torch.tanh(gate_gain * z)
    )


def _ladder_add_odd_tanh(s_z: torch.Tensor, odd_scale: float) -> torch.Tensor:
    """Four-scale even ladder plus a cheap additive odd branch on s_z."""
    return _tanh2_invcombo_ladder_second_deriv(s_z) + odd_scale * torch.tanh(s_z)


def _ladder_add_odd_delta(
    s_z: torch.Tensor, delta: torch.Tensor, odd_scale: float
) -> torch.Tensor:
    """Even ladder on global s_z plus additive odd from local-global deviation."""
    return _tanh2_invcombo_ladder_second_deriv(s_z) + odd_scale * torch.tanh(delta)


def _ladder_mix_add_odd_delta(
    s_z: torch.Tensor,
    s_x: torch.Tensor,
    *,
    sx_mix: float,
    odd_scale: float,
) -> torch.Tensor:
    """Even ladder on mixed s, plus additive odd from s_x - s_z."""
    s_eff = (1.0 - sx_mix) * s_z + sx_mix * s_x
    return _ladder_add_odd_delta(s_eff, s_x - s_z, odd_scale)


def _tanh2_second_and_third_from_tanh_arg(
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tanh²''(z) and its same-source odd companion tanh²'''(z)."""
    t = torch.tanh(z)
    s2 = (1.0 - t * t).clamp(min=1e-8)
    second = 2.0 * s2 * s2 - 4.0 * t * t * s2
    third = 2.0 * s2 * (-8.0 * t + 12.0 * t * t * t)
    return second, third


def _tanh2_invcombo_ladder_matched_odd_second_deriv(
    z: torch.Tensor,
    *,
    depth: int,
    odd_scale: float,
) -> torch.Tensor:
    """Geometric ladder where each even curvature rung gets a matched odd term.

    The odd term uses the chain-rule factor for each rung:
    d/ds [w_i * tanh²''(s/a_i)] = (w_i / a_i) * tanh²'''(s/a_i).
    """
    lam: torch.Tensor | float = 0.0
    for i in range(int(depth)):
        inv_temp = 1.0 / float(2**i)
        weight = inv_temp
        even, odd = _tanh2_second_and_third_from_tanh_arg(z * inv_temp)
        lam = lam + weight * even + odd_scale * weight * inv_temp * odd
    return lam


def _tanh2_invcombo_ladder_rung_oddgate_second_deriv(
    z: torch.Tensor,
    *,
    depth: int,
    gate_scale: float,
) -> torch.Tensor:
    """Cheap per-rung odd/even mix using the same scaled tanh argument.

    This keeps the mechanism of "each ladder rung gets an odd companion" while
    avoiding the extra cubic path from the exact third derivative.
    """
    lam: torch.Tensor | float = 0.0
    for i in range(int(depth)):
        inv_temp = 1.0 / float(2**i)
        weight = inv_temp
        t = torch.tanh(z * inv_temp)
        s2 = (1.0 - t * t).clamp(min=1e-8)
        even = 2.0 * s2 * s2 - 4.0 * t * t * s2
        lam = lam + weight * even * (1.0 + gate_scale * t)
    return lam


def diagnose_lambda_distribution(
    s_z: torch.Tensor, prefix: str = ""
) -> dict[str, torch.Tensor]:
    """Diagnose |s_z| distribution to check for saturation.

    Returns metrics indicating what fraction of units are in:
    - |s| < 1: high curvature zone (λ near peak)
    - 1 < |s| < 2: transition zone
    - |s| > 2: saturation zone (λ ≈ 0)
    """
    abs_s = s_z.abs()
    total = s_z.numel()

    metrics = {
        f"{prefix}|s_z|_mean": abs_s.mean(),
        f"{prefix}|s_z|_max": abs_s.max(),
        f"{prefix}|s_z|_std": abs_s.std(),
        f"{prefix}|s_z|<1": (abs_s < 1).float().mean(),
        f"{prefix}1<|s_z|<2": ((abs_s > 1) & (abs_s < 2)).float().mean(),
        f"{prefix}|s_z|>2": (abs_s > 2).float().mean(),
        f"{prefix}|s_z|>3": (abs_s > 3).float().mean(),
    }

    # Compute actual lambda values for tanh2 and softplus
    lam_tanh = _tanh2_second_deriv(s_z)
    lam_softplus = _softplus_second_deriv(s_z)

    metrics[f"{prefix}lam_tanh_mean"] = lam_tanh.abs().mean()
    metrics[f"{prefix}lam_tanh_active"] = (lam_tanh.abs() > 0.1).float().mean()
    metrics[f"{prefix}lam_softplus_mean"] = lam_softplus.mean()
    metrics[f"{prefix}lam_softplus_active"] = (lam_softplus > 0.1).float().mean()

    return metrics


def _tanh2_second_deriv_eps(z: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """tanh²''(z) + ε floor to avoid large-|s| saturation."""
    return _tanh2_second_deriv(z) + eps


def _tanh2_first_deriv(z: torch.Tensor) -> torch.Tensor:
    """First derivative of tanh(z)^2: 2 tanh(z) sech²(z)."""
    t = torch.tanh(z)
    s2 = (1.0 - t * t).clamp(min=1e-8)
    return 2.0 * t * s2


def _tanh2_third_deriv(z: torch.Tensor) -> torch.Tensor:
    """Third derivative of tanh(z)^2: 2 sech²(z)(-8 tanh(z) + 12 tanh³(z))."""
    t = torch.tanh(z)
    s2 = (1.0 - t * t).clamp(min=1e-8)
    return 2.0 * s2 * (-8.0 * t + 12.0 * t * t * t)


def _lam_ms_d1(s_z: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """ψ'' + βψ' fallback for saturated regions."""
    return _tanh2_second_deriv(s_z) + beta * _tanh2_first_deriv(s_z)


def _lam_ms_regulated(
    s_z: torch.Tensor,
    beta: torch.Tensor,
    sharp_logit: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Trust ψ'' when |ψ'''| is small; else blend in |ψ'| floor."""
    lam2 = _tanh2_second_deriv(s_z)
    lam1 = _tanh2_first_deriv(s_z)
    g = torch.sigmoid(-sharp_logit * (_tanh2_third_deriv(s_z).abs() - tau))
    lam_fb = lam2 + beta * lam1.abs()
    return g * lam2 + (1.0 - g) * lam_fb


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square norm over the last dim. x: (..., D)."""
    return x * x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()


def _causal_prefix_mean(x: torch.Tensor) -> torch.Tensor:
    """Causal prefix mean z_t = mean(x_0..x_t). x: (B, T, D)."""
    cum = torch.cumsum(x, dim=1)
    t = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype)
    return cum / t.view(1, -1, 1)


def _causal_prefix_std(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Causal prefix std per dim. x: (B, T, D)."""
    cum = torch.cumsum(x, dim=1)
    cum2 = torch.cumsum(x * x, dim=1)
    t = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
    mean = cum / t
    var = cum2 / t - mean * mean
    return torch.sqrt(var.clamp(min=eps))


def _causal_window_mean(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """z_t = mean(x_{max(0,t-W+1)}..x_t). x: (B, T, D)."""
    w = max(1, int(window_size))
    cum = torch.cumsum(x, dim=1)
    t_len = x.size(1)
    idx = torch.arange(t_len, device=x.device)
    start = (idx - w + 1).clamp(min=0)
    zeros = x.new_zeros(x.size(0), 1, x.size(2))
    cum_ext = torch.cat([zeros, cum], dim=1)
    win_sum = cum - cum_ext[:, start]
    count = (idx - start + 1).to(dtype=x.dtype).view(1, -1, 1)
    return win_sum / count


def _global_seq_mean(x: torch.Tensor) -> torch.Tensor:
    """Full-sequence mean broadcast to each token. x: (B, T, D)."""
    return x.mean(dim=1, keepdim=True).expand_as(x)


def _symmetric_window_mean(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Centered symmetric window mean (non-causal). x: (B, T, D)."""
    return _symmetric_window_mean_cumsum(x, window_size)


def _symmetric_window_mean_cumsum(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Centered symmetric window mean via cumsum (faster than avg_pool1d on MPS)."""
    w = max(1, int(window_size))
    t_len = x.size(1)
    if w >= t_len:
        return _global_seq_mean(x)
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    x_pad = F.pad(x, (0, 0, pad_left, pad_right), mode="replicate")
    cum = torch.cumsum(x_pad, dim=1)
    c0 = torch.cat([x.new_zeros(x.size(0), 1, x.size(2)), cum], dim=1)
    idx = torch.arange(t_len, device=x.device)
    win_sum = c0[:, idx + w] - c0[:, idx]
    return win_sum / float(w)


def _causal_window_attn_pool(
    x: torch.Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    window_size: int,
) -> torch.Tensor:
    """Causal attention pool; band mask limits keys to last W tokens."""
    w = max(1, min(int(window_size), x.size(1)))
    q = q_proj(x)
    k = k_proj(x)
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    t_len = x.size(1)
    idx = torch.arange(t_len, device=x.device)
    lag = idx.unsqueeze(1) - idx.unsqueeze(0)
    valid = (lag >= 0) & (lag < w)
    scores = scores.masked_fill(~valid, float("-inf"))
    alpha = F.softmax(scores, dim=-1)
    return torch.matmul(alpha, x)


def _causal_multihead_qkv(
    x: torch.Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    out_proj: nn.Linear | None,
    n_head: int,
) -> torch.Tensor:
    """Full causal multi-head QKV: softmax(QK^T)V, then optional out_proj."""
    bsz, t_len, dim = x.shape
    heads = max(1, int(n_head))
    if dim % heads != 0:
        raise ValueError(f"dim {dim} not divisible by n_head {heads}")
    hs = dim // heads
    q = q_proj(x).view(bsz, t_len, heads, hs).transpose(1, 2)
    k = k_proj(x).view(bsz, t_len, heads, hs).transpose(1, 2)
    v = v_proj(x).view(bsz, t_len, heads, hs).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (hs ** -0.5)
    idx = torch.arange(t_len, device=x.device)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
    scores = scores.masked_fill(~causal, float("-inf"))
    alpha = F.softmax(scores, dim=-1)
    y = torch.matmul(alpha, v).transpose(1, 2).contiguous().view(bsz, t_len, dim)
    if out_proj is not None:
        y = out_proj(y)
    return y


def _causal_temporal_smooth(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """x̃_t = (1-α)x_t + α x_{t-1}; x̃_0 = x_0. x: (B, T, D)."""
    a = alpha.clamp(0.0, 1.0)
    prev = torch.zeros_like(x)
    if x.size(1) > 1:
        prev[:, 1:] = x[:, :-1]
    return (1.0 - a) * x + a * prev


def _causal_ema(x: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """z_t = ρ z_{t-1} + (1-ρ) x_t; z_0 = x_0. x: (B, T, D).

    Closed form via lower-triangular weights; on MPS this beats Python-loop EMA.
    """
    r = rho.clamp(0.0, 0.999)
    t_len = x.size(1)
    idx = torch.arange(t_len, device=x.device, dtype=x.dtype)
    lag = idx.unsqueeze(1) - idx.unsqueeze(0)
    mask = lag >= 0
    powers = torch.pow(r, lag.clamp(min=0)) * mask
    weights = (1.0 - r) * powers
    weights[:, 0] = powers[:, 0]
    return torch.einsum("tk,bkd->btd", weights, x)


class _RankSpectralCore(nn.Module):
    """Form A: static rank-r symmetric map W = sum_j lam_j u_j u_j^T."""

    def __init__(self, dim: int, *, rank: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.rank = int(rank) if rank is not None else 4 * dim
        self.directions = nn.Parameter(torch.randn(self.rank, dim) * 0.02)
        self.lam = nn.Parameter(torch.full((self.rank,), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coef = x @ self.directions.transpose(-2, -1)
        return (coef * self.lam) @ self.directions


class _PhiSpectralCore(nn.Module):
    """Sum_j tanh(w_j^T x + b_j)^2 Hessian map with m learnable units."""

    def __init__(self, dim: int, *, num_units: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.weight.transpose(-2, -1)
        s = proj + self.bias
        lam = _tanh2_second_deriv(s)
        return torch.einsum("...m,md->...d", lam * proj, self.weight)


class _PhiPrefixSpectralCore(nn.Module):
    """H4_phi with lambda_j from causal prefix z_t, map still uses x_t."""

    def __init__(self, dim: int, *, num_units: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = _causal_prefix_mean(x)
        proj_x = x @ self.weight.transpose(-2, -1)
        s_z = z @ self.weight.transpose(-2, -1) + self.bias
        lam = _tanh2_second_deriv(s_z)
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiPrefixWindowSpectralCore(nn.Module):
    """H4_phi: λ from causal sliding-window mean z_t, map still uses x_t."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = _causal_window_mean(x, self.window_size)
        proj_x = x @ self.weight.transpose(-2, -1)
        s_z = z @ self.weight.transpose(-2, -1) + self.bias
        lam = _tanh2_second_deriv(s_z)
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiPrefixAttnSpectralCore(nn.Module):
    """H4_phi: λ from windowed causal attention pool z_t, map still uses x_t."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pool_dim = int(pool_dim)
        self.window_size = int(window_size)
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        self.q_proj = nn.Linear(dim, self.pool_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.pool_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = _causal_window_attn_pool(x, self.q_proj, self.k_proj, self.window_size)
        proj_x = x @ self.weight.transpose(-2, -1)
        s_z = z @ self.weight.transpose(-2, -1) + self.bias
        lam = _tanh2_second_deriv(s_z)
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiPrefixAttnEpsSpectralCore(nn.Module):
    """prefix_attn + λ = tanh²''(s) + ε; outer activation uses Swish in HessianFFN."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        lam_eps_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pool_dim = int(pool_dim)
        self.window_size = int(window_size)
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        self.q_proj = nn.Linear(dim, self.pool_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.pool_dim, bias=False)
        self.lam_eps = nn.Parameter(torch.tensor(float(lam_eps_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = _causal_window_attn_pool(x, self.q_proj, self.k_proj, self.window_size)
        proj_x = x @ self.weight.transpose(-2, -1)
        s_z = z @ self.weight.transpose(-2, -1) + self.bias
        eps = self.lam_eps.clamp(min=1e-6)
        lam = _tanh2_second_deriv_eps(s_z, eps)
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiPrefixMultiScaleCore(nn.Module):
    """Multi-scale z for λ; blend of sharp x_t and high-pass x_t-z_short for proj.

    λ path: z = fuse(z_attn, z_win, z_long) with learnable weights, then tanh²''(s_z).
    proj path: γ·x + (1-γ)·(x - z_win) with learnable γ (init → mostly x_t).
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pool_dim = int(pool_dim)
        self.window_size = int(window_size)
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        self.q_proj = nn.Linear(dim, self.pool_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.pool_dim, bias=False)
        # init: favor attn > win > long prefix mean
        self.scale_logits = nn.Parameter(torch.tensor([2.0, 0.5, -1.0]))
        # init γ≈0.88 → proj mostly x_t, partial high-pass
        self.hp_logit = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_attn = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win = _causal_window_mean(x, self.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = (
            scale_w[0] * z_attn
            + scale_w[1] * z_win
            + scale_w[2] * z_long
        )
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        lam = _tanh2_second_deriv(s_z)
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiPrefixMsDecoupleCore(_PhiPrefixMultiScaleCore):
    """Prefix-MS context (attn + window + long) with asymmetric U/V decouple.

    Simplified LM-Track: same z/projection path as ``h4_phi768_prefix_ms`` but
    output uses independent ``out_weight`` (V) instead of co-axial U.
    No multwin / invcombo / ladder stack.

    ``__init__`` sets ``V←U`` for scratch step~0 coaxial tie only.  Pretrain swap
    overwrites with ``U←W1`` then ``V←W2`` via ``apply_analytic_translation``.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            pool_dim=pool_dim,
            window_size=window_size,
        )
        self.out_weight = nn.Parameter(torch.empty_like(self.weight))
        with torch.no_grad():
            self.out_weight.copy_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_attn = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win = _causal_window_mean(x, self.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = scale_w[0] * z_attn + scale_w[1] * z_win + scale_w[2] * z_long
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w_in = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w_in
        s_z = z @ w_in + self.bias
        lam = _tanh2_second_deriv(s_z)
        gated = lam * proj_x
        if self.training:
            with torch.no_grad():
                self.last_diag = {
                    "s_z_abs_mean": float(s_z.abs().mean()),
                    "lam_near_zero_frac": float((lam < 1e-3).float().mean()),
                }
            if getattr(self, "capture_factor_lr", False):
                gated.retain_grad()
                self._factor_gated = gated
                self._factor_cache = {
                    "proj_in": proj_in.detach(),
                    "z": z.detach(),
                    "s_z": s_z.detach(),
                    "lam": lam.detach(),
                    "proj_x": proj_x.detach(),
                    "weight_shape": self.weight.shape,
                }
        return gated @ self.out_weight


class _PhiPrefixMsDecoupleCtxSplitCore(_PhiPrefixMsDecoupleCore):
    """Decouple content, context routing, and output into U, A, and V.

    Forward order: ``proj_x = x @ U`` → ``λ = ψ''(z @ A)`` → ``out = (λ ⊙ proj_x) @ V``.

    ``weight`` (U) learns only the content projection, ``context_weight`` (A)
    learns how aggregate context controls the Hessian gate, and ``out_weight``
    (V) learns the output map.

    **Scratch / GPT-2 ctxsplit FT**: ``A←U``, ``V←U`` so step~0 is the
    symmetric operator.  Training uses official ``λ = ψ''(z @ A + b_A)``.

    **Pretrain warm-start**: ``U←W1``, ``V←W2`` only (same as
    ``global_decouple_min``).  ``s = U x + b`` is that translation; ``λ`` is
    not translated — native ``ψ''(s)``.  A is stored from U; it does not
    replace ``s`` in the gate.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        lambda_geometry: DecoupleMinLambda = "global_tanh2",
        native_mix_init: float = -10.0,
        tanh2_temperature: float = 1.0,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            pool_dim=pool_dim,
            window_size=window_size,
        )
        self.lambda_geometry = lambda_geometry
        self.context_weight = nn.Parameter(torch.empty_like(self.weight))
        # Keep the MLP expansion/output biases separate from the context→λ bias.
        # Scratch training starts them at zero; pretrained transfer can inherit
        # b1/b2 without overloading ``bias`` on the A leg.
        self.content_bias = nn.Parameter(torch.zeros(self.num_units))
        self.out_bias = nn.Parameter(torch.zeros(dim))
        with torch.no_grad():
            self.context_weight.copy_(self.weight)
        # Learnable oracle×native mix for mod geometries; init ≈0 (σ(-10)≈0).
        self.native_mix_logit = nn.Parameter(torch.tensor(float(native_mix_init)))
        self.tanh2_temperature = float(tanh2_temperature)
        self.tanh2_amplitude = "paper"
        self.tanh2_amplitude_scale = 1.0
        self.tanh2_peak_centers: tuple[float, ...] | None = None
        self.tanh2_peak_shift = 0.0
        # Ortho Hessian leg only: ψ''(s/τ) inside δ. τ=1 is the previous native factor.
        self.hessian_factor_temperature = 1.0
        # After UV warm-start: λ = ψ''(s) on content s, not z @ A.
        self.lambda_from_content_s: bool = False
        self.lambda_on_content_s: bool = False
        self.phi_init_content_gate: bool = False
        self.mod_native_uv_only: bool = False
        self.geom_diag_enabled: bool = False

    def _uses_mod_native_ua_split(self) -> bool:
        geom = getattr(self, "lambda_geometry", "global_tanh2")
        return geom in (
            "proj_oracle_mod_native_simple",
            "proj_oracle_mod_native_ortho",
        )

    def _resolve_lambda(
        self, s: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Resolve λ; mod-native uses oracle on content ``s`` and native on A-leg ``gate``."""
        from tlb.ffn.mlp_hessian_analytic import (
            lambda_decouple_geometry,
            lambda_oracle_mod_native_ortho,
            lambda_oracle_mod_native_simple,
        )

        geom = getattr(self, "lambda_geometry", "global_tanh2")
        if getattr(self, "phi_init_content_gate", False):
            mix: float | torch.Tensor = 0.0
        else:
            mix = torch.sigmoid(self.native_mix_logit)
        if geom == "proj_oracle_mod_native_simple":
            return lambda_oracle_mod_native_simple(s, mix=mix, gate=gate)
        if geom == "proj_oracle_mod_native_ortho":
            return lambda_oracle_mod_native_ortho(
                s,
                mix=mix,
                gate=gate,
                temperature=float(getattr(self, "hessian_factor_temperature", 1.0)),
            )
        if geom == "global_tanh2":
            return tanh2_softened(
                s,
                float(getattr(self, "tanh2_temperature", 1.0)),
                amplitude=str(getattr(self, "tanh2_amplitude", "paper")),
                amplitude_scale=float(getattr(self, "tanh2_amplitude_scale", 1.0)),
                peak_centers=getattr(self, "tanh2_peak_centers", None),
                peak_shift=float(getattr(self, "tanh2_peak_shift", 0.0)),
            )
        return lambda_decouple_geometry(s, geom)

    def _context_z(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(z, z_win)`` used by ``λ = ψ''(z @ A)`` and the high-pass mix."""
        if getattr(self, "use_full_qkv", False):
            z = _causal_multihead_qkv(
                x,
                self.q_proj,
                self.k_proj,
                self.v_proj,
                getattr(self, "z_out_proj", None),
                int(getattr(self, "n_head", 1)),
            )
            z_win = _causal_window_mean(x, self.window_size)
            return z, z_win
        z_attn = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win = _causal_window_mean(x, self.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = scale_w[0] * z_attn + scale_w[1] * z_win + scale_w[2] * z_long
        return z, z_win

    def native_lambda(self, x: torch.Tensor) -> torch.Tensor:
        """Paper gate ``λ = ψ''(z @ A + b)``.  Not GELU/s."""
        lam, _proj_x, _s_z = self.native_factors(x)
        return lam

    def native_factors(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return native ``(λ, proj_x, s_gate)``.

        Mod-native (U/A split): ``Φ`` anchors on ``s=Ux+b``; ``δ`` or ``g_native`` acts
        on ``g=A x̃+b`` with ``x̃=proj_in(x)`` (no ``z@A``).  Step-0 swap uses ``x̃=x``.
        Scratch ctxsplit uses ``λ=ψ''(z @ A)`` on prefix-ms ``z``.
        """
        s_content = x @ self.weight.transpose(-2, -1) + self.content_bias
        if self._uses_mod_native_ua_split():
            if getattr(self, "phi_init_content_gate", False):
                g_a = x @ self.context_weight.transpose(-2, -1) + self.bias
                return self._resolve_lambda(s_content, g_a), s_content, g_a
            if getattr(self, "mod_native_uv_only", False):
                return self._resolve_lambda(s_content), s_content, s_content
            z, z_win = self._context_z(x)
            gamma = torch.sigmoid(self.hp_logit)
            proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
            proj_x = proj_in @ self.weight.transpose(-2, -1) + self.content_bias
            g_a = proj_in @ self.context_weight.transpose(-2, -1) + self.bias
            return self._resolve_lambda(s_content, g_a), proj_x, g_a
        if getattr(self, "phi_init_content_gate", False):
            return self._resolve_lambda(s_content), s_content, s_content
        z, z_win = self._context_z(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        proj_x = proj_in @ self.weight.transpose(-2, -1) + self.content_bias
        use_content_lambda = bool(
            getattr(self, "lambda_on_content_s", False)
            or getattr(self, "lambda_from_content_s", False)
        )
        if use_content_lambda:
            s_gate = s_content
        else:
            s_gate = z @ self.context_weight.transpose(-2, -1) + self.bias
        return self._resolve_lambda(s_gate), proj_x, s_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam, proj_x, s_z = self.native_factors(x)
        gated = lam * proj_x
        if self.training or getattr(self, "geom_diag_enabled", False):
            with torch.no_grad():
                u_norm = self.weight.norm(dim=1).clamp(min=1e-8)
                v_norm = self.out_weight.norm(dim=1).clamp(min=1e-8)
                uv_cos = (
                    (self.weight * self.out_weight).sum(dim=1) / (u_norm * v_norm)
                ).mean()
                feat = gated @ self.out_weight
                feat_coax = gated @ self.weight
                delta = feat - feat_coax
                coax_rms = float(feat_coax.pow(2).mean().sqrt().item())
                delta_rms = float(delta.pow(2).mean().sqrt().item())
                s_flat = proj_x.reshape(-1)
                g_flat = s_z.reshape(-1)
                lam_flat = lam.reshape(-1)
                if s_flat.numel() > 1:
                    s_c = s_flat - s_flat.mean()
                    g_c = g_flat - g_flat.mean()
                    sg_corr = float(
                        (s_c * g_c).sum()
                        / (s_c.pow(2).sum().sqrt() * g_c.pow(2).sum().sqrt() + 1e-8)
                    )
                    l_c = lam_flat - lam_flat.mean()
                    ls_corr = float(
                        (l_c * s_c).sum()
                        / (l_c.pow(2).sum().sqrt() * s_c.pow(2).sum().sqrt() + 1e-8)
                    )
                else:
                    sg_corr = float("nan")
                    ls_corr = float("nan")
                self.last_diag = {
                    "s_content_abs_mean": float(proj_x.abs().mean()),
                    "s_gate_abs_mean": float(s_z.abs().mean()),
                    "s_z_abs_mean": float(s_z.abs().mean()),
                    "lam_near_zero_frac": float((lam < 1e-3).float().mean()),
                    "lam_frac_negative": float((lam < 0).float().mean()),
                    "lam_rms": float(lam.pow(2).mean().sqrt().item()),
                    "uv_cos_mean": float(uv_cos.item()),
                    "feat_coax_rms": coax_rms,
                    "feat_decouple_delta_rms": delta_rms,
                    "decouple_to_coax_ratio": delta_rms / max(coax_rms, 1e-8),
                    "s_gate_s_content_corr": sg_corr,
                    "lambda_s_content_corr": ls_corr,
                    "active_unit_frac": float((lam.abs() > 1e-3).float().mean()),
                }
        return gated @ self.out_weight + self.out_bias


class _PhiPrefixMsDecoupleCtxSplitModNativeGeluZCore(_PhiPrefixMsDecoupleCtxSplitCore):
    """Mod-native oracle on content ``s=Ux+b``; A-leg gate from GELU-field prefix-ms context.

    Content / readout (mod-native):
      ``s = x @ U + b_content``, ``λ = Φ(s)·(1 + mix·δ(g))``, ``out = (λ ⊙ proj_x) @ V``.

    Gate ``g`` (W1–W2 cross term):
      ``g_field = GELU(proj_in @ U + b_content)`` → prefix-ms pools on ``g_field``
      → ``z_dim = z_g @ U`` → ``g = z_dim @ A + b_A``.

    Step-0: ``phi_init_content_gate`` + ``mix=0`` → exact ``Φ(s)`` swap (same as mod-native).
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        lambda_geometry: DecoupleMinLambda = "proj_oracle_mod_native_ortho",
        native_mix_init: float = -10.0,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            pool_dim=pool_dim,
            window_size=window_size,
            lambda_geometry=lambda_geometry,
            native_mix_init=native_mix_init,
        )
        self.mod_native_gelu_z_gate = True
        self.ctx_q_proj = nn.Linear(self.num_units, self.pool_dim, bias=False)
        self.ctx_k_proj = nn.Linear(self.num_units, self.pool_dim, bias=False)

    def _gelu_field_z_dim(
        self, x: torch.Tensor, proj_in: torch.Tensor
    ) -> torch.Tensor:
        """Prefix-ms context on ``GELU(U proj_in + b_content)`` → ``z_dim``."""
        proj_s = proj_in @ self.weight.transpose(-2, -1) + self.content_bias
        g_field = F.gelu(proj_s)
        z_attn_g = _causal_window_attn_pool(
            g_field, self.ctx_q_proj, self.ctx_k_proj, self.window_size
        )
        z_win_g = _causal_window_mean(g_field, self.window_size)
        z_long_g = _causal_prefix_mean(g_field)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z_g = (
            scale_w[0] * z_attn_g
            + scale_w[1] * z_win_g
            + scale_w[2] * z_long_g
        )
        return z_g @ self.weight

    def native_factors(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_content = x @ self.weight.transpose(-2, -1) + self.content_bias
        if getattr(self, "phi_init_content_gate", False):
            g_a = x @ self.context_weight.transpose(-2, -1) + self.bias
            return self._resolve_lambda(s_content, g_a), s_content, g_a
        if getattr(self, "mod_native_uv_only", False):
            return self._resolve_lambda(s_content), s_content, s_content
        _z, z_win = self._context_z(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        proj_x = proj_in @ self.weight.transpose(-2, -1) + self.content_bias
        z_dim = self._gelu_field_z_dim(x, proj_in)
        g_a = z_dim @ self.context_weight.transpose(-2, -1) + self.bias
        return self._resolve_lambda(s_content, g_a), proj_x, g_a


class _PhiPrefixMsDecoupleCtxSplitFullGeluZCore(_PhiPrefixMsDecoupleCtxSplitCore):
    """Full ctxsplit content path + GELU-field z aggregation for ``λ = ψ''(z @ A)``.

    Content (full prefix_ms ctxsplit):
      ``proj_in = γ·x + (1-γ)·(x - z_win_x)`` → ``proj_s = proj_in @ U + b`` → ``g = GELU(proj_s)``

    Context routing (path-2 / GELU aggregate):
      multiscale causal pools on ``g`` → ``z_dim = z_g @ U`` → ``s_z = z_dim @ A + b`` → ``λ = ψ''(s_z)``

    Readout: ``(λ ⊙ g) @ V``.  ``ctx_lambda_mix_logit`` init ≈ -4 blends ``λ→1`` at swap so
    ``g @ V`` matches MLP while the tanh²(z@A) leg remains in-graph for training.
    """

    context_weight_zero_init: bool = True

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        ctx_lambda_mix_init: float = -4.0,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            pool_dim=pool_dim,
            window_size=window_size,
        )
        self.ctx_q_proj = nn.Linear(self.num_units, self.pool_dim, bias=False)
        self.ctx_k_proj = nn.Linear(self.num_units, self.pool_dim, bias=False)
        self.ctx_lambda_mix_logit = nn.Parameter(
            torch.tensor(float(ctx_lambda_mix_init))
        )

    def _gelu_z_gated(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        z_attn_x = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win_x = _causal_window_mean(x, self.window_size)
        z_long_x = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win_x)
        w_in = self.weight.transpose(-2, -1)
        proj_s = proj_in @ w_in + self.bias
        g = F.gelu(proj_s)
        z_attn_g = _causal_window_attn_pool(
            g, self.ctx_q_proj, self.ctx_k_proj, self.window_size
        )
        z_win_g = _causal_window_mean(g, self.window_size)
        z_long_g = _causal_prefix_mean(g)
        z_g = (
            scale_w[0] * z_attn_g
            + scale_w[1] * z_win_g
            + scale_w[2] * z_long_g
        )
        z_dim = z_g @ self.weight
        s_z = z_dim @ self.context_weight.transpose(-2, -1) + self.bias
        lam_ctx = _tanh2_second_deriv(s_z)
        mix = torch.sigmoid(self.ctx_lambda_mix_logit)
        lam = (1.0 - mix) + mix * lam_ctx
        gated = lam * g
        diag = {
            "g_abs_mean": float(g.abs().mean()),
            "s_z_abs_mean": float(s_z.abs().mean()),
            "lam_near_zero_frac": float((lam_ctx < 1e-3).float().mean()),
            "lam_mean": float(lam.mean()),
            "ctx_lambda_mix": float(mix.item()),
        }
        return gated, lam, diag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated, lam, diag = self._gelu_z_gated(x)
        if self.training:
            with torch.no_grad():
                self.last_diag = diag
        return gated @ self.out_weight


class _PhiPrefixMsDecoupleCtxSplitFullGeluZActiveRankCore(
    _PhiPrefixMsDecoupleCtxSplitFullGeluZCore
):
    """Full gelu-z ctxsplit + token-level active-rank grad mask on |λ|."""

    token_rank_enabled: bool = False
    token_rank_inverse: bool = False
    token_rank_gate: str = "uv"
    token_rank_keep_frac: float = 1.0
    token_rank_inactive_scale: float = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated, lam, diag = self._gelu_z_gated(x)
        if self.training:
            with torch.no_grad():
                self.last_diag = diag
        out_bias = torch.zeros(self.dim, device=gated.device, dtype=gated.dtype)
        return _token_active_rank_readout(
            self,
            gated=gated,
            lam=lam,
            out_bias=out_bias,
        )


class _PhiPrefixMsDecoupleCtxSplitXTanhCore(_PhiPrefixMsDecoupleCore):
    """Ctxsplit bridge: raw ``x @ U`` content leg, ``λ = tanh²''(proj_x)``, A via mild resonance.

    Skips hp / z-win mixing on the expansion input.  ``resonance_logit`` init ≈ -4 keeps
    step~0 dominated by the content gate while ``context_weight`` (A) trains under
    context@4 LR during GPT-2 multi-layer FT.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        resonance_init: float = -4.0,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            pool_dim=pool_dim,
            window_size=window_size,
        )
        self.context_weight = nn.Parameter(torch.empty_like(self.weight))
        with torch.no_grad():
            self.context_weight.copy_(self.weight)
        self.resonance_logit = nn.Parameter(torch.tensor(float(resonance_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_attn = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win = _causal_window_mean(x, self.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = scale_w[0] * z_attn + scale_w[1] * z_win + scale_w[2] * z_long
        w_in = self.weight.transpose(-2, -1)
        proj_x = x @ w_in + self.bias
        lam = _tanh2_second_deriv(proj_x)
        a_hidden = z @ self.context_weight.transpose(-2, -1)
        alpha = torch.sigmoid(self.resonance_logit)
        gated = lam * (proj_x + alpha * a_hidden)
        if self.training:
            with torch.no_grad():
                self.last_diag = {
                    "proj_x_abs_mean": float(proj_x.abs().mean()),
                    "lam_near_zero_frac": float((lam < 1e-3).float().mean()),
                    "resonance_alpha": float(alpha.item()),
                }
        return gated @ self.out_weight


class _PhiPrefixMsDecoupleCtxSplitGeluAggCore(_PhiPrefixMsDecoupleCore):
    """Ctxsplit bridge: per-token GELU, then prefix-MS z-style aggregation for ``λ(z_g @ A)``.

    ``g = GELU(x @ U + b)`` aligns the hidden with the native MLP activation; causal
    multiscale pools operate on ``g`` (not raw ``x``) before the context routing leg.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            pool_dim=pool_dim,
            window_size=window_size,
        )
        self.context_weight = nn.Parameter(torch.empty_like(self.weight))
        with torch.no_grad():
            self.context_weight.copy_(self.weight)
        # Pool in hidden (unit) space — ``g`` is (…, num_units), not model dim.
        self.ctx_q_proj = nn.Linear(self.num_units, self.pool_dim, bias=False)
        self.ctx_k_proj = nn.Linear(self.num_units, self.pool_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_in = self.weight.transpose(-2, -1)
        proj_x = x @ w_in + self.bias
        g = F.gelu(proj_x)
        z_attn = _causal_window_attn_pool(
            g, self.ctx_q_proj, self.ctx_k_proj, self.window_size
        )
        z_win = _causal_window_mean(g, self.window_size)
        z_long = _causal_prefix_mean(g)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z_g = scale_w[0] * z_attn + scale_w[1] * z_win + scale_w[2] * z_long
        # Hidden-space aggregate → model dim via U, then standard A routing (M×D).
        z_dim = z_g @ self.weight
        s_ctx = z_dim @ self.context_weight.transpose(-2, -1) + self.bias
        lam = _tanh2_second_deriv(s_ctx)
        gated = lam * g
        if self.training:
            with torch.no_grad():
                self.last_diag = {
                    "g_abs_mean": float(g.abs().mean()),
                    "s_ctx_abs_mean": float(s_ctx.abs().mean()),
                    "lam_near_zero_frac": float((lam < 1e-3).float().mean()),
                }
        return gated @ self.out_weight


class _PhiGlobalDecoupleMinABroadcastCore(_PhiPrefixMsDecoupleCtxSplitCore):
    """GPT-2 multi-layer FT: A digests attention z; U/V co-consume shared hidden.

    â = z @ A^T + b  (attention summary from A)
    proj_x = proj_in @ U^T
    proj_shared = proj_x + â   — U and V both see A's hidden (resonance, not coaxial gate)
    λ = oracle(proj_shared)    — MLP-like gate on the shared state
    out = (λ ⊙ proj_shared) @ V + b_out
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        lambda_geometry: DecoupleMinLambda = "proj_oracle_fast",
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            pool_dim=pool_dim,
            window_size=window_size,
        )
        self.lambda_geometry = lambda_geometry
        self.out_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_attn = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win = _causal_window_mean(x, self.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = scale_w[0] * z_attn + scale_w[1] * z_win + scale_w[2] * z_long
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w_in = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w_in
        a_hidden = z @ self.context_weight.transpose(-2, -1) + self.bias
        proj_shared = proj_x + a_hidden
        if self.lambda_geometry in ("proj_oracle", "proj_oracle_fast"):
            from tlb.ffn.mlp_hessian_analytic import lambda_oracle_gelu

            lam = lambda_oracle_gelu(proj_shared)
        elif self.lambda_geometry == "global_oracle":
            from tlb.ffn.mlp_hessian_analytic import lambda_oracle_gelu

            lam = lambda_oracle_gelu(a_hidden)
            if lam.ndim == 2 and proj_shared.ndim == 3:
                lam = lam.unsqueeze(1)
        else:
            lam = _tanh2_second_deriv(a_hidden)
            if lam.ndim == 2 and proj_shared.ndim == 3:
                lam = lam.unsqueeze(1)
        gated = lam * proj_shared
        return gated @ self.out_weight + self.out_bias


class _PhiGlobalDecoupleMinResonanceCore(_PhiPrefixMsDecoupleCtxSplitCore):
    """Scheme-A gated resonance + UA→V training protocol.

    â = z @ A^T                    (context leg, no duplicate bias)
    proj_x = proj_in @ U^T + b     (MLP-aligned expansion; λ anchor)
    α = σ(resonance_logit), init ≈ 0
    proj_shared = proj_x + α · â   (U/V co-consume; scale-controlled)
    λ = oracle(proj_x)             (not oracle(sum))
    out = (λ ⊙ proj_shared) @ V + b_out
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        lambda_geometry: DecoupleMinLambda = "proj_oracle_fast",
        resonance_init: float = -4.0,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            pool_dim=pool_dim,
            window_size=window_size,
        )
        self.lambda_geometry = lambda_geometry
        self.out_bias = nn.Parameter(torch.zeros(dim))
        self.resonance_logit = nn.Parameter(torch.tensor(float(resonance_init)))

    def _lambda_from_proj_x(self, proj_x: torch.Tensor) -> torch.Tensor:
        from tlb.ffn.mlp_hessian_analytic import lambda_decouple_geometry

        return lambda_decouple_geometry(proj_x, self.lambda_geometry)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_attn = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win = _causal_window_mean(x, self.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = scale_w[0] * z_attn + scale_w[1] * z_win + scale_w[2] * z_long
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w_in = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w_in + self.bias
        a_hidden = z @ self.context_weight.transpose(-2, -1)
        alpha = torch.sigmoid(self.resonance_logit)
        proj_shared = proj_x + alpha * a_hidden
        lam = self._lambda_from_proj_x(proj_x)
        gated = lam * proj_shared
        return gated @ self.out_weight + self.out_bias


class _PhiPrefixMsDecoupleSeqCore(_PhiPrefixMsDecoupleCore):
    """Sequential decouple: λ from U only; τ-scaled s_z + mild 2-scale tanh²'' (anti-saturation).

    V (out_weight) is output-only — dual-λ is intentionally not used. Phase LR (flat U,
    tail V) is configured via ``hessian_phase_groups`` in training yaml.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        lam_temp_init: float = 2.0,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            pool_dim=pool_dim,
            window_size=window_size,
        )
        self.lam_temp_log = nn.Parameter(torch.tensor(math.log(float(lam_temp_init))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_attn = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win = _causal_window_mean(x, self.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = scale_w[0] * z_attn + scale_w[1] * z_win + scale_w[2] * z_long
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w_in = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w_in
        s_z = z @ w_in + self.bias
        tau = torch.exp(self.lam_temp_log).clamp(min=0.5, max=8.0)
        s = s_z / tau
        lam = _tanh2_second_deriv(s) + 0.25 * _tanh2_second_deriv(s / 2.0)
        gated = lam * proj_x
        if self.training:
            with torch.no_grad():
                self.last_diag = {
                    "s_z_abs_mean": float(s_z.abs().mean()),
                    "lam_near_zero_frac": float((lam < 1e-3).float().mean()),
                }
            if getattr(self, "capture_factor_lr", False):
                gated.retain_grad()
                self._factor_gated = gated
                self._factor_cache = {
                    "proj_in": proj_in.detach(),
                    "z": z.detach(),
                    "s_z": s_z.detach(),
                    "lam": lam.detach(),
                    "proj_x": proj_x.detach(),
                    "weight_shape": self.weight.shape,
                }
        return gated @ self.out_weight


class _PhiPrefixMsDecoupleDualCore(_PhiPrefixMultiScaleCore):
    """Prefix-MS with dual λ: curvature from U (proj) and V (output) directions.

    Step-0 matches coax when ``out_weight`` is a copy of ``weight`` and
    ``out_unit_bias`` is zero (lam = 0.5 * (lam_u + lam_v) with lam_u == lam_v).
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            pool_dim=pool_dim,
            window_size=window_size,
        )
        self.out_weight = nn.Parameter(torch.empty_like(self.weight))
        self.out_unit_bias = nn.Parameter(torch.zeros(self.num_units))
        self.out_bias = nn.Parameter(torch.zeros(dim))
        with torch.no_grad():
            self.out_weight.copy_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_attn = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win = _causal_window_mean(x, self.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = scale_w[0] * z_attn + scale_w[1] * z_win + scale_w[2] * z_long
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w_in = self.weight.transpose(-2, -1)
        w_out = self.out_weight.transpose(-2, -1)
        proj_x = proj_in @ w_in
        s_u = z @ w_in + self.bias
        s_v = z @ w_out + self.out_unit_bias
        lam_u = _tanh2_second_deriv(s_u)
        lam_v = _tanh2_second_deriv(s_v)
        lam = 0.5 * (lam_u + lam_v)
        return (lam * proj_x) @ self.out_weight + self.out_bias


# Geometric ladder up to block_size=256 (+ prefix in core forward).
MULTWIN_DEEP_WINDOWS: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)


class _PhiPrefixMsMultwinCore(nn.Module):
    """prefix_ms proj; λ z = softmax blend of multi-scale causal window means + prefix."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        # init: anchor at m=768 (6*dim) with std=0.02, scale relatively for other sizes
        base_m = 6 * dim
        init_std = 0.02 * math.sqrt(base_m / self.num_units)
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * init_std)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        n_scales = len(self.window_sizes) + 1
        logits = torch.full((n_scales,), -1.0)
        if self.window_size in self.window_sizes:
            logits[self.window_sizes.index(self.window_size)] = 2.0
        else:
            logits[0] = 2.0
        self.scale_logits = nn.Parameter(logits)
        self.hp_logit = nn.Parameter(torch.tensor(2.0))

    def _fused_z(self, x: torch.Tensor) -> torch.Tensor:
        z, _z_win = self._fused_z_and_window(x)
        return z

    def _fused_z_and_window(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused multi-window z and the primary ``window_size`` mean.

        ``forward`` needs both, and ``window_size`` is usually already present in
        ``window_sizes``. Reusing it avoids an extra cumsum/window-mean pass.
        """
        z_parts = [_causal_window_mean(x, w) for w in self.window_sizes]
        z_win = None
        for w, z_part in zip(self.window_sizes, z_parts):
            if w == self.window_size:
                z_win = z_part
                break
        if z_win is None:
            z_win = _causal_window_mean(x, self.window_size)
        z_parts.append(_causal_prefix_mean(x))
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = sum(sw * zp for sw, zp in zip(scale_w, z_parts))
        return z, z_win

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_win = self._fused_z_and_window(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        lam = _tanh2_second_deriv(s_z)
        return (lam * proj_x) @ self.weight


class _PhiPrefixMsMultwinSoftplusCore(_PhiPrefixMsMultwinCore):
    """multwin with softplus''(s_z) instead of tanh²''(s_z).

    softplus'' is always positive, smooth, avoids hard saturation of tanh''.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_win = self._fused_z_and_window(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        lam = _softplus_second_deriv(s_z)  # Use softplus instead of tanh2
        return (lam * proj_x) @ self.weight


class _PhiPrefixMsMultwinTempCore(_PhiPrefixMsMultwinCore):
    """multwin with temperature-scaled s_z before computing tanh²''.

    Temperature scaling helps avoid hard saturation by shrinking s_z magnitudes.
    tau is learnable, initialized to 2.0 (divide by 2 by default).
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
        temp_init: float = 2.0,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            window_size=window_size,
            window_sizes=window_sizes,
        )
        # log(tau) parameter for stability; tau = exp(log_tau)
        self.temp_logit = nn.Parameter(torch.tensor(math.log(float(temp_init))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_win = self._fused_z_and_window(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        # Temperature scaling: s_z_scaled = s_z / tau
        tau = torch.exp(self.temp_logit)
        s_z_scaled = s_z / tau.clamp(min=0.5)  # floor at 0.5 for stability
        lam = _tanh2_second_deriv(s_z_scaled)
        return (lam * proj_x) @ self.weight


class _PhiPrefixMsMultwinActivCore(_PhiPrefixMsMultwinCore):
    """multwin with configurable activation function for lambda computation.

    Supports: 'rational', 'shifted_sp', 'mish', 'tanh2' (default)
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
        activ: str = "tanh2",
        activ_param: float | None = None,
        s_z_rms_norm: bool = False,
        s_z_gain_init: float = 1.5,
        s_z_gain_mix: str = "s",
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            window_size=window_size,
            window_sizes=window_sizes,
        )
        self.activ = activ
        self.activ_param = activ_param
        self.s_z_rms_norm = bool(s_z_rms_norm)
        self.s_z_gain_active = True
        self.s_z_gain_alpha = 1.0
        self.s_z_gain_mix = s_z_gain_mix
        if self.s_z_rms_norm:
            self.s_z_gain_log = nn.Parameter(torch.tensor(math.log(float(s_z_gain_init))))

        # Learnable epsilon for rational if not provided
        if activ == "rational":
            # default 0.5, learnable
            param = activ_param if activ_param is not None else 0.5
            self.rational_eps = nn.Parameter(torch.tensor(math.log(param)))
        elif activ == "shifted_sp":
            # default 2.0, learnable
            param = activ_param if activ_param is not None else 2.0
            self.shifted_sp_shift = nn.Parameter(torch.tensor(float(param)))
        elif activ == "invcombo_ladder_odd_learn":
            # β in [0, 0.2], initialized near 0.05.
            init_beta = activ_param if activ_param is not None else 0.05
            init_ratio = min(max(float(init_beta) / 0.2, 1e-4), 1.0 - 1e-4)
            self.odd_beta_logit = nn.Parameter(
                torch.tensor(math.log(init_ratio / (1.0 - init_ratio)))
            )
        self.diag_enabled = False
        self.last_diag: dict[str, float | list[float]] | None = None

    def set_s_z_gain_active(self, active: bool) -> None:
        self.s_z_gain_active = bool(active)
        self.s_z_gain_alpha = 1.0 if active else 0.0

    def set_s_z_gain_alpha(self, alpha: float) -> None:
        self.s_z_gain_alpha = float(max(0.0, min(1.0, alpha)))

    def set_s_z_gain_init(self, gain: float) -> None:
        if self.s_z_rms_norm:
            with torch.no_grad():
                self.s_z_gain_log.fill_(math.log(float(gain)))

    def _compute_lambda(self, s_z: torch.Tensor) -> torch.Tensor:
        """Compute lambda using the specified activation function."""
        if self.activ == "rational":
            eps = torch.exp(self.rational_eps).clamp(min=0.1, max=2.0)
            # Use expanded eps to maintain gradient flow
            return _rational_second_deriv_v2(s_z, eps)
        elif self.activ == "shifted_sp":
            shift = self.shifted_sp_shift.clamp(min=0.5, max=4.0)
            return _shifted_softplus_second_deriv_v2(s_z, shift)
        elif self.activ == "mish":
            return _mish_second_deriv(s_z)
        elif self.activ == "swish":
            return _swish_second_deriv(s_z)
        elif self.activ == "signedfloor":
            return _tanh2_signed_floor_second_deriv(s_z)
        elif self.activ == "tanhsp":
            return _tanh2_softplus_blend_second_deriv(s_z)
        elif self.activ == "invcombo":
            return _tanh2_invcombo_second_deriv(s_z)
        elif self.activ == "invcombo_strong":
            return _tanh2_invcombo_strong_second_deriv(s_z)
        elif self.activ == "invcombo_deep":
            return _tanh2_invcombo_deep_second_deriv(s_z)
        elif self.activ == "invcombo_ladder":
            return _tanh2_invcombo_ladder_second_deriv(s_z)
        elif self.activ == "invcombo_ladder_odd_small":
            return _tanh2_invcombo_ladder_odd_second_deriv(s_z, 0.05)
        elif self.activ == "invcombo_ladder_odd_mid":
            return _tanh2_invcombo_ladder_odd_second_deriv(s_z, 0.10)
        elif self.activ == "invcombo_ladder_odd_learn":
            beta = 0.2 * torch.sigmoid(self.odd_beta_logit)
            return _tanh2_invcombo_ladder_odd_second_deriv(s_z, beta)
        elif self.activ == "invcombo_ladder_oddgate":
            return _tanh2_invcombo_ladder_oddgate_second_deriv(s_z, 0.10)
        elif self.activ == "invcombo_ladder_oddgate_m50":
            return _tanh2_invcombo_ladder_hot_oddgate_second_deriv(
                s_z, gate_scale=0.50, gate_gain=1.0
            )
        elif self.activ == "invcombo_ladder_oddgate_hot_m50":
            return _tanh2_invcombo_ladder_hot_oddgate_second_deriv(
                s_z, gate_scale=0.50, gate_gain=4.0
            )
        elif self.activ == "invcombo_ladder_oddgate_amp2_hot_m50":
            return _tanh2_invcombo_ladder_hot_oddgate_second_deriv(
                s_z, gate_scale=0.50, gate_gain=4.0, amp=2.0
            )
        elif self.activ == "invcombo_ladder6_odd_m10":
            return _tanh2_invcombo_ladder_rung_oddgate_second_deriv(
                s_z, depth=6, gate_scale=0.10
            )
        elif self.activ == "invcombo_ladder6_odd_m20":
            return _tanh2_invcombo_ladder_rung_oddgate_second_deriv(
                s_z, depth=6, gate_scale=0.20
            )
        elif self.activ == "invcombo_ladder6_odd_m35":
            return _tanh2_invcombo_ladder_rung_oddgate_second_deriv(
                s_z, depth=6, gate_scale=0.35
            )
        elif self.activ == "invcombo_ladder8_odd_m20":
            return _tanh2_invcombo_ladder_rung_oddgate_second_deriv(
                s_z, depth=8, gate_scale=0.20
            )
        elif self.activ == "softplus":
            return _softplus_second_deriv(s_z)
        else:  # default tanh2
            return _tanh2_second_deriv(s_z)

    def _prepare_s_z(self, s_z_raw: torch.Tensor) -> torch.Tensor:
        """RMS-normalize s_z then apply learnable gain for λ operating range."""
        if not self.s_z_rms_norm or not self.s_z_gain_active:
            return s_z_raw
        alpha = float(self.s_z_gain_alpha)
        if alpha <= 0.0:
            return s_z_raw
        rms = s_z_raw.pow(2).mean().sqrt().clamp(min=1e-6)
        gain = torch.exp(self.s_z_gain_log).clamp(min=0.25, max=8.0)
        s_z_scaled = gain * s_z_raw / rms
        if alpha >= 1.0:
            return s_z_scaled
        return (1.0 - alpha) * s_z_raw + alpha * s_z_scaled

    def _scaled_s_z(self, s_z_raw: torch.Tensor) -> torch.Tensor:
        rms = s_z_raw.pow(2).mean().sqrt().clamp(min=1e-6)
        gain = torch.exp(self.s_z_gain_log).clamp(min=0.25, max=8.0)
        return gain * s_z_raw / rms

    def _lambda_and_diag_s_z(self, s_z_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.s_z_rms_norm or not self.s_z_gain_active:
            return self._compute_lambda(s_z_raw), s_z_raw
        alpha = float(self.s_z_gain_alpha)
        if self.s_z_gain_mix == "lambda":
            lam_raw = self._compute_lambda(s_z_raw)
            if alpha <= 0.0:
                return lam_raw, s_z_raw
            s_z_scaled = self._scaled_s_z(s_z_raw)
            lam_scaled = self._compute_lambda(s_z_scaled)
            if alpha >= 1.0:
                return lam_scaled, s_z_scaled
            return (1.0 - alpha) * lam_raw + alpha * lam_scaled, s_z_scaled
        s_z = self._prepare_s_z(s_z_raw)
        return self._compute_lambda(s_z), s_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_win = self._fused_z_and_window(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z_raw = z @ w + self.bias
        lam, s_z = self._lambda_and_diag_s_z(s_z_raw)
        if self.diag_enabled:
            with torch.no_grad():
                feat = lam * proj_x
                scale_w = F.softmax(self.scale_logits, dim=0)
                diag: dict[str, float | list[float]] = {
                    "s_z_rms": float(s_z.pow(2).mean().sqrt().item()),
                    "s_z_raw_rms": float(s_z_raw.pow(2).mean().sqrt().item()),
                    "s_z_abs_mean": float(s_z.abs().mean().item()),
                    "s_z_abs_max": float(s_z.abs().max().item()),
                    "s_z_abs_gt_1": float((s_z.abs() > 1.0).float().mean().item()),
                    "s_z_abs_gt_2": float((s_z.abs() > 2.0).float().mean().item()),
                    "s_z_abs_gt_4": float((s_z.abs() > 4.0).float().mean().item()),
                    "lam_rms": float(lam.pow(2).mean().sqrt().item()),
                    "lam_abs_mean": float(lam.abs().mean().item()),
                    "lam_abs_max": float(lam.abs().max().item()),
                    "lam_pos_frac": float((lam > 0).float().mean().item()),
                    "lam_neg_frac": float((lam < 0).float().mean().item()),
                    "lam_near_zero_frac": float((lam.abs() < 0.05).float().mean().item()),
                    "proj_x_rms": float(proj_x.pow(2).mean().sqrt().item()),
                    "feat_rms": float(feat.pow(2).mean().sqrt().item()),
                    "z_rms": float(z.pow(2).mean().sqrt().item()),
                    "z_win_rms": float(z_win.pow(2).mean().sqrt().item()),
                    "proj_in_rms": float(proj_in.pow(2).mean().sqrt().item()),
                    "gamma": float(gamma.item()),
                    "scale_w": [float(v.item()) for v in scale_w],
                }
                if self.s_z_rms_norm:
                    diag["s_z_gain"] = float(torch.exp(self.s_z_gain_log).clamp(min=0.25, max=8.0).item())
                    diag["s_z_gain_active"] = float(self.s_z_gain_active)
                    diag["s_z_gain_alpha"] = float(self.s_z_gain_alpha)
                    diag["s_z_gain_mix_lambda"] = float(self.s_z_gain_mix == "lambda")
                self.last_diag = diag
        return (lam * proj_x) @ self.weight


class _PhiPrefixMsMultwinLadderMixCore(_PhiPrefixMsMultwinCore):
    """Four-scale ladder with additive odd and optional s_x/s_z mixing.

    Modes:
      addodd_tanh:      ladder(s_z) + β tanh(s_z)
      addodd_delta:     ladder(s_z) + β tanh(s_x - s_z)
      mix_addodd_delta: ladder((1-α)s_z + α s_x) + β tanh(s_x - s_z)
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
        mix_mode: str = "addodd_delta",
        odd_scale: float = 0.10,
        sx_mix: float = 0.25,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            window_size=window_size,
            window_sizes=window_sizes,
        )
        self.mix_mode = mix_mode
        self.odd_scale = float(odd_scale)
        self.sx_mix = float(sx_mix)
        self.diag_enabled = False
        self.last_diag: dict[str, float | list[float]] | None = None

    def _compute_lambda(self, s_z: torch.Tensor, s_x: torch.Tensor) -> torch.Tensor:
        if self.mix_mode == "addodd_tanh":
            return _ladder_add_odd_tanh(s_z, self.odd_scale)
        if self.mix_mode == "addodd_delta":
            return _ladder_add_odd_delta(s_z, s_x - s_z, self.odd_scale)
        if self.mix_mode == "mix_addodd_delta":
            return _ladder_mix_add_odd_delta(
                s_z, s_x, sx_mix=self.sx_mix, odd_scale=self.odd_scale
            )
        raise ValueError(f"unknown ladder mix mode: {self.mix_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_win = self._fused_z_and_window(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        s_x = proj_x + self.bias
        lam = self._compute_lambda(s_z, s_x)
        if self.diag_enabled:
            with torch.no_grad():
                feat = lam * proj_x
                delta = s_x - s_z
                scale_w = F.softmax(self.scale_logits, dim=0)
                self.last_diag = {
                    "s_z_rms": float(s_z.pow(2).mean().sqrt().item()),
                    "s_x_rms": float(s_x.pow(2).mean().sqrt().item()),
                    "delta_rms": float(delta.pow(2).mean().sqrt().item()),
                    "s_z_abs_mean": float(s_z.abs().mean().item()),
                    "delta_abs_mean": float(delta.abs().mean().item()),
                    "lam_rms": float(lam.pow(2).mean().sqrt().item()),
                    "lam_abs_mean": float(lam.abs().mean().item()),
                    "lam_pos_frac": float((lam > 0).float().mean().item()),
                    "lam_neg_frac": float((lam < 0).float().mean().item()),
                    "proj_x_rms": float(proj_x.pow(2).mean().sqrt().item()),
                    "feat_rms": float(feat.pow(2).mean().sqrt().item()),
                    "gamma": float(gamma.item()),
                    "scale_w": [float(v.item()) for v in scale_w],
                    "mix_mode": self.mix_mode,
                }
        return (lam * proj_x) @ self.weight


class _PhiPrefixMsMultwinLadderCrossCore(_PhiPrefixMsMultwinCore):
    """Route A: four-scale ladder Hessian + zero-init low-rank cross mix.

    feat = H_ladder(s_z) · proj_x  +  α · (proj_in @ U(z) @ V(z)^T)

    U(z), V(z) are fixed learnable factors with z-dependent rank scales; α and
    z-scales are zero-initialized so step-0 matches plain invcombo_ladder.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
        cross_rank: int = 64,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            window_size=window_size,
            window_sizes=window_sizes,
        )
        self.cross_rank = int(cross_rank)
        self.cross_u = nn.Parameter(torch.randn(dim, self.cross_rank) * 0.02)
        self.cross_v = nn.Parameter(torch.randn(dim, self.cross_rank) * 0.02)
        self.cross_u_scale = nn.Linear(dim, self.cross_rank, bias=False)
        self.cross_v_scale = nn.Linear(dim, self.cross_rank, bias=False)
        nn.init.zeros_(self.cross_u_scale.weight)
        nn.init.zeros_(self.cross_v_scale.weight)
        self.cross_alpha = nn.Parameter(torch.tensor(0.0))
        self.diag_enabled = False
        self.last_diag: dict[str, float | list[float]] | None = None

    def _cross_branch(self, proj_in: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        u_scale = 1.0 + self.cross_u_scale(z)
        v_scale = 1.0 + self.cross_v_scale(z)
        mid = (proj_in @ self.cross_u) * u_scale
        return self.cross_alpha * (mid * v_scale) @ self.cross_v.transpose(-2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_win = self._fused_z_and_window(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        lam = _tanh2_invcombo_ladder_second_deriv(s_z)
        feat_hess = (lam * proj_x) @ self.weight
        cross = self._cross_branch(proj_in, z)
        feat = feat_hess + cross
        if self.diag_enabled:
            with torch.no_grad():
                scale_w = F.softmax(self.scale_logits, dim=0)
                hess_rms = float(feat_hess.pow(2).mean().sqrt().item())
                cross_rms = float(cross.pow(2).mean().sqrt().item())
                self.last_diag = {
                    "s_z_rms": float(s_z.pow(2).mean().sqrt().item()),
                    "lam_rms": float(lam.pow(2).mean().sqrt().item()),
                    "lam_pos_frac": float((lam > 0).float().mean().item()),
                    "lam_neg_frac": float((lam < 0).float().mean().item()),
                    "proj_x_rms": float(proj_x.pow(2).mean().sqrt().item()),
                    "feat_hess_rms": hess_rms,
                    "feat_cross_rms": cross_rms,
                    "cross_to_hess_ratio": cross_rms / max(hess_rms, 1e-8),
                    "cross_alpha": float(self.cross_alpha.item()),
                    "feat_rms": float(feat.pow(2).mean().sqrt().item()),
                    "gamma": float(gamma.item()),
                    "scale_w": [float(v.item()) for v in scale_w],
                }
        return feat


class _PhiPrefixMsMultwinLadderBilinearCore(_PhiPrefixMsMultwinCore):
    """Route B: ladder Hessian + zero-init parallel bilinear cross-potential branch.

    Φ = Φ_ladder(s_z) + Σ_r φ( (u_r^T proj_in) · (v_r^T z) )

    Parallel to Route A (same ladder backbone); cross term uses bilinear potential
    with local u and context z, weight = 2s φ''(s) + φ'(s) on both output legs.
    α is zero-initialized so step-0 matches plain invcombo_ladder.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
        bilinear_rank: int = 64,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            window_size=window_size,
            window_sizes=window_sizes,
        )
        self.bilinear_rank = int(bilinear_rank)
        self.bilinear_u = nn.Parameter(
            torch.randn(dim, self.bilinear_rank) * 0.02
        )
        self.bilinear_v = nn.Parameter(
            torch.randn(dim, self.bilinear_rank) * 0.02
        )
        self.bilinear_alpha = nn.Parameter(torch.tensor(0.0))
        self.diag_enabled = False
        self.last_diag: dict[str, float | list[float]] | None = None

    def _bilinear_branch(self, proj_in: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        u_dot = proj_in @ self.bilinear_u
        v_dot = z @ self.bilinear_v
        s_bilin = u_dot * v_dot
        t = torch.tanh(s_bilin)
        s2 = (1.0 - t * t).clamp(min=1e-8)
        phi_p = 2.0 * t * s2
        phi_pp = 2.0 * s2 * s2 - 4.0 * t * t * s2
        weight = 2.0 * s_bilin * phi_pp + phi_p
        u = self.bilinear_u
        v = self.bilinear_v
        branch = torch.matmul(weight * v_dot, u.transpose(-2, -1)) + torch.matmul(
            weight * u_dot, v.transpose(-2, -1)
        )
        return self.bilinear_alpha * branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_win = self._fused_z_and_window(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        lam = _tanh2_invcombo_ladder_second_deriv(s_z)
        feat_hess = (lam * proj_x) @ self.weight
        bilinear = self._bilinear_branch(proj_in, z)
        feat = feat_hess + bilinear
        if self.diag_enabled:
            with torch.no_grad():
                scale_w = F.softmax(self.scale_logits, dim=0)
                u_dot = proj_in @ self.bilinear_u
                v_dot = z @ self.bilinear_v
                s_bilin = u_dot * v_dot
                hess_rms = float(feat_hess.pow(2).mean().sqrt().item())
                bilin_rms = float(bilinear.pow(2).mean().sqrt().item())
                self.last_diag = {
                    "s_z_rms": float(s_z.pow(2).mean().sqrt().item()),
                    "s_bilin_rms": float(s_bilin.pow(2).mean().sqrt().item()),
                    "lam_rms": float(lam.pow(2).mean().sqrt().item()),
                    "lam_pos_frac": float((lam > 0).float().mean().item()),
                    "lam_neg_frac": float((lam < 0).float().mean().item()),
                    "proj_x_rms": float(proj_x.pow(2).mean().sqrt().item()),
                    "feat_hess_rms": hess_rms,
                    "feat_bilinear_rms": bilin_rms,
                    "bilinear_to_hess_ratio": bilin_rms / max(hess_rms, 1e-8),
                    "bilinear_alpha": float(self.bilinear_alpha.item()),
                    "u_dot_rms": float(u_dot.pow(2).mean().sqrt().item()),
                    "v_dot_rms": float(v_dot.pow(2).mean().sqrt().item()),
                    "feat_rms": float(feat.pow(2).mean().sqrt().item()),
                    "gamma": float(gamma.item()),
                    "scale_w": [float(v.item()) for v in scale_w],
                }
        return feat


class _PhiPrefixMsMultwinLadderDecoupleCore(_PhiPrefixMsMultwinCore):
    """Ladder with decoupled input/output directions per unit (break m co-axis).

    feat = Σ_j λ_j(s_z) · (u_j^T proj_in) · v_j
         = (diag(λ) · (proj_in @ U)) @ V

    λ uses s_z = z @ U + b; U (``weight``) is shared for proj and curvature;
    V (``out_weight``) is zero-init copy of U so step-0 matches co-axial ladder.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            window_size=window_size,
            window_sizes=window_sizes,
        )
        self.out_weight = nn.Parameter(torch.empty_like(self.weight))
        with torch.no_grad():
            self.out_weight.copy_(self.weight)
        self.diag_enabled = False
        self.last_diag: dict[str, float | list[float]] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_win = self._fused_z_and_window(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w_in = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w_in
        s_z = z @ w_in + self.bias
        lam = _tanh2_invcombo_ladder_second_deriv(s_z)
        feat = (lam * proj_x) @ self.out_weight
        if self.diag_enabled:
            with torch.no_grad():
                scale_w = F.softmax(self.scale_logits, dim=0)
                u_norm = self.weight.norm(dim=1).clamp(min=1e-8)
                v_norm = self.out_weight.norm(dim=1).clamp(min=1e-8)
                uv_cos = (
                    (self.weight * self.out_weight).sum(dim=1) / (u_norm * v_norm)
                ).mean()
                feat_coax = (lam * proj_x) @ self.weight
                delta = feat - feat_coax
                coax_rms = float(feat_coax.pow(2).mean().sqrt().item())
                delta_rms = float(delta.pow(2).mean().sqrt().item())
                self.last_diag = {
                    "s_z_rms": float(s_z.pow(2).mean().sqrt().item()),
                    "lam_rms": float(lam.pow(2).mean().sqrt().item()),
                    "lam_pos_frac": float((lam > 0).float().mean().item()),
                    "lam_neg_frac": float((lam < 0).float().mean().item()),
                    "proj_x_rms": float(proj_x.pow(2).mean().sqrt().item()),
                    "uv_cos_mean": float(uv_cos.item()),
                    "feat_coax_rms": coax_rms,
                    "feat_decouple_delta_rms": delta_rms,
                    "decouple_to_coax_ratio": delta_rms / max(coax_rms, 1e-8),
                    "feat_rms": float(feat.pow(2).mean().sqrt().item()),
                    "gamma": float(gamma.item()),
                    "scale_w": [float(v.item()) for v in scale_w],
                }
        return feat


class _PhiPrefixMsMultwinLadderDecoupleCtxSplitCore(_PhiPrefixMsMultwinLadderDecoupleCore):
    """Legacy full stack with U/A/V gradient split (same idea as simplified ctxsplit).

    ``weight`` (U) = content projection; ``context_weight`` (A) = z→λ routing;
    ``out_weight`` (V) = output.  Scratch: ``A←U`` in ``__init__`` for step~0 tie.
    Pretrain/FT: ordered ``U←W1``, ``A←W1``, ``V←W2`` via analytic translation.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            window_size=window_size,
            window_sizes=window_sizes,
        )
        self.context_weight = nn.Parameter(self.weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_win = self._fused_z_and_window(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w_in = self.weight.transpose(-2, -1)
        a_in = self.context_weight.transpose(-2, -1)
        proj_x = proj_in @ w_in
        s_z = z @ a_in + self.bias
        lam = _tanh2_invcombo_ladder_second_deriv(s_z)
        feat = (lam * proj_x) @ self.out_weight
        if self.diag_enabled:
            with torch.no_grad():
                scale_w = F.softmax(self.scale_logits, dim=0)
                u_norm = self.weight.norm(dim=1).clamp(min=1e-8)
                v_norm = self.out_weight.norm(dim=1).clamp(min=1e-8)
                uv_cos = (
                    (self.weight * self.out_weight).sum(dim=1) / (u_norm * v_norm)
                ).mean()
                feat_coax = (lam * proj_x) @ self.weight
                delta = feat - feat_coax
                coax_rms = float(feat_coax.pow(2).mean().sqrt().item())
                delta_rms = float(delta.pow(2).mean().sqrt().item())
                self.last_diag = {
                    "s_z_rms": float(s_z.pow(2).mean().sqrt().item()),
                    "lam_rms": float(lam.pow(2).mean().sqrt().item()),
                    "lam_pos_frac": float((lam > 0).float().mean().item()),
                    "lam_neg_frac": float((lam < 0).float().mean().item()),
                    "proj_x_rms": float(proj_x.pow(2).mean().sqrt().item()),
                    "uv_cos_mean": float(uv_cos.item()),
                    "feat_coax_rms": coax_rms,
                    "feat_decouple_delta_rms": delta_rms,
                    "decouple_to_coax_ratio": delta_rms / max(coax_rms, 1e-8),
                    "feat_rms": float(feat.pow(2).mean().sqrt().item()),
                    "gamma": float(gamma.item()),
                    "scale_w": [float(v.item()) for v in scale_w],
                }
        return feat


class _PhiGlobalMsMultwinCore(nn.Module):
    """Bidirectional multwin: λ(z) from symmetric windows + global mean (vision-safe).

    Mirrors ``_PhiPrefixMsMultwinCore`` but replaces causal prefix / causal windows
    with symmetric pooling over the full patch sequence. Suitable for bidirectional
    ViT blocks where raster order is not a causal time axis.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.num_units = int(num_units) if num_units is not None else 6 * dim
        base_m = 6 * dim
        init_std = 0.02 * math.sqrt(base_m / self.num_units)
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * init_std)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        n_scales = len(self.window_sizes) + 1
        logits = torch.full((n_scales,), -1.0)
        if self.window_size in self.window_sizes:
            logits[self.window_sizes.index(self.window_size)] = 2.0
        else:
            logits[0] = 2.0
        self.scale_logits = nn.Parameter(logits)
        self.hp_logit = nn.Parameter(torch.tensor(2.0))

    def _fused_z_and_window(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_parts = [_symmetric_window_mean(x, w) for w in self.window_sizes]
        z_win = None
        for w, z_part in zip(self.window_sizes, z_parts):
            if w == self.window_size:
                z_win = z_part
                break
        if z_win is None:
            z_win = _symmetric_window_mean(x, self.window_size)
        z_parts.append(_global_seq_mean(x))
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = sum(sw * zp for sw, zp in zip(scale_w, z_parts))
        return z, z_win

    def _legacy_forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_win = self._fused_z_and_window(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        lam = _tanh2_second_deriv(s_z)
        return (lam * proj_x) @ self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from tlb.kernels.hessian_dispatch import global_ms_tied, resolve_hessian_ffn_backend
        from tlb.kernels.hessian_ffn_fused import _is_global_only_scale

        if resolve_hessian_ffn_backend(x.device) == "fused" and _is_global_only_scale(
            self.scale_logits
        ):
            z_win = None
            for w in self.window_sizes:
                if w == self.window_size:
                    z_win = _symmetric_window_mean(x, w)
                    break
            if z_win is None:
                z_win = _symmetric_window_mean(x, self.window_size)
            return global_ms_tied(
                x,
                self.weight,
                self.bias,
                self.hp_logit,
                window_size=self.window_size,
                z_win=z_win,
            )
        return self._legacy_forward(x)


class _PhiGlobalMsMultwinDecoupleCore(_PhiGlobalMsMultwinCore):
    """Bidirectional multwin + decoupled U/V per unit (vision-safe decouple port)."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            window_size=window_size,
            window_sizes=window_sizes,
        )
        self.out_weight = nn.Parameter(torch.empty_like(self.weight))
        with torch.no_grad():
            self.out_weight.copy_(self.weight)

    def _legacy_forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_win = self._fused_z_and_window(x)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w_in = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w_in
        s_z = z @ w_in + self.bias
        lam = _tanh2_invcombo_ladder_second_deriv(s_z)
        return (lam * proj_x) @ self.out_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from tlb.kernels.hessian_dispatch import global_ms_decouple

        return global_ms_decouple(
            x,
            self.weight,
            self.bias,
            self.out_weight,
            self.hp_logit,
            self.scale_logits,
            self.window_sizes,
            self.window_size,
            lambda_activ="invcombo_ladder",
            legacy_forward=lambda: self._legacy_forward(x),
        )


class _PhiGlobalLightDecoupleCore(nn.Module):
    """Fast vision decouple: global λ(z), one cumsum local window for high-pass, tanh²''."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        local_window: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.local_window = int(local_window)
        self.num_units = int(num_units) if num_units is not None else 6 * dim
        base_m = 6 * dim
        init_std = 0.02 * math.sqrt(base_m / self.num_units)
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * init_std)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        self.hp_logit = nn.Parameter(torch.tensor(2.0))
        self.out_weight = nn.Parameter(torch.empty_like(self.weight))
        with torch.no_grad():
            self.out_weight.copy_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from tlb.kernels.hessian_dispatch import global_light_decouple

        return global_light_decouple(
            x,
            self.weight,
            self.bias,
            self.out_weight,
            self.hp_logit,
            local_window=self.local_window,
        )


class _PhiGlobalDecoupleMinCore(nn.Module):
    """h4_phi768 per-token map + U/V decouple (+ optional global λ(z)).

    ``lambda_geometry='proj_oracle'`` matches analytic MLP translation (vision swap).
    ``lambda_geometry='global_tanh2'`` uses global seq-mean λ(s_z) with tanh²''.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        lambda_geometry: DecoupleMinLambda = "proj_oracle",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.lambda_geometry = lambda_geometry
        self.num_units = int(num_units) if num_units is not None else 6 * dim
        base_m = 6 * dim
        init_std = 0.02 * math.sqrt(base_m / self.num_units)
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * init_std)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        self.out_weight = nn.Parameter(torch.empty_like(self.weight))
        self.out_bias = nn.Parameter(torch.zeros(dim))
        self.tanh2_temperature = 1.0
        self.tanh2_amplitude = "paper"
        with torch.no_grad():
            self.out_weight.copy_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if str(getattr(self, "tanh2_amplitude", "paper")) == "gelu_band":
            proj_x = F.linear(x, self.weight, self.bias)
            lam = tanh2_softened(
                proj_x,
                float(getattr(self, "tanh2_temperature", 1.0)),
                amplitude="gelu_band",
            )
            return torch.matmul(proj_x * lam, self.out_weight) + self.out_bias
        from tlb.kernels.hessian_dispatch import decouple_min

        return decouple_min(
            x,
            self.weight,
            self.bias,
            self.out_weight,
            out_bias=self.out_bias,
            lambda_geometry=self.lambda_geometry,
        )


class _SideUnitRankMixin:
    """Rank-``r`` block residual in unit space before shared ``V``.

    Oracle: ``h = λ(s) ⊙ s``, ``y = h @ V + b₂``.
    Side supplements missing rank **inside** the unit sheet (not on logits):

        ``h = λ(s) ⊙ s + B (A x)``,  ``y = h @ V + b₂``.

    ``A`` (``side_down``) is ``r × d``, independent of ``U``.
    ``B`` (``side_up``) is ``m × r``, zero at step 0 so swap stays oracle-exact.
    """

    side_rank: int

    def _init_side_unit_rank(
        self,
        dim: int,
        *,
        side_rank: int = 64,
        side_down_std: float = 0.02,
    ) -> None:
        r = int(side_rank)
        self.side_rank = r
        self.side_down = nn.Parameter(torch.randn(r, dim) * float(side_down_std))
        self.side_up = nn.Parameter(torch.zeros(int(self.num_units), r))

    def _rope_hidden(self, x: torch.Tensor) -> torch.Tensor:
        from tlb.ffn.mlp_hessian_analytic import lambda_decouple_geometry

        s = F.linear(x, self.weight, self.bias)
        return lambda_decouple_geometry(s, self.lambda_geometry) * s

    def _side_unit_delta(self, x: torch.Tensor) -> torch.Tensor:
        z = F.linear(x, self.side_down)
        return F.linear(z, self.side_up)

    def drop_side(self) -> None:
        with torch.no_grad():
            self.side_up.zero_()


class _SideUnitsMixin:
    """``m`` pseudo-units with ``λ ≡ 1`` before shared ``V``."""

    def _init_side_units(self, dim: int) -> None:
        n = int(self.num_units)
        self.side_unit_weight = nn.Parameter(torch.zeros(n, dim))
        self.side_unit_bias = nn.Parameter(torch.zeros(n))

    def _rope_hidden(self, x: torch.Tensor) -> torch.Tensor:
        from tlb.ffn.mlp_hessian_analytic import lambda_decouple_geometry

        s = F.linear(x, self.weight, self.bias)
        return lambda_decouple_geometry(s, self.lambda_geometry) * s

    def _side_units_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.side_unit_weight, self.side_unit_bias)

    def drop_side(self) -> None:
        with torch.no_grad():
            self.side_unit_weight.zero_()
            self.side_unit_bias.zero_()


class _PhiGlobalDecoupleMinSideCore(_SideUnitRankMixin, _PhiGlobalDecoupleMinCore):
    """Oracle rope plus rank-``r`` unit-space block ``B(Ax)`` read through ``V``."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        lambda_geometry: DecoupleMinLambda = "proj_oracle",
        side_rank: int = 64,
        side_down_std: float = 0.02,
    ) -> None:
        super().__init__(dim, num_units=num_units, lambda_geometry=lambda_geometry)
        self._init_side_unit_rank(
            dim, side_rank=side_rank, side_down_std=side_down_std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._rope_hidden(x) + self._side_unit_delta(x)
        return torch.matmul(h, self.out_weight) + self.out_bias


class _PhiGlobalDecoupleMinSideUpCore(_PhiGlobalDecoupleMinSideCore):
    """Same unit-space side; ``side_up`` name maps to ``hessian_out`` LR group."""


class _PhiGlobalDecoupleMinSideCurvCore(_SideUnitRankMixin, _PhiGlobalDecoupleMinCore):
    """Oracle rope plus curvature-gated side ``B(ψ''(z) ⊙ z)`` before shared ``V``.

    Main path stays the oracle first-order rope; only the rank-``r`` side leg uses
    native tanh²'' curvature on ``z = A x``.  ``B`` zero at step 0 ⇒ exact oracle.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        lambda_geometry: DecoupleMinLambda = "proj_oracle",
        side_rank: int = 64,
        side_down_std: float = 0.02,
        side_gate_temperature: float = 1.0,
    ) -> None:
        super().__init__(dim, num_units=num_units, lambda_geometry=lambda_geometry)
        self._init_side_unit_rank(
            dim, side_rank=side_rank, side_down_std=side_down_std
        )
        self.side_gate_temperature = float(side_gate_temperature)

    def _side_unit_delta(self, x: torch.Tensor) -> torch.Tensor:
        z = F.linear(x, self.side_down)
        tau = float(getattr(self, "side_gate_temperature", 1.0))
        if tau != 1.0:
            z = z / tau
        gate = _tanh2_second_deriv(z)
        return F.linear(gate * z, self.side_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._rope_hidden(x) + self._side_unit_delta(x)
        return torch.matmul(h, self.out_weight) + self.out_bias


class _PhiGlobalDecoupleMinSideUnitsCore(_SideUnitsMixin, _PhiGlobalDecoupleMinCore):
    """Oracle rope plus ``m`` extra units with fixed ``λ = 1`` before ``V``."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        lambda_geometry: DecoupleMinLambda = "proj_oracle",
    ) -> None:
        super().__init__(dim, num_units=num_units, lambda_geometry=lambda_geometry)
        self._init_side_units(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._rope_hidden(x) + self._side_units_hidden(x)
        return torch.matmul(h, self.out_weight) + self.out_bias


class _NativeCalibMixin:
    """Learnable ψ'' calibration: ``λ·u ≈ GELU(u)`` while keeping native tanh²''."""

    calib_mode: str = "grid_fit"

    def _init_native_calib_params(self) -> None:
        from tlb.ffn.mlp_hessian_analytic import default_native_calib_init

        amp, inner, shift = default_native_calib_init()
        self.lam_amplitude_log = nn.Parameter(torch.tensor(math.log(max(amp, 1e-4))))
        self.lam_inner_scale_log = nn.Parameter(torch.tensor(math.log(max(inner, 1e-4))))
        self.lam_shift = nn.Parameter(torch.tensor(float(shift)))

    def _native_calib_lambda(self, u: torch.Tensor) -> torch.Tensor:
        from tlb.ffn.mlp_hessian_analytic import lambda_native_calibrated

        amp = torch.exp(self.lam_amplitude_log).clamp(0.25, 4.0)
        inner = torch.exp(self.lam_inner_scale_log).clamp(0.25, 4.0)
        mode = getattr(self, "calib_mode", "grid_fit")
        return lambda_native_calibrated(
            u,
            amplitude=amp,
            inner_scale=inner,
            shift=self.lam_shift,
            mode=mode,  # type: ignore[arg-type]
        )

    def _resolve_lambda(
        self,
        gate_coord: torch.Tensor,
        *,
        content: torch.Tensor | None = None,
    ) -> torch.Tensor:
        geom = getattr(self, "lambda_geometry", "proj_oracle_fast")
        if geom == "proj_native_calib":
            return self._native_calib_lambda(gate_coord)
        from tlb.ffn.mlp_hessian_analytic import lambda_decouple_geometry

        return lambda_decouple_geometry(gate_coord, geom)


class _PhiGlobalDecoupleMinNativeCalibCore(_PhiGlobalDecoupleMinCore, _NativeCalibMixin):
    """decouple_min with learnable native ψ'' calibration on content ``s = Ux+b``.

    Forward: ``h = λ(s) ⊙ s``, ``y = h @ V`` where ``λ = A·ψ''(shift + α·s)`` (grid_fit)
    or ``λ = ψ''(α·s + β)`` (affine_gate via ``calib_mode``).
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        lambda_geometry: DecoupleMinLambda = "proj_native_calib",
        calib_mode: str = "grid_fit",
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            lambda_geometry=lambda_geometry,
        )
        self.calib_mode = str(calib_mode)
        self._init_native_calib_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.linear(x, self.weight, self.bias)
        lam = self._resolve_lambda(s, content=s)
        out = torch.matmul(s * lam, self.out_weight)
        return out + self.out_bias


class _PhiGlobalDecoupleMinUavCore(nn.Module, _NativeCalibMixin):
    """GPT-2 LM split: ``λ`` on content ``s`` only (main.tex eq.~5); ``A`` enriches content.

    Paper channel split (eq.~lm_channel_split) routes context through ``A``, but the
    **curvature gate stays on content coordinates** for GPT-2 warm-start:

        s_j = u_j^T x + b_j
        λ_j = calib(s_j) = A_amp · ψ''(shift + α_c · s_j)   [Scheme B; never ψ''(z@A)]
        â = z @ A + b_A                                      [context channel only]
        h_j = λ_j · (s_j + σ(mix) · â_j)
        y = h @ V + b_out

    ``mix_logit`` init ≈ -4 ⇒ σ(mix)≈0 at step-0 (matches decouple_min + oracle/calib on s).
    UA-then-V: train ``U`` and ``A`` while ``V`` frozen breaks coaxial UV co-adaptation.
    """

    use_raw_context: bool = True

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        lambda_geometry: DecoupleMinLambda = "proj_native_calib",
        calib_mode: str = "grid_fit",
        window_size: int = 64,
        use_raw_context: bool = True,
        mix_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.lambda_geometry = lambda_geometry
        self.calib_mode = str(calib_mode)
        self.window_size = max(1, int(window_size))
        self.use_raw_context = bool(use_raw_context)
        self.num_units = int(num_units) if num_units is not None else 6 * dim
        base_m = 6 * dim
        init_std = 0.02 * math.sqrt(base_m / self.num_units)
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * init_std)
        self.content_bias = nn.Parameter(torch.zeros(self.num_units))
        self.context_weight = nn.Parameter(torch.empty_like(self.weight))
        self.gate_bias = nn.Parameter(torch.zeros(self.num_units))
        self.mix_logit = nn.Parameter(torch.tensor(float(mix_init)))
        self.out_weight = nn.Parameter(torch.empty_like(self.weight))
        self.out_bias = nn.Parameter(torch.zeros(dim))
        with torch.no_grad():
            self.context_weight.copy_(self.weight)
        self._init_native_calib_params()

    def _context_z(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_raw_context:
            return x
        return _causal_window_mean(x, self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.linear(x, self.weight, self.content_bias)
        lam = self._resolve_lambda(s, content=s)
        z = self._context_z(x)
        a_hidden = F.linear(z, self.context_weight, self.gate_bias)
        mix = torch.sigmoid(self.mix_logit)
        gated = lam * (s + mix * a_hidden)
        out = torch.matmul(gated, self.out_weight)
        return out + self.out_bias


class _PhiGlobalDecoupleMin2dCore(nn.Module):
    """2D spectral Hessian decouple: ``Ψ(x)=∑_j ψ(s_j,t_j)`` with ``s=Ux+b``, ``t=Ax+b_A``.

    Scalar readout per unit (``y = h @ V``) implements the decouple projection of
    ``∇²Ψ·x`` without UAV-style ``λ(s)·(s + σ·t)``:

        h_j = ψ_ss(s,t)·s_j + ψ_st(s,t)·t_j + ψ_tt(s,t)·t_j

    Swap step-0: ``A←W1``, ``b_A←b1`` (same as ``U,b``), ``ψ_st=ψ_tt≈0`` via gated logits,
    ``ψ_ss`` from ``lambda_geometry`` (oracle GELU/s) ⇒ ``h ≈ GELU(s)`` (``t=s`` at init).
    The ``t`` leg is **not** multiplied by ``ψ_ss(s)``.
    """

    context_weight_zero_init: bool = True

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        lambda_geometry: DecoupleMinLambda = "proj_oracle_fast",
        cross_init: float = -12.0,
        tt_init: float = -12.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.lambda_geometry = lambda_geometry
        self.num_units = int(num_units) if num_units is not None else 6 * dim
        base_m = 6 * dim
        init_std = 0.02 * math.sqrt(base_m / self.num_units)
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * init_std)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        self.context_weight = nn.Parameter(torch.zeros(self.num_units, dim))
        self.gate_bias = nn.Parameter(torch.zeros(self.num_units))
        self.cross_logit = nn.Parameter(torch.tensor(float(cross_init)))
        self.tt_logit = nn.Parameter(torch.tensor(float(tt_init)))
        self.out_weight = nn.Parameter(torch.empty_like(self.weight))
        self.out_bias = nn.Parameter(torch.zeros(dim))
        with torch.no_grad():
            self.out_weight.copy_(self.weight)

    def _psi_ss(self, s: torch.Tensor) -> torch.Tensor:
        from tlb.ffn.mlp_hessian_analytic import lambda_decouple_geometry

        return lambda_decouple_geometry(s, self.lambda_geometry)

    @staticmethod
    def _psi_st(_s: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        """Unit mixed curvature for ``ψ(s,t)=…+α·s·t`` (constant ``ψ_st=α`` per unit)."""
        return torch.ones_like(_s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.linear(x, self.weight, self.bias)
        t = F.linear(x, self.context_weight, self.gate_bias)
        lam_ss = self._psi_ss(s)
        cross = torch.sigmoid(self.cross_logit) * self._psi_st(s, t)
        tt_gate = torch.sigmoid(self.tt_logit) * _tanh2_second_deriv(t)
        h = lam_ss * s + cross * t + tt_gate * t
        return torch.matmul(h, self.out_weight) + self.out_bias


class _PhiGlobalDecoupleMinOffaxisCore(nn.Module):
    """Multiplicative off-axis resonance: A gates U features; λ stays on U expansion.

    s = U x + b
    a = A RMSNorm(z),  z = causal_window_mean(x)
    c = U a
    g = 2 σ(c)  (= 1 + tanh(c/2));  A=0 ⇒ g=1 (step-0 = decouple_min)
    h = oracle(s) ⊙ s ⊙ g
    y = h @ V + b_out

    Pretrain init: U←W1, A←0, V←W2 (``context_weight_zero_init``).
    Multi-layer FT: tie ``context_weight`` across swap layers via
    ``tie_offaxis_shared_context`` so one residual-space A routes all blocks.

    A deliberately acts in the shared residual space (D×D), then each layer's U
    maps the routed context into its own FFN-unit basis.  Directly sharing an M×D
    gate assumes that row j of every pretrained W1 has the same semantics, which
    is not true across GPT-2 blocks.
    """

    context_weight_zero_init: bool = True
    window_size: int = 64

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        lambda_geometry: DecoupleMinLambda = "proj_oracle_fast",
        window_size: int = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.lambda_geometry = lambda_geometry
        self.window_size = max(1, int(window_size))
        self.num_units = int(num_units) if num_units is not None else 6 * dim
        base_m = 6 * dim
        init_std = 0.02 * math.sqrt(base_m / self.num_units)
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * init_std)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        self.context_weight = nn.Parameter(torch.zeros(dim, dim))
        self.out_weight = nn.Parameter(torch.empty_like(self.weight))
        self.out_bias = nn.Parameter(torch.zeros(dim))
        with torch.no_grad():
            self.out_weight.copy_(self.weight)

    def _lambda_from_s(self, s: torch.Tensor) -> torch.Tensor:
        from tlb.ffn.mlp_hessian_analytic import lambda_decouple_geometry

        return lambda_decouple_geometry(s, self.lambda_geometry)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.linear(x, self.weight, self.bias)
        z = _causal_window_mean(x, self.window_size)
        routed_context = F.linear(_rms_norm(z), self.context_weight)
        c = F.linear(routed_context, self.weight)
        gate = 2.0 * torch.sigmoid(c)
        lam = self._lambda_from_s(s)
        gated = lam * s * gate
        return gated @ self.out_weight + self.out_bias


def _record_lam_abs_mean(core: nn.Module, lam: torch.Tensor) -> None:
    if not core.training:
        return
    with torch.no_grad():
        core.last_lam = lam.detach()
        reduce_dims = tuple(range(lam.ndim - 1))
        core.last_lam_abs_mean = lam.detach().abs().mean(dim=reduce_dims)


def _token_active_rank_readout(
    core: nn.Module,
    *,
    gated: torch.Tensor,
    lam: torch.Tensor,
    out_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shared active-rank backward routing (per-token |λ| or layer-rank coord)."""
    from tlb.training.cross_layer_rank_coord import LayerCoordRankReadout

    _record_lam_abs_mean(core, lam)
    if out_bias is None:
        out_bias = getattr(core, "out_bias", None)
    if out_bias is None:
        out_bias = torch.zeros(
            core.out_weight.shape[1],
            device=gated.device,
            dtype=gated.dtype,
        )
    if not core.training or not core.token_rank_enabled:
        return gated @ core.out_weight + out_bias

    if bool(getattr(core, "use_layer_rank_coord", False)):
        layer_pos = int(getattr(core, "layer_rank_coord_pos", 0))
        return LayerCoordRankReadout.apply(
            gated,
            core.out_weight,
            out_bias,
            lam,
            layer_pos,
        )

    keep_frac = max(0.0, min(1.0, float(core.token_rank_keep_frac)))
    inactive = max(0.0, min(1.0, float(core.token_rank_inactive_scale)))
    k = max(1, min(gated.shape[-1], int(round(gated.shape[-1] * keep_frac))))
    rank_scores = lam.detach().abs()
    active_idx = torch.topk(
        rank_scores,
        k=k,
        dim=-1,
        largest=not core.token_rank_inverse,
        sorted=False,
    ).indices
    grad_scale = torch.full_like(lam.detach(), inactive)
    grad_scale.scatter_(-1, active_idx, 1.0)
    ones = torch.ones_like(grad_scale)
    gate = str(core.token_rank_gate).strip().lower()
    if gate == "u_only":
        u_scale, v_scale = grad_scale, ones
    elif gate == "v_only":
        u_scale, v_scale = ones, grad_scale
    else:
        u_scale = v_scale = grad_scale

    reduce_dims = tuple(range(grad_scale.ndim - 1))
    active_frequency = (grad_scale > inactive).float().mean(dim=reduce_dims)
    core.last_token_active_frequency = active_frequency
    core.last_token_active_union = int((active_frequency > 0).sum().item())
    core.last_token_active_count = k
    return _TokenRankReadout.apply(
        gated,
        core.out_weight,
        out_bias,
        u_scale,
        v_scale,
    )


class _TokenRankReadout(torch.autograd.Function):
    """Identity-forward readout with token/unit-specific backward routing.

    ``u_scale`` gates the upstream U/λ path; ``v_scale`` gates row ``j`` of V.
    Forward remains exactly ``gated @ V``.
    """

    @staticmethod
    def forward(
        ctx,
        gated: torch.Tensor,
        out_weight: torch.Tensor,
        out_bias: torch.Tensor,
        u_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(gated, out_weight, u_scale, v_scale)
        return gated @ out_weight + out_bias

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        gated, out_weight, u_scale, v_scale = ctx.saved_tensors
        grad_gated = (grad_output @ out_weight.transpose(0, 1)) * u_scale
        flat_gated = (gated * v_scale).reshape(-1, gated.shape[-1])
        flat_output = grad_output.reshape(-1, grad_output.shape[-1])
        grad_out_weight = flat_gated.transpose(0, 1) @ flat_output
        grad_out_bias = flat_output.sum(dim=0)
        return grad_gated, grad_out_weight, grad_out_bias, None, None


class _PhiGlobalDecoupleMinActiveRankCore(_PhiGlobalDecoupleMinCore):
    """Decouple-min with true token-level active-rank backward routing.

    Forward is identical to ``global_decouple_min``.  During training each
    ``(batch, token)`` selects its own top-|λ| units.  The resulting mask scales
    that token's contribution to both U and V gradients; it is never averaged
    into a batch-global row mask.
    """

    token_rank_enabled: bool = False
    token_rank_inverse: bool = False
    token_rank_gate: str = "uv"
    token_rank_keep_frac: float = 1.0
    token_rank_inactive_scale: float = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_x = F.linear(x, self.weight, self.bias)
        if self.lambda_geometry in (
            "proj_oracle_fast",
            "proj_oracle",
            "global_oracle",
            "proj_gelu_tanh2_taylor",
            "proj_oracle_x_native_norm0",
            "proj_oracle_div_native_norm0",
            "proj_oracle_geom_native_norm0",
            "proj_oracle_dot_norm_native",
            "proj_gelu_x_native_over_s",
            "proj_oracle_mod_native_ortho",
            "proj_oracle_mod_native_simple",
        ):
            from tlb.ffn.mlp_hessian_analytic import lambda_decouple_geometry

            lam = lambda_decouple_geometry(proj_x, self.lambda_geometry)
        else:
            from tlb.kernels.hessian_ffn_fused import _lambda_gate

            s_z = F.linear(x.mean(dim=1), self.weight, self.bias)
            lam = _lambda_gate(s_z)
            if lam.ndim == 2 and proj_x.ndim == 3:
                lam = lam.unsqueeze(1)
        gated = lam * proj_x
        return _token_active_rank_readout(
            self,
            gated=gated,
            lam=lam,
            out_bias=self.out_bias,
        )


class _PhiPrefixMsMultwinHybridCore(_PhiPrefixMsMultwinCore):
    """multwin + hybrid λ: blend λ(z) with λ(x) via learnable gate.

    λ_total = σ(gate) * λ(z) + (1-σ(gate)) * λ(x)
    Init favors context λ(z) (gate ≈ 0.88) allowing model to open local path.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
        mix_z_init: float = 2.0,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            window_size=window_size,
            window_sizes=window_sizes,
        )
        # init favors context λ(z) (≈0.88); model can learn to open local λ(x) path
        self.lam_mix_logit = nn.Parameter(torch.tensor(float(mix_z_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._fused_z(x)
        z_win = _causal_window_mean(x, self.window_size)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        s_x = proj_x + self.bias  # per-token local summary
        lam_z = _tanh2_second_deriv(s_z)
        lam_x = _tanh2_second_deriv(s_x)
        w_z = torch.sigmoid(self.lam_mix_logit)
        lam = w_z * lam_z + (1.0 - w_z) * lam_x
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


def _dim_segment_mean(x: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Per-token segment mean over feature dim, broadcast back. x: (B, T, D)."""
    b, t, d = x.shape
    seg = d // n_segments
    if seg * n_segments != d:
        raise ValueError(f"dim {d} not divisible by n_segments {n_segments}")
    xv = x.view(b, t, n_segments, seg)
    sm = xv.mean(dim=-1, keepdim=True)
    return sm.expand_as(xv).reshape(b, t, d)


class _PhiPrefixMsMultwinDsegCore(_PhiPrefixMsMultwinCore):
    """multwin temporal z + blend with multwin on dim-segment-smoothed x (for λ)."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        window_size: int = 64,
        window_sizes: tuple[int, ...] = (4, 16, 64),
        n_segments: int = 8,
        dseg_mix_init: float = 4.0,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            window_size=window_size,
            window_sizes=window_sizes,
        )
        self.n_segments = int(n_segments)
        # sigmoid(3)≈0.95 → step-0 ≈ plain multwin
        self.dseg_mix_logit = nn.Parameter(torch.tensor(float(dseg_mix_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_time = self._fused_z(x)
        x_dseg = _dim_segment_mean(x, self.n_segments)
        z_dseg = self._fused_z(x_dseg)
        mix = torch.sigmoid(self.dseg_mix_logit)
        z = mix * z_time + (1.0 - mix) * z_dseg
        z_win = _causal_window_mean(x, self.window_size)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        lam = _tanh2_second_deriv(s_z)
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiPrefixMultiScaleD1Core(nn.Module):
    """prefix_ms + λ = ψ'' + βψ' (first-derivative saturation fallback)."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        beta_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pool_dim = int(pool_dim)
        self.window_size = int(window_size)
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        self.q_proj = nn.Linear(dim, self.pool_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.pool_dim, bias=False)
        self.scale_logits = nn.Parameter(torch.tensor([2.0, 0.5, -1.0]))
        self.hp_logit = nn.Parameter(torch.tensor(2.0))
        self.lam_beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_attn = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win = _causal_window_mean(x, self.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = (
            scale_w[0] * z_attn
            + scale_w[1] * z_win
            + scale_w[2] * z_long
        )
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        lam = _lam_ms_d1(s_z, self.lam_beta)
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _MsPrefixBase(nn.Module):
    """Shared multi-scale z and proj path for prefix_ms descendants."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pool_dim = int(pool_dim)
        self.window_size = int(window_size)
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        self.q_proj = nn.Linear(dim, self.pool_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.pool_dim, bias=False)
        self.scale_logits = nn.Parameter(torch.tensor([2.0, 0.5, -1.0]))
        self.hp_logit = nn.Parameter(torch.tensor(2.0))

    def _fused_z(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale causal z (attn + window + prefix mean)."""
        z_attn = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win = _causal_window_mean(x, self.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        return (
            scale_w[0] * z_attn
            + scale_w[1] * z_win
            + scale_w[2] * z_long
        )

    def _ms_tensors(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return proj_x, s_z (context), s_x (local per-token)."""
        z = self._fused_z(x)
        z_win = _causal_window_mean(x, self.window_size)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        s_x = proj_x + self.bias
        return proj_x, s_z, s_x


class _PhiPrefixMsStatsCore(_MsPrefixBase):
    """prefix_ms proj; λ uses ``[z_ms, z_std]`` (mean + causal std per dim)."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
    ) -> None:
        super().__init__(
            dim,
            num_units=num_units,
            pool_dim=pool_dim,
            window_size=window_size,
        )
        m = self.num_units
        d = self.dim
        self.weight_lam = nn.Parameter(torch.empty(m, 2 * d))
        nn.init.zeros_(self.weight_lam)
        with torch.no_grad():
            self.weight_lam[:, :d].copy_(self.weight)

    def _ms_tensors(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self._fused_z(x)
        z_std = _causal_prefix_std(x)
        z_cat = torch.cat([z, z_std], dim=-1)
        z_win = _causal_window_mean(x, self.window_size)
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z_cat @ self.weight_lam.transpose(-2, -1) + self.bias
        s_x = proj_x + self.bias
        return proj_x, s_z, s_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_x, s_z, _s_x = self._ms_tensors(x)
        lam = _tanh2_second_deriv(s_z)
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiPrefixMoECore(_MsPrefixBase):
    """Top-K MoE over per-token λ experts on a fixed prefix_ms backbone.

    prefix_ms path is fixed (multi-scale z + γ-blend proj). λ is anchored at
    ψ''(s_z) (E0 = prefix_ms). Three auxiliary λ corrections are routed per token:

      E1: ψ''(s_x) − ψ''(s_z)     local deviation
      E2: (ψ''+βψ')(s_z) − ψ''(s_z)
      E3: (ψ''+ε)(s_z) − ψ''(s_z)

    Top-2 among E1–E3 with tanh gates (zero init → exact prefix_ms at step 0).
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        top_k: int = 2,
        beta_init: float = 0.0,
        eps_init: float = 0.0,
    ) -> None:
        super().__init__(
            dim, num_units=num_units, pool_dim=pool_dim, window_size=window_size
        )
        self.top_k = min(int(top_k), 3)

        # Route 3 auxiliary λ corrections (not the prefix_ms anchor)
        self.route = nn.Linear(dim, 3, bias=True)
        nn.init.zeros_(self.route.weight)
        nn.init.zeros_(self.route.bias)

        self.lam_beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.lam_eps = nn.Parameter(torch.tensor(float(eps_init)))

    def _aux_deltas(
        self, lam_base: torch.Tensor, s_z: torch.Tensor, s_x: torch.Tensor
    ) -> torch.Tensor:
        eps = self.lam_eps.clamp(min=0.0)
        deltas = [
            _tanh2_second_deriv(s_x) - lam_base,
            _lam_ms_d1(s_z, self.lam_beta) - lam_base,
            _tanh2_second_deriv_eps(s_z, eps) - lam_base,
        ]
        return torch.stack(deltas, dim=-1)  # (B, T, M, 3)

    def _aux_gates(self, x: torch.Tensor) -> torch.Tensor:
        """Top-k sparse tanh gates over 3 aux experts; zero init → no correction."""
        logits = self.route(x)
        gates = torch.tanh(logits)
        _, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        mask = torch.zeros_like(gates)
        mask.scatter_(-1, topk_idx, 1.0)
        return gates * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_x, s_z, s_x = self._ms_tensors(x)
        lam_base = _tanh2_second_deriv(s_z)
        aux = self._aux_deltas(lam_base, s_z, s_x)
        gates = self._aux_gates(x).unsqueeze(2)  # (B, T, 1, 3)
        lam = lam_base + (aux * gates).sum(dim=-1)
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiPrefixLambdaGlobalCore(_MsPrefixBase):
    """prefix_ms backbone + layer-global λ expert blend (no per-token routing).

    Same multi-scale z / proj as prefix_ms. λ modes are mixed with learnable
    scalars shared by all tokens (like scale_logits for z):

      E0: ψ''(s_z)           — prefix_ms default (init dominant)
      E1: ψ''(s_x)           — local per-position curvature
      E2: ψ'' + β ψ'(s_z)    — saturation fallback
      E3: ψ''(s_z) + ε       — curvature floor
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        beta_init: float = 0.0,
        eps_init: float = 0.0,
    ) -> None:
        super().__init__(
            dim, num_units=num_units, pool_dim=pool_dim, window_size=window_size
        )
        # init favors E0 ≈ prefix_ms
        self.lam_logits = nn.Parameter(torch.tensor([2.0, -1.0, -1.0, -1.0]))
        self.lam_beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.lam_eps = nn.Parameter(torch.tensor(float(eps_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_x, s_z, s_x = self._ms_tensors(x)
        lam_z = _tanh2_second_deriv(s_z)
        lam_x = _tanh2_second_deriv(s_x)
        lam_d1 = _lam_ms_d1(s_z, self.lam_beta)
        lam_eps = _tanh2_second_deriv_eps(s_z, self.lam_eps.clamp(min=0.0))
        w = F.softmax(self.lam_logits, dim=0)
        lam = (
            w[0] * lam_z
            + w[1] * lam_x
            + w[2] * lam_d1
            + w[3] * lam_eps
        )
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiPrefixMultiScalePtCore(_MsPrefixBase):
    """prefix_ms + hybrid λ: blend context ψ''(s_z) with local per-token ψ''(s_x)."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        mix_z_init: float = 2.0,
    ) -> None:
        super().__init__(
            dim, num_units=num_units, pool_dim=pool_dim, window_size=window_size
        )
        # init favors context λ (≈prefix_ms); model can open local path
        self.lam_mix_logit = nn.Parameter(torch.tensor(float(mix_z_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_x, s_z, s_x = self._ms_tensors(x)
        lam_z = _tanh2_second_deriv(s_z)
        lam_x = _tanh2_second_deriv(s_x)
        w_z = torch.sigmoid(self.lam_mix_logit)
        lam = w_z * lam_z + (1.0 - w_z) * lam_x
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiPrefixMultiScaleGluCore(_MsPrefixBase):
    """prefix_ms + per-token unit gate on spectral branch (GLU-style routing)."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        gate_bias_init: float = 2.0,
    ) -> None:
        super().__init__(
            dim, num_units=num_units, pool_dim=pool_dim, window_size=window_size
        )
        # small scale + positive bias → gate≈0.88 at init, near prefix_ms
        self.gate_scale = nn.Parameter(torch.full((self.num_units,), 0.01))
        self.gate_bias = nn.Parameter(
            torch.full((self.num_units,), float(gate_bias_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_x, s_z, s_x = self._ms_tensors(x)
        lam = _tanh2_second_deriv(s_z)
        gate = torch.sigmoid(self.gate_scale * s_x + self.gate_bias)
        return torch.einsum("...m,md->...d", lam * gate * proj_x, self.weight)


class _PhiPrefixMultiScalePtGluCore(_MsPrefixBase):
    """prefix_ms_pt hybrid λ + unit-level SwiGLU gate."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        mix_z_init: float = 2.0,
    ) -> None:
        super().__init__(
            dim, num_units=num_units, pool_dim=pool_dim, window_size=window_size
        )
        self.lam_mix_logit = nn.Parameter(torch.tensor(float(mix_z_init)))
        self.gate_scale = nn.Parameter(torch.full((self.num_units,), 0.01))
        self.gate_bias = nn.Parameter(torch.full((self.num_units,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_x, s_z, s_x = self._ms_tensors(x)
        lam_z = _tanh2_second_deriv(s_z)
        lam_x = _tanh2_second_deriv(s_x)
        w_z = torch.sigmoid(self.lam_mix_logit)
        lam = w_z * lam_z + (1.0 - w_z) * lam_x
        gate = torch.sigmoid(self.gate_scale * s_x + self.gate_bias)
        return torch.einsum("...m,md->...d", lam * gate * proj_x, self.weight)


class _PhiPrefixMultiScaleRegCore(nn.Module):
    """prefix_ms + ψ'''-gated blend of ψ'' and ψ''+β|ψ'|."""

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        pool_dim: int = 32,
        window_size: int = 64,
        beta_init: float = 0.1,
        d3_tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pool_dim = int(pool_dim)
        self.window_size = int(window_size)
        self.d3_tau = float(d3_tau)
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        self.q_proj = nn.Linear(dim, self.pool_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.pool_dim, bias=False)
        self.scale_logits = nn.Parameter(torch.tensor([2.0, 0.5, -1.0]))
        self.hp_logit = nn.Parameter(torch.tensor(2.0))
        self.lam_beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.sharp_logit = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_attn = _causal_window_attn_pool(
            x, self.q_proj, self.k_proj, self.window_size
        )
        z_win = _causal_window_mean(x, self.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(self.scale_logits, dim=0)
        z = (
            scale_w[0] * z_attn
            + scale_w[1] * z_win
            + scale_w[2] * z_long
        )
        gamma = torch.sigmoid(self.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w = self.weight.transpose(-2, -1)
        proj_x = proj_in @ w
        s_z = z @ w + self.bias
        lam = _lam_ms_regulated(s_z, self.lam_beta, self.sharp_logit, self.d3_tau)
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiPrefixEMASpectralCore(nn.Module):
    """H4_phi: λ from causal EMA(z_t) on x, map still uses x_t."""

    def __init__(self, dim: int, *, num_units: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        # init ρ≈0.9
        self.ema_logit = nn.Parameter(torch.tensor(2.1972245773362196))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho = torch.sigmoid(self.ema_logit)
        z = _causal_ema(x, rho)
        proj_x = x @ self.weight.transpose(-2, -1)
        s_z = z @ self.weight.transpose(-2, -1) + self.bias
        lam = _tanh2_second_deriv(s_z)
        return torch.einsum("...m,md->...d", lam * proj_x, self.weight)


class _PhiCausalSTCore(nn.Module):
    """Causal spatio-temporal: temporal smooth x̃, EMA summary z, λ(z), proj(x̃)."""

    def __init__(self, dim: int, *, num_units: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.num_units = int(num_units) if num_units is not None else 4 * dim
        self.weight = nn.Parameter(torch.randn(self.num_units, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_units))
        # init ≈ α=0.25 temporal mix, ρ=0.9 EMA
        self.alpha_t = nn.Parameter(torch.tensor(-1.0986122886681098))
        self.ema_logit = nn.Parameter(torch.tensor(2.1972245773362196))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha_t)
        rho = torch.sigmoid(self.ema_logit)
        x_smooth = _causal_temporal_smooth(x, alpha)
        z = _causal_ema(x_smooth, rho)
        w = self.weight.transpose(-2, -1)
        proj = x_smooth @ w
        s_z = z @ w + self.bias
        lam = _tanh2_second_deriv(s_z)
        return torch.einsum("...m,md->...d", lam * proj, self.weight)


class _PhiBlockDiagSpectralCore(nn.Module):
    """Block-diagonal Hessian: dim split into num_blocks, each block has independent spectral map.

    Each block uses its own prefix-mean z for lambda computation, allowing block-level
    context-dependent curvature while keeping cross-block interactions zero (block-diagonal).
    """

    def __init__(
        self,
        dim: int,
        *,
        num_units: int | None = None,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        assert dim % num_blocks == 0, f"dim {dim} must be divisible by num_blocks {num_blocks}"
        self.block_dim = dim // num_blocks

        # Total units split across blocks
        total_units = num_units if num_units is not None else 6 * dim
        # Ensure units divisible by blocks for simplicity
        assert total_units % num_blocks == 0, "num_units must be divisible by num_blocks"
        self.units_per_block = total_units // num_blocks

        # Per-block weights: each block has (units_per_block, block_dim)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.units_per_block, self.block_dim) * 0.02)
            for _ in range(num_blocks)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.units_per_block))
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> split into blocks, process each, concatenate."""
        b, t, d = x.shape
        assert d == self.dim

        # Split x into blocks along last dimension: list of (B, T, block_dim)
        x_blocks = torch.chunk(x, self.num_blocks, dim=-1)

        out_blocks = []
        for blk_idx, x_b in enumerate(x_blocks):
            # Causal prefix mean within this block
            cum = torch.cumsum(x_b, dim=1)
            t_idx = torch.arange(1, t + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
            z_b = cum / t_idx  # (B, T, block_dim)

            w = self.weights[blk_idx].transpose(-2, -1)  # (block_dim, units_per_block)
            proj_x = x_b @ w  # (B, T, units_per_block)
            s_z = z_b @ w + self.biases[blk_idx]  # (B, T, units_per_block)

            lam = _tanh2_second_deriv(s_z)
            out_b = torch.einsum("...m,md->...d", lam * proj_x, self.weights[blk_idx])
            out_blocks.append(out_b)

        # Concatenate block outputs
        return torch.cat(out_blocks, dim=-1)


class _LowRankStack(nn.Module):
    """L layers of GeLU(x @ U U^T) with rank r (params ≈ L·r·d)."""

    def __init__(self, dim: int, *, rank: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.randn(dim, rank) * 0.02) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for u in self.factors:
            w = u @ u.transpose(-2, -1)
            h = F.gelu(h @ w)
        return h


class HessianSpectralCore(nn.Module):
    """Constant-curvature map x @ W; H2 adds per-token diagonal scaling."""

    def __init__(self, dim: int, *, variant: HessianVariant = "h1") -> None:
        super().__init__()
        self.variant = variant
        self.potential = _SymmetricPotential(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.potential.weight_matrix()
        h = x @ w
        if self.variant == "h2":
            h = h * _tanh2_hessian_diag(x)
        return h


class _ResidualWideReadout(nn.Module):
    """``Linear(h) + branch(h)``; branch last layer zero-init → step-0 ≈ multwin readout."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        ff = 4 * dim
        self.base = nn.Linear(dim, dim)
        self.branch = nn.Sequential(
            nn.Linear(dim, ff),
            nn.GELU(),
            nn.Linear(ff, dim),
        )
        nn.init.zeros_(self.branch[2].weight)
        nn.init.zeros_(self.branch[2].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.base(h) + self.branch(h)


def _build_readout(dim: int, variant: HessianVariant) -> nn.Module | None:
    if variant not in _HAS_READOUT:
        return None
    if variant in (
        "h4_phi768_prefix_ms_multwin_resread",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_resread",
    ):
        return _ResidualWideReadout(dim)
    if variant in (
        "h3_a",
        "h4_phi768_prefix_ms_multwin_wide",
        "h4_phi1536_prefix_ms_multwin_wide",
        "h4_phi768_prefix_ms_multwin_invcombo_wide",
        "h4_phi768_prefix_ms_multwin_invcombo_strong_wide",
        "h4_phi768_prefix_ms_multwin_invcombo_deep_wide",
        "h4_phi768_prefix_ms_multwin_invcombo_ladder_wide",
    ):
        ff = 4 * dim
        return nn.Sequential(
            nn.Linear(dim, ff),
            nn.GELU(),
            nn.Linear(ff, dim),
        )
    return nn.Linear(dim, dim)


class HessianFFN(nn.Module):
    """FFN slot replacing standard 4d MLP."""

    def __init__(
        self,
        dim: int,
        *,
        variant: HessianVariant = "h1",
        topk_k: int = 16,
        stack_rank: int = 32,
        stack_layers: int = 3,
        dynamic_rank: int = 32,
        lambda_geometry: DecoupleMinLambda | None = None,
        num_units: int | None = None,
        calib_mode: str | None = None,
        uav_raw_context: bool | None = None,
        mix_init: float | None = None,
        tanh2_temperature: float | None = None,
        tanh2_amplitude: str | None = None,
        tanh2_amplitude_scale: float | None = None,
        tanh2_peak_centers: tuple[float, ...] | list[float] | None = None,
        tanh2_peak_shift: float | None = None,
        hessian_factor_temperature: float | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant
        self._lambda_geometry = lambda_geometry
        self._num_units = num_units
        self._calib_mode = calib_mode
        self._uav_raw_context = uav_raw_context
        self._mix_init = mix_init
        self._tanh2_temperature = (
            1.0 if tanh2_temperature is None else float(tanh2_temperature)
        )
        self._tanh2_amplitude = "paper" if tanh2_amplitude is None else str(tanh2_amplitude)
        self._tanh2_amplitude_scale = (
            1.0 if tanh2_amplitude_scale is None else float(tanh2_amplitude_scale)
        )
        self._tanh2_peak_centers = (
            None
            if not tanh2_peak_centers
            else tuple(float(c) for c in tanh2_peak_centers)
        )
        self._tanh2_peak_shift = 0.0 if tanh2_peak_shift is None else float(tanh2_peak_shift)
        self._hessian_factor_temperature = (
            1.0
            if hessian_factor_temperature is None
            else float(hessian_factor_temperature)
        )
        self.rank_core: _RankSpectralCore | None = None
        self.phi_core: (
            _PhiSpectralCore
            | _PhiPrefixSpectralCore
            | _PhiPrefixWindowSpectralCore
            | _PhiPrefixAttnSpectralCore
            | _PhiPrefixAttnEpsSpectralCore
            | _PhiPrefixMultiScaleCore
            | _PhiPrefixMultiScaleD1Core
            | _PhiPrefixMultiScaleRegCore
            | _PhiPrefixMultiScalePtCore
            | _PhiPrefixMultiScaleGluCore
            | _PhiPrefixMultiScalePtGluCore
            | _PhiPrefixMsDecoupleCtxSplitCore
            | _PhiPrefixEMASpectralCore
            | _PhiCausalSTCore
            | _PhiBlockDiagSpectralCore
            | _PhiPrefixMoECore
            | _PhiPrefixLambdaGlobalCore
            | _PhiPrefixMsStatsCore
            | _PhiPrefixMsMultwinCore
            | _PhiPrefixMsMultwinHybridCore
            | _PhiPrefixMsMultwinDsegCore
            | _PhiPrefixMsMultwinSoftplusCore
            | _PhiPrefixMsMultwinTempCore
            | _PhiPrefixMsMultwinActivCore
            | _PhiPrefixMsMultwinLadderDecoupleCtxSplitCore
            | None
        ) = None

        if variant == "h3_stack":
            self.stack: nn.Module | None = _LowRankStack(
                dim, rank=stack_rank, n_layers=stack_layers
            )
            self.spectral = None
        elif variant == "h4_rank":
            self.stack = None
            self.spectral = None
            self.rank_core = _RankSpectralCore(dim, rank=4 * dim)
        elif variant == "h4_phi":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiSpectralCore(dim, num_units=4 * dim)
        elif variant == "h4_phi768":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiSpectralCore(dim, num_units=6 * dim)
        elif variant == "h4_phi768_prefix":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixSpectralCore(dim, num_units=6 * dim)
        elif variant == "h4_phi768_prefix_ema":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixEMASpectralCore(dim, num_units=6 * dim)
        elif variant == "h4_phi768_prefix_win":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixWindowSpectralCore(dim, num_units=6 * dim, window_size=64)
        elif variant == "h4_phi768_prefix_attn":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixAttnSpectralCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_prefix_attn_eps":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixAttnEpsSpectralCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_prefix_ms":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMultiScaleCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_prefix_ms_decouple":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsDecoupleCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_prefix_ms_decouple_ctxsplit":
            self.stack = None
            self.spectral = None
            lg_ctx: DecoupleMinLambda = (
                self._lambda_geometry
                if self._lambda_geometry is not None
                else "global_tanh2"
            )
            self.phi_core = _PhiPrefixMsDecoupleCtxSplitCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                pool_dim=32,
                window_size=64,
                lambda_geometry=lg_ctx,
                native_mix_init=float(
                    self._mix_init if self._mix_init is not None else -10.0
                ),
                tanh2_temperature=float(self._tanh2_temperature),
            )
            self.phi_core.tanh2_amplitude = self._tanh2_amplitude
            self.phi_core.tanh2_amplitude_scale = float(self._tanh2_amplitude_scale)
            self.phi_core.tanh2_peak_centers = self._tanh2_peak_centers
            self.phi_core.tanh2_peak_shift = float(self._tanh2_peak_shift)
        elif variant == "h4_phi768_prefix_ms_decouple_ctxsplit_x_tanh":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsDecoupleCtxSplitXTanhCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                pool_dim=32,
                window_size=64,
            )
        elif variant == "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_agg":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsDecoupleCtxSplitGeluAggCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                pool_dim=32,
                window_size=64,
            )
        elif variant == "h4_phi768_prefix_ms_decouple_ctxsplit_mod_native_gelu_z":
            self.stack = None
            self.spectral = None
            lam_geom = self._lambda_geometry or "proj_oracle_mod_native_ortho"
            self.phi_core = _PhiPrefixMsDecoupleCtxSplitModNativeGeluZCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                pool_dim=32,
                window_size=64,
                lambda_geometry=lam_geom,  # type: ignore[arg-type]
                native_mix_init=float(
                    self._mix_init if self._mix_init is not None else -10.0
                ),
            )
            self.phi_core.hessian_factor_temperature = float(
                self._hessian_factor_temperature
            )
        elif variant == "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsDecoupleCtxSplitFullGeluZCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                pool_dim=32,
                window_size=64,
            )
        elif variant == "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z_active_rank":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsDecoupleCtxSplitFullGeluZActiveRankCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                pool_dim=32,
                window_size=64,
            )
        elif variant == "h4_phi768_prefix_ms_decouple_dual":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsDecoupleDualCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_prefix_ms_decouple_seq":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsDecoupleSeqCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_prefix_ms_stats":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsStatsCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_prefix_ms_multwin":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64)
            )
        elif variant == "h4_phi768_prefix_ms_multwin_deep":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinCore(
                dim,
                num_units=6 * dim,
                window_size=64,
                window_sizes=MULTWIN_DEEP_WINDOWS,
            )
        elif variant == "h4_phi768_prefix_ms_multwin_dseg":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinDsegCore(
                dim,
                num_units=6 * dim,
                window_size=64,
                window_sizes=(4, 16, 64),
                n_segments=8,
            )
        elif variant == "h4_phi768_prefix_ms_multwin_wide":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64)
            )
        elif variant == "h4_phi768_prefix_ms_multwin_resread":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64)
            )
        elif variant == "h4_phi768_prefix_ms_multwin_softplus":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinSoftplusCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64)
            )
        elif variant == "h4_phi768_prefix_ms_multwin_temp":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinTempCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64), temp_init=2.0
            )
        elif variant == "h4_phi768_prefix_ms_multwin_rational":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="rational", activ_param=0.5
            )
        elif variant == "h4_phi768_prefix_ms_multwin_shiftedsp":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="shifted_sp", activ_param=2.0
            )
        elif variant == "h4_phi768_prefix_ms_multwin_mish":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="mish"
            )
        elif variant == "h4_phi768_prefix_ms_multwin_swish":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="swish"
            )
        elif variant == "h4_phi768_prefix_ms_multwin_signedfloor":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="signedfloor"
            )
        elif variant == "h4_phi768_prefix_ms_multwin_tanhsp":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="tanhsp"
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="invcombo"
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_strong":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="invcombo_strong"
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_strong_wide":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="invcombo_strong"
            )
        elif variant in (
            "h4_phi768_prefix_ms_multwin_invcombo_deep",
            "h4_phi768_prefix_ms_multwin_invcombo_deep_wide",
        ):
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="invcombo_deep"
            )
        elif variant in (
            "h4_phi768_prefix_ms_multwin_invcombo_ladder",
            "h4_phi768_prefix_ms_multwin_invcombo_ladder_wide",
        ):
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="invcombo_ladder"
            )
        elif variant in (
            "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_small",
            "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_mid",
            "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_learn",
            "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate",
            "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m10",
            "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m20",
            "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m35",
            "h4_phi768_prefix_ms_multwin_invcombo_ladder8_odd_m20",
            "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_m50",
            "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_hot_m50",
            "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_amp2_hot_m50",
        ):
            self.stack = None
            self.spectral = None
            odd_activ = {
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_small": "invcombo_ladder_odd_small",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_mid": "invcombo_ladder_odd_mid",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_learn": "invcombo_ladder_odd_learn",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate": "invcombo_ladder_oddgate",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m10": "invcombo_ladder6_odd_m10",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m20": "invcombo_ladder6_odd_m20",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m35": "invcombo_ladder6_odd_m35",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder8_odd_m20": "invcombo_ladder8_odd_m20",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_m50": "invcombo_ladder_oddgate_m50",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_hot_m50": "invcombo_ladder_oddgate_hot_m50",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_amp2_hot_m50": "invcombo_ladder_oddgate_amp2_hot_m50",
            }[variant]
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ=odd_activ,
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_ladder_gain":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="invcombo_ladder",
                s_z_rms_norm=True,
                s_z_gain_init=1.5,
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_ladder_gain_lammix":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="invcombo_ladder",
                s_z_rms_norm=True,
                s_z_gain_init=1.5,
                s_z_gain_mix="lambda",
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_ladder_addodd_t10":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinLadderMixCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                mix_mode="addodd_tanh", odd_scale=0.10,
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_ladder_addodd_delta_t10":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinLadderMixCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                mix_mode="addodd_delta", odd_scale=0.10,
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_ladder_mix_addodd_delta":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinLadderMixCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                mix_mode="mix_addodd_delta", odd_scale=0.10, sx_mix=0.25,
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_ladder_cross_r64":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinLadderCrossCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                cross_rank=64,
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_ladder_bilinear_r64":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinLadderBilinearCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                bilinear_rank=64,
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_ladder_decouple":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinLadderDecoupleCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_ladder_decouple_ctxsplit":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinLadderDecoupleCtxSplitCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                window_size=64,
                window_sizes=(4, 16, 64),
            )
        elif variant == "h4_phi768_global_ms":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiGlobalMsMultwinCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
            )
        elif variant == "h4_phi768_global_decouple":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiGlobalMsMultwinDecoupleCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
            )
        elif variant == "h4_phi768_global_light_decouple":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiGlobalLightDecoupleCore(
                dim, num_units=6 * dim, local_window=4,
            )
        elif variant == "h4_phi768_global_decouple_min":
            self.stack = None
            self.spectral = None
            lg: DecoupleMinLambda = (
                self._lambda_geometry
                if self._lambda_geometry is not None
                else "proj_oracle"
            )
            self.phi_core = _PhiGlobalDecoupleMinCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                lambda_geometry=lg,
            )
            self.phi_core.tanh2_temperature = float(self._tanh2_temperature)
            self.phi_core.tanh2_amplitude = str(self._tanh2_amplitude)
        elif variant in (
            "h4_phi768_global_decouple_min_side",
            "h4_phi768_global_decouple_min_side_up",
            "h4_phi768_global_decouple_min_side_curv",
            "h4_phi768_global_decouple_min_side_units",
        ):
            self.stack = None
            self.spectral = None
            lg_side: DecoupleMinLambda = (
                self._lambda_geometry
                if self._lambda_geometry is not None
                else "proj_oracle_fast"
            )
            if variant == "h4_phi768_global_decouple_min_side_units":
                side_cls = _PhiGlobalDecoupleMinSideUnitsCore
            elif variant == "h4_phi768_global_decouple_min_side_curv":
                side_cls = _PhiGlobalDecoupleMinSideCurvCore
            elif variant == "h4_phi768_global_decouple_min_side_up":
                side_cls = _PhiGlobalDecoupleMinSideUpCore
            else:
                side_cls = _PhiGlobalDecoupleMinSideCore
            self.phi_core = side_cls(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                lambda_geometry=lg_side,
            )
        elif variant == "h4_phi768_global_decouple_min_abroadcast":
            self.stack = None
            self.spectral = None
            lg_ab: DecoupleMinLambda = (
                self._lambda_geometry
                if self._lambda_geometry is not None
                else "proj_oracle_fast"
            )
            self.phi_core = _PhiGlobalDecoupleMinABroadcastCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                lambda_geometry=lg_ab,
            )
        elif variant == "h4_phi768_global_decouple_min_resonance":
            self.stack = None
            self.spectral = None
            lg_res: DecoupleMinLambda = (
                self._lambda_geometry
                if self._lambda_geometry is not None
                else "proj_oracle_fast"
            )
            self.phi_core = _PhiGlobalDecoupleMinResonanceCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                lambda_geometry=lg_res,
            )
        elif variant == "h4_phi768_global_decouple_min_offaxis":
            self.stack = None
            self.spectral = None
            lg_off: DecoupleMinLambda = (
                self._lambda_geometry
                if self._lambda_geometry is not None
                else "proj_oracle_fast"
            )
            self.phi_core = _PhiGlobalDecoupleMinOffaxisCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                lambda_geometry=lg_off,
            )
        elif variant == "h4_phi768_global_decouple_min_active_rank":
            self.stack = None
            self.spectral = None
            lg_ar: DecoupleMinLambda = (
                self._lambda_geometry
                if self._lambda_geometry is not None
                else "proj_oracle_fast"
            )
            self.phi_core = _PhiGlobalDecoupleMinActiveRankCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                lambda_geometry=lg_ar,
            )
        elif variant == "h4_phi768_global_decouple_min_native_calib":
            self.stack = None
            self.spectral = None
            lg_nc: DecoupleMinLambda = (
                self._lambda_geometry
                if self._lambda_geometry is not None
                else "proj_native_calib"
            )
            self.phi_core = _PhiGlobalDecoupleMinNativeCalibCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                lambda_geometry=lg_nc,
                calib_mode=str(self._calib_mode or "grid_fit"),
            )
        elif variant == "h4_phi768_global_decouple_min_uav":
            self.stack = None
            self.spectral = None
            lg_uav: DecoupleMinLambda = (
                self._lambda_geometry
                if self._lambda_geometry is not None
                else "proj_native_calib"
            )
            self.phi_core = _PhiGlobalDecoupleMinUavCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                lambda_geometry=lg_uav,
                calib_mode=str(self._calib_mode or "grid_fit"),
                use_raw_context=bool(self._uav_raw_context if self._uav_raw_context is not None else True),
                mix_init=float(self._mix_init if self._mix_init is not None else -4.0),
            )
        elif variant == "h4_phi768_global_decouple_min_2d":
            self.stack = None
            self.spectral = None
            lg_2d: DecoupleMinLambda = (
                self._lambda_geometry
                if self._lambda_geometry is not None
                else "proj_oracle_fast"
            )
            self.phi_core = _PhiGlobalDecoupleMin2dCore(
                dim,
                num_units=self._num_units if self._num_units is not None else 6 * dim,
                lambda_geometry=lg_2d,
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_ladder_resread":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="invcombo_ladder"
            )
        elif variant == "h4_phi768_prefix_ms_multwin_invcombo_wide":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinActivCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64),
                activ="invcombo"
            )
        elif variant == "h4_phi1536_prefix_ms_multwin":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinCore(
                dim, num_units=12 * dim, window_size=64, window_sizes=(4, 16, 64)
            )
        elif variant == "h4_phi1536_prefix_ms_multwin_wide":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinCore(
                dim, num_units=12 * dim, window_size=64, window_sizes=(4, 16, 64)
            )
        elif variant == "h4_phi2304_prefix_ms_multwin":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinCore(
                dim, num_units=18 * dim, window_size=64, window_sizes=(4, 16, 64)
            )
        elif variant == "h4_phi3072_prefix_ms_multwin":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinCore(
                dim, num_units=24 * dim, window_size=64, window_sizes=(4, 16, 64)
            )
        elif variant == "h4_phi768_prefix_ms_multwin_hybrid":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMsMultwinHybridCore(
                dim, num_units=6 * dim, window_size=64, window_sizes=(4, 16, 64)
            )
        elif variant == "h4_phi768_prefix_ms_d1":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMultiScaleD1Core(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_prefix_ms_reg":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMultiScaleRegCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_prefix_ms_pt":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMultiScalePtCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_prefix_ms_glu":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMultiScaleGluCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_prefix_ms_pt_glu":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMultiScalePtGluCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h4_phi768_st":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiCausalSTCore(dim, num_units=6 * dim)
        elif variant == "h4_phi768_block_diag":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiBlockDiagSpectralCore(
                dim, num_units=6 * dim, num_blocks=8
            )
        elif variant == "h4_phi768_prefix_moe":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixMoECore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64, top_k=2
            )
        elif variant == "h4_phi768_prefix_lam_global":
            self.stack = None
            self.spectral = None
            self.phi_core = _PhiPrefixLambdaGlobalCore(
                dim, num_units=6 * dim, pool_dim=32, window_size=64
            )
        elif variant == "h3_form_a":
            self.stack = None
            self.spectral = None
            self.rank_core = _RankSpectralCore(dim, rank=dim)
        else:
            self.stack = None
            core_v: HessianVariant = "h2" if variant == "h2" else "h1"
            self.spectral = (
                HessianSpectralCore(dim, variant=core_v)
                if variant in _H3_SPECTRAL or variant in ("h1", "h2")
                else None
            )

        self.readout = _build_readout(dim, variant)
        self._spectral_act = F.silu if variant in _SWISH_SPECTRAL else F.gelu
        self.topk_branch: nn.Module | None = None
        if variant == "h3_topk":
            self.topk_k = topk_k
        elif variant == "h3_b":
            self.topk_branch = _LearnableTopKBranch(dim, k=topk_k)
        elif variant == "h3_c":
            self.topk_branch = _DynamicLowRankBranch(dim, rank=dynamic_rank)

    def _spectral_main(self, x: torch.Tensor) -> torch.Tensor:
        act = self._spectral_act
        if self.stack is not None:
            return self.stack(x)
        if self.rank_core is not None:
            return act(self.rank_core(x))
        if self.phi_core is not None:
            h = self.phi_core(x)
            if self.variant in _PHI_DIRECT_OUTPUT:
                return h
            return act(h)
        assert self.spectral is not None
        return act(self.spectral(x))

    def _main_path(self, x: torch.Tensor) -> torch.Tensor:
        h = self._spectral_main(x)
        if self.readout is not None:
            h = self.readout(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.variant in _H3_SPECTRAL
            or self.variant == "h3_stack"
            or self.variant in (
                "h4_rank",
                "h4_phi",
                "h4_phi768",
                "h4_phi768_prefix",
                "h4_phi768_prefix_ema",
                "h4_phi768_prefix_win",
                "h4_phi768_prefix_attn",
                "h4_phi768_prefix_attn_eps",
                "h4_phi768_prefix_ms",
                "h4_phi768_prefix_ms_decouple",
                "h4_phi768_prefix_ms_decouple_ctxsplit",
                "h4_phi768_prefix_ms_decouple_ctxsplit_x_tanh",
                "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_agg",
                "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z",
                "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z_active_rank",
                "h4_phi768_prefix_ms_decouple_dual",
                "h4_phi768_prefix_ms_decouple_seq",
                "h4_phi768_prefix_ms_stats",
                "h4_phi768_prefix_ms_multwin",
                "h4_phi768_prefix_ms_multwin_deep",
                "h4_phi768_prefix_ms_multwin_dseg",
                "h4_phi768_prefix_ms_multwin_wide",
                "h4_phi768_prefix_ms_multwin_resread",
                "h4_phi1536_prefix_ms_multwin",
                "h4_phi1536_prefix_ms_multwin_wide",
                "h4_phi2304_prefix_ms_multwin",
                "h4_phi3072_prefix_ms_multwin",
                "h4_phi768_prefix_ms_multwin_hybrid",
                "h4_phi768_prefix_ms_d1",
                "h4_phi768_prefix_ms_reg",
                "h4_phi768_prefix_ms_pt",
                "h4_phi768_prefix_ms_glu",
                "h4_phi768_prefix_ms_pt_glu",
                "h4_phi768_st",
                "h4_phi768_block_diag",
                "h4_phi768_prefix_moe",
                "h4_phi768_prefix_lam_global",
                "h3_form_a",
                "h4_phi768_prefix_ms_multwin_softplus",
                "h4_phi768_prefix_ms_multwin_temp",
                "h4_phi768_prefix_ms_multwin_rational",
                "h4_phi768_prefix_ms_multwin_shiftedsp",
                "h4_phi768_prefix_ms_multwin_mish",
                "h4_phi768_prefix_ms_multwin_swish",
                "h4_phi768_prefix_ms_multwin_signedfloor",
                "h4_phi768_prefix_ms_multwin_tanhsp",
                "h4_phi768_prefix_ms_multwin_invcombo",
                "h4_phi768_prefix_ms_multwin_invcombo_strong",
                "h4_phi768_prefix_ms_multwin_invcombo_deep",
                "h4_phi768_prefix_ms_multwin_invcombo_wide",
                "h4_phi768_prefix_ms_multwin_invcombo_strong_wide",
                "h4_phi768_prefix_ms_multwin_invcombo_deep_wide",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_small",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_mid",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_odd_learn",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m10",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m20",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder6_odd_m35",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder8_odd_m20",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_m50",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_hot_m50",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_oddgate_amp2_hot_m50",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_gain",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_gain_lammix",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_addodd_t10",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_addodd_delta_t10",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_mix_addodd_delta",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_cross_r64",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_bilinear_r64",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_decouple",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_decouple_ctxsplit",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_wide",
                "h4_phi768_prefix_ms_multwin_invcombo_ladder_resread",
                "h4_phi768_global_ms",
                "h4_phi768_global_decouple",
                "h4_phi768_global_light_decouple",
                "h4_phi768_global_decouple_min",
                "h4_phi768_global_decouple_min_side",
                "h4_phi768_global_decouple_min_side_up",
                "h4_phi768_global_decouple_min_side_curv",
                "h4_phi768_global_decouple_min_side_units",
                "h4_phi768_global_decouple_min_abroadcast",
                "h4_phi768_global_decouple_min_resonance",
                "h4_phi768_global_decouple_min_offaxis",
                "h4_phi768_global_decouple_min_active_rank",
                "h4_phi768_global_decouple_min_native_calib",
                "h4_phi768_global_decouple_min_uav",
                "h4_phi768_global_decouple_min_2d",
            )
        ):
            out = self._main_path(x)
            if self.variant == "h3_topk":
                out = out + F.gelu(_topk_fixed_tanh2_branch(x, self.topk_k))
            elif self.variant == "h3_b" and self.topk_branch is not None:
                out = out + F.gelu(self.topk_branch(x))
            elif self.variant == "h3_c" and self.topk_branch is not None:
                out = out + F.gelu(self.topk_branch(x))
            return out
        assert self.spectral is not None
        return F.gelu(self.spectral(x))


FullHessianCore = HessianSpectralCore
