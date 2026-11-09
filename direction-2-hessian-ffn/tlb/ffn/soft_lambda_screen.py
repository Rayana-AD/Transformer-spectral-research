"""Screen a soft tanh²'' family. tanh²'' stays the gate; temperature is the softness knob.

Softness measures are taken from how Smish / LightDepth differ from a hard peak
(lower slope, wider half-width, fewer sign flips inside the usual pre-activation
window). They are not a license to replace ψ with those activations.
"""
from __future__ import annotations

import torch

from tlb.ffn.hessian_core import _tanh2_second_deriv, tanh2_softened

# Pre-activation window where a swap / early-train gate actually lives.
OPERATING_WINDOW = 2.0
# Signed curvature must still appear inside this wider probe (paper λ is signed).
SIGN_PROBE = 6.0


def lambda_curve(temperature: float, grid: torch.Tensor) -> torch.Tensor:
    return tanh2_softened(grid, temperature)


def softness_metrics(temperature: float, *, grid: torch.Tensor | None = None) -> dict[str, float]:
    if grid is None:
        grid = torch.linspace(-4.0, 4.0, 1601)
    lam = lambda_curve(temperature, grid)
    ds = float(grid[1] - grid[0])
    dlam = (lam[1:] - lam[:-1]).abs() / ds
    lam0 = float(lambda_curve(temperature, torch.zeros(1)).item())
    half = 0.5 * abs(lam0)
    # First |s| where |λ| drops to half the origin peak, scanning outward.
    mid = grid.numel() // 2
    right = lam[mid:]
    xs = grid[mid:]
    below = (right.abs() <= half).nonzero(as_tuple=False)
    half_width = float(xs[int(below[0])]) if below.numel() else float(xs[-1])
    op = grid.abs() <= OPERATING_WINDOW
    wide = torch.linspace(-SIGN_PROBE, SIGN_PROBE, 2401)
    wide_lam = lambda_curve(temperature, wide)
    return {
        "tau": float(temperature),
        "lambda0": lam0,
        "mean_abs_dlambda": float(dlam.mean().item()),
        "half_width": half_width,
        "neg_frac_operating": float((lam[op] < 0).float().mean().item()),
        "neg_frac_full": float((lam < 0).float().mean().item()),
        "signed_lobe": bool((wide_lam < -1e-3).any().item()),
    }


def screen_temperatures(
    temperatures: list[float],
    *,
    grid: torch.Tensor | None = None,
) -> dict:
    """Keep τ=1 as the paper baseline. A τ>1 is worth a run when it is softer
    inside the operating window but still a signed tanh²'' gate.
    """
    rows = [softness_metrics(t, grid=grid) for t in temperatures]
    base = next(r for r in rows if abs(r["tau"] - 1.0) < 1e-8)
    worth: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        if abs(row["tau"] - 1.0) < 1e-8:
            row["decision"] = "baseline"
            continue
        reasons = []
        if abs(row["lambda0"] - base["lambda0"]) > 1e-4:
            reasons.append("lambda0 drifted")
        if not row["signed_lobe"]:
            reasons.append("lost signed tanh2 lobe")
        # Marginal stretches (τ≈1.25) move the curve but are not a different experiment.
        if row["mean_abs_dlambda"] > 0.95 * base["mean_abs_dlambda"]:
            reasons.append("slope drop < 5% vs τ=1")
        if row["half_width"] < 1.5 * base["half_width"]:
            reasons.append("half-width < 1.5× τ=1")
        if row["neg_frac_operating"] >= base["neg_frac_operating"]:
            reasons.append("operating window not quieter")
        if reasons:
            row["decision"] = "reject"
            row["reasons"] = reasons
            rejected.append(row)
        else:
            row["decision"] = "test"
            row["reasons"] = []
            worth.append(row)
    return {"baseline": base, "test": worth, "reject": rejected, "all": rows}
