"""Scalar metrics for soft-λ gate shapes (pre-train Gate-0 style checks)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tlb.ffn.hessian_core import _tanh2_second_deriv
from tlb.ffn.mlp_hessian_analytic import (
    SOFT_LAMBDA_GEOMETRY_FNS,
    gelu_scalar,
    lambda_decouple_geometry,
    lambda_oracle_gelu,
)


def _lambda_fn(name: str):
    if name in SOFT_LAMBDA_GEOMETRY_FNS:
        return SOFT_LAMBDA_GEOMETRY_FNS[name]
    return lambda s: lambda_decouple_geometry(s, name)


def evaluate_lambda_geometry(
    name: str,
    *,
    grid: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compare ``λ(s)·s`` to ``GELU(s)`` and report softness / native overlap."""
    if grid is None:
        grid = torch.linspace(-4.0, 4.0, 1601)
    s = grid
    target = gelu_scalar(s)
    lam_fn = _lambda_fn(name)
    lam = lam_fn(s)
    pred = lam * s
    rmse = float(((pred - target) ** 2).mean().sqrt().item())
    native = _tanh2_second_deriv(s)
    native_norm = native / 2.0
    oracle = lambda_oracle_gelu(s)
    neg_frac = float((lam < 0.0).float().mean().item())
    # Mean |dλ/ds| proxy via finite diff (steepness — lower ⇒ softer gate).
    ds = float(s[1] - s[0])
    dlam = (lam[1:] - lam[:-1]) / ds
    steepness = float(dlam.abs().mean().item())
    return {
        "rmse_vs_gelu": rmse,
        "lambda_at_zero": float(lam_fn(torch.zeros(1)).item()),
        "neg_gate_frac": neg_frac,
        "mean_abs_dlambda": steepness,
        "mean_abs_native_norm": float(native_norm.abs().mean().item()),
        "mean_abs_oracle": float(oracle.abs().mean().item()),
    }


def rank_soft_lambda_candidates(
    names: list[str],
    *,
    grid: torch.Tensor | None = None,
) -> list[dict]:
    rows = [{"name": n, **evaluate_lambda_geometry(n, grid=grid)} for n in names]
    rows.sort(key=lambda r: (r["rmse_vs_gelu"], r["mean_abs_dlambda"]))
    return rows
