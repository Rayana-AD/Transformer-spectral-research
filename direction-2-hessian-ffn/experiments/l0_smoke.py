#!/usr/bin/env python3
"""L0 smoke tests: gradients, numerics, attention diagnostics, Hessian timing."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tlb.attention.angular import compute_attention, vanilla_attention_scores
from tlb.attention.diagnostics import attention_diagnostics
from tlb.ffn.hessian_core import HessianFFN, HessianSpectralCore, nonlinear_potential
from tlb.training.utils import resolve_device, save_json


def _check_gradients(model: torch.nn.Module, x: torch.Tensor) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = out.norm()
    loss.backward()
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            assert torch.isfinite(g).all(), "non-finite gradient detected"
            norms.append(float(g.norm().item()))
    return {
        "loss": float(loss.item()),
        "grad_norm_mean": sum(norms) / max(len(norms), 1),
        "grad_norm_max": max(norms) if norms else 0.0,
        "num_params_with_grad": len(norms),
    }


def test_attention(device: torch.device) -> dict:
    b, h, t, d = 2, 4, 16, 32
    q = torch.randn(b, h, t, d, device=device, requires_grad=True)
    k = torch.randn(b, h, t, d, device=device, requires_grad=True)
    v = torch.randn(b, h, t, d, device=device, requires_grad=True)

    results: dict = {}
    for attn_type in ("vanilla", "cosine", "angular"):
        out, attn = compute_attention(
            q, k, v, attention_type=attn_type, alpha=2.0, normalize_mode="row"
        )
        loss = out.sum()
        loss.backward(retain_graph=True)
        diag = attention_diagnostics(attn.detach())
        if attn_type != "vanilla":
            row_sums = attn.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
        results[attn_type] = {
            "output_finite": bool(torch.isfinite(out).all().item()),
            "diagnostics": diag,
        }
        q.grad = k.grad = v.grad = None

    q2 = torch.randn(b, h, t, d, device=device)
    k2 = torch.randn(b, h, t, d, device=device)
    logits = vanilla_attention_scores(q2, k2)
    softmax_attn = F.softmax(logits, dim=-1)
    _, angular_attn = compute_attention(
        q2, k2, q2, attention_type="angular", alpha=2.0, normalize_mode="row"
    )
    results["comparison"] = {
        "softmax": attention_diagnostics(softmax_attn),
        "angular": attention_diagnostics(angular_attn),
    }
    return results


def test_draft_identity_bug(device: torch.device, dim: int = 32, seq: int = 16) -> dict:
    """Show draft recon_x = (x @ Q) @ Q.T is identity; spectral uses diag(λ)."""
    x = torch.randn(seq, dim, device=device)
    z = x.mean(0).detach().cpu().requires_grad_(True)
    h = torch.autograd.functional.hessian(nonlinear_potential, z)
    h = h.to(device=device, dtype=x.dtype)
    evals, evecs = torch.linalg.eigh(h.float().cpu())
    evals = evals.to(device=device, dtype=x.dtype)
    evecs = evecs.to(device=device, dtype=x.dtype)

    wrong = (x @ evecs) @ evecs.transpose(-2, -1)
    coef = x @ evecs
    right = (coef * evals.unsqueeze(0)) @ evecs.transpose(-2, -1)

    wrong_err = float((wrong - x).abs().mean().item())
    right_err = float((right - x).abs().mean().item())
    return {
        "draft_recon_equals_x_mean_abs_err": wrong_err,
        "spectral_nonidentity_mean_abs_delta": right_err,
        "draft_is_identity": wrong_err < 1e-5,
        "spectral_changes_x": right_err > 1e-4,
    }


def _gelu_spectral(core: HessianSpectralCore) -> nn.Module:
    class _Wrap(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.gelu(core(x))

    return _Wrap()


def test_hessian(device: torch.device, dim: int = 64, seq: int = 32) -> dict:
    x = torch.randn(2, seq, dim, device=device)
    results: dict = {"draft_bug": test_draft_identity_bug(device, dim=dim, seq=seq)}

    for mode in ("nonlinear", "h0", "h1"):
        if mode == "nonlinear":
            ffn: nn.Module = HessianFFN(dim).to(device)
        else:
            ffn = _gelu_spectral(HessianSpectralCore(dim, mode=mode).to(device))

        t0 = time.perf_counter()
        out = ffn(x)
        fwd_ms = (time.perf_counter() - t0) * 1000

        if mode in ("nonlinear", "h0"):
            xg = x.clone().detach().requires_grad_(True)
            t0 = time.perf_counter()
            ffn(xg).norm().backward()
            bwd_ms = (time.perf_counter() - t0) * 1000
            grad_info = {
                "input_grad_finite": bool(
                    xg.grad is not None and torch.isfinite(xg.grad).all().item()
                ),
                "learnable_params": sum(p.numel() for p in ffn.parameters()),
            }
        else:
            t0 = time.perf_counter()
            grad_info = _check_gradients(ffn, x.clone())
            bwd_ms = (time.perf_counter() - t0) * 1000

        spectral_only = HessianSpectralCore(dim, mode=mode).to(device)
        spec = spectral_only(x[0])
        spec_delta = float((spec - x[0]).abs().mean().item())

        results[mode] = {
            "forward_ms": fwd_ms,
            "backward_ms": bwd_ms,
            "output_shape": list(out.shape),
            "spectral_mean_abs_delta": spec_delta,
            "gradients": grad_info,
        }

    results["nonlinear_has_nontrivial_spectral"] = (
        results["nonlinear"]["spectral_mean_abs_delta"] > 1e-4
    )
    results["h0_spectral_near_zero"] = results["h0"]["spectral_mean_abs_delta"] < 1e-4
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="L0 smoke diagnostics")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "l0_diagnostics.json",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    report = {
        "device": str(device),
        "attention": test_attention(device),
        "hessian": test_hessian(device, dim=args.dim, seq=args.seq),
    }
    save_json(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
