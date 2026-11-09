"""Geometry / UV-coupling diagnostics for Hessian FFN (ctxsplit UAV, decouple)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tlb.ffn.hessian_core import HessianFFN, _tanh2_second_deriv


def enable_geometry_diagnostics(model: nn.Module) -> None:
    """Turn on forward geometry probes on supported ``phi_core`` modules."""
    for module in model.modules():
        if hasattr(module, "geom_diag_enabled"):
            module.geom_diag_enabled = True


def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
    t = t.float()
    abs_t = t.abs()
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "abs_mean": float(abs_t.mean().item()),
        "rms": float(t.pow(2).mean().sqrt().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "frac_negative": float((t < 0).float().mean().item()),
        "frac_near_zero": float((abs_t < 1e-3).float().mean().item()),
    }


def _row_cosine(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-8) -> float:
    """Mean cosine similarity between matching rows of ``a`` and ``b``."""
    if a.shape != b.shape or a.ndim != 2:
        return float("nan")
    u = a.norm(dim=1).clamp(min=eps)
    v = b.norm(dim=1).clamp(min=eps)
    cos = (a * b).sum(dim=1) / (u * v)
    return float(cos.mean().item())


def _batch_corr(flat_a: torch.Tensor, flat_b: torch.Tensor) -> float:
    a = flat_a.reshape(-1).float()
    b = flat_b.reshape(-1).float()
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt()
    if float(denom.item()) < 1e-12:
        return float("nan")
    return float((a * b).sum().item() / denom.item())


def probe_ctxsplit_core(core: nn.Module, x: torch.Tensor) -> dict[str, Any]:
    """One forward geometry snapshot for prefix_ms ctxsplit / UAV cores."""
    out: dict[str, Any] = {"core": type(core).__name__}
    with torch.no_grad():
        if hasattr(core, "native_factors"):
            lam, proj_x, s_gate = core.native_factors(x)
            s_content = proj_x
        elif hasattr(core, "_context_z") and hasattr(core, "weight"):
            z, z_win = core._context_z(x)
            gamma = torch.sigmoid(core.hp_logit)
            proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
            w_in = core.weight.transpose(-2, -1)
            a_in = core.context_weight.transpose(-2, -1)
            s_content = proj_in @ w_in
            if hasattr(core, "content_bias"):
                s_content = s_content + core.content_bias
            elif hasattr(core, "bias"):
                s_content = s_content + core.bias
            s_gate = z @ a_in + core.bias
            lam = _tanh2_second_deriv(s_gate)
        else:
            return out

        gated = lam * s_content
        v = core.out_weight
        feat = gated @ v
        feat_coax = gated @ core.weight

        u_norm = core.weight.norm(dim=1).clamp(min=1e-8)
        v_norm = v.norm(dim=1).clamp(min=1e-8)
        uv_cos_rows = (core.weight * v).sum(dim=1) / (u_norm * v_norm)

        delta = feat - feat_coax
        coax_rms = float(feat_coax.pow(2).mean().sqrt().item())
        delta_rms = float(delta.pow(2).mean().sqrt().item())

        active = (lam.abs() > 1e-3).float().mean()

        out.update(
            {
                "s_content": _tensor_stats(s_content),
                "s_gate": _tensor_stats(s_gate),
                "lambda": _tensor_stats(lam),
                "gated": _tensor_stats(gated),
                "s_gate_s_content_corr": _batch_corr(s_gate, s_content),
                "lambda_s_content_corr": _batch_corr(lam, s_content),
                "uv_cos_mean": float(uv_cos_rows.mean().item()),
                "uv_cos_abs_mean": float(uv_cos_rows.abs().mean().item()),
                "feat_rms": float(feat.pow(2).mean().sqrt().item()),
                "feat_coax_rms": coax_rms,
                "decouple_delta_rms": delta_rms,
                "decouple_to_coax_ratio": delta_rms / max(coax_rms, 1e-8),
                "active_unit_frac": float(active.item()),
                "gamma": float(torch.sigmoid(core.hp_logit).item())
                if hasattr(core, "hp_logit")
                else None,
            }
        )
    return out


def collect_channel_grad_norms(core: nn.Module) -> dict[str, float | None]:
    """Per-channel gradient norms after ``backward`` (U / A / V)."""
    rows: dict[str, float | None] = {}
    for name, param in (
        ("u", getattr(core, "weight", None)),
        ("a", getattr(core, "context_weight", None)),
        ("v", getattr(core, "out_weight", None)),
    ):
        if isinstance(param, nn.Parameter) and param.grad is not None:
            rows[f"grad_{name}_norm"] = float(param.grad.detach().norm().item())
        else:
            rows[f"grad_{name}_norm"] = None
    u_g = rows.get("grad_u_norm")
    v_g = rows.get("grad_v_norm")
    a_g = rows.get("grad_a_norm")
    if u_g and v_g and u_g > 0:
        rows["grad_v_to_u_ratio"] = float(v_g / u_g)
    if a_g and u_g and u_g > 0:
        rows["grad_a_to_u_ratio"] = float(a_g / u_g)
    if a_g and v_g and v_g > 0:
        rows["grad_a_to_v_ratio"] = float(a_g / v_g)
    return rows


def collect_geometry_from_last_diag(model: nn.Module, *, layer_tag: str = "") -> dict[str, Any]:
    """Read ``last_diag`` written during the preceding forward pass."""
    layers: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, HessianFFN):
            continue
        core = module.phi_core
        if core is None:
            continue
        diag = getattr(core, "last_diag", None)
        if not diag:
            continue
        row: dict[str, Any] = {
            "module": name,
            "variant": str(module.variant),
            **{k: float(v) for k, v in diag.items() if isinstance(v, (float, int))},
        }
        row.update(collect_channel_grad_norms(core))
        layers.append(row)
    return _summarize_geometry_layers(layers, layer_tag=layer_tag)


def _summarize_geometry_layers(
    layers: list[dict[str, Any]], *, layer_tag: str = ""
) -> dict[str, Any]:
    if not layers:
        return {"layers": 0, "tag": layer_tag}

    def _mean(key: str) -> float | None:
        vals = [float(r[key]) for r in layers if key in r and r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "tag": layer_tag,
        "layers": len(layers),
        "uv_cos_mean": _mean("uv_cos_mean"),
        "decouple_to_coax_ratio": _mean("decouple_to_coax_ratio"),
        "s_gate_s_content_corr": _mean("s_gate_s_content_corr"),
        "lambda_s_content_corr": _mean("lambda_s_content_corr"),
        "active_unit_frac": _mean("active_unit_frac"),
        "grad_u_norm": _mean("grad_u_norm"),
        "grad_a_norm": _mean("grad_a_norm"),
        "grad_v_norm": _mean("grad_v_norm"),
        "grad_a_to_u_ratio": _mean("grad_a_to_u_ratio"),
        "grad_v_to_u_ratio": _mean("grad_v_to_u_ratio"),
        "per_layer": layers,
    }
    if layers and "lam_frac_negative" in layers[0]:
        lam_neg = [float(r["lam_frac_negative"]) for r in layers]
        summary["lambda_frac_negative"] = sum(lam_neg) / len(lam_neg)
    return summary


def collect_geometry_from_model(
    model: nn.Module,
    ffn_input: torch.Tensor | None = None,
    *,
    layer_tag: str = "",
) -> dict[str, Any]:
    """Aggregate geometry over Hessian layers (prefer ``last_diag`` from forward)."""
    if ffn_input is None:
        return collect_geometry_from_last_diag(model, layer_tag=layer_tag)

    layers: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, HessianFFN):
            continue
        core = module.phi_core
        if core is None:
            continue
        row: dict[str, Any] = {"module": name, "variant": str(module.variant)}
        if hasattr(core, "native_factors") or hasattr(core, "context_weight"):
            row.update(probe_ctxsplit_core(core, ffn_input))
        row.update(collect_channel_grad_norms(core))
        layers.append(row)
    return _summarize_geometry_layers(layers, layer_tag=layer_tag)


def summarize_geometry_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse step-wise geometry records into first/last deltas."""
    rows = [h for h in history if "hessian_geom" in h]
    if not rows:
        return {}
    first, last = rows[0]["hessian_geom"], rows[-1]["hessian_geom"]
    keys = (
        "uv_cos_mean",
        "decouple_to_coax_ratio",
        "s_gate_s_content_corr",
        "grad_a_to_u_ratio",
        "grad_v_to_u_ratio",
        "active_unit_frac",
    )
    delta = {}
    for k in keys:
        if k in first and k in last and first[k] is not None and last[k] is not None:
            delta[f"{k}_delta"] = float(last[k]) - float(first[k])
    return {"first": first, "last": last, "delta": delta, "n_snapshots": len(rows)}


def attach_uav_plane_hooks(
    model: nn.Module,
    store: list[dict[str, Any]],
    *,
    n_bins: int = 28,
    max_points: int = 6000,
) -> list[Any]:
    """Histogram ``(content proj, gate s)`` on each Hessian FFN during the next forward."""
    import numpy as np

    s_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    handles = []

    def _hook(layer_idx: int):
        def fn(mod: nn.Module, inputs: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
            core = getattr(mod, "phi_core", None)
            if core is None or not hasattr(core, "native_factors"):
                return
            _lam, proj, gate = core.native_factors(inputs[0])
            p = proj.detach().reshape(-1).float()
            g = gate.detach().reshape(-1).float()
            if p.numel() > max_points:
                idx = torch.randint(0, p.numel(), (max_points,), device=p.device)
                p, g = p[idx], g[idx]
            pp = p.cpu().numpy()
            gg = g.cpu().numpy()
            counts, _, _ = np.histogram2d(gg, pp, bins=[s_edges, s_edges])
            if pp.size > 2:
                pc, gc = pp - pp.mean(), gg - gg.mean()
                corr = float((pc * gc).sum() / (np.sqrt((pc ** 2).sum() * (gc ** 2).sum()) + 1e-8))
            else:
                corr = float("nan")
            store.append(
                {
                    "layer": layer_idx,
                    "corr": corr,
                    "content_abs": float(np.abs(pp).mean()) if pp.size else float("nan"),
                    "gate_abs": float(np.abs(gg).mean()) if gg.size else float("nan"),
                    "counts": counts.astype(int).tolist(),
                    "edges": s_edges.tolist(),
                }
            )

        return fn

    i = 0
    for module in model.modules():
        if isinstance(module, HessianFFN) and module.phi_core is not None:
            handles.append(module.register_forward_hook(_hook(i)))
            i += 1
    return handles
