"""Memory-Track geometry probes for NIAH (coaxial encoder, no UAV decouple).

Logs native ``(λ, s, proj_x)`` on encoder/decoder FFN inputs, split by needle vs
haystack regions, so training dynamics can be read against the analytic ψ'' well
instead of treating the block as a black-box MLP.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlb.ffn.hessian_core import HessianFFN, _PhiSpectralCore, _tanh2_second_deriv
from tlb.ffn.standard import StandardFFN
from tlb.models.seq2seq_model import Seq2SeqModel

# Central well of ψ''(s): |s| < artanh(1/√3).
_CENTRAL_LOBE_HALF = math.atanh(1.0 / math.sqrt(3.0))


def niah_region_masks(
    src: torch.Tensor,
    src_pad: torch.Tensor,
    *,
    hay_len: int,
    needle_len: int,
) -> dict[str, torch.Tensor]:
    """Boolean masks over encoder source positions (B, T)."""
    b, t = src.shape
    pos = torch.arange(t, device=src.device).view(1, t).expand(b, t)
    valid = ~src_pad
    needle = valid & (pos >= hay_len) & (pos < hay_len + needle_len)
    hay = (
        valid
        & ((pos < hay_len) | (pos >= hay_len + needle_len))
        & (pos < 2 * hay_len + needle_len)
    )
    delim = valid & (pos == 2 * hay_len + needle_len)
    return {"needle": needle, "hay": hay, "delim": delim, "valid": valid}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> tuple[float, int]:
    if values.ndim > 2:
        values = values.reshape(values.size(0), values.size(1), -1).mean(dim=-1)
    sel = values[mask]
    n = int(sel.numel())
    if n == 0:
        return float("nan"), 0
    return float(sel.mean().item()), n


def _masked_frac(condition: torch.Tensor, mask: torch.Tensor) -> tuple[float, int]:
    if condition.ndim > 2:
        condition = condition.reshape(condition.size(0), condition.size(1), -1).mean(
            dim=-1
        ) > 0
    sel = condition[mask]
    n = int(sel.numel())
    if n == 0:
        return float("nan"), 0
    return float(sel.float().mean().item()), n


def hessian_native_factors(
    core: nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(λ, proj_x, s_gate)`` for coaxial or prefix-ms cores."""
    if hasattr(core, "native_factors"):
        return core.native_factors(x)

    if isinstance(core, _PhiSpectralCore):
        proj_x = x @ core.weight.transpose(-2, -1)
        s_gate = proj_x + core.bias
        lam = _tanh2_second_deriv(s_gate)
        return lam, proj_x, s_gate

    if hasattr(core, "weight") and hasattr(core, "bias") and hasattr(core, "hp_logit"):
        from tlb.ffn.hessian_core import (
            _causal_prefix_mean,
            _causal_window_attn_pool,
            _causal_window_mean,
        )

        z_attn = _causal_window_attn_pool(
            x, core.q_proj, core.k_proj, core.window_size
        )
        z_win = _causal_window_mean(x, core.window_size)
        z_long = _causal_prefix_mean(x)
        scale_w = F.softmax(core.scale_logits, dim=0)
        z = scale_w[0] * z_attn + scale_w[1] * z_win + scale_w[2] * z_long
        gamma = torch.sigmoid(core.hp_logit)
        proj_in = gamma * x + (1.0 - gamma) * (x - z_win)
        w_in = core.weight.transpose(-2, -1)
        proj_x = proj_in @ w_in
        if hasattr(core, "content_bias"):
            proj_x = proj_x + core.content_bias
        s_gate = z @ w_in + core.bias
        lam = _tanh2_second_deriv(s_gate)
        return lam, proj_x, s_gate

    raise TypeError(f"unsupported Hessian core for memory probe: {type(core)!r}")


def _region_native_stats(
    lam: torch.Tensor,
    proj_x: torch.Tensor,
    s_gate: torch.Tensor,
    ff_out: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | int]:
    gated = lam * proj_x
    s_abs_tok = s_gate.abs().mean(dim=-1)
    lam_abs_tok = lam.abs().mean(dim=-1)
    gated_abs_tok = gated.abs().mean(dim=-1)
    ff_rms_tok = ff_out.pow(2).mean(dim=-1).sqrt()
    in_lobe = s_gate.abs().mean(dim=-1) < _CENTRAL_LOBE_HALF
    lam_neg = lam.mean(dim=-1) < 0

    row: dict[str, float | int] = {}
    for key, tensor in (
        ("s_abs_mean", s_abs_tok),
        ("lambda_abs_mean", lam_abs_tok),
        ("gated_abs_mean", gated_abs_tok),
        ("ff_out_rms", ff_rms_tok),
    ):
        val, _ = _masked_mean(tensor, mask)
        row[key] = val
    _, n = _masked_mean(s_abs_tok, mask)
    row["n_tokens"] = n
    frac, _ = _masked_frac(in_lobe, mask)
    row["s_in_central_lobe_frac"] = frac
    frac, _ = _masked_frac(lam_neg, mask)
    row["lambda_neg_frac"] = frac
    return row


def _region_standard_stats(
    x_norm: torch.Tensor,
    ff_out: torch.Tensor,
    ffn: StandardFFN,
    mask: torch.Tensor,
) -> dict[str, float | int]:
    h1 = ffn.act(ffn.linear1(x_norm))
    ff_rms_tok = ff_out.pow(2).mean(dim=-1).sqrt()
    h1_abs_tok = h1.abs().mean(dim=-1)
    ff_mean, n = _masked_mean(ff_rms_tok, mask)
    h_mean, _ = _masked_mean(h1_abs_tok, mask)
    return {
        "ff_out_rms": ff_mean,
        "hidden_abs_mean": h_mean,
        "n_tokens": n,
    }


def _contrast(
    needle: dict[str, float | int], hay: dict[str, float | int], key: str
) -> float:
    n = float(needle.get(key, float("nan")))
    h = float(hay.get(key, float("nan")))
    if math.isnan(n) or math.isnan(h):
        return float("nan")
    return n - h


def _masked_unit_pairs(
    s_gate: torch.Tensor,
    proj_x: torch.Tensor,
    lam: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten (B,T,M) native coordinates on masked token positions."""
    b, t, m = s_gate.shape
    tok = mask.reshape(-1)
    s = s_gate.reshape(b * t, m)[tok].reshape(-1)
    p = proj_x.reshape(b * t, m)[tok].reshape(-1)
    l = lam.reshape(b * t, m)[tok].reshape(-1)
    if s.numel() == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    if s.numel() > max_points:
        idx = torch.randperm(s.numel(), device=s.device)[:max_points]
        s, p, l = s[idx], p[idx], l[idx]
    return (
        s.detach().cpu().numpy().astype(np.float64),
        p.detach().cpu().numpy().astype(np.float64),
        l.detach().cpu().numpy().astype(np.float64),
    )


def _histogram2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_edges: Sequence[float],
    y_edges: Sequence[float],
) -> list[list[int]]:
    if x.size == 0:
        z = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=np.int64)
    else:
        z, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    return z.astype(np.int64).tolist()


def _collect_native_plane2d(
    lam: torch.Tensor,
    proj_x: torch.Tensor,
    s_gate: torch.Tensor,
    mask: torch.Tensor,
    *,
    n_bins: int,
    max_points: int,
) -> dict[str, Any]:
    """Binned occupancy on native planes ``(s, proj)`` and ``(s, λ)``."""
    s_edges = np.linspace(-1.2, 1.2, n_bins + 1).tolist()
    p_edges = np.linspace(-1.5, 1.5, n_bins + 1).tolist()
    l_edges = np.linspace(-2.5, 2.5, n_bins + 1).tolist()
    s, p, l = _masked_unit_pairs(
        s_gate, proj_x, lam, mask, max_points=max_points
    )
    return {
        "n_points": int(s.size),
        "s_proj": {
            "counts": _histogram2d(s, p, x_edges=s_edges, y_edges=p_edges),
            "s_edges": s_edges,
            "proj_edges": p_edges,
        },
        "s_lambda": {
            "counts": _histogram2d(s, l, x_edges=s_edges, y_edges=l_edges),
            "s_edges": s_edges,
            "lambda_edges": l_edges,
        },
    }


def _collect_visible_pipeline(
    x_norm: torch.Tensor,
    lam: torch.Tensor,
    proj_x: torch.Tensor,
    s_gate: torch.Tensor,
    ff_out: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | int]:
    """Per-region magnitudes for the native map ``x → (s, proj) → λ·proj → Δx``."""
    gated = lam * proj_x
    x_rms = x_norm.pow(2).mean(dim=-1).sqrt()
    gated_tok = gated.abs().mean(dim=-1)
    ff_rms = ff_out.pow(2).mean(dim=-1).sqrt()
    row: dict[str, float | int] = {}
    for key, tensor in (
        ("x_in_rms", x_rms),
        ("s_abs_mean", s_gate.abs().mean(dim=-1)),
        ("proj_abs_mean", proj_x.abs().mean(dim=-1)),
        ("gated_abs_mean", gated_tok),
        ("ff_out_rms", ff_rms),
    ):
        val, _ = _masked_mean(tensor, mask)
        row[key] = val
    x_mean = float(row["x_in_rms"])
    row["gated_to_x_ratio"] = float(row["gated_abs_mean"]) / max(x_mean, 1e-8)
    row["delta_to_x_ratio"] = float(row["ff_out_rms"]) / max(x_mean, 1e-8)
    _, n = _masked_mean(x_rms, mask)
    row["n_tokens"] = n
    return row


def _record_ffn_probe(
    store: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    side: str,
    layer_idx: int,
    ffn: nn.Module,
    x_norm: torch.Tensor,
    ff_out: torch.Tensor,
    enc_masks: dict[str, torch.Tensor] | None,
    dec_valid: torch.Tensor | None,
    collect_2d: bool = False,
    plane2d_layers: set[int] | None = None,
    n_bins: int = 20,
    max_points: int = 4096,
) -> None:
    key = (side, layer_idx)
    layer_row: dict[str, Any] = {
        "side": side,
        "layer": layer_idx,
        "ffn_type": getattr(ffn, "ffn_type", type(ffn).__name__),
    }
    if isinstance(ffn, HessianFFN) and ffn.phi_core is not None:
        core = ffn.phi_core
        lam, proj_x, s_gate = hessian_native_factors(core, x_norm)
        if side == "encoder" and enc_masks is not None:
            regions: dict[str, Any] = {}
            for name in ("needle", "hay", "delim"):
                regions[name] = _region_native_stats(
                    lam, proj_x, s_gate, ff_out, enc_masks[name]
                )
            regions["contrast"] = {
                "needle_minus_hay_s_abs": _contrast(
                    regions["needle"], regions["hay"], "s_abs_mean"
                ),
                "needle_minus_hay_gated_abs": _contrast(
                    regions["needle"], regions["hay"], "gated_abs_mean"
                ),
                "needle_minus_hay_ff_out_rms": _contrast(
                    regions["needle"], regions["hay"], "ff_out_rms"
                ),
                "needle_minus_hay_lobe_frac": _contrast(
                    regions["needle"], regions["hay"], "s_in_central_lobe_frac"
                ),
            }
            hess_row: dict[str, Any] = {
                "variant": str(ffn.variant),
                "coaxial": not hasattr(core, "out_weight"),
                "regions": regions,
            }
            if (
                collect_2d
                and plane2d_layers is not None
                and layer_idx in plane2d_layers
            ):
                plane2d: dict[str, Any] = {}
                visible: dict[str, Any] = {}
                for name in ("needle", "hay"):
                    plane2d[name] = _collect_native_plane2d(
                        lam,
                        proj_x,
                        s_gate,
                        enc_masks[name],
                        n_bins=n_bins,
                        max_points=max_points,
                    )
                    visible[name] = _collect_visible_pipeline(
                        x_norm, lam, proj_x, s_gate, ff_out, enc_masks[name]
                    )
                hess_row["plane2d"] = plane2d
                hess_row["visible"] = visible
            layer_row["hessian"] = hess_row
        elif dec_valid is not None:
            pooled = {
                "valid": _region_native_stats(lam, proj_x, s_gate, ff_out, dec_valid)
            }
            hess_row = {
                "variant": str(ffn.variant),
                "coaxial": not hasattr(core, "out_weight"),
                "pooled": pooled,
            }
            if (
                collect_2d
                and plane2d_layers is not None
                and layer_idx in plane2d_layers
            ):
                hess_row["plane2d"] = {
                    "valid": _collect_native_plane2d(
                        lam,
                        proj_x,
                        s_gate,
                        dec_valid,
                        n_bins=n_bins,
                        max_points=max_points,
                    )
                }
                hess_row["visible"] = {
                    "valid": _collect_visible_pipeline(
                        x_norm, lam, proj_x, s_gate, ff_out, dec_valid
                    )
                }
            layer_row["hessian"] = hess_row
    elif isinstance(ffn, StandardFFN) and enc_masks is not None:
        regions = {
            name: _region_standard_stats(x_norm, ff_out, ffn, enc_masks[name])
            for name in ("needle", "hay", "delim")
        }
        regions["contrast"] = {
            "needle_minus_hay_ff_out_rms": _contrast(
                regions["needle"], regions["hay"], "ff_out_rms"
            ),
            "needle_minus_hay_hidden_abs": _contrast(
                regions["needle"], regions["hay"], "hidden_abs_mean"
            ),
        }
        layer_row["standard"] = {"regions": regions}
    store.setdefault(key, []).append(layer_row)


def _sum_plane_counts(groups: list[dict[str, Any]], *path: str) -> list[list[int]] | None:
    arrays: list[np.ndarray] = []
    for g in groups:
        node: Any = g
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if node is None:
            continue
        arrays.append(np.asarray(node, dtype=np.int64))
    if not arrays:
        return None
    return np.sum(arrays, axis=0).astype(np.int64).tolist()


def _pool_probe_store(
    store: dict[tuple[str, int], list[dict[str, Any]]], side: str
) -> dict[str, Any]:
    rows = [(k, v) for k, v in store.items() if k[0] == side]
    if not rows:
        return {}
    pooled_layers: list[dict[str, Any]] = []
    for (_, layer_idx), group in sorted(rows, key=lambda kv: kv[0][1]):
        template = group[0]
        if "hessian" in template:
            if "regions" in template["hessian"]:
                out: dict[str, Any] = {
                    "layer": layer_idx,
                    "variant": template["hessian"]["variant"],
                    "coaxial": template["hessian"]["coaxial"],
                    "regions": {},
                    "contrast": {},
                }
                for region in ("needle", "hay", "delim"):
                    keys = template["hessian"]["regions"][region].keys()
                    out["regions"][region] = {
                        k: float(
                            sum(g["hessian"]["regions"][region][k] for g in group)
                            / len(group)
                        )
                        for k in keys
                    }
                for k in template["hessian"]["regions"]["contrast"]:
                    out["contrast"][k] = float(
                        sum(g["hessian"]["regions"]["contrast"][k] for g in group)
                        / len(group)
                    )
                if "plane2d" in template["hessian"]:
                    out["plane2d"] = {}
                    for region in template["hessian"]["plane2d"]:
                        ref = template["hessian"]["plane2d"][region]
                        out["plane2d"][region] = {
                            "n_points": int(
                                sum(
                                    g["hessian"]["plane2d"][region].get("n_points", 0)
                                    for g in group
                                )
                            ),
                            "s_proj": {
                                "counts": _sum_plane_counts(
                                    group,
                                    "hessian",
                                    "plane2d",
                                    region,
                                    "s_proj",
                                    "counts",
                                ),
                                "s_edges": ref["s_proj"]["s_edges"],
                                "proj_edges": ref["s_proj"]["proj_edges"],
                            },
                            "s_lambda": {
                                "counts": _sum_plane_counts(
                                    group,
                                    "hessian",
                                    "plane2d",
                                    region,
                                    "s_lambda",
                                    "counts",
                                ),
                                "s_edges": ref["s_lambda"]["s_edges"],
                                "lambda_edges": ref["s_lambda"]["lambda_edges"],
                            },
                        }
                if "visible" in template["hessian"]:
                    out["visible"] = {}
                    for region in template["hessian"]["visible"]:
                        keys = template["hessian"]["visible"][region].keys()
                        out["visible"][region] = {
                            k: float(
                                sum(g["hessian"]["visible"][region][k] for g in group)
                                / len(group)
                            )
                            for k in keys
                        }
            else:
                out = {
                    "layer": layer_idx,
                    "variant": template["hessian"]["variant"],
                    "coaxial": template["hessian"]["coaxial"],
                    "pooled": {
                        "valid": {
                            k: float(
                                sum(
                                    g["hessian"]["pooled"]["valid"][k] for g in group
                                )
                                / len(group)
                            )
                            for k in template["hessian"]["pooled"]["valid"]
                        }
                    },
                }
                if "plane2d" in template["hessian"]:
                    ref = template["hessian"]["plane2d"]["valid"]
                    out["plane2d"] = {
                        "valid": {
                            "n_points": int(
                                sum(
                                    g["hessian"]["plane2d"]["valid"].get("n_points", 0)
                                    for g in group
                                )
                            ),
                            "s_proj": {
                                "counts": _sum_plane_counts(
                                    group,
                                    "hessian",
                                    "plane2d",
                                    "valid",
                                    "s_proj",
                                    "counts",
                                ),
                                "s_edges": ref["s_proj"]["s_edges"],
                                "proj_edges": ref["s_proj"]["proj_edges"],
                            },
                            "s_lambda": {
                                "counts": _sum_plane_counts(
                                    group,
                                    "hessian",
                                    "plane2d",
                                    "valid",
                                    "s_lambda",
                                    "counts",
                                ),
                                "s_edges": ref["s_lambda"]["s_edges"],
                                "lambda_edges": ref["s_lambda"]["lambda_edges"],
                            },
                        }
                    }
                if "visible" in template["hessian"]:
                    keys = template["hessian"]["visible"]["valid"].keys()
                    out["visible"] = {
                        "valid": {
                            k: float(
                                sum(g["hessian"]["visible"]["valid"][k] for g in group)
                                / len(group)
                            )
                            for k in keys
                        }
                    }
            pooled_layers.append(out)
        elif "standard" in template:
            out = {"layer": layer_idx, "regions": {}, "contrast": {}}
            for region in ("needle", "hay", "delim"):
                keys = template["standard"]["regions"][region].keys()
                out["regions"][region] = {
                    k: float(
                        sum(g["standard"]["regions"][region][k] for g in group)
                        / len(group)
                    )
                    for k in keys
                }
            for k in template["standard"]["regions"]["contrast"]:
                out["contrast"][k] = float(
                    sum(g["standard"]["regions"]["contrast"][k] for g in group)
                    / len(group)
                )
            pooled_layers.append(out)
    return {"layers": pooled_layers}


@torch.no_grad()
def probe_niah_memory_geometry(
    model: Seq2SeqModel,
    cfg: dict,
    layout: dict,
    device: torch.device,
    *,
    batches: int = 2,
    batch_size: int | None = None,
    probe_seed: int = 4242,
) -> dict[str, Any]:
    """Probe encoder + decoder FFN geometry on fixed NIAH validation batches."""
    from tlb.tasks.seq2seq_tasks import build_seq2seq_batch

    was_training = model.training
    model.eval()

    hay_len = int(cfg["niah_hay_len"])
    needle_len = int(cfg["niah_needle_len"])
    bs = int(batch_size or cfg.get("val_batch", 64))
    collect_2d = bool(cfg.get("memory_geom_2d", False))
    n_bins = int(cfg.get("memory_geom_2d_bins", 20))
    max_points = int(cfg.get("memory_geom_2d_max_points", 4096))
    enc_plane_layers = (
        {len(model.encoder_layers) - 1} if collect_2d and model.encoder_layers else set()
    )
    dec_plane_layers = (
        {len(model.decoder_layers) - 1} if collect_2d and model.decoder_layers else set()
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(probe_seed)

    probe_store: dict[tuple[str, int], list[dict[str, Any]]] = {}
    memory_cosines: list[float] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    last_encoder_memory: list[torch.Tensor] = []

    enc_masks_ref: dict[str, torch.Tensor | None] = {"masks": None}
    dec_valid_ref: dict[str, torch.Tensor | None] = {"valid": None}

    def _last_enc_hook(_mod: nn.Module, _inp: tuple, output: torch.Tensor) -> None:
        last_encoder_memory.append(output)

    def _enc_hook(layer_idx: int):
        layer = model.encoder_layers[layer_idx]

        def hook(
            _mod: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            x_norm = inputs[0]
            masks = enc_masks_ref["masks"]
            if masks is None:
                return
            _record_ffn_probe(
                probe_store,
                side="encoder",
                layer_idx=layer_idx,
                ffn=layer.ffn,  # type: ignore[arg-type]
                x_norm=x_norm,
                ff_out=output,
                enc_masks=masks,
                dec_valid=None,
                collect_2d=collect_2d,
                plane2d_layers=enc_plane_layers,
                n_bins=n_bins,
                max_points=max_points,
            )

        return hook

    def _dec_hook(layer_idx: int):
        layer = model.decoder_layers[layer_idx]

        def hook(
            _mod: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            x_norm = inputs[0]
            dec_valid = dec_valid_ref["valid"]
            if dec_valid is None:
                return
            _record_ffn_probe(
                probe_store,
                side="decoder",
                layer_idx=layer_idx,
                ffn=layer.ffn,  # type: ignore[arg-type]
                x_norm=x_norm,
                ff_out=output,
                enc_masks=None,
                dec_valid=dec_valid,
                collect_2d=collect_2d,
                plane2d_layers=dec_plane_layers,
                n_bins=n_bins,
                max_points=max_points,
            )

        return hook

    if len(model.encoder_layers) > 0:
        handles.append(
            model.encoder_layers[-1].register_forward_hook(_last_enc_hook)
        )
    for i, layer in enumerate(model.encoder_layers):
        handles.append(layer.ffn.register_forward_hook(_enc_hook(i)))  # type: ignore[attr-defined]
    for i, layer in enumerate(model.decoder_layers):
        handles.append(layer.ffn.register_forward_hook(_dec_hook(i)))  # type: ignore[attr-defined]

    try:
        for _ in range(batches):
            src, tgt_in, _labels, src_pad, tgt_pad = build_seq2seq_batch(
                "niah", bs, layout, cfg, device, generator=gen
            )
            enc_masks_ref["masks"] = niah_region_masks(
                src, src_pad, hay_len=hay_len, needle_len=needle_len
            )
            dec_valid_ref["valid"] = ~tgt_pad

            last_encoder_memory.clear()
            model(
                src,
                tgt_in,
                src_key_padding_mask=src_pad,
                tgt_key_padding_mask=tgt_pad,
            )
            if last_encoder_memory and enc_masks_ref["masks"] is not None:
                memory = last_encoder_memory[-1]
                needle_m = enc_masks_ref["masks"]["needle"].unsqueeze(-1)
                hay_m = enc_masks_ref["masks"]["hay"].unsqueeze(-1)
                n_vec = (memory * needle_m).sum(dim=(0, 1)) / needle_m.sum().clamp(min=1)
                h_vec = (memory * hay_m).sum(dim=(0, 1)) / hay_m.sum().clamp(min=1)
                memory_cosines.append(
                    float(
                        F.cosine_similarity(n_vec.unsqueeze(0), h_vec.unsqueeze(0)).item()
                    )
                )
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    enc = _pool_probe_store(probe_store, "encoder")
    dec = _pool_probe_store(probe_store, "decoder")
    last_enc = enc.get("layers", [{}])[-1] if enc else {}
    return {
        "task": "niah",
        "batches": batches,
        "batch_size": bs,
        "hay_len": hay_len,
        "needle_len": needle_len,
        "central_lobe_halfwidth": _CENTRAL_LOBE_HALF,
        "memory_needle_hay_cosine": float(sum(memory_cosines) / max(len(memory_cosines), 1)),
        "encoder": enc,
        "decoder": dec,
        "encoder_last_contrast": last_enc.get("contrast", {}),
        "encoder_last_needle": (last_enc.get("regions") or {}).get("needle", {}),
        "encoder_last_hay": (last_enc.get("regions") or {}).get("hay", {}),
    }


def summarize_memory_geometry_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """First/last snapshot deltas for paper tables."""
    rows = [h for h in history if "memory_geometry" in h]
    if not rows:
        return {}
    first, last = rows[0]["memory_geometry"], rows[-1]["memory_geometry"]
    keys = (
        "needle_minus_hay_s_abs",
        "needle_minus_hay_gated_abs",
        "needle_minus_hay_ff_out_rms",
        "needle_minus_hay_lobe_frac",
    )
    delta: dict[str, float] = {}
    fc = first.get("encoder_last_contrast") or {}
    lc = last.get("encoder_last_contrast") or {}
    for k in keys:
        if k in fc and k in lc:
            fv, lv = fc[k], lc[k]
            if not (math.isnan(fv) or math.isnan(lv)):
                delta[k] = float(lv - fv)
    return {
        "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
        "n_snapshots": len(rows),
        "memory_cosine_first": first.get("memory_needle_hay_cosine"),
        "memory_cosine_last": last.get("memory_needle_hay_cosine"),
        "encoder_contrast_delta": delta,
        "first": first,
        "last": last,
    }
