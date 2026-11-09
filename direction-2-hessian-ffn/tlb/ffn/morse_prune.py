"""Morse-style unit pruning for decoupled Hessian FFN cores.

Prunes rank-1 unit rows (``weight`` / ``bias`` / ``out_weight``) after warm init
while keeping the full m=6d analytic graph available for reproduction runs.

Strategies (low → high persistence in typical ViT / LM setups):

- ``mlp_aligned`` / ``spectral_tail``: keep prefix rows ``[0 : mlp_width_mult * dim)``.
- ``weight_norm``: keep top-k units by ``||w_j|| * ||v_j||`` (Morse activity proxy).

All knobs are optional; absent config → no-op (existing experiments unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

MorseStrategy = Literal["mlp_aligned", "spectral_tail", "weight_norm"]


@dataclass(frozen=True)
class MorsePruneConfig:
    strategy: MorseStrategy = "mlp_aligned"
    keep_units: int | None = None
    mlp_width_mult: float = 4.0


def parse_morse_prune_config(cfg: dict[str, Any]) -> MorsePruneConfig | None:
    """Parse optional Morse prune keys from a run / experiment config dict."""
    raw = cfg.get("hessian_morse_prune")
    if raw is None:
        return None
    if isinstance(raw, str):
        strategy: MorseStrategy = raw  # type: ignore[assignment]
        keep_units_raw = cfg.get("hessian_morse_keep_units")
        mlp_mult_raw = cfg.get("hessian_morse_mlp_width_mult", 4.0)
        return MorsePruneConfig(
            strategy=strategy,
            keep_units=int(keep_units_raw) if keep_units_raw is not None else None,
            mlp_width_mult=float(mlp_mult_raw),
        )
    if isinstance(raw, dict):
        strategy = str(raw.get("strategy", "mlp_aligned"))  # type: ignore[assignment]
        keep_raw = raw.get("keep_units", cfg.get("hessian_morse_keep_units"))
        mlp_mult = raw.get("mlp_width_mult", cfg.get("hessian_morse_mlp_width_mult", 4.0))
        return MorsePruneConfig(
            strategy=strategy,
            keep_units=int(keep_raw) if keep_raw is not None else None,
            mlp_width_mult=float(mlp_mult),
        )
    raise TypeError(f"hessian_morse_prune must be str or dict, got {type(raw)!r}")


def _resolve_keep_units(
    core: nn.Module,
    cfg: MorsePruneConfig,
) -> int:
    m = int(core.weight.shape[0])
    dim = int(core.weight.shape[1])
    if cfg.keep_units is not None:
        return min(int(cfg.keep_units), m)
    return min(int(cfg.mlp_width_mult * dim), m)


def select_morse_unit_indices(
    core: nn.Module,
    cfg: MorsePruneConfig,
) -> torch.Tensor:
    """Return sorted unit row indices to keep."""
    if not hasattr(core, "weight") or not hasattr(core, "bias"):
        raise TypeError("Morse prune requires phi_core.weight and phi_core.bias")
    m = int(core.weight.shape[0])
    keep_k = _resolve_keep_units(core, cfg)
    if keep_k >= m:
        return torch.arange(m, device=core.weight.device)

    strategy = cfg.strategy
    if strategy in ("mlp_aligned", "spectral_tail"):
        return torch.arange(keep_k, device=core.weight.device, dtype=torch.long)

    if strategy == "weight_norm":
        w = core.weight.detach().float()
        if _core_has_decouple_out(core):
            v = core.out_weight.detach().float()
        else:
            v = w
        scores = w.norm(dim=1) * v.norm(dim=1)
        _, idx = torch.topk(scores, k=keep_k, largest=True, sorted=False)
        return idx.sort().values.to(device=core.weight.device)

    raise ValueError(f"unknown Morse prune strategy: {strategy!r}")


def physical_prune_phi_core(
    core: nn.Module,
    keep_indices: torch.Tensor,
) -> dict[str, int | str]:
    """Physically slice decoupled unit rows; updates ``core.num_units`` when present."""
    idx = keep_indices.long()
    m_before = int(core.weight.shape[0])
    k = int(idx.numel())
    if k >= m_before:
        return {
            "units_before": m_before,
            "units_after": m_before,
            "pruned_units": 0,
            "method": "physical_slice",
            "skipped": "already_at_or_below_target",
        }

    with torch.no_grad():
        new_weight = nn.Parameter(core.weight.data.index_select(0, idx).clone())
        new_bias = nn.Parameter(core.bias.data.index_select(0, idx).clone())
        core.weight = new_weight
        core.bias = new_bias
        if _core_has_decouple_out(core):
            core.out_weight = nn.Parameter(
                core.out_weight.data.index_select(0, idx).clone()
            )
        if hasattr(core, "context_weight") and isinstance(
            core.context_weight, nn.Parameter
        ):
            core.context_weight = nn.Parameter(
                core.context_weight.data.index_select(0, idx).clone()
            )
    if hasattr(core, "num_units"):
        core.num_units = k
    return {
        "units_before": m_before,
        "units_after": k,
        "pruned_units": m_before - k,
        "method": "physical_slice",
    }


def _iter_prunable_phi_cores(model: nn.Module):
    for module in model.modules():
        core = getattr(module, "phi_core", None)
        if core is None:
            continue
        if not hasattr(core, "weight") or not hasattr(core, "bias"):
            continue
        yield core


def _core_has_decouple_out(core: nn.Module) -> bool:
    return hasattr(core, "out_weight") and isinstance(core.out_weight, nn.Parameter)


def count_hessian_phi_core_params(model: nn.Module) -> int:
    total = 0
    for core in _iter_prunable_phi_cores(model):
        for name in ("weight", "bias", "out_weight", "out_bias", "context_weight"):
            if hasattr(core, name):
                p = getattr(core, name)
                if isinstance(p, nn.Parameter):
                    total += int(p.numel())
    return total


def apply_hessian_morse_pruning(
    model: nn.Module,
    cfg: MorsePruneConfig,
) -> dict[str, Any]:
    """Apply Morse pruning to every ``phi_core`` with unit rows in ``model``."""
    layer_stats: list[dict[str, int | str | bool]] = []
    for core in _iter_prunable_phi_cores(model):
        keep_target = _resolve_keep_units(core, cfg)
        keep_idx = select_morse_unit_indices(core, cfg)
        stat = physical_prune_phi_core(core, keep_idx)
        stat["strategy"] = cfg.strategy
        stat["keep_units_target"] = keep_target
        stat["decouple_out"] = _core_has_decouple_out(core)
        layer_stats.append(stat)

    params_before = sum(int(s["units_before"]) for s in layer_stats) if layer_stats else 0
    params_after = sum(int(s["units_after"]) for s in layer_stats) if layer_stats else 0
    return {
        "strategy": cfg.strategy,
        "keep_units": cfg.keep_units,
        "mlp_width_mult": cfg.mlp_width_mult,
        "layers": len(layer_stats),
        "units_before_total": params_before,
        "units_after_total": params_after,
        "phi_core_params": count_hessian_phi_core_params(model),
        "layer_stats": layer_stats,
    }
