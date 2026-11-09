"""Training utilities."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

IGNORE_LABEL = -100


def resolve_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    return F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_LABEL)


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    mask = labels != IGNORE_LABEL
    if not mask.any():
        return 0.0
    correct = (preds[mask] == labels[mask]).float().mean()
    return float(correct.item())


DECOUPLE_HESSIAN_VARIANT = "h4_phi768_prefix_ms_multwin_invcombo_ladder_decouple"


def schedule_lr_multiplier(
    step: int,
    cfg: dict[str, Any],
    *,
    total_steps: int,
    lr_min_ratio: float,
) -> float:
    schedule = str(cfg.get("lr_schedule", "cosine"))
    warmup_steps = int(cfg.get("warmup_steps", 0))
    if schedule == "flat_cosine":
        flat_steps = int(
            cfg.get("flat_steps", round(float(cfg.get("flat_ratio", 0.0)) * total_steps))
        )
        if step < max(0, flat_steps):
            return 1.0
        tail_steps = max(1, total_steps - max(0, flat_steps))
        return training_lr_multiplier(
            step - max(0, flat_steps),
            total_steps=tail_steps,
            warmup_steps=0,
            schedule="cosine",
            lr_min_ratio=lr_min_ratio,
        )
    return training_lr_multiplier(
        step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        schedule=schedule,
        lr_min_ratio=lr_min_ratio,
    )


def flat_schedule_steps(cfg: dict[str, Any], *, total_steps: int) -> int:
    if str(cfg.get("lr_schedule", "cosine")) != "flat_cosine":
        return 0
    return int(
        cfg.get("flat_steps", round(float(cfg.get("flat_ratio", 0.0)) * total_steps))
    )


from tlb.training.hessian_factor_lr import (
    assign_hessian_factor_param,
    attach_hessian_factor_lr,
    build_hessian_factor_param_groups,
)


def build_training_optimizer(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    ffn_type: str,
) -> torch.optim.Optimizer:
    """AdamW with optional Standard MLP or Hessian param-group LR multipliers."""
    base_lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0.01))
    std_groups = ffn_type == "standard" and bool(cfg.get("standard_param_groups", False))
    hess_phase = ffn_type == "hessian" and bool(cfg.get("hessian_phase_groups", False))
    hess_factor = ffn_type == "hessian" and bool(cfg.get("hessian_factor_lr", False))
    hess_groups = ffn_type == "hessian" and (
        bool(cfg.get("hessian_param_groups", False)) or hess_phase or hess_factor
    )
    if not std_groups and not hess_groups:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=base_lr, weight_decay=weight_decay)

    groups: dict[str, dict[str, Any]]
    if std_groups:
        exp_mult = float(cfg.get("standard_expansion_lr_mult", 2.0))
        con_mult = float(cfg.get("standard_contraction_lr_mult", 0.5))
        groups = {
            "base": {"params": [], "lr_mult": 1.0},
            "std_expansion": {"params": [], "lr_mult": exp_mult},
            "std_contraction": {"params": [], "lr_mult": con_mult},
        }
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if ".ffn.linear1." in name:
                groups["std_expansion"]["params"].append(param)
            elif ".ffn.linear2." in name:
                groups["std_contraction"]["params"].append(param)
            else:
                groups["base"]["params"].append(param)
    else:
        separate_out = bool(cfg.get("hessian_separate_out_weight", False)) or hess_phase
        if hess_factor:
            groups, _ = build_hessian_factor_param_groups(cfg)
        elif hess_phase:
            core_flat = float(cfg.get("hessian_flat_core_lr_mult", 4.0))
            out_flat = float(cfg.get("hessian_flat_out_lr_mult", 2.0))
            cross_flat = float(cfg.get("hessian_flat_cross_lr_mult", 2.0))
            readout_flat = float(cfg.get("hessian_flat_readout_lr_mult", 0.25))
            other_flat = float(cfg.get("hessian_flat_other_lr_mult", 1.0))
            core_tail = float(cfg.get("hessian_tail_core_lr_mult", 1.0))
            out_tail = float(cfg.get("hessian_tail_out_lr_mult", 4.0))
            cross_tail = float(cfg.get("hessian_tail_cross_lr_mult", 2.0))
            readout_tail = float(cfg.get("hessian_tail_readout_lr_mult", 1.0))
            other_tail = float(cfg.get("hessian_tail_other_lr_mult", 1.0))
            groups = {
                "base": {
                    "params": [],
                    "lr_mult": 1.0,
                    "lr_mult_flat": 1.0,
                    "lr_mult_tail": 1.0,
                },
                "hessian_core": {
                    "params": [],
                    "lr_mult": core_flat,
                    "lr_mult_flat": core_flat,
                    "lr_mult_tail": core_tail,
                },
                "hessian_cross": {
                    "params": [],
                    "lr_mult": cross_flat,
                    "lr_mult_flat": cross_flat,
                    "lr_mult_tail": cross_tail,
                },
                "hessian_readout": {
                    "params": [],
                    "lr_mult": readout_flat,
                    "lr_mult_flat": readout_flat,
                    "lr_mult_tail": readout_tail,
                },
                "hessian_other": {
                    "params": [],
                    "lr_mult": other_flat,
                    "lr_mult_flat": other_flat,
                    "lr_mult_tail": other_tail,
                },
                "hessian_out": {
                    "params": [],
                    "lr_mult": out_flat,
                    "lr_mult_flat": out_flat,
                    "lr_mult_tail": out_tail,
                },
            }
        else:
            core_mult = float(cfg.get("hessian_core_lr_mult", 1.0))
            context_mult = float(cfg.get("hessian_context_lr_mult", core_mult))
            readout_mult = float(cfg.get("hessian_readout_lr_mult", 1.0))
            hessian_other_mult = float(cfg.get("hessian_other_lr_mult", 1.0))
            cross_mult = float(cfg.get("hessian_cross_lr_mult", 1.0))
            out_mult = float(cfg.get("hessian_out_lr_mult", core_mult))
            groups = {
                "base": {"params": [], "lr_mult": 1.0},
                "hessian_core": {"params": [], "lr_mult": core_mult},
                "hessian_context": {"params": [], "lr_mult": context_mult},
                "hessian_cross": {"params": [], "lr_mult": core_mult * cross_mult},
                "hessian_readout": {"params": [], "lr_mult": readout_mult},
                "hessian_other": {"params": [], "lr_mult": hessian_other_mult},
            }
            if separate_out:
                groups["hessian_out"] = {"params": [], "lr_mult": out_mult}
        cross_markers = ("cross_", "bilinear_")
        if not separate_out:
            cross_markers = cross_markers + ("out_weight",)
        out_markers = ("out_weight", "out_unit_bias", "out_bias", "side_up")
        context_markers = (
            "context_weight",
            "gate_bias",
            "cross_logit",
            "tt_logit",
            "scale_logits",
            "q_proj",
            "k_proj",
            "ctx_q_proj",
            "ctx_k_proj",
            "resonance_logit",
            "ctx_lambda_mix_logit",
        )
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if ".ffn.phi_core." in name:
                if hess_factor:
                    assign_hessian_factor_param(
                        name,
                        param,
                        groups,
                        separate_out=separate_out,
                        cross_markers=cross_markers,
                        out_markers=out_markers,
                    )
                elif separate_out and any(m in name for m in out_markers):
                    groups["hessian_out"]["params"].append(param)
                elif any(m in name for m in cross_markers):
                    groups["hessian_cross"]["params"].append(param)
                elif (
                    not hess_phase
                    and not hess_factor
                    and (
                        any(m in name for m in context_markers)
                        or name.endswith(".phi_core.bias")
                    )
                ):
                    groups["hessian_context"]["params"].append(param)
                else:
                    groups["hessian_core"]["params"].append(param)
            elif ".ffn.readout." in name:
                groups["hessian_readout"]["params"].append(param)
            elif ".ffn." in name:
                groups["hessian_other"]["params"].append(param)
            else:
                groups["base"]["params"].append(param)
    param_groups: list[dict[str, Any]] = []
    for group_name, group in groups.items():
        if group["params"]:
            default_mult = float(group.get("lr_mult", group.get("lr_mult_flat", 1.0)))
            entry: dict[str, Any] = {
                "params": group["params"],
                "lr": base_lr * default_mult,
                "lr_mult": default_mult,
                "name": group_name,
                "weight_decay": weight_decay,
            }
            if hess_phase:
                entry["lr_mult_flat"] = float(group["lr_mult_flat"])
                entry["lr_mult_tail"] = float(group["lr_mult_tail"])
            param_groups.append(entry)
    opt = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    if ffn_type == "hessian" and hess_factor:
        attach_hessian_factor_lr(model, opt, cfg)
    return opt


def build_hessian_optimizer(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    ffn_type: str,
) -> torch.optim.Optimizer:
    return build_training_optimizer(model, cfg, ffn_type=ffn_type)


def set_optimizer_lr(
    opt: torch.optim.Optimizer,
    cfg: dict[str, Any],
    *,
    step: int,
    total_steps: int,
    lr_min_ratio: float,
) -> None:
    """Apply lr = base · η_global(t) · m_group (factor×order groups are static)."""
    global_mult = schedule_lr_multiplier(
        step, cfg, total_steps=total_steps, lr_min_ratio=lr_min_ratio
    )
    base_lr = float(cfg["lr"])
    use_phase = bool(cfg.get("hessian_phase_groups", False))
    flat_steps = flat_schedule_steps(cfg, total_steps=total_steps)
    in_flat = step < max(0, flat_steps)
    for pg in opt.param_groups:
        if use_phase and "lr_mult_flat" in pg:
            lr_mult = float(pg["lr_mult_flat"] if in_flat else pg["lr_mult_tail"])
        else:
            lr_mult = float(pg.get("lr_mult", 1.0))
        pg["lr"] = base_lr * lr_mult * global_mult


def training_lr_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    schedule: str = "cosine",
    lr_min_ratio: float = 0.01,
) -> float:
    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"unknown schedule: {schedule!r}")
    ws = max(0, int(warmup_steps))
    if ws > 0 and step < ws:
        return float(step + 1) / float(ws)
    tail = max(1, int(total_steps) - ws)
    t = float(step - ws) / float(tail)
    t = min(max(t, 0.0), 1.0)
    return lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * t))


def apply_lr_schedule(
    opt: torch.optim.Optimizer,
    step: int,
    cfg: dict[str, Any],
    *,
    total_steps: int,
    lr_min_ratio: float,
) -> None:
    mult = schedule_lr_multiplier(
        step, cfg, total_steps=total_steps, lr_min_ratio=lr_min_ratio
    )
    base_lr = float(cfg["lr"])
    for pg in opt.param_groups:
        lr_mult = float(pg.get("lr_mult", 1.0))
        pg["lr"] = base_lr * lr_mult * mult


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def task_layout(task: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute vocab sizes and max_len for a synthetic task."""
    vocab_content = int(cfg["vocab_content"])
    pad_id = vocab_content
    delim_id = vocab_content + 1
    arrow_id = vocab_content + 2

    if task == "gap_copy":
        max_len = int(cfg["n_max"]) + int(cfg["gap_max"]) + 1 + int(cfg["n_max"])
        vocab_size = vocab_content + 2
    elif task == "niah":
        max_len = 2 * int(cfg["niah_hay_len"]) + 2 * int(cfg["niah_needle_len"]) + 1
        vocab_size = vocab_content + 2
    elif task == "kv_assoc":
        p = int(cfg["kv_num_pairs"])
        max_len = 3 * p + max(0, p - 1) * 0 + 1 + int(cfg["kv_dist_max"]) + 2
        vocab_size = vocab_content + 3
    else:
        raise ValueError(f"unknown task: {task!r}")

    return {
        "vocab_content": vocab_content,
        "pad_id": pad_id,
        "delim_id": delim_id,
        "arrow_id": arrow_id,
        "vocab_size": vocab_size,
        "max_len": max_len,
        "dim_ff": int(cfg.get("dim_feedforward", 4 * int(cfg["dim"]))),
    }


def build_batch(
    task: str,
    batch_size: int,
    layout: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
    *,
    fixed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from tlb.tasks.memory_tasks import (
        build_gap_copy_batch,
        build_kv_assoc_batch,
        build_niah_fixed_batch,
    )

    vc = layout["vocab_content"]
    pad_id = layout["pad_id"]
    delim_id = layout["delim_id"]
    max_len = layout["max_len"]

    if task == "gap_copy":
        return build_gap_copy_batch(
            batch_size,
            int(cfg["n_min"]),
            int(cfg["n_max"]),
            int(cfg["gap_min"]),
            int(cfg["gap_max"]),
            vc,
            pad_id,
            delim_id,
            max_len,
            device,
            fixed_n=int(cfg["n_max"]) if fixed else None,
            fixed_gap=int(cfg["gap_max"]) if fixed else None,
        )
    if task == "niah":
        return build_niah_fixed_batch(
            batch_size,
            int(cfg["niah_hay_len"]),
            int(cfg["niah_needle_len"]),
            vc,
            pad_id,
            delim_id,
            max_len,
            device,
        )
    if task == "kv_assoc":
        return build_kv_assoc_batch(
            batch_size,
            int(cfg["kv_num_pairs"]),
            int(cfg["kv_dist_min"]),
            int(cfg["kv_dist_max"]),
            vc,
            pad_id,
            delim_id,
            layout["arrow_id"],
            max_len,
            device,
            fixed_dist_len=int(cfg["kv_dist_max"]) if fixed else None,
            fixed_query_pair_idx=0 if fixed else None,
        )
    raise ValueError(f"unknown task: {task!r}")
