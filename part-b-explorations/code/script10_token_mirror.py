# Script10 — Per-token TinyMirror branch vs standard Transformer encoder.
# Each block: Pre-LN MHA residual, then Pre-LN FFN + parallel TinyMirror (dim→r→dim, nonlinear in r),
# optional --mirror-attn-mem: extra lightweight self-attn memory (Q,K→head_dim, V→dim) on same residual.
# merged on residual. Default r=--mirror-dim; --mirror-full-width sets r=dim. Mirror nonlinearity:
# --mirror-activation (default silu; light_depth_t2.0 / phys_tau_softplus_z_b1.0|b3.0|b5.0 from activation_extras).
# Tasks: identity / delimiter / gap_copy / dup_tail_copy / niah / niah_prefix / kv_assoc / story_assoc.
# Training: defaults AdamW (wd=0.01) + cosine (no warmup); --pos-enc learned|none|sinusoidal. --train-recipe moderate:
# matched baseline + Adam, const LR (inherits --pos-enc).
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_tasks import (
    build_dup_tail_copy_batch,
    build_gap_copy_batch,
    build_kv_assoc_batch,
    build_niah_fixed_batch,
    build_niah_prefix_batch,
    build_story_assoc_batch,
)
from paper_style import apply_paper_style
from activation_extras import LightDepthAct, ScatteringOpticalDepthAct

IGNORE_LABEL = -100

# TinyMirror bottleneck (aligned with script5/7 ViT MLP swap IDs).
MIRROR_BOTTLENECK_ACTIVATIONS = (
    "silu",
    "none",
    "light_depth_t2.0",
    "phys_tau_softplus_z_b1.0",
    "phys_tau_softplus_z_b3.0",
    "phys_tau_softplus_z_b5.0",
)


def tiny_mirror_bottleneck_activation(act_id: str) -> nn.Module | None:
    if act_id == "silu":
        return nn.SiLU()
    if act_id == "none":
        return None
    if act_id == "light_depth_t2.0":
        return LightDepthAct(2.0)
    if act_id == "phys_tau_softplus_z_b1.0":
        return ScatteringOpticalDepthAct(1.0, "softplus_z")
    if act_id == "phys_tau_softplus_z_b3.0":
        return ScatteringOpticalDepthAct(3.0, "softplus_z")
    if act_id == "phys_tau_softplus_z_b5.0":
        return ScatteringOpticalDepthAct(5.0, "softplus_z")
    raise ValueError(f"unknown TinyMirror bottleneck activation: {act_id}")
DEFAULT_OUTPUT = Path("results_script10_token_mirror")


def sinusoidal_position_encoding(max_len: int, dim: int) -> torch.Tensor:
    """(max_len, dim) fixed sinusoidal PE (Vaswani et al.), added to token embeddings."""
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal PE requires even dim, got {dim}")
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(max_len, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def training_lr_multiplier(
    step: int,
    *,
    total_opt_steps: int,
    warmup_steps: int,
    schedule: str,
    lr_min_ratio: float,
) -> float:
    """Per-step LR multiplier (applied on top of --lr). step in [0, total_opt_steps)."""
    if schedule == "none":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"unknown lr schedule: {schedule!r}")
    ws = max(0, int(warmup_steps))
    if ws > 0 and step < ws:
        return float(step + 1) / float(ws)
    tail = max(1, int(total_opt_steps) - ws)
    t = float(step - ws) / float(tail)
    t = min(1.0, max(0.0, t))
    cos_part = 0.5 * (1.0 + math.cos(math.pi * t))
    lo = float(lr_min_ratio)
    return lo + (1.0 - lo) * cos_part


def build_script10_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(args.lr))
    raise ValueError(f"unknown optimizer: {args.optimizer!r}")


def _transformer_legacy_baseline(args: argparse.Namespace) -> bool:
    return getattr(args, "transformer_baseline", "legacy") == "legacy"


def _transformer_pos_enc(args: argparse.Namespace) -> str:
    return "none" if _transformer_legacy_baseline(args) else str(args.pos_enc)


# ---------------------------------------------------------------------------
# Batch builders (delimiter + identity local; gap/dup from memory_tasks)
# ---------------------------------------------------------------------------
def build_copy_batch(
    batch_size: int,
    n_min: int,
    n_max: int,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    max_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    *,
    fixed_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if vocab_content < 1:
        raise ValueError("vocab_content must be >= 1")
    if pad_id != vocab_content or delim_id != vocab_content + 1:
        raise ValueError("pad_id / delim_id mismatch for delimiter copy")
    tokens_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for _ in range(batch_size):
        if fixed_n is not None:
            n = int(fixed_n)
        elif generator is not None:
            n = int(torch.randint(n_min, n_max + 1, (1,), generator=generator).item())
        else:
            n = int(torch.randint(n_min, n_max + 1, (1,)).item())
        if n < 1:
            raise ValueError("delimiter copy requires n >= 1")
        content = torch.randint(0, vocab_content, (n,), device=device)
        tail = torch.randint(0, vocab_content, (n,), device=device)
        seq = torch.cat(
            [
                content,
                torch.tensor([delim_id], device=device, dtype=torch.long),
                tail,
            ],
            dim=0,
        )
        lab = torch.full((seq.shape[0],), IGNORE_LABEL, dtype=torch.long, device=device)
        lab[n + 1 : 2 * n + 1] = content
        if seq.shape[0] < max_len:
            pad_n = max_len - seq.shape[0]
            seq = F.pad(seq, (0, pad_n), value=pad_id)
            lab = F.pad(lab, (0, pad_n), value=IGNORE_LABEL)
        tokens_list.append(seq)
        labels_list.append(lab)
    tokens = torch.stack(tokens_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    padding_mask = tokens == pad_id
    return tokens, labels, padding_mask


def build_identity_copy_batch(
    batch_size: int,
    n_min: int,
    n_max: int,
    vocab_content: int,
    pad_id: int,
    max_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    *,
    fixed_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if vocab_content < 1:
        raise ValueError("vocab_content must be >= 1")
    if pad_id != vocab_content:
        raise ValueError("identity expects pad_id == vocab_content")
    tokens_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for _ in range(batch_size):
        if fixed_n is not None:
            L = int(fixed_n)
        elif generator is not None:
            L = int(torch.randint(n_min, n_max + 1, (1,), generator=generator).item())
        else:
            L = int(torch.randint(n_min, n_max + 1, (1,)).item())
        if L < 1:
            raise ValueError("identity copy requires L >= 1")
        seq = torch.randint(0, vocab_content, (L,), device=device)
        lab = seq.clone()
        if L < max_len:
            pad_n = max_len - L
            seq = F.pad(seq, (0, pad_n), value=pad_id)
            lab = F.pad(lab, (0, pad_n), value=IGNORE_LABEL)
        tokens_list.append(seq)
        labels_list.append(lab)
    tokens = torch.stack(tokens_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    padding_mask = tokens == pad_id
    return tokens, labels, padding_mask


def val_set_meta_dict(args: argparse.Namespace) -> dict[str, Any]:
    vc = int(args.vocab_content)
    if args.task == "identity":
        vocab_size = vc + 1
    elif args.task in ("kv_assoc", "story_assoc"):
        vocab_size = vc + 3
    else:
        vocab_size = vc + 2
    return {
        "version": 3,
        "task": args.task,
        "val_batch": args.val_batch,
        "val_n": "random" if args.val_random_n else args.val_n,
        "n_min": args.n_min,
        "n_max": args.n_max,
        "vocab_content": vc,
        "vocab_size": vocab_size,
        "val_set_seed": args.val_set_seed,
        "gap_min": int(args.gap_min) if args.task == "gap_copy" else None,
        "gap_max": int(args.gap_max) if args.task == "gap_copy" else None,
        "val_gap": (
            "random"
            if args.task == "gap_copy" and args.val_random_n
            else (int(args.val_gap) if args.task == "gap_copy" else None)
        ),
        "dup_tail": bool(args.task == "dup_tail_copy"),
        "niah_hay_len": int(args.niah_hay_len) if args.task == "niah" else None,
        "niah_needle_len": int(args.niah_needle_len) if args.task == "niah" else None,
        "niah_prefix_needle_len": (
            int(args.niah_prefix_needle_len) if args.task == "niah_prefix" else None
        ),
        "niah_prefix_len_min": (
            int(args.niah_prefix_len_min) if args.task == "niah_prefix" else None
        ),
        "niah_prefix_len_max": (
            int(args.niah_prefix_len_max) if args.task == "niah_prefix" else None
        ),
        "niah_prefix_val_len": (
            int(args.niah_prefix_val_len) if args.task == "niah_prefix" else None
        ),
        "niah_prefix_val_ood": (
            bool(getattr(args, "niah_prefix_val_ood", False))
            if args.task == "niah_prefix"
            else None
        ),
        "kv_num_pairs": int(args.kv_num_pairs) if args.task == "kv_assoc" else None,
        "kv_dist_min": int(args.kv_dist_min) if args.task == "kv_assoc" else None,
        "kv_dist_max": int(args.kv_dist_max) if args.task == "kv_assoc" else None,
        "kv_val_dist_len": int(args.kv_val_dist_len) if args.task == "kv_assoc" else None,
        "kv_val_query_idx": int(args.kv_val_query_idx) if args.task == "kv_assoc" else None,
        "kv_pair_gap_min": (
            int(args.kv_pair_gap_min) if args.task == "kv_assoc" else None
        ),
        "kv_pair_gap_max": (
            int(args.kv_pair_gap_max) if args.task == "kv_assoc" else None
        ),
        "story_num_facts": int(args.story_num_facts) if args.task == "story_assoc" else None,
        "story_open_min": int(args.story_open_min) if args.task == "story_assoc" else None,
        "story_open_max": int(args.story_open_max) if args.task == "story_assoc" else None,
        "story_pair_gap_min": (
            int(args.story_pair_gap_min) if args.task == "story_assoc" else None
        ),
        "story_pair_gap_max": (
            int(args.story_pair_gap_max) if args.task == "story_assoc" else None
        ),
        "story_dist_min": int(args.story_dist_min) if args.task == "story_assoc" else None,
        "story_dist_max": int(args.story_dist_max) if args.task == "story_assoc" else None,
        "story_val_open_len": (
            int(args.story_val_open_len) if args.task == "story_assoc" else None
        ),
        "story_val_dist_len": (
            int(args.story_val_dist_len) if args.task == "story_assoc" else None
        ),
        "story_val_query_idx": (
            int(args.story_val_query_idx) if args.task == "story_assoc" else None
        ),
    }


def meta_compatible(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    if stored.get("task", "delimiter") != current.get("task", "delimiter"):
        return False
    keys = (
        "version",
        "val_batch",
        "val_n",
        "n_min",
        "n_max",
        "vocab_content",
        "vocab_size",
        "val_set_seed",
        "gap_min",
        "gap_max",
        "val_gap",
        "dup_tail",
        "niah_hay_len",
        "niah_needle_len",
        "niah_prefix_needle_len",
        "niah_prefix_len_min",
        "niah_prefix_len_max",
        "niah_prefix_val_len",
        "niah_prefix_val_ood",
        "kv_num_pairs",
        "kv_dist_min",
        "kv_dist_max",
        "kv_val_dist_len",
        "kv_val_query_idx",
        "kv_pair_gap_min",
        "kv_pair_gap_max",
        "story_num_facts",
        "story_open_min",
        "story_open_max",
        "story_pair_gap_min",
        "story_pair_gap_max",
        "story_dist_min",
        "story_dist_max",
        "story_val_open_len",
        "story_val_dist_len",
        "story_val_query_idx",
    )
    return all(stored.get(k) == current.get(k) for k in keys)


def save_fixed_val_set(
    path: Path,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    pad_mask: torch.Tensor,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tokens": tokens.cpu(),
            "labels": labels.cpu(),
            "pad_mask": pad_mask.cpu(),
            "meta": meta,
        },
        path,
    )


def load_fixed_val_set(
    path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    blob = torch.load(path, map_location=device)
    tokens = blob["tokens"].to(device)
    labels = blob["labels"].to(device)
    pad_mask = blob["pad_mask"].to(device)
    return tokens, labels, pad_mask, blob["meta"]


def ensure_fixed_val_set(
    args: argparse.Namespace,
    val_path: Path,
    max_len: int,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    meta = val_set_meta_dict(args)
    cpu = torch.device("cpu")
    if val_path.is_file() and not args.regen_val_set:
        t, lab, pm, old_meta = load_fixed_val_set(val_path, cpu)
        if meta_compatible(old_meta, meta):
            print(f"Loaded fixed val set from {val_path}")
            return t.to(device), lab.to(device), pm.to(device)
        print(f"Val set meta mismatch; regenerating {val_path}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.val_set_seed)
    if args.task == "identity":
        t, lab, pm = build_identity_copy_batch(
            args.val_batch,
            args.n_min,
            args.n_max,
            vocab_content,
            pad_id,
            max_len,
            cpu,
            generator=gen,
            fixed_n=None if args.val_random_n else args.val_n,
        )
    elif args.task == "gap_copy":
        t, lab, pm = build_gap_copy_batch(
            args.val_batch,
            args.n_min,
            args.n_max,
            int(args.gap_min),
            int(args.gap_max),
            vocab_content,
            pad_id,
            delim_id,
            max_len,
            cpu,
            gen,
            fixed_n=None if args.val_random_n else args.val_n,
            fixed_gap=None if args.val_random_n else int(args.val_gap),
        )
    elif args.task == "dup_tail_copy":
        t, lab, pm = build_dup_tail_copy_batch(
            args.val_batch,
            args.n_min,
            args.n_max,
            vocab_content,
            pad_id,
            delim_id,
            max_len,
            cpu,
            generator=gen,
            fixed_n=None if args.val_random_n else args.val_n,
        )
    elif args.task == "niah":
        t, lab, pm = build_niah_fixed_batch(
            args.val_batch,
            int(args.niah_hay_len),
            int(args.niah_needle_len),
            vocab_content,
            pad_id,
            delim_id,
            max_len,
            cpu,
            generator=gen,
        )
    elif args.task == "niah_prefix":
        t, lab, pm = build_niah_prefix_batch(
            args.val_batch,
            int(args.niah_prefix_needle_len),
            int(args.niah_prefix_len_min),
            int(args.niah_prefix_len_max),
            vocab_content,
            pad_id,
            delim_id,
            max_len,
            cpu,
            generator=gen,
            fixed_total_len=int(args.niah_prefix_val_len),
        )
    elif args.task == "kv_assoc":
        arrow_id = vocab_content + 2
        t, lab, pm = build_kv_assoc_batch(
            args.val_batch,
            int(args.kv_num_pairs),
            int(args.kv_dist_min),
            int(args.kv_dist_max),
            vocab_content,
            pad_id,
            delim_id,
            arrow_id,
            max_len,
            cpu,
            gen,
            pair_gap_min=int(args.kv_pair_gap_min),
            pair_gap_max=int(args.kv_pair_gap_max),
            fixed_dist_len=int(args.kv_val_dist_len),
            fixed_query_pair_idx=int(args.kv_val_query_idx),
        )
    elif args.task == "story_assoc":
        arrow_id = vocab_content + 2
        t, lab, pm = build_story_assoc_batch(
            args.val_batch,
            int(args.story_num_facts),
            int(args.story_open_min),
            int(args.story_open_max),
            int(args.story_pair_gap_min),
            int(args.story_pair_gap_max),
            int(args.story_dist_min),
            int(args.story_dist_max),
            vocab_content,
            pad_id,
            delim_id,
            arrow_id,
            max_len,
            cpu,
            gen,
            fixed_open_len=int(args.story_val_open_len),
            fixed_dist_len=int(args.story_val_dist_len),
            fixed_query_pair_idx=int(args.story_val_query_idx),
        )
    else:
        t, lab, pm = build_copy_batch(
            args.val_batch,
            args.n_min,
            args.n_max,
            vocab_content,
            pad_id,
            delim_id,
            max_len,
            cpu,
            generator=gen,
            fixed_n=None if args.val_random_n else args.val_n,
        )
    save_fixed_val_set(val_path, t, lab, pm, meta)
    print(f"Saved fixed val set ({args.val_batch} seqs) to {val_path}")
    return t.to(device), lab.to(device), pm.to(device)


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    B, L, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * L, V),
        labels.reshape(B * L),
        ignore_index=IGNORE_LABEL,
        reduction="mean",
    )


def accuracy_on_labeled(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float, float]:
    pred = logits.argmax(dim=-1)
    B = labels.shape[0]
    all_acc: list[float] = []
    first_acc: list[float] = []
    second_acc: list[float] = []
    for b in range(B):
        lab = labels[b]
        m = (lab != IGNORE_LABEL).nonzero(as_tuple=False).flatten()
        if m.numel() == 0:
            continue
        correct = pred[b, m] == lab[m]
        all_acc.append(float(correct.float().mean().item()))
        n = correct.shape[0]
        h = n // 2
        if h > 0:
            first_acc.append(float(correct[:h].float().mean().item()))
        if n > h:
            second_acc.append(float(correct[h:].float().mean().item()))

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return _mean(all_acc), _mean(first_acc), _mean(second_acc)


def levenshtein_distance(a: list[int], b: list[int]) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[lb]


def copy_segment_em_and_norm_levenshtein(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    pred = logits.argmax(dim=-1)
    B = labels.shape[0]
    ems: list[float] = []
    nlevs: list[float] = []
    for b in range(B):
        lab = labels[b]
        m = (lab != IGNORE_LABEL).nonzero(as_tuple=False).flatten()
        if m.numel() == 0:
            continue
        p = pred[b, m].detach().cpu().tolist()
        g = lab[m].detach().cpu().tolist()
        n = len(g)
        ems.append(1.0 if p == g else 0.0)
        nlevs.append(levenshtein_distance(p, g) / max(1, n))
    return (
        float(np.mean(ems)) if ems else float("nan"),
        float(np.mean(nlevs)) if nlevs else float("nan"),
    )


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def resolve_device(name: str) -> torch.device:
    key = name.strip().lower()
    if key == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def parse_acc_hit_thresholds(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def first_steps_val_acc_ge(
    rows: list[dict], thresholds: list[float]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for thr in thresholds:
        tag = format(thr, "g").replace(".", "p").replace("-", "neg")
        key = f"first_step_acc_ge_{tag}"
        hit: str | None = None
        for r in rows:
            if float(r["val_acc_all"]) >= thr:
                hit = str(r["step"])
                break
        result[key] = hit if hit is not None else "—"
    return result


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def effective_mirror_dim(dim: int, args: argparse.Namespace) -> int:
    """Logged width: ``dim`` for diagonal or --mirror-full-width; else --mirror-dim."""
    if bool(getattr(args, "mirror_no_silu", False)):
        return int(dim)
    if bool(getattr(args, "mirror_full_width", False)):
        return int(dim)
    return int(args.mirror_dim)


def bottleneck_mirror_r(dim: int, args: argparse.Namespace) -> int:
    """Inner ``r`` for bottleneck TinyMirror (ignored when ``--mirror-no-silu`` diagonal)."""
    if bool(getattr(args, "mirror_full_width", False)):
        return int(dim)
    return int(args.mirror_dim)


class TinyMirror(nn.Module):
    """Per-token mirror branch; starts disabled (zero contribution to residual).

    - **Bottleneck** (default): ``Linear(dim→r)``, optional nonlinearity on ``r``, ``Linear(r→dim)``;
      ``proj_out`` zero-init. Nonlinearity from ``bottleneck_activation`` (see ``--mirror-activation``).
    - **Diagonal** (``--mirror-no-silu``): per-channel affine ``scale * x + bias`` in ``dim``,
      no rank change; ``scale`` and ``bias`` zero-init (bottleneck activation unused).
    """

    def __init__(
        self,
        dim: int,
        mirror_dim: int = 8,
        *,
        bottleneck_activation: str = "silu",
        diagonal: bool = False,
    ) -> None:
        super().__init__()
        self.diagonal = bool(diagonal)
        if self.diagonal:
            self.diag_scale = nn.Parameter(torch.zeros(dim))
            self.diag_bias = nn.Parameter(torch.zeros(dim))
            return
        if bottleneck_activation not in MIRROR_BOTTLENECK_ACTIVATIONS:
            raise ValueError(
                f"bottleneck_activation must be one of {MIRROR_BOTTLENECK_ACTIVATIONS}"
            )
        self.proj_in = nn.Linear(dim, mirror_dim)
        self.act = tiny_mirror_bottleneck_activation(bottleneck_activation)
        self.proj_out = nn.Linear(mirror_dim, dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.diagonal:
            return self.diag_scale * x + self.diag_bias
        h = self.proj_in(x)
        if self.act is not None:
            h = self.act(h)
        return self.proj_out(h)


class TokenAttentionMemory(nn.Module):
    """Single low-rank self-attention memory branch (distinct from block MHA).

    ``attn = softmax(Q K^T / sqrt(head_dim))``, ``mem = attn @ V(x)`` with
    ``Q,K: dim→head_dim``, ``V: dim→dim``. ``V`` is zero-initialized so the
    branch starts with zero contribution (like TinyMirror ``proj_out``).
    """

    def __init__(self, dim: int, head_dim: int) -> None:
        super().__init__()
        if head_dim < 1:
            raise ValueError("head_dim must be >= 1")
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, head_dim)
        self.k_proj = nn.Linear(dim, head_dim)
        self.v_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scale = 1.0 / math.sqrt(float(self.head_dim))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).expand_as(scores),
                float("-inf"),
            )
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.matmul(attn, v)


class TransformerBlockWithTokenMirrors(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        dim_feedforward: int,
        mirror_dim: int,
        *,
        dropout: float = 0.0,
        ffn_activation: str = "relu",
        mirror_scale: float = 1.0,
        mirror_activation: str = "silu",
        mirror_diagonal: bool = False,
        mirror_attn_mem: bool = False,
        mirror_attn_mem_head_dim: int = 16,
    ) -> None:
        super().__init__()
        if ffn_activation not in ("relu", "gelu", "silu"):
            raise ValueError("ffn_activation must be relu, gelu, or silu")
        if mirror_activation not in MIRROR_BOTTLENECK_ACTIVATIONS:
            raise ValueError(
                f"mirror_activation must be one of {MIRROR_BOTTLENECK_ACTIVATIONS}"
            )
        if mirror_dim < 1:
            raise ValueError("mirror_dim must be >= 1")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if ffn_activation == "relu":
            self.ffn_act = F.relu
        elif ffn_activation == "gelu":
            self.ffn_act = F.gelu
        else:
            self.ffn_act = F.silu
        self.mirror = TinyMirror(
            dim,
            mirror_dim=mirror_dim,
            bottleneck_activation=mirror_activation,
            diagonal=mirror_diagonal,
        )
        self.mirror_scale = float(mirror_scale)
        self.attn_mem: TokenAttentionMemory | None = None
        if mirror_attn_mem:
            self.attn_mem = TokenAttentionMemory(dim, mirror_attn_mem_head_dim)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)
        x_norm = self.norm2(x)
        ff = self.linear2(self.dropout2(self.ffn_act(self.linear1(x_norm))))
        mirror_out = self.mirror_scale * self.mirror(x_norm)
        extra = mirror_out
        if self.attn_mem is not None:
            extra = extra + self.attn_mem(x_norm, src_key_padding_mask)
        x = x + self.dropout2(ff) + extra
        return x


class SeqModelTokenMirror(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        nhead: int,
        dim_feedforward: int | None = None,
        *,
        max_seq_len: int,
        pos_enc: str = "learned",
        mirror_dim: int = 8,
        ffn_activation: str = "relu",
        mirror_scale: float = 1.0,
        mirror_activation: str = "silu",
        mirror_diagonal: bool = False,
        mirror_attn_mem: bool = False,
        mirror_attn_mem_head_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * dim
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        if pos_enc not in ("learned", "none", "sinusoidal"):
            raise ValueError("pos_enc must be 'learned', 'none', or 'sinusoidal'")
        if mirror_dim < 1:
            raise ValueError("mirror_dim must be >= 1")
        self.pos_enc = pos_enc
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = (
            nn.Embedding(max_seq_len, dim) if pos_enc == "learned" else None
        )
        if pos_enc == "sinusoidal":
            self.register_buffer(
                "pos_sin",
                sinusoidal_position_encoding(max_seq_len, dim),
                persistent=False,
            )
        else:
            self.pos_sin = None  # type: ignore[assignment]
        self.blocks = nn.ModuleList(
            [
                TransformerBlockWithTokenMirrors(
                    dim,
                    nhead,
                    dim_feedforward,
                    mirror_dim,
                    dropout=dropout,
                    ffn_activation=ffn_activation,
                    mirror_scale=mirror_scale,
                    mirror_activation=mirror_activation,
                    mirror_diagonal=mirror_diagonal,
                    mirror_attn_mem=mirror_attn_mem,
                    mirror_attn_mem_head_dim=mirror_attn_mem_head_dim,
                )
                for _ in range(depth)
            ]
        )
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _b, seq_len = token_ids.shape
        x = self.embed(token_ids)
        if self.pos_embed is not None:
            pos = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
            x = x + self.pos_embed(pos)
        elif self.pos_enc == "sinusoidal":
            x = x + self.pos_sin[:seq_len].to(device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=src_key_padding_mask)
        return self.lm_head(x)


class TransformerEncoderLayerAttn(nn.Module):
    """Matches script8 baseline: Post-LN MHA + Post-LN FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation not in ("relu", "gelu", "silu"):
            raise ValueError("activation must be relu, gelu, or silu")
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        else:
            self.act = F.silu

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(src)
        attn_out, _ = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = src + self.dropout1(attn_out)
        h2 = self.norm2(src)
        ff = self.linear2(self.dropout2(self.act(self.linear1(h2))))
        src = src + self.dropout2(ff)
        return src


class SeqModelTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        nhead: int,
        dim_feedforward: int | None = None,
        *,
        max_seq_len: int,
        pos_enc: str = "learned",
        ffn_activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * dim
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        if pos_enc not in ("learned", "none", "sinusoidal"):
            raise ValueError("pos_enc must be 'learned', 'none', or 'sinusoidal'")
        self.pos_enc = pos_enc
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = (
            nn.Embedding(max_seq_len, dim) if pos_enc == "learned" else None
        )
        if pos_enc == "sinusoidal":
            self.register_buffer(
                "pos_sin",
                sinusoidal_position_encoding(max_seq_len, dim),
                persistent=False,
            )
        else:
            self.pos_sin = None  # type: ignore[assignment]
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerAttn(
                    dim, nhead, dim_feedforward, dropout=dropout, activation=ffn_activation
                )
                for _ in range(depth)
            ]
        )
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _b, seq_len = token_ids.shape
        x = self.embed(token_ids)
        if self.pos_embed is not None:
            pos = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
            x = x + self.pos_embed(pos)
        elif self.pos_enc == "sinusoidal":
            x = x + self.pos_sin[:seq_len].to(device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_run(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    model_name: str,
    model: nn.Module,
    device: torch.device,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    max_len: int,
    val_tokens: torch.Tensor,
    val_labels: torch.Tensor,
    val_pad_mask: torch.Tensor,
    num_steps: int | None = None,
    csv_suffix: str = "",
    legacy_transformer: bool = False,
    extra_row_fields: dict[str, str | float | int] | None = None,
    on_log_step: Callable[[int, nn.Module], None] | None = None,
    after_step: Callable[[int, nn.Module], None] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 777)
    best_val_acc = float("nan")
    best_val_step = 0

    steps_limit = int(args.steps) if num_steps is None else int(num_steps)
    total_opt_steps = steps_limit + 1
    if legacy_transformer:
        opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    else:
        opt = build_script10_optimizer(args, model)
    for step in range(steps_limit + 1):
        if legacy_transformer:
            lr_mult = 1.0
            cur_lr = float(args.lr)
        else:
            lr_mult = training_lr_multiplier(
                step,
                total_opt_steps=total_opt_steps,
                warmup_steps=int(args.warmup_steps),
                schedule=str(args.lr_schedule),
                lr_min_ratio=float(args.lr_min_ratio),
            )
            cur_lr = float(args.lr) * lr_mult
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        model.train()
        if args.task == "identity":
            tokens, labels, pad_mask = build_identity_copy_batch(
                args.batch,
                args.n_min,
                args.n_max,
                vocab_content,
                pad_id,
                max_len,
                device,
                generator=gen,
            )
        elif args.task == "gap_copy":
            tokens, labels, pad_mask = build_gap_copy_batch(
                args.batch,
                args.n_min,
                args.n_max,
                int(args.gap_min),
                int(args.gap_max),
                vocab_content,
                pad_id,
                delim_id,
                max_len,
                device,
                gen,
            )
        elif args.task == "dup_tail_copy":
            tokens, labels, pad_mask = build_dup_tail_copy_batch(
                args.batch,
                args.n_min,
                args.n_max,
                vocab_content,
                pad_id,
                delim_id,
                max_len,
                device,
                generator=gen,
            )
        elif args.task == "niah":
            tokens, labels, pad_mask = build_niah_fixed_batch(
                args.batch,
                int(args.niah_hay_len),
                int(args.niah_needle_len),
                vocab_content,
                pad_id,
                delim_id,
                max_len,
                device,
                generator=gen,
            )
        elif args.task == "niah_prefix":
            tokens, labels, pad_mask = build_niah_prefix_batch(
                args.batch,
                int(args.niah_prefix_needle_len),
                int(args.niah_prefix_len_min),
                int(args.niah_prefix_len_max),
                vocab_content,
                pad_id,
                delim_id,
                max_len,
                device,
                generator=gen,
                fixed_total_len=None,
            )
        elif args.task == "kv_assoc":
            arrow_id = vocab_content + 2
            tokens, labels, pad_mask = build_kv_assoc_batch(
                args.batch,
                int(args.kv_num_pairs),
                int(args.kv_dist_min),
                int(args.kv_dist_max),
                vocab_content,
                pad_id,
                delim_id,
                arrow_id,
                max_len,
                device,
                gen,
                pair_gap_min=int(args.kv_pair_gap_min),
                pair_gap_max=int(args.kv_pair_gap_max),
                fixed_dist_len=None,
                fixed_query_pair_idx=None,
            )
        elif args.task == "story_assoc":
            arrow_id = vocab_content + 2
            tokens, labels, pad_mask = build_story_assoc_batch(
                args.batch,
                int(args.story_num_facts),
                int(args.story_open_min),
                int(args.story_open_max),
                int(args.story_pair_gap_min),
                int(args.story_pair_gap_max),
                int(args.story_dist_min),
                int(args.story_dist_max),
                vocab_content,
                pad_id,
                delim_id,
                arrow_id,
                max_len,
                device,
                gen,
                fixed_open_len=None,
                fixed_dist_len=None,
                fixed_query_pair_idx=None,
            )
        else:
            tokens, labels, pad_mask = build_copy_batch(
                args.batch,
                args.n_min,
                args.n_max,
                vocab_content,
                pad_id,
                delim_id,
                max_len,
                device,
                generator=gen,
            )

        opt.zero_grad()
        logits = model(tokens, src_key_padding_mask=pad_mask)
        ce = masked_ce_loss(logits, labels)
        ce.backward()
        opt.step()

        if after_step is not None:
            after_step(step, model)

        if step % args.log_every == 0 or step == steps_limit:
            model.eval()
            with torch.no_grad():
                v_logits = model(val_tokens, src_key_padding_mask=val_pad_mask)
                vloss = masked_ce_loss(v_logits, val_labels).item()
                acc, acc1, acc2 = accuracy_on_labeled(v_logits, val_labels)
                _, v_nlev = copy_segment_em_and_norm_levenshtein(v_logits, val_labels)
            ce_f = float(ce.item())
            if math.isfinite(acc) and (
                not math.isfinite(best_val_acc) or acc > best_val_acc
            ):
                best_val_acc = acc
                best_val_step = step
            row_tr: dict[str, str | float | int] = {
                "model": model_name,
                "step": step,
                "lr": cur_lr,
                "train_loss": ce_f,
                "val_loss": vloss,
                "val_acc_all": acc,
                "val_acc_best": best_val_acc,
                "val_acc_best_step": best_val_step,
                "val_acc_copy_first_half": acc1,
                "val_acc_copy_second_half": acc2,
                "val_copy_norm_lev": v_nlev,
                "seed": args.seed,
                "mirror_full_width": bool(args.mirror_full_width),
                "mirror_effective_dim": effective_mirror_dim(int(args.dim), args),
                "mirror_dim": args.mirror_dim,
                "mirror_no_silu": bool(getattr(args, "mirror_no_silu", False)),
                "mirror_scale": float(args.mirror_scale),
                "mirror_activation": args.mirror_activation,
                "ffn_activation": args.ffn_activation,
                "mirror_attn_mem": bool(getattr(args, "mirror_attn_mem", False)),
                "mirror_attn_mem_head_dim": int(
                    getattr(args, "mirror_attn_mem_head_dim", 16)
                ),
                "n_max": args.n_max,
                "val_batch": args.val_batch,
                "val_n": "random" if args.val_random_n else args.val_n,
            }
            if extra_row_fields:
                row_tr.update(extra_row_fields)
            rows.append(row_tr)
            if on_log_step is not None:
                on_log_step(step, model)
            b_str = f"{best_val_acc:.3f}" if math.isfinite(best_val_acc) else "—"
            print(
                f"[{model_name}] step {step:5d} | lr {cur_lr:.2e} | train_ce {ce_f:.4f} | val_ce {vloss:.4f} | "
                f"acc {acc:.3f} |1st {acc1:.3f} | 2nd {acc2:.3f} | nLev {v_nlev:.3f} | "
                f"best_val_acc {b_str} @ {best_val_step}",
                flush=True,
            )
    b_done = f"{best_val_acc:.4f}" if math.isfinite(best_val_acc) else "—"
    print(
        f"[{model_name}] done | best_val_acc {b_done} @ step {best_val_step}",
        flush=True,
    )
    suf = csv_suffix if csv_suffix else ""
    save_csv(out_dir / f"training_{model_name}_seed{args.seed}{suf}.csv", rows)
    return rows


class CalibrationFailed(RuntimeError):
    """Early-stop: task too hard or not improving under --calibrate-steps."""


def _ease_task_args(args: argparse.Namespace) -> None:
    """Relax gap_copy / dup_tail_copy for --auto-ease-on-fail (delimiter not adjusted)."""
    if args.task == "gap_copy":
        args.n_min = max(3, int(args.n_min) - 1)
        args.n_max = max(int(args.n_min), int(args.n_max) - 1)
        args.gap_max = max(int(args.gap_min), int(args.gap_max) - 2)
        args.vocab_content = max(8, int(args.vocab_content) - 2)
        if not args.val_random_n:
            args.val_n = min(max(int(args.val_n), int(args.n_min)), int(args.n_max))
            args.val_gap = min(
                max(int(args.val_gap), int(args.gap_min)), int(args.gap_max)
            )
        print(
            f"[script10] auto-ease → gap_copy n in [{args.n_min},{args.n_max}] "
            f"gap in [{args.gap_min},{args.gap_max}] vocab={args.vocab_content}",
            flush=True,
        )
    elif args.task == "dup_tail_copy":
        nn = max(3, int(args.n_min) - 1)
        args.n_min = args.n_max = nn
        args.val_n = nn
        args.vocab_content = max(8, int(args.vocab_content) - 2)
        print(
            f"[script10] auto-ease → dup_tail n={nn} vocab={args.vocab_content}",
            flush=True,
        )
    else:
        raise CalibrationFailed("auto-ease only supports gap_copy / dup_tail_copy")


def plot_loss_and_acc(
    rows_by_model: dict[str, list[dict]],
    out_dir: Path,
    *,
    figure_basename: str = "figure_script10_loss_acc",
) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    ax_tr, ax_val_lin, ax_val_log, ax_acc = axes
    n_max_pts = max(len(r) for r in rows_by_model.values())
    use_markers = n_max_pts <= 40
    mrk = "o" if use_markers else None
    ms = 4 if use_markers else 0
    if len(rows_by_model) == 1:
        only = next(iter(rows_by_model.keys()))
        plot_title = f"Script10: {only} only"
    else:
        plot_title = "Script10: token_mirror vs transformer (single run)"

    for name, rows in sorted(rows_by_model.items()):
        xs = [r["step"] for r in rows]
        ax_tr.plot(
            xs, [r["train_loss"] for r in rows], label=name, marker=mrk, markersize=ms, lw=1.2
        )
        ax_val_lin.plot(
            xs, [r["val_loss"] for r in rows], label=name, marker=mrk, markersize=ms, lw=1.2
        )
        ax_val_log.plot(
            xs, [r["val_loss"] for r in rows], label=name, marker=mrk, markersize=ms, lw=1.2
        )
        ax_acc.plot(
            xs, [r["val_acc_all"] for r in rows], label=name, marker=mrk, markersize=ms, lw=1.2
        )

    ax_tr.set_ylabel("Train CE")
    ax_tr.set_title(plot_title)
    ax_tr.legend(fontsize=7, ncol=2, loc="upper right")
    ax_tr.grid(True, ls=":", alpha=0.45)
    ax_val_lin.set_ylabel("Val CE (linear)")
    ax_val_lin.grid(True, ls=":", alpha=0.45)
    ax_val_log.set_ylabel("Val CE (log)")
    ax_val_log.set_yscale("log")
    ax_val_log.grid(True, which="both", ls=":", alpha=0.45)
    ax_acc.set_ylabel("Val acc (labeled)")
    ax_acc.set_xlabel("Step")
    ax_acc.legend(fontsize=7, loc="best")
    ax_acc.grid(True, ls=":", alpha=0.45)
    acc_vals = [float(r["val_acc_all"]) for rows in rows_by_model.values() for r in rows]
    finite = [v for v in acc_vals if np.isfinite(v)]
    if finite:
        lo, hi = min(finite), max(finite)
        pad = max(0.02, (hi - lo) * 0.25, hi * 0.1)
        y_lo = max(0.0, lo - pad)
        y_hi = hi + pad
        if y_hi > 1.0:
            y_hi = min(1.05, y_hi)
        else:
            y_hi = min(1.0, y_hi)
        ax_acc.set_ylim(y_lo, y_hi)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{figure_basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{figure_basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_training_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_multi_seed_training_curves(root_out: Path, *, title_note: str = "") -> None:
    seed_dirs = sorted(
        p for p in root_out.iterdir() if p.is_dir() and p.name.startswith("seed_")
    )
    if not seed_dirs:
        raise FileNotFoundError(f"no seed_* subdirs under {root_out}")
    models = ["token_mirror", "transformer"]
    series: dict[str, list[list[dict[str, str]]]] = {m: [] for m in models}
    for sd in seed_dirs:
        for m in models:
            paths = sorted(sd.glob(f"training_{m}_seed*.csv"))
            if paths:
                series[m].append(load_training_csv(paths[0]))
    if not series["token_mirror"]:
        raise FileNotFoundError(
            f"missing training_token_mirror_seed*.csv under {root_out}/seed_*"
        )
    n_seeds = len(series["token_mirror"])
    for m in models:
        if series[m] and len(series[m]) != n_seeds:
            raise ValueError(f"expected {n_seeds} seeds for {m}, got {len(series[m])}")
    ref_steps = [int(r["step"]) for r in series["token_mirror"][0]]
    for m in models:
        if not series[m]:
            continue
        for idx, rows in enumerate(series[m]):
            if [int(r["step"]) for r in rows] != ref_steps:
                raise ValueError(f"{m}: seed {idx} step grid mismatch")

    def stack_metric(model: str, key: str) -> np.ndarray:
        return np.array(
            [[float(r[key]) for r in rows] for rows in series[model]], dtype=float
        )

    apply_paper_style()
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    ax_tr, ax_val_lin, ax_val_log, ax_acc = axes
    s_arr = np.array(ref_steps, dtype=float)
    colors = {"token_mirror": "C0", "transformer": "C1"}
    labels_short = {"token_mirror": "Token mirror", "transformer": "Transformer"}

    for m in models:
        if not series[m]:
            continue
        Yt = stack_metric(m, "train_loss")
        Yv = stack_metric(m, "val_loss")
        Ya = stack_metric(m, "val_acc_all")
        ddof = 1 if Yt.shape[0] > 1 else 0
        mt, st = Yt.mean(0), Yt.std(0, ddof=ddof)
        mv, sv = Yv.mean(0), Yv.std(0, ddof=ddof)
        ma, sa = Ya.mean(0), Ya.std(0, ddof=ddof)
        c = colors[m]
        ls = labels_short[m]
        ax_tr.plot(s_arr, mt, label=f"{ls} (mean)", color=c, lw=1.5)
        ax_tr.fill_between(s_arr, mt - st, mt + st, color=c, alpha=0.22)
        ax_val_lin.plot(s_arr, mv, label=f"{ls} (mean)", color=c, lw=1.5)
        ax_val_lin.fill_between(s_arr, mv - sv, mv + sv, color=c, alpha=0.22)
        ax_val_log.plot(s_arr, mv, label=f"{ls} (mean)", color=c, lw=1.5)
        ax_val_log.fill_between(s_arr, mv - sv, mv + sv, color=c, alpha=0.22)
        ax_acc.plot(s_arr, ma, label=f"{ls} (mean)", color=c, lw=1.5)
        ax_acc.fill_between(s_arr, ma - sa, ma + sa, color=c, alpha=0.22)

    subt = f"Multi-seed mean ± std (N={n_seeds})"
    if title_note:
        subt = f"{subt} — {title_note}"
    ax_tr.set_title(subt)
    ax_tr.set_ylabel("Train CE")
    ax_tr.legend(fontsize=7, ncol=2, loc="upper right")
    ax_tr.grid(True, ls=":", alpha=0.45)
    ax_val_lin.set_ylabel("Val CE (linear)")
    ax_val_lin.grid(True, ls=":", alpha=0.45)
    ax_val_log.set_ylabel("Val CE (log)")
    ax_val_log.set_yscale("log")
    ax_val_log.grid(True, which="both", ls=":", alpha=0.45)
    ax_acc.set_ylabel("Val acc (labeled)")
    ax_acc.set_xlabel("Step")
    ax_acc.legend(fontsize=7, loc="best")
    ax_acc.grid(True, ls=":", alpha=0.45)
    acc_vals: list[float] = []
    for m in models:
        if series[m]:
            acc_vals.extend(stack_metric(m, "val_acc_all").mean(0).tolist())
    finite = [v for v in acc_vals if np.isfinite(v)]
    if finite:
        lo, hi = min(finite), max(finite)
        pad = max(0.02, (hi - lo) * 0.25, hi * 0.1)
        y_lo = max(0.0, lo - pad)
        y_hi = hi + pad
        if y_hi > 1.0:
            y_hi = min(1.05, y_hi)
        else:
            y_hi = min(1.0, y_hi)
        ax_acc.set_ylim(y_lo, y_hi)
    ax_acc.set_xlim(float(s_arr.min()), float(s_arr.max()))
    fig.tight_layout()
    root_out.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        root_out / "figure_script10_loss_acc_multi_seed.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        root_out / "figure_script10_loss_acc_multi_seed.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def write_short_metrics_report(
    out_dir: Path,
    *,
    seed: int,
    task: str,
    rows_mirror: list[dict],
    rows_tfmr: list[dict],
    first_hits_mirror: dict[str, str] | None,
    first_hits_tfmr: dict[str, str] | None,
    val_line_override: str | None = None,
    baseline_only: bool = False,
) -> None:
    lines: list[str] = []
    lines.append("=== Script10 metrics report ===")
    if baseline_only:
        lines.append("Run: Transformer baseline only (--baseline-only).")
    else:
        lines.append(
            "Compare: token_mirror (per-token TinyMirror branch) vs transformer."
        )
    lines.append(f"seed={seed}")
    lines.append(f"task={task}")
    if val_line_override:
        lines.append(val_line_override)
    elif rows_mirror:
        vr = rows_mirror[0]
        lines.append(
            f"val: cached batch size={vr.get('val_batch')}, n={vr.get('val_n')!r} "
            "(shared across both models)."
        )
    elif rows_tfmr:
        vr = rows_tfmr[0]
        lines.append(
            f"val: cached batch size={vr.get('val_batch')}, n={vr.get('val_n')!r} "
            "(transformer run)."
        )
    lines.append("")

    def block(name: str, rows: list[dict]) -> None:
        last = rows[-1]
        lines.append(f"[{name}] final @ step {last['step']}")
        lines.append(f"  train_ce={float(last['train_loss']):.6f}")
        lines.append(f"  val_ce={float(last['val_loss']):.6f}")
        lines.append(f"  val_acc_all={float(last['val_acc_all']):.6f}")
        if last.get("val_acc_best") is not None:
            lines.append(
                f"  best_val_acc={float(last['val_acc_best']):.6f} @ step {last['val_acc_best_step']}"
            )
        lines.append("")

    if rows_mirror:
        block("token_mirror", rows_mirror)
    block("transformer", rows_tfmr)
    if first_hits_mirror or first_hits_tfmr:
        lines.append(
            "[Time-to-accuracy] first logged step where val_acc_all >= threshold (— = never)."
        )
        keys = sorted(
            set(first_hits_mirror or []) | set(first_hits_tfmr or []),
            key=lambda s: s,
        )
        for k in keys:
            mv = (
                (first_hits_mirror or {}).get(k, "")
                if rows_mirror
                else "—"
            )
            tv = (first_hits_tfmr or {}).get(k, "")
            lines.append(f"  {k}: mirror={mv} tfmr={tv}")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (out_dir / f"metrics_report_seed{seed}.txt").write_text(text, encoding="utf-8")
    print(text)


def run_one_seed(args: argparse.Namespace, subdir: str | None = None) -> dict[str, float]:
    device = resolve_device(args.device)
    print(f"Device: {device} (requested {args.device!r})")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _baseline_only = bool(getattr(args, "baseline_only", False))
    if _baseline_only and int(getattr(args, "calibrate_steps", 0) or 0) > 0:
        raise ValueError("--baseline-only cannot be used with --calibrate-steps > 0")

    if args.dim % args.nhead != 0:
        raise ValueError("dim must be divisible by nhead.")
    if not _baseline_only and not bool(getattr(args, "mirror_full_width", False)):
        md = int(args.mirror_dim)
        if md < 1 or md > int(args.dim):
            raise ValueError(
                "Without --mirror-full-width: require 1 <= mirror-dim <= dim."
            )
    if (
        args.task not in ("niah", "niah_prefix", "kv_assoc", "story_assoc")
        and not args.val_random_n
    ):
        if args.val_n < args.n_min or args.val_n > args.n_max:
            raise ValueError("val_n must satisfy n_min <= val_n <= n_max (or --val-random-n).")
    if args.task == "gap_copy":
        if int(args.gap_min) < 0 or int(args.gap_max) < int(args.gap_min):
            raise ValueError("gap_copy requires 0 <= gap-min <= gap-max")
        if not args.val_random_n:
            if int(args.val_gap) < int(args.gap_min) or int(args.val_gap) > int(args.gap_max):
                raise ValueError("val-gap must lie in [gap-min, gap-max] for gap_copy")
    if args.task == "niah":
        if int(args.niah_hay_len) < 0 or int(args.niah_needle_len) < 1:
            raise ValueError("niah requires --niah-hay-len >= 0 and --niah-needle-len >= 1")
    if args.task == "niah_prefix":
        nl = int(args.niah_prefix_needle_len)
        lo = int(args.niah_prefix_len_min)
        hi = int(args.niah_prefix_len_max)
        vl = int(args.niah_prefix_val_len)
        ood = bool(getattr(args, "niah_prefix_val_ood", False))
        if nl < 1 or lo > hi or lo < 2 * nl + 1 or hi > 1024 or vl > 1024:
            raise ValueError(
                "niah_prefix: invalid needle or length bounds (cap len_max/val_len at 1024)"
            )
        if vl < 2 * nl + 1:
            raise ValueError("niah_prefix: val_len must be >= 2*needle_len+1")
        if ood:
            if vl <= hi:
                raise ValueError(
                    "niah_prefix: --niah-prefix-val-ood requires val_len > len_max"
                )
        elif vl < lo or vl > hi:
            raise ValueError("niah_prefix: val_len must be in [len_min, len_max] (or use val-ood)")
    if args.task == "kv_assoc":
        P = int(args.kv_num_pairs)
        vc = int(args.vocab_content)
        key_hi = vc // 2
        if P < 1 or key_hi < P or (vc - key_hi) < P:
            raise ValueError(
                "kv_assoc: need kv-num-pairs >= 1 and vocab_content large enough "
                "(keys in [0,floor(v/2)), values in [floor(v/2),v))."
            )
        dm, dx = int(args.kv_dist_min), int(args.kv_dist_max)
        if dm < 0 or dx < dm:
            raise ValueError("kv_assoc: require 0 <= kv-dist-min <= kv-dist-max")
        vdl = int(args.kv_val_dist_len)
        if vdl < dm or vdl > dx:
            raise ValueError(
                "kv_assoc: kv-val-dist-len must lie in [kv-dist-min, kv-dist-max] "
                "for in-distribution validation distractor length."
            )
        pg0, pg1 = int(args.kv_pair_gap_min), int(args.kv_pair_gap_max)
        if pg0 < 0 or pg1 < pg0:
            raise ValueError("kv_assoc: need 0 <= kv-pair-gap-min <= kv-pair-gap-max")
    if args.task == "story_assoc":
        P = int(args.story_num_facts)
        vc = int(args.vocab_content)
        key_hi = vc // 2
        if P < 1 or key_hi < P or (vc - key_hi) < P:
            raise ValueError(
                "story_assoc: need story-num-facts >= 1 and vocab_content large enough "
                "(keys in [0,floor(v/2)), values in [floor(v/2),v))."
            )
        om, ox = int(args.story_open_min), int(args.story_open_max)
        if om < 0 or ox < om:
            raise ValueError("story_assoc: need 0 <= story-open-min <= story-open-max")
        pg0, pg1 = int(args.story_pair_gap_min), int(args.story_pair_gap_max)
        if pg0 < 0 or pg1 < pg0:
            raise ValueError(
                "story_assoc: need 0 <= story-pair-gap-min <= story-pair-gap-max"
            )
        dm, dx = int(args.story_dist_min), int(args.story_dist_max)
        if dm < 0 or dx < dm:
            raise ValueError("story_assoc: require 0 <= story-dist-min <= story-dist-max")
        vol = int(args.story_val_open_len)
        if vol < om or vol > ox:
            raise ValueError(
                "story_assoc: story-val-open-len must lie in [story-open-min, story-open-max]."
            )
        vdl = int(args.story_val_dist_len)
        if vdl < dm or vdl > dx:
            raise ValueError(
                "story_assoc: story-val-dist-len must lie in [story-dist-min, story-dist-max]."
            )
        qix = int(args.story_val_query_idx)
        if qix < 0 or qix >= P:
            raise ValueError("story_assoc: story-val-query-idx must be in [0, P-1].")

    vocab_content = args.vocab_content
    pad_id = vocab_content
    if args.task == "identity":
        vocab_size = vocab_content + 1
        delim_id = vocab_content + 1
    elif args.task in ("kv_assoc", "story_assoc"):
        delim_id = vocab_content + 1
        vocab_size = vocab_content + 3
    else:
        delim_id = vocab_content + 1
        vocab_size = vocab_content + 2
    if args.task == "identity":
        max_len = args.n_max
    elif args.task == "gap_copy":
        max_len = int(args.n_max) + int(args.gap_max) + 1 + int(args.n_max)
    elif args.task == "dup_tail_copy":
        max_len = 3 * int(args.n_max) + 1
    elif args.task == "niah":
        max_len = 2 * int(args.niah_hay_len) + 2 * int(args.niah_needle_len) + 1
    elif args.task == "niah_prefix":
        max_len = max(int(args.niah_prefix_len_max), int(args.niah_prefix_val_len))
    elif args.task == "kv_assoc":
        Pk = int(args.kv_num_pairs)
        max_len = (
            3 * Pk
            + max(0, Pk - 1) * int(args.kv_pair_gap_max)
            + 1
            + int(args.kv_dist_max)
            + 2
        )
    elif args.task == "story_assoc":
        Ps = int(args.story_num_facts)
        max_len = (
            int(args.story_open_max)
            + 3 * Ps
            + max(0, Ps - 1) * int(args.story_pair_gap_max)
            + 1
            + int(args.story_dist_max)
            + 2
        )
    else:
        max_len = 2 * args.n_max + 1
    dim_ff = 4 * args.dim

    gap_note = ""
    if args.task == "gap_copy":
        gap_note = f" gap in [{args.gap_min},{args.gap_max}] val_gap={args.val_gap}"
    niah_note = ""
    if args.task == "niah":
        niah_note = (
            f" niah hay_each={args.niah_hay_len} needle_len={args.niah_needle_len}"
        )
    niah_pref_note = ""
    if args.task == "niah_prefix":
        _ood = bool(getattr(args, "niah_prefix_val_ood", False))
        niah_pref_note = (
            f" niah_prefix needle@0..{int(args.niah_prefix_needle_len) - 1} "
            f"core_len U[{args.niah_prefix_len_min},{args.niah_prefix_len_max}] "
            f"val_len={args.niah_prefix_val_len}"
            f"{' val_OOD' if _ood else ''}"
        )
    kv_note = ""
    if args.task == "kv_assoc":
        kv_note = (
            f" kv_assoc P={args.kv_num_pairs} pair_gap U[{args.kv_pair_gap_min},{args.kv_pair_gap_max}] "
            f"distr U[{args.kv_dist_min},{args.kv_dist_max}] "
            f"val_d={args.kv_val_dist_len} val_qidx={args.kv_val_query_idx}"
        )
    story_note = ""
    if args.task == "story_assoc":
        story_note = (
            f" story_assoc facts={args.story_num_facts} open U[{args.story_open_min},{args.story_open_max}] "
            f"pair_gap U[{args.story_pair_gap_min},{args.story_pair_gap_max}] "
            f"distr U[{args.story_dist_min},{args.story_dist_max}] "
            f"val_open={args.story_val_open_len} val_d={args.story_val_dist_len} val_qidx={args.story_val_query_idx}"
        )
    _mr = effective_mirror_dim(int(args.dim), args)
    _attn_mem = bool(getattr(args, "mirror_attn_mem", False))
    _attn_hd = int(getattr(args, "mirror_attn_mem_head_dim", 16))
    print(
        f"[script10] RUN | task={args.task} | vocab_content={vocab_content} vocab_size={vocab_size} | "
        f"n in [{args.n_min},{args.n_max}] val_n={'random' if args.val_random_n else args.val_n}"
        f"{gap_note}{niah_note}{niah_pref_note}{kv_note}{story_note} | "
        f"mirror_full_width={bool(args.mirror_full_width)} mirror_eff_r={_mr} "
        f"mirror_no_silu_diag={bool(getattr(args, 'mirror_no_silu', False))} "
        f"mirror_act={args.mirror_activation} ffn_act={args.ffn_activation} "
        f"mirror_attn_mem={_attn_mem} mirror_attn_mem_hd={_attn_hd} | "
        f"baseline_only={_baseline_only} | "
        f"max_len={max_len} | steps={args.steps} log_every={args.log_every} | "
        f"preset={args.preset or 'none'}",
        flush=True,
    )
    _pe_note = (
        f"pos_enc={args.pos_enc} (token_mirror; transformer={_transformer_pos_enc(args)} "
        f"via --transformer-baseline {getattr(args, 'transformer_baseline', 'legacy')})"
    )
    print(
        f"[script10] train: optim={args.optimizer} weight_decay={args.weight_decay} lr={args.lr} | "
        f"lr_schedule={args.lr_schedule} warmup_steps={args.warmup_steps} lr_min_ratio={args.lr_min_ratio} | "
        f"{_pe_note}",
        flush=True,
    )
    if args.task == "identity":
        print(
            "[script10] WARNING: task=identity is same-index labeling; with bidirectional MHA val acc "
            "often jumps to ~1 quickly (not a useful memory benchmark). Prefer "
            "--preset script10-delimiter-copy or --task delimiter / gap_copy / dup_tail_copy.",
            flush=True,
        )
    if args.task == "niah":
        print(
            "[script10] NOTE: task=niah uses a TransformerEncoder (bidirectional MHA). Retrieval is easier "
            "than causal LM needle tests; this still checks whether the baseline can copy the buried needle "
            "to the tail under CE on the last needle_len positions.",
            flush=True,
        )
    if args.task == "niah_prefix":
        print(
            "[script10] NOTE: task=niah_prefix = needle at positions 0..K-1, random filler, junk at tail; "
            "CE only on last K positions (copy prefix needle). Bidirectional encoder can attend back to prefix. "
            "Use --niah-prefix-val-ood + val_len > len_max for harder length extrapolation.",
            flush=True,
        )
    if args.task == "kv_assoc":
        print(
            "[script10] NOTE: task=kv_assoc = P key→value pairs as [key,ARROW,value] triples in random pair order "
            "(optional random gaps between triples), then DELIM, random distractor, query [key,ARROW]; "
            "CE on query ARROW predicts the value (content, not fixed slot position).",
            flush=True,
        )
    if args.task == "story_assoc":
        print(
            "[script10] NOTE: task=story_assoc = random opening tokens (pseudo story), then same shuffled facts + "
            "distractor + content QA query [entity,ARROW] (CE on ARROW → detail).",
            flush=True,
        )

    root_out = Path(args.output_dir)
    out_dir = root_out / subdir if subdir else root_out
    out_dir.mkdir(parents=True, exist_ok=True)

    vsp = Path(args.val_set_file)
    val_path = vsp if vsp.is_absolute() else root_out / vsp
    val_tokens, val_labels, val_pad_mask = ensure_fixed_val_set(
        args,
        val_path,
        max_len,
        vocab_content,
        pad_id,
        delim_id,
        device,
    )

    mirror: SeqModelTokenMirror | None = None
    _mdiag = bool(getattr(args, "mirror_no_silu", False))
    _r_block = (
        int(args.mirror_dim)
        if _mdiag
        else bottleneck_mirror_r(int(args.dim), args)
    )
    _mirror_report = effective_mirror_dim(int(args.dim), args)
    if not _baseline_only:
        mirror = SeqModelTokenMirror(
            vocab_size,
            args.dim,
            args.depth,
            args.nhead,
            dim_feedforward=dim_ff,
            max_seq_len=max_len,
            pos_enc=args.pos_enc,
            mirror_dim=_r_block,
            ffn_activation=args.ffn_activation,
            mirror_scale=args.mirror_scale,
            mirror_activation=args.mirror_activation,
            mirror_diagonal=_mdiag,
            mirror_attn_mem=_attn_mem,
            mirror_attn_mem_head_dim=_attn_hd,
            dropout=0.0,
        ).to(device)
    tfmr = SeqModelTransformer(
        vocab_size,
        args.dim,
        args.depth,
        args.nhead,
        dim_feedforward=dim_ff,
        max_seq_len=max_len,
        pos_enc=_transformer_pos_enc(args),
        ffn_activation=args.ffn_activation,
        dropout=0.0,
    ).to(device)

    param_rows: list[dict[str, Any]] = []
    if not _baseline_only:
        assert mirror is not None
        param_rows.append(
            {
                "model": "token_mirror",
                "params": count_parameters(mirror),
                "dim": args.dim,
                "depth": args.depth,
                "mirror_full_width": bool(args.mirror_full_width),
                "mirror_effective_dim": _mirror_report,
                "mirror_dim_arg": args.mirror_dim,
                "mirror_no_silu": bool(getattr(args, "mirror_no_silu", False)),
                "mirror_scale": float(args.mirror_scale),
                "mirror_activation": args.mirror_activation,
                "ffn_activation": args.ffn_activation,
                "mirror_attn_mem": _attn_mem,
                "mirror_attn_mem_head_dim": _attn_hd,
                "dim_feedforward": dim_ff,
                "nhead": args.nhead,
            }
        )
    param_rows.append(
        {
            "model": "transformer",
            "params": count_parameters(tfmr),
            "dim": args.dim,
            "depth": args.depth,
            "mirror_full_width": "",
            "mirror_effective_dim": "",
            "mirror_dim_arg": "",
            "mirror_no_silu": "",
            "mirror_scale": "",
            "mirror_activation": "",
            "mirror_attn_mem": "",
            "mirror_attn_mem_head_dim": "",
            "ffn_activation": args.ffn_activation,
            "dim_feedforward": dim_ff,
            "nhead": args.nhead,
        }
    )
    save_csv(out_dir / f"param_counts_seed{args.seed}.csv", param_rows)
    print("Parameter counts:", param_rows)
    if _transformer_legacy_baseline(args):
        print(
            "[script10] transformer baseline=legacy: Adam, constant lr (no schedule), no pos enc; "
            "token_mirror uses --optimizer / --lr-schedule / --pos-enc. "
            "Use --transformer-baseline matched for identical training recipe on both.",
            flush=True,
        )

    cal_steps = int(getattr(args, "calibrate_steps", 0) or 0)
    if cal_steps > 0 and not _baseline_only:
        min_best = float(getattr(args, "calibrate_min_best_acc", 0.28))
        rise_tol = float(getattr(args, "calibrate_rise_tol", 0.02))
        print(
            f"[script10] CALIBRATION: token_mirror {cal_steps} steps "
            f"(need best_val_acc>={min_best}, last vs prev rise tol={rise_tol})...",
            flush=True,
        )
        cal_rows = train_run(
            args,
            out_dir,
            model_name="token_mirror",
            model=mirror,
            device=device,
            vocab_content=vocab_content,
            pad_id=pad_id,
            delim_id=delim_id,
            max_len=max_len,
            val_tokens=val_tokens,
            val_labels=val_labels,
            val_pad_mask=val_pad_mask,
            num_steps=cal_steps,
            csv_suffix="_calib",
        )
        best = float(cal_rows[-1]["val_acc_best"])
        if best < min_best:
            raise CalibrationFailed(
                f"best_val_acc={best:.3f} < {min_best} after {cal_steps} steps"
            )
        if len(cal_rows) >= 2:
            last = float(cal_rows[-1]["val_acc_all"])
            prev = float(cal_rows[-2]["val_acc_all"])
            if last + rise_tol < prev:
                raise CalibrationFailed(
                    f"val acc not rising (last {last:.3f} vs prev {prev:.3f}, tol={rise_tol})"
                )
        print(
            "[script10] calibration passed; reinitializing models for full run.",
            flush=True,
        )
        assert mirror is not None
        mirror = SeqModelTokenMirror(
            vocab_size,
            args.dim,
            args.depth,
            args.nhead,
            dim_feedforward=dim_ff,
            max_seq_len=max_len,
            pos_enc=args.pos_enc,
            mirror_dim=_r_block,
            ffn_activation=args.ffn_activation,
            mirror_scale=args.mirror_scale,
            mirror_activation=args.mirror_activation,
            mirror_diagonal=_mdiag,
            mirror_attn_mem=_attn_mem,
            mirror_attn_mem_head_dim=_attn_hd,
            dropout=0.0,
        ).to(device)
        tfmr = SeqModelTransformer(
            vocab_size,
            args.dim,
            args.depth,
            args.nhead,
            dim_feedforward=dim_ff,
            max_seq_len=max_len,
            pos_enc=_transformer_pos_enc(args),
            ffn_activation=args.ffn_activation,
            dropout=0.0,
        ).to(device)

    _leg_t = _transformer_legacy_baseline(args)
    rows_mirror: list[dict[str, str | float | int]] = []
    if not _baseline_only:
        assert mirror is not None
        rows_mirror = train_run(
            args,
            out_dir,
            model_name="token_mirror",
            model=mirror,
            device=device,
            vocab_content=vocab_content,
            pad_id=pad_id,
            delim_id=delim_id,
            max_len=max_len,
            val_tokens=val_tokens,
            val_labels=val_labels,
            val_pad_mask=val_pad_mask,
            legacy_transformer=False,
        )
    rows_tfmr = train_run(
        args,
        out_dir,
        model_name="transformer",
        model=tfmr,
        device=device,
        vocab_content=vocab_content,
        pad_id=pad_id,
        delim_id=delim_id,
        max_len=max_len,
        val_tokens=val_tokens,
        val_labels=val_labels,
        val_pad_mask=val_pad_mask,
        legacy_transformer=_leg_t,
    )

    if _baseline_only:
        plot_loss_and_acc({"transformer": rows_tfmr}, out_dir)
    else:
        plot_loss_and_acc(
            {"token_mirror": rows_mirror, "transformer": rows_tfmr}, out_dir
        )

    thrs = parse_acc_hit_thresholds(args.acc_hit_thresholds)
    fh_m = (
        first_steps_val_acc_ge(rows_mirror, thrs)
        if thrs and rows_mirror
        else {}
    )
    fh_t = first_steps_val_acc_ge(rows_tfmr, thrs) if thrs else {}
    val_line: str | None = None
    if args.task == "niah":
        val_line = (
            f"val: cached batch size={args.val_batch}, "
            f"niah hay_each={args.niah_hay_len} needle_len={args.niah_needle_len} "
            "(shared across both models)."
        )
    elif args.task == "niah_prefix":
        val_line = (
            f"val: cached batch size={args.val_batch}, "
            f"niah_prefix fixed core_len={args.niah_prefix_val_len} "
            f"needle_len={args.niah_prefix_needle_len} "
            f"train core_len U[{args.niah_prefix_len_min},{args.niah_prefix_len_max}] "
            "(shared across both models)."
        )
    elif args.task == "kv_assoc":
        val_line = (
            f"val: cached batch size={args.val_batch}, "
            f"kv_assoc P={args.kv_num_pairs} pair_gap in [{args.kv_pair_gap_min},{args.kv_pair_gap_max}] "
            f"val_distr_len={args.kv_val_dist_len} query_pair_idx={args.kv_val_query_idx} "
            "(shared across both models)."
        )
    elif args.task == "story_assoc":
        val_line = (
            f"val: cached batch size={args.val_batch}, "
            f"story_assoc P={args.story_num_facts} val_open_len={args.story_val_open_len} "
            f"val_distr_len={args.story_val_dist_len} query_pair_idx={args.story_val_query_idx} "
            "(shared across both models)."
        )
    write_short_metrics_report(
        out_dir,
        seed=args.seed,
        task=args.task,
        rows_mirror=rows_mirror,
        rows_tfmr=rows_tfmr,
        first_hits_mirror=fh_m or None,
        first_hits_tfmr=fh_t or None,
        val_line_override=val_line,
        baseline_only=_baseline_only,
    )

    lt = rows_tfmr[-1]
    summary: dict[str, Any]
    if _baseline_only:
        summary = {
            "seed": args.seed,
            "mirror_final_val": float("nan"),
            "tfmr_final_val": lt["val_loss"],
            "mirror_final_acc": float("nan"),
            "tfmr_final_acc": lt["val_acc_all"],
            "acc_hit_thresholds": args.acc_hit_thresholds,
            "subdir": subdir or ".",
            "baseline_only": True,
            "ffn_activation": args.ffn_activation,
        }
        for k, v in fh_t.items():
            summary[f"tfmr_{k}"] = v
    else:
        lm = rows_mirror[-1]
        summary = {
            "seed": args.seed,
            "mirror_final_val": lm["val_loss"],
            "tfmr_final_val": lt["val_loss"],
            "mirror_final_acc": lm["val_acc_all"],
            "tfmr_final_acc": lt["val_acc_all"],
            "acc_hit_thresholds": args.acc_hit_thresholds,
            "subdir": subdir or ".",
            "baseline_only": False,
            "ffn_activation": args.ffn_activation,
            "mirror_activation": args.mirror_activation,
            "mirror_attn_mem": bool(getattr(args, "mirror_attn_mem", False)),
            "mirror_attn_mem_head_dim": int(
                getattr(args, "mirror_attn_mem_head_dim", 16)
            ),
        }
        for k, v in fh_m.items():
            summary[f"mirror_{k}"] = v
        for k, v in fh_t.items():
            summary[f"tfmr_{k}"] = v
    save_csv(out_dir / f"summary_seed{args.seed}.csv", [summary])
    return {
        "mirror_final_val": float(summary["mirror_final_val"]),
        "tfmr_final_val": float(summary["tfmr_final_val"]),
        "mirror_final_acc": float(summary["mirror_final_acc"]),
        "tfmr_final_acc": float(summary["tfmr_final_acc"]),
    }


def _cli_explicit_option(opt: str) -> bool:
    """True if argv contains --opt or --opt= (so preset does not clobber user intent)."""
    prefix = f"{opt}="
    return any(a == opt or a.startswith(prefix) for a in sys.argv)


def apply_script10_train_recipe(args: argparse.Namespace) -> None:
    """Optional middle-ground training for matched comparisons: Adam (no AdamW), no heavy LR schedule."""
    recipe = str(getattr(args, "train_recipe", "default"))
    if recipe == "default":
        return
    if recipe != "moderate":
        raise ValueError(f"unknown train_recipe: {recipe!r}")

    if not _cli_explicit_option("--transformer-baseline"):
        args.transformer_baseline = "matched"
    if not _cli_explicit_option("--optimizer"):
        args.optimizer = "adam"
    if not _cli_explicit_option("--weight-decay"):
        args.weight_decay = 0.0
    if not _cli_explicit_option("--lr-schedule"):
        args.lr_schedule = "none"
    if not _cli_explicit_option("--warmup-steps"):
        args.warmup_steps = 0
    if not _cli_explicit_option("--lr-min-ratio"):
        args.lr_min_ratio = 0.5
    # Only nudge lr if still at parser default (presets may set e.g. 3.5e-4 for niah_prefix).
    if not _cli_explicit_option("--lr") and float(args.lr) == 1e-3:
        args.lr = 5e-4

    print(
        "[script10] train_recipe=moderate: matched baseline (unless you set --transformer-baseline), "
        "Adam (not AdamW; no decoupled WD), constant lr (no warmup/cosine), "
        "default lr=5e-4 when --lr is still 1e-3. Omits AdamW as the usual strongest Acc driver in this stack.",
        flush=True,
    )


def apply_script10_preset(args: argparse.Namespace) -> None:
    """Presets: gap/dup tuned for ~30%+ val best_acc in a few hundred steps; delimiter kept softer but finicky."""
    user_steps = _cli_explicit_option("--steps")
    user_log = _cli_explicit_option("--log-every")
    user_lr = _cli_explicit_option("--lr")
    user_lmr = _cli_explicit_option("--lr-min-ratio")
    if args.preset in ("script10-easy-copy", "script10-delimiter-copy"):
        args.task = "delimiter"
        args.n_min = 6
        args.n_max = 8
        args.val_n = 7
        args.vocab_content = 14
        args.val_batch = 256
        args.batch = 48
        if not user_steps:
            args.steps = 3000
        if not user_log:
            args.log_every = 100
        args.dim = max(args.dim, 128)
        args.depth = max(args.depth, 3)
        pname = "script10-delimiter-copy"
        if args.preset == "script10-easy-copy":
            print(
                "note: --preset script10-easy-copy is a legacy alias for script10-delimiter-copy "
                "(delimiter cross-copy, NOT identity).",
                flush=True,
            )
            pname = "script10-easy-copy (alias → delimiter)"
        print(
            f"preset {pname}: delimiter n~U[6,8] val n=7 vocab=14 len≤17; "
            f"steps={args.steps} (default 3000 unless --steps). Prefer gap/dup presets for stable curves.",
            flush=True,
        )
    elif args.preset == "script10-gap-mid":
        args.task = "gap_copy"
        args.n_min = args.n_max = 3
        args.val_n = 3
        args.gap_min = 2
        args.gap_max = 5
        args.val_gap = 3
        args.vocab_content = 10
        args.val_batch = 256
        args.batch = 48
        if not user_steps:
            args.steps = 3500
        if not user_log:
            args.log_every = 50
        args.dim = max(args.dim, 128)
        args.depth = max(args.depth, 3)
        print(
            "preset script10-gap-mid: gap_copy n=3 gap~U[2,5] val gap=3 vocab=10 "
            f"max_len≤12; CE on tail (3 pos); steps={args.steps} (default 3500 unless --steps).",
            flush=True,
        )
    elif args.preset == "script10-dup-small":
        args.task = "dup_tail_copy"
        args.n_min = args.n_max = 3
        args.val_n = 3
        args.vocab_content = 10
        args.val_batch = 256
        args.batch = 48
        if not user_steps:
            args.steps = 3500
        if not user_log:
            args.log_every = 50
        args.dim = max(args.dim, 128)
        args.depth = max(args.depth, 3)
        print(
            "preset script10-dup-small: dup_tail_copy n=3 vocab=10 seq_len=10 "
            f"CE on tail_B (3 tok); steps={args.steps} (default 3500 unless --steps).",
            flush=True,
        )
    elif args.preset == "script10-niah-fixed-long":
        args.task = "niah"
        args.niah_hay_len = 12
        args.niah_needle_len = 3
        args.vocab_content = 12
        args.val_batch = 256
        args.batch = 48
        if not user_steps:
            args.steps = 1000
        if not user_log:
            args.log_every = 50
        args.dim = max(args.dim, 128)
        args.depth = max(args.depth, 3)
        core = 2 * int(args.niah_hay_len) + 2 * int(args.niah_needle_len) + 1
        print(
            "preset script10-niah-fixed-long: niah with longer hay (more middle interference) "
            f"hay_each={args.niah_hay_len} needle_len={args.niah_needle_len} vocab={args.vocab_content} "
            f"max_len={core}; CE on last{args.niah_needle_len}. "
            f"steps={args.steps} (default 1000 unless --steps).",
            flush=True,
        )
    elif args.preset == "script10-niah-fixed":
        args.task = "niah"
        args.niah_hay_len = 6
        args.niah_needle_len = 3
        args.vocab_content = 12
        args.val_batch = 256
        args.batch = 48
        if not user_steps:
            args.steps = 1000
        if not user_log:
            args.log_every = 50
        args.dim = max(args.dim, 128)
        args.depth = max(args.depth, 3)
        core = 2 * int(args.niah_hay_len) + 2 * int(args.niah_needle_len) + 1
        print(
            "preset script10-niah-fixed: niah with shorter hay_each (less middle interference) "
            f"hay_each={args.niah_hay_len} needle_len={args.niah_needle_len} vocab={args.vocab_content} "
            f"max_len={core}; CE on last{args.niah_needle_len} = needle (encoder / bidirectional). "
            f"Use script10-niah-fixed-long for hay_each=12. steps={args.steps} (default 1000 unless --steps).",
            flush=True,
        )
    elif args.preset == "script10-niah-prefix-easy":
        args.task = "niah_prefix"
        args.niah_prefix_needle_len = 3
        args.niah_prefix_len_min = 25
        args.niah_prefix_len_max = 40
        args.niah_prefix_val_len = 32
        args.niah_prefix_val_ood = False
        args.vocab_content = 12
        args.val_batch = 256
        args.batch = 48
        if not user_steps:
            args.steps = 2000
        if not user_log:
            args.log_every = 50
        args.dim = max(args.dim, 128)
        args.depth = max(args.depth, 3)
        print(
            "preset script10-niah-prefix-easy: short seq U[25,40] val=32 K=3 vocab=12 (often reaches val acc 1). "
            f"max_len={args.niah_prefix_len_max}. steps={args.steps}.",
            flush=True,
        )
    elif args.preset == "script10-niah-prefix-hard":
        args.task = "niah_prefix"
        args.niah_prefix_needle_len = 6
        args.niah_prefix_len_min = 72
        args.niah_prefix_len_max = 112
        args.niah_prefix_val_len = 176
        args.niah_prefix_val_ood = True
        args.vocab_content = 64
        args.val_batch = 256
        args.batch = 48
        if not user_steps:
            args.steps = 5000
        if not user_log:
            args.log_every = 100
        args.dim = max(args.dim, 128)
        args.depth = max(args.depth, 3)
        _mx = max(int(args.niah_prefix_len_max), int(args.niah_prefix_val_len))
        print(
            "preset script10-niah-prefix-hard: K=6 vocab=64 train U[72,112] val_OOD=176; "
            f"max_len={_mx}. steps={args.steps} (default 5000).",
            flush=True,
        )
    elif args.preset == "script10-niah-prefix":
        args.task = "niah_prefix"
        # In-distribution val; lower default lr slows convergence so tfmr often ends ~0.75–0.9 at 3k steps (not 1.0). Raise --lr toward 1e-3 if too slow.
        args.niah_prefix_needle_len = 6
        args.niah_prefix_len_min = 60
        args.niah_prefix_len_max = 88
        args.niah_prefix_val_len = 76
        args.niah_prefix_val_ood = False
        args.vocab_content = 54
        args.val_batch = 256
        args.batch = 48
        if not user_lr:
            args.lr = 3.5e-4
        if not user_steps:
            args.steps = 3000
        if not user_log:
            args.log_every = 50
        args.dim = max(args.dim, 128)
        args.depth = max(args.depth, 3)
        print(
            "preset script10-niah-prefix: K=6 vocab=54 train U[60,88] val=76 in-dist; "
            f"max_len={args.niah_prefix_len_max}. "
            f"default lr={args.lr} (override with --lr). steps={args.steps} (default 3000 unless --steps).",
            flush=True,
        )
    elif args.preset == "script10-kv-assoc":
        args.task = "kv_assoc"
        args.kv_num_pairs = 5
        args.kv_pair_gap_min = 0
        args.kv_pair_gap_max = 2
        args.kv_dist_min = 10
        args.kv_dist_max = 32
        args.kv_val_dist_len = 20
        args.kv_val_query_idx = 2
        args.vocab_content = 32
        args.val_batch = 256
        args.batch = 48
        args.n_min = 6
        args.n_max = 12
        args.val_n = 8
        if not user_steps:
            args.steps = 4000
        if not user_log:
            args.log_every = 100
        args.dim = max(args.dim, 128)
        args.depth = max(args.depth, 3)
        _Pk = int(args.kv_num_pairs)
        _kl = (
            3 * _Pk
            + max(0, _Pk - 1) * int(args.kv_pair_gap_max)
            + 1
            + int(args.kv_dist_max)
            + 2
        )
        print(
            "preset script10-kv-assoc: associative KV — P=5, shuffled pair order + pair_gap U[0,2], "
            f"train distractor U[{args.kv_dist_min},{args.kv_dist_max}], val distractor={args.kv_val_dist_len}, "
            f"val_qidx={args.kv_val_query_idx}; max_len={_kl}. steps={args.steps} (default 4000).",
            flush=True,
        )
    elif args.preset == "script10-kv-assoc-easy":
        args.task = "kv_assoc"
        args.kv_num_pairs = 3
        args.kv_pair_gap_min = 0
        args.kv_pair_gap_max = 1
        args.kv_dist_min = 4
        args.kv_dist_max = 12
        args.kv_val_dist_len = 8
        args.kv_val_query_idx = 1
        args.vocab_content = 16
        args.val_batch = 256
        args.batch = 48
        args.n_min = 6
        args.n_max = 12
        args.val_n = 8
        if not user_steps:
            args.steps = 1000
        if not user_log:
            args.log_every = 100
        if not user_lmr:
            args.lr_min_ratio = 0.35
        args.dim = max(args.dim, 128)
        args.depth = max(args.depth, 3)
        _Pk = int(args.kv_num_pairs)
        _kl = (
            3 * _Pk
            + max(0, _Pk - 1) * int(args.kv_pair_gap_max)
            + 1
            + int(args.kv_dist_max)
            + 2
        )
        print(
            "preset script10-kv-assoc-easy: easier KV — P=3, pair_gap U[0,1], "
            f"train distractor U[{args.kv_dist_min},{args.kv_dist_max}], val distractor={args.kv_val_dist_len}, "
            f"val_qidx={args.kv_val_query_idx}; max_len={_kl}. steps={args.steps} (default 1000 unless --steps). "
            f"Default lr_min_ratio={args.lr_min_ratio} (cosine tail; override --lr-min-ratio). "
            "Use script10-kv-assoc for the harder setting.",
            flush=True,
        )
    elif args.preset == "script10-story-assoc-easy":
        args.task = "story_assoc"
        args.story_num_facts = 3
        args.story_open_min = 4
        args.story_open_max = 10
        args.story_val_open_len = 7
        args.story_pair_gap_min = 0
        args.story_pair_gap_max = 1
        args.story_dist_min = 4
        args.story_dist_max = 12
        args.story_val_dist_len = 8
        args.story_val_query_idx = 1
        args.vocab_content = 16
        args.val_batch = 256
        args.batch = 48
        args.n_min = 6
        args.n_max = 12
        args.val_n = 8
        if not user_steps:
            args.steps = 1200
        if not user_log:
            args.log_every = 100
        if not user_lmr:
            args.lr_min_ratio = 0.35
        args.dim = max(args.dim, 128)
        args.depth = max(args.depth, 3)
        _Ps = int(args.story_num_facts)
        _sl = (
            int(args.story_open_max)
            + 3 * _Ps
            + max(0, _Ps - 1) * int(args.story_pair_gap_max)
            + 1
            + int(args.story_dist_max)
            + 2
        )
        print(
            "preset script10-story-assoc-easy: opening U[4,10] then P=3 shuffled facts, pair_gap U[0,1], "
            f"distr U[{args.story_dist_min},{args.story_dist_max}], val_open={args.story_val_open_len} "
            f"val_d={args.story_val_dist_len} val_qidx={args.story_val_query_idx}; max_len={_sl}. "
            f"steps={args.steps} (default 1200 unless --steps).",
            flush=True,
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Script10: TinyMirror per token vs Transformer")
    p.add_argument(
        "--aggregate-plots-only",
        action="store_true",
        help="Read seed_*/training_*.csv and write figure_script10_loss_acc_multi_seed.{png,pdf}.",
    )
    p.add_argument(
        "--task",
        type=str,
        choices=(
            "delimiter",
            "identity",
            "gap_copy",
            "dup_tail_copy",
            "niah",
            "niah_prefix",
            "kv_assoc",
            "story_assoc",
        ),
        default="identity",
        help="Synthetic tasks; kv_assoc / story_assoc = shuffled key→value facts + distractor + query (see --kv-*, --story-*).",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="",
        choices=(
            "",
            "script10-easy-copy",
            "script10-delimiter-copy",
            "script10-gap-mid",
            "script10-dup-small",
            "script10-niah-fixed",
            "script10-niah-fixed-long",
            "script10-kv-assoc",
            "script10-kv-assoc-easy",
            "script10-story-assoc-easy",
            "script10-niah-prefix",
            "script10-niah-prefix-hard",
            "script10-niah-prefix-easy",
        ),
        help="script10-delimiter-copy: cross-span delimiter copy (recommended). easy-copy: same, legacy name.",
    )
    p.add_argument(
        "--niah-hay-len",
        type=int,
        default=12,
        help="niah: random hay tokens on each side of the needle (fixed layout).",
    )
    p.add_argument(
        "--niah-needle-len",
        type=int,
        default=3,
        help="niah: needle length; CE on the last this many positions.",
    )
    p.add_argument(
        "--niah-prefix-needle-len",
        type=int,
        default=6,
        help="niah_prefix: needle tokens at positions 0..K-1; CE on last K.",
    )
    p.add_argument(
        "--niah-prefix-len-min",
        type=int,
        default=60,
        help="niah_prefix: min core seq length (needle + filler + junk, no pad).",
    )
    p.add_argument(
        "--niah-prefix-len-max",
        type=int,
        default=88,
        help="niah_prefix: max train core length; padding max_len = max(this, val_len).",
    )
    p.add_argument(
        "--niah-prefix-val-len",
        type=int,
        default=76,
        help="niah_prefix: fixed val core length; in [len_min,len_max] unless --niah-prefix-val-ood.",
    )
    p.add_argument(
        "--niah-prefix-val-ood",
        action="store_true",
        help="niah_prefix: val_len > len_max (length OOD); training stays U[len_min,len_max].",
    )
    p.add_argument(
        "--kv-num-pairs",
        type=int,
        default=5,
        help="kv_assoc: number of (key,value) pairs; keys from [0,floor(vocab/2)), values from rest.",
    )
    p.add_argument(
        "--kv-dist-min",
        type=int,
        default=8,
        help="kv_assoc: min distractor tokens between study block and query (train U[min,max]).",
    )
    p.add_argument(
        "--kv-dist-max",
        type=int,
        default=24,
        help="kv_assoc: max distractor length (sets max_len with P).",
    )
    p.add_argument(
        "--kv-val-dist-len",
        type=int,
        default=16,
        help="kv_assoc: distractor length for fixed validation cache (in [dist-min,dist-max]).",
    )
    p.add_argument(
        "--kv-val-query-idx",
        type=int,
        default=0,
        help="kv_assoc: which pair index (0..P-1) is queried on validation.",
    )
    p.add_argument(
        "--kv-pair-gap-min",
        type=int,
        default=0,
        help="kv_assoc: min random content tokens between consecutive [key,ARROW,value] triples (train/val).",
    )
    p.add_argument(
        "--kv-pair-gap-max",
        type=int,
        default=0,
        help="kv_assoc: max gap between triples; breaks fixed slot positions vs PE.",
    )
    p.add_argument(
        "--story-num-facts",
        type=int,
        default=5,
        help="story_assoc: number of (entity,detail) pairs (same pools as kv_assoc).",
    )
    p.add_argument(
        "--story-open-min",
        type=int,
        default=4,
        help="story_assoc: min opening story tokens before facts (train U[min,max]).",
    )
    p.add_argument(
        "--story-open-max",
        type=int,
        default=12,
        help="story_assoc: max opening length.",
    )
    p.add_argument(
        "--story-pair-gap-min",
        type=int,
        default=0,
        help="story_assoc: min gap tokens between consecutive triples in the fact block.",
    )
    p.add_argument(
        "--story-pair-gap-max",
        type=int,
        default=0,
        help="story_assoc: max gap between triples.",
    )
    p.add_argument(
        "--story-dist-min",
        type=int,
        default=8,
        help="story_assoc: min distractor tokens between facts+DELIM and query.",
    )
    p.add_argument(
        "--story-dist-max",
        type=int,
        default=24,
        help="story_assoc: max distractor length.",
    )
    p.add_argument(
        "--story-val-open-len",
        type=int,
        default=8,
        help="story_assoc: fixed opening length for cached val (in [open-min,open-max]).",
    )
    p.add_argument(
        "--story-val-dist-len",
        type=int,
        default=16,
        help="story_assoc: fixed distractor length for val (in [dist-min,dist-max]).",
    )
    p.add_argument(
        "--story-val-query-idx",
        type=int,
        default=0,
        help="story_assoc: which fact index (0..P-1) is queried on validation.",
    )
    p.add_argument("--gap-min", type=int, default=0)
    p.add_argument("--gap-max", type=int, default=0)
    p.add_argument("--val-gap", type=int, default=0)
    p.add_argument(
        "--ffn-activation",
        type=str,
        choices=("relu", "gelu", "silu"),
        default="relu",
        help="Encoder FFN activation (Pre-LN branch in token_mirror blocks; transformer baseline uses the same). "
        "TinyMirror bottleneck uses --mirror-activation.",
    )
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help="Train only the Transformer baseline (skip token_mirror). Incompatible with --calibrate-steps > 0.",
    )
    p.add_argument(
        "--mirror-full-width",
        action="store_true",
        help="TinyMirror inner width r = model dim (same dim→r→dim + --mirror-activation, no width reduction). "
        "Default: r = --mirror-dim (historical experiments).",
    )
    p.add_argument(
        "--mirror-linear",
        action="store_true",
        help="Deprecated. Same as --mirror-full-width (SiLU is not removed).",
    )
    p.add_argument(
        "--mirror-dim",
        type=int,
        default=8,
        help="TinyMirror inner width r when --mirror-full-width is off (Linear(dim,r)→act→Linear(r,dim)).",
    )
    p.add_argument(
        "--mirror-no-silu",
        action="store_true",
        help="TinyMirror: diagonal affine (scale*x+bias per dim), no bottleneck on mirror. FFN unchanged.",
    )
    p.add_argument(
        "--mirror-activation",
        type=str,
        choices=MIRROR_BOTTLENECK_ACTIVATIONS,
        default="silu",
        help="TinyMirror bottleneck nonlinearity after Linear(dim→r). Ignored with --mirror-no-silu. "
        "light_depth_t2.0 = LightDepthAct(2); phys_tau_softplus_z_b*.0 = ScatteringOpticalDepthAct(β, softplus_z).",
    )
    p.add_argument(
        "--mirror-scale",
        type=float,
        default=1.0,
        help="Multiply TinyMirror residual before adding (default 1).",
    )
    p.add_argument(
        "--mirror-attn-mem",
        action="store_true",
        help="Per block: add lightweight self-attention memory on the norm2 residual (low-rank Q/K, full V), "
        "alongside TinyMirror. V is zero-init so the branch starts at zero.",
    )
    p.add_argument(
        "--mirror-attn-mem-head-dim",
        type=int,
        default=16,
        help="Q/K head dimension for --mirror-attn-mem (V maps dim→dim). Try 16 or 24.",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45",
        help="Comma-separated training seeds; multi_seed_*.csv. Use '' for single --seed.",
    )
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--vocab-content", type=int, default=16)
    p.add_argument("--n-min", type=int, default=8)
    p.add_argument("--n-max", type=int, default=12)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--val-batch", type=int, default=256)
    p.add_argument("--val-n", type=int, default=10)
    p.add_argument("--val-random-n", action="store_true")
    p.add_argument("--val-set-file", type=str, default="fixed_val_set.pt")
    p.add_argument("--val-set-seed", type=int, default=999_001)
    p.add_argument("--regen-val-set", action="store_true")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--optimizer",
        type=str,
        choices=("adamw", "adam"),
        default="adamw",
        help="Optimizer for token_mirror (and for Transformer when --transformer-baseline matched).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW decoupled weight decay (default 0.01; set 0 to disable; ignored for --optimizer adam).",
    )
    p.add_argument(
        "--lr-schedule",
        type=str,
        choices=("cosine", "none"),
        default="cosine",
        help="LR schedule for token_mirror (and for Transformer when --transformer-baseline matched).",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear LR warmup steps (default 0=off; only used with --lr-schedule cosine).",
    )
    p.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.1,
        help="After cosine anneal, LR floor is lr * this (only for --lr-schedule cosine).",
    )
    p.add_argument(
        "--pos-enc",
        type=str,
        choices=("learned", "none", "sinusoidal"),
        default="learned",
        help="Position encoding: learned Embedding, none, or fixed sinusoidal (Transformer). "
        "Matched transformer follows this; legacy baseline still forces none.",
    )
    p.add_argument(
        "--transformer-baseline",
        type=str,
        choices=("matched", "legacy"),
        default="legacy",
        help="legacy: weaker Transformer (Adam, fixed lr, no PE) so easy tasks stay informative; matched: same optimizer/schedule/PE as token_mirror.",
    )
    p.add_argument(
        "--train-recipe",
        type=str,
        choices=("default", "moderate"),
        default="default",
        help="moderate: matched baseline unless set; Adam (not AdamW), weight_decay=0, constant lr; "
        "lr=5e-4 when --lr still default 1e-3. Inherits --pos-enc (default learned).",
    )
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--acc-hit-thresholds",
        type=str,
        default="0.25,0.50,0.75",
        help="Comma-separated val_acc targets for first-step tracking.",
    )
    p.add_argument(
        "--calibrate-steps",
        type=int,
        default=0,
        help="If >0: train token_mirror only this many steps first; require best/ rising or CalibrationFailed.",
    )
    p.add_argument(
        "--calibrate-min-best-acc",
        type=float,
        default=0.28,
        help="Calibration: min val_acc_best after --calibrate-steps (0.28 fits gap/dup presets; use 0.30 if you eased task further).",
    )
    p.add_argument(
        "--calibrate-rise-tol",
        type=float,
        default=0.02,
        help="Calibration: pass if last val_acc >= previous logged val_acc - tol.",
    )
    p.add_argument(
        "--auto-ease-on-fail",
        action="store_true",
        help="On CalibrationFailed, relax gap_copy/dup_tail (not delimiter) up to 2 times and regen val.",
    )
    return p


def _parse_training_seeds(args: argparse.Namespace) -> list[int]:
    s = args.seeds.strip()
    if s:
        return [int(x) for x in s.split(",") if x.strip()]
    return [args.seed]


def _run_one_seed_with_optional_retry(
    args: argparse.Namespace, subdir: str | None = None
) -> dict[str, float]:
    rounds = 3 if args.auto_ease_on_fail else 1
    for attempt in range(rounds):
        try:
            return run_one_seed(args, subdir=subdir)
        except CalibrationFailed as e:
            print(f"[script10] {e}", flush=True)
            if (
                args.auto_ease_on_fail
                and attempt + 1 < rounds
                and args.task in ("gap_copy", "dup_tail_copy")
            ):
                _ease_task_args(args)
                args.regen_val_set = True
                continue
            raise SystemExit(3) from e
    raise RuntimeError("unreachable")


def resolve_mirror_deprecated_cli(args: argparse.Namespace) -> None:
    if getattr(args, "mirror_linear", False):
        print(
            "[script10] note: --mirror-linear is deprecated → use --mirror-full-width "
            "(inner r=dim; SiLU unchanged).",
            flush=True,
        )
        args.mirror_full_width = True


def main() -> None:
    args = build_argparser().parse_args()
    apply_script10_preset(args)
    apply_script10_train_recipe(args)
    resolve_mirror_deprecated_cli(args)
    if args.aggregate_plots_only:
        root = Path(args.output_dir)
        note = ""
        agg_path = root / "multi_seed_aggregate.csv"
        if agg_path.is_file():
            with agg_path.open(newline="", encoding="utf-8") as f:
                agg_rows = list(csv.DictReader(f))
            if agg_rows:
                r0 = agg_rows[0]
                note = (
                    f"task={r0.get('task', '')}, steps={r0.get('steps', '')}, "
                    f"n in [{r0.get('n_min', '')},{r0.get('n_max', '')}]"
                )
        plot_multi_seed_training_curves(root, title_note=note)
        print(f"Wrote figure_script10_loss_acc_multi_seed.png and .pdf under {root}")
        return

    seeds = _parse_training_seeds(args)
    root_out = Path(args.output_dir)
    if len(seeds) > 1:
        multi: list[dict[str, Any]] = []
        for i, sd in enumerate(seeds):
            args.seed = sd
            if i > 0:
                args.regen_val_set = False
            fin = _run_one_seed_with_optional_retry(args, subdir=f"seed_{sd}")
            multi.append(
                {
                    "seed": sd,
                    "mirror_final_val_ce": fin["mirror_final_val"],
                    "mirror_final_val_acc": fin["mirror_final_acc"],
                    "tfmr_final_val_ce": fin["tfmr_final_val"],
                    "tfmr_final_val_acc": fin["tfmr_final_acc"],
                }
            )
        save_csv(root_out / "multi_seed_summary.csv", multi)

        def _std(xs: list[float]) -> float:
            return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0

        m_ce = [float(r["mirror_final_val_ce"]) for r in multi]
        m_acc = [float(r["mirror_final_val_acc"]) for r in multi]
        t_ce = [float(r["tfmr_final_val_ce"]) for r in multi]
        t_acc = [float(r["tfmr_final_val_acc"]) for r in multi]
        agg_row: dict[str, Any] = {
            "task": args.task,
            "n_seeds": len(seeds),
            "steps": args.steps,
            "n_min": args.n_min,
            "n_max": args.n_max,
            "val_n": "random" if args.val_random_n else args.val_n,
            "mirror_full_width": bool(args.mirror_full_width),
            "mirror_effective_dim": effective_mirror_dim(int(args.dim), args),
            "mirror_dim": args.mirror_dim,
            "mirror_no_silu": bool(getattr(args, "mirror_no_silu", False)),
            "mirror_activation": args.mirror_activation,
            "ffn_activation": args.ffn_activation,
            "mirror_attn_mem": bool(getattr(args, "mirror_attn_mem", False)),
            "mirror_attn_mem_head_dim": int(
                getattr(args, "mirror_attn_mem_head_dim", 16)
            ),
            "mirror_val_ce_mean": float(np.mean(m_ce)),
            "mirror_val_ce_std": _std(m_ce),
            "mirror_val_acc_mean": float(np.mean(m_acc)),
            "mirror_val_acc_std": _std(m_acc),
            "tfmr_val_ce_mean": float(np.mean(t_ce)),
            "tfmr_val_ce_std": _std(t_ce),
            "tfmr_val_acc_mean": float(np.mean(t_acc)),
            "tfmr_val_acc_std": _std(t_acc),
        }
        save_csv(root_out / "multi_seed_aggregate.csv", [agg_row])
        print(
            f"Wrote multi_seed_summary.csv and multi_seed_aggregate.csv under {root_out}",
            flush=True,
        )
        plot_multi_seed_training_curves(
            root_out,
            title_note=(
                f"task={args.task}, steps={args.steps}, "
                f"n in [{args.n_min},{args.n_max}]"
            ),
        )
        print(f"Wrote figure_script10_loss_acc_multi_seed.png and .pdf under {root_out}")
    else:
        _run_one_seed_with_optional_retry(args)


if __name__ == "__main__":
    main()
