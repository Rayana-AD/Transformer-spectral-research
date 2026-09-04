# Script 7：在预训练 ViT 上重做 script6 的 Exp1–3（输入幅值扫描、物理/基线激活对比、分类深度–精度–N/c 探针）
# 依赖：transformers、torch、numpy、matplotlib；CIFAR 需 torchvision。
# 用法：python script7_vit_criticality.py exp1|exp2|exp3|exp4 --help
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from activation_extras import LightDepthAct, ScatteringOpticalDepthAct
from paper_style import apply_paper_style

try:
    from transformers import ViTModel
    from transformers.utils import logging as tf_logging

    tf_logging.set_verbosity_error()
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"需要 transformers: pip install transformers\n{e}") from e

from script5_vit_representation_svd import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    discover_cached_vit_models,
    hidden_states_to_metrics,
    resolve_dtype,
    truncate_encoder,
)

BASE_SEED = 42
OUTPUT_ROOT = Path("results_script7")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _load_per_layer_prefill_csv(path: Path) -> list[dict]:
    """读入已跑过的 per-layer 摘要行（与 exp3_vit_per_layer_summary 同列）。"""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict] = []
    for r in rows:
        def ffloat(k: str) -> float:
            v = r.get(k, "")
            if v is None or v == "":
                return float("nan")
            return float(v)

        def fint(k: str) -> int:
            return int(r[k])

        ct_raw = r.get("c_t_to_next", "")
        c_tn = None if ct_raw in ("", "None", "nan") else float(ct_raw)
        out.append(
            {
                "layer_index": fint("layer_index"),
                "encoder_depth": fint("encoder_depth"),
                "best_test_acc": ffloat("best_test_acc"),
                "forward_only": fint("forward_only"),
                "per_layer_probe": fint("per_layer_probe"),
                "cls_activation": r["cls_activation"],
                "freeze_backbone": fint("freeze_backbone"),
                "c_bar_geom_probe": ffloat("c_bar_geom_probe"),
                "mean_abs_c_minus_1": ffloat("mean_abs_c_minus_1"),
                "nuclear_norm_layer": ffloat("nuclear_norm_layer"),
                "c_t_to_next": c_tn,
                "representation": r.get("representation", "per_layer_linear_probe_cls_frozen_full_vit"),
            }
        )
    return out


def append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def grad_norm_total(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.data.pow(2).sum().item())
    return math.sqrt(total)


def vit_grad_probe_sum_last_hidden(
    vit: ViTModel, pixel_values: torch.Tensor, grad_explode_thresh: float
) -> tuple[float, list[float], bool]:
    """Exp1 风格：last_hidden_state.sum() 反传；返回 ViT 总 grad 范数、各 encoder.layer 子范数、是否通过有限性/阈值。"""
    vit.train()
    for p in vit.parameters():
        p.requires_grad_(True)
        p.grad = None
    vit_out = vit(pixel_values, output_hidden_states=False)
    loss = vit_out.last_hidden_state.sum()
    loss.backward()
    ok = True
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in vit.parameters()):
        ok = False
    gtot = grad_norm_total(vit)
    if not math.isfinite(gtot):
        ok = False
    if gtot > grad_explode_thresh:
        ok = False
    blocks = [grad_norm_total(vit.encoder.layer[i]) for i in range(len(vit.encoder.layer))]
    vit.zero_grad(set_to_none=True)
    return gtot, blocks, ok


def nuclear_c_bar_geom(nuc_chain: list[float]) -> float:
    """c_bar_geom = (N_L/N_0)^(1/L)，L 为相邻 hidden 状态过渡数。"""
    if len(nuc_chain) < 2 or nuc_chain[0] <= 1e-30:
        return float("nan")
    depth = len(nuc_chain) - 1
    ratio = nuc_chain[-1] / nuc_chain[0]
    if ratio <= 0 or not math.isfinite(ratio):
        return float("nan")
    return float(ratio ** (1.0 / depth))


def apply_vit_mlp_activation(model: ViTModel, act_id: str) -> None:
    """gelu：不改动。其余：替换 encoder 各层 MLP 激活（共享同一模块实例，无参数）。"""
    if act_id == "gelu":
        return
    if act_id == "relu":
        act: nn.Module = nn.ReLU()
    elif act_id == "light_depth_t2.0":
        act = LightDepthAct(2.0)
    elif act_id == "phys_tau_softplus_z_b1.0":
        act = ScatteringOpticalDepthAct(1.0, "softplus_z")
    else:
        raise ValueError(f"Unknown act_id: {act_id}")
    for layer in model.encoder.layer:
        layer.intermediate.intermediate_act_fn = act


def load_vit(
    *,
    model_id: str,
    snapshot_path: str | None,
    local_only: bool,
    dtype: torch.dtype,
) -> ViTModel:
    load_path = snapshot_path if snapshot_path else model_id
    try:
        return ViTModel.from_pretrained(load_path, local_files_only=local_only, torch_dtype=dtype)
    except Exception as e:
        print(f"加载 ViT 失败: {e}", file=sys.stderr)
        cached = discover_cached_vit_models()
        if cached:
            print("本地缓存示例:", file=sys.stderr)
            for m in cached[:6]:
                print(f"  {m}", file=sys.stderr)
        raise SystemExit(1) from e


def make_pixel_batch(
    *,
    batch: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    input_kind: str,
) -> torch.Tensor:
    set_seed(seed)
    if input_kind == "randn":
        return torch.randn(batch, 3, image_size, image_size, device=device, dtype=dtype)
    if input_kind == "imagenet_norm_randn":
        x = torch.randn(batch, 3, image_size, image_size, device=device, dtype=dtype)
        mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
        return (x - mean) / std
    raise ValueError(input_kind)


class ViTClassifier(nn.Module):
    """ViT backbone + Linear 分类头（CLS）。支持 output_hidden_states 探针。"""

    def __init__(self, vit: ViTModel, num_classes: int) -> None:
        super().__init__()
        self.vit = vit
        self.classifier = nn.Linear(vit.config.hidden_size, num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        out = self.vit(pixel_values, output_hidden_states=output_hidden_states)
        h = out.last_hidden_state[:, 0, :]
        logits = self.classifier(h)
        if output_hidden_states:
            assert out.hidden_states is not None
            return logits, out.hidden_states
        return logits


def run_exp1(args: argparse.Namespace) -> None:
    """输入幅值 α扫描 + 前向 N/c + 全模型反传 grad 探针。"""
    out = OUTPUT_ROOT / "exp1_vit_alpha"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)  # type: ignore[arg-type]
    load_path: str = args.snapshot_path if args.snapshot_path else args.model_id
    local_only = args.local_files_only and not args.allow_download

    model = load_vit(
        model_id=args.model_id,
        snapshot_path=args.snapshot_path,
        local_only=local_only,
        dtype=dtype,
    )
    model = model.to(device=device, dtype=dtype)
    image_size = model.config.image_size
    raw_rows: list[dict] = []

    alphas = np.logspace(math.log10(args.alpha_min), math.log10(args.alpha_max), args.alpha_steps)
    layer_raw_rows: list[dict] = []

    for alpha in alphas:
        for si in range(args.num_seeds):
            seed = BASE_SEED + 1_000 + si
            base = make_pixel_batch(
                batch=args.batch,
                image_size=image_size,
                device=device,
                dtype=dtype,
                seed=seed,
                input_kind=args.input_kind,
            )
            pixel_values = (alpha * base).to(dtype=dtype)

            model.eval()
            fwd_ok = True
            try:
                with torch.no_grad():
                    layer_rows, c_ts = _forward_metrics_from_model(model, pixel_values)
                nuc_chain = [r["nuclear_norm"] for r in layer_rows]
                fwd_ok = all(math.isfinite(n) for n in nuc_chain)
            except Exception:
                fwd_ok = False
                layer_rows, c_ts = [], []
                nuc_chain = []
            c_bar = nuclear_c_bar_geom(nuc_chain) if nuc_chain else float("nan")

            model.train()
            for p in model.parameters():
                p.requires_grad_(True)
                p.grad = None
            pv = (
                alpha
                * make_pixel_batch(
                    batch=args.batch,
                    image_size=image_size,
                    device=device,
                    dtype=dtype,
                    seed=seed + 17,
                    input_kind=args.input_kind,
                )
            ).to(dtype=dtype)
            grad_ok = True
            gnorm = float("nan")
            try:
                vit_out = model(pv, output_hidden_states=False)
                loss = vit_out.last_hidden_state.sum()
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    grad_ok = False
                gnorm = grad_norm_total(model)
                if not math.isfinite(gnorm):
                    grad_ok = False
                if gnorm > args.grad_explode_thresh:
                    grad_ok = False
            except RuntimeError:
                grad_ok = False
            model.zero_grad(set_to_none=True)

            c_finite = [c for c in c_ts if c is not None and math.isfinite(c)]
            c_mean_arith = float(np.mean(c_finite)) if c_finite else float("nan")

            for j, cv in enumerate(c_ts):
                if cv is None or not math.isfinite(cv):
                    continue
                layer_raw_rows.append(
                    {
                        "model": str(load_path),
                        "alpha": float(alpha),
                        "input_kind": args.input_kind,
                        "seed": seed,
                        "layer_transition": j + 1,
                        "c_t": float(cv),
                        "grad_norm": gnorm,
                        "forward_finite": int(fwd_ok),
                        "grad_probe_ok": int(grad_ok),
                    }
                )

            raw_rows.append(
                {
                    "model": str(load_path),
                    "alpha": float(alpha),
                    "input_kind": args.input_kind,
                    "seed": seed,
                    "c_bar_geom": c_bar,
                    "c_mean_arith": c_mean_arith,
                    "nuclear_final": nuc_chain[-1] if nuc_chain else float("nan"),
                    "forward_finite": int(fwd_ok),
                    "grad_probe_ok": int(grad_ok),
                    "grad_norm": gnorm,
                }
            )

    save_csv(out / "exp1_vit_alpha_raw.csv", raw_rows)
    save_csv(out / "exp1_vit_alpha_layer_raw.csv", layer_raw_rows)

    cfg = {
        "command": "exp1",
        "model_id": args.model_id,
        "snapshot_path": args.snapshot_path,
        "local_files_only": bool(args.local_files_only),
        "allow_download": bool(args.allow_download),
        "device": args.device,
        "dtype": args.dtype,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_steps": args.alpha_steps,
        "num_seeds": args.num_seeds,
        "batch": args.batch,
        "input_kind": args.input_kind,
        "grad_explode_thresh": args.grad_explode_thresh,
        "note": "Script4 (results_script4) training CSV has no per-layer c_l; c_l vs alpha/grad figures use this Exp1 export.",
    }
    (out / "exp1_vit_run_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    summary: list[dict] = []
    for alpha in alphas:
        rows = [r for r in raw_rows if abs(r["alpha"] - alpha) < 1e-12]
        c_vals = np.array([r["c_bar_geom"] for r in rows], dtype=float)
        g_vals = np.array([r["grad_norm"] for r in rows if math.isfinite(r["grad_norm"])], dtype=float)
        summary.append(
            {
                "alpha": float(alpha),
                "c_bar_mean": float(np.nanmean(c_vals)),
                "forward_finite_frac": float(np.mean([r["forward_finite"] for r in rows])),
                "grad_probe_ok_frac": float(np.mean([r["grad_probe_ok"] for r in rows])),
                "grad_norm_median": float(np.median(g_vals)) if g_vals.size else float("nan"),
            }
        )
    save_csv(out / "exp1_vit_alpha_summary.csv", summary)

    stable = [s for s in summary if s["forward_finite_frac"] >= 0.99 and s["grad_probe_ok_frac"] >= 0.99]
    note = (
        f"median_c_bar_in_stable_band={float(np.median([s['c_bar_mean'] for s in stable])):.4f} (empirical)"
        if stable
        else "no stable band"
    )

    apply_paper_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.8))
    xs = [s["alpha"] for s in summary]
    ax0.plot(xs, [s["c_bar_mean"] for s in summary], "o-", ms=3)
    ax0.set_xscale("log")
    ax0.set_xlabel(r"Input scale $\alpha$")
    ax0.set_ylabel(r"$\bar c_{\mathrm{geom}}$")
    ax0.set_title("ViT Exp1: geometric mean layer ratio vs $\\alpha$")
    ax0.grid(True, which="both", ls=":", alpha=0.5)

    gmed = [s["grad_norm_median"] for s in summary]
    ax1.plot(xs, gmed, "s-", color="C1", ms=3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Input scale $\alpha$")
    ax1.set_ylabel("Median grad norm (full ViT probe)")
    ax1.axhline(args.grad_explode_thresh, color="gray", ls="--", lw=0.9)
    ax1.set_title("Gradient probe vs $\\alpha$")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    fig.suptitle(f"Script7 Exp1 ViT | {note}", fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(out / "fig_exp1_vit_alpha.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig_exp1_vit_alpha.pdf", bbox_inches="tight")
    plt.close(fig)

    # 逐层 c_l：与 alpha、全模型 grad 探针对齐（同一次 (alpha, seed) 上前向 + 反传）
    if layer_raw_rows:
        z = 1.96
        transitions = sorted({int(r["layer_transition"]) for r in layer_raw_rows})
        apply_paper_style()
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for li in transitions:
            xs_m, ys_m, ys_lo, ys_hi = [], [], [], []
            for alpha in alphas:
                vals = np.array(
                    [
                        r["c_t"]
                        for r in layer_raw_rows
                        if int(r["layer_transition"]) == li and abs(float(r["alpha"]) - float(alpha)) < 1e-12
                    ],
                    dtype=float,
                )
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                ci = z * s / math.sqrt(vals.size) if vals.size > 1 else 0.0
                xs_m.append(float(alpha))
                ys_m.append(m)
                ys_lo.append(m - ci)
                ys_hi.append(m + ci)
            if xs_m:
                ax.fill_between(xs_m, ys_lo, ys_hi, alpha=0.12)
                ax.plot(xs_m, ys_m, "o-", ms=3, label=rf"$\ell={li}$")
        ax.axhline(1.0, color="gray", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel(r"Input scale $\alpha$")
        ax.set_ylabel(r"Layer ratio $c_\ell = N_{\ell+1}/N_\ell$")
        ax.set_title(r"ViT Exp1: $c_\ell$ vs $\alpha$ (mean $\pm$ 95\% CI over seeds)")
        ax.legend(fontsize=6, ncol=2, loc="best")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out / "fig_exp1_vit_c_ell_vs_alpha.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / "fig_exp1_vit_c_ell_vs_alpha.pdf", bbox_inches="tight")
        plt.close(fig)

        apply_paper_style()
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        fwd_ok_mask = np.array([bool(r["grad_probe_ok"]) for r in layer_raw_rows], dtype=bool)
        c_arr = np.array([r["c_t"] for r in layer_raw_rows], dtype=float)
        g_arr = np.array([r["grad_norm"] for r in layer_raw_rows], dtype=float)
        a_arr = np.array([r["alpha"] for r in layer_raw_rows], dtype=float)
        ell_arr = np.array([r["layer_transition"] for r in layer_raw_rows], dtype=int)
        sel = fwd_ok_mask & np.isfinite(c_arr) & np.isfinite(g_arr) & (g_arr > 0)
        sc = ax.scatter(
            c_arr[sel],
            g_arr[sel],
            c=np.log10(a_arr[sel]),
            s=12,
            alpha=0.55,
            cmap="viridis",
            ec="none",
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"$\log_{10}\alpha$")
        ax.set_xlabel(r"$c_\ell$")
        ax.set_ylabel("Grad norm (full ViT probe)")
        ax.set_yscale("log")
        ax.set_title(r"ViT Exp1: $c_\ell$ vs gradient probe (color $=\alpha$)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out / "fig_exp1_vit_c_ell_vs_grad.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / "fig_exp1_vit_c_ell_vs_grad.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Exp1 wrote {out} | {note}")


def _forward_metrics_from_model(model: ViTModel, pixel_values: torch.Tensor):
    with torch.no_grad():
        o = model(pixel_values, output_hidden_states=True)
    assert o.hidden_states is not None
    return hidden_states_to_metrics(o.hidden_states)


def run_exp2(args: argparse.Namespace) -> None:
    out = OUTPUT_ROOT / "exp2_vit_physical"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)  # type: ignore[arg-type]
    local_only = args.local_files_only and not args.allow_download
    configs = [
        ("gelu", "gelu"),
        ("relu", "relu"),
        ("light_depth_t2.0", "light_depth_t2.0"),
        ("phys_tau_softplus_z_b1.0", "phys_tau_softplus_z_b1.0"),
    ]

    raw_layer: list[dict] = []
    raw_c: list[dict] = []

    for act_label, act_id in configs:
        for enc_depth in args.encoder_depths:
            for si in range(args.num_seeds):
                model = load_vit(
                    model_id=args.model_id,
                    snapshot_path=args.snapshot_path,
                    local_only=local_only,
                    dtype=dtype,
                )
                model = model.to(device=device, dtype=dtype)
                truncate_encoder(model, enc_depth)
                apply_vit_mlp_activation(model, act_id)
                model.eval()
                seed = BASE_SEED + si
                pixel_values = make_pixel_batch(
                    batch=args.batch,
                    image_size=model.config.image_size,
                    device=device,
                    dtype=dtype,
                    seed=seed,
                    input_kind=args.input_kind,
                )
                with torch.no_grad():
                    layer_rows, c_ts = _forward_metrics_from_model(model, pixel_values)
                nuc_chain = [r["nuclear_norm"] for r in layer_rows]
                c_bar = nuclear_c_bar_geom(nuc_chain)
                for ell, r in enumerate(layer_rows):
                    raw_layer.append(
                        {
                            "activation_id": act_label,
                            "encoder_depth": enc_depth,
                            "seed": seed,
                            "layer_index": r["layer_index"],
                            "nuclear_norm": r["nuclear_norm"],
                            "ln_nuclear": float(math.log(r["nuclear_norm"]) if r["nuclear_norm"] > 0 else float("nan")),
                            "c_bar_geom": c_bar,
                        }
                    )
                for j, cv in enumerate(c_ts):
                    if cv is not None:
                        raw_c.append(
                            {
                                "activation_id": act_label,
                                "encoder_depth": enc_depth,
                                "seed": seed,
                                "layer_transition": j + 1,
                                "c_t": float(cv),
                                "abs_c_minus_1": float(abs(cv - 1.0)),
                            }
                        )
                del model

    save_csv(out / "exp2_vit_lnuclear_raw.csv", raw_layer)
    save_csv(out / "exp2_vit_c_raw.csv", raw_c)

    z = 1.96
    for enc_depth in args.encoder_depths:
        n_trans = enc_depth
        apply_paper_style()
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
        for ax, (act_label, _) in zip(axes, configs):
            data = [
                [
                    r["c_t"]
                    for r in raw_c
                    if r["activation_id"] == act_label
                    and r["encoder_depth"] == enc_depth
                    and r["layer_transition"] == li
                ]
                for li in range(1, n_trans + 1)
            ]
            ax.boxplot(data, positions=list(range(1, n_trans + 1)), widths=0.35, showfliers=False)
            ax.axhline(1.0, color="gray", ls="--", lw=0.9)
            ax.set_xlabel(r"Transition $\ell$")
            ax.set_ylabel(r"$c_\ell$")
            ax.set_title(act_label[:22])
            ax.grid(True, ls=":", alpha=0.5)
        fig.suptitle(f"Script7 Exp2 ViT $c_\\ell$ (encoder_depth={enc_depth})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"fig_exp2_vit_c_box_depth{enc_depth}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / f"fig_exp2_vit_c_box_depth{enc_depth}.pdf", bbox_inches="tight")
        plt.close(fig)

        apply_paper_style()
        fig, ax = plt.subplots(figsize=(7, 3.8))
        max_layer_idx = enc_depth
        layers_idx = list(range(max_layer_idx + 1))
        for act_label, _ in configs:
            means, lo, hi = [], [], []
            for ell in layers_idx:
                vals = np.array(
                    [
                        r["ln_nuclear"]
                        for r in raw_layer
                        if r["activation_id"] == act_label and r["encoder_depth"] == enc_depth and r["layer_index"] == ell
                    ],
                    dtype=float,
                )
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    means.append(float("nan"))
                    lo.append(float("nan"))
                    hi.append(float("nan"))
                    continue
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                ci = z * s / math.sqrt(vals.size) if vals.size > 1 else 0.0
                means.append(m)
                lo.append(m - ci)
                hi.append(m + ci)
            ax.fill_between(layers_idx, lo, hi, alpha=0.15)
            ax.plot(layers_idx, means, "o-", label=act_label)
        ax.set_xlabel(r"Layer index (0 = post-embedding)")
        ax.set_ylabel(r"$\ln(N_\ell)$")
        ax.set_title(f"ViT Exp2 ln nuclear (encoder_depth={enc_depth})")
        ax.legend(fontsize=7)
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out / f"fig_exp2_vit_ln_N_depth{enc_depth}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / f"fig_exp2_vit_ln_N_depth{enc_depth}.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Exp2 wrote {out}")


PER_LAYER_PROBE_RATIONALE = """\
ViT Exp3 — per-layer linear probe (full L-layer encoder)

Why not only "truncate depth k and train end-to-end"?
  Training with truncate_encoder(vit, k) measures task accuracy of a k-block
  network. That mixes (i) representation at depth k with (ii) a different
  architecture (fewer layers).

What this mode does
  - Load the full pretrained ViT (all L encoder blocks). Backbone frozen.
  - One forward defines hidden_states[ell], ell = 0 ... L (0 = post-embedding).
  - For each ell, train a separate Linear head on CLS hidden_states[ell][:, 0, :].
  - Report best test accuracy of that probe vs ell.

Why it matches N_ell and c_ell
  Nuclear norms and layer ratios c are computed from the same frozen forward
  (same batch geometry as elsewhere in Script7). Accuracy is "linearly
  readable" from the representation at each depth, comparable across ell on a
  single backbone — no re-instantiating truncated models for each k.

中文摘要
  截断 k 层训练：Acc 是「只有 k 个 block 的端到端网络」的表现。
  全层线性探针：固定完整 L 层前向，每层 CLS 上单独训一个线性分类头；
  Acc 表示「若仅用第 ell 层表示做线性决策」的好坏，与同一前向链上的核范数、c_ell 对齐，
  无需为每个 k 更换 backbone 结构。
"""



def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        r, _ = spearmanr(x, y)
        return float(r) if math.isfinite(r) else float("nan")
    except ImportError:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean()
        ry -= ry.mean()
        sx, sy = rx.std(), ry.std()
        if sx < 1e-12 or sy < 1e-12:
            return float("nan")
        return float((rx * ry).mean() / (sx * sy))


def plot_exp3_per_layer_acc_vs_layer(
    out: Path,
    summary: list[dict],
    full_L: int,
    *,
    fig_stem: str = "fig_exp3_vit_per_layer_acc_vs_layer",
    note_name: str = "exp3_vit_per_layer_spearman_note.txt",
) -> tuple[float, float]:
    """双 y 轴：逐层线性探针准确率与 prefix $\\bar c_{geom}$。无标题；图例置于图下方避免遮挡。"""
    summary = sorted(summary, key=lambda r: int(r["layer_index"]))
    depths = np.array([s["layer_index"] for s in summary], dtype=float)
    accs = np.array([float(s["best_test_acc"]) for s in summary], dtype=float)

    def _joint_steep_layer_boundaries(
        max_lines: int = 3, min_frac: float = 0.18
    ) -> list[float]:
        """层间 $\\ell-1\\to\\ell$ 上 |Δacc|·|Δ\\bar c| 较大的边界（x = \\ell-0.5），表示两曲线同一步都变化明显。"""
        scores: list[tuple[int, float]] = []
        for i in range(1, len(depths)):
            if not (
                np.isfinite(accs[i])
                and np.isfinite(accs[i - 1])
                and np.isfinite(cbars[i])
                and np.isfinite(cbars[i - 1])
            ):
                continue
            da = abs(float(accs[i]) - float(accs[i - 1]))
            dc = abs(float(cbars[i]) - float(cbars[i - 1]))
            scores.append((int(depths[i]), da * dc))
        if not scores:
            return []
        mx = max(s[1] for s in scores)
        if mx <= 0:
            return []
        thr = max(min_frac * mx, 1e-15)
        ranked = sorted((s for s in scores if s[1] >= thr), key=lambda t: -t[1])[:max_lines]
        seen: set[float] = set()
        xs: list[float] = []
        for ell, _ in ranked:
            xb = float(ell) - 0.5
            if xb not in seen:
                seen.add(xb)
                xs.append(xb)
        return xs

    def _fcell(key: str) -> list[float]:
        out: list[float] = []
        for s in summary:
            v = s.get(key)
            if v is None or v == "" or str(v).lower() in ("nan", "none"):
                out.append(float("nan"))
            else:
                out.append(float(v))
        return out

    mc = np.array(_fcell("mean_abs_c_minus_1"), dtype=float)
    cbars = np.array(_fcell("c_bar_geom_probe"), dtype=float)

    vlines = _joint_steep_layer_boundaries()

    finite_acc = np.isfinite(accs)
    if finite_acc.sum() >= 2:
        rho_d = spearman_rho(depths[finite_acc], accs[finite_acc])
        mcf = mc[finite_acc]
        af = accs[finite_acc]
        fin_c = np.isfinite(mcf) & np.isfinite(af)
        rho_c = spearman_rho(mcf[fin_c], af[fin_c]) if fin_c.sum() >= 2 else float("nan")
    else:
        rho_d, rho_c = float("nan"), float("nan")

    apply_paper_style()
    dodge = 0.07 if depths.size == 1 else 0.0
    x_acc = depths - dodge
    x_c = depths + dodge
    ms = 7.2
    z_acc, z_c = 5, 4
    col_acc, col_c = "#1565c0", "#c62828"
    mew = 1.25
    mk_empty = {"marker": "o", "markersize": ms, "markerfacecolor": "none", "markeredgewidth": mew}
    fig, ax = plt.subplots(figsize=(7.0, 3.85))
    fig.subplots_adjust(left=0.11, right=0.88, top=0.96, bottom=0.24)
    ax2 = ax.twinx()
    for xb in vlines:
        ax.axvline(xb, color="#64748b", ls=":", lw=1.15, alpha=0.88, zorder=1.2)

    if depths.size <= 1:
        (ln_acc,) = ax.plot(
            x_acc,
            accs,
            "-",
            color=col_acc,
            markeredgecolor=col_acc,
            linewidth=1.45,
            zorder=z_acc,
            label="Test accuracy (linear probe)",
            **mk_empty,
        )
    else:
        (ln_acc,) = ax.plot(
            depths,
            accs,
            "-",
            color=col_acc,
            markeredgecolor=col_acc,
            linewidth=1.45,
            zorder=z_acc,
            label="Test accuracy (linear probe)",
            **mk_empty,
        )
    ax.set_xlabel(r"Layer index $\ell$")
    ax.set_ylabel("Test accuracy (CIFAR-10)")
    ax.set_xlim(depths.min() - 0.4, depths.max() + 0.4)
    if depths.size <= 1:
        (ln_c,) = ax2.plot(
            x_c,
            cbars,
            "--",
            color=col_c,
            markeredgecolor=col_c,
            linewidth=1.35,
            dashes=(6, 3),
            zorder=z_c,
            alpha=0.95,
            label=r"$\bar c_{\mathrm{geom}}$ (nuclear norms $N_0,\ldots,N_\ell$)",
            **mk_empty,
        )
    else:
        (ln_c,) = ax2.plot(
            depths,
            cbars,
            "--",
            color=col_c,
            markeredgecolor=col_c,
            linewidth=1.35,
            dashes=(6, 3),
            zorder=z_c,
            alpha=0.95,
            label=r"$\bar c_{\mathrm{geom}}$ (nuclear norms $N_0,\ldots,N_\ell$)",
            **mk_empty,
        )
    ax2.set_ylabel(r"$\bar c_{\mathrm{geom}}$ from $N_0,\ldots,N_\ell$")
    ax.grid(True, ls=":", alpha=0.5)
    h1, l1 = [ln_acc], ["Test accuracy (linear probe)"]
    h2, l2 = [ln_c], [r"$\bar c_{\mathrm{geom}}$ ($N_0,\ldots,N_\ell$)"]
    fig.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.06),
        bbox_transform=fig.transFigure,
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        fontsize=8,
        columnspacing=1.4,
        handletextpad=0.5,
        borderaxespad=0.6,
    )
    fig.savefig(out / f"{fig_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{fig_stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    with (out / note_name).open("w", encoding="utf-8") as f:
        f.write(
            f"rho(acc, layer_index)={rho_d}\nrho(acc, mean|c-1| prefix)={rho_c}\n"
            "per-layer linear probe on frozen full ViT; observational.\n"
        )
    return rho_d, rho_c


def _normalize_cls_for_probe(h: torch.Tensor, mode: str) -> torch.Tensor:
    """Optional CLS normalization before linear probe (depth-confound control)."""
    if not mode or mode == "none":
        return h
    if mode == "l2":
        return F.normalize(h, p=2, dim=-1, eps=1e-6)
    raise ValueError(f"unknown probe_normalize: {mode!r}")


def _run_exp3_per_layer_probe(
    args: argparse.Namespace,
    out: Path,
    device: torch.device,
    local_only: bool,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    full_L: int,
) -> None:
    """完整 L 层 ViT 冻结；对 hidden_states 每一层 CLS 单独训练线性探针，并记录 N/c（单次前向）。"""
    rationale_path = out / "exp3_vit_per_layer_probe_rationale.txt"
    rationale_path.write_text(PER_LAYER_PROBE_RATIONALE.strip() + "\n", encoding="utf-8")

    set_seed(BASE_SEED)
    vit = load_vit(
        model_id=args.model_id,
        snapshot_path=args.snapshot_path,
        local_only=local_only,
        dtype=torch.float32,
    )
    apply_vit_mlp_activation(vit, args.cls_activation)
    vit = vit.to(device)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)

    x0, _ = next(iter(test_loader))
    x0 = x0.to(device)
    with torch.no_grad():
        out0 = vit(x0, output_hidden_states=True)
    hs0 = out0.hidden_states
    assert hs0 is not None
    n_hs = len(hs0)
    hidden_size = vit.config.hidden_size

    layer_rows, c_ts = hidden_states_to_metrics(hs0)
    nucs = [r["nuclear_norm"] for r in layer_rows]

    pe = max(1, int(getattr(args, "per_layer_epochs", 1)))
    probe_norm = getattr(args, "probe_normalize", "none") or "none"
    ell_lo = max(0, int(getattr(args, "per_layer_from", 0)))
    ell_hi_arg = int(getattr(args, "per_layer_to", -1))
    ell_hi = n_hs - 1 if ell_hi_arg < 0 else min(ell_hi_arg, n_hs - 1)
    if ell_lo > ell_hi:
        raise SystemExit(f"per_layer_from ({ell_lo}) > per_layer_to ({ell_hi})")

    pref_csv = getattr(args, "per_layer_prefill_csv", None)
    summary: list[dict] = []
    if pref_csv:
        summary.extend(_load_per_layer_prefill_csv(Path(pref_csv)))
    layer_raw: list[dict] = []
    for ell, n in enumerate(nucs):
        ct = c_ts[ell] if ell < len(c_ts) else None
        layer_raw.append({"layer_index": ell, "nuclear_norm": n, "c_t": ct})

    for ell in range(ell_lo, ell_hi + 1):
        set_seed(BASE_SEED + 17_000 + ell)
        probe = nn.Linear(hidden_size, 10).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_acc = 0.0
        for ep in range(pe):
            probe.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.no_grad():
                    o = vit(xb, output_hidden_states=True)
                    h_all = o.hidden_states
                    assert h_all is not None
                    h_cls = _normalize_cls_for_probe(h_all[ell][:, 0, :].detach(), probe_norm)
                logits = probe(h_cls)
                opt.zero_grad()
                F.cross_entropy(logits, yb).backward()
                opt.step()
            probe.eval()
            correct = tot = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    o = vit(xb, output_hidden_states=True)
                    h_all = o.hidden_states
                    assert h_all is not None
                    h_cls = _normalize_cls_for_probe(h_all[ell][:, 0, :], probe_norm)
                    pred = probe(h_cls).argmax(1)
                    correct += (pred == yb).sum().item()
                    tot += yb.numel()
            ep_acc = correct / max(tot, 1)
            best_acc = max(best_acc, ep_acc)
            print(
                f"Exp3 per_layer_probe layer={ell}/{n_hs - 1} norm={probe_norm} "
                f"epoch {ep + 1}/{pe} test_acc={ep_acc:.4f} best={best_acc:.4f}",
                flush=True,
            )

        prefix = nucs[: ell + 1]
        c_bar_pre = nuclear_c_bar_geom(prefix)
        cs_pre = [
            abs(float(c_ts[j]) - 1.0)
            for j in range(ell)
            if c_ts[j] is not None and math.isfinite(float(c_ts[j]))
        ]
        mean_abs_c1 = float(np.mean(cs_pre)) if cs_pre else float("nan")

        summary.append(
            {
                "layer_index": ell,
                "encoder_depth": ell,
                "best_test_acc": best_acc,
                "forward_only": 0,
                "per_layer_probe": 1,
                "cls_activation": args.cls_activation,
                "probe_normalize": probe_norm,
                "freeze_backbone": 1,
                "c_bar_geom_probe": c_bar_pre,
                "mean_abs_c_minus_1": mean_abs_c1,
                "nuclear_norm_layer": nucs[ell],
                "c_t_to_next": c_ts[ell] if ell < len(c_ts) else None,
                "representation": f"per_layer_linear_probe_cls_frozen_full_vit_{probe_norm}",
            }
        )
        del probe

    del vit

    summary.sort(key=lambda r: int(r["layer_index"]))

    sum_name = "exp3_vit_per_layer_summary.csv"
    lay_name = "exp3_vit_per_layer_layer_raw.csv"
    fig_stem = "fig_exp3_vit_per_layer_acc_vs_layer"
    note_name = "exp3_vit_per_layer_spearman_note.txt"

    save_csv(out / sum_name, summary)
    save_csv(out / lay_name, layer_raw)

    plot_exp3_per_layer_acc_vs_layer(out, summary, full_L, fig_stem=fig_stem, note_name=note_name)
    print(f"Exp3 per_layer_probe wrote {out} | rationale: {rationale_path}")


def run_exp3(args: argparse.Namespace) -> None:
    """CIFAR-10：224缩放 + ImageNet 归一化。

 默认：截断 encoder 深度 k，训练 ViT+头，记录 Acc 与探针 N、c。
    --per_layer_probe：完整 L 层 ViT冻结，每层 CLS 单独线性探针 Acc（见 exp3_vit_per_layer_probe_rationale.txt）。
    --forward_only：不训练，仅前向 N/c。
    """
    if getattr(args, "torch_num_threads", None) is not None:
        n = max(1, int(args.torch_num_threads))
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    fig_csv = getattr(args, "per_layer_figure_only", None)
    if fig_csv:
        p = Path(fig_csv).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--per_layer_figure_only: 不是文件: {p}")
        summary_rows = _load_per_layer_prefill_csv(p)
        out_dir = p.parent
        full_l = max(int(r["layer_index"]) for r in summary_rows)
        plot_exp3_per_layer_acc_vs_layer(out_dir, summary_rows, full_l)
        print(f"Exp3:已从 CSV 重画逐层图 -> {out_dir}", flush=True)
        return

    out = OUTPUT_ROOT / "exp3_vit_cifar"
    probe_norm = getattr(args, "probe_normalize", "none") or "none"
    if probe_norm != "none":
        out = OUTPUT_ROOT / f"exp3_vit_cifar_{probe_norm}probe"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)  # type: ignore[arg-type]
    local_only = args.local_files_only and not args.allow_download

    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise SystemExit(f"Exp3 需要 torchvision: {e}") from e

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    forward_only = bool(getattr(args, "forward_only", False))
    per_layer_probe = bool(getattr(args, "per_layer_probe", False))
    if forward_only and per_layer_probe:
        raise SystemExit("不能同时使用 --forward_only 与 --per_layer_probe")
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    nw = int(getattr(args, "num_workers", 0))
    if not forward_only:
        train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
        if getattr(args, "train_subset_n", None):
            n = min(int(args.train_subset_n), len(train_ds))
            train_ds = torch.utils.data.Subset(train_ds, list(range(n)))
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, num_workers=nw, pin_memory=False
        )
    else:
        train_loader = None
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=nw, pin_memory=False
    )

    probe = load_vit(
        model_id=args.model_id,
        snapshot_path=args.snapshot_path,
        local_only=local_only,
        dtype=torch.float32,
    )
    full_L = len(probe.encoder.layer)
    del probe

    if per_layer_probe:
        assert train_loader is not None
        _run_exp3_per_layer_probe(
            args, out, device, local_only, train_loader, test_loader, full_L
        )
        return

    raw_depths = args.depths.strip().lower()
    if raw_depths in ("all", "full", "1..l"):
        depth_list = list(range(1, full_L + 1))
    else:
        depth_list = [int(x.strip()) for x in args.depths.split(",") if x.strip()]
    depth_list = [d for d in depth_list if 1 <= d <= full_L]
    if not depth_list:
        raise SystemExit(f"无有效深度（ViT encoder 共 {full_L} 层，可用 --depths all 表示 1..{full_L}）")

    summary: list[dict] = []
    layer_raw: list[dict] = []
    ckpt_csv = getattr(args, "epoch_checkpoint_csv", None)
    ckpt_dir = getattr(args, "epoch_checkpoint_dir", None)
    if ckpt_csv:
        ckpt_csv = Path(ckpt_csv)
    if ckpt_dir:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_fields = [
        "encoder_depth",
        "epoch",
        "epochs_total",
        "last_epoch_test_acc",
        "best_test_acc_so_far",
        "train_subset_n",
        "freeze_backbone",
        "cls_activation",
    ]
    subset_n_str = str(int(args.train_subset_n)) if getattr(args, "train_subset_n", None) else ""

    for k in depth_list:
        set_seed(BASE_SEED + k)
        vit = load_vit(
            model_id=args.model_id,
            snapshot_path=args.snapshot_path,
            local_only=local_only,
            dtype=torch.float32,
        )
        truncate_encoder(vit, k)
        apply_vit_mlp_activation(vit, args.cls_activation)
        model = ViTClassifier(vit, num_classes=10).to(device)
        if forward_only:
            best_acc = float("nan")
            model.eval()
            print(f"Exp3 forward_only depth={k}/{full_L} (probe batch)", flush=True)
        else:
            if args.freeze_backbone:
                for p in model.vit.parameters():
                    p.requires_grad_(False)
                for p in model.classifier.parameters():
                    p.requires_grad_(True)
            opt = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            best_acc = 0.0
            progress_path = out / "exp3_vit_progress.json"
            assert train_loader is not None
            for ep in range(args.epochs):
                if args.freeze_backbone:
                    model.vit.eval()
                    model.classifier.train()
                else:
                    model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    logits = model(xb)
                    F.cross_entropy(logits, yb).backward()
                    opt.step()
                model.eval()
                correct = tot = 0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        pred = model(xb).argmax(1)
                        correct += (pred == yb).sum().item()
                        tot += yb.numel()
                ep_acc = correct / max(tot, 1)
                best_acc = max(best_acc, ep_acc)
                try:
                    payload = {
                        "encoder_depth": k,
                        "epoch_done": ep + 1,
                        "epochs_total": args.epochs,
                        "last_epoch_test_acc": ep_acc,
                        "best_test_acc_so_far": best_acc,
                    }
                    progress_path.write_text(json.dumps(payload, indent=0) + "\n", encoding="utf-8")
                except OSError:
                    pass
                print(
                    f"Exp3 depth={k} epoch {ep + 1}/{args.epochs} "
                    f"test_acc={ep_acc:.4f} best_test_acc={best_acc:.4f}",
                    flush=True,
                )
                if ckpt_csv is not None:
                    append_csv_row(
                        ckpt_csv,
                        {
                            "encoder_depth": k,
                            "epoch": ep + 1,
                            "epochs_total": args.epochs,
                            "last_epoch_test_acc": ep_acc,
                            "best_test_acc_so_far": best_acc,
                            "train_subset_n": subset_n_str,
                            "freeze_backbone": int(args.freeze_backbone),
                            "cls_activation": args.cls_activation,
                        },
                        ckpt_fields,
                    )
                if ckpt_dir is not None:
                    ckpt_path = ckpt_dir / f"depth{k}_epoch{ep + 1}.pt"
                    sd = {kk: vv.detach().cpu() for kk, vv in model.state_dict().items()}
                    torch.save(sd, ckpt_path)

            model.eval()
        xb, _ = next(iter(test_loader))
        xb = xb.to(device)
        with torch.no_grad():
            _, hs = model(xb, output_hidden_states=True)
        layer_rows, c_ts = hidden_states_to_metrics(hs)
        nucs = [r["nuclear_norm"] for r in layer_rows]
        c_bar = nuclear_c_bar_geom(nucs)
        cs = [abs(c - 1.0) for c in c_ts if c is not None and math.isfinite(c)]
        mean_abs_c1 = float(np.mean(cs)) if cs else float("nan")

        summary.append(
            {
                "encoder_depth": k,
                "best_test_acc": best_acc,
                "forward_only": int(forward_only),
                "cls_activation": args.cls_activation,
                "freeze_backbone": int(args.freeze_backbone),
                "c_bar_geom_probe": c_bar,
                "mean_abs_c_minus_1": mean_abs_c1,
                "nuclear_final_probe": nucs[-1] if nucs else float("nan"),
                "representation": "vit_hidden_states_seq_x_hidden_svd_mean_batch",
            }
        )
        for ell, n in enumerate(nucs):
            ct = c_ts[ell] if ell < len(c_ts) else None
            layer_raw.append({"encoder_depth": k, "layer_index": ell, "nuclear_norm": n, "c_t": ct})
        del model

    if forward_only:
        sum_name = "exp3_vit_forward_summary.csv"
        lay_name = "exp3_vit_forward_layer_raw.csv"
        fig_stem = "fig_exp3_vit_forward_cbar_vs_depth"
        note_name = "exp3_vit_forward_spearman_note.txt"
    else:
        sum_name = "exp3_vit_depth_summary.csv"
        lay_name = "exp3_vit_depth_layer_raw.csv"
        fig_stem = "fig_exp3_vit_acc_vs_depth"
        note_name = "exp3_vit_spearman_note.txt"

    save_csv(out / sum_name, summary)
    save_csv(out / lay_name, layer_raw)

    depths = np.array([s["encoder_depth"] for s in summary], dtype=float)
    accs = np.array([s["best_test_acc"] for s in summary], dtype=float)
    mc = np.array([s["mean_abs_c_minus_1"] for s in summary], dtype=float)
    finite_acc = np.isfinite(accs)
    if finite_acc.sum() >= 2:
        rho_d = spearman_rho(depths[finite_acc], accs[finite_acc])
        mcf = mc[finite_acc]
        af = accs[finite_acc]
        fin_c = np.isfinite(mcf) & np.isfinite(af)
        rho_c = spearman_rho(mcf[fin_c], af[fin_c]) if fin_c.sum() >= 2 else float("nan")
    else:
        rho_d, rho_c = float("nan"), float("nan")

    if forward_only:
        rho_txt = "forward-only probe (no training); acc n/a"
    elif depths.size >= 2 and finite_acc.sum() >= 2:
        rho_txt = f"Spearman(acc,depth)={rho_d:.3f}, Spearman(acc,|c-1|)={rho_c:.3f}"
    else:
        rho_txt = "Spearman: n/a (need ≥2 depths or trained acc)"

    apply_paper_style()
    cbars = np.array([s["c_bar_geom_probe"] for s in summary], dtype=float)
    ms = 8
    z_acc, z_c = 5, 4
    dodge = 0.07 if depths.size == 1 else 0.0
    x_acc = depths - dodge
    x_c = depths + dodge

    if forward_only or not finite_acc.any():
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(depths, cbars, "s-", color="C1", markersize=ms, label=r"$\bar c_{\mathrm{geom}}$ (probe)")
        ax.set_xlabel("ViT encoder depth (truncated)")
        ax.set_ylabel(r"$\bar c_{\mathrm{geom}}$ (probe)")
        ax.set_title(f"Script7 Exp3 ViT+CIFAR | {rho_txt}\n(single test batch, observational)")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=7)
    else:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        if depths.size <= 1:
            ax.plot(x_acc, accs, "o", color="C0", markersize=ms, zorder=z_acc, label="best test acc (CIFAR-10)")
        else:
            ax.plot(depths, accs, "o-", color="C0", markersize=ms, zorder=z_acc, label="best test acc (CIFAR-10)")
        ax.set_xlabel("ViT encoder depth (truncated)")
        ax.set_ylabel("Accuracy")
        ax2 = ax.twinx()
        if depths.size <= 1:
            ax2.plot(x_c, cbars, "s", color="C1", markersize=ms, zorder=z_c, alpha=0.9, label=r"$\bar c_{\mathrm{geom}}$")
        else:
            ax2.plot(depths, cbars, "s--", color="C1", markersize=ms, zorder=z_c, alpha=0.85, label=r"$\bar c_{\mathrm{geom}}$")
        ax2.set_ylabel(r"$\bar c_{\mathrm{geom}}$ (probe)")
        ax.set_title(f"Script7 Exp3 ViT+CIFAR | {rho_txt}\n(observational)")
        ax.grid(True, ls=":", alpha=0.5)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(out / f"{fig_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{fig_stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    with (out / note_name).open("w", encoding="utf-8") as f:
        f.write(
            f"rho(acc, depth)={rho_d}\nrho(acc, mean|c-1|)={rho_c}\nObservational only.\n"
        )
    print(f"Exp3 wrote {out}")


def run_exp4(args: argparse.Namespace) -> None:
    """CIFAR-10：完整 ViT 冻结 + Exp3 同款逐层线性探针；固定测试 batch 上扫 probe_alpha。
    记录 c_\\ell、各 encoder block 的 grad 子范数（Exp1 同款 last_hidden_state.sum() 反传）、ViT 总 grad。
    可选：每层探针训完后在该层 CLS 上算单 batch CE。"""
    if getattr(args, "torch_num_threads", None) is not None:
        n = max(1, int(args.torch_num_threads))
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    out = OUTPUT_ROOT / "exp4_vit_cifar"
    out.mkdir(parents=True, exist_ok=True)
    rationale = out / "exp4_vit_grad_c_rationale.txt"
    rationale.write_text(
        "Script7 Exp4: CIFAR-10 + full ViT, frozen backbone, per-layer linear probes (same protocol as Exp3).\n"
        "After training each Linear head on CLS hidden_states[ell], scan probe_alpha on one fixed test batch.\n"
        "- c_l, N_l: hidden_states + hidden_states_to_metrics.\n"
        "- grad: last_hidden_state.sum().backward(); ||.||_2 over all ViT params and per encoder.layer[i] "
        "(subnorm at block i pairs with c_t[i] = N_{i+1}/N_i).\n"
        "- probe_ce (optional): one-batch CE at alpha=1 after training the probe at that ell.\n"
        "See exp3_vit_per_layer_probe_rationale.txt for why per-layer probes on a full encoder.\n",
        encoding="utf-8",
    )

    device = torch.device(args.device)
    local_only = args.local_files_only and not args.allow_download

    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise SystemExit(f"Exp4 需要 torchvision: {e}") from e

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    if getattr(args, "train_subset_n", None):
        nsub = min(int(args.train_subset_n), len(train_ds))
        train_ds = torch.utils.data.Subset(train_ds, list(range(nsub)))
    nw = int(getattr(args, "num_workers", 0))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=nw, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=nw, pin_memory=False
    )

    probe_alphas = [float(x.strip()) for x in args.probe_alphas.split(",") if x.strip()]
    if not probe_alphas:
        raise SystemExit("probe_alphas 不能为空")

    print("Exp4: loading ViT...", flush=True)
    set_seed(BASE_SEED)
    vit = load_vit(
        model_id=args.model_id,
        snapshot_path=args.snapshot_path,
        local_only=local_only,
        dtype=torch.float32,
    )
    apply_vit_mlp_activation(vit, args.cls_activation)
    vit = vit.to(device)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)

    full_L = len(vit.encoder.layer)
    x0, _ = next(iter(test_loader))
    x0 = x0.to(device)
    print(
        f"Exp4: reference forward on device={device} batch={x0.shape[0]} "
        f"(then N/c via SVD on CPU — silent minutes if batch is large)",
        flush=True,
    )
    with torch.no_grad():
        out0 = vit(x0, output_hidden_states=True)
    hs0 = out0.hidden_states
    assert hs0 is not None
    n_hs = len(hs0)
    hidden_size = vit.config.hidden_size

    print("Exp4: hidden_states_to_metrics (SVD) starting...", flush=True)
    layer_rows_ref, c_ts_ref = hidden_states_to_metrics(hs0)
    print("Exp4: reference N/c metrics done.", flush=True)
    nucs_ref = [r["nuclear_norm"] for r in layer_rows_ref]

    pe = max(1, int(getattr(args, "per_layer_epochs", 1)))
    ell_lo = max(0, int(getattr(args, "per_layer_from", 0)))
    ell_hi_arg = int(getattr(args, "per_layer_to", -1))
    ell_hi = n_hs - 1 if ell_hi_arg < 0 else min(ell_hi_arg, n_hs - 1)
    if ell_lo > ell_hi:
        raise SystemExit(f"per_layer_from ({ell_lo}) > per_layer_to ({ell_hi})")

    pref_csv = getattr(args, "per_layer_prefill_csv", None)
    per_layer_summary: list[dict] = []
    if pref_csv:
        per_layer_summary.extend(_load_per_layer_prefill_csv(Path(pref_csv)))

    xb_probe, yb_probe = next(iter(test_loader))
    xb_probe = xb_probe.to(device)
    yb_probe = yb_probe.to(device)

    print(
        f"Exp4: per-layer linear probes layers {ell_lo}..{ell_hi}, epochs={pe}",
        flush=True,
    )
    for ell in range(ell_lo, ell_hi + 1):
        set_seed(BASE_SEED + 19_000 + ell)
        probe = nn.Linear(hidden_size, 10).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_acc = 0.0
        for ep in range(pe):
            probe.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.no_grad():
                    o = vit(xb, output_hidden_states=True)
                    h_all = o.hidden_states
                    assert h_all is not None
                    h_cls = h_all[ell][:, 0, :].detach()
                logits = probe(h_cls)
                opt.zero_grad()
                F.cross_entropy(logits, yb).backward()
                opt.step()
            probe.eval()
            correct = tot = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    o = vit(xb, output_hidden_states=True)
                    h_all = o.hidden_states
                    assert h_all is not None
                    pred = probe(h_all[ell][:, 0, :]).argmax(1)
                    correct += (pred == yb).sum().item()
                    tot += yb.numel()
            ep_acc = correct / max(tot, 1)
            best_acc = max(best_acc, ep_acc)
            print(
                f"Exp4 per_layer_probe layer={ell}/{n_hs - 1} epoch {ep + 1}/{pe} "
                f"test_acc={ep_acc:.4f} best={best_acc:.4f}",
                flush=True,
            )

        probe_ce = float("nan")
        if args.log_probe_ce:
            try:
                with torch.no_grad():
                    o = vit(xb_probe, output_hidden_states=True)
                    h_all = o.hidden_states
                    assert h_all is not None
                    probe_ce = float(
                        F.cross_entropy(probe(h_all[ell][:, 0, :]), yb_probe).item()
                    )
            except Exception:
                probe_ce = float("nan")

        prefix = nucs_ref[: ell + 1]
        c_bar_pre = nuclear_c_bar_geom(prefix)
        cs_pre = [
            abs(float(c_ts_ref[j]) - 1.0)
            for j in range(ell)
            if c_ts_ref[j] is not None and math.isfinite(float(c_ts_ref[j]))
        ]
        mean_abs_c1 = float(np.mean(cs_pre)) if cs_pre else float("nan")

        per_layer_summary.append(
            {
                "layer_index": ell,
                "encoder_depth": ell,
                "best_test_acc": best_acc,
                "forward_only": 0,
                "per_layer_probe": 1,
                "per_layer_epochs": pe,
                "cls_activation": args.cls_activation,
                "freeze_backbone": 1,
                "c_bar_geom_probe": c_bar_pre,
                "mean_abs_c_minus_1": mean_abs_c1,
                "nuclear_norm_layer": nucs_ref[ell],
                "c_t_to_next": c_ts_ref[ell] if ell < len(c_ts_ref) else None,
                "probe_ce_batch": probe_ce,
                "representation": "per_layer_linear_probe_cls_frozen_full_vit_exp4",
            }
        )
        del probe

    print("Exp4: grad probe (sum last_hidden_state) at alpha=1 on probe batch...", flush=True)
    gtot1, blocks1, gok1 = vit_grad_probe_sum_last_hidden(
        vit, xb_probe.to(dtype=torch.float32), args.grad_explode_thresh
    )
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    for row in per_layer_summary:
        li = int(row["layer_index"])
        row["grad_norm_vit_total_alpha1"] = gtot1
        row["grad_probe_ok_alpha1"] = int(gok1)
        row["grad_norm_encoder_block_alpha1"] = (
            float(blocks1[li]) if li < len(blocks1) else float("nan")
        )

    per_layer_summary.sort(key=lambda r: int(r["layer_index"]))
    save_csv(out / "exp4_vit_per_layer_summary.csv", per_layer_summary)

    alpha_summary: list[dict] = []
    layer_raw: list[dict] = []

    print(
        f"Exp4: probe_alpha sweep {probe_alphas} (each: ViT forward + SVD + backward; "
        f"can take a few minutes per alpha on CPU SVD)",
        flush=True,
    )
    for alpha in probe_alphas:
        pv = (alpha * xb_probe).to(dtype=torch.float32)
        fwd_ok = True
        c_ts: list[float | None] = []
        nucs: list[float] = []
        try:
            with torch.no_grad():
                o = vit(pv, output_hidden_states=True)
                hs = o.hidden_states
                assert hs is not None
                layer_rows, c_ts = hidden_states_to_metrics(hs)
                nucs = [r["nuclear_norm"] for r in layer_rows]
        except Exception:
            fwd_ok = False

        c_bar = nuclear_c_bar_geom(nucs) if nucs else float("nan")
        cs = [abs(c - 1.0) for c in c_ts if c is not None and math.isfinite(c)]
        mean_abs_c1 = float(np.mean(cs)) if cs else float("nan")

        grad_ok = False
        gtot = float("nan")
        blocks: list[float] = []
        if fwd_ok:
            try:
                gtot, blocks, grad_ok = vit_grad_probe_sum_last_hidden(
                    vit, pv, args.grad_explode_thresh
                )
            except RuntimeError:
                grad_ok = False
                gtot, blocks = float("nan"), []
            vit.eval()
            for p in vit.parameters():
                p.requires_grad_(False)

        alpha_summary.append(
            {
                "probe_alpha": float(alpha),
                "c_bar_geom": c_bar,
                "mean_abs_c_minus_1": mean_abs_c1,
                "grad_norm_vit_sum_probe": gtot,
                "grad_probe_ok": int(grad_ok),
                "forward_finite": int(fwd_ok),
                "cls_activation": args.cls_activation,
            }
        )

        if fwd_ok and blocks:
            n_bl = len(blocks)
            for ell, n in enumerate(nucs):
                ct = c_ts[ell] if ell < len(c_ts) else None
                gblk = float(blocks[ell]) if ell < n_bl else float("nan")
                layer_raw.append(
                    {
                        "probe_alpha": float(alpha),
                        "layer_index": ell,
                        "nuclear_norm": n,
                        "c_t": ct,
                        "grad_norm_encoder_block": gblk,
                        "grad_norm_vit_sum_probe": gtot,
                    }
                )
        print(
            f"Exp4 probe_alpha={alpha:.4g} c_bar={c_bar:.4f} grad_total={gtot:.4g} grad_ok={grad_ok}",
            flush=True,
        )

    del vit
    save_csv(out / "exp4_vit_alpha_summary.csv", alpha_summary)
    save_csv(out / "exp4_vit_layer_raw.csv", layer_raw)

    apply_paper_style()
    if per_layer_summary:
        depths = np.array([s["layer_index"] for s in per_layer_summary], dtype=float)
        accs = np.array([s["best_test_acc"] for s in per_layer_summary], dtype=float)
        mc = np.array([s["mean_abs_c_minus_1"] for s in per_layer_summary], dtype=float)
        cbars = np.array([s["c_bar_geom_probe"] for s in per_layer_summary], dtype=float)
        finite_acc = np.isfinite(accs)
        if finite_acc.sum() >= 2:
            rho_d = spearman_rho(depths[finite_acc], accs[finite_acc])
            mcf = mc[finite_acc]
            af = accs[finite_acc]
            fin_c = np.isfinite(mcf) & np.isfinite(af)
            rho_c = spearman_rho(mcf[fin_c], af[fin_c]) if fin_c.sum() >= 2 else float("nan")
        else:
            rho_d, rho_c = float("nan"), float("nan")
        rho_txt_pl = (
            f"Spearman(acc,layer)={rho_d:.3f}, Spearman(acc,|c-1|)={rho_c:.3f}"
            if depths.size >= 2 and finite_acc.sum() >= 2
            else "Spearman: n/a"
        )
        ms = 8
        z_acc, z_c = 5, 4
        dodge = 0.07 if depths.size == 1 else 0.0
        x_acc = depths - dodge
        x_c = depths + dodge
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        if depths.size <= 1:
            ax.plot(x_acc, accs, "o", color="C0", markersize=ms, zorder=z_acc, label="linear probe acc")
        else:
            ax.plot(depths, accs, "o-", color="C0", markersize=ms, zorder=z_acc, label="linear probe acc")
        ax.set_xlabel(r"Layer index $\ell$ (0 = post-embedding)")
        ax.set_ylabel("Accuracy (CIFAR-10)")
        ax2 = ax.twinx()
        if depths.size <= 1:
            ax2.plot(
                x_c,
                cbars,
                "s",
                color="C1",
                markersize=ms,
                zorder=z_c,
                alpha=0.9,
                label=r"$\bar c_{\mathrm{geom}}$ (prefix)",
            )
        else:
            ax2.plot(
                depths,
                cbars,
                "s--",
                color="C1",
                markersize=ms,
                zorder=z_c,
                alpha=0.85,
                label=r"$\bar c_{\mathrm{geom}}$ (prefix)",
            )
        ax2.set_ylabel(r"$\bar c_{\mathrm{geom}}$ on layers $0..\ell$")
        ax.set_title(
            f"Script7 Exp4 | full {full_L}-block ViT frozen | {rho_txt_pl}\n"
            r"(per-layer linear probe + $\alpha=1$ prefix $\bar c$)"
        )
        ax.grid(True, ls=":", alpha=0.5)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fig_exp4_vit_per_layer_acc_vs_layer.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / "fig_exp4_vit_per_layer_acc_vs_layer.pdf", bbox_inches="tight")
        plt.close(fig)

        gb = np.array(
            [float(s["grad_norm_encoder_block_alpha1"]) for s in per_layer_summary],
            dtype=float,
        )
        fin_g = finite_acc & np.isfinite(gb) & (gb > 0)
        rho_ag = spearman_rho(accs[fin_g], gb[fin_g]) if fin_g.sum() >= 2 else float("nan")
        with (out / "exp4_vit_per_layer_spearman_note.txt").open("w", encoding="utf-8") as f:
            f.write(
                f"{rho_txt_pl}\nrho(acc, grad_block@alpha=1)={rho_ag}\n"
                "per-layer linear probe; observational.\n"
            )

    if layer_raw:
        c_pts, g_pts, a_pts = [], [], []
        for r in layer_raw:
            ct = r.get("c_t")
            if ct is None or not math.isfinite(float(ct)):
                continue
            gb = r.get("grad_norm_encoder_block")
            if gb is None or not math.isfinite(float(gb)) or float(gb) <= 0:
                continue
            c_pts.append(float(ct))
            g_pts.append(float(gb))
            a_pts.append(float(r["probe_alpha"]))
        if c_pts:
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            sc = ax.scatter(
                c_pts,
                g_pts,
                c=a_pts,
                s=22,
                alpha=0.65,
                cmap="coolwarm",
                ec="k",
                linewidths=0.2,
            )
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label(r"probe $\alpha$")
            ax.set_xlabel(r"$c_\ell$")
            ax.set_ylabel(r"$\|\nabla\|_2$ (encoder block $\ell$, sum-CLS probe)")
            ax.set_yscale("log")
            ax.set_title(
                "Exp4: $c_\ell$ vs per-block grad (frozen full ViT, per-layer linear probes trained)"
            )
            ax.grid(True, which="both", ls=":", alpha=0.5)
            fig.tight_layout()
            fig.savefig(out / "fig_exp4_c_vs_grad.png", dpi=300, bbox_inches="tight")
            fig.savefig(out / "fig_exp4_c_vs_grad.pdf", bbox_inches="tight")
            plt.close(fig)

    if len(probe_alphas) >= 2 and alpha_summary:
        apply_paper_style()
        alpha_summary.sort(key=lambda s: float(s["probe_alpha"]))
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.plot(
            [s["probe_alpha"] for s in alpha_summary],
            [s["grad_norm_vit_sum_probe"] for s in alpha_summary],
            "o-",
            ms=4,
            color="C0",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Input scale $\alpha$ (probe)")
        ax.set_ylabel(r"$\|\nabla\|_2$ (ViT total, sum-CLS probe)")
        ax.set_title("Exp4: total grad norm vs probe $\\alpha$")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out / "fig_exp4_grad_vs_alpha.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / "fig_exp4_grad_vs_alpha.pdf", bbox_inches="tight")
        plt.close(fig)

        apply_paper_style()
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.plot(
            [s["probe_alpha"] for s in alpha_summary],
            [s["c_bar_geom"] for s in alpha_summary],
            "s--",
            ms=4,
            color="C1",
        )
        ax.axhline(1.0, color="gray", ls=":", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel(r"Input scale $\alpha$ (probe)")
        ax.set_ylabel(r"$\bar c_{\mathrm{geom}}$")
        ax.set_title("Exp4: $\\bar c_{geom}$ vs probe $\\alpha$")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out / "fig_exp4_cbar_vs_alpha.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / "fig_exp4_cbar_vs_alpha.pdf", bbox_inches="tight")
        plt.close(fig)

    rho_txt = ""
    if len(alpha_summary) >= 2:
        c_b = np.array([float(s["c_bar_geom"]) for s in alpha_summary], dtype=float)
        g_b = np.array([float(s["grad_norm_vit_sum_probe"]) for s in alpha_summary], dtype=float)
        fin = np.isfinite(c_b) & np.isfinite(g_b) & (g_b > 0)
        if fin.sum() >= 2:
            rho_txt = f"Spearman(c_bar, grad_total)={spearman_rho(c_b[fin], g_b[fin]):.3f}\n"
    with (out / "exp4_vit_spearman_note.txt").open("w", encoding="utf-8") as f:
        f.write(
            rho_txt
            + "Observational; single test batch per alpha; full ViT frozen + per-layer linear probes.\n"
        )

    print(f"Exp4 wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Script7: ViT criticality experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model_id", type=str, default="google/vit-base-patch16-224")
    common.add_argument("--snapshot_path", type=str, default=None)
    common.add_argument("--local_files_only", action="store_true")
    common.add_argument("--allow_download", action="store_true")
    common.add_argument("--device", type=str, default="cpu")
    common.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    p1 = sub.add_parser("exp1", parents=[common], help="Input alpha sweep + grad probe")
    p1.add_argument("--alpha_min", type=float, default=0.05)
    p1.add_argument("--alpha_max", type=float, default=20.0)
    p1.add_argument("--alpha_steps", type=int, default=25)
    p1.add_argument("--num_seeds", type=int, default=4)
    p1.add_argument("--batch", type=int, default=2)
    p1.add_argument("--input_kind", choices=["imagenet_norm_randn", "randn"], default="imagenet_norm_randn")
    p1.add_argument("--grad_explode_thresh", type=float, default=1e6)
    p1.set_defaults(func=run_exp1)

    p2 = sub.add_parser("exp2", parents=[common], help="Activation comparison on ViT")
    p2.add_argument("--encoder_depths", type=int, nargs="+", default=[12])
    p2.add_argument("--num_seeds", type=int, default=6)
    p2.add_argument("--batch", type=int, default=2)
    p2.add_argument("--input_kind", choices=["imagenet_norm_randn", "randn"], default="imagenet_norm_randn")
    p2.set_defaults(func=run_exp2)

    p3 = sub.add_parser("exp3", parents=[common], help="CIFAR-10 classification vs truncated depth")
    p3.add_argument(
        "--depths",
        type=str,
        default="4,6,8,10,12",
        help="逗号分隔深度；关键字 all/full/1..l 表示 1..L（L 为 encoder 层数）",
    )
    p3.add_argument(
        "--forward_only",
        action="store_true",
        help="不训练：对每个 depth 仅加载截断 ViT + 测试集一个 batch 前向，写 N/c 探针（输出 exp3_vit_forward_*.csv）",
    )
    p3.add_argument(
        "--per_layer_probe",
        action="store_true",
        help="完整 L 层 ViT 冻结 backbone：对每层 CLS 单独训线性头，得每层 Acc；写 exp3_vit_per_layer_* 与理由说明 rationale文件",
    )
    p3.add_argument(
        "--per_layer_epochs",
        type=int,
        default=1,
        help="仅 --per_layer_probe：每层 linear probe 训练轮数（默认 1）",
    )
    p3.add_argument(
        "--per_layer_from",
        type=int,
        default=0,
        help="仅 --per_layer_probe：起始 layer_index（含），0=post-embedding",
    )
    p3.add_argument(
        "--per_layer_to",
        type=int,
        default=-1,
        help="仅 --per_layer_probe：结束层下标（含）；-1 表示到最后一层",
    )
    p3.add_argument(
        "--per_layer_prefill_csv",
        type=str,
        default=None,
        help="仅 --per_layer_probe：合并已有摘要行（如中断前已跑完的层），再跑 from..to",
    )
    p3.add_argument(
        "--probe_normalize",
        type=str,
        default="none",
        choices=("none", "l2"),
        help="仅 --per_layer_probe：CLS 向量 L2 归一化后再训 linear probe（控制 norm scale confound）",
    )
    p3.add_argument("--epochs", type=int, default=8)
    p3.add_argument("--batch", type=int, default=32)
    p3.add_argument("--lr", type=float, default=2e-4)
    p3.add_argument("--weight_decay", type=float, default=0.05)
    p3.add_argument("--cls_activation", type=str, default="gelu", help="gelu|relu|light_depth_t2.0|phys_tau_softplus_z_b1.0")
    p3.add_argument("--freeze_backbone", action="store_true", help="仅训分类头（更快）")
    p3.add_argument(
        "--train_subset_n",
        type=int,
        default=None,
        help="仅用前 N 张训练图（调试用；默认全量 CIFAR-10 训练集）",
    )
    p3.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker 数；0 单进程最省 CPU（MPS/CUDA 时常用 0）",
    )
    p3.add_argument(
        "--torch_num_threads",
        type=int,
        default=None,
        help="限制 PyTorch CPU 算子线程数；配合 MPS 时可设 2–4 以降低 CPU 占用",
    )
    p3.add_argument(
        "--epoch_checkpoint_csv",
        type=str,
        default=None,
        help="若设置，每个 epoch 结束追加一行 CSV（acc等，便于中断后留痕）",
    )
    p3.add_argument(
        "--epoch_checkpoint_dir",
        type=str,
        default=None,
        help="若设置，每个 epoch 结束将 model.state_dict()（CPU tensor）存为 depth{k}_epoch{e}.pt",
    )
    p3.add_argument(
        "--per_layer_figure_only",
        type=str,
        default=None,
        metavar="CSV",
        help="仅从 exp3_vit_per_layer_summary.csv 重画 fig_exp3_vit_per_layer_acc_vs_layer（不训练、不下载数据）",
    )
    p3.set_defaults(func=run_exp3)

    p4 = sub.add_parser(
        "exp4",
        parents=[common],
        help="CIFAR-10：全量 ViT 冻结 + 逐层线性探针（同 Exp3）；再扫 probe_alpha 做 c–grad",
    )
    p4.add_argument(
        "--per_layer_epochs",
        type=int,
        default=1,
        help="每层线性探针训练轮数（默认 1，省时间；可加至 2+ 略稳）",
    )
    p4.add_argument(
        "--per_layer_from",
        type=int,
        default=0,
        help="起始 layer_index（含），0=post-embedding",
    )
    p4.add_argument(
        "--per_layer_to",
        type=int,
        default=-1,
        help="结束 layer_index（含）；-1 表示到最后一层",
    )
    p4.add_argument(
        "--per_layer_prefill_csv",
        type=str,
        default=None,
        help="合并已有 per-layer 摘要行后再跑 from..to（与 Exp3 相同）",
    )
    p4.add_argument(
        "--probe_alphas",
        type=str,
        default="1.0",
        help="探针阶段输入缩放 α（逗号分隔，训练仍用 α=1）；多值时画 grad、c_bar_geom vs α",
    )
    p4.add_argument(
        "--batch",
        type=int,
        default=8,
        help="DataLoader batch；小一点可显著加快 CPU 上 N/c 的 SVD（默认 8）",
    )
    p4.add_argument("--lr", type=float, default=2e-4)
    p4.add_argument("--weight_decay", type=float, default=0.05)
    p4.add_argument(
        "--cls_activation",
        type=str,
        default="gelu",
        help="gelu|relu|light_depth_t2.0|phys_tau_softplus_z_b1.0",
    )
    p4.add_argument(
        "--train_subset_n",
        type=int,
        default=None,
        help="仅用前 N 张训练图（调试）",
    )
    p4.add_argument("--num_workers", type=int, default=0)
    p4.add_argument(
        "--torch_num_threads",
        type=int,
        default=None,
        help="限制 PyTorch CPU 线程数",
    )
    p4.add_argument(
        "--grad_explode_thresh",
        type=float,
        default=1e6,
        help="超过则标记 grad_probe_ok=0（同 Exp1）",
    )
    p4.add_argument(
        "--log_probe_ce",
        action="store_true",
        help="记录单 batch CE（对照标量，非梯度范数替代）",
    )
    p4.set_defaults(func=run_exp4)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
