# Script 5：预训练 ViT 前向 — 每层 token×hidden 表示矩阵的 SVD（核范数 / 谱范数）与层间比 c_t = N_{t+1}/N_t
# 激活消融：冻结权重下替换 MLP 非线性（gelu / relu / light_depth / phys 等，与 script7 对齐）。
# 可选：截断 encoder 深度、float16/bfloat16 前向（精度雪崩相关现象学）。
from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from activation_extras import LightDepthAct, ScatteringOpticalDepthAct
from paper_style import apply_paper_style

try:
    from transformers import ViTModel
    from transformers.utils import logging as tf_logging

    tf_logging.set_verbosity_error()
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "需要安装 transformers：pip install transformers\n" f"原始错误: {e}"
    ) from e

BASE_SEED = 42
RANK_THRESHOLD = 1e-6
COND_EPS = 1e-9
OUTPUT_DIR = Path("results_script5")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

InputKind = Literal["randn", "imagenet_norm_randn", "demo_bee"]
DtypeName = Literal["float32", "float16", "bfloat16"]

# 与 script7 `apply_vit_mlp_activation` 对齐的 id（CSV /图文件名原样保存）
ALLOWED_ACTIVATIONS = (
    "gelu",
    "relu",
    "light_depth_t2.0",
    "phys_tau_softplus_z_b1.0",
)


def analyze_matrix(M: torch.Tensor) -> tuple[int, float, float, float]:
    """与 script3 一致：对二维矩阵 M 做 SVD，返回秩、谱范数、核范数、条件数。"""
    s = np.linalg.svd(M.detach().float().cpu().numpy(), compute_uv=False)
    rank = int(np.sum(s > RANK_THRESHOLD))
    spectral = float(s[0]) if rank > 0 else 0.0
    nuclear = float(np.sum(s))
    cond = float(s[0] / s[-1]) if (rank > 0 and s[-1] > COND_EPS) else float("inf")
    return rank, spectral, nuclear, cond


def hf_hub_root() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def discover_cached_vit_models() -> list[str]:
    """扫描 hub 目录下名称含 vit 的已缓存模型文件夹（models--*）。"""
    root = hf_hub_root()
    if not root.is_dir():
        return []
    found: list[str] = []
    for p in sorted(root.glob("models--*vit*")):
        snaps = list(p.glob("snapshots/*"))
        if not snaps:
            continue
        snap = snaps[0]
        ok = (snap / "config.json").exists() and any(
            (snap / f).exists()
            for f in (
                "pytorch_model.bin",
                "model.safetensors",
                "pytorch_model.bin.index.json",
                "model.safetensors.index.json",
            )
        )
        if ok:
            # models--org--rest -> org/rest 作为 from_pretrained id
            rest = p.name.removeprefix("models--")
            if "--" in rest:
                org, repo_id = rest.split("--", 1)
                name = f"{org}/{repo_id}"
            else:
                name = rest
            found.append(name)
    return found


def replace_vit_mlp_activation(model: ViTModel, act_id: str) -> None:
    """gelu：不改动。其余：替换各 encoder block 的 MLP 激活（与 script7 一致）。"""
    if act_id == "gelu":
        return
    if act_id == "relu":
        act: nn.Module = nn.ReLU()
    elif act_id == "light_depth_t2.0":
        act = LightDepthAct(2.0)
    elif act_id == "phys_tau_softplus_z_b1.0":
        act = ScatteringOpticalDepthAct(1.0, "softplus_z")
    else:
        raise ValueError(f"Unknown activation id: {act_id}")
    for layer in model.encoder.layer:
        layer.intermediate.intermediate_act_fn = act


def truncate_encoder(model: ViTModel, max_layers: int | None) -> None:
    if max_layers is None:
        return
    full = model.encoder.layer
    n = len(full)
    if max_layers >= n:
        return
    model.encoder.layer = nn.ModuleList([full[i] for i in range(max_layers)])


def resolve_dtype(name: DtypeName) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    return torch.bfloat16


def iter_inputs(
    *,
    kinds: list[InputKind],
    batch: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Iterator[tuple[InputKind, torch.Tensor]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    for kind in kinds:
        if kind == "randn":
            x = torch.randn(batch, 3, image_size, image_size, device=device, dtype=dtype)
        elif kind == "imagenet_norm_randn":
            x = torch.randn(batch, 3, image_size, image_size, device=device, dtype=dtype)
            mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
            x = (x - mean) / std
        else:
            # torchvision 自带示例图（若安装且文件存在）；否则退回 ImageNet 归一化高斯图
            try:
                import torchvision.transforms as T
                from pathlib import Path as P

                from torchvision.io import read_image

                tv_root = P(__import__("torchvision").__file__).parent
                for name in ("dog2.jpg", "astronaut.jpg"):
                    img_path = tv_root / "prototype" / "datasets" / "assets" / name
                    if img_path.is_file():
                        im = read_image(str(img_path)).float() / 255.0
                        im = T.Resize((image_size, image_size))(im)
                        im = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(im)
                        x = im.unsqueeze(0).to(device=device, dtype=dtype)
                        if batch > 1:
                            x = x.repeat(batch, 1, 1, 1)
                        break
                else:
                    raise FileNotFoundError("no bundled torchvision demo image")
            except Exception:
                x = torch.randn(batch, 3, image_size, image_size, device=device, dtype=dtype)
                mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
                std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
                x = (x - mean) / std
        yield kind, x


def hidden_states_to_metrics(
    hidden_states: tuple[torch.Tensor, ...],
) -> tuple[list[dict], list[float | None]]:
    """
    hidden_states[i]: (B, seq, d)。对每个 batch 元素单独做 SVD，再对该层取 batch 平均指标。
    返回每层一条聚合记录（用于与 script 风格一致），以及逐层的 c_t（基于聚合后的核范数）。
    """
    per_layer_nuclear: list[float] = []
    per_layer_spectral: list[float] = []
    per_layer_rank: list[float] = []
    per_layer_cond_med: list[float] = []

    layer_records: list[dict] = []

    for idx, h in enumerate(hidden_states):
        # h: (B, L, d) -> 每个样本一张矩阵
        b = h.shape[0]
        nucs, specs, ranks, conds = [], [], [], []
        for bi in range(b):
            M = h[bi]  # (seq, d)
            rk, sp, nuc, cond = analyze_matrix(M)
            nucs.append(nuc)
            specs.append(sp)
            ranks.append(float(rk))
            conds.append(cond if np.isfinite(cond) else np.nan)
        per_layer_nuclear.append(float(np.mean(nucs)))
        per_layer_spectral.append(float(np.mean(specs)))
        per_layer_rank.append(float(np.mean(ranks)))
        finite = [c for c in conds if np.isfinite(c) and c > 0]
        per_layer_cond_med.append(float(np.median(finite)) if finite else float("nan"))

        layer_records.append(
            {
                "layer_index": idx,
                "nuclear_norm": per_layer_nuclear[-1],
                "spectral_norm": per_layer_spectral[-1],
                "rank_mean": per_layer_rank[-1],
                "condition_median": per_layer_cond_med[-1],
            }
        )

    c_ts: list[float | None] = []
    for i in range(len(per_layer_nuclear)):
        if i + 1 < len(per_layer_nuclear) and per_layer_nuclear[i] >1e-30:
            c_ts.append(per_layer_nuclear[i + 1] / per_layer_nuclear[i])
        else:
            c_ts.append(None)

    return layer_records, c_ts


def run_forward_metrics(
    model: ViTModel,
    pixel_values: torch.Tensor,
) -> tuple[list[dict], list[float | None]]:
    with torch.no_grad():
        out = model(pixel_values, output_hidden_states=True)
    hs = out.hidden_states
    assert hs is not None
    return hidden_states_to_metrics(hs)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize_raw(records: list[dict]) -> list[dict]:
    keys = sorted(
        {(r["activation"], r["input_kind"], r["layer_index"]) for r in records}
    )
    z = 1.96
    out: list[dict] = []
    for act, ik, li in keys:
        rows = [r for r in records if r["activation"] == act and r["input_kind"] == ik and r["layer_index"] == li]
        n = len(rows)
        nuc = np.array([r["nuclear_norm"] for r in rows], dtype=float)
        sp = np.array([r["spectral_norm"] for r in rows], dtype=float)
        rk = np.array([r["rank_mean"] for r in rows], dtype=float)
        ct_vals = np.array(
            [r["c_t"] for r in rows if r["c_t"] is not None and np.isfinite(r["c_t"])],
            dtype=float,
        )
        nuc_std = float(np.std(nuc, ddof=1)) if n > 1 else 0.0
        sp_std = float(np.std(sp, ddof=1)) if n > 1 else 0.0
        rec = {
            "activation": act,
            "input_kind": ik,
            "layer_index": li,
            "n_runs": n,
            "nuclear_mean": float(np.mean(nuc)),
            "nuclear_std": nuc_std,
            "nuclear_ci95_halfwidth": float(z * nuc_std / np.sqrt(n)) if n > 1 else 0.0,
            "spectral_mean": float(np.mean(sp)),
            "spectral_std": sp_std,
            "rank_mean": float(np.mean(rk)),
            "rank_std": float(np.std(rk, ddof=1)) if n > 1 else 0.0,
        }
        if ct_vals.size:
            rec["c_t_mean"] = float(np.mean(ct_vals))
            rec["c_t_std"] = float(np.std(ct_vals, ddof=1)) if ct_vals.size > 1 else 0.0
            rec["c_t_q25"] = float(np.quantile(ct_vals, 0.25))
            rec["c_t_q50"] = float(np.quantile(ct_vals, 0.50))
            rec["c_t_q75"] = float(np.quantile(ct_vals, 0.75))
        else:
            rec["c_t_mean"] = float("nan")
            rec["c_t_std"] = float("nan")
            rec["c_t_q25"] = float("nan")
            rec["c_t_q50"] = float("nan")
            rec["c_t_q75"] = float("nan")
        out.append(rec)
    out.sort(key=lambda r: (r["activation"], r["input_kind"], r["layer_index"]))
    return out


def plot_triptych(
    summary: list[dict],
    *,
    activation: str,
    input_kind: str,
    out_path: Path,
) -> None:
    rows = [r for r in summary if r["activation"] == activation and r["input_kind"] == input_kind]
    if not rows:
        return
    rows.sort(key=lambda r: r["layer_index"])
    layers = [r["layer_index"] for r in rows]
    nuc = np.array([r["nuclear_mean"] for r in rows])
    nuc_e = np.array([r["nuclear_ci95_halfwidth"] for r in rows])
    ct_mean = []
    ct_err = []
    ct_layers = []
    for r in rows:
        if np.isfinite(r.get("c_t_mean", float("nan"))):
            ct_layers.append(r["layer_index"])
            ct_mean.append(r["c_t_mean"])
            z = 1.96
            n = max(r["n_runs"], 1)
            ct_err.append(z * r.get("c_t_std", 0.0) / np.sqrt(n))

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    axes[0].errorbar(layers, nuc, yerr=nuc_e, fmt="o-", capsize=3, markersize=4)
    axes[0].set_xlabel(r"Layer index $\ell$ (0 = post-embedding)")
    axes[0].set_ylabel(r"Nuclear norm $N_\ell$")
    axes[0].set_title(r"$N_\ell$ vs depth")
    axes[0].grid(True, ls=":", alpha=0.5)

    log_n = np.log(np.clip(nuc, 1e-30, None))
    # CI on log：用一阶 delta 近似 log的误差条（可选）；此处仅画均值曲线
    axes[1].plot(layers, log_n, "s-", markersize=4)
    axes[1].set_xlabel(r"Layer index $\ell$")
    axes[1].set_ylabel(r"$\ln(N_\ell)$")
    axes[1].set_title(r"Semi-log: $\ln(N_\ell)$ vs $\ell$")
    axes[1].grid(True, ls=":", alpha=0.5)

    if ct_mean:
        axes[2].errorbar(ct_layers, ct_mean, yerr=ct_err, fmt="^-", capsize=3, markersize=4)
    axes[2].axhline(1.0, color="gray", ls="--", lw=0.9, alpha=0.8)
    axes[2].set_xlabel(r"Layer index $\ell$ ($c_\ell = N_{\ell+1}/N_\ell$)")
    axes[2].set_ylabel(r"$c_\ell$")
    axes[2].set_title(r"Layerwise ratio $c_\ell$")
    axes[2].grid(True, ls=":", alpha=0.5)

    fig.suptitle(f"ViT | activation={activation} | input={input_kind}", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_ct_hist(
    raw: list[dict],
    *,
    activation: str,
    input_kind: str,
    out_path: Path,
) -> None:
    vals = [
        float(r["c_t"])
        for r in raw
        if r["activation"] == activation
        and r["input_kind"] == input_kind
        and r["c_t"] is not None
        and np.isfinite(r["c_t"])
    ]
    if len(vals) < 2:
        return
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(vals, bins=min(40, max(10, len(vals) // 5)), density=True, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(r"$c_\ell = N_{\ell+1}/N_\ell$")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of $c_\\ell$ ({activation}, {input_kind})")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ViT representation SVD and c_t experiment (script 5)")
    p.add_argument("--model_id", type=str, default="google/vit-base-patch16-224")
    p.add_argument("--snapshot_path", type=str, default=None, help="本地 snapshot 目录，若设则优先于 model_id")
    p.add_argument("--local_files_only", action="store_true", help="仅从缓存加载（不尝试联网）")
    p.add_argument("--allow_download", action="store_true", help="允许联网下载（覆盖 local_files_only）")
    p.add_argument("--list_cached_vit", action="store_true", help="列出 hub 中已缓存的 vit 模型后退出")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--num_seeds", type=int, default=8)
    p.add_argument(
        "--input_kinds",
        type=str,
        default="randn,imagenet_norm_randn",
        help="逗号分隔：randn, imagenet_norm_randn, demo_bee",
    )
    p.add_argument(
        "--activations",
        type=str,
        default="gelu,relu,light_depth_t2.0,phys_tau_softplus_z_b1.0",
        help="逗号分隔：gelu | relu | light_depth_t2.0 | phys_tau_softplus_z_b1.0",
    )
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_layers", type=int, default=None, help="仅保留前 k 个 encoder block")
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_cached_vit:
        found = discover_cached_vit_models()
        print("Cached ViT-like models under hub:")
        for m in found:
            print(" ", m)
        if not found:
            print(" (none)")
        raise SystemExit(0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kinds: list[InputKind] = []
    for part in args.input_kinds.split(","):
        part = part.strip()
        if part in ("randn", "imagenet_norm_randn", "demo_bee"):
            kinds.append(part)  # type: ignore[arg-type]

    activations: list[str] = []
    for part in args.activations.split(","):
        part = part.strip().lower()
        if part in ALLOWED_ACTIVATIONS:
            activations.append(part)
        elif part:
            raise SystemExit(
                f"Unknown activation {part!r}. Allowed: {', '.join(ALLOWED_ACTIVATIONS)}"
            )
    if not activations:
        raise SystemExit("No valid --activations given.")

    dtype = resolve_dtype(args.dtype)  # type: ignore[arg-type]
    device = torch.device(args.device)

    load_kw: dict = {"torch_dtype": dtype}
    load_path = args.snapshot_path if args.snapshot_path else args.model_id
    local_only = args.local_files_only and not args.allow_download

    try:
        probe = ViTModel.from_pretrained(
            load_path,
            local_files_only=local_only,
            **load_kw,
        )
    except Exception as e:
        print(f"加载失败: {e}", file=sys.stderr)
        cached = discover_cached_vit_models()
        if cached:
            print("提示：已缓存的模型 id示例（可 --model_id 指定）：", file=sys.stderr)
            for m in cached[:8]:
                print(f"  {m}", file=sys.stderr)
        raise SystemExit(1) from e

    image_size = probe.config.image_size
    del probe

    raw_records: list[dict] = []

    for act in activations:
        model_act = ViTModel.from_pretrained(
            load_path,
            local_files_only=local_only,
            **load_kw,
        )
        model_act = model_act.to(device=device, dtype=dtype)
        model_act.eval()
        truncate_encoder(model_act, args.max_layers)
        replace_vit_mlp_activation(model_act, act)

        for seed_idx in range(args.num_seeds):
            seed = BASE_SEED + seed_idx
            for input_kind, pixels in iter_inputs(
                kinds=kinds,
                batch=args.batch,
                image_size=image_size,
                device=device,
                dtype=dtype,
                seed=seed,
            ):
                layer_rows, c_ts = run_forward_metrics(model_act, pixels)
                for row, c_t in zip(layer_rows, c_ts):
                    raw_records.append(
                        {
                            "model_id": str(load_path),
                            "seed": seed,
                            "input_kind": input_kind,
                            "activation": act,
                            "dtype": args.dtype,
                            "max_layers": args.max_layers if args.max_layers is not None else len(model_act.encoder.layer),
                            "layer_index": row["layer_index"],
                            "nuclear_norm": row["nuclear_norm"],
                            "spectral_norm": row["spectral_norm"],
                            "rank_mean": row["rank_mean"],
                            "condition_median": row["condition_median"],
                            "condition_number": row["condition_median"],
                            "c_t": c_t,
                        }
                    )

    raw_path = out_dir / "vit_svd_raw.csv"
    save_csv(raw_path, raw_records)
    summary = summarize_raw(raw_records)
    save_csv(out_dir / "vit_svd_summary.csv", summary)

    fig_dir = out_dir / "figures"
    for act in activations:
        for ik in kinds:
            safe_act = act.replace("/", "_")
            safe_ik = ik.replace("/", "_")
            plot_triptych(
                summary,
                activation=act,
                input_kind=ik,
                out_path=fig_dir / f"vit_norms_{safe_act}_{safe_ik}.png",
            )
            plot_ct_hist(
                raw_records,
                activation=act,
                input_kind=ik,
                out_path=fig_dir / f"vit_ct_hist_{safe_act}_{safe_ik}.png",
            )

    print(f"Wrote {raw_path} ({len(raw_records)} rows)")
    print(f"Wrote {out_dir / 'vit_svd_summary.csv'} ({len(summary)} rows)")
    print(f"Figures under {fig_dir}")


if __name__ == "__main__":
    main()
