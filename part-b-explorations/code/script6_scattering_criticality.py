# Script 6：临界散射系数（经验 c_crit）、物理激活稳住 c��1、深度–分类精度联动
# 用法：
#   python script6_scattering_criticality.py exp1 --help
#   python script6_scattering_criticality.py exp2 --help
#   python script6_scattering_criticality.py exp3 --help
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from activation_extras import LightDepthAct, ScatteringOpticalDepthAct, build_activation
from paper_style import apply_paper_style

BASE_SEED = 42
RANK_THRESHOLD = 1e-6
COND_EPS = 1e-9
OUTPUT_ROOT = Path("results_script6")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def analyze_matrix(M: torch.Tensor) -> tuple[int, float, float, float]:
    """与 script3 一致：对二维矩阵 M 做 SVD。"""
    s = np.linalg.svd(M.detach().float().cpu().numpy(), compute_uv=False)
    rank = int(np.sum(s > RANK_THRESHOLD))
    spectral = float(s[0]) if rank > 0 else 0.0
    nuclear = float(np.sum(s))
    cond = float(s[0] / s[-1]) if (rank > 0 and s[-1] > COND_EPS) else float("inf")
    return rank, spectral, nuclear, cond


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


class ScatteringLayer(nn.Module):
    def __init__(self, dim: int, act: nn.Module) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.act = act

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.lin(x)
        return z, self.act(z)


def forward_stack_nuclear_chain(layers: nn.ModuleList, x0: torch.Tensor) -> tuple[list[float], bool]:
    """返回 [N0, N1, ..., NL]（N0 为输入 x0 的核范数，其后为每层激活后）及前向是否全有限。"""
    finite = True
    _, _, n0, _ = analyze_matrix(x0)
    nucs: list[float] = [float(n0)]
    x = x0
    with torch.no_grad():
        for layer in layers:
            _, x = layer(x)
            if not torch.isfinite(x).all():
                finite = False
            _, _, n, _ = analyze_matrix(x)
            nucs.append(float(n))
    return nucs, finite


def nuclear_chain_to_c(nuc_chain: list[float]) -> tuple[list[float | None], float]:
    """nuc_chain: [N0,...,NL]，c[i]=N_{i+1}/N_i，长度 L；几何平均 c_bar = (NL/N0)^(1/L)。"""
    c_list: list[float | None] = []
    for i in range(len(nuc_chain) - 1):
        a, b = nuc_chain[i], nuc_chain[i + 1]
        if a > 1e-30 and math.isfinite(a) and math.isfinite(b):
            c_list.append(b / a)
        else:
            c_list.append(None)
    depth = len(c_list)
    if depth == 0 or nuc_chain[0] <= 1e-30:
        return c_list, float("nan")
    ratio = nuc_chain[-1] / nuc_chain[0]
    if not math.isfinite(ratio) or ratio <= 0:
        return c_list, float("nan")
    c_bar = float(ratio ** (1.0 / depth))
    return c_list, c_bar


def grad_norm_total(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.data.pow(2).sum().item())
    return math.sqrt(total)


# --- Exp1 -----------------------------------------------------------------


def run_exp1(args: argparse.Namespace) -> None:
    out = OUTPUT_ROOT / "exp1_alpha"
    out.mkdir(parents=True, exist_ok=True)
    alphas = np.logspace(math.log10(args.alpha_min), math.log10(args.alpha_max), args.alpha_steps)
    raw: list[dict] = []

    for alpha in alphas:
        set_seed(args.weight_seed)
        layers = nn.ModuleList(
            [ScatteringLayer(args.dim, nn.ReLU()) for _ in range(args.depth)]
        )
        for p in layers.parameters():
            p.requires_grad_(True)

        for si in range(args.num_seeds):
            set_seed(BASE_SEED + 10_000 + si)
            x0 = alpha * torch.randn(args.dim, args.dim, dtype=torch.float32)
            layers.eval()
            with torch.no_grad():
                chain, fwd_ok = forward_stack_nuclear_chain(layers, x0)
            c_list, c_bar = nuclear_chain_to_c(chain)
            c_finite = [c for c in (c_list or []) if c is not None and math.isfinite(c)]
            c_mean_arith = float(np.mean(c_finite)) if c_finite else float("nan")

            # 反传探针：同一权重，新输入样本
            xb = (alpha * torch.randn(args.dim, args.dim, dtype=torch.float32)).requires_grad_(False)
            layers.train()
            for p in layers.parameters():
                p.grad = None
            x = xb
            grad_ok = True
            gnorm = float("nan")
            try:
                for layer in layers:
                    _, x = layer(x)
                if not torch.isfinite(x).all():
                    grad_ok = False
                else:
                    loss = x.sum()
                    loss.backward()
                    if not all(
                        (p.grad is None or torch.isfinite(p.grad).all()) for p in layers.parameters()
                    ):
                        grad_ok = False
                    gnorm = grad_norm_total(layers)
                    if not math.isfinite(gnorm):
                        grad_ok = False
                    if gnorm > args.grad_explode_thresh:
                        grad_ok = False  # 计划：梯度爆炸判据
            except RuntimeError:
                grad_ok = False

            raw.append(
                {
                    "alpha": float(alpha),
                    "input_seed": BASE_SEED + 10_000 + si,
                    "depth": args.depth,
                    "dim": args.dim,
                    "c_bar_geom": c_bar,
                    "c_mean_arith": c_mean_arith,
                    "nuclear_final": chain[-1] if chain else float("nan"),
                    "nuclear_input": chain[0] if chain else float("nan"),
                    "forward_finite": int(fwd_ok),
                    "grad_probe_ok": int(grad_ok),
                    "grad_norm": gnorm,
                }
            )

    save_csv(out / "exp1_alpha_raw.csv", raw)

    # 按 alpha 汇总
    summary_rows: list[dict] = []
    for alpha in alphas:
        rows = [r for r in raw if abs(r["alpha"] - alpha) < 1e-12]
        c_vals = np.array([r["c_bar_geom"] for r in rows], dtype=float)
        g_vals = np.array([r["grad_norm"] for r in rows if math.isfinite(r["grad_norm"])], dtype=float)
        summary_rows.append(
            {
                "alpha": float(alpha),
                "c_bar_mean": float(np.nanmean(c_vals)),
                "c_bar_std": float(np.nanstd(c_vals, ddof=1)) if len(c_vals) > 1 else 0.0,
                "forward_finite_frac": float(np.mean([r["forward_finite"] for r in rows])),
                "grad_probe_ok_frac": float(np.mean([r["grad_probe_ok"] for r in rows])),
                "grad_norm_median": float(np.median(g_vals)) if g_vals.size else float("nan"),
            }
        )
    save_csv(out / "exp1_alpha_summary.csv", summary_rows)

    # 经验稳定带：前向且 grad 探针均好的 alpha
    stable = [s for s in summary_rows if s["forward_finite_frac"] >= 0.99 and s["grad_probe_ok_frac"] >= 0.99]
    c_crit_note = "n/a"
    if stable:
        cm = np.median([s["c_bar_mean"] for s in stable])
        c_crit_note = f"median_c_bar_in_stable_band={cm:.4f} (empirical, not a theorem)"

    apply_paper_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.8))
    xs = [s["alpha"] for s in summary_rows]
    ax0.plot(xs, [s["c_bar_mean"] for s in summary_rows], "o-", ms=3)
    ax0.set_xscale("log")
    ax0.set_xlabel(r"Input scale $\alpha$")
    ax0.set_ylabel(r"$\bar c_{\mathrm{geom}}$ (mean over seeds)")
    ax0.set_title("Geometric mean layer ratio vs input scale")
    ax0.grid(True, which="both", ls=":", alpha=0.5)

    gmed = [s["grad_norm_median"] for s in summary_rows]
    ax1.plot(xs, gmed, "s-", color="C1", ms=3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Input scale $\alpha$")
    ax1.set_ylabel("Median grad norm (probe)")
    ax1.axhline(args.grad_explode_thresh, color="gray", ls="--", lw=0.9, label="explode thresh")
    ax1.set_title("Gradient probe vs $\\alpha$")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    fig.suptitle(f"Exp1 ReLU stack depth={args.dim} L={args.depth} | {c_crit_note}", fontsize=9, y=1.05)
    fig.tight_layout()
    fig.savefig(out / "fig_exp1_alpha_cbar_grad.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig_exp1_alpha_cbar_grad.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Exp1 wrote {out} | {c_crit_note}")


# --- Exp2 -----------------------------------------------------------------


def run_exp2(args: argparse.Namespace) -> None:
    out = OUTPUT_ROOT / "exp2_physical"
    out.mkdir(parents=True, exist_ok=True)

    configs: list[tuple[str, Callable[[], nn.Module]]] = [
        ("gelu", lambda: nn.GELU()),
        ("relu", lambda: nn.ReLU()),
        ("light_depth_t2.0", lambda: LightDepthAct(2.0)),
    ]

    raw_layer: list[dict] = []
    raw_c: list[dict] = []

    for act_id, factory in configs:
        for depth in args.depths_exp2:
            for si in range(args.num_seeds_exp2):
                set_seed(BASE_SEED + si)
                layers = nn.ModuleList([ScatteringLayer(args.dim_exp2, factory()) for _ in range(depth)])
                x0 = torch.randn(args.dim_exp2, args.dim_exp2)
                with torch.no_grad():
                    chain, _ = forward_stack_nuclear_chain(layers, x0)
                c_list, c_bar = nuclear_chain_to_c(chain)
                for ell, n in enumerate(chain):
                    raw_layer.append(
                        {
                            "activation_id": act_id,
                            "depth": depth,
                            "seed": BASE_SEED + si,
                            "layer_index": ell,
                            "nuclear_norm": n,
                            "ln_nuclear": float(math.log(n) if n > 0 else float("nan")),
                            "c_bar_geom": c_bar,
                        }
                    )
                for j, cv in enumerate(c_list):
                    if cv is not None:
                        raw_c.append(
                            {
                                "activation_id": act_id,
                                "depth": depth,
                                "seed": BASE_SEED + si,
                                "layer_transition": j + 1,
                                "c_t": float(cv),
                                "abs_c_minus_1": float(abs(cv - 1.0)),
                            }
                        )

    save_csv(out / "exp2_lnuclear_raw.csv", raw_layer)
    save_csv(out / "exp2_c_raw.csv", raw_c)

    # 图：每个 depth 一张 — c_t 箱线（按层）
    z = 1.96
    for depth in args.depths_exp2:
        apply_paper_style()
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for ax, (act_id, _) in zip(axes, configs):
            data = [
                [r["c_t"] for r in raw_c if r["activation_id"] == act_id and r["depth"] == depth and r["layer_transition"] == li + 1]
                for li in range(depth)
            ]
            pos = list(range(1, depth + 1))
            ax.boxplot(data, positions=pos, widths=0.35, showfliers=False)
            ax.axhline(1.0, color="gray", ls="--", lw=0.9)
            ax.set_xlabel(r"Layer transition $\ell$")
            ax.set_ylabel(r"$c_\ell$")
            ax.set_title(act_id.replace("_", " "))
            ax.grid(True, ls=":", alpha=0.5)
        fig.suptitle(
            f"Exp2 $c_\\ell$ distribution (ScatteringLayer stack, depth={depth}, seeds={args.num_seeds_exp2})",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(out / f"fig_exp2_c_boxplot_depth{depth}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / f"fig_exp2_c_boxplot_depth{depth}.pdf", bbox_inches="tight")
        plt.close(fig)

    # ln(N) 均值 ± 95% CI 按层
    for depth in args.depths_exp2:
        apply_paper_style()
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for act_id, _ in configs:
            layers_idx = list(range(depth + 1))
            means, lo, hi = [], [], []
            for ell in layers_idx:
                vals = np.array(
                    [
                        r["ln_nuclear"]
                        for r in raw_layer
                        if r["activation_id"] == act_id and r["depth"] == depth and r["layer_index"] == ell
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
            ax.fill_between(layers_idx, lo, hi, alpha=0.2)
            ax.plot(layers_idx, means, "o-", label=act_id)
        ax.set_xlabel(r"Layer index $\ell$ (0 = input)")
        ax.set_ylabel(r"$\ln(N_\ell)$")
        ax.set_title(f"Exp2 nuclear norm (log) vs depth={depth}")
        ax.legend(fontsize=7)
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out / f"fig_exp2_ln_N_depth{depth}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / f"fig_exp2_ln_N_depth{depth}.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Exp2 wrote {out}")


# --- Exp3 -----------------------------------------------------------------


class MLPClassifier(nn.Module):
    """Flatten图像或向量；Linear→h；L 个 Linear+BN+Act；分类头。"""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_layers: int,
        num_classes: int,
        activation_name: str,
        **act_kwargs,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.act0 = build_activation(activation_name, **act_kwargs)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "lin": nn.Linear(hidden, hidden),
                        "bn": nn.BatchNorm1d(hidden),
                        "act": build_activation(activation_name, **act_kwargs),
                    }
                )
            )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor, return_hiddens: bool = False):
        x = x.flatten(1)
        h = self.act0(self.in_proj(x))
        hiddens = [h]
        for blk in self.blocks:
            h = blk["act"](blk["bn"](blk["lin"](h)))
            hiddens.append(h)
        logits = self.head(h)
        if return_hiddens:
            return logits, hiddens
        return logits


def hidden_list_to_norms(hiddens: list[torch.Tensor]) -> tuple[list[float], list[float | None], float]:
    """hiddens: 每层 (B,h)。逐样本 analyze_matrix(1,h)，再对核范数取 batch 平均；c 用相邻平均核范数之比。"""
    nucs: list[float] = []
    for h in hiddens:
        b = h.shape[0]
        nv = []
        for bi in range(b):
            _, _, n, _ = analyze_matrix(h[bi : bi + 1])
            nv.append(n)
        nucs.append(float(np.mean(nv)))
    c_list, c_bar = nuclear_chain_to_c(nucs)
    return nucs, c_list, c_bar


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


def run_exp3(args: argparse.Namespace) -> None:
    out = OUTPUT_ROOT / "exp3_cls"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.dataset == "cifar10":
        try:
            from torchvision import datasets, transforms
        except ImportError as e:
            raise SystemExit(f"需要 torchvision 以使用 CIFAR-10: {e}") from e

        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
        test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
        in_dim = 3 * 32 * 32
        num_classes = 10
    else:
        #合成多类（优先 sklearn；否则简单高斯混合）
        set_seed(BASE_SEED)
        try:
            from sklearn.datasets import make_classification

            X, y = make_classification(
                n_samples=args.synth_samples,
                n_features=args.synth_features,
                n_informative=max(8, args.synth_features // 2),
                n_redundant=4,
                n_classes=10,
                n_clusters_per_class=1,
                random_state=BASE_SEED,
            )
        except ImportError:
            n = args.synth_samples
            d = args.synth_features
            X = torch.randn(n, d)
            y = torch.randint(0, 10, (n,))
            for c in range(10):
                mask = y == c
                if mask.any():
                    X[mask] += 0.35 * torch.randn(1, d) * c
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        n = X.shape[0]
        perm = torch.randperm(n)
        X, y = X[perm], y[perm]
        n_tr = int(0.8 * n)
        train_ds = torch.utils.data.TensorDataset(X[:n_tr], y[:n_tr])
        test_ds = torch.utils.data.TensorDataset(X[n_tr:], y[n_tr:])
        in_dim = args.synth_features
        num_classes = 10

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    depth_list = [int(x) for x in args.depths_exp3.split(",")]
    summary: list[dict] = []
    layer_raw: list[dict] = []

    for num_layers in depth_list:
        set_seed(BASE_SEED + num_layers)
        model = MLPClassifier(
            in_dim,
            args.hidden,
            num_layers,
            num_classes,
            args.cls_activation,
            **{},
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val = 0.0
        for ep in range(args.epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()

            model.eval()
            correct = 0
            tot = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb).argmax(1)
                    correct += (pred == yb).sum().item()
                    tot += yb.numel()
            acc = correct / max(tot, 1)
            best_val = max(best_val, acc)

        # 探针：一批测试数据上的 N、c
        model.eval()
        xb, _ = next(iter(test_loader))
        xb = xb.to(device)
        with torch.no_grad():
            _, hiddens = model(xb, return_hiddens=True)
        nucs, c_list, c_bar = hidden_list_to_norms(hiddens)
        cs = [abs(c - 1.0) for c in c_list if c is not None and math.isfinite(c)]
        mean_abs_c1 = float(np.mean(cs)) if cs else float("nan")

        summary.append(
            {
                "num_layers": num_layers,
                "hidden": args.hidden,
                "best_val_acc": best_val,
                "representation": "post_act_hidden_Bxh_svd_per_sample_mean",
                "c_bar_geom_probe": c_bar,
                "mean_abs_c_minus_1": mean_abs_c1,
                "nuclear_final_probe": nucs[-1] if nucs else float("nan"),
            }
        )
        for ell, n in enumerate(nucs):
            ct = c_list[ell] if ell < len(c_list) else None
            layer_raw.append(
                {
                    "num_layers": num_layers,
                    "layer_index": ell,
                    "nuclear_norm": n,
                    "c_t": ct,
                }
            )

    save_csv(out / "exp3_depth_summary.csv", summary)
    save_csv(out / "exp3_depth_layer_raw.csv", layer_raw)

    depths = np.array([s["num_layers"] for s in summary], dtype=float)
    accs = np.array([s["best_val_acc"] for s in summary], dtype=float)
    mc = np.array([s["mean_abs_c_minus_1"] for s in summary], dtype=float)
    rho_acc_depth = spearman_rho(depths, accs)
    rho_acc_mc = spearman_rho(mc, accs)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(depths, accs, "o-", label="best val acc")
    ax.set_xlabel("Depth (MLP blocks)")
    ax.set_ylabel("Accuracy")
    ax2 = ax.twinx()
    ax2.plot(depths, [s["c_bar_geom_probe"] for s in summary], "s--", color="C1", alpha=0.8, label=r"$\bar c_{\mathrm{geom}}$ (probe)")
    ax.set_title(
        f"Exp3 Acc vs depth | Spearman(acc,depth)={rho_acc_depth:.3f}, Spearman(acc,|c-1|)={rho_acc_mc:.3f}\n(observational)"
    )
    ax.grid(True, ls=":", alpha=0.5)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_exp3_acc_vs_depth.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig_exp3_acc_vs_depth.pdf", bbox_inches="tight")
    plt.close(fig)

    with (out / "exp3_spearman_note.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"Spearman rho(acc, depth) = {rho_acc_depth}\n"
            f"Spearman rho(acc, mean| c-1 |) = {rho_acc_mc}\n"
            "These are observational associations, not causal claims.\n"
        )

    print(f"Exp3 wrote {out}")


# --- CLI ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Script 6: scattering criticality experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("exp1", help="Alpha sweep on ReLU scattering stack")
    p1.add_argument("--dim", type=int, default=16)
    p1.add_argument("--depth", type=int, default=8)
    p1.add_argument("--alpha_min", type=float, default=0.01)
    p1.add_argument("--alpha_max", type=float, default=100.0)
    p1.add_argument("--alpha_steps", type=int, default=40)
    p1.add_argument("--num_seeds", type=int, default=8)
    p1.add_argument("--weight_seed", type=int, default=999)
    p1.add_argument("--grad_explode_thresh", type=float, default=1e6)
    p1.set_defaults(func=run_exp1)

    p2 = sub.add_parser("exp2", help="Physical activations vs ReLU (c spread, ln N)")
    p2.add_argument("--dim_exp2", type=int, default=16)
    p2.add_argument("--depths_exp2", type=int, nargs="+", default=[8, 12])
    p2.add_argument("--num_seeds_exp2", type=int, default=20)
    p2.set_defaults(func=run_exp2)

    p3 = sub.add_parser("exp3", help="Classification: sweep depth, record Acc and probe N,c")
    p3.add_argument("--dataset", choices=["cifar10", "synthetic"], default="synthetic")
    p3.add_argument("--synth_samples", type=int, default=4000)
    p3.add_argument("--synth_features", type=int, default=64)
    p3.add_argument("--hidden", type=int, default=256)
    p3.add_argument("--depths_exp3", type=str, default="4,6,8,10,12")
    p3.add_argument("--epochs", type=int, default=35)
    p3.add_argument("--batch", type=int, default=128)
    p3.add_argument("--lr", type=float, default=1e-3)
    p3.add_argument("--cls_activation", type=str, default="relu")
    p3.add_argument("--device", type=str, default="cpu")
    p3.set_defaults(func=run_exp3)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
