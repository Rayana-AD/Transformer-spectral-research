# 脚本 3：多层散射累积效应（多激活 + LightDepth + 物理光深 τ(z)：z·exp(-τ(z))）
# 实验目的：对比 Smish / LightDepth / Beer–Lambert 型散射衰减等在深度上的核范数与条件数演化
# 仅重画论文 3x3 条件数组图：python script3_depth_scattering.py --export-grid-only
# 右图曾“断在 layer 8”的原因：深层条件数对许多 seed 为 inf，旧代码用 inf 汇总后在线性 y 轴上无法绘制后续层。
# 现用对数 y 轴 + nan 断点 + 有限 seed 比例标注。
import csv
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from activation_extras import LightDepthAct, ScatteringOpticalDepthAct, Smish
from paper_style import apply_paper_style

BASE_SEED = 42
DIM = 16
DEPTHS = [2, 4, 8, 12]
NUM_SEEDS = 20
RANK_THRESHOLD = 1e-6
COND_EPS = 1e-9
OUTPUT_DIR = Path("results_script3")
PAPER_DIR = Path("paper_assets")

# (CSV/文件名 id, 每层新建激活的工厂)
# t=1 的 LightDepth 与 Smish 在映射上等价，但论文图中仍保留「light-depth, t=1」条目以便与 Smish 对照排版。
ACTIVATION_CONFIGS: list[tuple[str, Callable[[], nn.Module]]] = [
    ("smish", lambda: Smish()),
    ("light_depth_t1.0", lambda: LightDepthAct(1.0)),
    ("light_depth_t0.5", lambda: LightDepthAct(0.5)),
    ("light_depth_t2.0", lambda: LightDepthAct(2.0)),
    ("light_depth_t4.0", lambda: LightDepthAct(4.0)),
    # 物理衰减：z为线性层输出；τ(z)≥0；y = z·exp(-τ)。β 控制光深对 z 的灵敏度。
    ("phys_tau_softplus_z_b0.5", lambda: ScatteringOpticalDepthAct(0.5, "softplus_z")),
    ("phys_tau_softplus_z_b1.0", lambda: ScatteringOpticalDepthAct(1.0, "softplus_z")),
    ("phys_tau_softplus_z2_b1.0", lambda: ScatteringOpticalDepthAct(1.0, "softplus_z2")),
    ("relu", lambda: nn.ReLU()),
    ("gelu", lambda: nn.GELU()),
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class ScatteringLayer(nn.Module):
    def __init__(self, dim: int, act: nn.Module) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.act = act

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.lin(x)
        return z, self.act(z)


def analyze_matrix(M: torch.Tensor) -> tuple[int, float, float, float]:
    s = np.linalg.svd(M.detach().cpu().numpy(), compute_uv=False)
    rank = int(np.sum(s > RANK_THRESHOLD))
    spectral = float(s[0]) if rank > 0 else 0.0
    nuclear = float(np.sum(s))
    cond = float(s[0] / s[-1]) if (rank > 0 and s[-1] > COND_EPS) else float("inf")
    return rank, spectral, nuclear, cond


def load_summary_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                if v == "" or k == "activation_id":
                    continue
                try:
                    r[k] = float(v) if any(c in v for c in ".eE") else int(v)
                except ValueError:
                    if v.lower() in ("nan", "inf"):
                        r[k] = float(v)
            rows.append(r)
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict]) -> list[dict]:
    out = []
    z = 1.96
    keys = sorted({(r["activation_id"], r["depth"], r["layer"]) for r in records})
    for activation_id, depth, layer in keys:
        rows = [r for r in records if r["activation_id"] == activation_id and r["depth"] == depth and r["layer"] == layer]
        n = len(rows)
        nuc = np.array([r["nuclear_norm"] for r in rows], dtype=float)
        cond_all = np.array([r["condition_number"] for r in rows], dtype=float)
        finite_mask = np.isfinite(cond_all) & (cond_all > 0)
        cond_finite = cond_all[finite_mask]
        rank = np.array([r["rank"] for r in rows], dtype=float)
        nuc_std = float(np.std(nuc, ddof=1)) if n > 1 else 0.0

        cond_finite_frac = float(np.mean(finite_mask))
        if cond_finite.size:
            cond_median_finite = float(np.median(cond_finite))
            cond_p90_finite = float(np.quantile(cond_finite, 0.9))
            cond_median_log10 = float(np.median(np.log10(cond_finite)))
        else:
            cond_median_finite = float("nan")
            cond_p90_finite = float("nan")
            cond_median_log10 = float("nan")

        out.append(
            {
                "activation_id": activation_id,
                "depth": depth,
                "layer": layer,
                "n": n,
                "rank_mean": float(np.mean(rank)),
                "rank_std": float(np.std(rank, ddof=1)) if n > 1 else 0.0,
                "nuclear_mean": float(np.mean(nuc)),
                "nuclear_std": nuc_std,
                "nuclear_ci95_halfwidth": float(z * nuc_std / np.sqrt(n)) if n > 1 else 0.0,
                "cond_finite_frac": cond_finite_frac,
                "cond_median_finite": cond_median_finite,
                "cond_p90_finite": cond_p90_finite,
                "cond_median_log10": cond_median_log10,
            }
        )
    return out


CONDITION_GRID_ORDER: list[list[str]] = [
    ["gelu", "relu", "smish"],
    ["light_depth_t1.0", "light_depth_t2.0", "light_depth_t4.0"],
]


def activation_panel_title_latex(activation_id: str) -> str:
    return {
        "gelu": r"$\mathrm{GELU}$",
        "relu": r"$\mathrm{ReLU}$",
        "smish": r"$\mathrm{Smish}$",
        "light_depth_t1.0": r"Light-depth, $t=1$",
        "light_depth_t2.0": r"Light-depth, $t=2$",
        "light_depth_t4.0": r"Light-depth, $t=4$",
        "phys_tau_softplus_z_b0.5": r"$z\mathrm{e}^{-\tau}$, $\tau=\mathrm{softplus}(0.5 z)$",
        "phys_tau_softplus_z_b1.0": r"$z\mathrm{e}^{-\tau}$, $\tau=\mathrm{softplus}(z)$",
        "phys_tau_softplus_z2_b1.0": r"$z\mathrm{e}^{-\tau}$, $\tau=\mathrm{softplus}(z^2)$",
    }.get(activation_id, activation_id.replace("_", r"\_"))


def export_depth8_condition_grid(summary: list[dict], depth: int = 8) -> None:
    """论文用 $2\times3$：仅条件数子图（对数主轴 + 有限样本比例次轴）。"""
    apply_paper_style()
    nrows = len(CONDITION_GRID_ORDER)
    ncols = len(CONDITION_GRID_ORDER[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 6.2), sharex=True)
    y_left_label = r"Median $\kappa$"
    y_right_label = r"Fraction finite $\kappa$"
    x_label = r"Layer $\ell$"
    for i in range(nrows):
        for j in range(ncols):
            aid = CONDITION_GRID_ORDER[i][j]
            ax = axes[i, j]
            rows = sorted(
                [r for r in summary if r["depth"] == depth and r["activation_id"] == aid],
                key=lambda r: r["layer"],
            )
            if not rows:
                ax.set_visible(False)
                continue
            layers = [r["layer"] for r in rows]
            cond_med = [r["cond_median_finite"] for r in rows]
            cond_frac = [r["cond_finite_frac"] for r in rows]
            cond_plot = [c if np.isfinite(c) else np.nan for c in cond_med]
            ax.plot(layers, cond_plot, "o-", color="C1", markersize=4)
            ax.set_yscale("log")
            ax.set_xticks(list(range(1, depth + 1)))
            ax.grid(True, which="both", ls=":", alpha=0.45)
            ax.set_title(activation_panel_title_latex(aid), fontsize=10)
            if j == 0:
                ax.set_ylabel(y_left_label)
            if i == nrows - 1:
                ax.set_xlabel(x_label)
            ax2 = ax.twinx()
            ax2.plot(layers, cond_frac, "s--", color="gray", alpha=0.6, markersize=3)
            ax2.set_ylim(-0.05, 1.05)
            if j == ncols - 1:
                ax2.set_ylabel(y_right_label, color="gray", fontsize=9)
                ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
            else:
                ax2.set_yticklabels([])
    fig.tight_layout()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_DIR / "figure_script3_depth8_condition_grid.png", dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_DIR / "figure_script3_depth8_condition_grid.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "depth8_condition_grid_paper.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PAPER_DIR / 'figure_script3_depth8_condition_grid.pdf'}")


def plot_depth_curves(depth: int, activation_id: str, summary: list[dict]) -> None:
    rows = sorted(
        [r for r in summary if r["depth"] == depth and r["activation_id"] == activation_id],
        key=lambda r: r["layer"],
    )
    if not rows:
        return
    layers = [r["layer"] for r in rows]
    nuclear = [r["nuclear_mean"] for r in rows]
    cond_med = [r["cond_median_finite"] for r in rows]
    cond_frac = [r["cond_finite_frac"] for r in rows]

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(layers, nuclear, "o-")
    axes[0].set_xlabel(r"Layer $\ell$")
    axes[0].set_ylabel(r"Mean $\|\mathbf{H}\|_{\ast}$")
    axes[0].set_title(rf"$\|\mathbf{{H}}\|_{{\ast}}$ mean ($N={depth}$)" + "\n" + activation_panel_title_latex(activation_id), fontsize=10)
    axes[0].set_xticks(list(range(1, depth + 1)))

    cond_plot = [c if np.isfinite(c) else np.nan for c in cond_med]
    axes[1].plot(layers, cond_plot, "o-", color="C1")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Layer $\ell$")
    axes[1].set_ylabel(r"Median $\kappa$ (finite runs)")
    axes[1].set_title(rf"$\kappa$ ($N={depth}$)" + "\n" + activation_panel_title_latex(activation_id), fontsize=10)
    axes[1].set_xticks(list(range(1, depth + 1)))
    axes[1].grid(True, which="both", ls=":", alpha=0.5)

    ax2 = axes[1].twinx()
    ax2.plot(layers, cond_frac, "s--", color="gray", alpha=0.65, markersize=4)
    ax2.set_ylabel(r"Fraction finite $\kappa$", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    safe = activation_id.replace(".", "p")
    fig.savefig(OUTPUT_DIR / f"depth_{depth}_{safe}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"depth_{depth}_{safe}.pdf", bbox_inches="tight")
    plt.close(fig)


def depth_scattering_experiment() -> None:
    records: list[dict] = []
    print("=== Multi-layer scattering: Smish vs LightDepth(t) vs baselines ===")
    for activation_id, act_factory in ACTIVATION_CONFIGS:
        print(f"\n-- activation_id={activation_id} --")
        for depth in DEPTHS:
            for seed in range(NUM_SEEDS):
                set_seed(BASE_SEED + seed)
                layers = nn.ModuleList([ScatteringLayer(DIM, act_factory()) for _ in range(depth)]).eval()
                x = torch.randn(DIM, DIM)
                with torch.no_grad():
                    for i, layer in enumerate(layers, start=1):
                        _, x = layer(x)
                        rank, spectral, nuclear, cond = analyze_matrix(x)
                        records.append(
                            {
                                "seed": BASE_SEED + seed,
                                "activation_id": activation_id,
                                "depth": depth,
                                "layer": i,
                                "rank": rank,
                                "spectral_norm": spectral,
                                "nuclear_norm": nuclear,
                                "condition_number": cond,
                            }
                        )

    summary = summarize(records)
    save_csv(OUTPUT_DIR / "depth_raw.csv", records)
    save_csv(OUTPUT_DIR / "depth_summary.csv", summary)

    for activation_id, _ in ACTIVATION_CONFIGS:
        for depth in DEPTHS:
            depth_rows = [r for r in summary if r["depth"] == depth and r["activation_id"] == activation_id]
            if not depth_rows:
                continue
            final = max(depth_rows, key=lambda r: r["layer"])
            cm = final["cond_median_finite"]
            cm_s = f"{cm:.2e}" if np.isfinite(cm) else "nan(all_inf?)"
            print(
                f"{activation_id:18s} depth={depth:2d} | final nuc={final['nuclear_mean']:.3f}±{final['nuclear_std']:.3f} | "
                f"cond_med={cm_s} | finite_frac={final['cond_finite_frac']:.0%}"
            )
            plot_depth_curves(depth, activation_id, summary)

    export_depth8_condition_grid(summary, depth=8)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--export-grid-only":
        summ = load_summary_csv(OUTPUT_DIR / "depth_summary.csv")
        export_depth8_condition_grid(summ, depth=8)
        raise SystemExit(0)
    print("=== Script3: depth scattering experiment ===")
    print(f"torch={torch.__version__}, numpy={np.__version__}, dim={DIM}, seeds={NUM_SEEDS}, depths={DEPTHS}")
    depth_scattering_experiment()
