#基础实验・激活函数可视化 + 矩阵不变量（原始 Code1 完整版）
#实验目的：直观展示激活函数的漫反射结构变化，定量验证秩 / 谱范数 / 核范数 / 条件数的变化规律
#输出：三联热力图 figure_activation_triptych + 基础定量 CSV；期刊表见 paper_assets（若另行导出）

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from paper_style import apply_paper_style


BASE_SEED = 42
MATRIX_DIM = 16
DEVICE = "cpu"
NUM_SEEDS = 20
RANK_THRESHOLD = 1e-6
COND_EPS = 1e-9
OUTPUT_DIR = Path("results_script1")
PAPER_DIR = Path("paper_assets")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def analyze_matrix(M: torch.Tensor) -> tuple[int, float, float, float]:
    s = np.linalg.svd(M.detach().cpu().numpy(), compute_uv=False)
    rank = int(np.sum(s > RANK_THRESHOLD))
    spectral = float(s[0]) if rank > 0 else 0.0
    nuclear = float(np.sum(s))
    cond = float(s[0] / s[-1]) if (rank > 0 and s[-1] > COND_EPS) else float("inf")
    return rank, spectral, nuclear, cond


def summarize_records(records: list[dict]) -> list[dict]:
    summary = []
    z = 1.96
    for name in sorted({r["activation"] for r in records}):
        rows = [r for r in records if r["activation"] == name]
        n = len(rows)
        rank_values = np.array([r["rank"] for r in rows], dtype=float)
        spectral_values = np.array([r["spectral_norm"] for r in rows], dtype=float)
        nuclear_values = np.array([r["nuclear_norm"] for r in rows], dtype=float)
        finite_cond = np.array([r["condition_number"] for r in rows if np.isfinite(r["condition_number"])], dtype=float)
        rank_drop_ratio = float(np.mean(rank_values < MATRIX_DIM))
        cond_p50 = float(np.median(finite_cond)) if finite_cond.size > 0 else float("inf")
        cond_p90 = float(np.quantile(finite_cond, 0.9)) if finite_cond.size > 0 else float("inf")
        nuc_mean = float(np.mean(nuclear_values))
        nuc_std = float(np.std(nuclear_values, ddof=1)) if n > 1 else 0.0
        nuc_ci = z * nuc_std / np.sqrt(n) if n > 1 else 0.0

        summary.append(
            {
                "activation": name,
                "n": n,
                "rank_mean": float(np.mean(rank_values)),
                "rank_std": float(np.std(rank_values, ddof=1)) if n > 1 else 0.0,
                "spectral_mean": float(np.mean(spectral_values)),
                "spectral_std": float(np.std(spectral_values, ddof=1)) if n > 1 else 0.0,
                "nuclear_mean": nuc_mean,
                "nuclear_std": nuc_std,
                "nuclear_ci95_halfwidth": float(nuc_ci),
                "cond_median_finite": cond_p50,
                "cond_p90_finite": cond_p90,
                "rank_drop_ratio": rank_drop_ratio,
            }
        )
    return summary


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def visualize_activation_effect() -> None:
    class TinyActivationTest(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(MATRIX_DIM, MATRIX_DIM)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    apply_paper_style()
    set_seed(BASE_SEED)
    model = TinyActivationTest().to(DEVICE).eval()
    x = torch.randn(1, MATRIX_DIM, MATRIX_DIM, device=DEVICE)
    with torch.no_grad():
        z = model(x)
        z_relu = torch.nn.functional.relu(z)
        z_gelu = torch.nn.functional.gelu(z)

    fig = plt.figure(figsize=(10.5, 3.4), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    titles = (
        r"Pre-activation $\mathbf{Z}$ (linear output)",
        r"$\mathrm{ReLU}(\mathbf{Z})$",
        r"$\mathrm{GELU}(\mathbf{Z})$",
    )
    for ax, tensor, title in zip(axes, (z, z_relu, z_gelu), titles):
        im = ax.imshow(tensor[0].cpu().numpy(), cmap="RdBu", vmin=-2, vmax=2)
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, cax=cax, label=r"Entry value")
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for base in (OUTPUT_DIR, PAPER_DIR):
        base.mkdir(parents=True, exist_ok=True)
        fig.savefig(base / "figure_activation_triptych.png", dpi=300, bbox_inches="tight")
        fig.savefig(base / "figure_activation_triptych.pdf", bbox_inches="tight")
    plt.close(fig)


def analyze_matrix_invariants() -> None:
    activations = {"none": nn.Identity(), "relu": nn.ReLU(), "gelu": nn.GELU()}
    records: list[dict] = []
    for seed in range(NUM_SEEDS):
        set_seed(BASE_SEED + seed)
        X = torch.randn(MATRIX_DIM, MATRIX_DIM, device=DEVICE)
        fc = nn.Linear(MATRIX_DIM, MATRIX_DIM, bias=False, device=DEVICE).eval()
        with torch.no_grad():
            X_lin = fc(X)
        for name, act in activations.items():
            with torch.no_grad():
                X_act = act(X_lin)
            rank, spectral, nuclear, cond = analyze_matrix(X_act)
            records.append(
                {
                    "seed": BASE_SEED + seed,
                    "activation": name,
                    "rank": rank,
                    "spectral_norm": spectral,
                    "nuclear_norm": nuclear,
                    "condition_number": cond,
                }
            )

    summary = summarize_records(records)
    save_csv(OUTPUT_DIR / "invariants_raw.csv", records)
    save_csv(OUTPUT_DIR / "invariants_summary.csv", summary)

    print("=" * 90)
    print("activation | rank(mean±std) | spectral(mean±std) | nuclear(mean±std) | cond_p50 | rank_drop")
    print("=" * 90)
    for row in summary:
        print(
            f"{row['activation']:>9s} | "
            f"{row['rank_mean']:.2f}±{row['rank_std']:.2f} | "
            f"{row['spectral_mean']:.3f}±{row['spectral_std']:.3f} | "
            f"{row['nuclear_mean']:.3f}±{row['nuclear_std']:.3f} | "
            f"{row['cond_median_finite']:.2f} | "
            f"{row['rank_drop_ratio']:.2%}"
        )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Script1: activation visualization + matrix invariants ===")
    print(f"torch={torch.__version__}, numpy={np.__version__}, dim={MATRIX_DIM}, seeds={NUM_SEEDS}")
    visualize_activation_effect()
    analyze_matrix_invariants()
