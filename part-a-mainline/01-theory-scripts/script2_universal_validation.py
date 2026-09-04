import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


BASE_SEED = 42
TEST_DIMENSIONS = [8, 16, 32, 64]
TEST_ACTIVATIONS = {
    "none": nn.Identity(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "elu": nn.ELU(),
}
NUM_SEEDS = 20
RANK_THRESHOLD = 1e-6
COND_EPS = 1e-9
OUTPUT_DIR = Path("results_script2")
PAPER_DIR = Path("paper_assets")
# 期刊表只展示两个典型宽度（全量扫描仍见 universal_summary.csv）
TABLE_EXPORT_DIMS: list[int] = [16, 64]


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


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict]) -> list[dict]:
    summary: list[dict] = []
    z = 1.96
    keys = sorted({(r["dim"], r["activation"]) for r in records})
    for dim, act_name in keys:
        rows = [r for r in records if r["dim"] == dim and r["activation"] == act_name]
        n = len(rows)
        rank_vals = np.array([r["rank"] for r in rows], dtype=float)
        spectral_vals = np.array([r["spectral_norm"] for r in rows], dtype=float)
        nuclear_vals = np.array([r["nuclear_norm"] for r in rows], dtype=float)
        cond_vals = np.array([r["condition_number"] for r in rows if np.isfinite(r["condition_number"])], dtype=float)
        nuc_std = float(np.std(nuclear_vals, ddof=1)) if n > 1 else 0.0
        summary.append(
            {
                "dim": dim,
                "activation": act_name,
                "n": n,
                "rank_mean": float(np.mean(rank_vals)),
                "rank_std": float(np.std(rank_vals, ddof=1)) if n > 1 else 0.0,
                "spectral_mean": float(np.mean(spectral_vals)),
                "spectral_std": float(np.std(spectral_vals, ddof=1)) if n > 1 else 0.0,
                "nuclear_mean": float(np.mean(nuclear_vals)),
                "nuclear_std": nuc_std,
                "nuclear_ci95_halfwidth": float(z * nuc_std / np.sqrt(n)) if n > 1 else 0.0,
                "cond_median_finite": float(np.median(cond_vals)) if cond_vals.size else float("inf"),
                "cond_p90_finite": float(np.quantile(cond_vals, 0.9)) if cond_vals.size else float("inf"),
                "rank_drop_ratio": float(np.mean(rank_vals < dim)),
            }
        )
    return summary


def export_universal_booktabs(summary: list[dict]) -> None:
    """期刊三线表：谱范数、核范数、条件数等（表注自洽，不引用其他脚本）；仅 TABLE_EXPORT_DIMS。"""
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out = PAPER_DIR / "table_script2_universal_booktabs.tex"
    dims = TABLE_EXPORT_DIMS
    cols_per_n = 4
    acts_order = ["none", "relu", "gelu", "silu", "mish", "elu"]
    act_label = {
        "none": r"$\mathrm{Linear}$",
        "relu": r"$\mathrm{ReLU}$",
        "gelu": r"$\mathrm{GELU}$",
        "silu": r"$\mathrm{SiLU}$",
        "mish": r"$\mathrm{Mish}$",
        "elu": r"$\mathrm{ELU}$",
    }
    tab_spec = "l" + "c" * (cols_per_n * len(dims))
    lines = [
        r"% Auto-generated from results_script2/universal_summary.csv",
        r"% CSV: full $n\in\{8,16,32,64\}$; this table: $n\in\{16,64\}$ only.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Summaries of the feature matrix $\mathbf{M}=\sigma(W\mathbf{X})\in\mathbb{R}^{n\times n}$, "
        r"where $\mathbf{X}$ has i.i.d.\ standard normal entries, $W$ is one random linear map per random seed, "
        r"and $\sigma$ is the listed pointwise activation (identity for $\mathrm{Linear}$). "
        r"\emph{Rank} counts singular values above a fixed numerical threshold. "
        r"The \emph{spectral norm} $\|\mathbf{M}\|_{2}=\sigma_{\max}(\mathbf{M})$ is the largest singular value; "
        r"the \emph{nuclear norm} $\|\mathbf{M}\|_{\ast}=\sum_i \sigma_i(\mathbf{M})$ is the sum of singular values. "
        r"These two norms and rank are reported as mean $\pm$ standard deviation over $20$ seeds. "
        r"The \emph{median condition number} $\mathrm{med}\,\kappa$ uses "
        r"$\kappa(\mathbf{M})=\sigma_{\max}(\mathbf{M})/\sigma_{\min}(\mathbf{M})$ on realizations with finite $\kappa$. "
        r"Representative widths $n\in\{16,64\}$; additional $n$ are in \texttt{universal\_summary.csv}.}",
        r"\label{tab:universal-sweep}",
        r"\small",
        rf"\begin{{tabular}}{{{tab_spec}}}",
        r"\toprule",
    ]
    header1 = "".join(
        [rf" & \multicolumn{{{cols_per_n}}}{{c}}{{$n={d}$}}" for d in dims]
    )
    lines.append(header1 + r" \\")
    cmid_parts = []
    c0 = 2
    for _ in dims:
        c1 = c0 + cols_per_n - 1
        cmid_parts.append(rf"\cmidrule(lr){{{c0}-{c1}}}")
        c0 = c1 + 1
    lines.append("".join(cmid_parts) + r" \\")
    sub_cols = (
        r" & $\mathrm{rank}$ & $\|\mathbf{M}\|_2$ & $\|\mathbf{M}\|_{\ast}$ & $\mathrm{med}\,\kappa$"
    ) * len(dims)
    lines.append(r"$\mathrm{Activation}$" + sub_cols + r" \\")
    lines.append(r"\midrule")
    lookup = {(r["dim"], r["activation"]): r for r in summary}
    for act in acts_order:
        row = [act_label[act]]
        for d in dims:
            rec = lookup.get((d, act))
            if rec is None:
                row.extend(["--"] * cols_per_n)
                continue
            rk = f"${rec['rank_mean']:.2f} \\pm {rec['rank_std']:.2f}$"
            sp = f"${rec['spectral_mean']:.2f} \\pm {rec['spectral_std']:.2f}$"
            nuc = f"${rec['nuclear_mean']:.2f} \\pm {rec['nuclear_std']:.2f}$"
            kap = rec["cond_median_finite"]
            kap_s = f"{kap:.2f}" if np.isfinite(kap) else r"$\infty$"
            row.extend([rk, sp, nuc, kap_s])
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def universal_validation() -> None:
    records: list[dict] = []
    print("=== Universal validation: dimensions + activations ===")
    for dim in TEST_DIMENSIONS:
        for seed in range(NUM_SEEDS):
            set_seed(BASE_SEED + seed)
            X = torch.randn(dim, dim)
            fc = nn.Linear(dim, dim, bias=False).eval()
            with torch.no_grad():
                X_lin = fc(X)
            for name, act in TEST_ACTIVATIONS.items():
                with torch.no_grad():
                    X_act = act(X_lin)
                rank, spectral, nuclear, cond = analyze_matrix(X_act)
                records.append(
                    {
                        "seed": BASE_SEED + seed,
                        "dim": dim,
                        "activation": name,
                        "rank": rank,
                        "spectral_norm": spectral,
                        "nuclear_norm": nuclear,
                        "condition_number": cond,
                    }
                )

    summary = summarize(records)
    save_csv(OUTPUT_DIR / "universal_raw.csv", records)
    save_csv(OUTPUT_DIR / "universal_summary.csv", summary)
    export_universal_booktabs(summary)

    for dim in TEST_DIMENSIONS:
        print(f"\n=== dim={dim} ===")
        dim_rows = [r for r in summary if r["dim"] == dim]
        for row in dim_rows:
            print(
                f"{row['activation']:6s} | "
                f"rank {row['rank_mean']:.2f}±{row['rank_std']:.2f} | "
                f"spec {row['spectral_mean']:.2f}±{row['spectral_std']:.2f} | "
                f"nuc {row['nuclear_mean']:.2f}±{row['nuclear_std']:.2f} | "
                f"cond_p50 {row['cond_median_finite']:.2f}"
            )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Script2: universal validation ===")
    print(f"torch={torch.__version__}, numpy={np.__version__}, seeds={NUM_SEEDS}, dims={TEST_DIMENSIONS}")
    universal_validation()
