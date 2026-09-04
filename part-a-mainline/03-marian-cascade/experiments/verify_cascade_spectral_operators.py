#!/usr/bin/env python3
"""Explicit, verifiable spectral calculations for value-path cascade operators.

Parallels script2 (activation universal sweep) but for linear smoothers S_spat, S_temp.
Produces CSV + markdown report for paper/rebuttal "Activation → Cascade" bridge.

Run:
  .venv_minepy/bin/python experiments/verify_cascade_spectral_operators.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tlb_value_smooth import apply_value_cascade  # noqa: E402

OUTPUT = _ROOT / "results_cascade_spectral_verify"
BASE_SEED = 42
NUM_SEEDS = 20
RANK_THRESHOLD = 1e-6
COND_EPS = 1e-9
SEQ_DIMS = [16, 64, 128]
FEAT_DIMS = [16, 64]
ALPHA_S = 0.1
ALPHA_T = 0.2

ACTIVATIONS = {
    "linear": nn.Identity(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def analyze_matrix(M: torch.Tensor | np.ndarray) -> Dict[str, float]:
    if isinstance(M, torch.Tensor):
        arr = M.detach().cpu().numpy()
    else:
        arr = M
    s = np.linalg.svd(arr, compute_uv=False)
    rank = int(np.sum(s > RANK_THRESHOLD))
    spectral = float(s[0]) if rank > 0 else 0.0
    nuclear = float(np.sum(s))
    cond = float(s[0] / s[-1]) if (rank > 0 and s[-1] > COND_EPS) else float("inf")
    fro = float(np.linalg.norm(arr, ord="fro"))
    return {
        "rank": rank,
        "spectral_norm": spectral,
        "nuclear_norm": nuclear,
        "condition_number": cond,
        "frobenius_norm": fro,
    }


def build_temporal_operator(
    S: int, alpha_t: float, mode: str = "circular_roll"
) -> np.ndarray:
    P = np.zeros((S, S), dtype=np.float64)
    if mode == "circular_roll":
        for i in range(S):
            P[i, (i - 1) % S] = 1.0
    elif mode == "causal_prev":
        P[0, 0] = 1.0
        for i in range(1, S):
            P[i, i - 1] = 1.0
    else:
        raise ValueError(mode)
    I = np.eye(S)
    return (1.0 - alpha_t) * I + alpha_t * P


def build_spatial_operator(S: int, alpha_s: float, mode: str = "spatial_global") -> np.ndarray:
    I = np.eye(S)
    if mode == "spatial_global":
        J = np.ones((S, S), dtype=np.float64) / float(S)
        return (1.0 - alpha_s) * I + alpha_s * J
    if mode == "prefix_mean":
        rows = []
        for t in range(S):
            row = np.zeros(S, dtype=np.float64)
            row[: t + 1] = 1.0 / float(t + 1)
            rows.append(row)
        ref = np.stack(rows, axis=0)
        return (1.0 - alpha_s) * I + alpha_s * ref
    raise ValueError(mode)


def operator_spectral_norm(S: np.ndarray) -> float:
    return float(np.linalg.svd(S, compute_uv=False)[0])


def temporal_frequency_gains(S: int, alpha_t: float) -> np.ndarray:
    gains = []
    for k in range(S):
        omega = 2.0 * math.pi * k / S
        lam = (1.0 - alpha_t) + alpha_t * np.exp(-1j * omega)
        gains.append(abs(lam))
    return np.array(gains, dtype=np.float64)


def apply_torch_cascade(V: torch.Tensor, alpha_s: float, alpha_t: float, order: str):
    v4 = V.view(1, 1, V.shape[0], V.shape[1])
    return apply_value_cascade(
        v4,
        alpha_s,
        alpha_t,
        order=order,  # type: ignore[arg-type]
        temporal_mode="circular_roll",
        spatial_mode="spatial_global",
    )


def verify_operator_matches_torch(S_len: int, D: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    set_seed(0)
    V = torch.randn(S_len, D)
    V_np = V.numpy()
    checks = [
        ("circular_roll", "spatial_global"),
        ("causal_prev", "prefix_mean"),
    ]
    for tmode, smode in checks:
        S_t = build_temporal_operator(S_len, ALPHA_T, tmode)
        S_s = build_spatial_operator(S_len, ALPHA_S, smode)
        v4 = V.view(1, 1, S_len, D)
        _, _, v_t_torch = apply_value_cascade(
            v4,
            0.0,
            ALPHA_T,
            order="spatial_first",
            temporal_mode=tmode,  # type: ignore[arg-type]
            spatial_mode=smode,  # type: ignore[arg-type]
        )
        _, v_s_torch, _ = apply_value_cascade(
            v4,
            ALPHA_S,
            0.0,
            order="spatial_first",
            temporal_mode=tmode,  # type: ignore[arg-type]
            spatial_mode=smode,  # type: ignore[arg-type]
        )
        rows.append(
            {
                "temporal_mode": tmode,
                "spatial_mode": smode,
                "max_err_temporal": float(
                    np.max(np.abs(S_t @ V_np - v_t_torch[0, 0].numpy()))
                ),
                "max_err_spatial": float(
                    np.max(np.abs(S_s @ V_np - v_s_torch[0, 0].numpy()))
                ),
            }
        )
    return rows


def verify_noncommutativity(S_len: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for tmode, smode in (
        ("circular_roll", "spatial_global"),
        ("causal_prev", "prefix_mean"),
    ):
        S_t = build_temporal_operator(S_len, ALPHA_T, tmode)
        S_s = build_spatial_operator(S_len, ALPHA_S, smode)
        comm = S_s @ S_t - S_t @ S_s
        rows.append(
            {
                "S_len": float(S_len),
                "temporal_mode": tmode,
                "spatial_mode": smode,
                "norm_S_temp": operator_spectral_norm(S_t),
                "norm_S_spat": operator_spectral_norm(S_s),
                "norm_commutator": float(np.linalg.norm(comm, ord=2)),
                "norm_S_spat_S_temp": operator_spectral_norm(S_s @ S_t),
                "norm_S_temp_S_spat": operator_spectral_norm(S_t @ S_s),
            }
        )
    return rows


def verify_attention_bound(S_len: int, D: int, n_trials: int = 50) -> Dict[str, float]:
    ratios = []
    for i in range(n_trials):
        set_seed(100 + i)
        V = np.random.randn(S_len, D)
        A = np.random.rand(S_len, S_len)
        A = A / A.sum(axis=-1, keepdims=True)
        S_t = build_temporal_operator(S_len, ALPHA_T)
        S_s = build_spatial_operator(S_len, ALPHA_S)
        Vp = S_t @ (S_s @ V)
        O = A @ Vp
        s_o = np.linalg.svd(O, compute_uv=False)
        s_v = np.linalg.svd(Vp, compute_uv=False)
        nuc_o = float(np.sum(s_o))
        nuc_v = float(np.sum(s_v))
        norm_a = float(np.linalg.svd(A, compute_uv=False)[0])
        if nuc_v > 1e-12:
            ratios.append(nuc_o / (norm_a * nuc_v + 1e-12))
    arr = np.array(ratios)
    return {
        "n_trials": n_trials,
        "max_ratio_nuc_O_over_normA_nucV": float(np.max(arr)),
        "mean_ratio": float(np.mean(arr)),
    }


def sweep_synthetic() -> List[Dict]:
    rows: List[Dict] = []
    for S_len in SEQ_DIMS:
        for D in FEAT_DIMS:
            for seed in range(NUM_SEEDS):
                set_seed(BASE_SEED + seed)
                V0 = torch.randn(S_len, D)
                W = torch.randn(D, D)
                Z = V0 @ W.T

                for act_name, act in ACTIVATIONS.items():
                    with torch.no_grad():
                        M = act(Z).numpy()
                    stats = analyze_matrix(M)
                    rows.append(
                        {
                            "family": "activation",
                            "S": S_len,
                            "D": D,
                            "seed": BASE_SEED + seed,
                            "stage": act_name,
                            **stats,
                        }
                    )

                S_t = build_temporal_operator(S_len, ALPHA_T)
                S_s = build_spatial_operator(S_len, ALPHA_S)
                stages = {
                    "raw": V0.numpy(),
                    "spatial_only": S_s @ V0.numpy(),
                    "temporal_only": S_t @ V0.numpy(),
                    "spatial_first": S_t @ (S_s @ V0.numpy()),
                    "temporal_first": S_s @ (S_t @ V0.numpy()),
                }
                for stage_name, M in stages.items():
                    stats = analyze_matrix(M)
                    rows.append(
                        {
                            "family": "cascade",
                            "S": S_len,
                            "D": D,
                            "seed": BASE_SEED + seed,
                            "stage": stage_name,
                            **stats,
                        }
                    )
    return rows


def summarize_sweep(rows: List[Dict]) -> List[Dict]:
    keys = sorted(
        {
            (r["family"], r["S"], r["D"], r["stage"])
            for r in rows
            if r["family"] in ("activation", "cascade")
        }
    )
    out: List[Dict] = []
    for family, S_len, D, stage in keys:
        sub = [
            r
            for r in rows
            if r["family"] == family
            and r["S"] == S_len
            and r["D"] == D
            and r["stage"] == stage
        ]
        nuc = np.array([r["nuclear_norm"] for r in sub], dtype=float)
        cond = np.array(
            [r["condition_number"] for r in sub if np.isfinite(r["condition_number"])],
            dtype=float,
        )
        rank = np.array([r["rank"] for r in sub], dtype=float)
        out.append(
            {
                "family": family,
                "S": S_len,
                "D": D,
                "stage": stage,
                "n": len(sub),
                "nuclear_mean": float(np.mean(nuc)),
                "nuclear_std": float(np.std(nuc, ddof=1)) if len(sub) > 1 else 0.0,
                "cond_median": float(np.median(cond)) if cond.size else float("nan"),
                "rank_mean": float(np.mean(rank)),
            }
        )
    return out


def marian_probe_consistency() -> List[Dict]:
    enc_csv = (
        _ROOT
        / "results_probe_independent/schemeA_3x2x3_20260424_0137"
        / "cascade_s0.1_t0.2_seed42/probe_data/probe_train_enc_0.csv"
    )
    rows: List[Dict] = []
    if not enc_csv.is_file():
        return rows

    with enc_csv.open(newline="", encoding="utf-8") as f:
        rdr = list(csv.DictReader(f))
    raw_n = np.array([float(r["V_raw__nuclear"]) for r in rdr])
    first_n = np.array([float(r["V_first__nuclear"]) for r in rdr])
    second_n = np.array([float(r["V_second__nuclear"]) for r in rdr])

    synth_rf, synth_rs = [], []
    for seed in range(NUM_SEEDS):
        set_seed(BASE_SEED + seed)
        V = torch.randn(128, 64).numpy()
        S_t = build_temporal_operator(128, ALPHA_T)
        S_s = build_spatial_operator(128, ALPHA_S)
        v1 = S_s @ V
        v2 = S_t @ v1
        synth_rf.append(analyze_matrix(v1)["nuclear_norm"] / analyze_matrix(V)["nuclear_norm"])
        synth_rs.append(analyze_matrix(v2)["nuclear_norm"] / analyze_matrix(V)["nuclear_norm"])

    rows.append(
        {
            "metric": "nuclear_ratio_first_to_raw",
            "marian_probe_mean": float(np.mean(first_n / raw_n)),
            "synthetic_128x64_mean": float(np.mean(synth_rf)),
        }
    )
    rows.append(
        {
            "metric": "nuclear_ratio_second_to_raw",
            "marian_probe_mean": float(np.mean(second_n / raw_n)),
            "synthetic_128x64_mean": float(np.mean(synth_rs)),
        }
    )
    return rows


def activation_vs_cascade_table(summary: List[Dict]) -> List[str]:
    """Side-by-side rows mirroring script2 Table style at S=64,D=64."""
    lines = [
        "| stage | nuclear (mean) | med κ |",
        "|---|---:|---:|",
    ]
    act = {
        r["stage"]: r
        for r in summary
        if r["family"] == "activation" and r["S"] == 64 and r["D"] == 64
    }
    cas = {
        r["stage"]: r
        for r in summary
        if r["family"] == "cascade" and r["S"] == 64 and r["D"] == 64
    }
    for name in ("linear", "gelu", "relu", "silu"):
        if name in act:
            r = act[name]
            lines.append(
                f"| act:{name} | {r['nuclear_mean']:.1f} | {r['cond_median']:.1f} |"
            )
    for name in ("raw", "spatial_only", "temporal_only", "spatial_first"):
        if name in cas:
            r = cas[name]
            lines.append(
                f"| cascade:{name} | {r['nuclear_mean']:.1f} | {r['cond_median']:.1f} |"
            )
    return lines


def write_markdown_report(
    noncomm: List[Dict],
    bound: Dict[str, float],
    verify: List[Dict],
    summary: List[Dict],
    probe_rows: List[Dict],
) -> None:
    gains = temporal_frequency_gains(64, ALPHA_T)
    lines = [
        "# Cascade spectral operator verification",
        "",
        "Reproducible via `experiments/verify_cascade_spectral_operators.py`.",
        "",
        "## 1. Torch vs explicit matrix operators",
        "",
    ]
    for row in verify:
        lines.append(
            f"- {row['temporal_mode']}+{row['spatial_mode']}: "
            f"temporal err **{row['max_err_temporal']:.2e}**, "
            f"spatial err **{row['max_err_spatial']:.2e}**"
        )
    lines.extend(["", "## 2. Commutativity (α_s=0.1, α_t=0.2)", ""])
    for row in noncomm:
        if int(row["S_len"]) != 64:
            continue
        lines.append(
            f"- S=64, {row['temporal_mode']}+{row['spatial_mode']}: "
            f"||[S_spat,S_temp]||_2=**{row['norm_commutator']:.4f}**"
        )
    lines.append(
        "- Encoder (circular+global): commutator ≈ 0 ⇒ final V is order-invariant; "
        "order only relabels intermediate probe stages."
    )
    lines.extend(
        [
            "",
            "## 3. Bound ||O||_* ≤ ||A||_2 ||V'||_* (numeric)",
            f"- max ratio over {bound['n_trials']} trials: **{bound['max_ratio_nuc_O_over_normA_nucV']:.4f}**",
            "",
            "## 4. Temporal low-pass |λ(ω)| at S=64",
            f"- DC k=0: **{gains[0]:.4f}**; Nyquist k=32: **{gains[32]:.4f}**",
            "",
            "## 5. Activation vs cascade (S=64,D=64, 20 seeds; parallels script2)",
            "",
        ]
    )
    lines.extend(activation_vs_cascade_table(summary))
    lines.append("")
    if probe_rows:
        lines.append("## 6. Marian enc_0 probe vs synthetic 128×64")
        for r in probe_rows:
            lines.append(
                f"- {r['metric']}: probe **{r['marian_probe_mean']:.4f}**, "
                f"synthetic **{r['synthetic_128x64_mean']:.4f}**"
            )
    (OUTPUT / "verification_report.md").write_text("\n".join(lines), encoding="utf-8")


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    verify = verify_operator_matches_torch(64, 64)
    save_csv(OUTPUT / "operator_torch_consistency.csv", verify)
    noncomm: List[Dict] = []
    for S in (8, 16, 64, 128):
        noncomm.extend(verify_noncommutativity(S))
    save_csv(OUTPUT / "noncommutativity.csv", noncomm)
    bound = verify_attention_bound(64, 64)
    save_csv(OUTPUT / "attention_bound_check.csv", [bound])
    freq_rows = []
    for S in (16, 64, 128):
        for k, g in enumerate(temporal_frequency_gains(S, ALPHA_T)):
            freq_rows.append({"S": S, "freq_bin": k, "gain_abs": float(g)})
    save_csv(OUTPUT / "temporal_frequency_gains.csv", freq_rows)
    sweep = sweep_synthetic()
    save_csv(OUTPUT / "synthetic_sweep_raw.csv", sweep)
    summary = summarize_sweep(sweep)
    save_csv(OUTPUT / "synthetic_sweep_summary.csv", summary)
    probe_rows = marian_probe_consistency()
    save_csv(OUTPUT / "marian_probe_vs_synthetic.csv", probe_rows)
    write_markdown_report(noncomm, bound, verify, summary, probe_rows)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
