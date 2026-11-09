#!/usr/bin/env python3
"""Plot NIAH Memory-Track native geometry from ``memory_geometry`` training logs.

Shows coaxial encoder |s| / central-lobe occupancy on needle vs hay, and how
needle–hay contrast evolves alongside val accuracy — not a simulated ψ'' basin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "memory_track_geom"

MEM_COLOR = "#2A9178"
STD_COLOR = "#9B6BA3"
NEEDLE = "#C44E52"
HAY = "#4C72B0"
REF = "#C4884A"


def _load_geom_series(path: Path) -> dict[str, np.ndarray]:
    row = json.loads(path.read_text())
    steps, val_acc = [], []
    needle_s, hay_s = [], []
    needle_lobe, hay_lobe = [], []
    contrast_s, contrast_gated, mem_cos = [], [], []
    for pt in row.get("history", []):
        g = pt.get("memory_geometry")
        if not g:
            continue
        step = int(pt["step"])
        steps.append(step)
        needle = g.get("encoder_last_needle") or {}
        hay = g.get("encoder_last_hay") or {}
        contrast = g.get("encoder_last_contrast") or {}
        needle_s.append(float(needle.get("s_abs_mean", np.nan)))
        hay_s.append(float(hay.get("s_abs_mean", np.nan)))
        needle_lobe.append(float(needle.get("s_in_central_lobe_frac", np.nan)))
        hay_lobe.append(float(hay.get("s_in_central_lobe_frac", np.nan)))
        contrast_s.append(
            float(
                contrast.get(
                    "needle_minus_hay_s_abs",
                    contrast.get("needle_minus_hay_ff_out_rms", np.nan),
                )
            )
        )
        contrast_gated.append(
            float(
                contrast.get(
                    "needle_minus_hay_gated_abs",
                    contrast.get("needle_minus_hay_hidden_abs", np.nan),
                )
            )
        )
        mem_cos.append(float(g.get("memory_needle_hay_cosine", np.nan)))
        # val_acc may live in adjacent history point
        val_acc.append(np.nan)

    # merge val_acc from history
    acc_by_step = {
        int(pt["step"]): float(pt["val_acc"])
        for pt in row.get("history", [])
        if "val_acc" in pt
    }
    val_acc = [acc_by_step.get(s, np.nan) for s in steps]

    return {
        "label": row.get("label", path.stem),
        "variant": row.get("encoder_hessian_variant") or row.get("ffn_type"),
        "steps": np.asarray(steps),
        "val_acc": np.asarray(val_acc),
        "needle_s": np.asarray(needle_s),
        "hay_s": np.asarray(hay_s),
        "needle_lobe": np.asarray(needle_lobe),
        "hay_lobe": np.asarray(hay_lobe),
        "contrast_s": np.asarray(contrast_s),
        "contrast_gated": np.asarray(contrast_gated),
        "mem_cos": np.asarray(mem_cos),
        "lobe_half": float(
            (row.get("memory_geometry_summary") or {})
            .get("first", {})
            .get("central_lobe_halfwidth", np.arctanh(1 / np.sqrt(3)))
        ),
    }


def _psi_second(s: np.ndarray) -> np.ndarray:
    t = np.tanh(s)
    return 2.0 * (1.0 - t * t) * (1.0 - 3.0 * t * t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NIAH memory-track geometry")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "niah_memory_track_geom",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
    )
    args = parser.parse_args()

    hess_path = args.results_dir / "niah_hessian_split_seed0.json"
    std_path = args.results_dir / "niah_standard_seed0.json"
    if not hess_path.exists():
        raise SystemExit(f"missing {hess_path} — run l5_seq2seq_synthetic with geom config first")

    hess = _load_geom_series(hess_path)
    std = _load_geom_series(std_path) if std_path.exists() else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lobe_half = hess["lobe_half"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))

    ax = axes[0, 0]
    ax.plot(hess["steps"], hess["needle_s"], "o-", color=NEEDLE, label="needle |s|", ms=4)
    ax.plot(hess["steps"], hess["hay_s"], "s-", color=HAY, label="hay |s|", ms=4)
    ax.axhline(lobe_half, color=REF, ls="--", lw=1, label=rf"central lobe ($|s|={lobe_half:.2f}$)")
    ax.set_xlabel("step")
    ax.set_ylabel("encoder last-layer mean |s|")
    ax.set_title("Coaxial gate coordinate (Memory-Track encoder)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    # Lobe occupancy stays at 1 on this run; retrieval is the moving evidence.
    acc = hess["val_acc"]
    ax.plot(hess["steps"], np.where(np.isfinite(acc), acc, np.nan), "o-", color=MEM_COLOR, ms=4, label="val acc")
    ax.plot(hess["steps"], hess["mem_cos"], "s-", color=REF, ms=4, label="needle–hay cosine")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xlabel("step")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Retrieval: accuracy up, memories split")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(
        hess["steps"],
        hess["contrast_s"],
        "o-",
        color=MEM_COLOR,
        label="Δ|s| (needle−hay)",
        ms=4,
    )
    ax.plot(
        hess["steps"],
        hess["contrast_gated"],
        "^-",
        color=MEM_COLOR,
        alpha=0.7,
        label="Δ|λ·proj|",
        ms=4,
    )
    if std is not None:
        std_contrast = std["contrast_s"]
        if np.all(np.isnan(std_contrast)) and len(std["contrast_gated"]):
            std_contrast = std["contrast_gated"]
        ax.plot(
            std["steps"],
            std_contrast,
            "s--",
            color=STD_COLOR,
            label="Std Δ FF rms",
            ms=4,
        )
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("needle − hay contrast")
    ax.set_title("Memory write separation")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    # |s| contrast is tiny; the write is in the map output.
    # Reuse contrast_gated only when it is the native gated contrast.
    ax.plot(hess["steps"], hess["needle_s"], "o-", color=NEEDLE, ms=4, label="needle |s|")
    ax.plot(hess["steps"], hess["hay_s"], "s-", color=HAY, ms=4, label="hay |s|")
    ax.set_xlabel("step")
    ax.set_ylabel("mean |s|")
    ax.set_title("Gate coordinate moves; lobe stays full")
    ax.legend(fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / f"fig_memory_track_geom.{ext}", dpi=160)
    plt.close(fig)

    # Reference ψ'' inset figure
    fig2, axr = plt.subplots(figsize=(4.5, 2.8))
    s = np.linspace(-1.5, 1.5, 400)
    axr.plot(s, _psi_second(s), color=REF, lw=1.6)
    axr.axvspan(-lobe_half, lobe_half, alpha=0.12, color=REF, label="central lobe")
    if len(hess["needle_s"]):
        axr.axvline(float(hess["needle_s"][-1]), color=NEEDLE, ls="--", label="final needle |s|")
        axr.axvline(float(hess["hay_s"][-1]), color=HAY, ls=":", label="final hay |s|")
    axr.set_xlabel("gate coordinate s")
    axr.set_ylabel("ψ''(s)")
    axr.set_title("Analytic reference (not simulated training)")
    axr.legend(fontsize=8)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(args.output_dir / f"fig_psi_reference.{ext}", dpi=160)
    plt.close(fig2)

    print(f"→ {args.output_dir}")


if __name__ == "__main__":
    main()
