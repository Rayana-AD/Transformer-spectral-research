#!/usr/bin/env python3
"""
汇总 scheme A 级联批跑目录下各子实验的 result.json，生成：
  - runs.csv                 每次 run 一行
  - summary_by_config.csv    按 (order, alpha_s, alpha_t) 对 3 seed 的 mean/std
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple


def _f(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or not math.isfinite(x)):
        return ""
    if isinstance(x, (float, int)):
        return f"{float(x):.6g}"
    return str(x)


def load_run(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def shallow_nuclear_train_mean(r: Dict[str, Any]) -> str:
    try:
        k = r.get("first_probe_layer") or (
            "dec_0" if r.get("probe_target") == "decoder" else "enc_0"
        )
        v = r["probe_summaries"][k]["V_raw"]["nuclear"]["mean"]
        return f"{float(v):.4f}"
    except (KeyError, TypeError, ValueError):
        return ""


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: aggregate_schemeA_results.py <OUT_ROOT_DIR>", file=sys.stderr)
        sys.exit(1)
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print("Not a directory:", root, file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, str]] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        rj = p / "result.json"
        if not rj.is_file():
            continue
        r = load_run(rj)
        if not r:
            continue
        exp = r.get("exp_name", p.name)
        is_rev = bool(r.get("is_reverse")) or "cascade_rev" in str(exp)
        order = "temporal_first" if is_rev else "spatial_first"
        mic_fin = 0
        nmic = 0
        for _L, m in (r.get("train_assoc_loss_vs_metrics_by_layer") or {}).items():
            for _k, cell in (m or {}).items():
                if isinstance(cell, dict) and "mic" in cell:
                    nmic += 1
                    v = cell.get("mic")
                    if v is not None and math.isfinite(float(v)):
                        mic_fin += 1
        rows.append(
            {
                "dir": p.name,
                "exp_name": str(exp),
                "order": order,
                "alpha_spatial": _f(r.get("alpha_spatial")),
                "alpha_temporal": _f(r.get("alpha_temporal")),
                "is_reverse": "1" if is_rev else "0",
                "seed": _f(r.get("seed")),
                "bleu": _f(r.get("bleu")),
                "train_loss": _f(r.get("train_loss")),
                "train_ppl": _f(r.get("train_ppl")),
                "max_train_batches": _f(r.get("max_train_batches")),
                "max_eval_batches": _f(r.get("max_eval_batches")),
                "n_eval_batch_bleu": str(len(r.get("eval_per_batch_bleu") or [])),
                "enc0_train_Vraw_nuclear_mean": shallow_nuclear_train_mean(r),
                "probe_target": str(r.get("probe_target", "")),
                "bleu_w4": _f((r.get("bleu_by_decoder_window") or {}).get("4")),
                "bleu_w16": _f((r.get("bleu_by_decoder_window") or {}).get("16")),
                "bleu_w64": _f((r.get("bleu_by_decoder_window") or {}).get("64")),
                "bleu_wfull": _f((r.get("bleu_by_decoder_window") or {}).get("full")),
                "cka_dec2": _f((r.get("cka_mean_by_layer") or {}).get("dec_2")),
                "cka_dec5": _f((r.get("cka_mean_by_layer") or {}).get("dec_5")),
                "grad_decoder_mean": _f(
                    (r.get("gradient_norm_summary") or {}).get("grad_decoder", {}).get("mean")
                ),
                "mic_train_finite_cells": f"{mic_fin}/{nmic}" if nmic else "",
            }
        )

    if not rows:
        print("No result.json under", root)
        return

    runs_csv = root / "schemeA_runs.csv"
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", runs_csv, "n=", len(rows))

    # 按 (order, s, t) 汇总
    by: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    by_ppl: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        key = (row["order"], row["alpha_spatial"], row["alpha_temporal"])
        try:
            by[key].append(float(row["bleu"]))
        except (TypeError, ValueError):
            pass
        try:
            by_ppl[key].append(float(row["train_ppl"]))
        except (TypeError, ValueError):
            pass

    summ = []
    for (order, s, t), bleus in sorted(by.items()):
        if not bleus:
            continue
        ppls = by_ppl.get((order, s, t), [])
        summ.append(
            {
                "order": order,
                "alpha_spatial": s,
                "alpha_temporal": t,
                "n_runs": str(len(bleus)),
                "bleu_mean": f"{mean(bleus):.4f}" if bleus else "",
                "bleu_std": f"{pstdev(bleus):.4f}" if len(bleus) > 1 else "0.0000",
                "train_ppl_mean": f"{mean(ppls):.4f}" if ppls else "",
                "train_ppl_std": f"{pstdev(ppls):.4f}" if len(ppls) > 1 else "0.0000",
            }
        )
    s_csv = root / "schemeA_summary_by_config.csv"
    with open(s_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "order",
                "alpha_spatial",
                "alpha_temporal",
                "n_runs",
                "bleu_mean",
                "bleu_std",
                "train_ppl_mean",
                "train_ppl_std",
            ],
        )
        w.writeheader()
        w.writerows(summ)
    print("Wrote", s_csv, "groups=", len(summ))


if __name__ == "__main__":
    main()
