#!/usr/bin/env python3
"""
探针数据导出 + **传播/级联** 作图（见 ``probe_mech_plots``）。

- CSV：各层训练步、各层 eval 批次（``probe_data/``）。
- 图：级联 $V^{\\mathrm{raw}}\\!\\to V^{(1)}\\!\\to V^{(2)}$、
  **eval 上按 batch 序 × 多 encoder 线**、Spearman 用 **层×指标热力图** + TSV 表，

不再默认生成与「单条柱状 Spearman / 每层级联 hexbin 墙」等重复的旧式面板。

输出：``<exp_dir>/probe_plots/`` 下以 ``mech_`` 为前缀的 PNG 及 ``mech_correlation_table.tsv``。

依赖：numpy, matplotlib（见 requirements.txt）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 训练/导出 CSV 与多面板图共用的 (stage, metric_key) — 与 ``compute_token_metrics`` 一致
DEFAULT_TRAIN_METRIC_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("V_raw", "nuclear"),
    ("V_raw", "temporal_cosine_mean"),
    ("V_raw", "temporal_var_mean"),
    ("V_raw", "spatial_var_mean"),
    ("V_raw", "temporal_diff_mean"),
    ("V_first", "nuclear"),
    ("V_second", "nuclear"),
)


def _stage_mathtext(stage: str) -> str:
    return {
        "V_raw": r"$V^{\mathrm{raw}}$",
        "V_first": r"$V^{(1)}$",
        "V_second": r"$V^{(2)}$",
    }.get(stage, stage)


def _key_mathtext(key: str) -> str:
    """Matplotlib mathtext（内置解析器，与完整 LaTeX 略有差异）。"""
    m = {
        "nuclear": r"nuclear norm $\Vert V\Vert_{*}$",
        "temporal_var_mean": r"temp. variance $\overline{\mathrm{Var}_t}$",
        "spatial_var_mean": r"spat. variance $\overline{\mathrm{Var}_s}$",
        "temporal_diff_mean": r"temp. roughness $\overline{|\Delta V|}$",
        "temporal_cosine_mean": r"adj. cosine $\overline{\cos}$",
    }
    if key in m:
        return m[key]
    return key.replace("_", r"\_").replace(" ", r"\ ")


def _pair_axis_label(stage: str, key: str) -> str:
    return f"{_stage_mathtext(stage)} — {_key_mathtext(key)}"


def _pair_suptitle_line(stage: str, key: str) -> str:
    return f"{_stage_mathtext(stage)} | {_key_mathtext(key)}"

# Eval 图中用于与 BLEU 对照的层与标量（默认最浅层）
DEFAULT_EVAL_LAYER = "enc_0"
DEFAULT_EVAL_METRIC = ("V_raw", "nuclear")


def _col_name(stage: str, key: str) -> str:
    return f"{stage}__{key}"


def measurements_to_train_rows(
    measurements: List[Dict[str, Any]],
    metric_pairs: Sequence[Tuple[str, str]] = DEFAULT_TRAIN_METRIC_PAIRS,
) -> Tuple[List[str], List[List[float]]]:
    """将单层 measurements 转为表头 + 行数据（用于 CSV）。"""
    headers = ["step", "loss"] + [_col_name(s, k) for s, k in metric_pairs]
    rows: List[List[float]] = []
    for m in measurements:
        step = float(m.get("step", 0))
        loss = m.get("loss")
        loss_f = float(loss) if loss is not None and not (isinstance(loss, float) and np.isnan(loss)) else float("nan")
        row = [step, loss_f]
        for stage, key in metric_pairs:
            try:
                row.append(float(m[stage][key]))
            except (KeyError, TypeError):
                row.append(float("nan"))
        rows.append(row)
    return headers, rows


def save_train_csv_for_probes(
    probes: List[Any],
    out_dir: Path,
    metric_pairs: Sequence[Tuple[str, str]] = DEFAULT_TRAIN_METRIC_PAIRS,
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for p in probes:
        name = getattr(p, "layer_name", "layer")
        path = out_dir / f"probe_train_{name}.csv"
        headers, rows = measurements_to_train_rows(p.measurements, metric_pairs)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)
        written.append(path)
    return written


def save_eval_batch_csv(
    batch_rows: List[Dict[str, Any]],
    out_dir: Path,
    layer: str = DEFAULT_EVAL_LAYER,
    metric_pairs: Sequence[Tuple[str, str]] = DEFAULT_TRAIN_METRIC_PAIRS,
) -> Optional[Path]:
    """多 batch eval：单层的 ``probe_eval_{layer}.csv``（同批次的 BLEU + 该层探针列）。"""
    if not batch_rows:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"probe_eval_{layer}.csv"
    headers = ["batch_idx", "bleu"] + [_col_name(s, k) for s, k in metric_pairs]
    rows: List[List[float]] = []
    for r in batch_rows:
        bi = int(r.get("batch_idx", 0))
        bleu = float(r.get("bleu", 0.0))
        layer_obj = (r.get("layers") or {}).get(layer) or {}
        row: List[float] = [bi, bleu]
        for stage, key in metric_pairs:
            try:
                row.append(float(layer_obj[stage][key]))
            except (KeyError, TypeError):
                row.append(float("nan"))
        rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return path


def save_eval_batch_csv_per_layer(
    batch_rows: List[Dict[str, Any]],
    out_dir: Path,
    metric_pairs: Sequence[Tuple[str, str]] = DEFAULT_TRAIN_METRIC_PAIRS,
) -> List[Path]:
    if not batch_rows or not (batch_rows[0].get("layers") or {}):
        return []
    written: List[Path] = []
    for layer in sorted(batch_rows[0]["layers"].keys()):
        p = save_eval_batch_csv(
            batch_rows, out_dir, layer=layer, metric_pairs=metric_pairs
        )
        if p is not None:
            written.append(p)
    out_dir = Path(out_dir)
    if not (out_dir / "probe_eval_batches.csv").is_file():
        for candidate in ("probe_eval_enc_0.csv", "probe_eval_dec_0.csv"):
            cpath = out_dir / candidate
            if cpath.is_file():
                (out_dir / "probe_eval_batches.csv").write_bytes(cpath.read_bytes())
                written.append(out_dir / "probe_eval_batches.csv")
                break
    return written


def save_eval_full_json(batch_rows: List[Dict[str, Any]], out_dir: Path) -> Optional[Path]:
    """完整 eval 行（含所有层）JSON，用于事后自定义绘图。"""
    if not batch_rows:
        return None
    out_dir = Path(out_dir)
    path = out_dir / "probe_eval_batch_rows.json"
    with open(path, "w") as f:
        json.dump(batch_rows, f, indent=2, default=float)
    return path


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or x.size == 0:
        return x.copy()
    pad = win // 2
    xp = np.pad(x.astype(float), (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    out = np.convolve(xp, kernel, mode="valid")
    if out.size >= x.size:
        return out[: x.size]
    return out


def export_and_plot(
    probes: List[Any],
    eval_batch_rows: List[Dict[str, Any]],
    exp_dir: Path,
    plot_train: bool = True,
    plot_eval: bool = True,
    save_csv: bool = True,
    save_eval_json: bool = False,
    rolling_window: int = 0,
    train_assoc_by_layer: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    eval_assoc_by_layer: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    train_spearman_by_layer: Optional[Dict[str, Dict[str, float]]] = None,
    eval_spearman_by_layer: Optional[Dict[str, Dict[str, float]]] = None,
    include_legacy_step_plots: bool = False,
    include_eval_batch_axis_plots: bool = False,
    train_plot_stride: int = 1,
) -> Dict[str, Any]:
    """
    写入 CSV/可选 JSON；作图由 ``probe_mech_plots.run_mechanism_plot_suite`` 完成
   （沿深度/三阶段/池化相关散点、多关联系数热力图 + TSV。默认不画以 step / val batch 序
   为主横轴 的图，除非 ``include_*_legacy`` 为真）。

    ``train_assoc_by_layer`` / ``eval_assoc_by_layer``：来自 ``result.json`` 的嵌套
    关联系数（每标量 含 spearman, pearson, kendall_tau, mic）。未传时可用仅 Spearman
    的 ``train_spearman_by_layer`` / ``eval_spearman_by_layer`` 画热力图（旧格式）。
    """
    from probe_mech_plots import run_mechanism_plot_suite

    exp_dir = Path(exp_dir)
    data_dir = exp_dir / "probe_data"
    plot_dir = exp_dir / "probe_plots"
    data_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Any] = {
        "data_dir": str(data_dir),
        "plot_dir": str(plot_dir),
        "csv": [],
        "plots": [],
        "train_plot_stride": int(train_plot_stride) if train_plot_stride else 1,
    }

    if save_csv:
        out["csv"].extend([str(p) for p in save_train_csv_for_probes(probes, data_dir)])
        out["csv"].extend(
            str(p) for p in save_eval_batch_csv_per_layer(eval_batch_rows, data_dir)
        )
    if save_eval_json:
        j = save_eval_full_json(eval_batch_rows, data_dir)
        if j:
            out["eval_json"] = str(j)

    if plot_train or plot_eval:
        out["plots"].extend(
            run_mechanism_plot_suite(
                probes,
                eval_batch_rows,
                plot_dir,
                data_dir,
                train_assoc_by_layer=train_assoc_by_layer,
                eval_assoc_by_layer=eval_assoc_by_layer,
                train_spearman_by_layer=train_spearman_by_layer,
                eval_spearman_by_layer=eval_spearman_by_layer,
                rolling_window=rolling_window,
                do_train=plot_train,
                do_eval=plot_eval,
                include_legacy_step_plots=include_legacy_step_plots,
                include_eval_batch_axis_plots=include_eval_batch_axis_plots,
                train_plot_stride=train_plot_stride,
            )
        )

    return out


def plot_from_csv(
    train_csv: Path,
    eval_csv: Optional[Path],
    out_dir: Path,
    rolling_window: int = 11,
) -> List[Path]:
    """
    事后仅根据 CSV 重画（无需重新跑训练）。train_csv 为单层 probe_train_enc_0.csv 等。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # 简单读 CSV
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return saved
    step = np.array([float(r["step"]) for r in rows], dtype=float)
    loss = np.array([float(r["loss"]) if r.get("loss") not in ("", None) else np.nan for r in rows], dtype=float)
    # 动态找 V_raw__nuclear 列
    nuc_col = _col_name("V_raw", "nuclear")
    if nuc_col in rows[0]:
        nuc = np.array([float(r[nuc_col]) for r in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        n = len(step)
        hb = ax.hexbin(
            nuc,
            loss,
            gridsize=min(50, max(20, int(np.sqrt(max(n, 1))))),
            cmap="viridis",
            mincnt=1,
        )
        plt.colorbar(hb, ax=ax, label="count")
        ax.set_xlabel("V_raw nuclear")
        ax.set_ylabel("loss")
        ax.set_title("Loss vs nuclear (from CSV)")
        fig.tight_layout()
        p = out_dir / "train_loss_vs_Vraw_nuclear_hexbin_from_csv.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        saved.append(p)

    return saved
