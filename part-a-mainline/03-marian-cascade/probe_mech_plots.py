#!/usr/bin/env python3
"""
Transformer 层间传播 / 级联探针 的可视化（与 ``probe_visualization`` 的 CSV 列一致）。

目标
----
1. **沿深度（encoder 层下标）** 的级联：同一步或时间平均，横轴 = 层，三条折线
   对应 :math:`V^{\\mathrm{raw}}, V^{(1)}, V^{(2)}`（传播机制主图，不以 step 为主轴）。
2. **同画布上的阶段演化**：横轴 = 三阶段下标 0,1,2，多条折线 = 各层，展示单层内
   级联形状及层间对比。
3. **与 loss / PPL 的关系**：每层的 hexbin 保留；另提供 **全层点云** 散点
   (核范数 vs :math:`\\mathcal{L}` / PPL)，色 = 层，直接对应相关性与传播。
4. 相关性：多方法（Spearman / Pearson / Kendall / MIC）**层×指标** 热力图
   + ``mech_correlation_table.tsv``（列含 ``method``）。
5. 逐步 / 按 eval batch 序号的曲线 **可选**（默认不生成），避免与「深度 + 相关」
   主叙事抢权重。

作图用 matplotlib mathtext（``$...$``），不用裸 ``V_raw`` 下划线字串。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 与 probe_visualization 共享展示键（仅 raw 在 eval 上跨层最可比；级联在 train/ eval 用三阶段）
CASCADE_METRIC_KEYS: Tuple[str, ...] = (
    "nuclear",
    "temporal_cosine_mean",
    "temporal_var_mean",
    "spatial_var_mean",
    "temporal_diff_mean",
)

STAGES: Tuple[str, ...] = ("V_raw", "V_first", "V_second")
STAGE_COL = {"V_raw": "C0", "V_first": "C1", "V_second": "C2"}
STAGE_LABEL = {"V_raw": r"$V^{\mathrm{raw}}$", "V_first": r"$V^{(1)}$", "V_second": r"$V^{(2)}$"}
# 池化 (nuc vs L) 图：同层三线区分阶段
STAGE_LINESTYLE: Dict[str, Any] = {
    "V_raw": "-",
    "V_first": (0, (5, 2)),
    "V_second": (0, (1, 1.2)),
}

# 与 run_probe_independent.ASSOC_METHOD_KEYS 一致（热力图/TSV）
ASSOC_HEATMAP_METHODS: Tuple[str, ...] = (
    "spearman",
    "pearson",
    "kendall_tau",
    "mic",
)
NEG_ONE_TO_ONE = frozenset({"spearman", "pearson", "kendall_tau"})


def _import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _strided_sequence(mlist: List[Any], stride: int) -> List[Any]:
    """作图用：对训练测探针行按时间隔行取样，不修改原始 ``measurements``。"""
    if not mlist or stride is None or int(stride) <= 1:
        return mlist
    s = int(stride)
    return mlist[::s]


def _key_short(key: str) -> str:
    m = {
        "nuclear": "nuc",
        "temporal_cosine_mean": "t_cos",
        "temporal_var_mean": "t_var",
        "spatial_var_mean": "s_var",
        "temporal_diff_mean": "t_dif",
    }
    return m.get(key, key)


def _layer_index_from_name(probe: Any) -> int:
    name = getattr(probe, "layer_name", "enc_0")
    if isinstance(name, str) and name.startswith("enc_"):
        try:
            return int(name.split("_", 1)[1])
        except (ValueError, IndexError):
            return 0
    if isinstance(name, str) and name.startswith("dec_"):
        try:
            return int(name.split("_", 1)[1])
        except (ValueError, IndexError):
            return 0
    return 0


def _min_train_steps(probes: List[Any]) -> int:
    lens = [len(getattr(p, "measurements", None) or []) for p in probes]
    return min(lens) if lens else 0


def _apply_roll(y: np.ndarray, rw: int) -> np.ndarray:
    if rw <= 1 or y.size == 0:
        return y
    pad = rw // 2
    xp = np.pad(y.astype(float), (pad, pad), mode="edge")
    k = np.ones(rw) / rw
    out = np.convolve(xp, k, mode="valid")
    return out[: y.size] if out.size >= y.size else y


def _draw_hex(
    fig,
    ax,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    n = int(x.size)
    if n < 2:
        return
    if n > 80:
        g = int(min(45, max(10, int(np.sqrt(n)))))
        h = ax.hexbin(
            x,
            y,
            gridsize=g,
            cmap="viridis",
            mincnt=1,
            linewidths=0.0,
            edgecolors="none",
        )
        cbar = fig.colorbar(h, ax=ax, fraction=0.05, pad=0.1)
        cbar.set_label("points / bin", fontsize=8)
    else:
        ax.scatter(x, y, s=10, alpha=0.45, c="0.25", edgecolors="none")
    ax.grid(False)


def plot_train_cascade_stages_timeseries(
    probes: List[Any],
    out_dir: Path,
    metric_keys: Sequence[str] = CASCADE_METRIC_KEYS,
    rolling_window: int = 1,
    train_plot_stride: int = 1,
) -> List[Path]:
    """
    每层一图：多子图（每物理量一行），子图内三条线 = raw / first / second。
    """
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for probe in probes:
        mlist = _strided_sequence(
            getattr(probe, "measurements", None) or [], train_plot_stride
        )
        name = getattr(probe, "layer_name", "layer")
        n = len(mlist)
        if n < 2:
            continue
        steps = np.array(
            [float(m.get("step", i + 1)) for i, m in enumerate(mlist)], dtype=float
        )
        nrows = len(metric_keys)
        fig, axes = plt.subplots(nrows, 1, figsize=(7.0, 2.0 * nrows), sharex=True, squeeze=False)
        for ri, mkey in enumerate(metric_keys):
            ax = axes[ri, 0]
            for st in STAGES:
                try:
                    y = np.array([float(m[st][mkey]) for m in mlist], dtype=float)
                except (KeyError, TypeError, IndexError):
                    continue
                if rolling_window > 1 and n >= rolling_window:
                    y = _apply_roll(y, rolling_window)
                ax.plot(
                    steps,
                    y,
                    color=STAGE_COL[st],
                    lw=1.1,
                    label=STAGE_LABEL[st],
                    alpha=0.9,
                )
            ax.set_ylabel(_key_short(mkey), fontsize=9)
            if ri == 0:
                ax.legend(loc="upper right", fontsize=7, ncol=3)
            ax.grid(False)
        axes[-1, 0].set_xlabel(r"train step $t$ (minibatch index; legacy)")
        fig.suptitle(
            f"Train (legacy) | {name} | "
            + r"$V^{\mathrm{raw}},V^{(1)},V^{(2)}$ vs $t$ | " + f"n={n}",
            fontsize=10,
        )
        fig.tight_layout()
        p = out_dir / f"mech_train_cascade_stages__{name}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
    return saved


def plot_train_cascade_nuclear_loss_ppl(
    probes: List[Any],
    out_dir: Path,
    train_plot_stride: int = 1,
) -> List[Path]:
    """每层 2×3：行= loss / PPL，列= V_raw, V(1), V(2) 核范数 对 y 的 hexbin。"""
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for probe in probes:
        mlist = _strided_sequence(
            getattr(probe, "measurements", None) or [], train_plot_stride
        )
        name = getattr(probe, "layer_name", "layer")
        if len(mlist) < 2:
            continue
        try:
            loss = np.array(
                [float(m["loss"]) if m.get("loss") is not None else float("nan") for m in mlist],
                dtype=float,
            )
        except (KeyError, TypeError):
            continue
        ppl = np.exp(np.clip(loss, -50, 50))
        fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.5), squeeze=False)
        for ci, st in enumerate(STAGES):
            try:
                x = np.array([float(m[st]["nuclear"]) for m in mlist], dtype=float)
            except (KeyError, TypeError):
                continue
            for ri, (yv, ylab) in enumerate(
                [(loss, r"$\mathcal{L}$"), (ppl, "PPL")],
            ):
                ax = axes[ri, ci]
                o = np.isfinite(x) & np.isfinite(yv)
                if o.sum() < 2:
                    continue
                _draw_hex(fig, ax, x[o], yv[o])
                if ri == 0:
                    ax.set_title(
                        f"{STAGE_LABEL[st]}  nuclear (norm.)",
                        fontsize=9,
                    )
                ax.set_ylabel(ylab, fontsize=9)
        sp = int(train_plot_stride) if train_plot_stride and int(train_plot_stride) > 1 else 1
        st_note = f" | plot stride={sp}" if sp > 1 else ""
        fig.suptitle(
            f"Train | {name} | nuclear vs $\\mathcal{{L}}$/PPL (hexbin) | n={len(mlist)}{st_note}",
            fontsize=10,
        )
        fig.tight_layout()
        p = out_dir / f"mech_train_cascade_nuclear_L_PPL__{name}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
    return saved


def plot_train_depth_cascade(
    probes: List[Any],
    out_dir: Path,
    metric_key: str = "nuclear",
    aggregate: str = "mean",
) -> List[Path]:
    """
    横轴 = **encoder 层**（深），纵轴 = 标量，三条线 = 三阶段同指标。
    ``aggregate``：``"mean"`` 对训练时间维平均；``"last"`` 用各层对齐后的最后一步。
    """
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not probes:
        return []
    nstep = _min_train_steps(probes)
    if nstep < 1:
        return []
    L = len(probes)
    x = np.arange(L, dtype=float)
    fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * L + 2), 3.4))
    for st in STAGES:
        vals = np.empty(L, dtype=float)
        for li, p in enumerate(probes):
            mlist = getattr(p, "measurements", None) or []
            if not mlist:
                vals[li] = float("nan")
                continue
            if aggregate == "last":
                t = len(mlist) - 1
                try:
                    vals[li] = float(mlist[t][st][metric_key])
                except (KeyError, TypeError, IndexError):
                    vals[li] = float("nan")
            else:
                row = []
                for t in range(min(nstep, len(mlist))):
                    try:
                        row.append(float(mlist[t][st][metric_key]))
                    except (KeyError, TypeError, IndexError):
                        pass
                vals[li] = float(np.mean(row)) if row else float("nan")
        ax.plot(
            x,
            vals,
            "o-",
            ms=3.5,
            lw=1.1,
            label=STAGE_LABEL[st],
            color=STAGE_COL[st],
        )
    ax.set_xlabel("encoder layer index (depth $l$)")
    ax.set_ylabel(_key_short(metric_key))
    ag = "time-mean" if aggregate == "mean" else "last step"
    ax.set_title(
        rf"Train | cascade along depth | {ag} | {_key_short(metric_key)} | "
        rf"$K={L}$ layers, $T \geq {nstep}$ train steps",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [getattr(p, "layer_name", f"enc_{i}") for i, p in enumerate(probes)],
        rotation=0,
        fontsize=8,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(False)
    fig.tight_layout()
    p = out_dir / f"mech_train_depth_cascade__{_key_short(metric_key)}__{aggregate}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return [p]


def plot_train_stage_path_all_layers(
    probes: List[Any],
    out_dir: Path,
    metric_key: str = "nuclear",
    aggregate: str = "mean",
) -> List[Path]:
    """
    单画布：横轴 = 级联三阶段 0$\\to$1$\\to$2，纵轴 = 标量；**每层一条折线**
    展示 $V^{\\mathrm{raw}}\\!\\to\\!V^{(1)}\\!\\to\\!V^{(2)}$ 在同一指标上的形变与层间对比。
    """
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not probes:
        return []
    nstep = _min_train_steps(probes)
    if nstep < 1:
        return []
    x_stage = np.arange(3, dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cmap = plt.get_cmap("viridis" if len(probes) > 2 else "tab10")
    for li, p in enumerate(probes):
        mlist = getattr(p, "measurements", None) or []
        if not mlist:
            continue
        ys: List[float] = []
        for st in STAGES:
            if aggregate == "last":
                t = len(mlist) - 1
                try:
                    ys.append(float(mlist[t][st][metric_key]))
                except (KeyError, TypeError, IndexError):
                    ys.append(float("nan"))
            else:
                row = []
                for t in range(min(nstep, len(mlist))):
                    try:
                        row.append(float(mlist[t][st][metric_key]))
                    except (KeyError, TypeError, IndexError):
                        pass
                ys.append(float(np.mean(row)) if row else float("nan"))
        if len(ys) != 3 or not any(np.isfinite(ys)):
            continue
        col = cmap(li / max(1, len(probes) - 1) if len(probes) > 1 else 0.0)
        ax.plot(
            x_stage,
            ys,
            "o-",
            ms=3,
            lw=1.0,
            color=col,
            alpha=0.9,
            label=getattr(p, "layer_name", f"enc_{li}"),
        )
    ax.set_xticks(x_stage)
    ax.set_xticklabels(
        [STAGE_LABEL[s] for s in STAGES],
        fontsize=9,
    )
    ax.set_xlabel("cascade stage within self-attn value (same $l$)", fontsize=9)
    ax.set_ylabel(_key_short(metric_key), fontsize=9)
    ag = "time-mean" if aggregate == "mean" else "last step"
    ax.set_title(
        r"Train | intra-layer path $V^{\mathrm{raw}}\!\to\!V^{(1)}\!\to\!V^{(2)}$ | "
        + f"{ag} | {_key_short(metric_key)}"
    )
    if len(probes) <= 12:
        ax.legend(loc="best", fontsize=6, ncol=2)
    else:
        ax.legend(loc="best", fontsize=5, ncol=3)
    ax.grid(False)
    fig.tight_layout()
    p = out_dir / f"mech_train_stage_path_all_layers__{_key_short(metric_key)}__{aggregate}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return [p]


def plot_train_pooled_nuclear_vs_loss_ppl(
    probes: List[Any],
    out_dir: Path,
    train_plot_stride: int = 1,
) -> List[Path]:
    """
    **每层一图**（2 行：$\\mathcal{L}$ / PPL；横轴 = 该层该阶段的 nuclear）：
    同一张小图里用颜色 + 线型画满三阶段 (raw / 1st / 2nd)，**仅该层** 的时序折线，避免
    多层叠在一张里「意面图」难读。

    点仍按时间顺序在 (nuclear, L) 或 (nuclear, PPL) 平面上连接；每阶段一条轨迹。
    """
    from collections import defaultdict

    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # layer_idx -> st -> list of (t_step, nuc, lof, ppl) in time order
    by_layer: Dict[int, Dict[str, List[Tuple[float, float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    ts = int(train_plot_stride) if train_plot_stride and int(train_plot_stride) > 0 else 1
    for p in probes:
        li = int(_layer_index_from_name(p))
        mlist = _strided_sequence(getattr(p, "measurements", None) or [], ts)
        for m in mlist:
            t_step = float(m.get("step", 0) or 0)
            lo = m.get("loss")
            if lo is None or not np.isfinite(lo):
                continue
            lof = float(lo)
            ppl = float(np.exp(np.clip(lof, -50, 50)))
            for st in STAGES:
                try:
                    nuc = float(m[st]["nuclear"])
                except (KeyError, TypeError, IndexError, ValueError):
                    continue
                by_layer[li][st].append((t_step, nuc, lof, ppl))

    for li in list(by_layer.keys()):
        for st in STAGES:
            by_layer[li][st].sort(key=lambda r: r[0])
    if not by_layer or not any(by_layer[li][st] for li in by_layer for st in STAGES):
        return []

    n_layers = 1 + max(
        (int(_layer_index_from_name(p)) for p in probes),
        default=0,
    )
    st_note = f" | plot stride = {ts}" if ts > 1 else ""
    saved: List[Path] = []
    for li in range(n_layers):
        title_layer = f"enc_{li}"
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(4.0, 4.2), sharex=True)
        any_line = False
        for st in STAGES:
            pts = sorted(by_layer.get(li, {}).get(st, []), key=lambda r: r[0])
            if len(pts) < 1:
                continue
            nxs = [r[1] for r in pts]
            loss_y = [r[2] for r in pts]
            ppl_y = [r[3] for r in pts]
            col = STAGE_COL[st]
            ls = STAGE_LINESTYLE[st]
            ax0.plot(
                nxs,
                loss_y,
                marker="o",
                ms=3,
                lw=1.1,
                color=col,
                linestyle=ls,
                alpha=0.9,
                label=STAGE_LABEL[st],
            )
            ax1.plot(
                nxs,
                ppl_y,
                marker="o",
                ms=3,
                lw=1.1,
                color=col,
                linestyle=ls,
                alpha=0.9,
            )
            any_line = True
        if not any_line:
            plt.close(fig)
            continue
        ax0.set_ylabel(r"$\mathcal{L}$", fontsize=9)
        ax0.set_title(
            r"Train | layer "
            + title_layer
            + r" | nuclear (three stages) vs $\mathcal{L}$ / PPL (time-ordered lines)"
            + st_note,
            fontsize=8,
        )
        ax0.grid(False)
        ax0.legend(loc="best", fontsize=6, handlelength=2.2)
        ax1.set_ylabel("PPL", fontsize=9)
        ax1.set_xlabel(r"nuclear norm ($\Vert\cdot\Vert_{*}$)", fontsize=8)
        ax1.grid(False)
        fig.tight_layout()
        p = out_dir / f"mech_train_pooled_nuclear_vs_L_PPL__{title_layer}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
    return saved


def plot_train_global_L_PPL_Vraw_enc0(
    probes: List[Any],
    out_dir: Path,
    rolling_window: int = 1,
    train_plot_stride: int = 1,
) -> List[Path]:
    """
    **全局**训练曲线：同一小批次上全模型共用的 loss / PPL，以及 enc_0 的 $V^{\\mathrm{raw}}$ nuclear
    对训练步 $t$（minibatch 序），与每 epoch 多 batch 的「监控」同构；**所有层**的 loss 相同，故
    只画一张，不重复 `enc_5` 等 6 张。

    若 ``--max-train-batches`` 仅 2–4 则只有 2–4 个点，曲线退化为短折线 —— 需把 max 调大
    才能看到多步波动。
    """
    plt = _import_plt()
    if not probes:
        return []
    p0 = probes[0]
    lay0 = getattr(p0, "layer_name", "enc_0")
    mlist = _strided_sequence(
        getattr(p0, "measurements", None) or [],
        int(train_plot_stride) if train_plot_stride else 1,
    )
    n = len(mlist)
    if n < 1:
        return []
    ts = int(train_plot_stride) if train_plot_stride and int(train_plot_stride) > 0 else 1
    steps = np.array(
        [float(m.get("step", i + 1)) for i, m in enumerate(mlist)], dtype=float
    )
    loss = np.array(
        [
            float(m["loss"]) if m.get("loss") is not None else float("nan")
            for m in mlist
        ],
        dtype=float,
    )
    ppl = np.exp(np.clip(loss, -50, 50))
    v0 = []
    for m in mlist:
        try:
            v0.append(float(m["V_raw"]["nuclear"]))
        except (KeyError, TypeError, ValueError, IndexError):
            v0.append(float("nan"))
    v0 = np.array(v0, dtype=float)
    if rolling_window > 1 and n >= rolling_window:
        loss = _apply_roll(loss, rolling_window)
        ppl = _apply_roll(ppl, rolling_window)
        v0 = _apply_roll(v0, rolling_window)
    # 不用 squeeze=False：(3,1) 在 squeeze False 时 axes 为 2D ndarray，(ax0,ax1,ax2) 会错
    fig, axs = plt.subplots(3, 1, figsize=(6.8, 5.0), sharex=True)
    ax0, ax1, ax2 = (np.ravel(axs)[:3])
    ax0.plot(steps, loss, "o-", color="0.2", ms=3, lw=0.9)
    ax0.set_ylabel(r"$\mathcal{L}$ (global, same for all probed layers)")
    ax0.set_title(
        r"Train | global: $\mathcal{L}$, PPL, and $V^{\mathrm{raw}}$ nuclear (first probed layer) vs $t$"
    )
    ax0.grid(False)
    ax1.plot(steps, ppl, "o-", color="0.25", ms=3, lw=0.9)
    ax1.set_ylabel("PPL")
    ax1.grid(False)
    ax2.plot(steps, v0, "o-", color="C0", ms=3, lw=0.9)
    ax2.set_ylabel(
        f"{lay0}: " + r"$V^{\mathrm{raw}}$ nuc."
    )
    ax2.set_xlabel(r"train step $t$ (minibatch index)")
    ax2.grid(False)
    foot: List[str] = []
    if n < 4:
        foot.append(
            f"only T={n} point(s) on plot after stride — use larger --max-train-batches for smoother L/PPL"
        )
    if ts > 1:
        foot.append(
            f"plot stride={ts} (subsampled for clarity; see CSV for all steps)"
        )
    if foot:
        fig.text(0.5, 0.01, " | ".join(foot), ha="center", fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.05 if foot else 0.01, 1, 1))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"mech_train_global_L_PPL_Vraw_{lay0}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [p]


def plot_eval_depth_cascade(
    eval_batch_rows: List[Dict[str, Any]],
    out_dir: Path,
    metric_key: str = "nuclear",
    aggregate: str = "last",
) -> List[Path]:
    """
    Eval：横轴 = 层，三条线 = 三阶段。默认 ``last``=最后一个 val batch 上的断面；
    ``mean``=对全部 batch 平均（不把 batch 序当自变量，只作汇总）。
    """
    plt = _import_plt()
    if not eval_batch_rows or not eval_batch_rows[0].get("layers"):
        return []
    out_dir = Path(out_dir)
    rows = list(eval_batch_rows)
    if aggregate == "last":
        use_rows = [rows[-1]]
    else:
        use_rows = rows
    layers = sorted(rows[0]["layers"].keys())
    L = len(layers)
    x = np.arange(L, dtype=float)
    fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * L + 2), 3.4))
    for st in STAGES:
        ys = []
        for layer in layers:
            acc = []
            for r in use_rows:
                try:
                    acc.append(
                        float(r["layers"][layer][st][metric_key])
                    )
                except (KeyError, TypeError, ValueError):
                    pass
            if not acc:
                ys.append(float("nan"))
            else:
                ys.append(float(np.mean(acc)))
        y = np.array(ys, dtype=float)
        o = np.isfinite(y)
        if o.sum() < 1:
            continue
        ax.plot(
            x[o],
            y[o],
            "o-",
            ms=3.5,
            lw=1.1,
            label=STAGE_LABEL[st],
            color=STAGE_COL[st],
        )
    ax.set_xlabel("encoder layer index (depth $l$)")
    ax.set_ylabel(_key_short(metric_key))
    ag = f"last batch (idx {rows[-1].get('batch_idx', '?')})" if aggregate == "last" else f"mean over {len(rows)} val batches"
    ax.set_title(f"Eval | cascade along depth | {ag}")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=8)
    ax.legend(loc="best", fontsize=7)
    ax.grid(False)
    fig.tight_layout()
    p = out_dir / f"mech_eval_depth_cascade__{_key_short(metric_key)}__{aggregate}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [p]


def plot_eval_stage_path_all_layers(
    eval_batch_rows: List[Dict[str, Any]],
    out_dir: Path,
    metric_key: str = "nuclear",
    aggregate: str = "last",
) -> List[Path]:
    """
    Eval 单画布：横轴 = 三阶段 0$\\to$1$\\to$2，各层折线。数据来自最后一 val batch 或全体 batch 平均。
    """
    plt = _import_plt()
    if not eval_batch_rows or not eval_batch_rows[0].get("layers"):
        return []
    out_dir = Path(out_dir)
    rows = list(eval_batch_rows)
    if aggregate == "last":
        use_rows = [rows[-1]]
    else:
        use_rows = rows
    layers = sorted(rows[0]["layers"].keys())
    x_stage = np.arange(3, dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cmap = plt.get_cmap("viridis" if len(layers) > 2 else "tab10")
    for li, layer in enumerate(layers):
        ys: List[float] = []
        for st in STAGES:
            acc = []
            for r in use_rows:
                try:
                    acc.append(
                        float(r["layers"][layer][st][metric_key])
                    )
                except (KeyError, TypeError, ValueError):
                    pass
            if not acc:
                ys.append(float("nan"))
            else:
                ys.append(float(np.mean(acc)))
        if not any(np.isfinite(ys)):
            continue
        col = cmap(li / max(1, len(layers) - 1) if len(layers) > 1 else 0.0)
        ax.plot(
            x_stage,
            ys,
            "o-",
            ms=3,
            lw=1.0,
            color=col,
            alpha=0.9,
            label=layer,
        )
    ax.set_xticks(x_stage)
    ax.set_xticklabels([STAGE_LABEL[s] for s in STAGES], fontsize=9)
    ax.set_xlabel(r"cascade stage in value (eval)", fontsize=9)
    ax.set_ylabel(_key_short(metric_key), fontsize=9)
    ag = f"last batch" if aggregate == "last" else f"mean of {len(rows)} batches"
    ax.set_title(
        r"Eval | $V^{\mathrm{raw}}\!\to\!V^{(1)}\!\to\!V^{(2)}$ by layer | " + f"{ag} | {_key_short(metric_key)}"
    )
    if len(layers) <= 12:
        ax.legend(loc="best", fontsize=6, ncol=2)
    else:
        ax.legend(loc="best", fontsize=5, ncol=3)
    ax.grid(False)
    fig.tight_layout()
    p = out_dir / f"mech_eval_stage_path_all_layers__{_key_short(metric_key)}__{aggregate}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [p]


def plot_train_loss_ppl(
    probes: List[Any],
    out_dir: Path,
    rolling_window: int = 1,
    train_plot_stride: int = 1,
) -> List[Path]:
    """
    每层一个子图：loss 与 PPL 随 $t$（**legacy**）。**同一** ``loss`` 在每一层重复，
    与 ``mech_train_global_L_PPL_Vraw_enc0.png`` 信息重叠；看全局更推荐后者。
    """
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for probe in probes:
        mlist = _strided_sequence(
            getattr(probe, "measurements", None) or [], train_plot_stride
        )
        name = getattr(probe, "layer_name", "layer")
        n = len(mlist)
        if n < 2:
            continue
        loss = np.array(
            [float(m["loss"]) if m.get("loss") is not None else float("nan") for m in mlist],
            dtype=float,
        )
        ppl = np.exp(np.clip(loss, -50, 50))
        steps = np.array(
            [float(m.get("step", i + 1)) for i, m in enumerate(mlist)], dtype=float
        )
        if rolling_window > 1 and n >= rolling_window:
            loss = _apply_roll(loss, rolling_window)
            ppl = _apply_roll(ppl, rolling_window)
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.5, 4.0), sharex=True)
        ax0.plot(steps, loss, color="0.2", lw=0.9)
        ax0.set_ylabel(r"$\mathcal{L}$")
        ax0.grid(False)
        ax1.plot(steps, ppl, color="0.2", lw=0.9)
        ax1.set_ylabel("PPL")
        ax1.set_xlabel("step")
        ax0.set_title(f"Train | {name} | loss & PPL (aligned with probe rows)")
        fig.tight_layout()
        p = out_dir / f"mech_train_loss_ppl__{name}.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
    return saved


def plot_eval_bleu_line(batch_rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    plt = _import_plt()
    if len(batch_rows) < 1:
        return []
    out_dir = Path(out_dir)
    j = np.array([int(r.get("batch_idx", i)) for i, r in enumerate(batch_rows)])
    b = np.array([float(r["bleu"]) for r in batch_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(j, b, "o-", ms=3, color="0.15", lw=0.9)
    ax.set_xlabel(r"eval batch index (val, order=loader)")
    ax.set_ylabel("corpus BLEU (same batch gen)")
    ax.set_title(f"Eval | per-batch BLEU | K={len(batch_rows)}")
    ax.grid(False)
    fig.tight_layout()
    p = out_dir / "mech_eval_bleu_per_batch.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [p]


def plot_eval_Vraw_metric_all_layers(
    batch_rows: List[Dict[str, Any]],
    out_dir: Path,
    stage: str = "V_raw",
    metric_key: str = "nuclear",
) -> List[Path]:
    """x=batch 序号，y=标量，多条线=各 encoder 层（同阶段同键）。"""
    plt = _import_plt()
    if len(batch_rows) < 2 or not batch_rows[0].get("layers"):
        return []
    out_dir = Path(out_dir)
    j = np.array([int(r.get("batch_idx", i)) for i, r in enumerate(batch_rows)])
    layers = sorted(batch_rows[0]["layers"].keys())
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for li, layer in enumerate(layers):
        ys = []
        for r in batch_rows:
            try:
                ys.append(float(r["layers"][layer][stage][metric_key]))
            except (KeyError, TypeError):
                ys.append(float("nan"))
        y = np.array(ys, dtype=float)
        o = np.isfinite(y)
        if o.sum() < 1:
            continue
        ax.plot(
            j[o],
            y[o],
            "o-",
            ms=2.5,
            lw=0.8,
            label=layer,
            color=f"C{li % 10}",
        )
    ax.set_xlabel(r"eval batch index $j$ (legacy)")
    yst = STAGE_LABEL.get(stage, stage)
    ax.set_ylabel(
        f"{_key_short(metric_key)} @ {yst}",
    )
    ax.set_title(
        r"Eval (legacy) | " + f"{_key_short(metric_key)} vs $j$ — all layers",
    )
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(False)
    fig.tight_layout()
    p = out_dir / f"mech_eval__{stage}__{metric_key}__all_layers.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [p]


def plot_eval_cascade_nuclear_by_layer(
    batch_rows: List[Dict[str, Any]],
    out_dir: Path,
) -> List[Path]:
    """每层一图：batch 序号上三条线 = 三阶段核范数（看级联在跨 batch 上是否同向）。"""
    plt = _import_plt()
    if not batch_rows or not batch_rows[0].get("layers"):
        return []
    out_dir = Path(out_dir)
    j = np.array([int(r.get("batch_idx", i)) for i, r in enumerate(batch_rows)])
    saved: List[Path] = []
    for layer in sorted(batch_rows[0]["layers"].keys()):
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        for st in STAGES:
            ys = []
            for r in batch_rows:
                try:
                    ys.append(float(r["layers"][layer][st]["nuclear"]))
                except (KeyError, TypeError):
                    ys.append(float("nan"))
            y = np.array(ys, dtype=float)
            o = np.isfinite(y)
            if o.sum() < 1:
                continue
            ax.plot(
                j[o],
                y[o],
                "o-",
                ms=2,
                lw=0.9,
                label=STAGE_LABEL[st],
                color=STAGE_COL[st],
            )
        ax.set_xlabel(r"eval batch index $j$ (legacy)")
        ax.set_ylabel("nuclear norm")
        ax.set_title(
            f"Eval (legacy) | {layer} | "
            + r"$V^{\mathrm{raw}}$ vs $V^{(1)}$ vs $V^{(2)}$ (nuclear) vs $j$",
        )
        ax.legend(loc="best", fontsize=7)
        ax.grid(False)
        fig.tight_layout()
        p = out_dir / f"mech_eval_cascade_nuclear__{layer}.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
    return saved


def _sorted_assoc_metric_names(
    d_by_layer: Optional[Dict[str, Dict[str, Any]]], max_rows: int = 32
) -> List[str]:
    if not d_by_layer:
        return []
    u: set = set()
    for d in d_by_layer.values():
        u |= set((d or {}).keys())
    names = sorted(u)
    return names[:max_rows] if max_rows else names


# 关联系数热力图：只保留与「标量/均值类」探针行对应的键（不要 std 等）
def _is_meanish_metric_suffix(suf: str) -> bool:
    if suf in ("std",) or "std" in suf:
        return False
    if suf.endswith("_mean") or suf in ("nuclear", "mean", "l2_norm"):
        return True
    return False


def _meanish_suffix_order() -> List[str]:
    return [
        "nuclear",
        "mean",
        "l2_norm",
        "temporal_cosine_mean",
        "temporal_var_mean",
        "spatial_var_mean",
        "temporal_diff_mean",
    ]


def _metric_suffix_display(suf: str) -> str:
    """热力图 y 轴：无下划线，阶段名在子图标题中表达。"""
    m = {
        "nuclear": r"Nuclear $\Vert\cdot\Vert_{*}$",
        "mean": "Elem. mean",
        "l2_norm": r"$L^2$ norm",
        "temporal_cosine_mean": r"Temp.~$\overline{\cos}$",
        "temporal_var_mean": "Temp. var.",
        "spatial_var_mean": "Spat. var.",
        "temporal_diff_mean": r"Temp.~$|\Delta|$",
    }
    if suf in m:
        return m[suf]
    return suf.replace("_", " ")


def _layer_display_name(layer: str) -> str:
    if layer.startswith("enc_"):
        return layer.split("_", 1)[1]
    return layer.replace("_", " ")


def _metric_key_exists_for_layer(
    dbl: Dict[str, Dict[str, Any]], layer: str, full_key: str
) -> bool:
    d0 = dbl.get(layer) or {}
    return full_key in d0 and d0.get(full_key) is not None


def _metric_key_in_any_layer(dbl: Dict[str, Dict[str, Any]], full_key: str) -> bool:
    return any(_metric_key_exists_for_layer(dbl, ly, full_key) for ly in dbl)


def _nested_assoc_has_any_finite(
    dbl: Optional[Dict[str, Dict[str, Any]]], method: str
) -> bool:
    """嵌套关联系数表里是否至少有一个该 method 的有限值；全 NaN 则不要输出空白 heatmap（如未装 minepy 的 MIC）。"""
    if not dbl:
        return False
    for d0 in dbl.values():
        for cell in (d0 or {}).values():
            v: Any
            if isinstance(cell, dict) and method in cell:
                v = cell.get(method)
            elif method == "spearman" and not isinstance(cell, dict):
                v = cell
            else:
                v = None
            if v is not None and np.isfinite(v):
                return True
    return False


def _sorted_metric_names(
    d_by_layer: Optional[Dict[str, Dict[str, float]]],
    max_rows: int = 28,
) -> List[str]:
    if not d_by_layer:
        return []
    u: set = set()
    for d in d_by_layer.values():
        u |= set((d or {}).keys())
    names = sorted(u)
    return names[:max_rows] if max_rows else names


def plot_correlation_heatmaps(
    train_assoc: Optional[Dict[str, Dict[str, Dict[str, float]]]],
    eval_assoc: Optional[Dict[str, Dict[str, Dict[str, float]]]],
    out_dir: Path,
    train_spearman_by_layer: Optional[Dict[str, Dict[str, float]]] = None,
    eval_spearman_by_layer: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Path]:
    """
    层 × 探针标量 矩阵。若给嵌套 ``assoc``（多方法）则 **每种关联系数** 各出一张
    ( Spearman / Pearson / Kendall :math:`\\tau` 用 $[-1,1]$ 色条；MIC 用 $[0,1]$ )。
    无嵌套时退回仅 Spearman 的扁平表（与旧版文件名兼容）。
    """
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    def build_matrix(
        dbl: Dict[str, Dict[str, Any]], layers: List[str], metrics: List[str], method: str
    ) -> np.ndarray:
        mmat = np.full((len(metrics), len(layers)), np.nan, dtype=float)
        for c, layer in enumerate(layers):
            d = dbl.get(layer) or {}
            for r, mk in enumerate(metrics):
                cell = d.get(mk)
                if isinstance(cell, dict) and method in cell:
                    v = cell.get(method)
                elif method == "spearman" and not isinstance(cell, dict):
                    v = cell
                else:
                    v = None
                if v is not None and np.isfinite(v):
                    mmat[r, c] = float(v)
        return mmat

    def one_method(
        title: str,
        dbl: Optional[Dict[str, Dict[str, Any]]],
        fname: str,
        method: str,
    ) -> None:
        if not dbl:
            return
        layers = sorted(dbl.keys())
        if not layers:
            return
        n_layers = len(layers)
        order = _meanish_suffix_order()
        st_metrics: List[List[str]] = []
        for st in STAGES:
            mnames: List[str] = []
            for suf in order:
                if not _is_meanish_metric_suffix(suf):
                    continue
                fk = f"{st}.{suf}"
                if _metric_key_in_any_layer(dbl, fk):
                    mnames.append(fk)
            st_metrics.append(mnames)
        n_row_max = max((len(m) for m in st_metrics), default=0)
        if n_row_max < 1:
            return
        wcell = 0.18 + 0.3 * n_layers
        hcell = 0.55 + 0.24 * n_row_max
        fig, axs = plt.subplots(
            1, 3, figsize=(3 * wcell, hcell), squeeze=True, constrained_layout=True
        )
        if method in NEG_ONE_TO_ONE:
            cbl = (
                r"Spearman $\rho$"
                if method == "spearman"
                else (r"Pearson $r$" if method == "pearson" else r"Kendall $\tau$")
            )
        else:
            cbl = "MIC"
        im_last: Any = None
        plotted_ax: List[Any] = []
        for ax, st, mnames in zip(np.ravel(axs), STAGES, st_metrics):
            if not mnames:
                ax.set_axis_off()
                ax.set_title(f"{STAGE_LABEL[st]}", fontsize=9)
                ax.text(
                    0.5,
                    0.5,
                    "no rows",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                continue
            mmat = build_matrix(dbl, layers, mnames, method)
            if method in NEG_ONE_TO_ONE:
                vlim = 1.0
                im = ax.imshow(
                    mmat,
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=-vlim,
                    vmax=vlim,
                    interpolation="nearest",
                )
            else:
                im = ax.imshow(
                    mmat,
                    aspect="auto",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
            im_last = im
            plotted_ax.append(ax)
            ax.set_xticks(np.arange(n_layers))
            ax.set_xticklabels(
                [_layer_display_name(ly) for ly in layers], rotation=0, fontsize=7
            )
            ax.set_xlabel("layer", fontsize=7)
            ax.set_yticks(np.arange(len(mnames)))
            ylabs = [_metric_suffix_display(mk.split(".", 1)[1]) for mk in mnames]
            ax.set_yticklabels(ylabs, fontsize=6.5)
            ax.set_title(f"{STAGE_LABEL[st]}", fontsize=9)
        if im_last is None or not plotted_ax:
            plt.close(fig)
            return
        fig.suptitle(
            title
            + " (mean / nuclear / $L^2$; no *std* rows; cols = enc layer index)",
            fontsize=8,
        )
        cbar = fig.colorbar(
            im_last, ax=plotted_ax, label=cbl, fraction=0.025, pad=0.1
        )
        cbar.ax.tick_params(labelsize=7)
        p = out_dir / fname
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    def one_flat_spearman(
        title: str, dbl: Optional[Dict[str, Dict[str, float]]], fname: str
    ) -> None:
        if not dbl:
            return
        layers = sorted(dbl.keys())
        if not layers:
            return
        n_layers = len(layers)
        order = _meanish_suffix_order()

        def _flat_build_for_metrics(
            mnames: List[str],
        ) -> np.ndarray:
            mmat = np.full((len(mnames), n_layers), np.nan, dtype=float)
            for c, layer in enumerate(layers):
                d0 = dbl.get(layer) or {}
                for r, mk in enumerate(mnames):
                    v = d0.get(mk)
                    if v is not None and np.isfinite(v):
                        mmat[r, c] = float(v)
            return mmat

        st_metrics: List[List[str]] = []
        for st in STAGES:
            mnames = []
            for suf in order:
                if not _is_meanish_metric_suffix(suf):
                    continue
                fk = f"{st}.{suf}"
                if any(fk in (dbl.get(ly) or {}) for ly in layers):
                    mnames.append(fk)
            st_metrics.append(mnames)
        n_row_max = max((len(m) for m in st_metrics), default=0)
        if n_row_max < 1:
            return
        wcell = 0.18 + 0.3 * n_layers
        hcell = 0.55 + 0.24 * n_row_max
        fig, axs = plt.subplots(
            1, 3, figsize=(3 * wcell, hcell), squeeze=True, constrained_layout=True
        )
        im: Any = None
        plotted_ax: List[Any] = []
        for ax, st, mnames in zip(np.ravel(axs), STAGES, st_metrics):
            if not mnames:
                ax.set_axis_off()
                ax.set_title(f"{STAGE_LABEL[st]}", fontsize=9)
                continue
            mmat = _flat_build_for_metrics(mnames)
            im = ax.imshow(
                mmat,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-1.0,
                vmax=1.0,
                interpolation="nearest",
            )
            plotted_ax.append(ax)
            ax.set_xticks(np.arange(n_layers))
            ax.set_xticklabels(
                [_layer_display_name(ly) for ly in layers], rotation=0, fontsize=7
            )
            ax.set_xlabel("layer", fontsize=7)
            ax.set_yticks(np.arange(len(mnames)))
            ylabs = [_metric_suffix_display(mk.split(".", 1)[1]) for mk in mnames]
            ax.set_yticklabels(ylabs, fontsize=6.5)
            ax.set_title(f"{STAGE_LABEL[st]}", fontsize=9)
        if im is None or not plotted_ax:
            plt.close(fig)
            return
        fig.suptitle(
            title
            + " (Spearman, mean / nuclear / $L^2$; cols = enc layer index)",
            fontsize=8,
        )
        cbar = fig.colorbar(
            im, ax=plotted_ax, label=r"Spearman $\rho$", fraction=0.025, pad=0.1
        )
        cbar.ax.tick_params(labelsize=7)
        p = out_dir / fname
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    def _is_assoc_nested(d: Optional[Dict]) -> bool:
        if not d:
            return False
        keys = set(ASSOC_HEATMAP_METHODS)
        for d0 in d.values():
            for v in (d0 or {}).values():
                if isinstance(v, dict) and keys & set(v.keys()):
                    return True
        return False

    if _is_assoc_nested(train_assoc):
        for method in ASSOC_HEATMAP_METHODS:
            fname = f"mech_corr_heatmap_train_loss__{method}.png"
            if not _nested_assoc_has_any_finite(train_assoc, method):
                stale = out_dir / fname
                if stale.is_file():
                    try:
                        stale.unlink()
                    except OSError:
                        pass
                continue
            one_method(
                f"Train: loss vs probe | {method}",
                train_assoc,
                fname,
                method,
            )
    else:
        one_flat_spearman(
            "Train: Spearman (rank loss, rank probe) — rows=metrics, cols=layer",
            train_spearman_by_layer,
            "mech_corr_heatmap_train_loss.png",
        )

    if _is_assoc_nested(eval_assoc):
        for method in ASSOC_HEATMAP_METHODS:
            fname = f"mech_corr_heatmap_eval_bleu__{method}.png"
            if not _nested_assoc_has_any_finite(eval_assoc, method):
                stale = out_dir / fname
                if stale.is_file():
                    try:
                        stale.unlink()
                    except OSError:
                        pass
                continue
            one_method(
                f"Eval: batch BLEU vs probe | {method}",
                eval_assoc,
                fname,
                method,
            )
    else:
        one_flat_spearman(
            "Eval: Spearman (rank batch BLEU, rank probe) — rows=metrics, cols=layer",
            eval_spearman_by_layer,
            "mech_corr_heatmap_eval_bleu.png",
        )

    return saved


def write_correlation_tsv(
    train_assoc: Optional[Dict[str, Dict[str, Dict[str, float]]]],
    eval_assoc: Optional[Dict[str, Dict[str, Dict[str, float]]]],
    data_dir: Path,
    train_spearman_by_layer: Optional[Dict[str, Dict[str, float]]] = None,
    eval_spearman_by_layer: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[Path]:
    path = Path(data_dir) / "mech_correlation_table.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, str, str, str, str]] = []

    def from_nested(
        tag: str, dbl: Optional[Dict[str, Dict[str, Dict[str, float]]]]
    ) -> None:
        if not dbl:
            return
        for layer, d0 in sorted(dbl.items()):
            for mname, mets in sorted((d0 or {}).items()):
                if not isinstance(mets, dict):
                    continue
                for mk in sorted(mets.keys()):
                    v = mets[mk]
                    if v is not None and (isinstance(v, (float, int)) and np.isfinite(v)):
                        rows.append(
                            (tag, layer, mname, mk, f"{float(v):.5f}")
                        )

    from_nested("train", train_assoc)
    from_nested("eval", eval_assoc)

    if not rows and (train_spearman_by_layer or eval_spearman_by_layer):
        for tag, dbl in (("train", train_spearman_by_layer), ("eval", eval_spearman_by_layer)):
            if not dbl:
                continue
            for layer, d0 in sorted(dbl.items()):
                for mname, v in sorted((d0 or {}).items()):
                    if v is not None and np.isfinite(v):
                        rows.append(
                            (tag, layer, mname, "spearman", f"{float(v):.5f}")
                        )
    if not rows:
        return None
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["split", "layer", "metric", "method", "value"])
        w.writerows(rows)
    return path


def plot_probe_summaries_cascade_nuclear_bars(
    probe_summaries: Dict[str, Any],
    out_dir: Path,
) -> List[Path]:
    """
    仅来自 ``result.json`` 的 ``probe_summaries``：每层 **训练汇总的** mean nuclear，
    对 V_raw / V(1) / V(2) 做分组柱（非逐步曲线，可与完整 CSV 作图区分）。
    """
    plt = _import_plt()
    if not probe_summaries:
        return []
    out_dir = Path(out_dir)
    layers = sorted(
        k for k in probe_summaries.keys() if k.startswith("enc_")
    ) or sorted(probe_summaries.keys())
    if not layers:
        return []
    x = np.arange(len(layers), dtype=float)
    w = 0.24
    fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * len(layers)), 3.4))
    leg = {
        "V_raw": r"$V^{\mathrm{raw}}$",
        "V_first": r"$V^{(1)}$",
        "V_second": r"$V^{(2)}$",
    }
    for i, st in enumerate(STAGES):
        vals: List[float] = []
        for L in layers:
            try:
                vals.append(
                    float(probe_summaries[L][st]["nuclear"]["mean"])
                )
            except (KeyError, TypeError, ValueError):
                vals.append(float("nan"))
        ax.bar(
            x + (i - 1) * w,
            vals,
            width=w * 0.95,
            label=leg[st],
            color=STAGE_COL[st],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_ylabel("mean nuclear (train summary, result.json)", fontsize=9)
    ax.set_title("Aggregate (no time axis): cascade nuclear by layer from probe_summaries")
    ax.legend(loc="best", fontsize=7)
    ax.grid(False, axis="y")
    fig.tight_layout()
    p = out_dir / "mech_from_result__layer_mean_nuclear_cascade_bars.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [p]


def run_mechanism_plot_suite(
    probes: List[Any],
    eval_batch_rows: List[Dict[str, Any]],
    plot_dir: Path,
    data_dir: Path,
    train_assoc_by_layer: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    eval_assoc_by_layer: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    train_spearman_by_layer: Optional[Dict[str, Dict[str, float]]] = None,
    eval_spearman_by_layer: Optional[Dict[str, Dict[str, float]]] = None,
    rolling_window: int = 0,
    do_train: bool = True,
    do_eval: bool = True,
    include_legacy_step_plots: bool = False,
    include_eval_batch_axis_plots: bool = False,
    train_plot_stride: int = 1,
) -> List[str]:
    """
    生成传播分析图 + 汇总表。返回所写入的 PNG/TSV 路径列表（字符串）。

    默认 **不** 生成以 train ``step`` / eval ``batch`` 为主横轴 的图；需要时打开
    ``include_legacy_step_plots`` / ``include_eval_batch_axis_plots``。

    ``train_plot_stride``：仅作图时每隔几行训练测针 subsample，**不影响** ``measurements`` 与 CSV。
    """
    plot_dir = Path(plot_dir)
    data_dir = Path(data_dir)
    n0 = len(probes[0].measurements) if probes else 0
    rw = rolling_window
    if rw <= 0 and n0 >= 500:
        rw = max(11, n0 // 200)
    elif rw <= 0:
        rw = 1
    tps = int(train_plot_stride) if train_plot_stride and int(train_plot_stride) > 0 else 1

    out: List[str] = []

    if do_train:
        for agg in ("mean", "last"):
            out.extend(
                str(p) for p in plot_train_depth_cascade(
                    probes, plot_dir, metric_key="nuclear", aggregate=agg
                )
            )
            out.extend(
                str(p) for p in plot_train_stage_path_all_layers(
                    probes, plot_dir, metric_key="nuclear", aggregate=agg
                )
            )
        out.extend(
            str(p)
            for p in plot_train_pooled_nuclear_vs_loss_ppl(
                probes, plot_dir, train_plot_stride=tps
            )
        )
        out.extend(
            str(p)
            for p in plot_train_global_L_PPL_Vraw_enc0(
                probes,
                plot_dir,
                rolling_window=rw,
                train_plot_stride=tps,
            )
        )
        out.extend(
            str(p)
            for p in plot_train_cascade_nuclear_loss_ppl(
                probes, plot_dir, train_plot_stride=tps
            )
        )
        if include_legacy_step_plots:
            out.extend(
                str(p)
                for p in plot_train_cascade_stages_timeseries(
                    probes,
                    plot_dir,
                    rolling_window=rw,
                    train_plot_stride=tps,
                )
            )
            out.extend(
                str(p)
                for p in plot_train_loss_ppl(
                    probes,
                    plot_dir,
                    rolling_window=rw,
                    train_plot_stride=tps,
                )
            )

    if do_eval and eval_batch_rows:
        for agg in ("last", "mean"):
            out.extend(
                str(p) for p in plot_eval_depth_cascade(
                    eval_batch_rows, plot_dir, metric_key="nuclear", aggregate=agg
                )
            )
            out.extend(
                str(p) for p in plot_eval_stage_path_all_layers(
                    eval_batch_rows, plot_dir, metric_key="nuclear", aggregate=agg
                )
            )
        if include_eval_batch_axis_plots:
            out.extend(
                str(p) for p in plot_eval_bleu_line(eval_batch_rows, plot_dir)
            )
            for stage, mkey in (
                ("V_raw", "nuclear"),
                ("V_raw", "temporal_cosine_mean"),
            ):
                out.extend(
                    str(p)
                    for p in plot_eval_Vraw_metric_all_layers(
                        eval_batch_rows, plot_dir, stage=stage, metric_key=mkey
                    )
                )
            out.extend(
                str(p)
                for p in plot_eval_cascade_nuclear_by_layer(
                    eval_batch_rows, plot_dir
                )
            )

    if do_train or do_eval:
        out.extend(
            str(p)
            for p in plot_correlation_heatmaps(
                train_assoc_by_layer,
                eval_assoc_by_layer,
                plot_dir,
                train_spearman_by_layer=train_spearman_by_layer,
                eval_spearman_by_layer=eval_spearman_by_layer,
            )
        )
        tsv = write_correlation_tsv(
            train_assoc_by_layer,
            eval_assoc_by_layer,
            data_dir,
            train_spearman_by_layer=train_spearman_by_layer,
            eval_spearman_by_layer=eval_spearman_by_layer,
        )
        if tsv is not None:
            out.append(str(tsv))
    return out
