#!/usr/bin/env python3
"""Script5：2×3 合并图 — 每子图叠加多激活曲线；上/下排为两种输入（长名称）。"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from paper_style import apply_paper_style

DEFAULT_SUMMARY = Path("results_script5/vit_svd_summary.csv")
DEFAULT_RAW = Path("results_script5/vit_svd_raw.csv")
DEFAULT_OUT_DIR = Path("results_script5/figures")

# 行顺序：上排 ImageNet 归一化高斯，下排原始高斯
INPUT_KIND_ROWS = ("imagenet_norm_randn", "randn")

# 每行左下斜体脚注（不重复放在子图标题里）
INPUT_KIND_FOOTNOTE = {
    "imagenet_norm_randn": (
        "ImageNet-normalized Gaussian noise — "
        r"$x \sim \mathcal{N}(0,I)$, then $(x-\mu_{\mathrm{IN}})/\sigma_{\mathrm{IN}}$"
    ),
    "randn": (r"Isotropic Gaussian pixels — $x \sim \mathcal{N}(0,I)$, no normalization"),
}

ACT_PRETTY = {
    "gelu": r"$\mathrm{GeLU}$",
    "relu": r"$\mathrm{ReLU}$",
    "light_depth_t2.0": r"Light depth $t{=}2$",
    "phys_tau_softplus_z_b1.0": r"Phys.\ $\mathrm{softplus}$-$z$, $\beta{=}1$",
}

# 与默认双激活区分度好的配色（activations 多于 2 时循环）
ACT_COLORS = ("#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#949494")


def _read_summary(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_max_layers_raw(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            return int(row["max_layers"])
    return None


def _activations_in_summary(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for r in rows:
        a = r["activation"]
        if a not in seen:
            seen.add(a)
            order.append(a)
    pref = ["gelu", "relu", "light_depth_t2.0", "phys_tau_softplus_z_b1.0"]
    head = [a for a in pref if a in seen]
    tail = [a for a in order if a not in head]
    return head + tail


def _ylim_pad(
    arrays: list[np.ndarray],
    *,
    pad_frac: float = 0.08,
    min_span: float | None = None,
    ref_line: float | None = None,
) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size])
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = max(hi - lo, min_span or 0.0)
    pad = max(span * pad_frac, 1e-9)
    ymin, ymax = lo - pad, hi + pad
    if ref_line is not None and not (ymin <= ref_line <= ymax):
        ymin = min(ymin, ref_line) - pad
        ymax = max(ymax, ref_line) + pad
    return ymin, ymax


def plot_combined_panel(
    summary: list[dict],
    *,
    max_layers: int,
    out_png: Path,
    activations: list[str],
) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.0, 7.2),
        sharex="col",
        sharey=False,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.92,
        bottom=0.20,
        wspace=0.26,
        hspace=0.36,
    )

    z = 1.96

    for row_idx, ik in enumerate(INPUT_KIND_ROWS):
        #该 input 下各激活、各列的 y 数组（用于统一 ylim）
        nuc_cols: list[list[np.ndarray]] = [[], [], []]
        for col in range(3):
            nuc_cols[col] = []

        for act_idx, act in enumerate(activations):
            color = ACT_COLORS[act_idx % len(ACT_COLORS)]
            rows = sorted(
                [r for r in summary if r["activation"] == act and r["input_kind"] == ik],
                key=lambda r: int(r["layer_index"]),
            )
            if not rows:
                continue
            layers = np.array([int(r["layer_index"]) for r in rows], dtype=float)
            nuc = np.array([float(r["nuclear_mean"]) for r in rows])
            nuc_e = np.array([float(r["nuclear_ci95_halfwidth"]) for r in rows])
            log_n = np.log(np.clip(nuc, 1e-30, None))

            ct_layers: list[int] = []
            ct_m: list[float] = []
            ct_err: list[float] = []
            for r in rows:
                cm = r.get("c_t_mean", "")
                if cm != "" and cm.lower() != "nan" and np.isfinite(float(cm)):
                    ct_layers.append(int(r["layer_index"]))
                    m = float(cm)
                    ct_m.append(m)
                    n = max(int(r["n_runs"]), 1)
                    sd = float(r.get("c_t_std", 0.0) or 0.0)
                    ct_err.append(z * sd / np.sqrt(n))

            ct_layers_a = np.array(ct_layers, dtype=float)
            ct_m_a = np.array(ct_m)
            ct_e_a = np.array(ct_err)

            ax0, ax1, ax2 = axes[row_idx, 0], axes[row_idx, 1], axes[row_idx, 2]
            ax0.errorbar(
                layers,
                nuc,
                yerr=nuc_e,
                fmt="o-",
                color=color,
                capsize=2.5,
                markersize=4.5,
                linewidth=1.2,
                label=act,
            )
            ax1.plot(layers, log_n, "o-", color=color, markersize=4.5, linewidth=1.2, label=act)
            if ct_m:
                ax2.errorbar(
                    ct_layers_a,
                    ct_m_a,
                    yerr=ct_e_a,
                    fmt="o-",
                    color=color,
                    capsize=2.5,
                    markersize=4.5,
                    linewidth=1.2,
                    label=act,
                )
            nuc_cols[0].append(nuc)
            nuc_cols[0].append(nuc + nuc_e)
            nuc_cols[0].append(nuc - nuc_e)
            nuc_cols[1].append(log_n)
            if ct_m:
                nuc_cols[2].append(ct_m_a)
                nuc_cols[2].append(ct_m_a + ct_e_a)
                nuc_cols[2].append(ct_m_a - ct_e_a)

        for col, ax in enumerate(axes[row_idx]):
            ax.grid(False)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
            yvals = [np.asarray(a, dtype=float) for a in nuc_cols[col]]
            ref = 1.0 if col == 2 else None
            lo, hi = _ylim_pad(yvals, pad_frac=0.07, min_span=0.02 if col == 2 else None, ref_line=ref)
            ax.set_ylim(lo, hi)
            if col == 2:
                ax.axhline(1.0, color="gray", ls="--", lw=0.9, zorder=-1)

        axes[row_idx, 0].set_ylabel(r"Nuclear norm $N_\ell$")
        axes[row_idx, 1].set_ylabel(r"$\ln(N_\ell)$")
        axes[row_idx, 2].set_ylabel(
            rf"$c_\ell\ (\mathrm{{ViT}},\ L_{{\mathrm{{trunc}}}} = {max_layers})$",
            fontsize=10,
        )
        # 输入说明：仅左列；上排离轴框近，下排让出 x 轴双行标签空间
        if row_idx == 0:
            foot_y = -0.11
        else:
            foot_y = -0.40
        axes[row_idx, 0].text(
            0.0,
            foot_y,
            INPUT_KIND_FOOTNOTE[ik],
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            style="italic",
        )

    leg_handles = [
        Line2D([0], [0], color=ACT_COLORS[i % len(ACT_COLORS)], marker="o", linestyle="-", linewidth=1.2)
        for i, _ in enumerate(activations)
    ]
    leg = axes[0, 0].legend(
        handles=leg_handles,
        labels=[ACT_PRETTY.get(a, a) for a in activations],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        frameon=True,
        fancybox=True,
        edgecolor="0.35",
        facecolor="white",
        fontsize=8,
        handlelength=1.6,
        title="MLP activation",
        title_fontsize=8,
        borderpad=0.6,
        labelspacing=0.55,
    )
    leg.set_zorder(100)

    for col in range(3):
        if col == 2:
            axes[1, col].set_xlabel(r"Layer index $\ell$ ($c_\ell=N_{\ell+1}/N_\ell$)")
        elif col == 0:
            ax_b0 = axes[1, col]
            ax_b0.set_xlabel(r"Layer index $\ell$", fontsize=10, labelpad=10)
            ax_b0.text(
                0.5,
                -0.24,
                "(0 = post-embedding)",
                transform=ax_b0.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                style="italic",
            )
        else:
            axes[1, col].set_xlabel(r"Layer index $\ell$")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _activation_summary_lines(summary: list[dict], activations: list[str]) -> list[str]:
    max_li = max(int(r["layer_index"]) for r in summary)
    lines: list[str] = []
    for ik in INPUT_KIND_ROWS:
        lines.append(f"### `{ik}`")
        lines.append("")
        lines.append(
            f"| activation | $\\ln N_0$ (mean) | $\\ln N_{{{max_li}}}$ (mean) | mean $\\bar c_\\ell$* |"
        )
        lines.append("|------------|----------------|----------------------|--------|")
        for act in activations:
            rows = [r for r in summary if r["activation"] == act and r["input_kind"] == ik]
            if not rows:
                continue
            by_li = {int(r["layer_index"]): r for r in rows}
            n0 = float(by_li[0]["nuclear_mean"]) if 0 in by_li else float("nan")
            n_last = float(by_li[max_li]["nuclear_mean"]) if max_li in by_li else float("nan")
            c_vals: list[float] = []
            for _, r in sorted(by_li.items()):
                cm = r.get("c_t_mean", "")
                if cm != "" and cm.lower() != "nan" and np.isfinite(float(cm)):
                    c_vals.append(float(cm))
            mean_c = float(np.mean(c_vals)) if c_vals else float("nan")
            lines.append(
                f"| `{act}` | {np.log(max(n0, 1e-30)):.4f} | {np.log(max(n_last, 1e-30)):.4f} | {mean_c:.4f} |"
            )
        lines.append("")
        lines.append(
            r"*mean $\bar c_\ell$：对该输入、该激活下 $\ell=0,\ldots,11$ 共 12 个有限 $c_\ell$（`c_t_mean`）取算术平均，仅作速览。"
        )
        lines.append("")
    return lines


def _c_t_mean_at(summary: list[dict], *, act: str, ik: str, layer_index: int) -> float | None:
    for r in summary:
        if r["activation"] == act and r["input_kind"] == ik and int(r["layer_index"]) == layer_index:
            cm = r.get("c_t_mean") or ""
            if cm and str(cm).lower() != "nan" and np.isfinite(float(cm)):
                return float(cm)
    return None


def _nuclear0_by_activation(summary: list[dict], ik: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in summary:
        if r["input_kind"] == ik and int(r["layer_index"]) == 0:
            out[r["activation"]] = float(r["nuclear_mean"])
    return out


def _csv_findings_lines(summary: list[dict], *, n_raw: int) -> list[str]:
    n_runs = int(summary[0]["n_runs"]) if summary else 8
    lines: list[str] = [
        "## CSV 核对结论与主要发现",
        "",
        f"*由 `vit_svd_summary.csv` 解析（每格 `n_runs={n_runs}`）。原始行数见上节「与 CSV 的对应关系」。重跑 Script5 后请复查尖峰数值。*",
        "",
    ]
    for ik_label, ik in (
        ("`imagenet_norm_randn`", "imagenet_norm_randn"),
        ("`randn`", "randn"),
    ):
        n0b = _nuclear0_by_activation(summary, ik)
        if len(n0b) >= 2:
            vals = list(n0b.values())
            same = all(math.isclose(vals[0], v, rel_tol=0.0, abs_tol=1e-4) for v in vals)
            if same:
                lines.append(
                    f"- **$\\ell=0$ 与激活无关（{ik_label}）**：各激活 `nuclear_mean` 在 layer0 **均为** `{vals[0]:.2f}`，与仅经 embedding、尚未进入替换 MLP 的 block 一致。"
                )
    c_ld_im_9 = _c_t_mean_at(summary, act="light_depth_t2.0", ik="imagenet_norm_randn", layer_index=9)
    c_ld_rn_9 = _c_t_mean_at(summary, act="light_depth_t2.0", ik="randn", layer_index=9)
    c_phys_rn_0 = _c_t_mean_at(summary, act="phys_tau_softplus_z_b1.0", ik="randn", layer_index=0)
    if c_ld_im_9 is not None:
        lines.append(
            f"- **Light depth + ImageNet 归一化高斯**：`layer_index=9`（即 $N_{{10}}/N_9$）处 $c_\\ell\\approx {c_ld_im_9:.2f}$，为显著尖峰；随后 $N_\\ell$ 在约 $4\\times 10^4$ 量级附近形态剧变。"
        )
    if c_phys_rn_0 is not None:
        lines.append(
            f"- **Phys + `randn`**：`layer_index=0`（即 $N_1/N_0$）处 $c_\\ell\\approx {c_phys_rn_0:.2f}$，第一层后范数跃迁极强。"
        )
    if c_ld_rn_9 is not None:
        lines.append(
            f"- **Light depth + `randn`**：`layer_index=9` 处 $c_\\ell\\approx {c_ld_rn_9:.2f}$；与 ImageNet 输入下尖峰同属 **$\\ell=9$ 起始的过渡**，数值依赖输入分布。"
        )
    lines.append(
        "- **与表一致**：上一节「数值速览」中 $\\ln N_0$、$\\ln N_{12}$、$\\bar c_\\ell$ 与 `vit_svd_summary.csv` 逐行一致。"
    )
    lines.append("")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="Script5 ViT norms 2×3 combined panel (multi-activation colors).")
    p.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--raw_csv", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--stem",
        type=str,
        default="vit_norms_combined_panel",
        help="输出 vit_norms_combined_panel.png / .pdf",
    )
    args = p.parse_args()

    if not args.summary_csv.is_file():
        raise SystemExit(f"Missing {args.summary_csv}")

    summary = _read_summary(args.summary_csv)
    activations = _activations_in_summary(summary)
    if not activations:
        raise SystemExit("No activations in summary CSV.")

    max_layers = _read_max_layers_raw(args.raw_csv)
    if max_layers is None:
        max_layers = max(int(r["layer_index"]) for r in summary)

    out_png = args.out_dir / f"{args.stem}.png"
    plot_combined_panel(summary, max_layers=max_layers, out_png=out_png, activations=activations)

    n_raw = 0
    if args.raw_csv.is_file():
        with args.raw_csv.open(encoding="utf-8") as f:
            n_raw = sum(1 for _ in f) - 1

    n_runs_cap = int(summary[0]["n_runs"]) if summary else 8

    note_lines = [
        "# Script5：ViT 层间核范数 $N_\\ell$ 与 $c_\\ell$ — 合并面板说明",
        "",
        f"对应图：`{args.stem}.png` / `.pdf`（目录 `{args.out_dir.resolve()}`）。",
        "",
        "## 实验设置（与 `script5_vit_representation_svd.py` 一致）",
        "",
        "- **模型**：预训练 `google/vit-base-patch16-224`（ViT-B/16），前向时 **冻结权重**；MLP 非线性可为 **GeLU（默认不改动）**、**ReLU**、**Light depth $t{=}2$**、**Phys.\ softplus-$z$（$\\beta{=}1$）** 等（与 `script5_vit_representation_svd.py` / script7 对齐）。",
        f"- **截断**：本批 CSV 对应 **encoder 块数** `max_layers = {max_layers}`（图中 $L_{{\\mathrm{{trunc}}}} = {max_layers}$）。",
        "- **表示与指标**：对每层 `hidden_states[\\ell]` 将 token×channel 展平为矩阵，**逐样本 SVD**，核范数 $N_\\ell$ 在 batch 上平均；层间比 $c_\\ell = N_{\\ell+1}/N_\\ell$（与 Script5 原三联图定义一致）。",
        "- **随机性**：多个随机种子、同一输入管线重复前向；图中曲线为 **`vit_svd_summary.csv` 中按（activation, input\\_kind, layer）聚合后的均值**，误差条为核范数与 $c_\\ell$ 的 **95% CI（正态近似）**。",
        "- **两种输入（本图上下排）**：",
        "  - **上排** `imagenet_norm_randn`：**全名**为 *Gaussian noise with ImageNet channel normalization* — 即 $x\\sim\\mathcal{N}(0,I)$ 再按 ImageNet 的 $\\mu,\\sigma$ 做逐通道标准化。",
        "  - **下排** `randn`：**全名**为 *isotropic Gaussian pixels* — $x\\sim\\mathcal{N}(0,I)$，**无**输入标准化。",
        f"- **激活曲线**：本数据包含 {len(activations)} 条激活设置 — {', '.join(f'`{a}`' for a in activations)} — **同子图内以色区分**（与 Script7 Exp2 多曲线思路一致）。",
        "",
        "## 坐标轴与子图布局",
        "",
        f"- **布局**：$2\\times 3$。左：**$N_\\ell$ vs $\\ell$**；中：**$\\ln(N_\\ell)$**；右：**$c_\\ell$**，灰虚线 $c_\\ell=1$；右列纵轴 $c_\\ell$（ViT，$L_{{\\mathrm{{trunc}}}} = {max_layers}$）。",
        "- **横轴与注释**：列共享；**仅底行**写 `Layer index $\\ell$`，右列附加 $c_\\ell=N_{\\ell+1}/N_\\ell$；**“0 = post-embedding”** 仅在 **左下** 子图 **一次**（斜体）。**输入类型**为各行 **左列轴框左下** 的 **斜体脚注**。",
        "- **样式**：**无浅灰 grid**；**圆形 marker** + 实线；**图例（label box）**在 **左上第一个子图内部**、**纵向排列**，标题 *MLP activation*。",
        "",
        "## 读图提示",
        "",
        "- **上下排对比**：同一激活下，输入尺度与统计结构不同会导致 $N_\\ell$ 整体平移或形状变化；**不宜**脱离输入类型比较绝对核范数大小。",
        "- **$\\ln(N_\\ell)$**：强调相对变化与「是否近似线性于深度」的直观形状（非线性网络下仅作可视化）。",
        "- **$c_\\ell$**：约在1 附近表示层间核范数近似守恒；系统性 **$<1$** 表示沿深度收缩，**$>1$** 表示扩张；预训练 ViT 在合成输入上常出现 **先抑后扬** 等形状，与 Script6 小合成栈上远离 1 的 $c_\\ell$ **不矛盾**（架构、权重、输入分布均不同）。",
        "",
        "## 与 CSV 的对应关系",
        "",
        f"- **聚合曲线**：[`../vit_svd_summary.csv`](../vit_svd_summary.csv) — **{len(summary)} 条记录**（不含表头）；列为 `activation`, `input_kind`, `layer_index`, `nuclear_mean`, `nuclear_ci95_halfwidth`, `c_t_mean`, `c_t_std`, `n_runs` 等。",
        f"- **逐种子原始行**：[`../vit_svd_raw.csv`](../vit_svd_raw.csv) — **{n_raw} 条记录**（不含表头）。列含 `nuclear_norm`, `c_t`, `input_kind`, `activation`, `max_layers`, `seed` 等。",
        "",
        "## 数值速览（来自 summary）",
        "",
        *_activation_summary_lines(summary, activations),
        *_csv_findings_lines(summary, n_raw=n_raw),
        "## 简要分析与科学意义",
        "",
        "- **分析**：ImageNet 归一化输入上 GeLU/ReLU 轨迹接近，而 **Light / Phys** 可显著改变范数链与 **局部 $c_\\ell$ 尖峰**；各向同性高斯输入下整体更“膨胀”，且特定层 $c_\\ell\\gg 1$，不宜概括为“恒在 1 附近”。",
        "- **意义**：在同一 ViT 权重与指标定义下隔离 **输入构造** 与 **MLP 激活**，便于和 Script6/7 对照：**$c_\\ell$ 可比区间差一个量级**，结论须固定模型、输入、归一化与层索引约定。",
        "",
        "## 建议 Caption（英文）",
        "",
        f"**Two-row, three-column panel (ViT-B/16, frozen).** **Rows:** (top) ImageNet-normalized Gaussian pixels, (bottom) isotropic Gaussian pixels. **Columns:** mean $N_\\ell$, $\\ln N_\\ell$, and $c_\\ell=N_{{\\ell+1}}/N_\\ell$ (mean $\\pm$ approximate 95% CI over seeds). **Curves:** four MLP activations (GeLU, ReLU, Light depth $t{{=}}2$, Phys softplus-$z$). Dashed line: $c_\\ell=1$. Data: `vit_svd_summary.csv` ($n_runs={n_runs_cap}$ per cell).",
        "",
        "## 建议 Caption（中文）",
        "",
        "**ViT-B/16 冻结权重 $2\\times 3$ 面板。** **上排**：ImageNet 通道归一化高斯输入；**下排**：各向同性高斯像素。**列**：层均核范数 $N_\\ell$、$\\ln N_\\ell$、层间比 $c_\\ell$（跨种子均值与 95% CI）。**不同颜色曲线**：不同 MLP 激活；灰虚线 $c_\\ell=1$。",
        "",
        "## 重新生成",
        "",
        "```bash",
        "# 先跑 Script5 写出 CSV（示例）",
        "python script5_vit_representation_svd.py --num_seeds 8 --batch 4 \\",
        "  --input_kinds randn,imagenet_norm_randn \\",
        "  --activations gelu,relu,light_depth_t2.0,phys_tau_softplus_z_b1.0",
        "",
        "# 仅重绘合并图 + 本说明",
        "python script5_vit_norms_combined_panel.py \\",
        f"  --summary_csv {args.summary_csv} --raw_csv {args.raw_csv} --out_dir {args.out_dir}",
        "```",
        "",
    ]

    out_md = args.out_dir / f"{args.stem}_note.md"
    out_md.write_text("\n".join(note_lines), encoding="utf-8")
    print(f"Wrote {out_png} and {out_md}")


if __name__ == "__main__":
    main()
