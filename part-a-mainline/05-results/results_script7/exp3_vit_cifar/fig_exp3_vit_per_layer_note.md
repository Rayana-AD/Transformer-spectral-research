# Script7 Exp3 — 逐层线性探针图说明

对应图文件：`fig_exp3_vit_per_layer_acc_vs_layer.png` / `.pdf`（由 `exp3_vit_per_layer_summary.csv` 生成或更新）。

## 实验设置

- **骨干**：预训练 ViT（默认 `google/vit-base-patch16-224`），**完整 12 个 encoder block**，**backbone 冻结**（不更新 ViT 权重）。
- **数据**：CIFAR-10；图像 **Resize 224×224**，**ImageNet 均值/方差归一化**。
- **探针**：对每个隐藏层索引 \(\ell = 0,\ldots,L\)（\(\ell=0\) 为 patch embedding 之后、第 1 个 block 之前），取该层 **CLS token** 表示 \(h_\ell \in \mathbb{R}^d\)，单独训练一个 **线性分类头**（10 类），与其它层探针相互独立。
- **指标来源**：\(\bar c_{\mathrm{geom}}\) 与 prefix 上的 \(|c-1|\) 统计与 **同一次冻结前向** 上的核范数链一致（详见 `exp3_vit_per_layer_probe_rationale.txt`）。
- **本仓库 CSV 快照**：`exp3_vit_per_layer_summary.csv` 中 `cls_activation=gelu`，`per_layer_probe=1`，`freeze_backbone=1`。

## 目标

在**固定表示流**上，同时观察：

1. **任务可读性**：仅用第 \(\ell\) 层 CLS 做线性决策时，CIFAR-10 **测试准确率**随深度如何变化。
2. **几何前缀指标**：从第 0 层到第 \(\ell\) 层的核范数链上，**几何平均临界性标度** \(\bar c_{\mathrm{geom}}\)（prefix \(0\ldots\ell\)）随 \(\ell\) 的变化，并与准确率对照。


## 坐标轴含义

| 轴 | 含义 |
|----|------|
| **横轴** | **Layer index \(\ell\)**：与 Transformer 输出序列中的 **层索引**（`hidden_states[\ell]`）一致；本数据中 \(\ell=0,\ldots,12\)。 |
| **左纵轴** | **Test accuracy (CIFAR-10)**：该层线性探针在 **测试集**上的准确率（CSV：`best_test_acc`）。 |
| **右纵轴** | **\(\bar c_{\mathrm{geom}}\) from \(N_0,\ldots,N_\ell\)**：用前缀核范数 \(N_0,\ldots,N_\ell\) 定义的几何平均比 \(\bar c_{\mathrm{geom}} = (N_\ell/N_0)^{1/\ell}\)。\(\ell=0\) 时无定义（CSV 中常为缺失）。 |

## 图中竖点线（若有）

脚本在相邻层 \(\ell-1 \to \ell\) 上计算 \(| \Delta \mathrm{acc} | \times | \Delta \bar c_{\mathrm{geom}} |\)，在乘积不低于当前最大乘积的 **18%** 的区间中，取至多 **3** 个最大者，在对应 **层间** \(x=\ell-0.5\) 处画灰色竖点线：表示该步上 **准确率与 \(\bar c_{\mathrm{geom}}\) 同时出现较大变化**（非因果，仅便于读图）。  
**当前仓库 CSV** 下约为 \(x \in \{1.5,\,2.5,\,3.5\}\)（因 \(\ell=0\to1\) 时 \(\bar c_{\mathrm{geom}}\) 在 \(\ell=0\) 缺失，该步不参与筛选）。

## 结合 `exp3_vit_per_layer_summary.csv` 读图

以下为当前 CSV 中的数值趋势（读图时以你本地的最新 CSV 为准）。

- **准确率（左轴）**：从 \(\ell=0\) 的 **0.10** 单调升至 \(\ell=12\) 的 **0.8652**；\(\ell=0\to1\) 跃升最大（0.10 → 0.2882），之后稳步上升，后段 \(\ell=10\)–\(12\) 仍小幅增长（0.7888 → 0.8591 → 0.8652）。
- **\(\bar c_{\mathrm{geom}}\)（右轴）**：\(\ell=1\) 起有定义，约 **2.18**，随 \(\ell\) **总体下降**并在中后段在约 **1.26–1.35** 间波动；与左轴上升形成对照。
- **`c_t_to_next`（表中列，未直接画在图上）**：相邻层核范数比 \(N_{\ell+1}/N_\ell\)，例如 \(\ell=0\) 为 **2.18**，最后一层 \(\ell=12\) 为空（无“下一层”）。
- **相关（见 `exp3_vit_per_layer_spearman_note.txt`）**：Spearman(acc, layer_index) = **1.0**（深度与准确率完全单调同向）；Spearman(acc, mean_abs_c_minus_1) 约为 **-0.986**（与 CSV 列 `mean_abs_c_minus_1` 一致，对应 prefix 上 |c-1| 的均值）。均为观测性统计，非因果。

## 图注（建议 Caption）

**英文（期刊风格）**：Layer-wise **test accuracy** of linear probes on the CLS token at each depth of a frozen ViT-B/16 on CIFAR-10 (indigo, hollow circles), with \(\bar c_{\mathrm{geom}}=(N_\ell/N_0)^{1/\ell}\) from nuclear norms \(N_0,\ldots,N_\ell\) (red, dashed, same-size hollow circles). Vertical dotted lines mark layer boundaries where both curves take a comparatively large step in the same interval (see below). No title; legend below the panel.

**中文**：CIFAR-10、冻结 ViT 上逐层 CLS **测试准确率**（深蓝实线、空心圆）与由 \(N_0,\ldots,N_\ell\) 得到的 \(\bar c_{\mathrm{geom}}\)（红色虚线、**同尺寸空心圆**）。灰色竖点线标示相邻层间两条曲线**同时**变化较大的位置。图例在图下方。

## 重画图（不重新训练）

```bash
python script7_vit_criticality.py exp3 \
  --per_layer_figure_only results_script7/exp3_vit_cifar/exp3_vit_per_layer_summary.csv
```

会在 CSV 同目录写出 `fig_exp3_vit_per_layer_acc_vs_layer.png`、`.pdf` 并更新 `exp3_vit_per_layer_spearman_note.txt`。
