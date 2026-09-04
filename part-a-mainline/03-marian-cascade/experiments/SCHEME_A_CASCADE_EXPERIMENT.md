# 方案 A：级联多 α × 多顺序 × 多 Seed（完整探针记录）

**目的**：在 **Helsinki-NLP/opus-mt-de-en** 上、与 `run_probe_independent` 已对齐的设置下，系统比较  
（1）三组 **(α_spatial, α_temporal)**、（2）两种 **V 上平滑级联顺序**、（3）**3 个随机种子**，并落盘 `result.json`、探针 CSV/JSON、作图等。

## 1. 级联定义（与代码一致）

| 名称 | 标志 | 含义 |
|------|------|------|
| **Spatial→Temporal**（默认） | 无 `--reverse` | $V^{(1)} = \text{SpatialSmooth}(V^{\mathrm{raw}})$，$V^{(2)} = \text{TemporalSmooth}(V^{(1)})$；`order=spatial_first` |
| **Temporal→Spatial** | `--reverse` | $V^{(1)} = \text{TemporalSmooth}(V^{\mathrm{raw}})$，$V^{(2)} = \text{SpatialSmooth}(V^{(1)})$；`order=temporal_first` |

前向中实际用于 attention 的 $V$ 在训练时来自 **第二阶段输出** $V^{(2)}$ 路径（与 `apply_mixed_4d` 实现一致）。

## 2. 网格

- **α 三类**：(0.2, 0.2), (0.1, 0.2), (0.2, 0.1)
- **顺序**：2
- **Seed**：42, 43, 44  

→ 共 **18 次** 独立 `run_probe_independent` 子目录（每次含完整训练+eval+关联系数+可选探针产物）。

## 3. 环境与命令

在 **`tlb_mine`**（conda，含 `minepy`，见 `docs/CONDA_TLB_MINE.md`）中执行：

```bash
cd /path/to/TLB-Experiments
chmod +x experiments/run_schemeA_cascade_matrix.sh
./experiments/run_schemeA_cascade_matrix.sh
```

默认会写入：

`results_probe_independent/schemeA_3x2x3_<时间戳>/`

子目录名形如 `cascade_s0.2_t0.2_seed42` 与 `cascade_rev_s0.1_t0.2_seed43`。

可调环境变量（可选）：

- `OUT_ROOT`：输出根目录
- `MAX_TRAIN` / `MAX_EVAL`：训练与 val 上探针+BLEU 的 batch 上限（默认 32 / 8）
- `SEEDS`：覆盖种子列表
- `CONDA_ENV`：默认 `tlb_mine`

批跑结束会自动调用 `experiments/aggregate_schemeA_results.py` 生成：

- `schemeA_runs.csv`：每 run 一行（BLEU、train loss/PPL、enc₀ 等）
- `schemeA_summary_by_config.csv`：按 (order, α_s, α_t) 的 **3 seed 均值/标准差**

## 4. 每条 run 中应留存的产物

- `result.json`：BLEU、train 指标、**四类关联系数**、探针 `probe_summaries` 等
- `probe_data/*.csv`、可选 `probe_eval_batch_rows.json`
- `probe_plots/mech_*.png` 等

## 5. 与 v_smooth 文档的关系

`experiments/v_smooth_experiments_summary.md` 中的「多 seed 验证」是 **v_smooth 单法** 的延伸；**本方案 A** 是 **级联双平滑 + 探针+MIC** 的完整实验矩阵，结果目录与 v_smooth 的 `results_v_smooth_*` 独立。
