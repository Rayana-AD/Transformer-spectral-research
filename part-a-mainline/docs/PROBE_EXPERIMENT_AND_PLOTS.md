# 级联探针实验设置与作图说明

本文说明 `run_probe_independent.py` 中 **独立式级联探针** 的实验设定、输出目录、关联系数含义，以及 `probe_plots/` 下各 **mech\_\*** 系列图的读图方式（坐标轴、色标、对应关系与常见趋势）。重画无需重训时，可用 `probe_replot_from_saved.py` 从 `probe_data/` 与 `result.json` 再生成图。

---

## 1. 实验脚本与主要设定

- **入口脚本**：`run_probe_independent.py`  
- **作用**：在 MarianMT（或兼容路径）的 encoder self-attention 的 **4D value** 上注入 **三阶段级联**（$V^{\mathrm{raw}} \to V^{(1)} \to V^{(2)}$），在 **训练步** 上记录每层探针，在 **多 batch 验证** 上记录 eval 探针，并与 **loss / PPL、BLEU** 等标量作关联。  
- **调度方式**：`transformers` 若提供 `eager_attention_forward` 则挂接；否则 **patch `MarianAttention.forward`**。需能拿到 eager/可注入路径上的 $V$ 张量。  
- **与本文档最相关的常见参数**（见脚本 `--help`）：  
  - `alpha_spatial` / `alpha_temporal`：级联中空间/时间方向强度。  
  - `max-train-batches`：训练上最多记录多少 **minibatch 步**的探针；步数少则曲线/相关只有很短的序列。  
  - `max-eval-batches`：验证上取多少个 **batch** 的 teacher-forcing 探针 + 每批 **corpus BLEU**；作 **batch 序上的相关** 时，batch 数不宜过小（建议与脚本注释一致，例如 ≥4）。  
  - `--plot-probes` / `--export-probe-csv` / `--save-eval-probe-json`：是否写 `probe_plots/`、`probe_data/`，及是否把完整 eval 行存成 JSON 便于事后作图。  
  - `train-plot-stride`：**仅作图**时对训练行抽稀，不改变 CSV/原始 `measurements`。

**输出布局**（每次实验子目录，例如 `results_probe_independent/<exp_name>/`）：

| 路径 | 内容 |
|------|------|
| `result.json` | 标量结论文本、**关联系数**、探针 summary 等。 |
| `probe_data/probe_train_enc_*.csv` | 各层随 **step/loss** 的逐段指标。 |
| `probe_data/probe_eval_*.csv` | 各层随 **eval batch 序号 / BLEU** 的批级指标。 |
| `probe_data/probe_eval_batch_rows.json` | （可选）完整 eval 行，含多层的嵌套 `layers` 结构。 |
| `probe_data/mech_correlation_table.tsv` | 与热力图同源的关联系数表。 |
| `probe_plots/mech_*.png` | 本文件下一节所描述的作图。 |

---

## 2. 探针、级联与关联系数

- **三阶段**（在每一 encoder 层、每个 attention 步内）：$V^{\mathrm{raw}}$, $V^{(1)}$, $V^{(2)}$。默认作图在 **nuclear 范数** 上叙事较多，与 CSV/汇总中的 **temporal/spatial 方差、cosine 等** 一起写入。  
- **训练侧**：在固定 **时间序** 上，各层每个标量与 **同一步的全局 loss** 配对，计算秩/线性/依赖类关联（见下）。  
- **Eval 侧**：在 **val batch 序** 上，各标量与 **同批的 corpus BLEU** 配对，计算同类关联。  
- **关联系数**（与 `result.json` 中 `train_assoc_loss_vs_metrics_by_layer`、`eval_assoc_bleu_vs_metrics_by_layer` 一致；亦写入 `mech_correlation_table.tsv`）：  
  - **Spearman** $\rho$：秩相关，对单调查量、少量离群点较稳。  
  - **Pearson** $r$：线性相关。  
  - **Kendall** $\tau$：秩相关（另一系）。  
  - **MIC**（Maximal Information Coefficient）：检测非线性强弱，取值约 $[0,1]$。需安装 **minepy**；未安装或样本不足时 **全为 NaN**，程序 **不再写出** 对应热力图，并会删除过期的 `*__mic.png`，避免出现 **空白整图**。

---

## 3. 各系列图件读图说明

下列文件名中的 `{aggregate}` 多为 `time-mean` / `last` 对应的 **`mean` / `last`**；`{name}` 为 `enc_0`…等层名。默认 **不** 打开「以 train step 或 eval batch 为主横轴」的旧式全套图，除非在脚本中显式打开 legacy 选项。

### 3.1 沿深度与「层内三阶段」：train / eval

| 文件模式 | 画什么 | 横纵轴与读图 |
|----------|--------|----------------|
| `mech_train_depth_cascade__nuc__{mean,last}.png` | 训练上：各 **encoder 层** 在一种聚合（对时间平均或取最后一步）下，**三阶段** 的 **nuclear** 随 **层下标** 的折线。 | **横轴**：层（0…L−1 或 `enc_*` 标签）。**纵轴**：nuclear。**三条线/色**：$V^{\mathrm{raw}}$, $V^{(1)}$, $V^{(2)}$。  
| `mech_train_stage_path_all_layers__nuc__{mean,last}.png` | 同一单图：对每一层，在 **0→1→2 阶段** 上连一条折线；多 **层** 多色线。 | **横轴**：阶段 0,1,2（数学上等价于子图里标注的三阶段名）。**纵轴**：nuclear。看**单层**级联形变、**层间**谁高谁低。  
| `mech_eval_depth_cascade__nuc__{last,mean}.png` | 与 train depth 同构，但用 **val 多 batch** 在 **各层、各阶段** 上聚合成一条「沿深度」的线（**last=最后一批** 或 **mean=跨批平均**）。 | **横轴**：层；**纵轴**：nuclear。理解 **与 BLEU 分轨**：此图不直接画 BLEU，但可与 batch 级表对照。  
| `mech_eval_stage_path_all_layers__nuc__{last,mean}.png` | eval 上「层内 0→1→2」与 train 的 stage_path 对偶。 | 读法同 train 版。 |

**如何理解趋势**：`mean` 更稳、`last` 更贴近停训/当前 checkpoint。深度图上若**浅层/中层** 某阶段总抬高，说明级联在深度上的「堆积」与实验设定（$\alpha$、层数）一致；stage_path 若三线随阶段 **稳定下降**，与「范数在级联中收缩」的叙事一致（具体以你跑出的曲线为准）。

### 3.2 与 loss、PPL：训练侧池化/全局/hexbin

| 文件模式 | 画什么 | 读图 |
|----------|--------|------|
| `mech_train_pooled_nuclear_vs_L_PPL__enc_*.png` | **每层一图**；**横轴**=该层该步在**三阶段**上各自的 **nuclear**，**上子图**纵轴=全局 **$\mathcal{L}$**，**下子图**=**PPL**；时间序在 (nuclear, L) 或 (nuclear, PPL) 平面用折线**按时间连接**。三条折线=三阶段，**颜色+线型**区分。 | 看**同一层** 下核范数与 loss 是否**同向**随训练走动；PPL 与 loss 单调对应（PPL$=\exp L$ 尺度假象）。不叠其他层，避免「意面图」。步数少时轨迹短。  
| `mech_train_global_L_PPL_Vraw_enc0.png` | 仅 **enc\_0** 的 $V^{\mathrm{raw}}$ nuclear 与**全局** loss、PPL 对 **train step**；三行子图。 | **横轴**：step（小批次序）。**纵轴**：$\mathcal{L}$、PPL、nuclear。loss 各层相同，只出一张。 |
| `mech_train_cascade_nuclear_L_PPL__{name}.png` | **每层** 2×3：**行**= $\mathcal{L}$ 与 PPL，**列**=三阶段；**hexbin** 核范数与 loss/PPL 的联合密度。 | 看该层在训练时间上 **nuclear 与 L/PPL** 的**联合分布**；颜色为密度（多探针点时 bin 多）。  
| `mech_train_loss_ppl__{name}.png` | （**legacy**）按层的 loss/PPL 对 step 等；若未开 legacy 可能不存在。 | 以脚本开关为准。 |

**色标/密度**：hexbin 图右侧 colorbar 表示 **每 bin 内点数**（或脚本内 `colorbar` 标签）；与热力图的「相关系数」**不是**同一类色标。

### 3.3 Eval：BLEU 与多层的 legacy 图

| 文件 | 画什么 | 读图 |
|------|--------|------|
| `mech_eval_bleu_per_batch.png` | （legacy）**batch 序号 → BLEU** | 看验证 batch 间波动。 |
| `mech_eval__V_raw__nuclear__all_layers.png` 等 | （legacy）指定 **阶段/指标** 在 **多 encoder 层** 上对 **batch 序** 的曲线。 | 横轴为 eval **batch 索引** $j$；纵轴为探针；多条线为各层。 |
| `mech_eval_cascade_nuclear__{layer}.png` | 单层上 **三阶段** nuclear 对 **$j$** | 看该层随 batch 的级联与 BLEU 同屏叙事时需叠看 `bleu` 或 `result.json` 的 `eval_per_batch_bleu`。 |
| `mech_eval_depth_cascade` / `mech_eval_stage_path` | 已见 3.1。 |

### 3.4 关联系数热力图（`mech_corr_heatmap_*`）

- **文件**：`mech_corr_heatmap_train_loss__{spearman,pearson,kendall_tau}.png`（若 `mic` 有有限值会多 `mic`）。eval 为 `mech_corr_heatmap_eval_bleu__*`，命名同理。无嵌套旧结果时，可能仅有 `mech_corr_heatmap_train_loss.png` / `mech_corr_heatmap_eval_bleu.png`（仅 Spearman）。  
- **布局（当前版本）**：**1×3 子图**，左→右：$V^{\mathrm{raw}}$, $V^{(1)}$, $V^{(2)}$。  
- **横轴（列）**：**encoder 层**，刻度为 **0…L−1**（不再用 `enc_0` 下划线形式）。  
- **纵轴（行）**：仅保留 **与「标量/均值类」** 探针对应的行（nuclear、element mean、$L^2$、及各类 `*_mean`）；**不含** `*std*`。  
- **色标与含义**：  
  - Spearman / Pearson / Kendall：一般为 **$[-1,1]$**，**`RdBu_r`（或同级发散色图）**——**越红**越正相关，**越蓝**越负相关，**白**近 0。  
  - **MIC**：**$[0,1]$**，`viridis` 类**顺序色图**；值越大表示**可检测的非单调依赖**越强。无有效 MIC 时**不写该文件**（并删除历史空白 `*__mic.png`）。  
- **关系在衡量什么（train）**：每个格子 = 在**该层、该满名指标** 的时间序列 与 **同一步的 loss 序列** 的关联系数。  
- **关系在衡量什么（eval）**：每个格子 = **该层该指标** 的 **按 batch 序** 序列 与 **同批 BLEU 序列** 的关联系数。  
- **读趋势**：在 **nuclear / 大范数** 行常可看到与 loss 的**负/弱**关系（与「大范数是否利于收敛」的假设对读）；在 **方差/波动类** 行上信号可能更强。**batch 数太少** 时，eval 整图会**不稳定**，宜结合 `max-eval-batches` 与 TSV 看数值。  
- **图幅**：已压缩单栏宽度、略增高，使 **1×3** 不显得**过扁、过宽**；若行数多，仍建议 PDF/浏览器中放大看。

### 3.5 与 `result.json` 的汇总条图（若单独调用）

- `mech_from_result__layer_mean_nuclear_cascade_bars.png`：由 **探针汇总的** 各层、三阶段 **mean nuclear** 分组柱，**无时间轴**，与随 step 的 CSV 作图**互补**。

### 3.6 表：`mech_correlation_table.tsv`

- 文本表列出 **split（train/eval）**、**层**、**指标名**、**方法**、**数值**。作图时与热力图**同源**；可 grep 比读图更精。

---

## 4. 重画与复现

```bash
python probe_replot_from_saved.py /path/to/results_probe_independent/<exp_name>/
```

- 会读 `probe_data/*.csv` 与（若有）`probe_eval_batch_rows.json`，并从 `result.json` 取关联系数。  
- 若 `result.json` 中某类关联系数**全为 NaN**（如未装 minepy 的 MIC），对应热力图**不会输出**，并**尝试删除**该实验目录下过期的同名片。

---

## 5. 常见问题

- **为什么以前热力图有「空白张」？** 多为 **MIC 全 NaN** 仍用 `imshow` 画出了「全空矩阵」；现改为 **无有效数值则不输出** 并 **删旧 `*__mic.png`**。若需要 MIC，请安装 **minepy** 并保证 **样本点足够**、方差非退化。  
- **我明明 `requirements.txt` 里有 minepy，为什么 `result.json` 里 `mic` 还是 `NaN`？** 说明当前 Python 环境 **没装上或 import 失败**（例如没执行 `pip install -r requirements.txt`，或 `minepy` 编译/依赖问题）。`association_metrics` 里对 `from minepy import MINE` 是 **`try/except: pass`**，失败时 **不会报错**，三列里仍会有 Spearman/Pearson/Kendall，只有 **MIC 保持 NaN**。可在跑实验的同一环境里执行 `python -c "from minepy import MINE; print('ok')"` 自检。  
- **有 minepy 时 MIC 热图却一片深蓝、只零星黄格（0 / 1）？** 默认 **`max_eval_batches=4`** 时，eval 上每个格子只用了 **4 个 (BLEU, 标量) 对** 估 MIC。MINE 在 **极小 n** 下本来就很噪，**容易落在 0 或 1** 这类极端，和 Spearman 的连续取值观感差很多。这不是「没算 Spearman」的 bug；要更可信的 **MIC 热力图** 请把 `--max-eval-batches` 调大（如 8–20）后 **重训那一步的 eval+result.json**（仅重画 CSV 变不了关联系数）。  
- **train 上只有极少的点？** 调大 `--max-train-batches`。  
- **eval 相关很跳？** 调大 `max-eval-batches`，或看 Spearman/秩类是否更稳。  

本文档与代码版本一一对应；若以后增加新 `mech_*` 文件名，请在本表同期增补一行说明。
