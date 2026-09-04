# 研究过程日志（叙事版）

> 与 `part-a-mainline/docs/RESEARCH_GITHUB_PLAN.md` 互补：本文偏**可读叙事**；后者偏**文件清单与打包规则**。

## 起点

拆解 Transformer 底层机理。核心假设：**深度上的信息扭曲主要来自 FFN 点式激活对特征矩阵谱（rank、核范数、条件数）的累积作用**，而非仅来自 QK 注意力路由。为此设计 script1–11，从合成层wise metrics 到 Marian/ViT 探针。

## 时间线

### I. 合成谱理论（script1–4）→ Part A

| 脚本 | 问题 | 结论 |
|------|------|------|
| script1 | 单步激活 + 不变量可视化 | 建立「谱探针」语言 |
| script2 | 宽度×激活 universal sweep | 进入正文 Table 1 |
| script3 | 深度 × κ 网格 | Fig 2a |
| script4 | 玩具 Transformer 去噪训练 | Fig 2b |

**位置**：`part-a-mainline/01-theory-scripts/`；结果 CSV 在 `05-results/results_script2–4/`

### II. ViT 桥（script5–7）→ Part A（观测性）

冻结 ViT 层wise SVD、c_ℓ 与 Marian 几何 overlay。**理论未深入**，支撑「预训练几何 ↔ 激活扭曲」的讨论。

**位置**：`part-a-mainline/02-vit-bridge/`；`part-b-explorations/summaries/VIT_MARIAN_ROOT_CAUSE.md`

### III. 直接改 Q/K/V 谱 → Part B（失败）

- **规整核范数 / 全局 QKV rescale**：BLEU → 0（`results_marian_qkv_proj_nuc/summary.json`）
- **Q–K 耦合 / kv_inner 等**（Line A/B）：`tlb_attention_core.py`，`script11`；`results_marian_bleu_quick/summary.json` 中 kv_inner ≈ 0.15 vs vanilla ≈ 9.7
- **结论**：动 QK 路由或硬改投影谱会崩溃；引向 **只动 value 路径**

**你问「Q 质量实验是否做过」**：✅ 做过，见 `EXPERIMENT_ALIGNMENT_GUIDE.md`

### IV. V 平滑搜索 → 部分进 Part A，过程在 Part B

多种 spatial/temporal、强度网格 → `results_v_smooth_*`（**checkpoint 不公开**，~20GB+）。摘要：`experiments/v_smooth_experiments_summary.md`

### V. Scheme A 级联（论文主线）→ Part A

`tlb_value_smooth.py` + `run_probe_independent.py` + Scheme A shell 矩阵。

**未完全理论化的点**：
- **Spatial vs temporal 顺序**：Table 4 / `schemeA_full_report.md` 有实验；Encoder 顺序敏感、Decoder 顺序不敏感 → 优化路径 vs 算子交换性，**无 FFN 单独因果分解**
- **Decoder 侧问题**：Discussion 叙事 + grad/CKA 表 + Fig 31–32 不对称；**最佳 BLEU 上 Encoder 仍略高**；「锁定 Decoder」= 诊断结论，非单一 ablation 铁证

### VI. 架构分叉（script8–11）→ Part B（负面/探索）

| 假设 | 脚本 | 结果 |
|------|------|------|
| 正交记忆通道 | script8 | dual ≈ baseline，无可靠记忆优势 |
| FFN 中段双空间 | script9 | toy/探索，无 MT 增益 |
| Token mirror | script10 | 合成 LM，未迁移 Marian |
| TLB attention 模式 | script11 | 与 Q/K 实验同族 |

### VII. 审稿回应（2026-06+）→ 仅 summary 进 Part A

Causal-safe、正则基线、前缀泄漏 demo。**rebuttal 全文、Review、带作者 tex 不公开**。

## 两套 finetune 协议（读结果时必知）

1. **`marian_mt_finetune_tlb.py`**：800 pairs、~100 steps → vanilla BLEU ~15.5（stress-test）
2. **`run_probe_independent.py`**：默认 32 batches → vanilla ~18.16；Scheme A 矩阵多用此协议

公开仓库 README 中应写明，避免复现者混淆。

## 导航

- **主线**：根目录 [README.md](README.md) → `part-a-mainline/`
- **负面路径**：`part-b-explorations/README.md`
- **复现**：[REPRODUCTION.md](REPRODUCTION.md)
- **同步**：`scripts/copy_from_tlb.sh` + `scripts/supplement_repro_assets.sh`

源项目 dedup 日志：`TLB-Experiments/docs/DEDUP_LOG.md`
