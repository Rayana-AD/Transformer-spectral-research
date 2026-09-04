# TLB-Experiments → GitHub 研究型仓库规划

> **目的**：暂不投稿论文，将本项目整理为可公开的研究档案（research archive），保留完整科学过程——含正面与负面路径。  
> **与旧 `repr-toolkit-public.zip` 的区别**：那是求职用**脱敏工具包**；本规划是**完整研究叙事 + 可复现资产**，不刻意隐藏「曾写稿/被拒」的事实，但**不含**可直接 deanonymize 的投稿材料。

---

## 0. 项目过程梳理（与你的记忆对照）

### 初始动机
拆解 Transformer 底层机理；假设：**智能与信息扭曲很大程度来自激活函数（FFN 非线性）在深度上的累积**，而非仅来自 QK 路由。

### 演化时间线

| 阶段 | 时间（目录日期） | 假设 / 动作 | 脚本 / 模块 | 结果 |
|------|------------------|-------------|-------------|------|
| **I. 合成谱理论** | 2026-04 初 | 点式 σ 改变 rank / nuclear / κ；深度放大 | `script1`–`script4`, `activation_extras.py` | ✅ 进入正文 §2；支撑「谱探针语言」 |
| **II. 预训练几何桥** | 2026-04 中 | ViT 层wise c_ℓ、与 depth 纠缠 | `script5`–`script7`, `experiments/pretrained_random_activation_geometry_overlay.py` | ⚠️ 观测性；附录/讨论 |
| **III. 直接改 Q/K/V 谱** | 2026-04-22 | 规整核范数、K⊙V、Q 缩放 | `tlb_attention_core.py`, `marian_attn_tlb_patch.py`, `results_marian_qkv_proj_nuc/` 等 | ❌ **崩溃**（BLEU→0）→ Table collapse |
| **IV. V 平滑网格** | 2026-04-23 | 只动 value 路径、不动 QK | `marian_mt_finetune_tlb.py`, `results_v_smooth_grid_*` | ⚠️ 部分有效 → 引向 cascade |
| **V. Scheme A 级联** | 2026-04-24–25 | spatial×temporal 两阶段平滑 + 探针 | `tlb_value_smooth.py`, `run_probe_independent.py`, `results_probe_independent/schemeA_*` | ✅ **论文主线** |
| **VI. 架构分叉（负面）** | 2026-04-29+ | 正交记忆通道、FFN 中段拆、Mirror 支路 | `script8`–`script11`, `memory_tasks.py` | ❌/探索：dual≈baseline；toy 上无 MT 增益 |
| **VII. 审稿回应** | 2026-06-30+ | causal-safe、正则基线、Qwen 前缀、PPL 案例 | `experiments/run_*`, `rebuttal.md`, `results_rebuttal/` | 支持修订；**暂不公开发表** |

### 你记忆中的几个点 — 文件位置

| 记忆 | 是否做过 | 位置 |
|------|----------|------|
| 正交信息通道保「记忆力」 | ✅ 做过，未赢 | `script8_dual_space_memory.py`；`results_script8_*`；设计笔记 `3cursor_orthogonal_dual_space_scattering.md` |
| Q 质量 / Q–K 耦合 | ✅ **做过**，非主线 | `tlb_attention_core.py`（`kv_inner`, `nuc_inter_q`, `qv_inner`…）；`script11_tlb_attention.py`；`results_marian_bleu_quick/`（kv_inner BLEU≈0.15）；`EXPERIMENT_ALIGNMENT_GUIDE.md`（线 A/B） |
| 直接改/规整核范数提表现 | ✅ 做过，**不行** | `results_marian_qkv_proj_nuc/summary.json`；`results_marian_reverse_experiments_*` |
| V 平滑试多种直到定稿 | ✅ | `results_v_smooth_*`（~20GB 权重，公开只保留 CSV）；`experiments/v_smooth_experiments_summary.md` |
| temporal/spatial 顺序 ↔ 机理 | ⚠️ **实验有、理论浅** | Table 4 + `schemeA_full_report.md`；未单独 FFN 干预 |
| 锁定问题在 Decoder + FFN | ⚠️ **部分** | Enc/Dec 分跑 `--probe-target`；`tab:decoder-grad-cka`；**无** FFN 直接因果分解 |

---

## 1. 双盲策略：现在公开，将来再投稿？

| 策略 | 可行性 | 说明 |
|------|--------|------|
| **现在 public 研究 archive** | ✅ 推荐 | 标题避免 venue；不写「under review」；作者可匿名或实名（研究档案常见） |
| **将来要双盲再投** | ⚠️ 需计划 | 公开后 **arXiv/GitHub 可被搜到** → 多数 venue **不再算双盲**。若将来投稿：**新 preprint 前改 private**，或 **大幅改写+新实验** 当新稿 |
| **公开 → 再 private** | ✅ 技术可行 | GitHub 可 `make private`；已 fork/clone 的副本无法收回 |
| **本规划建议** | | Public = **研究过程档案**（含负面结果）；若再投 = **新 repo private fork** 或 **私有分支** 做修订，勿假设「同一 public repo 还能双盲」 |

**不应公开（即使研究 archive）：** `rebuttal.md`, `Review-Cascade.md`, `acl_latex.tex` 中带作者块、`*cursor*` 600KB 策略日志（可节选消毒后放入 Part B）。

---

## 2. 仓库顶层结构（两部分）

```
transformer-spectral-research/     # 建议公开 repo 名（中性，无 Cascade/SchemeA）
├── README.md                      # 研究总述 + 时间线 + 如何导航
├── RESEARCH_LOG.md                # 叙事版过程（本文档精简版）
├── CITATION.cff                   # 可选；Zenodo DOI 以后再加
├── requirements.txt
├── .gitignore                     # 见 §5
│
├── part-a-mainline/               # 论文主线（可复现 claim，非「要投稿」）
│   ├── README.md
│   ├── 01-theory-scripts/         # script1-4 + activation_extras + paper_style
│   ├── 02-vit-bridge/             # script5,7 + overlay experiment
│   ├── 03-marian-cascade/         # tlb_value_smooth, run_probe, marian_* patch
│   ├── 04-data/                   # data/opus_de_en/ (~1.4MB)
│   ├── 05-results/                # CSV/JSON/PDF/PNG only（无 checkpoint）
│   ├── 06-paper-draft/            # 可选：匿名化 tex 或仅 README 指向本地 private
│   └── docs/
│       └── EXPERIMENT_RESULTS_PAPER_FIGURES.md
│
└── part-b-explorations/           # 负面 / 分叉 / 时间线
    ├── README.md                  # 按时间排序的「假设 → 实验 → 结论」
    ├── timeline/
    │   ├── 01-dual-memory/        # script8 + summaries
    │   ├── 02-qkv-spectral-edits/
    │   ├── 03-vsmooth-search/
    │   ├── 04-arch-forks-9-11/
    │   └── 05-rebuttal-ablations/ # 仅 summary，无 rebuttal 全文
    ├── code/                      # script8-11, tlb_attention_core, memory_tasks
    ├── summaries/                 # 各 results_*/summary.json 聚合
    └── notes/
        ├── EXPERIMENT_ALIGNMENT_GUIDE.md
        └── v_smooth_experiments_summary.md
```

**旧 `repr-toolkit-public/`（Desktop）**：保留为 **求职工具子集**；与本 research repo **分 repo 或 monorepo 的 `tooling/` 子目录**。

---

## 3. Part A — 论文主线：文件清单

### 3.1 理论核心 scripts

| 文件 | 作用 |
|------|------|
| `script1_activation_visualization_invariants.py` | 单步激活热力图 + invariants |
| `script2_universal_validation.py` | 宽度×激活 sweep → Table 1 |
| `script3_depth_scattering.py` | 深度 κ 网格 → Fig 2a |
| `script4_scattering_transformer_training.py` | 玩具 Transformer 去噪 → Fig 2b |
| `activation_extras.py` | Smish, LightDepth, 光深激活 |
| `paper_style.py` | 绘图样式 |

### 3.2 ViT 桥（观测）

| 文件 | 作用 |
|------|------|
| `script5_vit_representation_svd.py` | 冻结 ViT 层 SVD |
| `script7_vit_criticality.py` | ViT + CIFAR 探针 |
| `script5_vit_norms_combined_panel.py` | 合并图 |
| `experiments/pretrained_random_activation_geometry_overlay.py` | Marian/ViT c_ℓ overlay |

### 3.3 Marian 级联（机制主实验）

| 文件 | 作用 |
|------|------|
| `tlb_value_smooth.py` | spatial/temporal + causal-safe |
| `marian_attn_tlb_patch.py` | HF Marian 注入 |
| `run_probe_independent.py` | Scheme A 探针 + 训练 |
| `marian_mt_finetune_tlb.py` | 完整 800/1epoch finetune |
| `probe_mech_plots.py`, `probe_visualization.py` | 图 |
| `mt_train_tlb.py` | 数据加载 `load_local_fixed_pairs` |
| `experiments/run_schemeA_*.sh` | 批量跑 |
| `experiments/circular_shift_leakage_demo.py` | 泄漏 toy |
| `experiments/verify_cascade_spectral_operators.py` | 算子数值验证 |

### 3.4 数据

| 路径 | 大小 | 说明 |
|------|------|------|
| `data/opus_de_en/train_5k.jsonl` | ~1.4MB | 实验用前 800 句 |
| `data/opus_de_en/val_500.jsonl` | | 实验用前 128 句 |
| `data/cifar-10-*` | ~340MB | 仅 Part A 若含 script7 exp3；可改 `git lfs` 或文档说明自下载 |

### 3.5 结果（建议打包：CSV/JSON/PDF/PNG，无 `.pt`/无 `hf_finetuned_*`）

| 路径 | 对应论文 |
|------|----------|
| `paper_assets/` | 主图主表 PDF/PNG |
| `results_script2/universal_summary.csv` | Table 1 |
| `results_script3/depth_summary.csv` + 图 | Fig 2a |
| `results_script4/training_loss_*.csv` + 图 | Fig 2b |
| `results_script7/exp3_vit_cifar/` | ViT 图 |
| `results_random_init_activation_geometry/` | c_ℓ overlay |
| `results_probe_independent/schemeA_3x2x3_20260424_0137/` | Enc 矩阵 |
| `results_probe_independent/schemeA_dec_3x2x3_20260425_1943/` | Dec 矩阵 |
| `results_probe_independent/paper_figures/` | schemeA 论文图 |
| `results_probe_independent/schemeA_dec_causal_ablation_20260630_1812/` | causal 表 |
| `results_marian_ffn_frozen_eval/` | collapse 表 |
| `results_prefix_invariance/` | causal 验证 |
| `results_cascade_spectral_verify/` | 算子 verify |
| `results_ppl_spike_analysis/ppl_spike_bad_cases.md` | PPL 案例 |

**体积提示**：`results_probe_independent/` ~88MB 可整包；**勿**复制 `results_v_smooth_*`（3–8GB checkpoint）。

### 3.6 论文稿（建议）

| 选项 | 做法 |
|------|------|
| **A. 不公开 tex** | Part A 只放 `docs/figure_table_map.md` + 图 PDF |
| **B. 匿名 tex** | 删 author/ack；放 `part-a-mainline/06-paper-draft/` |
| **C. Private 第二 repo** | 完整 `acl_latex.tex` + rebuttal 永不上 public |

---

## 4. Part B — 负面实验：文件清单

### 4.1 代码

| 文件 | 假设 | 结论 |
|------|------|------|
| `script8_dual_space_memory.py` | 正交 mem + smart 双流 | dual ≈ baseline（见 `results_script8_longseq_*`） |
| `script9_ffn_bottleneck_dual.py` | FFN 中段拆双路 | 无 clear win（`results_script9_*`） |
| `script10_token_mirror.py` | 并行低秩 mirror 支路 | toy 探索（`results_script10_*`） |
| `script11_tlb_attention.py` | TLB Q/K 模式 | toy；`results_script11_*` |
| `tlb_attention_core.py` | kv_inner, nuc_inter_q 等 | Marian 上多 collapse |
| `memory_tasks.py` | copy 任务构造 | script8/9/10 共用 |

### 4.2 关键 summary（只复制 JSON/CSV，不复制 GB 权重）

| 路径 | 一行结论 |
|------|----------|
| `results_marian_qkv_proj_nuc/summary.json` | QKV 核范数规整 → BLEU≈0 |
| `results_marian_bleu_quick/summary.json` | kv_inner finetune 失败 |
| `results_marian_ffn_frozen_eval/summary.json` | 换 FFN 激活 → 崩溃 |
| `results_marian_reverse_experiments_20260422/` | Line-C / reverse 对照 |
| `experiments/v_smooth_experiments_summary.md` | V 平滑搜索过程 |
| `results_script8_*/multi_seed_summary.csv` | dual 未超 baseline |
| `results_script9_*/note.md` | 各 run 备注 |

### 4.3 讨论笔记（可选公开节选）

| 文件 | 建议 |
|------|------|
| `EXPERIMENT_ALIGNMENT_GUIDE.md` | ✅ 公开 — 探针 vs 实验区分 |
| `Discussion_Conclusion.md` | ✅ 可公开 |
| `docs/VIT_MARIAN_ROOT_CAUSE.md` | ✅ 技术讨论 |
| `1cursor_*.md`, `rebuttal.md`, `Review-Cascade.md` | ❌ 私有或大幅删减 |
| `portfolio_sandbox/` | 求职用，**不进** research repo |

### 4.4 Part B README 建议结构（`RESEARCH_LOG.md`）

```markdown
## 假设 H1：激活深度累积 → 谱病态
- 脚本：1–4 | 结果：positive | 见 part-a/01-theory-scripts

## 假设 H2：正交记忆通道
- 脚本：8 | 结果：negative | BLEU/acc 未超 baseline

## 假设 H3：改 Q/K 耦合可替代 V 平滑
- 模块：tlb_attention_core, script11 | 结果：negative | kv_inner collapse

## 假设 H4：直接核范数约束/投影
- results_marian_qkv_proj_nuc | 结果：negative

## 假设 H5：仅 V 路径两阶段平滑（Scheme A）
- 见 part-a/03-marian-cascade | 结果：stress-test 下可测几何；非 SOTA MT

## 开放问题（未理论闭合）
- spatial/temporal 顺序 ↔ 优化路径（Enc order-sensitive, Dec 不敏感）
- Decoder fragility vs FFN 因果分解
- 800/128 规模与 32-batch probe 协议并存
```

---

## 5. `.gitignore` 建议

```gitignore
.venv*/
__pycache__/
*.pt
*.pth
**/hf_finetuned_*/
results_v_smooth_*/
results_final_*/
results_marian_bleu_m1run/
*.pdf
!paper_assets/**/*.pdf
!part-a/**/figures/**/*.pdf
```

（PDF 规则按你是否公开论文图调整。）

---

## 6. 与现有目录的关系

| 现有路径 | 处置 |
|----------|------|
| `/Users/rayana/Desktop/TLB-Experiments/` | **保持为工作母本**；不强制搬家 |
| `/Users/rayana/Desktop/repr-toolkit-public/` | 求职工具；独立 repo |
| `/Users/rayana/Desktop/TLB-Experiments/portfolio_sandbox/` | 打包实验隔离区；可保留 |
| **新建** `/Users/rayana/Desktop/github-research-staging/` | 按本规划轻量 staging → zip / git push |

---

## 7. 下一步（Agent 可执行）

1. 创建 `github-research-staging/` 目录树 + `MANIFEST.yaml`（路径清单，不复制大文件）
2. 运行脚本：按 MANIFEST 复制 CSV/JSON/小图 + 代码（symlink 或 copy）
3. 生成 `part-b/timeline/README.md` 自动索引 `results_*/summary.json`
4. **不**默认 push；你确认后再 `git init` + GitHub

---

*文档版本：2026-09-04 · 基于仓库全量扫描与 script1–11 对照*
