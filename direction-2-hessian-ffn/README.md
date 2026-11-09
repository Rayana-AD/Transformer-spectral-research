# 方向 2：Hessian 形式低秩 FFN（研究原型）

在标准 MLP-FFN 的 GeLU 路径上，引入 **与 Hessian / 曲率结构相关的低秩分解**（如可学习谱分支、\(L(z)L(z)^\top\) 型秩约束、ctx-split 读写），在 **记忆、因果问答（语言建模）、视觉** 三类任务上检验 **表示能力与优化几何**，而非单一 BLEU 分数。

> **性质**：原型展示为主。完整论文实验 **未** 纳入本仓库；见 [PROTOTYPE-NOTICE.md](PROTOTYPE-NOTICE.md)。

---

## 方法（一句话）

**Hessian-shaped FFN**：将 FFN 中间态 \(z\) 的演化写成可与 **二阶信息 / 低秩因子** 对齐的形式；通过 **ctx-split**（上下文分裂读写）、**谱门控** 与 **秩受限分支**，在保持参数量可控的前提下，显式调节 **层间秩轨迹与记忆容量**。

核心实现入口（同步自 `TLB-Algorithm/tlb/`）：

| 模块 | 路径 | 作用 |
|------|------|------|
| Hessian FFN 变体 | `tlb/ffn/hessian_core.py` | H1–H4 及 ctx-split / 低秩分支定义 |
| 标准 FFN 对照 | `tlb/ffn/standard.py` | 基线 MLP |
| 因果 LM 组装 | `tlb/models/causal_lm.py` | WikiText / 流式预训练壳 |
| Ctx-split 初始化 | `tlb/models/gpt2_ctxsplit_init.py` | 分裂读写结构 |
| 记忆几何探针 | `tlb/training/memory_track_probe.py` | 轨道秩、迹链指标 |
| 融合内核（可选） | `tlb/kernels/hessian_ffn_fused.py` | 性能实验用 |

---

## 任务模块（本仓库提供的检验入口）

以下脚本 **面向原型验证**，默认 **smoke 规模**；完整规模请在本地 `TLB-Algorithm` 配置集群上运行。

### 1. 记忆（Memory track）

- **目的**：检验 FFN 低秩分支是否改变 **序列记忆容量与遗忘曲线**。
- **入口**：`experiments/run_memory_capacity_sweep.py`、`experiments/run_sequential_forgetting_v2.py`（若已同步）
- **探针**：`experiments/plot_memory_track_geometry.py` → 示意输出在 `figures/memory_track_geom/`

### 2. 因果问答 / 语言建模（Causal LM）

- **目的**：在 **WikiText-103 等** 因果 LM 上对比 standard FFN vs Hessian / ctx-split FFN 的 **loss、主动秩、层间协调**。
- **入口**：`experiments/l4_causal_wt103_single.py`、`experiments/run_wt103_d256_lr_smoke.py`
- **配置示例**：`configs/wt103_scale_smoke/manifest.yaml`

### 3. 视觉（Vision）

- **目的**：将 **ViT / Galaxy-10 / CIFAR** 等视觉骨干的 FFN 替换为 Hessian 变体，检查 **准确率–速度–秩指标**。
- **入口**：`experiments/l_vit_cifar_smoke.py`、`experiments/l_vit_galaxy10_aion_ft_fast.py`
- **模型桥**：`tlb/models/vit_ffn_transfer.py`

---

## 快速运行（smoke）

```bash
cd direction-2-hessian-ffn
pip install -r requirements-snippet.txt

# 合成 / 数值 smoke（CPU 可）
python experiments/l0_smoke.py

# WT103 小规模对照（需数据，见 configs）
python experiments/run_wt103_d256_lr_smoke.py --help
```

数据路径与 Colab 大包 **不** 随仓库存储；请在本机 `TLB-Algorithm/data/` 按 `docs/COLAB_WT103_SCRATCH.md`（源仓库）准备。

---

## 与方向 1 的关系

| | 方向 1（value cascade） | 方向 2（Hessian FFN） |
|--|-------------------------|------------------------|
| 干预位置 | Attention **V** 路径 | **FFN** 块 |
| 主要问题 | 小数据微调时 **Enc/Dec 几何是否稳定** | 低秩 Hessian 结构能否支撑 **记忆与多模态任务** |
| 本仓库角色 | 完整诊断档案 + 图 | **原型代码 + smoke** |

---

## 同步源码

从作者本地算法树更新本目录：

```bash
bash scripts/sync_from_tlb_algorithm.sh
```

源路径默认：`~/Desktop/TLB-Algorithm`。
