# 方向 2：Hessian Feed-Forward Network（H-FFN）

## 背景

标准 Transformer 块中，FFN 占参数量与算力的主要部分，但结构仍是「扩维—非线性—投影」的静态 MLP，对每个 token 施加同一类映射。将 FFN 视为键值式联想记忆时，**变换表示** 与 **存放事实** 挤在同一条未索引的通道里，容易产生干扰；仅靠加宽中间层或 MoE，也难以显式刻画 **随输入变化的局部曲率**。

本方向把 FFN 写成由标量势 \(\Psi\) 诱导的 **Hessian 型算子**：在可学习的投影轴 \(u_j\) 上，用二阶门控 \(\lambda_j(x)=\psi''(u_j^\top x + b_j)\) 调节沿该方向的放大或抑制，整体保持与两层 MLP 相近的渐进计算量，但 **曲率与秩结构** 可解释、可探针。针对两类任务又衍生 **Memory-Track**（多尺度上下文门控，用于联想检索）与 **Decoupled H-FFN**（拆分内容、路由与发射，用于因果语言建模）；预训练迁移场景下可用 **解析 GELU 匹配** 做 FFN 暖启动。

## 方法

核心前向可概括为

\[
\text{H-FFN}(x)=\mathcal{H}(x)\,x=\sum_{j=1}^{m}\lambda_j(x)\,(u_j^\top x)\,u_j,
\quad
\Psi(x)=\sum_j \psi(u_j^\top x + b_j).
\]

实现中将 \(m\) 个秩一分支堆叠为低秩 Hessian 形式，并配合 **ctx-split**（上下文分裂读写）、**多尺度曲率阶梯** 与 **Decoupled** 变体；代码主入口为 `tlb/ffn/hessian_core.py`，标准 MLP 对照为 `tlb/ffn/standard.py`。因果 LM、ctx-split 组装与 ViT FFN 替换分别见 `tlb/models/causal_lm.py`、`gpt2_ctxsplit_init.py`、`vit_ffn_transfer.py`。

本目录收录的是 **可运行原型**（smoke 配置与代表脚本），不含完整实验流水线的全部 checkpoint 与私有数据。

## 结果（概要）

在与匹配训练预算的内部对照中，H-FFN 系列在以下三类任务上相对同规模 MLP 基线有一致趋势（具体协议与方差以本地完整实验为准；本仓库不打包主表）：

- **记忆 / 联想检索（Memory-Track）**：Needle-in-a-Haystack 类探测上，同预算下准确率显著高于匹配宽度的 MLP（论文主实验约 **96% 量级 vs 76% 量级**）。
- **因果语言建模（Decoupled H-FFN）**：WikiText-103 等语料上测试困惑度约 **1.1–1.4× 改善**（例如 WT103 约 **455 → 388** 量级）；流式 Web 教育语料上约 **1.15×** 量级。
- **视觉迁移**：GPT-2 FFN 解析替换微调约 **1.07×** PPL；ViT-B/16 上可在 **更少步数** 达到约 **+2.4% top-1** 的收敛表现。

示意性几何探针图见 `figures/memory_track_geom/`（迹链、曲率质量等，用于理解秩轨迹而非充当主结果表）。

## 本仓库包含什么

| 类别 | 路径 |
|------|------|
| 记忆容量与轨道探针 | `experiments/run_memory_capacity_sweep.py`，`tlb/training/memory_track_probe.py` |
| 因果 LM（WT103 smoke） | `experiments/l4_causal_wt103_single.py`，`run_wt103_d256_lr_smoke.py`，`configs/wt103_scale_smoke/` |
| 视觉 smoke | `experiments/l_vit_cifar_smoke.py`，`vit_forward_smoke.py` |
| 数值 / 合成 smoke | `experiments/l0_smoke.py` |

从本地算法树增量同步：

```bash
bash scripts/sync_from_tlb_algorithm.sh   # 默认源：~/Desktop/TLB-Algorithm
```

## 快速开始

```bash
pip install -r requirements-snippet.txt
python experiments/l0_smoke.py
```

WikiText 与 FineWeb 等数据需在本机 `TLB-Algorithm/data/` 自行准备；`run_wt103_d256_lr_smoke.py --help` 列出 smoke 用法。

## 与方向 1 的分工

方向 1 在 **attention 的 value 路径** 上做级联诊断（小数据 Marian 微调）；方向 2 在 **FFN 子层** 用 Hessian 低秩结构处理 **记忆、因果建模与视觉** 中的表示与优化几何。二者共享「谱 / 秩 / 曲率」语言，但 **干预位置与任务设定不同**。
