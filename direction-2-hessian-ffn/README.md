# 方向 2：Hessian Feed-Forward Network（H-FFN）

**仓库主页**已收录本节的背景、方法、结果与快速开始（见根目录 [README.md](../README.md) 中「方向 2」）。此处便于 **仅浏览方向 2 代码树** 时使用；表述与主页一致。

## 背景

标准 Transformer 块中，FFN 占参数量与算力的主要部分，但结构仍是「扩维—非线性—投影」的静态 MLP，对每个 token 施加同一类映射。将 FFN 视为键值式联想记忆时，**变换表示** 与 **存放事实** 挤在同一条未索引的通道里，容易产生干扰；仅靠加宽中间层或 MoE，也难以显式刻画 **随输入变化的局部曲率**。

本方向把 FFN 写成由标量势 $\Psi$ 诱导的 **Hessian 型算子**：在可学习的投影轴 $u_j$ 上，用二阶门控 $\lambda_j(x)=\psi^{\prime\prime}(u_j^\top x + b_j)$ 调节沿该方向的放大或抑制，整体保持与两层 MLP 相近的渐进计算量，但 **曲率与秩结构** 可解释、可探针。针对两类任务又衍生 **Memory-Track**（多尺度上下文门控，用于联想检索）与 **Decoupled H-FFN**（拆分内容、路由与发射，用于因果语言建模）；预训练迁移场景下可用 **解析 GELU 匹配** 做 FFN 暖启动。

## 方法

$$
\mathrm{H\text{-}FFN}(x)=\mathcal{H}(x)\,x=\sum_{j=1}^{m}\lambda_j(x)\,(u_j^\top x)\,u_j,
\qquad
\Psi(x)=\sum_j \psi(u_j^\top x + b_j).
$$

实现入口：`tlb/ffn/hessian_core.py`（对照 `tlb/ffn/standard.py`）；`tlb/models/causal_lm.py`、`gpt2_ctxsplit_init.py`、`vit_ffn_transfer.py`。本目录为 **可运行原型**，不含完整 checkpoint 与私有数据。

## 结果（概要）

- **记忆 / Memory-Track**：Needle-in-a-Haystack 类探测约 **96% vs 76%** 量级（同预算 MLP）。  
- **因果 LM / Decoupled H-FFN**：WT103 困惑度约 **455 → 388** 量级（约 **1.1–1.4×**）；FineWeb-Edu 约 **1.15×**。  
- **视觉**：GPT-2 FFN 替换约 **1.07×** PPL；ViT-B/16 约 **+2.4% top-1**、更少步数收敛。  

探针图：`figures/memory_track_geom/`。

## 代码与运行

| 类别 | 路径 |
|------|------|
| 记忆 | `experiments/run_memory_capacity_sweep.py`，`tlb/training/memory_track_probe.py` |
| 因果 LM | `experiments/l4_causal_wt103_single.py`，`run_wt103_d256_lr_smoke.py`，`configs/wt103_scale_smoke/` |
| 视觉 | `experiments/l_vit_cifar_smoke.py`，`vit_forward_smoke.py` |
| Smoke | `experiments/l0_smoke.py` |

```bash
pip install -r requirements-snippet.txt && python experiments/l0_smoke.py
bash scripts/sync_from_tlb_algorithm.sh
```
