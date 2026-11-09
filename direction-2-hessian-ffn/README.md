# 方向 2：Hessian Feed-Forward Network（H-FFN）研究原型

## 背景

Transformer 中 FFN 占参数与算力的大头，但长期保持「扩展至约 \(4d\) 再线性投影」的两层 MLP 形态：对所有 token 施加同一静态映射，难以在 **变换隐状态** 与 **可检索的长期关联** 之间兼顾，且缺少显式索引。在注意力侧已有核与几何视角的工作，而 FFN 子层相对缺少同样可解释的 **二阶、低秩** 构造。本方向追问：能否把 position-wise FFN 写成 **由标量势函数导出的 Hessian 作用于输入**，在 **与标准 MLP 相近的渐近成本** 下，把 **曲率门控、多尺度上下文与内容–路由–发射解耦** 写进结构本身。

该线与仓库 **方向 1（Marian value-path 谱诊断）** 相互独立：方向 1 动 attention 的 \(V\) 路径做训练期探针；方向 2 动 **FFN 子层**，面向 **记忆检索、因果语言建模与视觉迁移**。关联会议稿目前审稿反馈不佳、可能撤稿；本目录 **仅公开可验证的原型实现与 smoke 配置**，完整主表与再投稿材料见本地工作区与 [PROTOTYPE-NOTICE.md](PROTOTYPE-NOTICE.md)。

---

## 方法（概要）

**核心算子.** 将标量势 \(\Psi(x)=\sum_j \psi(u_j^\top x + b_j)\) 的 Hessian 缩并作用于 \(x\)：

\[
\text{H-FFN}(x)=\mathcal{H}(x)\,x=\sum_{j=1}^{m}\lambda_j(x)\,(u_j^\top x)\,u_j,\quad \lambda_j(x)=\psi''(u_j^\top x + b_j).
\]

几何上，这是 \(m\) 个 **秩一向外积** 的加权和：沿轴 \(u_j\) 投影，用 **局域二阶曲率** \(\lambda_j\) 缩放后再沿同一轴发射。实现上通过秩受限字典 \(U\in\mathbb{R}^{m\times d}\) 与逐坐标势 \(\psi\) 完成，可与两层 MLP 对齐参数规模。势函数采用有界、带符号的二阶导（如 \(\psi(s)=\tanh^2(s)\)），以在中央正激励与两侧阻尼井之间取得稳定门控，而非单纯替换 GELU 形状。

**两条任务向扩展（与论文叙述一致，细节见 `tlb/ffn/hessian_core.py`）：**

1. **Memory-Track（记忆）** — 面向 **联想检索 / Needle-in-a-Haystack** 类设置，在共轴轨道上叠加 **多尺度上下文门控**，使 FFN 更贴近「可寻址记忆轨」而非无索引宽层。
2. **Decoupled H-FFN（因果语言建模）** — **ctx-split** 等变体将 **内容、路由与发射** 拆开，减轻单一路径上的表示干扰，便于在 WikiText、流式语料等设置上对比标准 FFN 的困惑度与秩轨迹。

**预训练迁移（示意性）** — 在 GPT-2、ViT-B/16 等骨干上，可用 **解析匹配的 GELU 暖启动** 再做短微调或 FFN 置换，检验收敛与精度，而不在本仓库展开完整训练流水。

本仓库 **不包含** 论文全文实验矩阵；代码中的变体名（H1–H4、ctx-split、谱分支等）与 `hessian_core.py` 注释一一对应，供阅读实现与 smoke 跑通。

---

## 已报道结果（论文摘要级，非本仓复现包）

下列数字来自内部稿件 **摘要中的匹配预算实验**，在此 **仅作方法动机说明**；读者 **不应** 将本 Git 目录等同于可一键复现的完整论文包。扩大规模与再审稿后数字可能调整。

| 任务 | 对照 | H-FFN / Memory-Track 等（摘要） |
|------|------|--------------------------------|
| Needle-in-a-Haystack | 匹配 MLP \(\approx 75.7\%\pm 5.8\) | \(\approx 96.5\%\pm 0.7\) |
| WikiText-103 困惑度 | 标准 FFN \(455.3\pm 7.6\) | Decoupled H-FFN \(388.0\pm 2.3\)（约 \(1.17\times\) 增益） |
| 因果 LM（多语料） | — | 摘要称约 \(1.06\)–\(1.38\times\) PPL 改善量级 |
| GPT-2 微调 / ViT-B/16 | 基线 FFN | 摘要称约 \(1.07\times\) PPL 与更少步数达相近精度等 |

本目录提供的是 **任务入口与原型图**（如 `figures/memory_track_geom/` 中的轨迹示意），**不是** 上述行的官方复现脚本全集。

---

## 本仓库包含什么

| 类别 | 路径 |
|------|------|
| H-FFN 与低秩变体 | `tlb/ffn/hessian_core.py`，`standard.py` |
| 因果 LM / ctx-split 壳 | `tlb/models/causal_lm.py`，`gpt2_ctxsplit_init.py` |
| 视觉 FFN 置换 | `tlb/models/vit_ffn_transfer.py` |
| 记忆几何探针 | `tlb/training/memory_track_probe.py` |
| 记忆任务 smoke | `experiments/run_memory_capacity_sweep.py` |
| WT103 smoke | `experiments/l4_causal_wt103_single.py`，`configs/wt103_scale_smoke/` |
| 视觉 smoke | `experiments/l_vit_cifar_smoke.py` |
| 本地同步 | `scripts/sync_from_tlb_algorithm.sh`（源：`~/Desktop/TLB-Algorithm`） |

---

## 快速开始（smoke）

```bash
cd direction-2-hessian-ffn
pip install -r requirements-snippet.txt
python experiments/l0_smoke.py
```

WikiText / FineWeb 等数据 **未** 打包进 Git；在本地 `TLB-Algorithm` 准备数据后，可参见 `configs/wt103_scale_smoke/manifest.yaml` 与 `run_wt103_d256_lr_smoke.py --help`。

---

## 与方向 1 的界限

方向 1 研究 **小数据 Marian 微调** 时 attention **value** 的谱与级联探针；方向 2 研究 **FFN 子层** 的 Hessian 低秩形式与 **记忆 / LM / 视觉** 能力。二者共享「几何可读」动机，但 **干预位置、任务与公开粒度完全不同**。
