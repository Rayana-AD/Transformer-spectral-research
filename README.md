# Transformer 谱几何与 Value-Path 诊断研究档案

本仓库公开两条研究线：**方向 1** 关注预训练 Marian 在 **数据极少**、**AdamW 等优化尚未进入稳定 regime** 时，内部表示如何漂移；**方向 2** 为基于 Hessian 形式的 **低秩 FFN 原型**（见 [direction-2-hessian-ffn](direction-2-hessian-ffn/)，与会议投稿线分离，以原型展示为主）。

---

## 方向 1：Value 级联平滑在问什么？

级联平滑 **不是** 为了在充足数据与常规微调下调高 BLEU 的通用插件。它的主要用途是：

- 在 **平行句极少**（例如约 800 对）且 **优化步数很短** 时，给 Encoder / Decoder 的 **self-attention value 路径** 加上 **可控、可重复的读入算子**；
- 在 **训练过程中** 同步记录 **核范数、PPL、梯度峰值** 等指标，观察 **Encoder 与 Decoder 是否呈现不同的深度形态**，以及这些形态与 **损失尖峰、输出稳定性** 的关联有多强；
- 当 **AdamW 超参或标签编码不当** 导致 vanilla 微调明显掉点时，级联有时表现为 **相对 rescue**——这反映的是 **\(V\) 空间如何参与学习**（可写/不可写、前向低通），而不是「平滑本身等于更好的归纳偏置」。

在 **正确 `text_target` + 默认 AdamW + 全数据 1 epoch** 的对照下，vanilla 往往可达 ~20.8–21 BLEU；Scheme A **encoder-only、detach、legacy** 的级联与之接近。**请勿把旧 stress-test 下的 ~15 / ~18 数字与当前 matched 协议混谈**（见 [REPRODUCTION.md](REPRODUCTION.md)）。

---

## 一张图读懂：Encoder / Decoder 随层「膨胀–压缩」差异

训练期（Scheme A 探针，\( \alpha_s{=}0.1,\ \alpha_t{=}0.2 \)，多种子平均）中，**同一级联算子** 在两侧堆叠上呈现 **不同的核范数深度曲线**：

![Scheme A：Encoder 与 Decoder 训练期 value 核范数随层变化](figures/gallery/07_schemeA_depth_nuclear_norm.png)

| 模块 | 深度形态（定性） | 便于怎样读训练 |
|------|------------------|----------------|
| **Encoder** | 中段（约 layer 2）**抬升、保持较高容量**；双向表征在中层展开 | 与 **中间层 PPL/核范数** 联动时，常看到较平滑的压缩阶梯 |
| **Decoder** | 浅层 **范数最大**，向中层 **急剧压缩**（谷），深层部分回升 | 自回归 **历史→过滤→输出**；中层与 **train PPL 尖峰**、梯度峰值关联更强 |

级联三阶段在同一图里通常满足 **\( \|V^{\mathrm{raw}}\|_* > \|V^{(1)}\|_* > \|V^{(2)}\|_* \)**，说明算子主要起 **可度量的压缩探针** 作用；**Encoder vs Decoder 的形态差** 往往大于 spatial-first / temporal-first 的顺序差。更细的层–指标说明见 `part-a-mainline/05-results/` 下 Scheme A 归档与 `schemeA_train_depth_nuc_seedavg_notes.md`。

---

## 仓库结构

```
part-a-mainline/          方向 1 主线：合成谱理论、ViT 桥、Marian value 级联与归档结果
part-b-explorations/      方向 1：未走通的 Q/K/V、memory、FFN 编辑等探索
direction-2-hessian-ffn/  方向 2：Hessian 形式低秩 FFN 原型（记忆 / 因果 LM / 视觉）
figures/gallery/          方向 1 主图（Enc/Dec 深度曲线，07）
REPRODUCTION.md           复现步骤与协议说明
```

方向 1 细节仍见 [part-a-mainline/README.md](part-a-mainline/README.md)；负面实验见 [part-b-explorations/README.md](part-b-explorations/README.md)。

---

## 方法要点（方向 1，极简）

1. **谱诊断语言**（script1–4）：点式激活如何改变秩、核范数、条件数。  
2. **Value 级联（Scheme A）**：在 **不改变 \(QK^\top\) 路由** 的前提下，对 **\(V\)** 做 spatial × temporal 两阶段混合；用于 **训练期探针**，而非默认推理必开。  
3. **Encoder-only + detach + legacy（circular + global）** 与论文 Scheme A 探针矩阵一致；Decoder 因果安全版用于回应泄漏疑虑，**不宜与 Marian 全模型 BLEU 主结论混用**。

---

## 快速开始

```bash
pip install -r requirements.txt

# 合成谱实验（CPU，数分钟）
cd part-a-mainline/01-theory-scripts && python script2_universal_validation.py

# 阅读已归档 Scheme A 汇总（无需 GPU）
cat part-a-mainline/05-results/results_probe_independent/schemeA_3x2x3_20260424_0137/schemeA_full_report.md
```

完整流程：[REPRODUCTION.md](REPRODUCTION.md)

---

## 方向 2

[Hessian 低秩 FFN（H-FFN）](direction-2-hessian-ffn/) 为独立研究线：背景、方法摘要与论文级结果口径见该目录 README；代码为原型，知识产权说明见 **PROTOTYPE-NOTICE.md**。

---

## 引用与同步

供学术复现与过程审阅；引用请注明仓库 URL。投稿用全文与审稿材料保留在私有工作区。

本地完整实验树：`~/Desktop/TLB-Experiments/`（方向 1）、`~/Desktop/TLB-Algorithm/`（方向 2 源）。
