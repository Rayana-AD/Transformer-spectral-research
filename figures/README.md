# Figure Gallery（精选）

主页 [README](../README.md) 引用本目录中的 **6 张代表图**（方向 1、方向 2 各 3 张）。更高分辨率与 Scheme A 全系列见 `part-a-mainline/05-results/`。

| 文件 | 方向 | 读图要点 |
|------|------|----------|
| `01_activation_triptych.png` | 1 | 点式激活前后特征矩阵的视觉对比：非线性如何「涂抹」或重组能量，为后文核范数/条件数探针提供直觉。 |
| `02_script3_depth_conditioning.png` | 1 | 合成实验中条件数 κ 随深度与激活类型的累积：说明深度会放大谱病态，不仅是宽度问题。 |
| `07_schemeA_depth_nuclear_norm.png` | 1 | **主图**：Marian 微调训练期 value 张量核范数随层变化；Encoder 中段抬升、Decoder 浅高–中谷，体现两侧几何不对称。 |
| `d2_01_niah_stress_capacity.png` | 2 | **主图**：Needle-in-a-Haystack / 记忆容量 stress 下，Memory-Track（H-FFN）与匹配 MLP 的准确率对比。 |
| `d2_02_wt103_ppl.png` | 2 | WikiText-103 上 Decoupled H-FFN 与标准 FFN 的困惑度曲线（同预算）：因果语言建模增益的代表。 |
| `d2_03_memory_trace_chain.png` | 2 | 记忆轨道几何探针：层间迹链与秩轨迹如何随深度演化（解释机制，非单独主结果表）。 |

未列入 gallery 的归档图（如 Scheme A 热图、PPL 特写）仍在 `part-a-mainline/05-results/results_probe_independent/paper_figures/`。
