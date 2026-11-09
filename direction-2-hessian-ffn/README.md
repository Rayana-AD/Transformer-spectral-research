# 方向 2：Hessian 势函数 FFN（H-FFN）研究原型

标准 Transformer 的 FFN 仍是「扩维 → 点式激活 → 压回」的两层 MLP，参数量与算力中占比很高，却缺少与 **局部几何 / 二阶结构** 显式对齐的写法。本目录公开 **Hessian Feed-Forward Network（H-FFN）** 的可运行原型：把 FFN 写成标量势 \(\Psi\) 的 Hessian 与输入的缩并，在 **记忆检索、因果语言建模、视觉 FFN 替换** 三类设置下检验能力。

完整论文级实验表与大规模预训练 **不** 包含在本仓库（见 [PROTOTYPE-NOTICE.md](PROTOTYPE-NOTICE.md)）。下文的 **背景与方法** 与投稿稿一致；**结果** 仅概括已发表主张的类型，并以本仓库内的 **smoke 脚本与示意性几何图** 为例说明如何自行复现探针。

---

## 背景：标准 FFN 的三处结构限制

1. **静态稠密映射**：同一 MLP 作用于每个 token，中间宽度固定（常为 \(4d\)），难以按上下文调节有效秩。  
2. **联想存储无显式索引**：FFN 常被解释为键–值式记忆，但内容与路由纠缠在同一路径上，易产生干扰。  
3. **与曲率/二阶信息脱节**：注意力侧已有核、Hopfield 等几何视角；FFN 侧较少把 **输入依赖的曲率** 写进前向式子。

方向 2 的问题因此是：**能否用与 Hessian 对齐的、可低秩实现的 FFN，在匹配预算下改善记忆与因果建模，并转移到视觉骨干？**

---

## 方法

### 核心：势函数与 rank-one Hessian 缩并

将标量势写为沿可学习方向的求和：

\[
\Psi(x)=\sum_{j=1}^{m}\psi(s_j),\quad s_j = u_j^\top x + b_j .
\]

对 \(x\) 求导得到梯度与 Hessian；H-FFN 的前向定义为 **Hessian 与 \(x\) 的缩并**：

\[
\text{H-FFN}(x)=\mathcal{H}(x)\,x=\sum_{j=1}^{m}\lambda_j(x)\,(u_j^\top x)\,u_j,
\quad \lambda_j(x)=\psi''(s_j).
\]

几何上，这是对 \(m\) 个 **rank-one 方向** 的加权：先投影到轴 \(u_j\)，再用 **局部二阶曲率** \(\lambda_j\) 门控，沿同一方向发射。在实现上取 \(\psi=\tanh^2\) 等 **有界、可正可负** 的 \(\psi''\)，以避免 outlier 坐标上的不稳定放大（论文中有 \(\psi''\) 的闭式与多尺度 ladder 扩展）。

**代码入口**：`tlb/ffn/hessian_core.py`（含 H1–H4、ctx-split、低秩分支等变体）；对照基线 `tlb/ffn/standard.py`。

### 记忆：Memory-Track（同轴多尺度门控）

针对 **联想检索 / Needle-in-a-Haystack** 类任务，在核心 H-FFN 上增加 **同轴、多尺度的上下文门控**（coaxial Memory-Track），使 FFN 在保持与 MLP 相近渐近成本的前提下，显式调节 **沿轨道的秩与迹**。  

**本仓库**：`tlb/training/memory_track_probe.py`，`experiments/run_memory_capacity_sweep.py`；几何示意见 `figures/memory_track_geom/`（迹链、曲率质量等，**非** 完整 NIAH 主表）。

### 语言建模：Decoupled H-FFN 与 ctx-split

针对 **因果 LM**，将 **内容、路由、发射** 解耦，并支持 **ctx-split**（上下文分裂读写），在 WikiText-103 / 流式语料等设定下与 standard FFN 对照 **PPL 与主动秩轨迹**。  

**本仓库**：`tlb/models/causal_lm.py`、`tlb/models/gpt2_ctxsplit_init.py`；smoke 配置 `configs/wt103_scale_smoke/`，入口 `experiments/l4_causal_wt103_single.py`、`experiments/run_wt103_d256_lr_smoke.py`。

### 视觉：FFN 模块替换

在 **ViT** 等骨干上 **仅替换 FFN 子层** 为 Hessian 变体（权重转移与 analytic GELU 匹配等见论文 transfer 节），检验 **收敛速度与准确率–秩指标**。  

**本仓库**：`tlb/models/vit_ffn_transfer.py`，`experiments/l_vit_cifar_smoke.py`、`experiments/vit_forward_smoke.py`。

---

## 结果（主张类型与公开范围）

论文稿中的主要结论类型如下（**具体数字与置信区间以正式发表版为准**；本仓库 **不** 打包完整 seed 表与大 checkpoint）：

| 任务类型 | 对照 | 论文中报告的趋势（摘要级） |
|----------|------|---------------------------|
| 联想记忆（NIAH 等） | 匹配 MLP | Memory-Track 在匹配预算下 **明显提高** 检索准确率 |
| 因果 LM | standard FFN | Decoupled / ctx-split H-FFN 在 WT103、FineWeb-Edu 等上 **PPL 约 1.06–1.38× 改善** |
| 预训练迁移 | GPT-2 / ViT FFN 替换 | 微调 PPL 与 ViT 收敛 **有增益**（如更少 step 达到相近精度） |

**本目录可核验的内容**限于：

- **smoke 训练/前向** 能否跑通、loss 是否有限；  
- **记忆轨道几何图**（`figures/memory_track_geom/`）展示 **秩–迹–曲率** 探针形态，用于理解机制，**不能** 替代论文主表；  
- 配置文件中的 **小规模** recipe（`configs/wt103_scale_smoke/recipes/`）。

会议审稿 **得分不佳、可能撤稿**；公开原型旨在 **方法可审计**，扩大实验后可能再投。引用或产品时请参阅 [PROTOTYPE-NOTICE.md](PROTOTYPE-NOTICE.md)。

---

## 目录结构

```
tlb/ffn/              Hessian 核心与变体
tlb/models/           因果 LM、ViT 替换、ctx-split 组装
tlb/training/         记忆与几何探针
tlb/kernels/          可选融合内核（性能向）
experiments/          三类任务的 smoke / 探针脚本
configs/              小规模对照配置
figures/memory_track_geom/   记忆几何示意（非论文全套主图）
```

---

## 快速开始（原型）

```bash
cd direction-2-hessian-ffn
pip install -r requirements-snippet.txt

# 数值与模块 smoke（CPU 可）
python experiments/l0_smoke.py

# WT103 小规模对照（需自行准备语料，见本地 TLB-Algorithm 文档）
python experiments/run_wt103_d256_lr_smoke.py --help
```

从作者完整算法树更新本目录：

```bash
bash scripts/sync_from_tlb_algorithm.sh
```

默认源路径：`~/Desktop/TLB-Algorithm`。

---

## 与仓库方向 1 的区别

| | 方向 1（value cascade） | 方向 2（H-FFN） |
|--|-------------------------|------------------|
| 干预对象 | Attention **Value** 路径 | **FFN** 子层 |
| 主要场景 | Marian **极少数据微调** 的训练期诊断 | **记忆 / 因果 LM / 视觉** 能力与几何 |
| 本仓角色 | 完整归档 + Enc/Dec 深度主图 | **方法原型 + smoke + 示意几何** |
