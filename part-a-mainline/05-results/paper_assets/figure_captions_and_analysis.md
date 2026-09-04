# 图表说明与读图分析（论文草稿）

文中符号：$\mathbf{Z}$ 为线性层输出；$\mathbf{H}$ 为经激活后的特征矩阵；$\cdot_{\ast}$ 为核范数；$\kappa$ 为矩阵条件数（谱范数比）；$\ell$ 为层索引。

---

## Script1：激活前后矩阵热力图（三联图）

**文件**：`results_script1/figure_activation_triptych.pdf`（及 `paper_assets/` 下同名的副本）。

**坐标与标签**

- 三幅子图共用色条，纵横向无坐标刻度（矩阵下标省略）；色条标签为「Entry value」，表示矩阵元素数值。
- 子图标题（Matplotlib mathtext）：
  1. Pre-activation $\mathbf{Z}$（linear output）：随机线性层输出，未过激活。
  2. $\mathrm{ReLU}(\mathbf{Z})$：负半轴截断。
  3. $\mathrm{GELU}(\mathbf{Z})$：光滑门控，保留更多负值信息。

**Caption（英文期刊体例）**

> **Figure 1.** Heatmaps of a single $16\times 16$ feature matrix before and after pointwise activations. The left panel shows the linear output $\mathbf{Z}$; the center and right panels apply $\mathrm{ReLU}$ and $\mathrm{GELU}$, respectively. Color scale is fixed to $[-2,2]$ for cross-panel comparison.

**读图要点**

- ReLU 会在空间上产生大面积零区（秩与核范数往往更易下降）；GELU 保留负值通道，热力图通常更「满」，与后文矩阵不变量统计一致。

---

## Script 2：维度与激活普适性（三线表）

**文件**：`paper_assets/table_script2_universal_booktabs.tex`（由 `results_script2/universal_summary.csv` 生成）。

**版面**：每个典型 $n$ 下为 **数值秩**（超过阈值的奇异值个数）、**谱范数** $\mathbf{M}*2=\sigma*{\max}$、**核范数** $\mathbf{M}*{\ast}=\sum_i\sigma_i$（二者与秩均为20 个种子上的 **均值 $\pm$ 标准差**）、**条件数** $\kappa=\sigma*{\max}/\sigma_{\min}$ 在有限 $\kappa$ 实现上的 **中位数** $\mathrm{med}\kappa$。期刊表只印 `**TABLE_EXPORT_DIMS = [16, 64]`**；CSV 仍为全扫描 $n\in8,16,32,64$。表注正文不引用 Script1，符号在 caption 中单独定义。

**Caption（与 `table_script2_universal_booktabs.tex` 一致）**

> 说明 $\mathbf{M}=\sigma(W\mathbf{X})$ 的构造，并分别定义 rank、$\mathbf{M}*2$、$\mathbf{M}*{\ast}$、$\mathrm{med}\kappa$ 及统计方式；代表 $n\in16,64$，其余 $n$ 见 CSV。

**读表要点**

- 同一 $n$ 下，$\mathrm{Linear}$（恒等）往往核范数最大、$\kappa$ 也偏大；平滑激活（如 $\mathrm{GELU}$、$\mathrm{SiLU}$）常使核范数与 $\kappa$ 相对降低。
- 随 $n$ 增大，各类范数尺度上升，$\kappa$ 的量级通常增大，比较时应固定 $n$ 或在文中归一化说明。

---

## Script 3：深度散射——条件数 $3\times 3$ 组图

**文件**：`paper_assets/figure_script3_depth8_condition_grid.pdf`（及 `results_script3/depth8_condition_grid_paper.png`）。

更新数据后若只需重画组图，可运行：`python script3_depth_scattering.py --export-grid-only`（依赖 `results_script3/depth_summary.csv` 中已包含九种 `activation_id`；若缺行请先完整跑一遍 Script3）。

**布局**

- 共 $3\times 3$ 个子图，对应九种激活设定；深度固定 $N=8$。
- **横轴**：层索引 $\ell=1,\ldots,8$。
- **左纵轴（对数）**：$\kappa$ 的**中位数**（仅对条件数有限的随机种子）。
- **右纵轴（灰虚线）**：有限条件数种子所占**比例**（$0$–$1$）。

**行排列**

1. $\mathrm{GELU}$，$\mathrm{ReLU}$，$\mathrm{Smish}$
2. Light-depth，$t=1,2,4$

3.物理衰减 $z\mathrm{e}^{-\tau}$，$\tau=\mathrm{softplus}(\beta z)$（$\beta=0.5,1$）与 $\tau=\mathrm{softplus}(z^2)$

**Caption**

> **Figure 3.** Depth evolution of the condition number at $N{=}8$ scattering layers. Orange curves: median $\kappa$ over finite samples (log scale). Gray dashed curves: fraction of seeds with finite $\kappa$. When the fraction drops below one, some initializations yield singular or numerically singular features at that layer.

**读图要点**

- **橙色曲线上升**：随 $\ell$ 增大，条件数中位数上升，数值病态加剧。
- **灰线下降**：深层部分种子 $\kappa\to\infty$，说明秩塌缩或奇异矩阵出现；与左列核范数快速衰减的实验现象一致。
- $t=1$ 的 light-depth 与 Smish 在映射上接近 $z\sigma(z)$，曲线形态应相近；差异主要来自随机线性层抽样的数值波动。
- $\tau=\mathrm{softplus}(z^2)$ 往往对大幅度特征施加强衰减，**有限 $\kappa$ 比例**在更浅层即开始下降，可与 $\beta z$ 形式对照讨论「对称光深」与稳定性。

---

## Script 4：低秩去噪与验证损失

**文件**：`paper_assets/figure_script4_val_loss.pdf`；**表**：`paper_assets/table_script4_val_loss_booktabs.tex`。

**坐标与标签**

- **横轴**：训练步 $s$（图中聚焦 $0\leq s\leq 1000$）。
- **上图**：验证 $\mathcal{L}_{\mathrm{MSE}}$，**对数**纵轴，展示跨数量级的整体下降。
- **下图**：同一验证损失，**线性**纵轴；纵轴范围由 $s\in[150,1000]$ 内的损失自动留白，用于分辨曲线间**细微差异**。

**Caption**

> **Figure 2.** Validation MSE on a low-rank matrix denoising task with mixed non-Gaussian noise. Top: log scale. Bottom: linear scale with vertical range zoomed to the spread over late training steps to separate curves.

**表（三线表）**

- $\mathcal{L}_{\mathrm{val}}$ (final)：所报告窗口内最后一个记录步的验证损失。
- $\min \mathcal{L}_{\mathrm{val}}$ 与 Step of min：窗口内最优值及对应步数。

**读图要点**

- 若上图（对数）中曲线近乎重合，应结合下图线性放大判断排序；若噪声与模型容量固定，不同激活的差异可能较小，需通过更难任务或更长训练再拉开。
- 验证集与训练集分离，可避免「记住单一噪声实现」导致的虚假过拟合结论。

---

## LaTeX 排版提示

- 请使用 `\usepackage{booktabs}` 编译上述表格。
- 若全文启用 `babel` / `polyglossia`，注意数学模式中的连字符与标点。
- 矢量图优先使用本目录下的 `.pdf` 插图。