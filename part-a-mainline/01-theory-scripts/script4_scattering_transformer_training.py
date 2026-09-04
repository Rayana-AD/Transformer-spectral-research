# Script4：低秩矩阵去噪恢复，8 层堆叠 Transformer（散射随深度累积）
# 噪声：高斯 + Laplace + 均匀 + 稀疏脉冲（非高斯混合）
# 曲线：默认横轴画满 TRAIN_STEPS；另含「全区间线性」与「末段放大」子图 + 终值条形图。
# 若各激活曲线仍几乎重合：优先尝试提高噪声/增大 MATRIX_DIM/减小 RANK；再考虑加深（NUM_LAYERS）或略降 LR 拉长训练。
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from activation_extras import build_activation
from paper_style import apply_paper_style

BASE_SEED = 42
MATRIX_DIM = 16
RANK = 3
NUM_LAYERS = 8
# 混合噪声强度（整体强于原先单高斯）
NOISE_GAUSS_STD = 0.5
NOISE_LAPLACE_B = 0.35
NOISE_UNIFORM_HALF = 0.35
OUTLIER_PROB = 0.08
OUTLIER_SCALE = 1.2
NUM_TRAIN = 320
NUM_VAL = 80
BATCH = 32
LR = 1e-3
TRAIN_STEPS = 2500
EVAL_EVERY = 25
# 与训练步数一致，保证图上能看到 true final；若仅需早期对比可改小
PLOT_X_MAX = TRAIN_STEPS
DEVICE = "cpu"
OUTPUT_DIR = Path("results_script4")
PAPER_DIR = Path("paper_assets")

VAL_LOSS_LINEAR_WINDOW_START = 150
# 末段子图：只画最后这么多步（线性纵轴紧贴数据）
LINEAR_TAIL_LAST_STEPS = 700

ACTIVATION_LABEL_TEX: dict[str, str] = {
    "gelu": r"$\mathrm{GELU}$",
    "relu": r"$\mathrm{ReLU}$",
    "elu": r"$\mathrm{ELU}$",
    "mish": r"$\mathrm{Mish}$",
    "smish": r"$\mathrm{Smish}$",
    "light_depth_t2.0": r"Light-depth, $t{=}2$",
}


ACTIVATION_EXPERIMENTS: list[dict] = [
    {"id": "gelu", "name": "gelu"},
    {"id": "relu", "name": "relu"},
    {"id": "elu", "name": "elu"},
    {"id": "mish", "name": "mish"},
    {"id": "smish", "name": "smish"},
    {"id": "light_depth_t2.0", "name": "light_depth", "t": 2.0},
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_low_rank_matrix(n: int, r: int) -> torch.Tensor:
    u = torch.randn(n, r)
    v = torch.randn(n, r)
    return u @ v.t()


def laplace_noise(shape: tuple[int, ...], b: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    u = torch.rand(shape, device=device, dtype=dtype) - 0.5
    u = u.clamp(min=-0.499999, max=0.499999)
    return -torch.sign(u) * b * torch.log(1 - 2 * u.abs())


def add_mixed_noise(m: torch.Tensor) -> torch.Tensor:
    """高斯 + Laplace + 均匀 + 稀疏符号脉冲。"""
    n0, n1 = m.shape
    device, dtype = m.device, m.dtype
    y = m.clone()
    y = y + NOISE_GAUSS_STD * torch.randn(n0, n1, device=device, dtype=dtype)
    y = y + laplace_noise((n0, n1), NOISE_LAPLACE_B, device, dtype)
    y = y + NOISE_UNIFORM_HALF * (2 * torch.rand(n0, n1, device=device, dtype=dtype) - 1)
    mask = torch.rand(n0, n1, device=device, dtype=dtype) < OUTLIER_PROB
    spikes = OUTLIER_SCALE * torch.sign(torch.randn(n0, n1, device=device, dtype=dtype))
    y = y + mask.to(dtype) * spikes
    return y


def build_dataset(num: int, n: int, r: int, seed0: int) -> tuple[torch.Tensor, torch.Tensor]:
    clean = []
    noisy = []
    for i in range(num):
        set_seed(seed0 + i)
        m = make_low_rank_matrix(n, r)
        y = add_mixed_noise(m)
        clean.append(m)
        noisy.append(y)
    return torch.stack(clean, dim=0), torch.stack(noisy, dim=0)


class TransformerBlock(nn.Module):
    """单层：Pre-norm 自注意力 + FFN（与原先单层结构同族，堆叠 NUM_LAYERS 次）。"""

    def __init__(self, dim: int, act: nn.Module) -> None:
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            act,
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (self.dim**0.5), dim=-1)
        x = x + self.out(attn @ v)
        x = x + self.ffn(self.norm2(x))
        return x


class LowRankDenoiseTransformer(nn.Module):
    """输入 noisy Y 的行序列 (B, n, n)，经 NUM_LAYERS 层后输出 M 估计。"""

    def __init__(self, dim: int, num_layers: int, activation_name: str, **act_kwargs) -> None:
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, build_activation(activation_name, **act_kwargs))
                for _ in range(num_layers)
            ]
        )
        self.row_head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.row_head(x)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_one_config(exp: dict) -> list[dict]:
    act_id = exp["id"]
    act_name = exp["name"]
    act_kwargs = {k: exp[k] for k in exp if k not in ("id", "name")}

    set_seed(BASE_SEED)
    model = LowRankDenoiseTransformer(MATRIX_DIM, NUM_LAYERS, act_name, **act_kwargs).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_m, train_y = build_dataset(NUM_TRAIN, MATRIX_DIM, RANK, seed0=BASE_SEED + 10_000)
    val_m, val_y = build_dataset(NUM_VAL, MATRIX_DIM, RANK, seed0=BASE_SEED + 50_000)
    train_m = train_m.to(DEVICE)
    train_y = train_y.to(DEVICE)
    val_m = val_m.to(DEVICE)
    val_y = val_y.to(DEVICE)

    rows: list[dict] = []
    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, NUM_TRAIN, (BATCH,), device=DEVICE)
        xb = train_y[idx]
        mb = train_m[idx]

        opt.zero_grad()
        pred = model(xb)
        loss = nn.functional.mse_loss(pred, mb)
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0 or step == TRAIN_STEPS - 1:
            model.eval()
            with torch.no_grad():
                train_full = nn.functional.mse_loss(model(train_y), train_m).item()
                vloss = nn.functional.mse_loss(model(val_y), val_m).item()
            model.train()
            rows.append(
                {
                    "activation_id": act_id,
                    "step": step,
                    "train_loss_batch": float(loss.item()),
                    "train_loss_full": float(train_full),
                    "val_loss": float(vloss),
                }
            )
    return rows


def plot_curves(rows: list[dict]) -> None:
    apply_paper_style()
    aids = sorted({r["activation_id"] for r in rows})
    fig, (ax_log, ax_lin, ax_tail) = plt.subplots(
        3,
        1,
        figsize=(9, 10),
        sharex=False,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.1, 1.0, 0.95]},
    )
    xmax = PLOT_X_MAX
    x_tail0 = max(0, xmax - LINEAR_TAIL_LAST_STEPS)

    for aid in aids:
        sub = [r for r in rows if r["activation_id"] == aid and r["step"] <= xmax]
        x = [r["step"] for r in sub]
        y = [r["val_loss"] for r in sub]
        lbl = ACTIVATION_LABEL_TEX.get(aid, aid.replace("_", r"\_"))
        ax_log.plot(x, y, label=lbl, linestyle="-")
        ax_lin.plot(x, y, label=lbl, linestyle="-")
        sub_tail = [r for r in sub if r["step"] >= x_tail0]
        ax_tail.plot([r["step"] for r in sub_tail], [r["val_loss"] for r in sub_tail], label=lbl, linestyle="-")

    ax_log.set_ylabel(r"Validation $\mathcal{L}_{\mathrm{MSE}}$ (log)")
    ax_log.set_yscale("log")
    ax_log.set_title(rf"Low-rank denoising ({NUM_LAYERS}-layer Transformer), $0\leq s \leq {xmax}$")
    ax_log.legend(fontsize=7, ncol=2, loc="upper right")
    ax_log.grid(True, which="both", ls=":", alpha=0.45)
    ax_log.set_xlim(0, xmax)

    ax_lin.set_ylabel(r"Validation $\mathcal{L}_{\mathrm{MSE}}$ (linear)")
    mid_start = max(VAL_LOSS_LINEAR_WINDOW_START, xmax // 3)
    win = [r["val_loss"] for r in rows if mid_start <= r["step"] <= xmax]
    if win:
        lo, hi = float(np.min(win)), float(np.max(win))
        pad = max((hi - lo) * 0.15, 1e-4)
        ax_lin.set_ylim(max(lo - pad, 1e-6), hi + pad)
    ax_lin.grid(True, ls=":", alpha=0.45)
    ax_lin.set_xlim(0, xmax)
    ax_lin.set_title(rf"Linear scale (emphasize $s \geq {mid_start}$)")

    ax_tail.set_xlabel(r"Training step $s$")
    ax_tail.set_ylabel(r"$\mathcal{L}_{\mathrm{val}}$ (tail zoom)")
    ax_tail.set_title(rf"Last {LINEAR_TAIL_LAST_STEPS} steps (linear, tight axis)")
    ax_tail.set_xlim(x_tail0, xmax)
    win_t = [r["val_loss"] for r in rows if x_tail0 <= r["step"] <= xmax]
    if win_t:
        lo_t, hi_t = float(np.min(win_t)), float(np.max(win_t))
        span = max(hi_t - lo_t, 1e-5)
        ax_tail.set_ylim(lo_t - 0.08 * span, hi_t + 0.08 * span)
    ax_tail.grid(True, ls=":", alpha=0.45)
    ax_tail.legend(fontsize=6, ncol=2, loc="upper right")
    for base in (OUTPUT_DIR, PAPER_DIR):
        base.mkdir(parents=True, exist_ok=True)
        fig.savefig(base / "figure_script4_val_loss.png", dpi=300, bbox_inches="tight")
        fig.savefig(base / "figure_script4_val_loss.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "loss_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "loss_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_final_val_bars(summary: list[dict]) -> None:
    """按终值验证损失排序的水平条形图，便于读出微小差异。"""
    apply_paper_style()
    s = sorted(summary, key=lambda r: r["final_val_loss"], reverse=True)
    ys = np.arange(len(s))
    vals = [r["final_val_loss"] for r in s]
    labels = [ACTIVATION_LABEL_TEX.get(r["activation_id"], r["activation_id"]) for r in s]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    ax.barh(ys, vals, color="steelblue", height=0.65)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Final validation $\mathcal{L}_{\mathrm{MSE}}$")
    s_end = max(r["final_step"] for r in summary)
    ax.set_title(rf"Ranking at end of training ($s={s_end}$)")
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-5)
    ax.set_xlim(lo - 0.15 * span, hi + 0.25 * span)
    for i, r in enumerate(s):
        v = r["final_val_loss"]
        ax.text(v + 0.003 * span, i, f"{v:.4f}", va="center", fontsize=8)
    for base in (OUTPUT_DIR, PAPER_DIR):
        base.mkdir(parents=True, exist_ok=True)
        fig.savefig(base / "figure_script4_final_val_bar.png", dpi=300, bbox_inches="tight")
        fig.savefig(base / "figure_script4_final_val_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def export_script4_booktabs(summary: list[dict]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out = PAPER_DIR / "table_script4_val_loss_booktabs.tex"
    lines = [
        r"% Auto-generated from results_script4/training_loss_summary.csv",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Low-rank matrix denoising: validation mean-squared error after mixed non-Gaussian noise.}",
        r"\label{tab:script4-val-loss}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Activation & $\mathcal{L}_{\mathrm{val}}$ (final) & $\min \mathcal{L}_{\mathrm{val}}$ & Step of min \\",
        r"\midrule",
    ]
    for row in sorted(summary, key=lambda r: r["activation_id"]):
        aid = row["activation_id"]
        tex = ACTIVATION_LABEL_TEX.get(aid, aid.replace("_", r"\_"))
        lines.append(
            f"{tex} & {row['final_val_loss']:.4f} & {row['min_val_loss']:.4f} & {row['min_val_step']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--regen-summary-from-raw":
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        regen_from_raw()
        return

    print("=== Script4: low-rank denoising, stacked Transformer ===")
    print(
        f"torch={torch.__version__}, numpy={np.__version__}, n={MATRIX_DIM}, rank={RANK}, layers={NUM_LAYERS}, "
        f"noise: Gaussσ={NOISE_GAUSS_STD} LaplaceB={NOISE_LAPLACE_B} Unif±={NOISE_UNIFORM_HALF} "
        f"outlier p={OUTLIER_PROB} scale={OUTLIER_SCALE}, "
        f"train={NUM_TRAIN}, val={NUM_VAL}, batch={BATCH}, steps={TRAIN_STEPS}, plot_x<={PLOT_X_MAX}, lr={LR}"
    )
    all_rows: list[dict] = []
    for exp in ACTIVATION_EXPERIMENTS:
        print(f"\n-- {exp['id']} ({exp['name']}) --")
        rows = train_one_config(exp)
        all_rows.extend(rows)
        print(
            f"step0: train_full={rows[0]['train_loss_full']:.4f} val={rows[0]['val_loss']:.4f}"
        )
        print(
            f"final: train_full={rows[-1]['train_loss_full']:.4f} val={rows[-1]['val_loss']:.4f}"
        )

    save_csv(OUTPUT_DIR / "training_loss_raw.csv", all_rows)
    summary = build_training_summary(all_rows)
    save_csv(OUTPUT_DIR / "training_loss_summary.csv", summary)
    plot_curves(all_rows)
    plot_final_val_bars(summary)
    export_script4_booktabs(summary)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
