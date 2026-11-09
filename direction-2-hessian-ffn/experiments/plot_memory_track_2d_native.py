#!/usr/bin/env python3
"""Figures that show the native chain x → s → λ=ψ''(s) → λ⊙proj → Δx.

The coaxial encoder (s, proj) cloud is nearly the line s≈proj, so a 9-column
density strip does not show memory. These plots keep the two views that do:

* time chain of native magnitudes (needle vs hay) next to retrieval accuracy
* (s, λ) mass sitting on the analytic ψ'' curve, with hay spreading farther
* decoder (s, proj) — the plane where gate s comes from context z, not from x
* centroid path in (|s|, |Δx|), the 2D picture of that chain
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "memory_track_geom"

NEEDLE = "#C44E52"
HAY = "#4C72B0"
PSI = "#C4884A"
ACC = "#1A1A1A"
LOBE = float(np.arctanh(1.0 / np.sqrt(3.0)))


def _psi(s: np.ndarray) -> np.ndarray:
    t = np.tanh(s)
    return 2.0 * (1.0 - t * t) * (1.0 - 3.0 * t * t)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _snaps(path: Path) -> list[tuple[int, dict, float | None]]:
    row = json.loads(path.read_text())
    acc = {
        int(p["step"]): float(p["val_acc"])
        for p in row.get("history", [])
        if "val_acc" in p
    }
    out = []
    for p in row.get("history", []):
        if "memory_geometry" not in p:
            continue
        step = int(p["step"])
        out.append((step, p["memory_geometry"], acc.get(step)))
    return out


def _last(side: dict | None) -> dict | None:
    for layer in reversed((side or {}).get("layers") or []):
        if "visible" in layer or "plane2d" in layer:
            return layer
    return None


def _series(snaps: list[tuple[int, dict, float | None]], side: str, region: str, key: str):
    steps, vals, acc = [], [], []
    for step, geom, a in snaps:
        layer = _last(geom.get(side))
        block = (layer or {}).get("visible", {}).get(region)
        if not block or key not in block:
            reg = (layer or {}).get("regions", {}).get(region, {})
            if key not in reg:
                continue
            block = reg
        steps.append(step)
        vals.append(float(block[key]))
        acc.append(a if a is not None else np.nan)
    return np.asarray(steps), np.asarray(vals), np.asarray(acc)


def _std_ff(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        return np.array([]), np.array([]), np.array([])
    snaps = _snaps(path)
    steps, needle, hay = [], [], []
    for step, geom, _ in snaps:
        layers = (geom.get("encoder") or {}).get("layers") or []
        if not layers:
            continue
        regions = layers[-1].get("regions") or {}
        if "needle" not in regions or "hay" not in regions:
            continue
        steps.append(step)
        needle.append(float(regions["needle"].get("ff_out_rms", np.nan)))
        hay.append(float(regions["hay"].get("ff_out_rms", np.nan)))
    return np.asarray(steps), np.asarray(needle), np.asarray(hay)


def _marginal_s(region_plane: dict) -> tuple[np.ndarray, np.ndarray]:
    block = region_plane["s_lambda"]
    z = np.asarray(block["counts"], float)
    edges = np.asarray(block["s_edges"], float)
    mass = z.sum(axis=0)
    total = mass.sum()
    if total > 0:
        mass = mass / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, mass


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trace_chain(snaps, std_path: Path, out: Path) -> None:
    """Four native stages, needle vs hay, with retrieval accuracy on the last stage."""
    keys = [
        ("s_abs_mean", r"$|s|$", r"gate $s=Ux+b$"),
        ("lambda_abs_mean", r"$|\lambda|$", r"curvature $\lambda=\psi''(s)$"),
        ("gated_abs_mean", r"$|\lambda \odot \mathrm{proj}|$", r"gated content"),
        ("ff_out_rms", r"$|\Delta x|$", r"map output $\Delta x$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.15))
    for ax, (key, ylab, title) in zip(axes, keys):
        sn, vn, an = _series(snaps, "encoder", "needle", key)
        _, vh, _ = _series(snaps, "encoder", "hay", key)
        ax.plot(sn, vn, "o-", color=NEEDLE, ms=4.5, lw=1.6, label="needle")
        ax.plot(sn, vh, "s-", color=HAY, ms=4.5, lw=1.6, label="hay")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel(ylab)
        if key == "ff_out_rms":
            ax2 = ax.twinx()
            m = np.isfinite(an)
            ax2.plot(sn[m], an[m], color=ACC, ls="--", marker=".", ms=5, lw=1.1, label="val acc")
            ax2.set_ylim(-0.02, 1.05)
            ax2.set_ylabel("val accuracy")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        else:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        r"Encoder coaxial chain on $x$:  $s \rightarrow \lambda=\psi''(s) \rightarrow \lambda\odot\mathrm{proj} \rightarrow \Delta x$",
        fontsize=11,
        y=1.04,
    )
    fig.tight_layout()
    _save(fig, out, "fig_trace_chain")

    # Hay/needle amplification vs the Standard MLP residual, same task.
    st, sn_std, sh_std = _std_ff(std_path)
    _, vn, _ = _series(snaps, "encoder", "needle", "ff_out_rms")
    steps, vh, _ = _series(snaps, "encoder", "hay", "ff_out_rms")
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    if len(steps):
        ax.plot(steps, vh / np.maximum(vn, 1e-8), "o-", color="#2A9178", ms=5, label="Memory-Track  hay/needle")
    if len(st):
        ax.plot(st, sh_std / np.maximum(sn_std, 1e-8), "s--", color="#9B6BA3", ms=5, label="Standard MLP  hay/needle")
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$|\Delta x|_{\mathrm{hay}} \;/\; |\Delta x|_{\mathrm{needle}}$")
    ax.set_title(r"Where the write shows up: hay is amplified, needle is not")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out, "fig_trace_chain_ratio")


def _uav_series(path: Path) -> dict[str, np.ndarray]:
    """Recorded ctxsplit smoke: content |Ux| vs gate |Az|, and U·V cosine."""
    if not path.exists():
        return {}
    row = json.loads(path.read_text())[0]
    steps, content, gate, uv, corr = [], [], [], [], []
    for pt in row.get("history", []):
        g = pt.get("hessian_geom")
        if not g:
            continue
        layers = g.get("per_layer") or []
        if not layers:
            continue
        steps.append(int(pt["step"]))
        content.append(float(np.mean([L["s_content_abs_mean"] for L in layers])))
        gate.append(float(np.mean([L["s_z_abs_mean"] for L in layers])))
        uv.append(float(g["uv_cos_mean"]))
        corr.append(float(g.get("s_gate_s_content_corr", np.nan)))
    if not steps:
        return {}
    return {
        "step": np.asarray(steps),
        "content": np.asarray(content),
        "gate": np.asarray(gate),
        "uv": np.asarray(uv),
        "corr": np.asarray(corr),
    }


def _gelu_over_s(s: np.ndarray) -> np.ndarray:
    """Warm-start profile λ=GELU(s)/s. Not the native gate."""
    out = np.empty_like(s)
    near = np.abs(s) < 1e-3
    # d/ds GELU at 0 is Φ(0)=0.5
    out[near] = 0.5
    ss = s[~near]
    out[~near] = 0.5 * ss * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (ss + 0.044715 * ss ** 3))) / ss
    return out


def _ladder(s: np.ndarray, k_max: int = 3) -> np.ndarray:
    acc = np.zeros_like(s)
    for k in range(k_max + 1):
        acc = acc + (2.0 ** (-k)) * _psi(s / (2.0 ** k))
    return acc


def plot_curvature_mass(snaps, out: Path, uav_path: Path) -> None:
    """Mass and write path are the figure. λ shapes are a thin reference strip."""
    want = {0, 200, 399}
    picked = [(s, g) for s, g, a in snaps if s in want]
    if len(picked) < 2:
        picked = [(s, g) for s, g, _ in (snaps[0], snaps[len(snaps) // 2], snaps[-1])]
    styles = {0: (0.45, ":"), 200: (0.75, "--"), 399: (1.0, "-")}

    fig = plt.figure(figsize=(9.6, 4.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.42, 1.35], hspace=0.55, wspace=0.28)
    ax_psi = fig.add_subplot(gs[0, :])
    ax_mass = fig.add_subplot(gs[1, 0])
    ax_path = fig.add_subplot(gs[1, 1])

    s_line = np.linspace(-2.2, 2.2, 500)
    family = (
        (r"$\psi''$ (used)", _psi(s_line), PSI, 2.0, "-"),
        (r"$\tau{=}0.5$", _psi(s_line / 0.5), "#8C6A4A", 1.0, "-"),
        (r"$\tau{=}2$", _psi(s_line / 2.0), "#D4A574", 1.0, "-"),
        (r"ladder", _ladder(s_line), "#3D6B8C", 1.15, "-"),
        (r"GELU$/s$ (init only)", _gelu_over_s(s_line), "#9B6BA3", 1.15, "--"),
    )
    for label, y, color, lw, ls in family:
        ax_psi.plot(s_line, y, color=color, lw=lw, ls=ls, label=label)
    ax_psi.axhline(0, color="0.8", lw=0.6)
    ax_psi.set_xlim(-2.2, 2.2)
    ax_psi.set_ylim(-0.9, 4.05)
    ax_psi.set_ylabel(r"$\lambda(s)$", fontsize=8)
    ax_psi.tick_params(labelsize=7)
    ax_psi.legend(
        fontsize=6.5,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )

    for step, geom in picked:
        enc = _last(geom.get("encoder"))
        if not enc or "plane2d" not in enc:
            continue
        alpha, ls = styles.get(step, (1.0, "-"))
        for region, color in (("needle", NEEDLE), ("hay", HAY)):
            centers, mass = _marginal_s(enc["plane2d"][region])
            ax_mass.plot(
                centers, mass, color=color, ls=ls, lw=1.5, alpha=alpha,
                label=f"{region}  {step}",
            )
    ax_mass.axvline(-LOBE, color=PSI, ls=":", lw=0.8)
    ax_mass.axvline(LOBE, color=PSI, ls=":", lw=0.8)
    ax_mass.set_xlabel(r"gate coordinate $s$")
    ax_mass.set_ylabel("unit mass")
    ax_mass.set_xlim(-1.25, 1.25)
    ax_mass.set_title("Mass leaves 0; hay splits into two lobes")
    ax_mass.legend(fontsize=6.5, ncol=2, loc="upper right")

    steps, dx_n, acc = _series(snaps, "encoder", "needle", "ff_out_rms")
    _, dx_h, _ = _series(snaps, "encoder", "hay", "ff_out_rms")
    ax_path.plot(steps, dx_n, "o-", color=NEEDLE, ms=4.5, lw=1.5, label="needle")
    ax_path.plot(steps, dx_h, "s-", color=HAY, ms=4.5, lw=1.5, label="hay")
    ax_acc = ax_path.twinx()
    ok = np.isfinite(acc)
    ax_acc.plot(steps[ok], acc[ok], color="0.25", ls="--", marker=".", ms=5, lw=1.0, label="val acc")
    ax_acc.set_ylim(-0.02, 1.05)
    ax_acc.set_ylabel("val accuracy")
    ax_path.set_xlabel("step")
    ax_path.set_ylabel(r"mean $|\Delta x|$")
    ax_path.set_title(r"Hay $|\Delta x|$ rises with accuracy")
    h1, l1 = ax_path.get_legend_handles_labels()
    h2, l2 = ax_acc.get_legend_handles_labels()
    ax_path.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    _save(fig, out, "fig_curvature_mass")


def _heat(ax, z, edges, *, xlim):
    z = np.asarray(z, float)
    z = z / z.sum() if z.sum() else z
    pos = z[z > 0]
    vmin = max(float(pos.min()) if pos.size else 1e-6, 1e-6)
    im = ax.imshow(
        z,
        origin="lower",
        aspect="equal",
        extent=[edges[0], edges[-1], edges[0], edges[-1]],
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=max(float(z.max()), vmin * 10)),
    )
    lo, hi = xlim
    ax.plot([lo, hi], [lo, hi], color="white", ls="--", lw=0.8, alpha=0.85)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    return im


def plot_paired_heatmaps(snaps, uav_path: Path, out: Path) -> None:
    """Decoder plane and UAV ctxsplit plane, same grid, two rows."""
    by_step = {s: g for s, g, a in snaps}
    dec_steps = [s for s in (0, 200, 399) if s in by_step]
    uav_snaps = []
    if uav_path.exists():
        row = json.loads(uav_path.read_text())
        if isinstance(row, list):
            row = row[0]
        uav_snaps = [
            (int(pt["step"]), pt["uav_plane"])
            for pt in row.get("history", [])
            if pt.get("uav_plane")
        ]
    want_uav = (0, 60, 119)
    uav_pick = []
    for target in want_uav:
        if not uav_snaps:
            break
        uav_pick.append(min(uav_snaps, key=lambda item: abs(item[0] - target)))
    n = 3
    fig = plt.figure(figsize=(9.4, 6.6))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1, 1, 1, 0.06], wspace=0.18, hspace=0.38)
    im_dec = im_uav = None
    for col, step in enumerate(dec_steps):
        ax = fig.add_subplot(gs[0, col])
        dec = _last(by_step[step].get("decoder"))
        plane = (dec or {}).get("plane2d", {}).get("valid", {}).get("s_proj")
        if not plane:
            ax.axis("off")
            continue
        im_dec = _heat(ax, plane["counts"], plane["s_edges"], xlim=(-1.15, 1.15))
        acc = next((a for s, _, a in snaps if s == step), None)
        acc_txt = f"   acc {acc:.2f}" if acc is not None else ""
        ax.set_title(f"step {step}{acc_txt}", fontsize=9)
        ax.set_xlabel(r"gate $s$")
        if col == 0:
            ax.set_ylabel("Decoder\n" + r"content proj")
        else:
            ax.set_yticklabels([])
    for col in range(len(dec_steps), n):
        fig.add_subplot(gs[0, col]).axis("off")
    for col, (step, planes) in enumerate(uav_pick):
        ax = fig.add_subplot(gs[1, col])
        counts = np.sum([np.asarray(p["counts"], float) for p in planes], axis=0)
        im_uav = _heat(ax, counts, planes[0]["edges"], xlim=(-0.85, 0.85))
        corr = float(np.mean([p["corr"] for p in planes]))
        ax.set_title(f"step {step}    corr {corr:.2f}", fontsize=9)
        ax.set_xlabel(r"content $Ux$")
        if col == 0:
            ax.set_ylabel("UAV ctxsplit\n" + r"gate $Az$")
        else:
            ax.set_yticklabels([])
    if im_dec is not None:
        fig.colorbar(im_dec, cax=fig.add_subplot(gs[0, -1]), label="token density")
    if im_uav is not None:
        fig.colorbar(im_uav, cax=fig.add_subplot(gs[1, -1]), label="token density")
    out.mkdir(parents=True, exist_ok=True)
    _save(fig, out, "fig_two_heatmaps")


def plot_decoder_plane(snaps, out: Path) -> None:
    """Decoder (s, proj) at three steps — s is context, proj is content."""
    want_steps = [0, 200, 399]
    by_step = {s: g for s, g, _ in snaps}
    chosen = [s for s in want_steps if s in by_step] or [snaps[0][0], snaps[-1][0]]
    fig, axes = plt.subplots(1, len(chosen), figsize=(3.3 * len(chosen), 3.3), squeeze=False)
    last_im = None
    for ax, step in zip(axes[0], chosen):
        dec = _last(by_step[step].get("decoder"))
        plane = (dec or {}).get("plane2d", {}).get("valid", {}).get("s_proj")
        if not plane:
            ax.axis("off")
            continue
        z = np.asarray(plane["counts"], float)
        z = z / z.sum() if z.sum() else z
        se, pe = plane["s_edges"], plane["proj_edges"]
        positive = z[z > 0]
        vmin = max(float(positive.min()) if positive.size else 1e-6, 1e-6)
        last_im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            extent=[se[0], se[-1], pe[0], pe[-1]],
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=max(float(z.max()), vmin * 10)),
        )
        ax.plot([-1.2, 1.2], [-1.2, 1.2], color="white", ls="--", lw=0.7, alpha=0.7)
        se_a = np.asarray(se, float)
        pe_a = np.asarray(pe, float)
        sc = 0.5 * (se_a[:-1] + se_a[1:])
        pc = 0.5 * (pe_a[:-1] + pe_a[1:])
        ws, wp = z.sum(axis=0), z.sum(axis=1)
        s_std = float(np.sqrt(np.average((sc - np.average(sc, weights=ws)) ** 2, weights=ws))) if ws.sum() else 0.0
        p_std = float(np.sqrt(np.average((pc - np.average(pc, weights=wp)) ** 2, weights=wp))) if wp.sum() else 0.0
        acc = next((a for s, _, a in snaps if s == step), None)
        acc_txt = f"  acc {acc:.2f}" if acc is not None else ""
        ax.set_title(f"step {step}{acc_txt}\n" + rf"spread $\sigma_s$={s_std:.2f}  $\sigma_p$={p_std:.2f}", fontsize=9)
        ax.set_xlabel(r"gate $s$  (from context $z$)")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.4, 1.4)
    axes[0, 0].set_ylabel(r"content proj  (from $x$)")
    fig.suptitle(
        r"Decoder native plane: $s$ and proj are different coordinates (dashed $= s=\mathrm{proj}$)",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, label="mass")
    _save(fig, out, "fig_decoder_native_plane")


def plot_uav_heatmap(path: Path, out: Path) -> None:
    """Separate figure: ctxsplit (content, gate) mass leaving the diagonal."""
    if not path.exists():
        return
    row = json.loads(path.read_text())
    if isinstance(row, list):
        row = row[0]
    snaps = []
    for pt in row.get("history", []):
        planes = pt.get("uav_plane")
        if planes:
            snaps.append((int(pt["step"]), planes))
    if not snaps:
        return
    show = snaps if len(snaps) <= 4 else [snaps[0], snaps[len(snaps) // 3], snaps[2 * len(snaps) // 3], snaps[-1]]
    fig, axes = plt.subplots(1, len(show), figsize=(3.15 * len(show), 3.35), squeeze=False)
    last_im = None
    for ax, (step, planes) in zip(axes[0], show):
        counts = np.sum([np.asarray(p["counts"], float) for p in planes], axis=0)
        counts = counts / counts.sum() if counts.sum() else counts
        edges = planes[0]["edges"]
        pos = counts[counts > 0]
        vmin = max(float(pos.min()) if pos.size else 1e-6, 1e-6)
        last_im = ax.imshow(
            counts,
            origin="lower",
            aspect="equal",
            extent=[edges[0], edges[-1], edges[0], edges[-1]],
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=max(float(counts.max()), vmin * 10)),
        )
        ax.plot([-1, 1], [-1, 1], color="white", ls="--", lw=0.8, alpha=0.85)
        corr = float(np.mean([p["corr"] for p in planes]))
        cabs = float(np.mean([p["content_abs"] for p in planes]))
        gabs = float(np.mean([p["gate_abs"] for p in planes]))
        ax.set_title(
            f"step {step}\ncorr={corr:.2f}  |Ux|={cabs:.2f}  |Az|={gabs:.2f}",
            fontsize=9,
        )
        ax.set_xlabel(r"content $Ux$")
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-0.85, 0.85)
    axes[0, 0].set_ylabel(r"gate $Az$")
    fig.suptitle(
        r"UAV ctxsplit: heat on $(Ux,\; Az)$. Off the dashed line is the split.",
        fontsize=11,
        y=1.04,
    )
    fig.tight_layout()
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03, label="mass")
    out.mkdir(parents=True, exist_ok=True)
    _save(fig, out, "fig_uav_ctxsplit_heatmap")


def plot_centroid_path(snaps, out: Path) -> None:
    """Needle and hay trajectories in the (|s|, |Δx|) plane."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for region, color, marker in (("needle", NEEDLE, "o"), ("hay", HAY, "s")):
        steps, s_abs, _ = _series(snaps, "encoder", region, "s_abs_mean")
        _, dx, acc = _series(snaps, "encoder", region, "ff_out_rms")
        ax.plot(s_abs, dx, color=color, lw=1.4, label=region)
        ax.scatter(s_abs, dx, c=steps, cmap="cividis", s=36, marker=marker, zorder=3)
        for step, xs, ys, a in zip(steps, s_abs, dx, acc):
            if step in (0, 200, 399):
                tag = f"{step}"
                if np.isfinite(a):
                    tag += f"\nacc {a:.2f}"
                ax.annotate(tag, (xs, ys), textcoords="offset points", xytext=(6, 4), fontsize=7, color=color)
    ax.set_xlabel(r"mean $|s|$   (gate coordinate)")
    ax.set_ylabel(r"mean $|\Delta x|$   (what is written back)")
    ax.set_title("Same chain, as a path: hay leaves the needle")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, out, "fig_centroid_path")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "niah_memory_track_geom")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    hess = args.results_dir / "niah_hessian_split_seed0.json"
    snaps = _snaps(hess)
    if not snaps:
        raise SystemExit(f"no geometry in {hess}")
    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_trace_chain(snaps, args.results_dir / "niah_standard_seed0.json", args.output_dir)
    plot_curvature_mass(
        snaps,
        args.output_dir,
        ROOT / "results" / "wt103_ctxsplit_geom_smoke" / "summary.json",
    )
    plot_paired_heatmaps(
        snaps,
        ROOT / "results" / "uav_ctxsplit_plane" / "h4_phi768_prefix_ms_decouple_ctxsplit_seed0.json",
        ROOT / "figures" / "memory_track_geom",
    )
    print(f"→ {args.output_dir}")


if __name__ == "__main__":
    main()
