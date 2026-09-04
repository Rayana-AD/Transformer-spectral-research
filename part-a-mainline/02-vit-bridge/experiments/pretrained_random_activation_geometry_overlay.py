#!/usr/bin/env python3
"""Overlay random-init and pretrained activation geometry for ViT/Marian.

This script reads the existing random-init c_l rows produced by
`random_init_activation_geometry_compare.py`, computes matching pretrained rows
without truncating layers, and plots random vs pretrained side-by-side in the
same 3x3 panel.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from activation_extras import LightDepthAct
from paper_style import apply_paper_style
from script5_vit_representation_svd import IMAGENET_MEAN, IMAGENET_STD, hidden_states_to_metrics

try:
    from transformers import AutoTokenizer, MarianModel, ViTModel
    from transformers.utils import logging as tf_logging

    tf_logging.set_verbosity_error()
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"需要 transformers: pip install transformers\n{e}") from e


ACT_ORDER = ["relu", "gelu", "light_depth_t2.0"]
ACT_LABEL = {
    "relu": r"$\mathrm{ReLU}$",
    "gelu": r"$\mathrm{GeLU}$",
    "light_depth_t2.0": r"Light depth $t{=}2$",
}
FAMILIES = ["vit", "marian_encoder", "marian_decoder"]
FAMILY_LABEL = {
    "vit": "ViT",
    "marian_encoder": "Marian encoder",
    "marian_decoder": "Marian decoder",
}
VARIANTS = ["random", "pretrained"]
VARIANT_STYLE = {
    "random": {"facecolor": "white", "edgecolor": "#2563eb", "median": "#1d4ed8", "label": "random init"},
    "pretrained": {"facecolor": "#fed7aa", "edgecolor": "#c2410c", "median": "#9a3412", "label": "pretrained"},
}


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_activation(act_id: str) -> nn.Module:
    if act_id == "relu":
        return nn.ReLU()
    if act_id == "gelu":
        return nn.GELU()
    if act_id == "light_depth_t2.0":
        return LightDepthAct(2.0)
    raise ValueError(f"Unknown activation: {act_id}")


def set_vit_activation(model: ViTModel, act_id: str) -> None:
    act = make_activation(act_id)
    for layer in model.encoder.layer:
        layer.intermediate.intermediate_act_fn = act


def set_marian_activation(model: MarianModel, act_id: str) -> None:
    act = make_activation(act_id)
    for layer in model.encoder.layers:
        layer.activation_fn = act
    for layer in model.decoder.layers:
        layer.activation_fn = act


def c_rows_from_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    *,
    family: str,
    variant: str,
    activation_id: str,
    seed: int,
    note: str,
) -> list[dict]:
    layer_records, c_ts = hidden_states_to_metrics(hidden_states)
    rows: list[dict] = []
    for i, c_t in enumerate(c_ts[:-1]):
        if c_t is None:
            continue
        rows.append(
            {
                "model_family": family,
                "init_variant": variant,
                "activation_id": activation_id,
                "seed": seed,
                "layer_transition": i + 1,
                "transition_label": f"{i}->{i + 1}",
                "c_t": float(c_t),
                "nuclear_from": layer_records[i]["nuclear_norm"],
                "nuclear_to": layer_records[i + 1]["nuclear_norm"],
                "note": note,
            }
        )
    return rows


def normalize_random_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    mapping = {
        "vit_random": "vit",
        "marian_encoder_random": "marian_encoder",
        "marian_decoder_random": "marian_decoder",
    }
    for row in rows:
        family = mapping.get(row.get("model_family", ""), row.get("model_family", ""))
        if family not in FAMILIES:
            continue
        new = dict(row)
        new["model_family"] = family
        new["init_variant"] = "random"
        out.append(new)
    return out


def make_vit_pixels(args: argparse.Namespace, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    pixels = torch.randn(args.batch, 3, args.vit_image_size, args.vit_image_size, device=device)
    if args.vit_input_kind == "imagenet_norm_randn":
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        pixels = (pixels - mean) / std
    return pixels


GERMAN_SENTENCES = [
    "Das ist ein kurzer Test.",
    "Die Katze sitzt auf dem Tisch.",
    "Wir untersuchen die Geometrie der Schichten.",
    "Maschinelle Übersetzung ist empfindlich gegenüber Reihenfolge.",
    "Heute messen wir die Kernnormen.",
    "Ein kleines Beispiel reicht für die Vorwärtsanalyse.",
]
ENGLISH_SENTENCES = [
    "This is a short test.",
    "The cat sits on the table.",
    "We study the geometry of the layers.",
    "Machine translation is sensitive to order.",
    "Today we measure nuclear norms.",
    "A small example is enough for the forward analysis.",
]


def make_marian_inputs(
    args: argparse.Namespace,
    device: torch.device,
    tokenizer: AutoTokenizer | None,
    seed: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tokenizer is None or args.marian_input_kind == "random_ids":
        torch.manual_seed(seed)
        input_ids = torch.randint(3, vocab_size, (args.batch, args.seq_len), device=device)
        decoder_input_ids = torch.randint(3, vocab_size, (args.batch, args.decoder_seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, decoder_input_ids, attention_mask

    start = (seed - args.base_seed) % len(GERMAN_SENTENCES)
    src = [GERMAN_SENTENCES[(start + i) % len(GERMAN_SENTENCES)] for i in range(args.batch)]
    tgt = [ENGLISH_SENTENCES[(start + i) % len(ENGLISH_SENTENCES)] for i in range(args.batch)]
    enc = tokenizer(
        src,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.seq_len,
    )
    try:
        dec = tokenizer(
            text_target=tgt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.decoder_seq_len,
        )
        decoder_input_ids = dec["input_ids"]
    except TypeError:
        with tokenizer.as_target_tokenizer():
            dec = tokenizer(
                tgt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.decoder_seq_len,
            )
        decoder_input_ids = dec["input_ids"]
    return enc["input_ids"].to(device), decoder_input_ids.to(device), enc["attention_mask"].to(device)


def compute_pretrained_rows(args: argparse.Namespace, device: torch.device) -> list[dict]:
    rows: list[dict] = []
    tokenizer: AutoTokenizer | None = None
    if args.marian_input_kind == "tokenized_sentences":
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.marian_model_id,
                local_files_only=args.local_files_only,
            )
        except Exception as e:
            print(f"[warn] tokenizer load failed, falling back to random_ids: {e}", flush=True)
            tokenizer = None

    for act_id in ACT_ORDER:
        print(f"[pretrained] loading ViT {args.vit_model_id} activation={act_id}", flush=True)
        vit = ViTModel.from_pretrained(args.vit_model_id, local_files_only=args.local_files_only).to(device)
        set_vit_activation(vit, act_id)
        vit.eval()
        for si in range(args.num_seeds):
            seed = args.base_seed + si
            pixels = make_vit_pixels(args, device, seed)
            with torch.no_grad():
                out = vit(pixel_values=pixels, output_hidden_states=True, return_dict=True)
            assert out.hidden_states is not None
            rows.extend(
                c_rows_from_hidden_states(
                    out.hidden_states,
                    family="vit",
                    variant="pretrained",
                    activation_id=act_id,
                    seed=seed,
                    note=f"pretrained {args.vit_model_id}; full encoder; no truncate; activation swapped after load",
                )
            )
            del pixels, out
        del vit
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"[pretrained] loading Marian {args.marian_model_id} activation={act_id}", flush=True)
        marian = MarianModel.from_pretrained(args.marian_model_id, local_files_only=args.local_files_only).to(device)
        set_marian_activation(marian, act_id)
        marian.eval()
        vocab_size = int(marian.config.vocab_size)
        for si in range(args.num_seeds):
            seed = args.base_seed + si
            input_ids, decoder_input_ids, attention_mask = make_marian_inputs(args, device, tokenizer, seed, vocab_size)
            with torch.no_grad():
                out_m = marian(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
            assert out_m.encoder_hidden_states is not None
            assert out_m.decoder_hidden_states is not None
            rows.extend(
                c_rows_from_hidden_states(
                    out_m.encoder_hidden_states,
                    family="marian_encoder",
                    variant="pretrained",
                    activation_id=act_id,
                    seed=seed,
                    note=f"pretrained {args.marian_model_id} encoder; no truncate; activation swapped after load",
                )
            )
            rows.extend(
                c_rows_from_hidden_states(
                    out_m.decoder_hidden_states,
                    family="marian_decoder",
                    variant="pretrained",
                    activation_id=act_id,
                    seed=seed,
                    note=f"pretrained {args.marian_model_id} decoder; no truncate; activation swapped after load",
                )
            )
            del input_ids, decoder_input_ids, attention_mask, out_m
        del marian
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def box_vals(rows: list[dict], family: str, variant: str, act: str, layer_transition: int) -> list[float]:
    vals: list[float] = []
    for r in rows:
        if (
            r.get("model_family") == family
            and r.get("init_variant") == variant
            and r.get("activation_id") == act
            and int(r.get("layer_transition", 0)) == layer_transition
        ):
            vals.append(float(r["c_t"]))
    return vals


def max_layer(rows: list[dict], family: str) -> int:
    vals = [int(r["layer_transition"]) for r in rows if r.get("model_family") == family]
    return max(vals, default=0)


def nice_ylim(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo, hi = float(arr.min()), float(arr.max())
    span = max(hi - lo, 0.05)
    pad = max(span * 0.12, 0.012)
    ymin, ymax = lo - pad, hi + pad
    ymin = min(ymin, 1.0) - pad * 0.2
    ymax = max(ymax, 1.0) + pad * 0.2
    return ymin, ymax


def robust_ylim(vals: list[float]) -> tuple[float, float]:
    """Zoomed range: keep the dominant trend visible while allowing clipped markers."""
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(arr, 0.10))
    hi = float(np.quantile(arr, 0.93))
    span = max(hi - lo, 0.05)
    pad = max(span * 0.20, 0.018)
    ymin, ymax = lo - pad, hi + pad
    ymin = min(ymin, 1.0) - pad * 0.15
    ymax = max(ymax, 1.0) + pad * 0.15
    return ymin, ymax


def plot_clipped_median_markers(
    ax: plt.Axes,
    rows: list[dict],
    family: str,
    act: str,
    n_layer: int,
    ylim: tuple[float, float],
) -> None:
    ymin, ymax = ylim
    span = ymax - ymin
    y_top = ymax - span * 0.025
    y_bot = ymin + span * 0.025
    for variant, offset in (("random", -0.18), ("pretrained", 0.18)):
        style = VARIANT_STYLE[variant]
        for li in range(1, n_layer + 1):
            vals = box_vals(rows, family, variant, act, li)
            if not vals:
                continue
            med = float(np.median(vals))
            q25 = float(np.quantile(vals, 0.25))
            q75 = float(np.quantile(vals, 0.75))
            clipped_by_act: list[float] = []
            for act_id in ACT_ORDER:
                same = box_vals(rows, family, variant, act_id, li)
                if same:
                    clipped_by_act.append(float(np.median(same)))
            cross_act_stable = (
                len(clipped_by_act) == len(ACT_ORDER)
                and (all(v > ymax for v in clipped_by_act) or all(v < ymin for v in clipped_by_act))
            )
            stable_suffix = "*" if cross_act_stable else ""
            label = f"{med:.2f}{stable_suffix}"
            if med > ymax:
                ax.scatter(li + offset, y_top, marker="^", s=22, color=style["edgecolor"], zorder=5, clip_on=False)
                ax.annotate(
                    label,
                    xy=(li + offset, y_top),
                    xytext=(0, -11),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="black",
                    clip_on=False,
                )
            elif med < ymin:
                ax.scatter(li + offset, y_bot, marker="v", s=22, color=style["edgecolor"], zorder=5, clip_on=False)
                ax.annotate(
                    label,
                    xy=(li + offset, y_bot),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                    clip_on=False,
                )


def plot_overlay(rows: list[dict], out_dir: Path, stem: str, *, zoom: bool = False) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 8.6), sharex=False, sharey=False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.17, wspace=0.22, hspace=0.42)

    for row_idx, family in enumerate(FAMILIES):
        row_vals = [
            float(r["c_t"])
            for r in rows
            if r.get("model_family") == family and np.isfinite(float(r["c_t"]))
        ]
        row_ylim = (0.6, 1.6) if zoom else nice_ylim(row_vals)
        n_layer = max_layer(rows, family)
        for col_idx, act in enumerate(ACT_ORDER):
            ax = axes[row_idx, col_idx]
            ylim = row_ylim
            for variant, offset in (("random", -0.18), ("pretrained", 0.18)):
                data = [box_vals(rows, family, variant, act, li) for li in range(1, n_layer + 1)]
                positions = [li + offset for li in range(1, n_layer + 1)]
                if not any(data):
                    continue
                style = VARIANT_STYLE[variant]
                bp = ax.boxplot(
                    data,
                    positions=positions,
                    widths=0.28,
                    showfliers=False,
                    patch_artist=True,
                    manage_ticks=False,
                    medianprops=dict(color=style["median"], linewidth=1.6),
                    whiskerprops=dict(color=style["edgecolor"], linewidth=0.9),
                    capprops=dict(color=style["edgecolor"], linewidth=0.9),
                )
                for box in bp["boxes"]:
                    box.set(facecolor=style["facecolor"], edgecolor=style["edgecolor"], linewidth=0.95)
            ax.axhline(1.0, color="gray", ls="--", lw=0.85, zorder=0)
            ax.set_xlim(0.45, n_layer + 0.55)
            ax.set_ylim(*ylim)
            ax.set_xticks(range(1, n_layer + 1))
            ax.set_xticklabels([str(i) for i in range(1, n_layer + 1)])
            if zoom:
                ax.yaxis.set_major_locator(MultipleLocator(0.1))
                ax.yaxis.set_minor_locator(MultipleLocator(0.05))
                ax.tick_params(axis="y", which="major", length=4)
                ax.tick_params(axis="y", which="minor", length=2)
            ax.grid(False)
            if zoom:
                plot_clipped_median_markers(ax, rows, family, act, n_layer, ylim)
            if row_idx == 0:
                ax.annotate(
                    ACT_LABEL[act],
                    xy=(0.5, 1.0),
                    xycoords="axes fraction",
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    clip_on=False,
                )
            if col_idx == 0:
                ax.set_ylabel(rf"$c_\ell$ ({FAMILY_LABEL[family]})", fontsize=9.5)
            if row_idx == len(FAMILIES) - 1:
                ax.set_xlabel(r"Block/layer transition $\ell$ ($h_{\ell-1}\to h_\ell$)")

    handles = [
        Patch(
            facecolor=VARIANT_STYLE[v]["facecolor"],
            edgecolor=VARIANT_STYLE[v]["edgecolor"],
            label=VARIANT_STYLE[v]["label"],
        )
        for v in VARIANTS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.50, 0.055))
    if zoom:
        fig.text(
            0.50,
            0.025,
            r"Uniform y-range $0.6$--$1.6$; triangles mark clipped medians (* = clipped across all activations)",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.35",
        )
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for family in FAMILIES:
        for variant in VARIANTS:
            for act in ACT_ORDER:
                for li in range(1, max_layer(rows, family) + 1):
                    vals = box_vals(rows, family, variant, act, li)
                    if not vals:
                        continue
                    out.append(
                        {
                            "model_family": family,
                            "init_variant": variant,
                            "activation_id": act,
                            "layer_transition": li,
                            "n": len(vals),
                            "c_t_mean": float(np.mean(vals)),
                            "c_t_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "c_t_median": float(np.median(vals)),
                            "c_t_q25": float(np.quantile(vals, 0.25)),
                            "c_t_q75": float(np.quantile(vals, 0.75)),
                        }
                    )
    return out


def write_note(args: argparse.Namespace, out_dir: Path, rows: list[dict], summary: list[dict], stem: str) -> None:
    lines = [
        "# Random vs pretrained activation geometry overlay",
        "",
        f"Figure: `{stem}.png` / `.pdf`.",
        f"Zoomed trend figure: `{stem}_zoom.png` / `.pdf`.",
        "",
        "## Setup",
        "",
        f"- Pretrained ViT: `{args.vit_model_id}`.",
        f"- Pretrained Marian: `{args.marian_model_id}`.",
        "- No truncation: full available stacks are used.",
        "- Columns: ReLU, GeLU, Light depth `t=2`.",
        "- Legend: blue/white = random initialization; orange = pretrained checkpoint loaded, then activation swapped.",
        "- x-axis is transition index: `ell=1` means `h_0 -> h_1`, not absolute block id. The plotting code disables Matplotlib boxplot auto-ticks and explicitly labels `1..L`.",
        "- The zoomed figure uses one shared y-range for all subplots (`0.6..1.6`) with major ticks every `0.1` and minor ticks every `0.05`. Triangles mark medians outside the visible range and are annotated with the clipped median value in black. A trailing `*` means the same transition is clipped in the same direction across all three activations for that model/variant.",
        f"- Pretrained Marian input kind: `{args.marian_input_kind}`.",
        "",
        "## Experiment Settings",
        "",
        "- **Random initialization.** The random baseline comes from `experiments/random_init_activation_geometry_compare.py`: a randomly initialized ViT with 12 encoder blocks, and a randomly initialized Marian-style encoder-decoder with 6 encoder layers and 6 decoder layers. No checkpoint is loaded.",
        f"- **Pretrained checkpoints.** The pretrained ViT is `{args.vit_model_id}`; the pretrained Marian model is `{args.marian_model_id}`. Both are loaded from the Hugging Face cache when `--local_files_only` is used.",
        "- **No truncation.** All available layers/blocks are kept: 12 ViT encoder blocks, 6 Marian encoder layers, and 6 Marian decoder layers.",
        "- **Activation intervention.** For ViT, each encoder block MLP uses `layer.intermediate.intermediate_act_fn`. For Marian, both encoder and decoder FFN layers use `layer.activation_fn`. We compare ReLU, GeLU, and `LightDepthAct(2.0)`.",
        "- **Inputs.** ViT uses Gaussian images with ImageNet normalization. Marian pretrained runs use short tokenized German/English sentence pairs for encoder and decoder inputs; random Marian runs use random token IDs from the configured vocabulary.",
        "- **Metric.** For every hidden state `h_ell`, the token-by-hidden matrix is SVD-decomposed per sample; nuclear norms are averaged across the batch. The plotted quantity is `c_ell = N(h_ell) / N(h_{ell-1})`, so `ell=1` means the transition `h_0 -> h_1`.",
        "- **Replicates.** Each box is over six seeds (`42..47`). The pretrained weights are fixed; seed changes only the input batch.",
        "",
        "## Conference-style Caption",
        "",
        "**Figure 41. Random-initialized versus pretrained layerwise geometric response under activation interventions.** Rows show ViT, Marian encoder, and Marian decoder; columns show ReLU, GeLU, and Light depth $t=2$. Each boxplot reports the layer-transition ratio $c_\\ell=N(h_\\ell)/N(h_{\\ell-1})$, where $N(\\cdot)$ is the batch-averaged nuclear norm of the token-by-hidden representation matrix. Blue/white boxes are random initialization; orange boxes load pretrained checkpoints (`google/vit-base-patch16-224` and `Helsinki-NLP/opus-mt-de-en`) and then replace the FFN/MLP activation while keeping weights fixed. No layers are truncated. The zoomed version uses a shared y-axis range `0.6..1.6` in all subplots; triangles denote clipped medians and print their values; `*` marks transitions clipped in the same direction across all three activations.",
        "",
        "## How to Read the Figure",
        "",
        "- The x-axis is **transition index**, not the absolute block label. `1` means `h_0 -> h_1`; ViT has transitions `1..12`, Marian encoder/decoder have `1..6`.",
        "- The dashed horizontal line is `c_ell=1`. Values above 1 indicate nuclear-norm expansion across the transition; values below 1 indicate compression.",
        "- Compare **blue/white vs orange within the same panel** to see how pretraining reshapes the geometry for a fixed architecture and activation.",
        "- Compare **columns within a row** to see how the activation changes the geometry under the same model family.",
        "- In the zoomed figure, all subplots use the same y-axis range (`0.6..1.6`) and the same tick spacing. Upward/downward triangles mark medians outside the displayed range and print the clipped median in black. A `*` means that the same transition is clipped in the same direction for ReLU, GeLU, and Light depth within that model/variant. In particular, the Marian encoder's sixth transition is consistently low under pretraining (`0.36`, `0.37`, `0.47`), and the Marian decoder's sixth transition is consistently high (`4.16`, `4.26`, `4.12`), while pretrained ViT Light depth has a high late transition near `1.65` that is activation-specific rather than cross-activation stable.",
        "",
        "## Main Observations",
        "",
        "### Random initialization",
        "",
        "- **Random ViT:** all three activations have `c_ell > 1`, and the ratios gradually relax toward 1 with depth. ReLU has the strongest initial expansion, while Light depth `t=2` has the weakest.",
        "- **Random Marian Encoder:** the first transition `h_0 -> h_1` strongly expands the nuclear norm (about `1.57`), while later transitions stay close to `0.99-1.00`.",
        "- **Random Marian Decoder:** the first transition also strongly expands (about `1.56`), while later transitions are mostly slightly below 1 (about `0.985-0.989`).",
        "- These random-init trends isolate the **initial geometric bias** of the architecture and activation before task training. They show that residual Transformer blocks are not geometrically neutral at initialization: the first transition can dominate the nuclear-norm scale, and activation choice changes the strength of this initial expansion.",
        "",
        "### Pretrained geometric response",
        "",
        "- Under pretrained checkpoints, the observed activation-swap response should be interpreted as `observed c_ell change = activation-intrinsic geometric effect + mismatch/interactions with GeLU-pretrained weights`.",
        "- In other words, this panel does **not** claim that ReLU/GeLU/Light depth have a standalone universal ranking. It measures how sensitive an already trained GeLU representation flow is to activation perturbations.",
        "- **ViT GeLU:** random initialization has the trend `1.18 -> 1.13 -> 1.10 -> ... -> 1.03`, i.e. early expansion followed by relaxation toward 1. The pretrained GeLU trajectory instead compresses in early blocks (`~0.97 -> 0.95 -> 0.97`) and then rises toward or above 1 in later blocks. This means pretraining reshapes the layerwise geometry rather than merely smoothing the random-init curve.",
        "- **ViT Light depth:** pretrained weights are especially sensitive in the late ViT stack; around transition 10 the median `c_ell` is about `1.65`, far above the random-init Light depth curve.",
        "- **Marian Encoder:** after pretraining, the encoder is no longer the random pattern of first-step expansion followed by near-identity transitions. It shows a learned layer structure, including mid-stack compression and a strong final compression.",
        "- **Marian Decoder:** after pretraining, the final decoder transition expands extremely strongly (`c_ell` about `4.2`). This is qualitatively different from ViT and from the encoder, and is consistent with the decoder/generation side having a distinct geometric regime.",
        "",
        "## LaTeX Body Text",
        "",
        "```latex",
        r"\paragraph{Activation geometry before and after pretraining.}",
        r"Figure~\ref{fig41} compares layerwise geometric ratios under random initialization and pretrained checkpoints for three Transformer families: ViT, the Marian encoder, and the Marian decoder.  For each hidden state $h_\ell$, we compute the batch-averaged nuclear norm $N(h_\ell)$ of the token-by-hidden representation matrix and report the transition ratio",
        r"\begin{equation}",
        r"c_\ell = \frac{N(h_\ell)}{N(h_{\ell-1})},",
        r"\end{equation}",
        r"where $\ell=1$ denotes the transition $h_0\to h_1$.  The ViT row uses the full 12-block encoder, while Marian uses all 6 encoder layers and all 6 decoder layers; no stack is truncated.  The pretrained runs load \texttt{google/vit-base-patch16-224} and \texttt{Helsinki-NLP/opus-mt-de-en}, after which only the FFN/MLP activation is replaced by ReLU, GeLU, or Light depth $t=2$.",
        "",
        r"Random initialization exposes the initial geometric bias induced by architecture and activation.  In the ViT row, all three activations yield $c_\ell>1$ and gradually relax toward one as depth increases.  ReLU has the largest initial expansion, whereas Light depth $t=2$ has the weakest.  In the randomly initialized Marian encoder, the first transition expands strongly ($c_\ell\approx1.57$), but subsequent transitions remain near $0.99$--$1.00$.  The randomly initialized Marian decoder behaves similarly at the first transition ($c_\ell\approx1.56$), while later transitions are slightly contractive, around $0.985$--$0.989$.  Thus, before learning, the dominant geometric effect is an early residual/FFN expansion followed by near-critical propagation.",
        "",
        r"Pretraining qualitatively changes these trajectories.  For GeLU ViT, the random-initialized curve roughly follows $1.18\to1.13\to1.10\to\cdots\to1.03$, i.e. early expansion followed by relaxation toward one.  The pretrained GeLU curve instead compresses in the early blocks ($0.97\to0.95\to0.97$) and rises in the later blocks toward or above one.  This reversal indicates that pretraining does not merely smooth the initial geometry; it learns a task-adapted layerwise geometric trajectory in which early layers filter or compress representations and later layers re-expand or reorganize them.",
        "",
        r"Activation swaps under pretrained weights should therefore be interpreted as perturbations of a learned GeLU geometry rather than as pure activation-only effects.  The measured change in $c_\ell$ combines the intrinsic geometric bias of the activation with its interaction or mismatch with weights optimized under GeLU.  This interaction is especially visible for Light depth in pretrained ViT, where late transitions are strongly amplified, with transition 10 reaching a median $c_\ell\approx1.65$.  Marian shows a different pretrained response: the encoder develops mid-stack compression and a strong final compression, while the decoder exhibits an extreme final expansion ($c_\ell\approx4.2$).  The decoder behavior suggests that autoregressive generation induces a separate geometric regime from encoder-only visual recognition and from the translation encoder.",
        "",
        r"\begin{figure}[t]",
        r"  \centering",
        r"  \includegraphics[width=\linewidth]{fig_random_pretrained_activation_geometry_overlay_zoom.pdf}",
        r"  \caption{Random-initialized versus pretrained layerwise geometric response under activation interventions. Rows show ViT, Marian encoder, and Marian decoder; columns show ReLU, GeLU, and Light depth $t=2$. Each boxplot reports $c_\ell=N(h_\ell)/N(h_{\ell-1})$. Blue/white boxes are random initialization; orange boxes load pretrained checkpoints and then replace the FFN/MLP activation while keeping weights fixed. No layers are truncated. The displayed version uses a shared y-axis range $0.6$--$1.6$ for all subplots; triangles mark medians outside the visible range and annotate their values; $*$ marks transitions clipped in the same direction across all three activations.}",
        r"  \label{fig41}",
        r"\end{figure}",
        "```",
        "",
        "## Files",
        "",
        "- `pretrained_activation_c_raw.csv`: pretrained-only raw c rows.",
        "- `random_pretrained_activation_c_raw.csv`: merged random + pretrained raw c rows.",
        "- `random_pretrained_activation_c_summary.csv`: per transition summaries.",
        "",
        "## Median snapshots",
        "",
    ]
    for family in FAMILIES:
        lines.extend([f"### {FAMILY_LABEL[family]}", ""])
        n_layer = max_layer(rows, family)
        header = "| variant / activation | " + " | ".join(str(i) for i in range(1, n_layer + 1)) + " |"
        sep = "|" + "|".join(["---"] * (n_layer + 1)) + "|"
        lines.extend([header, sep])
        for variant in VARIANTS:
            for act in ACT_ORDER:
                cells: list[str] = []
                for li in range(1, n_layer + 1):
                    vals = [
                        r
                        for r in summary
                        if r["model_family"] == family
                        and r["init_variant"] == variant
                        and r["activation_id"] == act
                        and int(r["layer_transition"]) == li
                    ]
                    cells.append(f"{float(vals[0]['c_t_median']):.4f}" if vals else "—")
                lines.append(f"| `{variant}` / `{act}` | " + " | ".join(cells) + " |")
        lines.append("")
    (out_dir / "random_pretrained_activation_geometry_overlay_note.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=_ROOT / "results_random_init_activation_geometry")
    p.add_argument(
        "--random_raw",
        type=Path,
        default=_ROOT / "results_random_init_activation_geometry" / "random_init_activation_c_raw.csv",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--num_seeds", type=int, default=6)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--vit_model_id", default="google/vit-base-patch16-224")
    p.add_argument("--marian_model_id", default="Helsinki-NLP/opus-mt-de-en")
    p.add_argument("--vit_image_size", type=int, default=224)
    p.add_argument("--vit_input_kind", choices=["randn", "imagenet_norm_randn"], default="imagenet_norm_randn")
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--decoder_seq_len", type=int, default=32)
    p.add_argument("--marian_input_kind", choices=["tokenized_sentences", "random_ids"], default="tokenized_sentences")
    p.add_argument("--local_files_only", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random_rows = normalize_random_rows(read_csv(args.random_raw))
    device = resolve_device(args.device)
    pretrained_rows = compute_pretrained_rows(args, device)
    save_csv(args.out_dir / "pretrained_activation_c_raw.csv", pretrained_rows)
    merged = random_rows + pretrained_rows
    save_csv(args.out_dir / "random_pretrained_activation_c_raw.csv", merged)
    summary = summarize(merged)
    save_csv(args.out_dir / "random_pretrained_activation_c_summary.csv", summary)
    stem = "fig_random_pretrained_activation_geometry_overlay"
    plot_overlay(merged, args.out_dir, stem)
    plot_overlay(merged, args.out_dir, f"{stem}_zoom", zoom=True)
    write_note(args, args.out_dir, merged, summary, stem)
    print(f"Wrote overlay outputs under {args.out_dir}")


if __name__ == "__main__":
    main()
