# Script11 — TLB-Attention vs vanilla / ablations on the script10 synthetic LM stack.
# Reuses script10 presets, val cache, and train_run; trains a single SeqModelTransformerFlexible.
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import script10_token_mirror as s10
from script10_token_mirror import (
    IGNORE_LABEL,
    apply_script10_preset,
    apply_script10_train_recipe,
    build_argparser,
    ensure_fixed_val_set,
    masked_ce_loss,
    plot_loss_and_acc,
    resolve_device,
    save_csv,
    sinusoidal_position_encoding,
    train_run,
)
from tlb_attention_core import AttentionMode, SeqModelTransformerFlexible, _run_numerical_self_tests
from transformer_mechanics import probe_seq_model_flexible

DEFAULT_OUTPUT = Path("results_script11_tlb_attention")


def _jsonable_probe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable_probe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable_probe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (float, int, str)) or obj is None:
        return obj
    return str(obj)


def _transformer_pos_enc(args: argparse.Namespace) -> str:
    return s10._transformer_pos_enc(args)


def _transformer_legacy_baseline(args: argparse.Namespace) -> bool:
    return s10._transformer_legacy_baseline(args)


def validate_and_task_layout(args: argparse.Namespace) -> dict[str, Any]:
    """Mirror script10 run_one_seed checks; return vocab sizes and max_len."""
    if args.dim % args.nhead != 0:
        raise ValueError("dim must be divisible by nhead.")
    if (
        args.task not in ("niah", "niah_prefix", "kv_assoc", "story_assoc")
        and not args.val_random_n
    ):
        if args.val_n < args.n_min or args.val_n > args.n_max:
            raise ValueError("val_n must satisfy n_min <= val_n <= n_max (or --val-random-n).")
    if args.task == "gap_copy":
        if int(args.gap_min) < 0 or int(args.gap_max) < int(args.gap_min):
            raise ValueError("gap_copy requires 0 <= gap-min <= gap-max")
        if not args.val_random_n:
            if int(args.val_gap) < int(args.gap_min) or int(args.val_gap) > int(args.gap_max):
                raise ValueError("val-gap must lie in [gap-min, gap-max] for gap_copy")
    if args.task == "niah":
        if int(args.niah_hay_len) < 0 or int(args.niah_needle_len) < 1:
            raise ValueError("niah requires --niah-hay-len >= 0 and --niah-needle-len >= 1")
    if args.task == "niah_prefix":
        nl = int(args.niah_prefix_needle_len)
        lo = int(args.niah_prefix_len_min)
        hi = int(args.niah_prefix_len_max)
        vl = int(args.niah_prefix_val_len)
        ood = bool(getattr(args, "niah_prefix_val_ood", False))
        if nl < 1 or lo > hi or lo < 2 * nl + 1 or hi > 1024 or vl > 1024:
            raise ValueError("niah_prefix: invalid needle or length bounds")
        if vl < 2 * nl + 1:
            raise ValueError("niah_prefix: val_len must be >= 2*needle_len+1")
        if ood:
            if vl <= hi:
                raise ValueError("niah_prefix: --niah-prefix-val-ood requires val_len > len_max")
        elif vl < lo or vl > hi:
            raise ValueError("niah_prefix: val_len must be in [len_min, len_max]")
    if args.task == "kv_assoc":
        P = int(args.kv_num_pairs)
        vc = int(args.vocab_content)
        key_hi = vc // 2
        if P < 1 or key_hi < P or (vc - key_hi) < P:
            raise ValueError("kv_assoc: vocab too small for P pairs")
        dm, dx = int(args.kv_dist_min), int(args.kv_dist_max)
        if dm < 0 or dx < dm:
            raise ValueError("kv_assoc: require 0 <= kv-dist-min <= kv-dist-max")
        vdl = int(args.kv_val_dist_len)
        if vdl < dm or vdl > dx:
            raise ValueError("kv_assoc: kv-val-dist-len must lie in distr range")
        pg0, pg1 = int(args.kv_pair_gap_min), int(args.kv_pair_gap_max)
        if pg0 < 0 or pg1 < pg0:
            raise ValueError("kv_assoc: need 0 <= kv-pair-gap-min <= kv-pair-gap-max")
    if args.task == "story_assoc":
        P = int(args.story_num_facts)
        vc = int(args.vocab_content)
        key_hi = vc // 2
        if P < 1 or key_hi < P or (vc - key_hi) < P:
            raise ValueError("story_assoc: vocab too small for P facts")
        om, ox = int(args.story_open_min), int(args.story_open_max)
        if om < 0 or ox < om:
            raise ValueError("story_assoc: need 0 <= story-open-min <= story-open-max")
        pg0, pg1 = int(args.story_pair_gap_min), int(args.story_pair_gap_max)
        if pg0 < 0 or pg1 < pg0:
            raise ValueError("story_assoc: invalid pair gap range")
        dm, dx = int(args.story_dist_min), int(args.story_dist_max)
        if dm < 0 or dx < dm:
            raise ValueError("story_assoc: invalid distr range")
        vol = int(args.story_val_open_len)
        if vol < om or vol > ox:
            raise ValueError("story_assoc: story-val-open-len must lie in open range")
        vdl = int(args.story_val_dist_len)
        if vdl < dm or vdl > dx:
            raise ValueError("story_assoc: story-val-dist-len must lie in distr range")
        qix = int(args.story_val_query_idx)
        if qix < 0 or qix >= P:
            raise ValueError("story_assoc: story-val-query-idx must be in [0, P-1]")

    vocab_content = args.vocab_content
    pad_id = vocab_content
    if args.task == "identity":
        vocab_size = vocab_content + 1
        delim_id = vocab_content + 1
    elif args.task in ("kv_assoc", "story_assoc"):
        delim_id = vocab_content + 1
        vocab_size = vocab_content + 3
    else:
        delim_id = vocab_content + 1
        vocab_size = vocab_content + 2
    if args.task == "identity":
        max_len = args.n_max
    elif args.task == "gap_copy":
        max_len = int(args.n_max) + int(args.gap_max) + 1 + int(args.n_max)
    elif args.task == "dup_tail_copy":
        max_len = 3 * int(args.n_max) + 1
    elif args.task == "niah":
        max_len = 2 * int(args.niah_hay_len) + 2 * int(args.niah_needle_len) + 1
    elif args.task == "niah_prefix":
        max_len = max(int(args.niah_prefix_len_max), int(args.niah_prefix_val_len))
    elif args.task == "kv_assoc":
        Pk = int(args.kv_num_pairs)
        max_len = (
            3 * Pk
            + max(0, Pk - 1) * int(args.kv_pair_gap_max)
            + 1
            + int(args.kv_dist_max)
            + 2
        )
    elif args.task == "story_assoc":
        Ps = int(args.story_num_facts)
        max_len = (
            int(args.story_open_max)
            + 3 * Ps
            + max(0, Ps - 1) * int(args.story_pair_gap_max)
            + 1
            + int(args.story_dist_max)
            + 2
        )
    else:
        max_len = 2 * args.n_max + 1
    dim_ff = 4 * args.dim
    return {
        "vocab_content": vocab_content,
        "pad_id": pad_id,
        "delim_id": delim_id,
        "vocab_size": vocab_size,
        "max_len": max_len,
        "dim_ff": dim_ff,
    }


def build_flexible_model(
    args: argparse.Namespace,
    layout: dict[str, Any],
    device: torch.device,
) -> SeqModelTransformerFlexible:
    pos = _transformer_pos_enc(args)
    max_len = int(layout["max_len"])
    if pos == "sinusoidal":
        pos_sin = sinusoidal_position_encoding(max_len, int(args.dim))
    else:
        pos_sin = None
    return SeqModelTransformerFlexible(
        int(layout["vocab_size"]),
        int(args.dim),
        int(args.depth),
        int(args.nhead),
        dim_feedforward=int(layout["dim_ff"]),
        max_seq_len=max_len,
        pos_enc=pos,
        ffn_activation=args.ffn_activation,
        dropout=float(args.dropout),
        attention_mode=str(args.attention),  # type: ignore[arg-type]
        tlb_eps=float(args.tlb_eps),
        sinkhorn_iters=int(args.sinkhorn_iters),
        post_attn_layer_norm=bool(args.post_attn_ln),
        pos_sin_buffer=pos_sin,
    ).to(device)


def build_argparser_script11() -> argparse.ArgumentParser:
    p = build_argparser()
    p.description = "Script11: TLB-Attention and ablations (script10 synthetic tasks)"
    p.add_argument(
        "--attention",
        type=str,
        default="tlb",
        choices=(
            "vanilla",
            "tlb",
            "frob",
            "l2",
            "sinkhorn",
            "nuc_inter",
            "kv_mul",
            "kv_inner",
            "kv_x_inter",
            "kv_inner_nuc_inter",
            "qv_inner",
            "randv_inner",
            "nuc_inter_q",
            "v_jitter",
        ),
        help="Attention variant (vanilla = standard softmax on QK; tlb/frob/l2/sinkhorn = script11 variants).",
    )
    p.add_argument("--tlb-eps", type=float, default=1e-8, help="Epsilon for TLB / Frobenius / L2 denominators.")
    p.add_argument("--sinkhorn-iters", type=int, default=5, help="Sinkhorn iterations for sinkhorn mode.")
    p.add_argument(
        "--post-attn-ln",
        action="store_true",
        help="Apply LayerNorm on attention output before residual (ablation).",
    )
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout on attention and FFN.")
    p.add_argument(
        "--probe-every",
        type=int,
        default=0,
        help="If >0, run mechanistic probe on val batch every N steps (writes mechanics.jsonl).",
    )
    p.set_defaults(output_dir=str(DEFAULT_OUTPUT))
    p.add_argument(
        "--run-self-test",
        action="store_true",
        help="Run tlb_attention_core numerical checks and exit.",
    )
    p.add_argument(
        "--sanity-overfit",
        action="store_true",
        help="Train tiny model on a single batch for --steps (default 80) to verify loss can go near zero.",
    )
    return p


def _sanity_overfit(args: argparse.Namespace, device: torch.device) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _args = argparse.Namespace(**vars(args))
    # Identity copy labels every non-pad token — CE can go ~0 on one batch.
    _args.task = "identity"
    _args.n_min = _args.n_max = 8
    _args.val_n = 8
    _args.vocab_content = 16
    _args.batch = 4
    _args.val_batch = 4
    _args.steps = min(200, int(getattr(args, "steps", 150) or 150))
    _args.log_every = 25
    _args.dim = 64
    _args.depth = 2
    _args.nhead = 4
    _args.lr = 3e-3
    layout = validate_and_task_layout(_args)
    m = build_flexible_model(_args, layout, device)
    opt = torch.optim.Adam(m.parameters(), lr=float(_args.lr))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 1)
    from script10_token_mirror import build_identity_copy_batch

    pad_id = layout["pad_id"]
    vocab_content = layout["vocab_content"]
    max_len = layout["max_len"]
    tokens, labels, pad_mask = build_identity_copy_batch(
        _args.batch,
        _args.n_min,
        _args.n_max,
        vocab_content,
        pad_id,
        max_len,
        device,
        generator=gen,
        fixed_n=8,
    )
    for step in range(_args.steps + 1):
        m.train()
        opt.zero_grad()
        logits = m(tokens, src_key_padding_mask=pad_mask)
        loss = masked_ce_loss(logits, labels)
        loss.backward()
        opt.step()
        if step % _args.log_every == 0 or step == _args.steps:
            print(f"[sanity-overfit] step {step} train_ce={loss.item():.6f}", flush=True)
    if loss.item() > 0.05:
        print("[sanity-overfit] WARNING: loss did not drop to near zero; check implementation.", flush=True)
    else:
        print("[sanity-overfit] OK", flush=True)


def main() -> None:
    p = build_argparser_script11()
    args = p.parse_args()
    if args.run_self_test:
        _run_numerical_self_tests()
        print("Self-test OK.")
        return
    if args.sanity_overfit:
        device = resolve_device(args.device)
        _sanity_overfit(args, device)
        return

    apply_script10_preset(args)
    apply_script10_train_recipe(args)
    if getattr(args, "baseline_only", False):
        print("[script11] note: --baseline-only is redundant (script11 trains one model only).", flush=True)

    layout = validate_and_task_layout(args)
    device = resolve_device(args.device)
    print(f"[script11] attention={args.attention} task={args.task} preset={args.preset or 'none'}", flush=True)

    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)
    vsp = Path(args.val_set_file)
    val_path = vsp if vsp.is_absolute() else root_out / vsp
    val_tokens, val_labels, val_pad_mask = ensure_fixed_val_set(
        args,
        val_path,
        layout["max_len"],
        layout["vocab_content"],
        layout["pad_id"],
        layout["delim_id"],
        device,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_flexible_model(args, layout, device)

    mechanics_path = root_out / f"mechanics_seed{args.seed}_{args.attention}.jsonl"
    mechanics_f = None
    if int(args.probe_every) > 0:
        mechanics_f = mechanics_path.open("w", encoding="utf-8")

    def after_step_probe(step: int, mod: nn.Module) -> None:
        if int(args.probe_every) <= 0 or mechanics_f is None:
            return
        if step % int(args.probe_every) != 0 and step != int(args.steps):
            return
        probe = probe_seq_model_flexible(mod, val_tokens, val_pad_mask)  # type: ignore[arg-type]
        probe["step"] = step
        probe["attention"] = args.attention
        mechanics_f.write(json.dumps(_jsonable_probe(probe)) + "\n")
        mechanics_f.flush()

    extra = {
        "attention": args.attention,
        "tlb_eps": float(args.tlb_eps),
        "sinkhorn_iters": int(args.sinkhorn_iters),
        "post_attn_ln": bool(args.post_attn_ln),
    }
    rows = train_run(
        args,
        root_out,
        model_name=f"tfmr_{args.attention}",
        model=model,
        device=device,
        vocab_content=layout["vocab_content"],
        pad_id=layout["pad_id"],
        delim_id=layout["delim_id"],
        max_len=layout["max_len"],
        val_tokens=val_tokens,
        val_labels=val_labels,
        val_pad_mask=val_pad_mask,
        legacy_transformer=_transformer_legacy_baseline(args),
        extra_row_fields=extra,
        after_step=after_step_probe if int(args.probe_every) > 0 else None,
    )
    if mechanics_f is not None:
        mechanics_f.close()
        print(f"Wrote {mechanics_path}", flush=True)

    plot_loss_and_acc(
        {f"tfmr_{args.attention}": rows}, root_out, figure_basename="figure_script11_loss_acc"
    )
    print(f"Wrote figure_script11_loss_acc.png under {root_out}", flush=True)

    save_csv(
        root_out / f"summary_seed{args.seed}_{args.attention}.csv",
        [
            {
                "seed": args.seed,
                "attention": args.attention,
                "final_val_loss": rows[-1]["val_loss"],
                "final_val_acc": rows[-1]["val_acc_all"],
            }
        ],
    )


if __name__ == "__main__":
    main()
