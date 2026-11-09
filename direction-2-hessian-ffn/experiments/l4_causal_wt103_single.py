#!/usr/bin/env python3
"""Single-arm WT103 causal LM for Colab miniGPT preset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.l4_causal_wt103 import train_one
from tlb.data.wikitext103 import WikiText103Corpus
from tlb.ffn.hessian_core import HessianVariant
from tlb.training.utils import load_yaml_config, resolve_device, save_json

ARM_SPECS: dict[str, tuple[str, HessianVariant]] = {
    "standard": ("standard", "h1"),
    "ctxsplit": ("hessian", "h4_phi768_prefix_ms_decouple_ctxsplit"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=list(ARM_SPECS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "wikitext103",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    ffn_type, hess_var = ARM_SPECS[args.arm]
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label = f"{args.arm}_seed{args.seed}"
    print(f"Loading WikiText-103 from {args.data_dir} …", flush=True)
    corpus = WikiText103Corpus(
        args.data_dir,
        max_vocab=int(cfg.get("max_vocab", 32_000)),
        min_freq=int(cfg.get("min_freq", 2)),
    )
    corpus.prepare()
    print(
        f"  vocab={corpus.vocab_size} train_tokens={corpus.train_ids.numel()}",
        flush=True,
    )

    metrics = train_one(
        label,
        corpus,
        cfg,
        ffn_type=ffn_type,
        hessian_variant=hess_var,
        device=device,
        seed=args.seed,
    )
    out = args.output_dir / f"{label}.json"
    save_json(out, metrics)
    print(
        f"→ {out} val_ppl={metrics.get('val_ppl')} "
        f"test_ppl={metrics['test_ppl']:.2f} "
        f"tokens={metrics.get('tokens_seen', 0) / 1e6:.1f}M",
        flush=True,
    )


if __name__ == "__main__":
    main()
