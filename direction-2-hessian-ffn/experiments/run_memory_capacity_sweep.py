#!/usr/bin/env python3
"""L5 memory capacity curves: niah hay length, kv pairs, mk-niah needle count."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.l5_seq2seq_synthetic import VARIANT_SPECS, train_one
from tlb.training.utils import load_yaml_config, resolve_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "memory_capacity_sweep.yaml",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["standard", "hessian_split"],
        choices=list(VARIANT_SPECS.keys()),
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        default=["niah_hay", "kv_pairs", "mk_niah"],
        choices=["niah_hay", "kv_pairs", "mk_niah"],
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shorter sweeps for smoke (3 points each).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "memory_capacity_sweep",
    )
    args = parser.parse_args()

    base_cfg = load_yaml_config(args.config)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.fast:
        hay_sweep = [16, 48, 96]
        kv_sweep = [4, 10, 16]
        mk_sweep = [2, 4, 8]
    else:
        hay_sweep = [int(x) for x in base_cfg.get("niah_hay_sweep", [16, 32, 48, 64, 96])]
        kv_sweep = [int(x) for x in base_cfg.get("kv_pairs_sweep", [4, 8, 12, 16])]
        mk_sweep = [int(x) for x in base_cfg.get("mk_needles_sweep", [2, 4, 6, 8])]

    rows: list[dict] = []

    def run_probe(
        probe: str,
        task: str,
        sweep_key: str,
        sweep_val: int,
        variant: str,
    ) -> None:
        cfg = copy.deepcopy(base_cfg)
        if probe == "niah_hay":
            cfg["niah_hay_len"] = sweep_val
        elif probe == "kv_pairs":
            cfg["kv_num_pairs"] = sweep_val
        elif probe == "mk_niah":
            cfg["mk_num_needles"] = sweep_val

        ffn_type, enc_h, dec_h = VARIANT_SPECS[variant]
        label = f"{probe}_{sweep_key}{sweep_val}_{variant}_seed{args.seed}"
        print(f"[capacity] {label} …", flush=True)
        metrics = train_one(
            label,
            task,
            cfg,
            ffn_type=ffn_type,
            encoder_hessian_variant=enc_h,
            decoder_hessian_variant=dec_h,
            device=device,
            seed=args.seed,
        )
        row = {
            "probe": probe,
            "task": task,
            "variant": variant,
            "sweep_key": sweep_key,
            "sweep_value": sweep_val,
            "best_val_acc": metrics["best_val_acc"],
            "val_acc": metrics["val_acc"],
            "param_count": metrics["param_count"],
            "train_time_s": metrics["train_time_s"],
        }
        rows.append(row)
        save_json(args.output_dir / f"{label}.json", {**metrics, **row})
        print(
            f"  best_val_acc={row['best_val_acc']:.3f} time={row['train_time_s']:.0f}s",
            flush=True,
        )

    for variant in args.variants:
        if "niah_hay" in args.probes:
            for hay in hay_sweep:
                run_probe("niah_hay", "niah", "hay", hay, variant)
        if "kv_pairs" in args.probes:
            for pairs in kv_sweep:
                if pairs * 2 > int(base_cfg["vocab_content"]) // 2:
                    print(f"[skip] kv_pairs={pairs} exceeds vocab pool", flush=True)
                    continue
                run_probe("kv_pairs", "kv_assoc", "pairs", pairs, variant)
        if "mk_niah" in args.probes:
            for k in mk_sweep:
                run_probe("mk_niah", "mk_niah", "needles", k, variant)

    save_json(args.output_dir / "summary.json", rows)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"\n→ {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
