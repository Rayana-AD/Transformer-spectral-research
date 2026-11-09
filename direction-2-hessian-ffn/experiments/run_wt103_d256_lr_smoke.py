#!/usr/bin/env python3
"""WT103 d=256 LR smoke — compare ctxsplit recipes vs Standard (~500 steps, <45 min MPS).

Use locally before scaling miniGPT / FineWeb d256 to pick a ctxsplit LR overlay.

Example:
  python experiments/run_wt103_d256_lr_smoke.py --device mps --data-dir data/wikitext103
  python experiments/run_wt103_d256_lr_smoke.py --device mps --steps 300 --recipes baseline_out16_ro025 out16_ro035
  python experiments/run_wt103_d256_lr_smoke.py --skip-standard  # ctxsplit grid only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.l4_causal_wt103 import train_one
from tlb.data.wikitext103 import WikiText103Corpus
from tlb.training.utils import load_yaml_config, resolve_device, save_json

CTX_VARIANT = "h4_phi768_prefix_ms_decouple_ctxsplit"


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def merge_cfg(base_path: Path, overlay_path: Path | None) -> dict[str, Any]:
    cfg = load_yaml_config(base_path)
    if overlay_path is not None and overlay_path.is_file():
        cfg.update(load_yaml_config(overlay_path))
    return cfg


def recipe_done(output_dir: Path, recipe_id: str) -> bool:
    path = output_dir / f"ctxsplit_{recipe_id}_seed0.json"
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return row.get("status") in ("done", "finished")


def run_arm(
    label: str,
    cfg: dict,
    *,
    corpus: WikiText103Corpus,
    device,
    seed: int,
    ffn_type: str,
    hessian_variant: str,
    output_dir: Path,
) -> dict[str, Any]:
    out = output_dir / f"{label}.json"
    if out.is_file():
        row = json.loads(out.read_text(encoding="utf-8"))
        if row.get("status") in ("done", "finished"):
            print(f"  skip {label} (done val_ppl={row.get('val_ppl')})", flush=True)
            return row

    print(f"  train {label} …", flush=True)
    metrics = train_one(
        label,
        corpus,
        cfg,
        ffn_type=ffn_type,
        hessian_variant=hessian_variant,  # type: ignore[arg-type]
        device=device,
        seed=seed,
    )
    save_json(out, metrics)
    print(
        f"    val_ppl={metrics.get('val_ppl')} test_ppl={metrics.get('test_ppl'):.1f}",
        flush=True,
    )
    return metrics


def print_summary(
    rows: list[dict[str, Any]], *, std_val: float | None, steps: int
) -> None:
    print(f"\n=== WT103 d256 LR smoke summary @{steps} ===", flush=True)
    print(f"{'recipe':<22} {'val_ppl':>8} {'test_ppl':>9} {'ratio':>7}  label", flush=True)
    print("-" * 72, flush=True)
    if std_val is not None and std_val > 0:
        print(f"{'standard':<22} {std_val:8.1f} {'—':>9} {'1.00×':>7}  standard_seed0", flush=True)

    ctx_rows = [r for r in rows if r.get("kind") == "ctxsplit"]
    ctx_rows.sort(key=lambda r: float(r.get("val_ppl") or 1e9))
    for row in ctx_rows:
        val = row.get("val_ppl")
        test = row.get("test_ppl")
        ratio = (std_val / val) if std_val and val and val > 0 else None
        ratio_s = f"{ratio:.3f}×" if ratio else "—"
        val_s = f"{val:.1f}" if val is not None else "—"
        test_s = f"{test:.1f}" if test is not None else "—"
        print(
            f"{row['recipe_id']:<22} {val_s:>8} {test_s:>9} {ratio_s:>7}  {row.get('label', '')}",
            flush=True,
        )

    if ctx_rows and std_val:
        best = ctx_rows[0]
        base = next((r for r in ctx_rows if r["recipe_id"] == "baseline_out16_ro025"), None)
        print(
            f"\nBest ctxsplit: {best['recipe_id']} val_ppl={best.get('val_ppl'):.1f} "
            f"ratio={std_val / best['val_ppl']:.3f}×",
            flush=True,
        )
        if base and best["recipe_id"] != base["recipe_id"]:
            print(
                f"  vs baseline {base.get('val_ppl'):.1f} → "
                f"{(base['val_ppl'] - best['val_ppl']):+.1f} val_ppl",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs" / "wt103_scale_smoke" / "manifest.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "wt103_d256_lr_smoke",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "wikitext103")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None, help="default: manifest seed")
    parser.add_argument(
        "--recipes",
        nargs="+",
        default=None,
        help="subset of recipe ids (default: all in manifest)",
    )
    parser.add_argument("--skip-standard", action="store_true")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override steps in all configs (default: manifest steps or yaml)",
    )
    args = parser.parse_args()

    spec = load_manifest(args.manifest)
    seed = int(args.seed if args.seed is not None else spec.get("seed", 0))
    smoke_steps = int(
        args.steps if args.steps is not None else spec.get("steps", 500)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Loading WikiText-103 from {args.data_dir} …", flush=True)
    corpus = WikiText103Corpus(
        args.data_dir,
        max_vocab=32_000,
        min_freq=2,
    )
    corpus.prepare()
    print(f"  vocab={corpus.vocab_size} train_tokens={corpus.train_ids.numel()}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    std_val: float | None = None

    def with_steps(cfg: dict) -> dict:
        out = dict(cfg)
        out["steps"] = smoke_steps
        return out

    if not args.skip_standard:
        std_cfg = with_steps(load_yaml_config(ROOT / spec["standard_config"]))
        std_row = run_arm(
            f"standard_seed{seed}",
            std_cfg,
            corpus=corpus,
            device=device,
            seed=seed,
            ffn_type="standard",
            hessian_variant="h1",
            output_dir=args.output_dir,
        )
        std_val = std_row.get("val_ppl")
        summary_rows.append({"kind": "standard", **std_row})

    recipes = spec["recipes"]
    if args.recipes:
        wanted = set(args.recipes)
        recipes = [r for r in recipes if r["id"] in wanted]
        missing = wanted - {r["id"] for r in recipes}
        if missing:
            raise SystemExit(f"Unknown recipe ids: {sorted(missing)}")

    ctx_base = ROOT / spec["ctxsplit_base"]
    for recipe in recipes:
        rid = recipe["id"]
        overlay = ROOT / recipe["overlay"] if recipe.get("overlay") else None
        cfg = with_steps(merge_cfg(ctx_base, overlay))
        label = f"ctxsplit_{rid}_seed{seed}"
        row = run_arm(
            label,
            cfg,
            corpus=corpus,
            device=device,
            seed=seed,
            ffn_type="hessian",
            hessian_variant=CTX_VARIANT,
            output_dir=args.output_dir,
        )
        summary_rows.append(
            {
                "kind": "ctxsplit",
                "recipe_id": rid,
                "label": recipe.get("label", rid),
                "overlay": str(overlay) if overlay else None,
                **row,
            }
        )

    manifest_out = {
        "runner": "run_wt103_d256_lr_smoke",
        "seed": seed,
        "steps": smoke_steps,
        "standard_val_ppl": std_val,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": summary_rows,
    }
    save_json(args.output_dir / "smoke_summary.json", manifest_out)
    print_summary(summary_rows, std_val=std_val, steps=smoke_steps)
    print(f"\n→ {args.output_dir / 'smoke_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
