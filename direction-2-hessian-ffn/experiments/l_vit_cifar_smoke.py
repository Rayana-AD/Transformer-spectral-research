#!/usr/bin/env python3
"""Phase B: CIFAR-10 TLB-ViT trainable smoke (standard vs h4_phi768)."""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from tlb.data.cifar10 import build_cifar10_loaders
from tlb.training.utils import (
    apply_lr_schedule,
    build_hessian_optimizer,
    load_yaml_config,
    resolve_device,
    save_json,
)
from tlb.training.vision_utils import (
    build_vit_from_cfg,
    evaluate_loader,
    parse_vision_arm,
)


def train_one(
    label: str,
    cfg: dict,
    *,
    device: torch.device,
    seed: int,
    synthetic: bool | None = None,
) -> dict:
    torch.manual_seed(seed)
    ffn_type, hessian_variant = parse_vision_arm(label)
    train_loader, val_loader, num_classes = build_cifar10_loaders(
        cfg, synthetic=synthetic
    )
    model = build_vit_from_cfg(
        cfg,
        num_classes=num_classes,
        ffn_type=ffn_type,
        hessian_variant=hessian_variant,
        device=device,
    )
    opt = build_hessian_optimizer(model, cfg, ffn_type=ffn_type)
    steps = int(cfg["steps"])
    lr_min_ratio = float(cfg.get("lr_min", 1e-5)) / float(cfg["lr"])
    log_every = int(cfg.get("log_every", 50))
    eval_every = int(cfg.get("eval_every", 50))

    history: list[dict] = []
    best = {"val_top1": -1.0, "step": 0}
    data_iter = itertools.cycle(train_loader)
    t0 = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        images, labels = next(data_iter)
        images = images.to(device)
        labels = labels.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        apply_lr_schedule(
            opt, step, cfg, total_steps=steps, lr_min_ratio=lr_min_ratio
        )

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = float((pred == labels).float().mean().item())
            row = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_top1": acc,
            }
            history.append(row)
            print(
                f"[{label}] step={step} loss={row['train_loss']:.4f} "
                f"acc={acc:.3f}"
            )

        if step % eval_every == 0 or step == steps:
            metrics = evaluate_loader(model, val_loader, device, num_classes)
            print(
                f"[{label}] val step={step} loss={metrics['val_loss']:.4f} "
                f"top1={metrics['val_top1']:.3f} macro_f1={metrics['val_macro_f1']:.3f}"
            )
            if metrics["val_top1"] > best["val_top1"]:
                best = {
                    "val_top1": metrics["val_top1"],
                    "val_macro_f1": metrics["val_macro_f1"],
                    "val_loss": metrics["val_loss"],
                    "step": step,
                }
            model.train()

    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "ffn_type": ffn_type,
        "hessian_variant": hessian_variant if ffn_type == "hessian" else None,
        "seed": seed,
        "num_params": model.num_parameters(),
        "steps": steps,
        "elapsed_sec": elapsed,
        "best": best,
        "history_tail": history[-5:],
        "dataset": "cifar10_synthetic"
        if (synthetic if synthetic is not None else bool(cfg.get("synthetic", False)))
        else "cifar10",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "vit_cifar_smoke.yaml",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["standard", "h4_phi768"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic offline images (overrides config).",
    )
    parser.add_argument(
        "--real-cifar",
        action="store_true",
        help="Force real torchvision CIFAR-10 download.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    device = resolve_device(args.device)
    seeds = [args.seed] if args.seed is not None else list(cfg.get("seeds", [0]))
    out_dir = Path(cfg.get("output_dir", "results/vit_cifar_smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_flag: bool | None = None
    if args.synthetic:
        synthetic_flag = True
    elif args.real_cifar:
        synthetic_flag = False

    runs: list[dict] = []
    for seed in seeds:
        for label in args.variants:
            result = train_one(
                label, cfg, device=device, seed=int(seed), synthetic=synthetic_flag
            )
            path = out_dir / f"{label}_seed{seed}.json"
            save_json(path, result)
            print(f"wrote {path}")
            runs.append(result)

    summary = {
        "config": str(args.config),
        "device": str(device),
        "synthetic": runs[0]["dataset"].endswith("synthetic") if runs else None,
        "runs": [
            {
                "label": r["label"],
                "seed": r["seed"],
                "best": r["best"],
                "num_params": r["num_params"],
                "elapsed_sec": r["elapsed_sec"],
                "dataset": r["dataset"],
            }
            for r in runs
        ],
    }
    save_json(out_dir / "summary.json", summary)
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
