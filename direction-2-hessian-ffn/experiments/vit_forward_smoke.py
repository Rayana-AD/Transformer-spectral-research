#!/usr/bin/env python3
"""Phase A: random-tensor forward smoke for VisionTransformer (no dataset)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tlb.models.vit import (
    VISION_ALLOWED_HESSIAN_VARIANTS,
    VisionTransformer,
    estimate_seq_len,
    validate_vision_hessian_variant,
)
from tlb.training.utils import resolve_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "vit_forward_smoke" / "summary.json",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)

    # Reject causal variants
    blocked_ok = False
    try:
        validate_vision_hessian_variant("h4_phi768_prefix_ms")
    except ValueError:
        blocked_ok = True
    if not blocked_ok:
        raise RuntimeError("expected causal-prefix variant to be rejected")

    results: list[dict] = []
    arms = (
        ("standard", "h4_phi768"),
        ("hessian", "h4_phi768"),
        ("hessian", "h4_phi768_global_ms"),
        ("hessian", "h4_phi768_global_decouple"),
        ("hessian", "h4_phi768_global_light_decouple"),
        ("hessian", "h4_phi768_global_decouple_min"),
    )
    for ffn_type, hessian_variant in arms:
        model = VisionTransformer(
            img_size=args.img_size,
            patch_size=args.patch_size,
            num_classes=10,
            dim=128,
            depth=2,
            nhead=4,
            ffn_type=ffn_type,
            hessian_variant=hessian_variant,  # type: ignore[arg-type]
            attention_type="angular",
        ).to(device)
        x = torch.randn(
            args.batch_size, 3, args.img_size, args.img_size, device=device
        )
        logits = model(x)
        assert logits.shape == (args.batch_size, 10)
        results.append(
            {
                "ffn_type": ffn_type,
                "hessian_variant": hessian_variant if ffn_type == "hessian" else None,
                "logits_shape": list(logits.shape),
                "num_params": model.num_parameters(),
                "seq_len": estimate_seq_len(args.img_size, args.patch_size),
            }
        )
        print(
            f"ok ffn={ffn_type} logits={tuple(logits.shape)} "
            f"params={model.num_parameters()} seq_len={estimate_seq_len(args.img_size, args.patch_size)}"
        )

    summary = {
        "allowed_hessian": sorted(VISION_ALLOWED_HESSIAN_VARIANTS),
        "causal_prefix_rejected": blocked_ok,
        "arms": results,
    }
    save_json(args.out, summary)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
