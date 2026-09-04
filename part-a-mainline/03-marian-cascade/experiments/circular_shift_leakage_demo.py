#!/usr/bin/env python3
"""Toy demonstration: circular roll vs causal_prev prefix sensitivity on value tensors."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from tlb_value_smooth import apply_value_cascade


def pos0_sensitivity_last_token(
    temporal_mode: str,
    spatial_mode: str,
    alpha_s: float = 0.1,
    alpha_t: float = 0.2,
) -> float:
    """Change only last token; measure ||V_smooth[0] - V_smooth[0]'||."""
    v_a = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1)
    v_b = v_a.clone()
    v_b[..., -1, :] = 999.0
    _, _, s_a = apply_value_cascade(
        v_a, alpha_s, alpha_t,
        order="spatial_first",
        temporal_mode=temporal_mode,  # type: ignore[arg-type]
        spatial_mode=spatial_mode,  # type: ignore[arg-type]
    )
    _, _, s_b = apply_value_cascade(
        v_b, alpha_s, alpha_t,
        order="spatial_first",
        temporal_mode=temporal_mode,  # type: ignore[arg-type]
        spatial_mode=spatial_mode,  # type: ignore[arg-type]
    )
    return float((s_a[..., 0, :] - s_b[..., 0, :]).abs().max().item())


def main() -> None:
    rows = [
        ("circular_roll + spatial_global", "circular_roll", "spatial_global"),
        ("causal_prev + prefix_mean", "causal_prev", "prefix_mean"),
        ("causal_prev + spatial_global", "causal_prev", "spatial_global"),
        ("circular_roll + prefix_mean", "circular_roll", "prefix_mean"),
    ]
    print("pos0 sensitivity to last-token change (max abs diff at position 0):")
    for name, tm, sm in rows:
        d = pos0_sensitivity_last_token(tm, sm)
        print(f"  {name}: {d:.6f}")


if __name__ == "__main__":
    main()
