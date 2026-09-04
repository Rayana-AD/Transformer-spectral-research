"""Shared value-path spatial/temporal smoothing for attention (Marian, Qwen3, probes).

Temporal modes
--------------
- ``circular_roll``: ``V.roll(+1, seq)`` — wraps last token to position 0 (legacy / encoder OK).
- ``causal_prev``: mix with previous token only; position 0 unchanged (decoder-safe).

Spatial modes
-------------
- ``spatial_global``: sequence-wide mean (legacy / encoder OK).
- ``prefix_mean``: prefix-only cumulative mean; position t uses mean(V[0:t+1]) (decoder-safe).
"""
from __future__ import annotations

from typing import Literal, Tuple

import torch

TemporalMode = Literal["circular_roll", "causal_prev"]
SpatialMode = Literal["spatial_global", "prefix_mean"]
CascadeOrder = Literal["spatial_first", "temporal_first"]


def temporal_neighbor(value: torch.Tensor, mode: TemporalMode) -> torch.Tensor:
    """Return neighbor tensor aligned with ``value`` along dim=-2 (seq)."""
    if mode == "circular_roll":
        return value.roll(shifts=1, dims=-2)
    if mode == "causal_prev":
        # V_prev[t] = V[t-1] for t>0; V_prev[0] = V[0] (self, no future leakage)
        return torch.cat([value[..., :1, :], value[..., :-1, :]], dim=-2)
    raise ValueError(f"unknown temporal mode: {mode!r}")


def spatial_reference(value: torch.Tensor, mode: SpatialMode) -> torch.Tensor:
    """Return reference tensor broadcastable with ``value`` for spatial mixing."""
    if mode == "spatial_global":
        return value.mean(dim=-2, keepdim=True)
    if mode == "prefix_mean":
        cs = value.cumsum(dim=-2)
        seq_len = value.shape[-2]
        t = torch.arange(1, seq_len + 1, device=value.device, dtype=value.dtype)
        shape = [1] * (value.dim() - 2) + [seq_len, 1]
        t = t.view(*shape)
        return cs / t
    raise ValueError(f"unknown spatial mode: {mode!r}")


def apply_spatial_smooth(
    value: torch.Tensor,
    alpha_s: float,
    *,
    spatial_mode: SpatialMode = "spatial_global",
) -> torch.Tensor:
    ref = spatial_reference(value, spatial_mode)
    return (1.0 - alpha_s) * value + alpha_s * ref


def apply_temporal_smooth(
    value: torch.Tensor,
    alpha_t: float,
    *,
    temporal_mode: TemporalMode = "circular_roll",
) -> torch.Tensor:
    neighbor = temporal_neighbor(value, temporal_mode)
    return (1.0 - alpha_t) * value + alpha_t * neighbor


def apply_value_cascade(
    value: torch.Tensor,
    alpha_spatial: float,
    alpha_temporal: float,
    *,
    order: CascadeOrder = "spatial_first",
    temporal_mode: TemporalMode = "circular_roll",
    spatial_mode: SpatialMode = "spatial_global",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (V_raw, V_first, V_second) cascade stages without detaching."""
    v_raw = value
    if order == "spatial_first":
        v_first = apply_spatial_smooth(
            v_raw, alpha_spatial, spatial_mode=spatial_mode
        )
        v_second = apply_temporal_smooth(
            v_first, alpha_temporal, temporal_mode=temporal_mode
        )
    else:
        v_first = apply_temporal_smooth(
            v_raw, alpha_temporal, temporal_mode=temporal_mode
        )
        v_second = apply_spatial_smooth(
            v_first, alpha_spatial, spatial_mode=spatial_mode
        )
    return v_raw, v_first, v_second


def decoder_safe_modes() -> Tuple[TemporalMode, SpatialMode]:
    return "causal_prev", "prefix_mean"


def encoder_legacy_modes() -> Tuple[TemporalMode, SpatialMode]:
    return "circular_roll", "spatial_global"


def _run_self_tests() -> None:
    torch.manual_seed(0)
    v = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1)  # positions 0..7

    circ = temporal_neighbor(v, "circular_roll")
    assert circ[0, 0, 0, 0].item() == 7.0, "circular roll leaks last token to pos 0"

    causal = temporal_neighbor(v, "causal_prev")
    assert causal[0, 0, 0, 0].item() == 0.0, "causal_prev pos0 should be self"
    assert causal[0, 0, 3, 0].item() == 2.0, "causal_prev pos3 should be prev token"

    glob = spatial_reference(v, "spatial_global")
    assert glob.shape[-2] == 1
    assert abs(glob.mean().item() - 3.5) < 1e-5

    pref = spatial_reference(v, "prefix_mean")
    assert abs(pref[0, 0, 0, 0].item() - 0.0) < 1e-5
    assert abs(pref[0, 0, 3, 0].item() - 1.5) < 1e-5  # mean(0,1,2,3)

    _, v1, v2 = apply_value_cascade(
        v, 0.1, 0.2, order="spatial_first", temporal_mode="causal_prev", spatial_mode="prefix_mean"
    )
    assert v1.shape == v.shape and v2.shape == v.shape
    print("tlb_value_smooth self-tests OK")


if __name__ == "__main__":
    _run_self_tests()
