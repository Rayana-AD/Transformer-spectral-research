"""Shared activation modules for Script3 (depth scattering) and Script4 (training).

- Smish: x * sigmoid(x) (same family as Script3's former diffuse_act).
- LightDepth: x / (1 + exp(-x / t)), temperature t > 0.
- ScatteringOpticalDepth: y = z * exp(-τ(z)), with τ ≥ 0 from z (Beer–Lambert style attenuation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Smish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class LightDepthAct(nn.Module):
    """Optical-depth style gate: x / (1 + exp(-x/t)).

    Note: when t == 1, this equals x * sigmoid(x) (same as Smish). Use t != 1 to probe temperature.
    """

    def __init__(self, t: float) -> None:
        super().__init__()
        if t <= 0:
            raise ValueError("LightDepthAct requires t > 0")
        self.t = float(t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1.0 + torch.exp(-x / self.t))


class ScatteringOpticalDepthAct(nn.Module):
    """Beer–Lambert style: activation(z) = z · exp(-τ(z)), τ from local scattering / optical depth.

    z is the pre-activation tensor (e.g. Wx+b from the linear layer). τ must be ≥0 so exp(-τ) ∈ (0, 1].

    Modes (tau_from):
      - "softplus_z": τ = softplus(β z) — signed channel; negative z → small τ, weak damping.
      - "softplus_z2": τ = softplus(β z²) — scattering strength scales with local |field|².
    """

    def __init__(self, beta: float = 1.0, tau_from: str = "softplus_z") -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("ScatteringOpticalDepthAct requires beta > 0")
        self.beta = float(beta)
        if tau_from not in ("softplus_z", "softplus_z2"):
            raise ValueError('tau_from must be "softplus_z" or "softplus_z2"')
        self.tau_from = tau_from

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.tau_from == "softplus_z":
            tau = F.softplus(self.beta * z)
        else:
            tau = F.softplus(self.beta * (z * z))
        return z * torch.exp(-tau)


def build_activation(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "mish":
        return nn.Mish()
    if name == "silu":
        return nn.SiLU()
    if name == "smish":
        return Smish()
    if name == "light_depth":
        t = float(kwargs.get("t", 1.0))
        return LightDepthAct(t)
    if name in ("scattering_tau", "optical_depth", "phys_tau"):
        beta = float(kwargs.get("beta", 1.0))
        tau_from = str(kwargs.get("tau_from", "softplus_z"))
        return ScatteringOpticalDepthAct(beta=beta, tau_from=tau_from)
    raise ValueError(f"Unknown activation: {name}")
