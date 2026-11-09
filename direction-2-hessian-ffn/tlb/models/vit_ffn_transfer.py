"""Transfer torchvision ViT MLP weights into bidirectional Hessian FFN blocks."""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from tlb.ffn.hessian_core import DecoupleMinLambda, HessianFFN, HessianVariant
from tlb.ffn.mlp_hessian_analytic import (
    apply_analytic_translation,
    promote_decouple_min_to_global_ms,
    promote_decouple_min_to_global_ms_decouple,
    translate_standard_ffn,
)
from tlb.ffn.morse_prune import MorsePruneConfig, apply_hessian_morse_pruning
from tlb.ffn.standard import StandardFFN
from tlb.models.vit import validate_vision_hessian_variant

WarmInitStrategy = Literal[
    "fc1_copy",
    "fc1_svd",
    "analytic_decouple_min",
    "analytic_ms_decouple",
    "analytic_global_ms",
    "none",
]


class HessianMLPBlock(nn.Module):
    """Drop-in for torchvision MLPBlock: (B, N, D) -> (B, N, D)."""

    def __init__(
        self,
        dim: int,
        *,
        hessian_variant: HessianVariant = "h4_phi768",
        lambda_geometry: DecoupleMinLambda | None = None,
        num_units: int | None = None,
    ) -> None:
        super().__init__()
        validate_vision_hessian_variant(str(hessian_variant))
        self.ffn = HessianFFN(
            dim,
            variant=hessian_variant,
            lambda_geometry=lambda_geometry,
            num_units=num_units,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


def _torchvision_mlp_linears(mlp: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    fc1 = mlp[0]
    fc2 = mlp[3]
    if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
        raise TypeError(f"unexpected torchvision MLP layout: {mlp}")
    return fc1, fc2


def _torchvision_mlp_to_standard_ffn(fc1: nn.Linear, fc2: nn.Linear) -> StandardFFN:
    dim = int(fc1.in_features)
    hidden = int(fc1.out_features)
    mlp = StandardFFN(dim, dim_feedforward=hidden)
    with torch.no_grad():
        mlp.linear1.weight.copy_(fc1.weight.detach().float())
        mlp.linear1.bias.copy_(fc1.bias.detach().float())
        mlp.linear2.weight.copy_(fc2.weight.detach().float())
        mlp.linear2.bias.copy_(fc2.bias.detach().float())
    return mlp


def _promote_for_variant(tr, variant: str):
    if variant == "h4_phi768_global_ms":
        return promote_decouple_min_to_global_ms(tr, tie_mode="w1")
    if variant in ("h4_phi768_global_decouple", "h4_phi768_global_decouple_min"):
        if variant == "h4_phi768_global_decouple_min":
            return tr
        return promote_decouple_min_to_global_ms_decouple(tr)
    return tr


def warm_init_hessian_analytic_from_torchvision_mlp(
    hessian_mlp: HessianMLPBlock,
    tv_mlp: nn.Module,
    *,
    strategy: WarmInitStrategy,
) -> dict[str, float | int | str]:
    """Translate torchvision MLP weights into Hessian phi_core via analytic map."""
    fc1, fc2 = _torchvision_mlp_linears(tv_mlp)
    core = hessian_mlp.ffn.phi_core
    if not hasattr(core, "weight") or not hasattr(core, "bias"):
        raise TypeError(
            f"Hessian variant {hessian_mlp.ffn.variant!r} has no phi_core.weight/bias "
            "for analytic warm init"
        )

    variant = str(hessian_mlp.ffn.variant)
    num_units = int(core.weight.shape[0])
    std_mlp = _torchvision_mlp_to_standard_ffn(fc1, fc2)
    tr = translate_standard_ffn(
        std_mlp,
        hessian_variant="h4_phi768_global_decouple_min",
        hessian_hidden=num_units,
        lambda_mode="oracle_gelu",
        joint_mode="w1_only",
    )
    tr = _promote_for_variant(tr, variant)
    apply_analytic_translation(hessian_mlp.ffn, tr)

    rows = min(num_units, int(fc1.out_features))
    return {
        "strategy": strategy,
        "rows_from_fc1": int(rows),
        "rows_from_svd": 0,
        "num_units": num_units,
        "mlp_hidden": int(fc1.out_features),
        "dim": int(core.weight.shape[1]),
        "translation_variant": tr.variant,
    }


def warm_init_hessian_from_torchvision_mlp(
    hessian_mlp: HessianMLPBlock,
    tv_mlp: nn.Module,
    *,
    strategy: WarmInitStrategy = "fc1_svd",
) -> dict[str, float | int | str]:
    """Initialize ``hessian_mlp.ffn.phi_core.weight`` from torchvision ``MLPBlock``.

    Returns a small diagnostics dict for logging / analysis.
    """
    if strategy == "none":
        return {"strategy": "none", "rows_from_fc1": 0, "rows_from_svd": 0}

    if strategy in ("analytic_decouple_min", "analytic_ms_decouple", "analytic_global_ms"):
        expected = {
            "analytic_decouple_min": "h4_phi768_global_decouple_min",
            "analytic_ms_decouple": "h4_phi768_global_decouple",
            "analytic_global_ms": "h4_phi768_global_ms",
        }[strategy]
        if str(hessian_mlp.ffn.variant) != expected:
            raise ValueError(
                f"strategy {strategy!r} requires hessian_variant {expected!r}, "
                f"got {hessian_mlp.ffn.variant!r}"
            )
        return warm_init_hessian_analytic_from_torchvision_mlp(
            hessian_mlp, tv_mlp, strategy=strategy
        )

    fc1, fc2 = _torchvision_mlp_linears(tv_mlp)
    core = hessian_mlp.ffn.phi_core
    if not hasattr(core, "weight") or not hasattr(core, "bias"):
        raise TypeError(
            f"Hessian variant {hessian_mlp.ffn.variant!r} has no phi_core.weight/bias "
            "for MLP warm init"
        )

    w = core.weight
    b = core.bias
    dim = int(w.shape[1])
    num_units = int(w.shape[0])

    w1 = fc1.weight.detach().float()  # (H, D)
    b1 = fc1.bias.detach().float()
    w2 = fc2.weight.detach().float()  # (D, H)

    rows_fc1 = min(num_units, w1.shape[0])
    with torch.no_grad():
        w[:rows_fc1].copy_(w1[:rows_fc1])
        if b.shape[0] >= rows_fc1:
            b[:rows_fc1].copy_(b1[:rows_fc1])
        if hasattr(core, "out_weight"):
            ow = core.out_weight
            for j in range(rows_fc1):
                ow[j].copy_(w2[:, j])

        rows_svd = 0
        if strategy == "fc1_svd" and num_units > rows_fc1:
            # Use composition W2 @ W1 to fill remaining spectral rows.
            composed = w2 @ w1  # (D, D)
            u, _, _ = torch.linalg.svd(composed, full_matrices=False)
            need = num_units - rows_fc1
            take = min(need, u.shape[0])
            w[rows_fc1 : rows_fc1 + take].copy_(u[:take])
            rows_svd = int(take)
            if rows_fc1 + take < num_units:
                tail = num_units - rows_fc1 - take
                init_std = 0.02 * math.sqrt(6 * dim / num_units)
                w[rows_fc1 + take :].normal_(0.0, init_std)

        # Bias-neutral start for λ path when available.
        if hasattr(core, "hp_logit"):
            core.hp_logit.fill_(2.0)  # sigmoid≈0.88 → proj_in≈x
        if hasattr(core, "scale_logits"):
            core.scale_logits.fill_(-1.0)
            if hasattr(core, "window_sizes") and core.window_sizes:
                core.scale_logits[-1] = 2.0  # favor global scale

    return {
        "strategy": strategy,
        "rows_from_fc1": int(rows_fc1),
        "rows_from_svd": int(rows_svd),
        "num_units": num_units,
        "mlp_hidden": int(w1.shape[0]),
        "dim": dim,
    }


def swap_torchvision_mlps_to_hessian(
    model: nn.Module,
    *,
    hessian_variant: HessianVariant,
    warm_init: WarmInitStrategy = "fc1_svd",
    lambda_geometry: DecoupleMinLambda | None = None,
    num_units: int | None = None,
) -> list[dict[str, float | int | str]]:
    """Replace every encoder ``MLPBlock`` with ``_HessianMLP`` (+ optional warm init)."""
    diagnostics: list[dict[str, float | int | str]] = []
    layers = model.encoder.layers
    dim = int(model.hidden_dim) if hasattr(model, "hidden_dim") else 768
    n_layers = len(layers)
    lg_note = f" λ={lambda_geometry}" if lambda_geometry else ""
    m_note = f" m={num_units}" if num_units is not None else ""
    print(
        f"TLB: swapping {n_layers} MLP blocks → {hessian_variant}{lg_note}{m_note} "
        f"(warm={warm_init})...",
        flush=True,
    )
    for i, layer in enumerate(layers):
        old_mlp = layer.mlp
        new_mlp = HessianMLPBlock(
            dim,
            hessian_variant=hessian_variant,
            lambda_geometry=lambda_geometry,
            num_units=num_units,
        )
        if warm_init != "none":
            diag = warm_init_hessian_from_torchvision_mlp(
                new_mlp, old_mlp, strategy=warm_init
            )
        else:
            diag = {"strategy": "none", "rows_from_fc1": 0, "rows_from_svd": 0}
        diag["layer"] = i
        diagnostics.append(diag)
        layer.mlp = new_mlp
        print(f"TLB:   layer {i + 1}/{n_layers} warm-init done", flush=True)
    return diagnostics


def apply_hessian_morse_prune(
    model: nn.Module,
    *,
    strategy: str = "mlp_aligned",
    keep_units: int | None = None,
    mlp_width_mult: float = 4.0,
) -> dict[str, object]:
    """Physically prune low-persistence Hessian units (optional param control).

    Run after warm init on the default m=6d graph; does not affect configs that
    omit ``hessian_morse_prune``.
    """
    cfg = MorsePruneConfig(
        strategy=strategy,  # type: ignore[arg-type]
        keep_units=keep_units,
        mlp_width_mult=mlp_width_mult,
    )
    return apply_hessian_morse_pruning(model, cfg)


def freeze_hessian_spectral_tail(model: nn.Module, tail_start_row: int) -> dict[str, int]:
    """Stop training on unit rows ``[tail_start_row:]`` via grad hooks (spectral tail).

    Keeps m=6d graph but freezes expansion units beyond MLP width (4d).
    """
    start = int(tail_start_row)
    frozen_units = 0
    n_layers = 0

    def _tail_zero_grad_hook(row_start: int):
        def hook(grad: torch.Tensor) -> torch.Tensor | None:
            if grad is None:
                return None
            g = grad.clone()
            g[row_start:].zero_()
            return g

        return hook

    for layer in model.encoder.layers:
        mlp = layer.mlp
        if not hasattr(mlp, "ffn"):
            continue
        core = mlp.ffn.phi_core
        if not hasattr(core, "weight"):
            continue
        m = int(core.weight.shape[0])
        if start >= m:
            continue
        for attr in ("weight", "bias", "out_weight"):
            if not hasattr(core, attr):
                continue
            p = getattr(core, attr)
            if isinstance(p, torch.nn.Parameter):
                p.register_hook(_tail_zero_grad_hook(start))
        frozen_units = m - start
        n_layers += 1
    return {
        "tail_start_row": start,
        "frozen_units_per_layer": frozen_units,
        "layers": n_layers,
        "method": "grad_hook",
    }


def set_torchvision_trainable(
    model: nn.Module,
    *,
    freeze_encoder: bool,
    train_ffn: bool = True,
    train_head: bool = True,
) -> None:
    """Toggle requires_grad for torchvision ViT (head / encoder / Hessian FLP paths)."""
    for name, param in model.named_parameters():
        if name.startswith("heads."):
            param.requires_grad = train_head
        elif ".mlp." in name:
            param.requires_grad = train_ffn
        else:
            param.requires_grad = not freeze_encoder
