"""ImageNet-pretrained torchvision ViT with optional Hessian FFN swap."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tlb.ffn.hessian_core import DecoupleMinLambda, HessianVariant
from tlb.models.vit_ffn_transfer import (
    HessianMLPBlock,
    WarmInitStrategy,
    set_torchvision_trainable,
    swap_torchvision_mlps_to_hessian,
    warm_init_hessian_from_torchvision_mlp,
)

# Back-compat alias
_HessianMLP = HessianMLPBlock


def build_torchvision_vit_b16(
    *,
    num_classes: int = 10,
    ffn_type: str = "standard",
    hessian_variant: HessianVariant = "h4_phi768",
    pretrained: bool = True,
    freeze_encoder: bool = False,
    hessian_warm_init: WarmInitStrategy = "none",
    hessian_lambda_geometry: DecoupleMinLambda | None = None,
    hessian_num_units: int | None = None,
) -> nn.Module:
    """ViT-B/16; ``ffn_type=hessian`` replaces each block MLP with bidirectional Hessian."""
    try:
        from torchvision.models import ViT_B_16_Weights, vit_b_16
    except ImportError as exc:
        raise ImportError("torchvision required for pretrained ViT-B/16") from exc

    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    if pretrained:
        print("TLB: loading ImageNet ViT-B/16 weights...", flush=True)
    model = vit_b_16(weights=weights)
    if pretrained:
        print("TLB: ViT-B/16 weights loaded", flush=True)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    nn.init.trunc_normal_(model.heads.head.weight, std=0.02)
    nn.init.zeros_(model.heads.head.bias)

    warm_diag: list[dict[str, Any]] = []
    if ffn_type == "hessian":
        warm_diag = swap_torchvision_mlps_to_hessian(
            model,
            hessian_variant=hessian_variant,
            warm_init=hessian_warm_init,
            lambda_geometry=hessian_lambda_geometry,
            num_units=hessian_num_units,
        )
    elif ffn_type != "standard":
        raise ValueError(f"unknown ffn_type for torchvision vit: {ffn_type!r}")

    set_torchvision_trainable(
        model,
        freeze_encoder=freeze_encoder,
        train_ffn=True,
        train_head=True,
    )
    model._tlb_warm_init_diag = warm_diag  # noqa: SLF001 — experiment logging
    return model


def load_vit_checkpoint(model: nn.Module, path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        meta = {k: v for k, v in payload.items() if k != "state_dict"}
    else:
        state = payload
        meta = {}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        **meta,
    }


def save_vit_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if state_dict is None:
        if model is None:
            raise ValueError("save_vit_checkpoint requires model or state_dict")
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "state_dict": state_dict,
        **(meta or {}),
    }
    torch.save(payload, out)


def staged_swap_mlps_from_checkpoint(
    model: nn.Module,
    *,
    hessian_variant: HessianVariant,
    warm_init: WarmInitStrategy = "fc1_svd",
) -> list[dict[str, Any]]:
    """Swap MLP→Hessian on an already-loaded model (e.g. after standard FT checkpoint)."""
    diagnostics: list[dict[str, Any]] = []
    dim = int(model.hidden_dim) if hasattr(model, "hidden_dim") else 768
    for i, layer in enumerate(model.encoder.layers):
        old_mlp = layer.mlp
        new_mlp = HessianMLPBlock(dim, hessian_variant=hessian_variant)
        if warm_init != "none" and hasattr(old_mlp, "__getitem__"):
            try:
                diag = warm_init_hessian_from_torchvision_mlp(
                    new_mlp, old_mlp, strategy=warm_init
                )
            except TypeError:
                diag = {"strategy": "none", "rows_from_fc1": 0, "rows_from_svd": 0}
        else:
            diag = {"strategy": "none", "rows_from_fc1": 0, "rows_from_svd": 0}
        diag["layer"] = i
        diagnostics.append(diag)
        layer.mlp = new_mlp
    model._tlb_warm_init_diag = diagnostics  # noqa: SLF001
    return diagnostics


def count_trainable(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def count_trainable_ffn(model: nn.Module) -> int:
    """Trainable params in encoder MLP / Hessian FFN blocks only."""
    return sum(
        int(p.numel())
        for name, p in model.named_parameters()
        if p.requires_grad and ".mlp." in name
    )
