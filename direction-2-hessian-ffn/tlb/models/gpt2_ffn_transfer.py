"""Swap HuggingFace GPT-2 MLP blocks for TLB Hessian FFN (analytic warm-start)."""
from __future__ import annotations

from typing import Iterable, Literal

import torch
import torch.nn as nn

HybridMode = Literal["residual", "gated"]

from tlb.ffn.hessian_core import DecoupleMinLambda, HessianFFN, HessianVariant
from tlb.ffn.mlp_hessian_analytic import (
    apply_analytic_translation,
    promote_decouple_min_to_global_ms,
    promote_decouple_min_to_global_ms_decouple,
    translate_standard_ffn,
)
from tlb.ffn.standard import StandardFFN

# Pretrain MLP→Hessian swap must use oracle λ geometry (ViT T2g lesson).
PRETRAIN_SWAP_VARIANT: HessianVariant = "h4_phi768_global_decouple_min"
PRETRAIN_SWAP_LAMBDA: DecoupleMinLambda = "proj_oracle"

# Simplified ctxsplit bridges for GPT-2 multi-layer FT (not full tanh²(z@A) stack).
# Paper ctxsplit is installed by tlb.models.gpt2_ctxsplit_init, not this translator.
GPT2_CTXSPLIT_BRIDGE_VARIANTS = frozenset(
    {
        "h4_phi768_prefix_ms_decouple_ctxsplit_x_tanh",
        "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_agg",
        "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z",
        "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z_active_rank",
    }
)

GPT2_CTXSPLIT_WARM_INIT_VARIANTS = GPT2_CTXSPLIT_BRIDGE_VARIANTS


class GPT2HessianMLP(nn.Module):
    """Drop-in replacement for ``GPT2MLP`` / ``transformers GPT2MLP``."""

    def __init__(
        self,
        dim: int,
        *,
        hessian_variant: HessianVariant = PRETRAIN_SWAP_VARIANT,
        lambda_geometry: DecoupleMinLambda | None = None,
        num_units: int | None = None,
        calib_mode: str | None = None,
        uav_raw_context: bool | None = None,
        mix_init: float | None = None,
        tanh2_temperature: float | None = None,
        tanh2_amplitude: str | None = None,
    ) -> None:
        super().__init__()
        self.ffn = HessianFFN(
            dim,
            variant=hessian_variant,
            lambda_geometry=lambda_geometry,
            num_units=num_units,
            calib_mode=calib_mode,
            uav_raw_context=uav_raw_context,
            mix_init=mix_init,
            tanh2_temperature=tanh2_temperature,
            tanh2_amplitude=tanh2_amplitude,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ffn(hidden_states)


class GPT2HybridMLP(nn.Module):
    """Keep the native GPT-2 MLP and add a Hessian branch.

    ``residual``: ``y = MLP(x) + Hessian(x)`` with Hessian readout zeroed at install.
    ``gated``: ``y = (1-α) MLP(x) + α Hessian(x)`` with ``α = σ(alpha_logit)`` near 0.
    """

    def __init__(
        self,
        mlp: nn.Module,
        hessian: GPT2HessianMLP,
        *,
        mode: HybridMode = "residual",
        alpha_init: float = -6.0,
    ) -> None:
        super().__init__()
        if mode not in ("residual", "gated"):
            raise ValueError(f"hybrid mode must be residual or gated, got {mode!r}")
        self.mlp = mlp
        self.hessian = hessian
        self.mode = mode
        if mode == "gated":
            self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha_logit", torch.tensor(0.0))

    def gate_alpha(self) -> torch.Tensor:
        if self.mode != "gated":
            return self.alpha_logit.new_zeros(())
        return torch.sigmoid(self.alpha_logit)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        y_mlp = self.mlp(hidden_states)
        y_h = self.hessian(hidden_states)
        if self.mode == "residual":
            return y_mlp + y_h
        alpha = self.gate_alpha()
        return (1.0 - alpha) * y_mlp + alpha * y_h


def zero_hessian_readout(hessian_mlp: GPT2HessianMLP) -> None:
    """Make Hessian output identically 0 so a residual branch is function-preserving."""
    core = hessian_mlp.ffn.phi_core
    if core is None:
        raise TypeError("hessian mlp has no phi_core")
    with torch.no_grad():
        if hasattr(core, "out_weight"):
            core.out_weight.zero_()
        if hasattr(core, "out_bias") and isinstance(core.out_bias, torch.Tensor):
            core.out_bias.zero_()


def install_hybrid_mlp(
    model: nn.Module,
    layer_idx: int,
    *,
    mode: HybridMode,
    hessian_variant: HessianVariant = PRETRAIN_SWAP_VARIANT,
    lambda_geometry: DecoupleMinLambda | None = PRETRAIN_SWAP_LAMBDA,
    num_units: int | None = None,
    lambda_mode: str = "oracle_gelu",
    alpha_init: float = -6.0,
) -> dict[str, float | int | str]:
    """Wrap the native MLP at ``layer_idx`` with a residual or gated Hessian branch."""
    block = model.transformer.h[layer_idx]
    old_mlp = block.mlp
    if isinstance(old_mlp, (GPT2HessianMLP, GPT2HybridMLP)):
        raise TypeError(
            f"layer {layer_idx} already swapped ({type(old_mlp).__name__}); "
            "hybrid install expects the native GPT-2 MLP"
        )
    device = next(old_mlp.parameters()).device
    dim = int(model.config.n_embd)
    if num_units is None:
        raw_inner = getattr(model.config, "n_inner", None)
        num_units = int(raw_inner) if raw_inner is not None else 4 * dim
    hessian_mlp = GPT2HessianMLP(
        dim,
        hessian_variant=hessian_variant,
        lambda_geometry=lambda_geometry,
        num_units=num_units,
    ).to(device)
    meta = warm_init_hessian_from_gpt2_mlp(
        hessian_mlp,
        old_mlp,
        hessian_variant=str(hessian_variant),
        lambda_mode=lambda_mode,
    )
    if mode == "residual":
        zero_hessian_readout(hessian_mlp)
        meta["readout_zeroed"] = True
    hybrid = GPT2HybridMLP(
        old_mlp,
        hessian_mlp,
        mode=mode,
        alpha_init=alpha_init,
    ).to(device)
    block.mlp = hybrid
    meta["layer"] = layer_idx
    meta["hybrid_mode"] = mode
    meta["warm_init"] = True
    return meta


def freeze_hybrid_native_mlp(model: nn.Module, layer_indices: Iterable[int]) -> None:
    """Keep the original GPT-2 MLP frozen inside hybrid wrappers."""
    for idx in layer_indices:
        mlp = model.transformer.h[int(idx)].mlp
        if isinstance(mlp, GPT2HybridMLP):
            for param in mlp.mlp.parameters():
                param.requires_grad = False


def _gpt2_mlp_linears(mlp: nn.Module) -> tuple[nn.Module, nn.Module]:
    if hasattr(mlp, "c_fc") and hasattr(mlp, "c_proj"):
        return mlp.c_fc, mlp.c_proj
    raise TypeError(f"unsupported GPT-2 MLP layout: {type(mlp)!r}")


def _is_conv1d(layer: nn.Module) -> bool:
    return layer.__class__.__name__ == "Conv1D"


def gpt2_mlp_to_standard_ffn(mlp: nn.Module) -> StandardFFN:
    """Map GPT-2 ``c_fc`` / ``c_proj`` into ``StandardFFN`` (GELU MLP)."""
    fc1, fc2 = _gpt2_mlp_linears(mlp)
    if _is_conv1d(fc1):
        dim = int(fc1.weight.shape[0])
        hidden = int(fc1.weight.shape[1])
    else:
        dim = int(fc1.in_features)
        hidden = int(fc1.out_features)
    std = StandardFFN(dim, dim_feedforward=hidden, activation="gelu")
    with torch.no_grad():
        if _is_conv1d(fc1):
            std.linear1.weight.copy_(fc1.weight.T.detach().float())
            std.linear1.bias.copy_(fc1.bias.detach().float())
            std.linear2.weight.copy_(fc2.weight.T.detach().float())
            std.linear2.bias.copy_(fc2.bias.detach().float())
        else:
            std.linear1.weight.copy_(fc1.weight.detach().float())
            std.linear1.bias.copy_(fc1.bias.detach().float())
            std.linear2.weight.copy_(fc2.weight.detach().float())
            std.linear2.bias.copy_(fc2.bias.detach().float())
    return std


def _finalize_gpt2_hessian_warm_start(core: nn.Module, *, target_variant: str) -> None:
    """Post-translation hooks for GPT-2 clean-route variants."""
    if target_variant.endswith("_uav") and hasattr(core, "context_weight"):
        with torch.no_grad():
            core.context_weight.copy_(core.weight)
            if hasattr(core, "gate_bias") and hasattr(core, "content_bias"):
                core.gate_bias.copy_(core.content_bias)
    if target_variant.endswith("_2d"):
        # Ordered swap: U←W1 (translation), A←W1, b_A←b1; cross/tt gates ≈0 ⇒ step-0 ≈ oracle.
        with torch.no_grad():
            if hasattr(core, "context_weight") and hasattr(core, "weight"):
                core.context_weight.copy_(core.weight)
            if hasattr(core, "gate_bias") and hasattr(core, "bias"):
                core.gate_bias.copy_(core.bias)


def _promote_for_variant(tr, variant: str):
    if variant == "h4_phi768_global_ms":
        return promote_decouple_min_to_global_ms(tr, tie_mode="w1")
    if variant in (
        "h4_phi768_global_decouple_min",
        "h4_phi768_global_decouple_min_native_calib",
        "h4_phi768_global_decouple_min_uav",
        "h4_phi768_global_decouple_min_2d",
    ):
        return tr
    if variant == "h4_phi768_global_decouple":
        return promote_decouple_min_to_global_ms_decouple(tr)
    return tr


def warm_init_hessian_from_gpt2_mlp(
    hessian_mlp: GPT2HessianMLP,
    gpt2_mlp: nn.Module,
    *,
    hessian_variant: str | None = None,
    lambda_mode: str = "oracle_gelu",
) -> dict[str, float | int | str]:
    """Analytic translation GPT-2 MLP → Hessian FFN (U/V from W1/W2).

    Always translates via ``global_decouple_min`` + ``oracle_gelu``, then promotes
    tensors to the target variant.  For pretrain swap use
    ``global_decouple_min`` + ``proj_oracle_fast`` — **not** ``ctxsplit`` (tanh² λ).
    """
    target = hessian_variant or str(hessian_mlp.ffn.variant)
    if "ctxsplit" in target and target not in GPT2_CTXSPLIT_BRIDGE_VARIANTS:
        raise ValueError(
            f"Hessian variant {target!r} uses full tanh²(z@A) context λ and cannot "
            "match a pretrained GELU MLP at step-0. Use "
            f"{PRETRAIN_SWAP_VARIANT!r} + {PRETRAIN_SWAP_LAMBDA!r}, or a bridge in "
            f"{sorted(GPT2_CTXSPLIT_BRIDGE_VARIANTS)!r} for simplified ctxsplit FT."
        )

    std = gpt2_mlp_to_standard_ffn(gpt2_mlp)
    core = hessian_mlp.ffn.phi_core
    if core is None or not hasattr(core, "weight"):
        raise TypeError(f"variant {hessian_mlp.ffn.variant!r} has no phi_core.weight")
    num_units = int(core.weight.shape[0])
    tr = translate_standard_ffn(
        std,
        hessian_variant="h4_phi768_global_decouple_min",
        hessian_hidden=num_units,
        lambda_mode=lambda_mode,  # type: ignore[arg-type]
        joint_mode="w1_only",
    )
    tr = _promote_for_variant(tr, target)
    apply_analytic_translation(hessian_mlp.ffn, tr)
    if core is not None:
        _finalize_gpt2_hessian_warm_start(core, target_variant=target)
    if target in (
        "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z",
        "h4_phi768_prefix_ms_decouple_ctxsplit_gelu_z_active_rank",
    ):
        # Swap init: γ→1 (raw x content), A←0, λ-mix→1 so forward ≈ g @ V (MLP).
        core = hessian_mlp.ffn.phi_core
        if core is not None:
            with torch.no_grad():
                if hasattr(core, "hp_logit"):
                    core.hp_logit.fill_(10.0)
                if hasattr(core, "context_weight"):
                    core.context_weight.zero_()
                if hasattr(core, "ctx_lambda_mix_logit"):
                    core.ctx_lambda_mix_logit.fill_(-4.0)

    return {
        "num_units": num_units,
        "dim": int(core.weight.shape[1]),
        "translation_variant": tr.variant,
        "target_variant": target,
        "lambda_mode": lambda_mode,
        "lambda_geometry": getattr(core, "lambda_geometry", None),
    }


def swap_gpt2_mlp_with_hessian(
    model: nn.Module,
    layer_idx: int,
    *,
    hessian_variant: HessianVariant = PRETRAIN_SWAP_VARIANT,
    lambda_geometry: DecoupleMinLambda | None = PRETRAIN_SWAP_LAMBDA,
    num_units: int | None = None,
    lambda_mode: str = "oracle_gelu",
    warm_init: bool = True,
    calib_mode: str | None = None,
    uav_raw_context: bool | None = None,
    mix_init: float | None = None,
    tanh2_temperature: float | None = None,
    tanh2_amplitude: str | None = None,
) -> dict[str, float | int | str]:
    """Replace ``model.transformer.h[layer_idx].mlp`` with a Hessian FFN block."""
    block = model.transformer.h[layer_idx]
    old_mlp = block.mlp
    device = next(old_mlp.parameters()).device
    dim = int(model.config.n_embd)
    if num_units is None:
        raw_inner = getattr(model.config, "n_inner", None)
        num_units = int(raw_inner) if raw_inner is not None else 4 * dim
    hessian_mlp = GPT2HessianMLP(
        dim,
        hessian_variant=hessian_variant,
        lambda_geometry=lambda_geometry,
        num_units=num_units,
        calib_mode=calib_mode,
        uav_raw_context=uav_raw_context,
        mix_init=mix_init,
        tanh2_temperature=tanh2_temperature,
        tanh2_amplitude=tanh2_amplitude,
    ).to(device)
    if warm_init:
        meta = warm_init_hessian_from_gpt2_mlp(
            hessian_mlp,
            old_mlp,
            hessian_variant=str(hessian_variant),
            lambda_mode=lambda_mode,
        )
    else:
        meta = {
            "num_units": num_units,
            "dim": dim,
            "target_variant": str(hessian_variant),
            "warm_init": False,
            "init_mode": "cold",
            "lambda_mode": None,
        }
    block.mlp = hessian_mlp
    meta["layer"] = layer_idx
    meta["warm_init"] = warm_init
    meta["tanh2_temperature"] = float(tanh2_temperature if tanh2_temperature is not None else 1.0)
    meta["tanh2_amplitude"] = str(tanh2_amplitude or "paper")
    return meta


def tie_offaxis_shared_context(
    model: nn.Module,
    layer_indices: Iterable[int],
) -> nn.Parameter | None:
    """Tie one residual-space ``context_weight`` (A) across off-axis blocks.

    PyTorch registers the same Parameter reference on each block, while
    ``named_parameters`` de-duplicates it by identity.  This gives one optimizer
    update whose gradient is accumulated from every swapped layer.
    """
    shared: nn.Parameter | None = None
    for idx in layer_indices:
        block = model.transformer.h[int(idx)]
        ffn = getattr(block.mlp, "ffn", None)
        if ffn is None:
            continue
        core = getattr(ffn, "phi_core", None)
        if core is None or not hasattr(core, "context_weight"):
            continue
        ctx = core.context_weight
        if not isinstance(ctx, nn.Parameter):
            continue
        if shared is None:
            shared = ctx
            continue
        if ctx.shape != shared.shape:
            raise ValueError(
                "cannot share off-axis A across incompatible shapes: "
                f"{tuple(shared.shape)} vs {tuple(ctx.shape)} at layer {idx}"
            )
        if core.context_weight is shared:
            continue
        if "context_weight" in core._parameters:
            del core._parameters["context_weight"]
        core.context_weight = shared
    return shared


def swap_gpt2_mlps(
    model: nn.Module,
    layer_indices: Iterable[int],
    *,
    hessian_variant: HessianVariant = PRETRAIN_SWAP_VARIANT,
    lambda_geometry: DecoupleMinLambda | None = PRETRAIN_SWAP_LAMBDA,
    num_units: int | None = None,
    lambda_mode: str = "oracle_gelu",
    warm_init: bool = True,
    calib_mode: str | None = None,
    uav_raw_context: bool | None = None,
    mix_init: float | None = None,
    tanh2_temperature: float | None = None,
    tanh2_amplitude: str | None = None,
) -> list[dict[str, float | int | str]]:
    layers = [int(idx) for idx in layer_indices]
    metas = [
        swap_gpt2_mlp_with_hessian(
            model,
            idx,
            hessian_variant=hessian_variant,
            lambda_geometry=lambda_geometry,
            num_units=num_units,
            lambda_mode=lambda_mode,
            warm_init=warm_init,
            calib_mode=calib_mode,
            uav_raw_context=uav_raw_context,
            mix_init=mix_init,
            tanh2_temperature=tanh2_temperature,
            tanh2_amplitude=tanh2_amplitude,
        )
        for idx in layers
    ]
    if "offaxis" in str(hessian_variant) and len(layers) > 1:
        shared = tie_offaxis_shared_context(model, layers)
        if shared is not None:
            for row in metas:
                row["shared_context_weight"] = True
                row["shared_context_numel"] = int(shared.numel())
    return metas


def freeze_all_but_layers(
    model: nn.Module,
    layer_indices: Iterable[int],
    *,
    train_ln: bool = True,
    train_mlp_only: bool = True,
) -> None:
    """Freeze everything except selected transformer blocks (MLP or full block)."""
    for param in model.parameters():
        param.requires_grad = False
    layers = set(layer_indices)
    for idx, block in enumerate(model.transformer.h):
        if idx not in layers:
            continue
        if train_mlp_only:
            for param in block.mlp.parameters():
                param.requires_grad = True
            mix = getattr(block, "ffn_mix_logit", None)
            if isinstance(mix, nn.Parameter):
                mix.requires_grad = True
            if train_ln:
                for param in block.ln_2.parameters():
                    param.requires_grad = True
        else:
            for param in block.parameters():
                param.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


_V_CORE_ATTRS = ("out_weight", "out_bias", "out_unit_bias")


def set_hessian_v_requires_grad(
    model: nn.Module,
    layer_indices: Iterable[int],
    *,
    train_v: bool,
) -> int:
    """Freeze or unfreeze the V leg (``out_weight`` / ``out_bias``) on Hessian MLP blocks.

    U (``weight``) and A (``context_weight`` when present) stay untouched — used for
    multi-layer FT where V trains only after the UA leg (forward order).
    Returns the number of V parameters toggled.
    """
    toggled = 0
    for idx in layer_indices:
        block = model.transformer.h[idx]
        ffn = getattr(block.mlp, "ffn", None)
        if ffn is None:
            continue
        core = getattr(ffn, "phi_core", None)
        if core is None:
            continue
        for attr in _V_CORE_ATTRS:
            param = getattr(core, attr, None)
            if isinstance(param, nn.Parameter):
                param.requires_grad = train_v
                toggled += int(param.numel())
    return toggled


def hessian_v_trainable(model: nn.Module, layer_indices: Iterable[int]) -> bool:
    """True if any V parameter on the listed layers requires grad."""
    for idx in layer_indices:
        block = model.transformer.h[idx]
        ffn = getattr(block.mlp, "ffn", None)
        if ffn is None:
            continue
        core = getattr(ffn, "phi_core", None)
        if core is None:
            continue
        for attr in _V_CORE_ATTRS:
            param = getattr(core, attr, None)
            if isinstance(param, nn.Parameter) and param.requires_grad:
                return True
    return False


def mlp_step0_mse(
    old_mlp: nn.Module,
    new_mlp: nn.Module,
    *,
    dim: int,
    device: torch.device,
    batch: int = 4,
    seq: int = 64,
) -> float:
    """Diagnostic: MSE between original GPT-2 MLP and swapped Hessian @ init."""
    with torch.no_grad():
        x = torch.randn(batch, seq, dim, device=device)
        y_old = old_mlp(x)
        y_new = new_mlp(x)
        return float(((y_old - y_new) ** 2).mean().item())
