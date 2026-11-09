"""GPT-2 paper-ctxsplit warm-start: same protocol as ``global_decouple_min``.

Only **U/V are warm-inited** (``U←W1``, ``V←W2``, plus their biases).
``s = U x + b`` is that translation (micro ``U`` + bias, not an ``s`` fit).
``λ = ψ''(s)`` is computed natively from that ``s`` — not translated, not
``z @ A``, not GELU/s.  A is the third UAV channel, written from U.

* ``token_psi_align``: canonical UV warm-start + native UAV A.
* ``uv_w12``: UV copy only; leaves the pre-copy scratch A (legacy).
* ``lam_align`` / ``x_align``: legacy MSE paths.
"""
from __future__ import annotations

import inspect
import math
from typing import Any, Iterable, Literal, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from tlb.ffn.hessian_core import HessianFFN, HessianVariant, _tanh2_second_deriv

ResidualMode = Literal["add", "none", "gated"]
InitMode = Literal[
    "token_psi_align",
    "mod_native_swap",
    "psi_align",
    "x_lam_align",
    "lam_align",
    "uv_rank",
    "uv_w12",
    "gpt2_warm",
    "x_align",
    "gpt2_z_qkv",
    "scratch_tie",
]
RankScore = Literal["lambda_abs", "gelu_abs"]
ASplitMode = Literal[
    "copy_u",
    "joint_wv",
    "w2_col",
    "row_shuffle",
    "column_block_cyclic",
    "block_diagonal",
]
CTXSPLIT_VARIANT: HessianVariant = "h4_phi768_prefix_ms_decouple_ctxsplit"
MOD_NATIVE_GELU_Z_VARIANT: HessianVariant = (
    "h4_phi768_prefix_ms_decouple_ctxsplit_mod_native_gelu_z"
)


def cross_w1_w2_for_a(
    w1: torch.Tensor,
    w2: torch.Tensor,
    b1: torch.Tensor,
    mode: ASplitMode | str = "joint_wv",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``A/b_A`` from the per-unit ``W2(W1)`` cross term while ``U←W1`` unchanged.

    Standard MLP unit ``j``: ``σ(w_j^T x + b_{1,j}) · v_j`` with ``v_j = W2[:,j]``.
    ``A`` rows inherit that cross geometry (not ``A←U``):

    - ``joint_wv``: ``a_j ∝ w_j + v_j`` (analytic joint unit, same as ``joint_mode=sum``)
    - ``w2_col``: ``a_j = v_j`` scaled to ``||w_j||`` (contraction leg as gate direction)
    """
    split = str(mode).strip().lower()
    if w1.ndim != 2 or w2.ndim != 2 or b1.ndim != 1:
        raise ValueError(
            f"expected W1 (H,D), W2 (D,H), b1 (H,), got "
            f"{tuple(w1.shape)}, {tuple(w2.shape)}, {tuple(b1.shape)}"
        )
    hidden, dim = int(w1.shape[0]), int(w1.shape[1])
    if int(w2.shape[0]) != dim or int(w2.shape[1]) != hidden:
        raise ValueError(f"W2 layout {tuple(w2.shape)} != {(dim, hidden)}")
    if split == "copy_u":
        return w1.clone(), b1.clone()
    eps = 1e-8
    a_w = torch.empty_like(w1)
    for j in range(hidden):
        w_in = w1[j].float()
        v_j = w2[:, j].float()
        n_in = float(w_in.norm().item())
        n_out = float(v_j.norm().item())
        if split == "w2_col":
            if n_out < eps:
                a_w[j] = w_in
            else:
                a_w[j] = v_j * (n_in / (n_out + eps))
        elif split == "joint_wv":
            w_joint = w_in + v_j
            if float(w_joint.norm().item()) < eps:
                w_joint = w_in if n_in > eps else v_j
            w_joint = w_joint / (float(w_joint.norm().item()) + eps)
            w_joint = w_joint * math.sqrt(max(n_in, eps) * max(n_out, eps))
            a_w[j] = w_joint
        else:
            raise ValueError(f"cross_w1_w2_for_a mode {mode!r} is not a cross init")
    return a_w.to(device=w1.device, dtype=w1.dtype), b1.clone()


def split_w1_for_a(
    w1: torch.Tensor,
    b1: torch.Tensor,
    mode: ASplitMode | str = "joint_wv",
    *,
    w2: torch.Tensor | None = None,
    n_blocks: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``A/b_A`` while ``U`` keeps full ``W1`` (swap on ``s=Ux+b``).

    Cross modes (``joint_wv``, ``w2_col``) need ``W2`` and encode the MLP contraction
    leg per unit.  Step-0 oracle uses ``λ=Φ(s)`` with ``mix=0``; ``A`` is independent
    from ``U`` but does not affect the swapped output until FT opens the gate path.
    """
    split = str(mode).strip().lower()
    if split in ("joint_wv", "w2_col"):
        if w2 is None:
            raise ValueError(f"a_split={split!r} requires W2")
        return cross_w1_w2_for_a(w1, w2, b1, split)
    if split == "copy_u":
        return w1.clone(), b1.clone()
    if w1.ndim != 2 or b1.ndim != 1 or int(w1.shape[0]) != int(b1.shape[0]):
        raise ValueError(
            f"expected W1 (H,D) and b1 (H,), got {tuple(w1.shape)} and {tuple(b1.shape)}"
        )
    hidden, dim = int(w1.shape[0]), int(w1.shape[1])
    gen = torch.Generator(device=w1.device)
    gen.manual_seed(int(seed))
    if split == "row_shuffle":
        perm = torch.randperm(hidden, generator=gen, device=w1.device)
        return w1.index_select(0, perm).contiguous(), b1.index_select(0, perm).contiguous()
    n_blk = max(1, min(int(n_blocks), dim))
    block = dim // n_blk
    if split == "column_block_cyclic":
        # ``A = W1 P`` with ``P`` a block-cyclic column permutation (``x̃`` subspace mix).
        order = [(b + 1) % n_blk for b in range(n_blk)]
        a_w = torch.empty_like(w1)
        dst = 0
        for src_blk in order:
            c0 = src_blk * block
            c1 = dim if src_blk == n_blk - 1 else (src_blk + 1) * block
            width = c1 - c0
            a_w[:, dst : dst + width] = w1[:, c0:c1]
            dst += width
        return a_w, b1.clone()
    if split == "block_diagonal":
        # ``A = W1 R`` where ``R`` block-permutes input coordinates (paper ``a_j^T x̃``).
        perm = torch.randperm(n_blk, generator=gen, device=w1.device)
        r = torch.zeros(dim, dim, device=w1.device, dtype=w1.dtype)
        for dst_blk, src_blk in enumerate(perm.tolist()):
            r0 = int(src_blk) * block
            r1 = dim if int(src_blk) == n_blk - 1 else (int(src_blk) + 1) * block
            d0 = dst_blk * block
            d1 = dim if dst_blk == n_blk - 1 else (dst_blk + 1) * block
            r[r0:r1, d0:d1] = torch.eye(r1 - r0, d1 - d0, device=w1.device, dtype=w1.dtype)
        return (w1 @ r).contiguous(), b1.clone()
    raise ValueError(f"unknown a_split mode {mode!r}")


def _linear_weight(layer: nn.Module, in_features: int) -> torch.Tensor:
    """``(out, in)`` regardless of HuggingFace Conv1D vs Linear."""
    w = layer.weight.detach()
    if w.ndim != 2:
        raise TypeError(f"expected 2D weight, got {tuple(w.shape)}")
    if int(w.shape[0]) == int(in_features):
        return w.float().T.contiguous()
    return w.float().contiguous()


def gpt2_attn_qkv_o(
    attn: nn.Module, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear-layout Q, K, V, out and their biases from GPT-2 attention."""
    if not hasattr(attn, "c_attn") or not hasattr(attn, "c_proj"):
        raise TypeError(f"expected GPT-2 attention with c_attn/c_proj, got {type(attn)!r}")
    qkv = _linear_weight(attn.c_attn, dim)
    if int(qkv.shape[0]) < 3 * dim:
        raise ValueError(f"c_attn out dim {qkv.shape[0]} < 3d={3 * dim}")
    q, k, v = qkv[:dim], qkv[dim : 2 * dim], qkv[2 * dim : 3 * dim]
    qkv_b = attn.c_attn.bias.detach().float()
    bq, bk, bv = qkv_b[:dim], qkv_b[dim : 2 * dim], qkv_b[2 * dim : 3 * dim]
    o_w = _linear_weight(attn.c_proj, dim)
    o_b = attn.c_proj.bias.detach().float()
    return q, k, v, o_w, bq, bk, bv, o_b


def gpt2_attn_qk(attn: nn.Module, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, _v, _o, _bq, _bk, _bv, _ob = gpt2_attn_qkv_o(attn, dim)
    return q, k


class GPT2CtxSplitMLP(nn.Module):
    """U/A/V ctxsplit as the FFN map.  Bypasses HessianFFN GELU + Linear readout."""

    def __init__(
        self,
        dim: int,
        *,
        hessian_variant: HessianVariant = CTXSPLIT_VARIANT,
        num_units: int | None = None,
        lambda_geometry: str | None = None,
        mix_init: float = -10.0,
        hessian_factor_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.ffn = HessianFFN(
            dim,
            variant=hessian_variant,
            num_units=num_units,
            lambda_geometry=lambda_geometry,  # type: ignore[arg-type]
            mix_init=mix_init,
            hessian_factor_temperature=hessian_factor_temperature,
        )
        if self.ffn.phi_core is None:
            raise TypeError(f"{hessian_variant!r} has no phi_core")
        if not hasattr(self.ffn.phi_core, "context_weight"):
            raise TypeError(f"{hessian_variant!r} is not ctxsplit (no A / context_weight)")
        # Unused shell: paper ctxsplit already reads out through V.
        self.ffn.readout = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ffn.phi_core(hidden_states)


def scratch_tie_ctxsplit(mlp: GPT2CtxSplitMLP) -> dict[str, Any]:
    """Paper scratch init: A ← U, V ← U.  λ stays ψ''(z @ A).

    No W1/W2 translation.  Same tie as WT103 from-scratch ctxsplit so step-0
    matches the symmetric operator; U/A/V then train as separate channels.
    """
    core = mlp.ffn.phi_core
    with torch.no_grad():
        core.context_weight.copy_(core.weight)
        core.out_weight.copy_(core.weight)
    return {
        "init": "scratch_tie",
        "target_variant": str(mlp.ffn.variant),
        "num_units": int(core.weight.shape[0]),
        "dim": int(core.weight.shape[1]),
        "a_tied_to_u": True,
        "v_tied_to_u": True,
        "lambda_geometry": "tanh2_second_deriv",
        "oracle_gelu": False,
        "readout": None,
    }


def _copy_linear(dest: nn.Linear, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    dest.weight.copy_(weight.to(device=dest.weight.device, dtype=dest.weight.dtype))
    if dest.bias is not None and bias is not None:
        dest.bias.copy_(bias.to(device=dest.bias.device, dtype=dest.bias.dtype))


def gpt2_mlp_w1_w2(mlp: nn.Module, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear-layout W1 ``(H, D)`` and W2 ``(D, H)`` from GPT-2 ``c_fc`` / ``c_proj``."""
    if not hasattr(mlp, "c_fc") or not hasattr(mlp, "c_proj"):
        raise TypeError(f"expected GPT-2 MLP with c_fc/c_proj, got {type(mlp)!r}")
    w1 = _linear_weight(mlp.c_fc, dim)
    hidden = int(w1.shape[0])
    w2 = _linear_weight(mlp.c_proj, hidden)
    if int(w2.shape[0]) != dim or int(w2.shape[1]) != hidden:
        raise ValueError(f"W2 layout {tuple(w2.shape)} != {(dim, hidden)}")
    return w1, w2


def gpt2_mlp_b1_b2(mlp: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """GPT-2 MLP expansion/output biases ``b1`` and ``b2``."""
    if not hasattr(mlp, "c_fc") or not hasattr(mlp, "c_proj"):
        raise TypeError(f"expected GPT-2 MLP with c_fc/c_proj, got {type(mlp)!r}")
    if mlp.c_fc.bias is None or mlp.c_proj.bias is None:
        raise TypeError("GPT-2 MLP c_fc/c_proj must both have bias")
    return mlp.c_fc.bias.detach().float(), mlp.c_proj.bias.detach().float()


def warm_ctxsplit_uav_from_s(
    mlp: GPT2CtxSplitMLP,
    old_mlp: nn.Module,
    *,
    a_split: ASplitMode | str = "joint_wv",
    a_split_blocks: int = 4,
    a_split_seed: int = 0,
) -> dict[str, Any]:
    """``global_decouple_min``-style UV warm-start, then native ctxsplit UAV.

    ``U/b_content ← W1/b1`` (exact swap content).  ``V/b_out ← W2/b2``.
    ``A/b_A ← cross(W1,W2)`` per unit (default ``joint_wv``), **not** ``A←U``.
    Step-0 ``λ=Φ(s)`` with ``mix=0`` is unchanged by ``A``; FT uses the cross gate.
    """
    core = mlp.ffn.phi_core
    dim = int(core.weight.shape[1])
    w1, w2 = gpt2_mlp_w1_w2(old_mlp, dim)
    b1, b2 = gpt2_mlp_b1_b2(old_mlp)
    rows = min(int(core.weight.shape[0]), int(w1.shape[0]))
    device = core.weight.device
    dtype = core.weight.dtype
    w1_rows = w1[:rows].to(device=device, dtype=dtype)
    w2_cols = w2.to(device=device, dtype=dtype)
    b1_rows = b1[:rows].to(device=device, dtype=dtype)
    a_w, a_b = split_w1_for_a(
        w1_rows.float(),
        b1_rows.float(),
        a_split,
        w2=w2_cols.float(),
        n_blocks=a_split_blocks,
        seed=a_split_seed,
    )
    with torch.no_grad():
        core.weight[:rows].copy_(w1_rows)
        core.content_bias[:rows].copy_(b1_rows)
        core.context_weight[:rows].copy_(a_w.to(device=device, dtype=dtype))
        core.bias[:rows].copy_(a_b.to(device=device, dtype=dtype))
        core.out_weight[:rows].copy_(w2.T[:rows].to(device=device, dtype=dtype))
        core.out_bias.copy_(b2.to(device=device, dtype=dtype))
        core.hp_logit.fill_(10.0)
    core.lambda_from_content_s = True
    core.phi_init_content_gate = True
    split_name = str(a_split).strip().lower()
    return {
        "w1_content_copy": True,
        "w2_copy": True,
        "a_from_s": split_name == "copy_u",
        "a_cross_w1_w2": split_name in ("joint_wv", "w2_col"),
        "a_split": split_name,
        "a_split_blocks": int(a_split_blocks),
        "a_split_seed": int(a_split_seed),
        "lambda_from_content_s": True,
        "phi_init_content_gate": True,
        "rows_written": rows,
    }


def clear_phi_init_content_gate(
    model: nn.Module,
    layer_indices: Iterable[int],
    *,
    keep_lambda_on_content_s: bool = False,
) -> None:
    """End step-0 swap mode: ``proj_x`` opens to ctxsplit; λ may stay on content ``s``."""
    for idx in layer_indices:
        block = model.transformer.h[int(idx)]
        core = block.mlp.ffn.phi_core
        if core is None:
            continue
        if hasattr(core, "phi_init_content_gate"):
            core.phi_init_content_gate = False
        if hasattr(core, "lambda_from_content_s") and not keep_lambda_on_content_s:
            core.lambda_from_content_s = False


_UV_ONLY_TRAINABLE = frozenset({"weight", "content_bias", "out_weight", "out_bias"})


def set_mod_native_uv_only_trainable(
    model: nn.Module, layer_indices: Iterable[int]
) -> int:
    """Train only U/V (and their biases); freeze A, mix, hp, and prefix-ms z params."""
    frozen = 0
    for idx in layer_indices:
        core = model.transformer.h[int(idx)].mlp.ffn.phi_core
        if core is None:
            continue
        core.mod_native_uv_only = True
        for name, param in core.named_parameters():
            train = name in _UV_ONLY_TRAINABLE
            param.requires_grad_(train)
            if not train:
                frozen += int(param.numel())
    return frozen


def begin_mod_native_finetune(
    model: nn.Module,
    layer_indices: Iterable[int],
    *,
    hp_logit: float = 2.0,
    mix_logit: float = 0.0,
    uv_only: bool = False,
) -> dict[str, float]:
    """After step-0 oracle swap: open ctxsplit + native mix; U/A/V train independently.

    Step-0 keeps ``hp_logit≈10`` (``proj_in≈x``) and ``mix=0`` for exact ``Φ(s)`` swap.
    FT lowers ``hp_logit`` and sets ``native_mix_logit`` (default ``0`` → ``mix=0.5``).
    ``A`` is already the ``W2(W1)`` cross init (``joint_wv`` / ``w2_col``), not ``A←U``.
    """
    clear_phi_init_content_gate(
        model, layer_indices, keep_lambda_on_content_s=True
    )
    stats: dict[str, float | bool | int] = {
        "hp_logit": float(hp_logit),
        "mix_logit": float(mix_logit),
        "uv_only": bool(uv_only),
    }
    if uv_only:
        for idx in layer_indices:
            core = model.transformer.h[int(idx)].mlp.ffn.phi_core
            if core is None:
                continue
            with torch.no_grad():
                if hasattr(core, "native_mix_logit"):
                    core.native_mix_logit.fill_(float(mix_logit))
        stats["frozen_non_uv_params"] = set_mod_native_uv_only_trainable(
            model, layer_indices
        )
        return stats
    for idx in layer_indices:
        core = model.transformer.h[int(idx)].mlp.ffn.phi_core
        if core is None:
            continue
        with torch.no_grad():
            if hasattr(core, "hp_logit"):
                core.hp_logit.fill_(float(hp_logit))
            if hasattr(core, "native_mix_logit"):
                core.native_mix_logit.fill_(float(mix_logit))
    return stats


def warm_ctxsplit_mod_native_from_gpt2(
    mlp: GPT2CtxSplitMLP,
    old_mlp: nn.Module,
    *,
    lambda_geometry: str,
    a_split: ASplitMode | str = "joint_wv",
    a_split_blocks: int = 4,
    a_split_seed: int = 0,
) -> dict[str, Any]:
    """UV oracle swap on ``s``; ``A`` from ``W2(W1)`` cross init (default ``joint_wv``)."""
    meta = warm_ctxsplit_uav_from_s(
        mlp,
        old_mlp,
        a_split=a_split,
        a_split_blocks=a_split_blocks,
        a_split_seed=a_split_seed,
    )
    core = mlp.ffn.phi_core
    core.lambda_geometry = lambda_geometry  # type: ignore[assignment]
    meta["lambda_geometry"] = lambda_geometry
    meta["mod_native_ua_split"] = True
    meta["oracle_gelu"] = lambda_geometry in (
        "proj_oracle_fast",
        "proj_oracle",
        "proj_oracle_mod_native_simple",
        "proj_oracle_mod_native_ortho",
    )
    meta["mod_native"] = True
    if hasattr(core, "mod_native_gelu_z_gate"):
        meta["mod_native_gelu_z_gate"] = bool(core.mod_native_gelu_z_gate)
    return meta


def align_ctxsplit_mod_native_to_mlp(
    mlp: GPT2CtxSplitMLP,
    old_mlp: nn.Module,
    *,
    lambda_geometry: str,
    a_split: ASplitMode | str = "joint_wv",
    a_split_blocks: int = 4,
    a_split_seed: int = 0,
) -> dict[str, Any]:
    """Step-0 oracle mod-native swap; ``A`` from ``W2(W1)`` cross (``joint_wv`` / ``w2_col``)."""
    meta = warm_ctxsplit_mod_native_from_gpt2(
        mlp,
        old_mlp,
        lambda_geometry=lambda_geometry,
        a_split=a_split,
        a_split_blocks=a_split_blocks,
        a_split_seed=a_split_seed,
    )
    meta.update(
        {
            "init": "gpt2_mod_native_swap",
            "warm_target": "uv",
            "native_uav": True,
            "a_written": True,
            "a_tied_to_u": bool(meta.get("a_from_s")),
            "v_tied_to_u": False,
        }
    )
    return meta


def warm_ctxsplit_uv_from_gpt2_mlp(mlp: GPT2CtxSplitMLP, old_mlp: nn.Module) -> dict[str, Any]:
    """Copy U/b1←W1/b1 then V/b2←W2/b2.  A / z / λ are not written.

    Training still runs paper ctxsplit: ``λ = ψ''(z @ A)``.  The A-leg
    ``bias`` stays independent from the inherited content bias.
    """
    core = mlp.ffn.phi_core
    dim = int(core.weight.shape[1])
    a_before = core.context_weight.detach().clone()
    w1, w2 = gpt2_mlp_w1_w2(old_mlp, dim)
    b1, b2 = gpt2_mlp_b1_b2(old_mlp)
    rows = min(int(core.weight.shape[0]), int(w1.shape[0]))
    device = core.weight.device
    dtype = core.weight.dtype
    with torch.no_grad():
        core.weight[:rows].copy_(w1[:rows].to(device=device, dtype=dtype))
        core.out_weight[:rows].copy_(w2.T[:rows].to(device=device, dtype=dtype))
        core.content_bias[:rows].copy_(b1[:rows].to(device=device, dtype=dtype))
        core.out_bias.copy_(b2.to(device=device, dtype=dtype))
    if not torch.equal(core.context_weight.detach(), a_before):
        raise RuntimeError("uv_w12 wrote A; only U then V may be copied")
    return {
        "init": "gpt2_uv_w12",
        "warm_target": "uv",
        "target_variant": str(mlp.ffn.variant),
        "num_units": int(core.weight.shape[0]),
        "dim": dim,
        "a_tied_to_u": False,
        "v_tied_to_u": False,
        "a_written": False,
        "z_full_qkv": False,
        "lambda_geometry": "tanh2_second_deriv",
        "oracle_gelu": False,
        "readout": None,
    }


def gpt2_mlp_preact(
    old_mlp: nn.Module, x: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """s = x @ W1^T + b1 on real tokens. Returns s, W1, W2."""
    if x.ndim != 3:
        raise ValueError(f"rank x must be (B, T, D), got {tuple(x.shape)}")
    w1, w2 = gpt2_mlp_w1_w2(old_mlp, dim)
    b1, _b2 = gpt2_mlp_b1_b2(old_mlp)
    b1 = b1.to(device=x.device, dtype=x.dtype)
    w1 = w1.to(device=x.device, dtype=x.dtype)
    s = x @ w1.transpose(-2, -1) + b1
    return s, w1, w2


def mlp_token_unit_scores(
    s: torch.Tensor, old_mlp: nn.Module, *, score: RankScore
) -> torch.Tensor:
    """Per-token, per-unit activity on the MLP pre-activation ``(B, T, H)``."""
    act = getattr(old_mlp, "act", None)
    gelu_s = act(s) if callable(act) else F.gelu(s)
    if score == "gelu_abs":
        return gelu_s.abs()
    eps = 1e-6
    lam = torch.where(s.abs() < eps, torch.full_like(gelu_s, 0.5), gelu_s / s)
    return lam.abs()


def mlp_unit_scores(s: torch.Tensor, old_mlp: nn.Module, *, score: RankScore) -> torch.Tensor:
    """Per-unit activity, mean over batch and time."""
    token = mlp_token_unit_scores(s, old_mlp, score=score)
    return token.mean(dim=tuple(range(token.ndim - 1)))


def select_active_unit_indices(scores: torch.Tensor, keep_frac: float) -> torch.Tensor:
    """Top-``keep_frac`` units by score, indices in original MLP order."""
    frac = float(keep_frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac!r}")
    hidden = int(scores.numel())
    n_keep = max(1, min(hidden, int(round(frac * hidden))))
    _vals, idx = torch.topk(scores.reshape(-1), n_keep, largest=True, sorted=False)
    return idx.sort().values


def ctxsplit_init_needs_x(init: str) -> bool:
    """Init modes that fit on the swapped layer's own ``x = LN2(h')``."""
    return init in (
        "x_align",
        "lam_align",
        "psi_align",
        "x_lam_align",
    )


def gpt2_mlp_forward(old_mlp: nn.Module, x: torch.Tensor, dim: int) -> torch.Tensor:
    """MLP increment from ``c_fc`` / ``c_proj`` layout (no ``Module.forward`` required)."""
    s, _w1, w2 = gpt2_mlp_preact(old_mlp, x, dim)
    act = getattr(old_mlp, "act", None)
    hidden = act(s) if callable(act) else F.gelu(s)
    _b1, b2 = gpt2_mlp_b1_b2(old_mlp)
    return hidden @ w2.transpose(-2, -1) + b2.to(device=x.device, dtype=x.dtype)


def _frozen_mlp_teacher_targets(
    old_mlp: nn.Module,
    x: torch.Tensor,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level teacher field ``GELU(s)`` and MLP increment (no weight copy)."""
    was_training = old_mlp.training
    old_mlp.eval()
    try:
        field_tgt, _lam_oracle = mlp_active_field(old_mlp, x, dim)
        with torch.no_grad():
            if hasattr(old_mlp, "c_fc"):
                output_tgt = gpt2_mlp_forward(old_mlp, x, dim).detach()
            else:
                output_tgt = old_mlp(x).detach()
        return field_tgt.detach(), output_tgt
    finally:
        if was_training:
            old_mlp.train()


def _token_dim_mse(pred: torch.Tensor, tgt: torch.Tensor, *, label: str) -> torch.Tensor:
    """MSE in a matched ``(B, T, F)`` space — no cross-rank pooling.

    Mean over feature ``F`` at each ``(b, t)``, then mean over tokens.  Does
    **not** collapse ``H`` and ``D`` into one undifferentiated global mean.
    """
    if pred.shape != tgt.shape:
        raise ValueError(
            f"{label} dim mismatch: pred {tuple(pred.shape)} vs tgt {tuple(tgt.shape)}"
        )
    if pred.ndim != 3:
        raise ValueError(f"{label} expects (B, T, F), got {tuple(pred.shape)}")
    return (pred - tgt).pow(2).mean(dim=-1).mean()


def _native_psi_token_losses(
    mlp: GPT2CtxSplitMLP,
    x: torch.Tensor,
    field_tgt: torch.Tensor,
    output_tgt: torch.Tensor,
    lam_tgt: torch.Tensor,
    *,
    output_weight: float = 1.0,
    lambda_weight: float | None = None,
    field_weight: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MLP teacher → native ``ψ''(s_z)`` with per-token, dim-matched terms.

    ``x[b,t]`` drives ``z[b,t]`` through native prefix-ms aggregate (required).
    Targets stay in matched spaces only:

      * ``output`` — ``(B, T, D)`` MLP increment (primary φ-init signal)
      * ``λ``, ``field`` — ``(B, T, H)`` gate / hidden field

    H-space terms are scaled by ``D/H`` when combined with output so ``H=4D``
    does not dominate the batch loss.  Init moves **A + λ-bias** only; U/V/z
    remain native scratch for ctxsplit FT.
    """
    core = mlp.ffn.phi_core
    lam, proj_x, _s_z = core.native_factors(x)
    field_pred = lam * proj_x
    output_pred = field_pred @ core.out_weight + core.out_bias
    dim = int(output_pred.shape[-1])
    num_units = int(lam.shape[-1])
    h_to_d = float(dim) / float(num_units)

    lambda_loss = _token_dim_mse(lam, lam_tgt, label="lambda")
    field_loss = _token_dim_mse(field_pred, field_tgt, label="field")
    output_loss = _token_dim_mse(output_pred, output_tgt, label="output")

    lam_w = float(lambda_weight) if lambda_weight is not None else h_to_d
    field_w = float(field_weight) if field_weight is not None else 0.25 * h_to_d
    total = (
        float(output_weight) * output_loss
        + lam_w * lambda_loss
        + field_w * field_loss
    )
    return lambda_loss, field_loss, output_loss, total


def _run_native_psi_adam(
    mlp: GPT2CtxSplitMLP,
    x: torch.Tensor,
    field_tgt: torch.Tensor,
    output_tgt: torch.Tensor,
    lam_tgt: torch.Tensor,
    trainable: tuple[nn.Parameter, ...],
    *,
    steps: int,
    lr: float,
    output_weight: float = 1.0,
    lambda_weight: float | None = None,
    field_weight: float | None = None,
) -> tuple[float, float, float, float]:
    """Adam on ``trainable``; returns (lambda_mse, field_mse, output_mse, total)."""
    for param in mlp.parameters():
        param.requires_grad_(False)
    for param in trainable:
        param.requires_grad_(True)
    opt = torch.optim.Adam(list(trainable), lr=lr)
    lambda_mse = field_mse = output_mse = total_mse = 0.0
    for _ in range(max(0, int(steps))):
        opt.zero_grad()
        lambda_loss, field_loss, output_loss, loss = _native_psi_token_losses(
            mlp,
            x,
            field_tgt,
            output_tgt,
            lam_tgt,
            output_weight=output_weight,
            lambda_weight=lambda_weight,
            field_weight=field_weight,
        )
        loss.backward()
        opt.step()
        lambda_mse = float(lambda_loss.detach().item())
        field_mse = float(field_loss.detach().item())
        output_mse = float(output_loss.detach().item())
        total_mse = float(loss.detach().item())
    opt.zero_grad()
    for param in mlp.parameters():
        param.requires_grad_(True)
    return lambda_mse, field_mse, output_mse, total_mse


def mlp_lambda_gelu(old_mlp: nn.Module, x: torch.Tensor, dim: int) -> torch.Tensor:
    """Teacher gate ``λ = GELU(s)/s`` on this layer's real tokens.  Signed."""
    s, _w1, _w2 = gpt2_mlp_preact(old_mlp, x, dim)
    act = getattr(old_mlp, "act", None)
    gelu_s = act(s) if callable(act) else F.gelu(s)
    eps = 1e-6
    return torch.where(s.abs() < eps, torch.full_like(gelu_s, 0.5), gelu_s / s)


def mlp_active_field(
    old_mlp: nn.Module, x: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher hidden field ``GELU(s)`` and its gate ``GELU(s)/s``."""
    s, _w1, _w2 = gpt2_mlp_preact(old_mlp, x, dim)
    act = getattr(old_mlp, "act", None)
    gelu_s = act(s) if callable(act) else F.gelu(s)
    eps = 1e-6
    lam = torch.where(s.abs() < eps, torch.full_like(gelu_s, 0.5), gelu_s / s)
    return gelu_s, lam


def ctxsplit_init_needs_frozen_track(init: str) -> bool:
    """Active-rank init watches frozen MLPs, not the layer being replaced."""
    return init in ("uv_rank", "gpt2_warm")


def write_ctxsplit_u_then_tie(
    mlp: GPT2CtxSplitMLP,
    u_rows: torch.Tensor,
    *,
    harvest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write tracked-active U, then standard ctxsplit ``A,V ← U``.  No W2."""
    core = mlp.ffn.phi_core
    if tuple(u_rows.shape) != tuple(core.weight.shape):
        raise ValueError(
            f"harvested U {tuple(u_rows.shape)} != ctxsplit U {tuple(core.weight.shape)}"
        )
    with torch.no_grad():
        core.weight.copy_(u_rows.to(device=core.weight.device, dtype=core.weight.dtype))
    meta = scratch_tie_ctxsplit(mlp)
    meta["init"] = "gpt2_u_rank"
    meta["warm_target"] = "u"
    meta["a_written"] = True
    meta["z_full_qkv"] = False
    if harvest:
        meta.update(harvest)
    return meta


def collect_block_mlp_inputs(
    model: nn.Module,
    layer_indices: Iterable[int],
    input_ids: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Token-level FFN inputs ``x = LN2(h)`` from a real forward (eval, no dropout)."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _hook(idx: int):
        def pre_hook(_module: nn.Module, args: tuple) -> None:
            captured[idx] = args[0].detach()

        return pre_hook

    for idx in layer_indices:
        handles.append(model.transformer.h[int(idx)].mlp.register_forward_pre_hook(_hook(int(idx))))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    missing = [int(i) for i in layer_indices if int(i) not in captured]
    if missing:
        raise RuntimeError(f"MLP hooks missed layers {missing}")
    return captured


def align_ctxsplit_uv_to_mlp(
    mlp: GPT2CtxSplitMLP,
    old_mlp: nn.Module,
    x: torch.Tensor,
    *,
    steps: int = 64,
    lr: float = 0.001,
) -> dict[str, Any]:
    """Warm **U,V on x**: Adam on ``‖Hessian(x) − MLP(x)‖²``.  A/z/λ stay frozen."""
    if x.ndim != 3:
        raise ValueError(f"align x must be (B, T, D), got {tuple(x.shape)}")
    meta = scratch_tie_ctxsplit(mlp)
    core = mlp.ffn.phi_core
    x = x.to(device=core.weight.device, dtype=core.weight.dtype)
    was_mlp_training = old_mlp.training
    old_mlp.eval()
    try:
        with torch.no_grad():
            y_target = old_mlp(x).detach()
            mse0 = float(F.mse_loss(mlp(x), y_target).item())
    finally:
        if was_mlp_training:
            old_mlp.train()

    a_before = core.context_weight.detach().clone()
    trainable = (core.weight, core.out_weight)
    for param in core.parameters():
        param.requires_grad_(param is core.weight or param is core.out_weight)
    opt = torch.optim.Adam(list(trainable), lr=lr)
    mse = mse0
    for _ in range(max(0, int(steps))):
        opt.zero_grad()
        loss = F.mse_loss(mlp(x), y_target)
        loss.backward()
        opt.step()
        mse = float(loss.detach().item())
    opt.zero_grad()
    for param in core.parameters():
        param.requires_grad_(True)
    if not torch.equal(core.context_weight.detach(), a_before):
        raise RuntimeError("x-align wrote A; only U and V may move")
    meta["init"] = "gpt2_x_align"
    meta["warm_target"] = "x"
    meta["a_tied_to_u"] = False
    meta["v_tied_to_u"] = False
    meta["z_full_qkv"] = False
    meta["align_steps"] = int(steps)
    meta["align_lr"] = float(lr)
    meta["align_tokens"] = int(x.shape[0] * x.shape[1])
    meta["align_mse0"] = mse0
    meta["align_mse"] = mse
    return meta


def align_ctxsplit_a_to_mlp(
    mlp: GPT2CtxSplitMLP,
    old_mlp: nn.Module,
    x: torch.Tensor,
    *,
    steps: int = 64,
    lr: float = 0.001,
) -> dict[str, Any]:
    """Continue an existing U/V alignment by warming only A and λ-bias.

    This does not reinitialize U, V, A, z, or either inherited bias.  The
    native ``ψ''(z@A)`` forward is optimized against the same MLP increment
    target used by x-align.
    """
    if x.ndim != 3:
        raise ValueError(f"align x must be (B, T, D), got {tuple(x.shape)}")
    core = mlp.ffn.phi_core
    x = x.to(device=core.weight.device, dtype=core.weight.dtype)
    was_mlp_training = old_mlp.training
    old_mlp.eval()
    try:
        with torch.no_grad():
            y_target = old_mlp(x).detach()
            mse0 = float(F.mse_loss(mlp(x), y_target).item())
    finally:
        if was_mlp_training:
            old_mlp.train()

    u_before = core.weight.detach().clone()
    v_before = core.out_weight.detach().clone()
    q_before = core.q_proj.weight.detach().clone()
    content_bias_before = core.content_bias.detach().clone()
    out_bias_before = core.out_bias.detach().clone()
    for param in core.parameters():
        param.requires_grad_(param is core.context_weight or param is core.bias)
    opt = torch.optim.Adam((core.context_weight, core.bias), lr=lr)
    mse = mse0
    for _ in range(max(0, int(steps))):
        opt.zero_grad()
        loss = F.mse_loss(mlp(x), y_target)
        loss.backward()
        opt.step()
        mse = float(loss.detach().item())
    opt.zero_grad()
    for param in core.parameters():
        param.requires_grad_(True)

    unchanged = (
        ("U", core.weight, u_before),
        ("V", core.out_weight, v_before),
        ("z", core.q_proj.weight, q_before),
        ("content_bias", core.content_bias, content_bias_before),
        ("out_bias", core.out_bias, out_bias_before),
    )
    for name, value, before in unchanged:
        if not torch.equal(value.detach(), before):
            raise RuntimeError(f"A-align wrote {name}; only A and λ-bias may move")
    return {
        "a_align_steps": int(steps),
        "a_align_lr": float(lr),
        "a_align_mse0": mse0,
        "a_align_mse": mse,
    }


def align_ctxsplit_uv_then_a_to_mlp(
    mlp: GPT2CtxSplitMLP,
    old_mlp: nn.Module,
    x: torch.Tensor,
    *,
    uv_steps: int = 64,
    uv_lr: float = 0.001,
    a_steps: int = 64,
    a_lr: float = 0.001,
) -> dict[str, Any]:
    """Two-stage warm init: proven U/V x-align, then A-only continuation."""
    meta = align_ctxsplit_uv_to_mlp(
        mlp, old_mlp, x, steps=uv_steps, lr=uv_lr
    )
    uv_mse0 = float(meta["align_mse0"])
    uv_mse = float(meta["align_mse"])
    a_meta = align_ctxsplit_a_to_mlp(
        mlp, old_mlp, x, steps=a_steps, lr=a_lr
    )
    meta.update(a_meta)
    meta["init"] = "gpt2_x_lam_align"
    meta["warm_target"] = "x_then_a"
    meta["align_mse0"] = uv_mse0
    meta["uv_align_mse"] = uv_mse
    meta["align_mse"] = float(a_meta["a_align_mse"])
    meta["a_written"] = True
    return meta


def align_ctxsplit_token_psi_to_mlp(
    mlp: GPT2CtxSplitMLP,
    old_mlp: nn.Module,
    x: torch.Tensor | None = None,
    *,
    psi_steps: int = 128,
    psi_lr: float = 0.001,
    uv_steps: int = 0,
    uv_lr: float = 0.001,
    psi_output_weight: float = 1.0,
    psi_lambda_weight: float | None = None,
    psi_field_weight: float | None = None,
) -> dict[str, Any]:
    """UV warm-start; ``λ = ψ''(s)`` from translated content ``s``.  ``x`` unused."""
    del (
        x,
        psi_steps,
        psi_lr,
        uv_steps,
        uv_lr,
        psi_output_weight,
        psi_lambda_weight,
        psi_field_weight,
    )
    meta = scratch_tie_ctxsplit(mlp)
    meta.update(warm_ctxsplit_uav_from_s(mlp, old_mlp))
    meta.update(
        {
            "init": "gpt2_token_psi_align",
            "warm_target": "uv",
            "oracle_gelu": False,
            "native_uav": True,
            "a_written": True,
            "a_tied_to_u": True,
            "v_tied_to_u": False,
        }
    )
    return meta


def align_ctxsplit_lambda_to_mlp(
    mlp: GPT2CtxSplitMLP,
    old_mlp: nn.Module,
    x: torch.Tensor,
    *,
    steps: int = 64,
    lr: float = 0.001,
) -> dict[str, Any]:
    """Warm **A** through the native active field ``ψ''(s) ⊙ proj_x`` (legacy W1/W2 copy).

    Pretrained coordinates stay ordered: U/b1←W1/b1, A←U, V/b2←W2/b2.
    The teacher is ``GELU(W1x+b1)``; only A and the λ-path bias move.
    Training stays ``ψ''(z @ A)`` — no GELU pretence and no λ switch.
    """
    if x.ndim != 3:
        raise ValueError(f"align x must be (B, T, D), got {tuple(x.shape)}")
    core = mlp.ffn.phi_core
    dim = int(core.weight.shape[1])
    x = x.to(device=core.weight.device, dtype=core.weight.dtype)
    w1, w2 = gpt2_mlp_w1_w2(old_mlp, dim)
    b1, b2 = gpt2_mlp_b1_b2(old_mlp)
    if int(core.weight.shape[0]) != int(w1.shape[0]):
        raise ValueError(
            f"lam_align requires width {int(w1.shape[0])}, "
            f"got {int(core.weight.shape[0])}"
        )
    device = core.weight.device
    dtype = core.weight.dtype
    with torch.no_grad():
        core.weight.copy_(w1.to(device=device, dtype=dtype))
        core.context_weight.copy_(core.weight)
        core.out_weight.copy_(w2.T.to(device=device, dtype=dtype))
        core.content_bias.copy_(b1.to(device=device, dtype=dtype))
        core.out_bias.copy_(b2.to(device=device, dtype=dtype))
        # Calibration compares the same token-local expansion coordinate.
        core.hp_logit.fill_(10.0)
    with torch.no_grad():
        field_tgt, lam_tgt = mlp_active_field(old_mlp, x, dim)
        field_tgt = field_tgt.detach()
        lam_tgt = lam_tgt.detach()
        output_tgt = (
            field_tgt @ w2.T.to(device=device, dtype=dtype)
            + b2.to(device=device, dtype=dtype)
        )
        lam0, proj0, _s_z0 = core.native_factors(x)
        field_mse0 = float(F.mse_loss(lam0 * proj0, field_tgt).item())
        lambda_mse0 = float(F.mse_loss(lam0, lam_tgt).item())
        output_mse0 = float(F.mse_loss(mlp(x), output_tgt).item())

    u_before = core.weight.detach().clone()
    v_before = core.out_weight.detach().clone()
    q_before = core.q_proj.weight.detach().clone()
    for param in core.parameters():
        param.requires_grad_(param is core.context_weight or param is core.bias)
    opt = torch.optim.Adam((core.context_weight, core.bias), lr=lr)
    field_mse = field_mse0
    output_mse = output_mse0
    for _ in range(max(0, int(steps))):
        opt.zero_grad()
        lam, proj_x, _s_z = core.native_factors(x)
        field_pred = lam * proj_x
        output_pred = field_pred @ core.out_weight + core.out_bias
        field_loss = F.mse_loss(field_pred, field_tgt)
        output_loss = F.mse_loss(output_pred, output_tgt)
        # The MLP increment is the transfer contract.  The field term keeps
        # A's native unit activity anchored instead of exploiting V's nullspace.
        loss = output_loss + field_loss
        loss.backward()
        opt.step()
        field_mse = float(field_loss.detach().item())
        output_mse = float(output_loss.detach().item())
    opt.zero_grad()
    for param in core.parameters():
        param.requires_grad_(True)
    if not torch.equal(core.weight.detach(), u_before):
        raise RuntimeError("lam_align wrote U; only A and λ-bias may move")
    if not torch.equal(core.out_weight.detach(), v_before):
        raise RuntimeError("lam_align wrote V; only A and λ-bias may move")
    if not torch.equal(core.q_proj.weight.detach(), q_before):
        raise RuntimeError("lam_align wrote z; only A and λ-bias may move")
    with torch.no_grad():
        lam_final = core.native_lambda(x)
        lambda_mse = float(F.mse_loss(lam_final, lam_tgt).item())
        output_mse = float(F.mse_loss(mlp(x), output_tgt).item())
    meta = {
        "init": "gpt2_lam_align",
        "warm_target": "active_field",
        "target_variant": str(mlp.ffn.variant),
        "num_units": int(core.weight.shape[0]),
        "dim": dim,
        "a_tied_to_u": False,
        "v_tied_to_u": False,
        "a_written": True,
        "z_full_qkv": False,
        "lambda_geometry": "tanh2_second_deriv",
        "oracle_gelu": False,
        "readout": None,
    }
    meta["align_steps"] = int(steps)
    meta["align_lr"] = float(lr)
    meta["align_tokens"] = int(x.shape[0] * x.shape[1])
    meta["align_mse0"] = field_mse0
    meta["align_mse"] = field_mse
    meta["lambda_mse0"] = lambda_mse0
    meta["lambda_mse"] = lambda_mse
    meta["output_mse0"] = output_mse0
    meta["output_mse"] = output_mse
    return meta


def warm_ctxsplit_z_from_gpt2_attn(mlp: GPT2CtxSplitMLP, block: nn.Module) -> dict[str, Any]:
    """UAV stay scratch-tied.  Warm **z only** from this layer's full QKV attn.

    Attention yields one signal (softmax(QKᵀ)V → out).  That signal is z, not λ.
    λ is never initialized from attention or through A; after z is set it is
    always ``ψ''(z @ A)``.
    """
    meta = scratch_tie_ctxsplit(mlp)
    core = mlp.ffn.phi_core
    dim = int(core.weight.shape[1])
    q, k, v, o_w, bq, bk, bv, o_b = gpt2_attn_qkv_o(block.attn, dim)
    n_head = int(getattr(block.attn, "num_heads", getattr(block.attn, "n_head", 1)))
    device = core.weight.device
    dtype = core.weight.dtype
    core.q_proj = nn.Linear(dim, dim, bias=True).to(device=device, dtype=dtype)
    core.k_proj = nn.Linear(dim, dim, bias=True).to(device=device, dtype=dtype)
    core.v_proj = nn.Linear(dim, dim, bias=True).to(device=device, dtype=dtype)
    core.z_out_proj = nn.Linear(dim, dim, bias=True).to(device=device, dtype=dtype)
    with torch.no_grad():
        _copy_linear(core.q_proj, q, bq)
        _copy_linear(core.k_proj, k, bk)
        _copy_linear(core.v_proj, v, bv)
        _copy_linear(core.z_out_proj, o_w, o_b)
    core.n_head = n_head
    core.use_full_qkv = True
    meta["init"] = "gpt2_z_qkv"
    meta["warm_target"] = "z"
    meta["z_full_qkv"] = True
    meta["n_head"] = n_head
    return meta


def ctxsplit_lambda_stats(core: nn.Module, x: torch.Tensor) -> dict[str, float]:
    """λ = tanh²''(z @ A + b) on a mean-pooled token.  Not identically 1."""
    z = x.mean(dim=1) if x.ndim == 3 else x
    s_z = z @ core.context_weight.transpose(-2, -1) + core.bias
    lam = _tanh2_second_deriv(s_z)
    return {
        "lam_mean": float(lam.mean().item()),
        "lam_std": float(lam.std().item()),
        "lam_near_one_frac": float((lam - 1.0).abs().lt(0.05).float().mean().item()),
    }


class GPT2CtxSplitBlock(nn.Module):
    """FFN slot without the host increment.  Attn residual is unchanged.

    ``none``:  y = FFN(LN2(h'))
    ``gated``: y = (1-α) h' + α FFN(LN2(h'))
    """

    def __init__(self, src: nn.Module, *, residual: ResidualMode = "none") -> None:
        super().__init__()
        if residual not in ("gated", "none"):
            raise ValueError(f"wrapper residual must be gated or none, got {residual!r}")
        self.ln_1 = src.ln_1
        self.attn = src.attn
        self.ln_2 = src.ln_2
        self.mlp = src.mlp
        if hasattr(src, "crossattention"):
            self.crossattention = src.crossattention
        if hasattr(src, "ln_cross_attn"):
            self.ln_cross_attn = src.ln_cross_attn
        self.residual_mode = residual
        if residual == "gated":
            self.ffn_mix_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("ffn_mix_logit", torch.tensor(0.0))

    def ffn_alpha(self) -> torch.Tensor:
        if self.residual_mode != "gated":
            return self.ffn_mix_logit.new_ones(())
        return torch.sigmoid(self.ffn_mix_logit)

    @staticmethod
    def _call_attn(module: nn.Module, hidden_states, *args, **kwargs):
        """transformers 4.x uses ``layer_past``; 5.x uses ``past_key_values``."""
        params = inspect.signature(module.forward).parameters
        mapped = dict(kwargs)
        if "layer_past" in params and "layer_past" not in mapped and "past_key_values" in mapped:
            mapped["layer_past"] = mapped.pop("past_key_values")
        if "past_key_values" in params and "past_key_values" not in mapped and "layer_past" in mapped:
            mapped["past_key_values"] = mapped.pop("layer_past")
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            mapped = {k: v for k, v in mapped.items() if k in params}
        return module(hidden_states, *args, **mapped)

    def forward(self, hidden_states, *args, **kwargs):
        use_cache = bool(kwargs.get("use_cache", False))
        output_attentions = bool(kwargs.get("output_attentions", False))
        encoder_hidden_states = kwargs.get("encoder_hidden_states")

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self._call_attn(self.attn, hidden_states, *args, **kwargs)
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            attn_rest = attn_outputs[1:]
        else:
            attn_output = attn_outputs
            attn_rest = ()
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_outputs = self._call_attn(self.crossattention, hidden_states, *args, **kwargs)
            cross_out = cross_outputs[0] if isinstance(cross_outputs, tuple) else cross_outputs
            hidden_states = residual + cross_out

        stream = hidden_states
        ff = self.mlp(self.ln_2(stream))
        if self.residual_mode == "none":
            hidden_states = ff
        else:
            alpha = self.ffn_alpha()
            hidden_states = (1.0 - alpha) * stream + alpha * ff

        if use_cache:
            return (hidden_states,) + attn_rest
        if output_attentions:
            return (hidden_states,) + attn_rest[-1:]
        return (hidden_states,)


def install_ctxsplit_layer(
    model: nn.Module,
    layer_idx: int,
    *,
    hessian_variant: HessianVariant = CTXSPLIT_VARIANT,
    num_units: int | None = None,
    lambda_geometry: str | None = None,
    mix_init: float = -10.0,
    a_split: ASplitMode | str = "joint_wv",
    a_split_blocks: int = 4,
    a_split_seed: int = 0,
    residual: ResidualMode = "add",
    warm_init: bool = True,
    init: InitMode = "uv_rank",
    align_x: torch.Tensor | None = None,
    align_steps: int = 64,
    align_lr: float = 0.001,
    a_align_steps: int = 64,
    a_align_lr: float = 0.001,
    harvested_u: torch.Tensor | None = None,
    harvest_meta: Mapping[str, Any] | None = None,
    hessian_factor_temperature: float = 1.0,
) -> dict[str, Any]:
    """Replace one GPT-2 block MLP with paper ctxsplit.

    Default ``uv_rank`` writes U from frozen-MLP active-rank tracking, then
    ``A,V ← U``.  Training is ``λ = ψ''(z @ A)``.  Host slot stays ``h' + F``.
    """
    src = model.transformer.h[layer_idx]
    old_mlp = src.mlp
    device = next(old_mlp.parameters()).device
    dim = int(model.config.n_embd)
    if num_units is None:
        raw_inner = getattr(model.config, "n_inner", None)
        num_units = int(raw_inner) if raw_inner is not None else 4 * dim

    mode: InitMode = "uv_rank" if init == "gpt2_warm" else init
    if warm_init and mode == "uv_rank":
        if harvested_u is None:
            raise ValueError(
                "uv_rank requires harvested_u from frozen-layer rank tracking"
            )
        n_keep = int(harvested_u.shape[0])
        hessian_mlp = GPT2CtxSplitMLP(
            dim,
            hessian_variant=hessian_variant,
            num_units=n_keep,
            lambda_geometry=lambda_geometry,
            mix_init=mix_init,
            hessian_factor_temperature=hessian_factor_temperature,
        ).to(device)
        meta = write_ctxsplit_u_then_tie(
            hessian_mlp, harvested_u, harvest=harvest_meta
        )
    else:
        hessian_mlp = GPT2CtxSplitMLP(
            dim,
            hessian_variant=hessian_variant,
            num_units=num_units,
            lambda_geometry=lambda_geometry,
            mix_init=mix_init,
            hessian_factor_temperature=hessian_factor_temperature,
        ).to(device)
        if not warm_init:
            meta = {
                "init": "cold",
                "target_variant": str(hessian_variant),
                "num_units": num_units,
                "dim": dim,
                "oracle_gelu": False,
            }
        elif mode == "scratch_tie":
            meta = scratch_tie_ctxsplit(hessian_mlp)
        elif mode == "gpt2_z_qkv":
            meta = warm_ctxsplit_z_from_gpt2_attn(hessian_mlp, src)
        elif mode == "x_align":
            if align_x is None:
                raise ValueError("x_align requires align_x from a real GPT-2 forward")
            meta = align_ctxsplit_uv_to_mlp(
                hessian_mlp,
                old_mlp,
                align_x,
                steps=align_steps,
                lr=align_lr,
            )
        elif mode in ("lam_align", "psi_align"):
            if align_x is None:
                raise ValueError("lam_align requires align_x from a real GPT-2 forward")
            meta = align_ctxsplit_lambda_to_mlp(
                hessian_mlp,
                old_mlp,
                align_x,
                steps=align_steps,
                lr=align_lr,
            )
        elif mode == "token_psi_align":
            meta = align_ctxsplit_token_psi_to_mlp(hessian_mlp, old_mlp)
        elif mode == "mod_native_swap":
            if not lambda_geometry:
                raise ValueError(
                    "mod_native_swap requires lambda_geometry "
                    "(proj_oracle_mod_native_simple or proj_oracle_mod_native_ortho)"
                )
            meta = align_ctxsplit_mod_native_to_mlp(
                hessian_mlp,
                old_mlp,
                lambda_geometry=lambda_geometry,
                a_split=a_split,
                a_split_blocks=a_split_blocks,
                a_split_seed=a_split_seed,
            )
        elif mode == "x_lam_align":
            if align_x is None:
                raise ValueError(
                    "x_lam_align requires align_x from a real GPT-2 forward"
                )
            meta = align_ctxsplit_uv_then_a_to_mlp(
                hessian_mlp,
                old_mlp,
                align_x,
                uv_steps=align_steps,
                uv_lr=align_lr,
                a_steps=a_align_steps,
                a_lr=a_align_lr,
            )
        elif mode == "uv_w12":
            meta = warm_ctxsplit_uv_from_gpt2_mlp(hessian_mlp, old_mlp)
        else:
            raise ValueError(f"unknown ctxsplit init {mode!r}")
    src.mlp = hessian_mlp
    if residual != "add":
        model.transformer.h[layer_idx] = GPT2CtxSplitBlock(src, residual=residual).to(device)
    meta["layer"] = layer_idx
    meta["ffn_residual"] = residual
    meta["warm_init"] = warm_init
    return meta


def install_ctxsplit_layers(
    model: nn.Module,
    layer_indices: Iterable[int],
    *,
    hessian_variant: HessianVariant = CTXSPLIT_VARIANT,
    num_units: int | None = None,
    lambda_geometry: str | None = None,
    mix_init: float = -10.0,
    a_split: ASplitMode | str = "joint_wv",
    a_split_blocks: int = 4,
    a_split_seed: int = 0,
    residual: ResidualMode = "add",
    warm_init: bool = True,
    init: InitMode = "uv_rank",
    align_inputs: Mapping[int, torch.Tensor] | None = None,
    align_steps: int = 64,
    align_lr: float = 0.001,
    a_align_steps: int = 64,
    a_align_lr: float = 0.001,
    harvested_u: torch.Tensor | None = None,
    harvest_meta: Mapping[str, Any] | None = None,
    hessian_factor_temperature: float = 1.0,
) -> list:
    return [
        install_ctxsplit_layer(
            model,
            int(idx),
            hessian_variant=hessian_variant,
            num_units=num_units,
            lambda_geometry=lambda_geometry,
            mix_init=mix_init,
            a_split=a_split,
            a_split_blocks=a_split_blocks,
            a_split_seed=a_split_seed,
            residual=residual,
            warm_init=warm_init,
            init=init,
            align_x=None if align_inputs is None else align_inputs[int(idx)],
            align_steps=align_steps,
            align_lr=align_lr,
            a_align_steps=a_align_steps,
            a_align_lr=a_align_lr,
            harvested_u=harvested_u,
            harvest_meta=harvest_meta,
            hessian_factor_temperature=hessian_factor_temperature,
        )
        for idx in layer_indices
    ]
