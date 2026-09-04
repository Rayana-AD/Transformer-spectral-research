"""Mechanistic metrics for Transformer probes: entropy, nuclear norms, QK^T singular values."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from script8_dual_space_memory import (
    attention_row_entropy,
    batch_mean_nuclear,
    nuclear_chain_from_hidden,
)

RANK_THRESHOLD = 1e-6


def singular_values_np(M: torch.Tensor) -> np.ndarray:
    return np.linalg.svd(M.detach().float().cpu().numpy(), compute_uv=False)


def attention_row_entropy_mha(attn: torch.Tensor, eps: float = 1e-10) -> float:
    """Mean Shannon entropy of attention rows; attn (B, H, L, L) after softmax."""
    ent = -(attn * torch.log(attn.clamp_min(eps))).sum(dim=-1)  # (B, H, L)
    return float(ent.mean().item())


def hidden_nuclear_chain(
    hidden_list: list[torch.Tensor], k_trunc: int | None = None
) -> tuple[list[float], list[float | None], list[float | None]]:
    """Delegates to script8; hidden tensors (B, L, d)."""
    return nuclear_chain_from_hidden(hidden_list, k_trunc=k_trunc)


def qk_logits_per_head(
    h: torch.Tensor,
    in_proj: torch.nn.Linear,
    nhead: int,
    scale: float,
) -> torch.Tensor:
    """Compute pre-softmax logits (B, H, L, L) from pre-norm hidden h."""
    qkv = in_proj(h)
    d_model = h.size(-1)
    d_h = d_model // nhead
    b, t, _ = h.shape
    q, k, _ = qkv.chunk(3, dim=-1)
    q = q.view(b, t, nhead, d_h).transpose(1, 2)
    k = k.view(b, t, nhead, d_h).transpose(1, 2)
    return (q @ k.transpose(-2, -1)) / scale


def qk_singular_value_summary(
    logits: torch.Tensor,
    *,
    batch_idx: int = 0,
    head_idx: int = 0,
    top_k: int = 8,
) -> dict[str, Any]:
    """Singular values of QK^T/sqrt(d) for one head (L, L). Returns mean top-k and full vector prefix."""
    m = logits[batch_idx, head_idx].detach().float().cpu()
    s = singular_values_np(m)
    kk = min(top_k, int(s.size))
    return {
        "singular_values_top": s[:kk].tolist(),
        "spectral_norm": float(s[0]) if s.size > 0 else 0.0,
        "nuclear_norm_logits": float(np.sum(s)),
        "rank_est": int(np.sum(s > RANK_THRESHOLD)),
    }


def probe_seq_model_flexible(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    pad_mask: torch.Tensor | None,
) -> dict[str, Any]:
    """
    Run forward_collect_attentions + hidden chain + per-layer entropy + QK summary.
    Expects SeqModelTransformerFlexible with forward_collect_attentions.
    """
    from tlb_attention_core import SeqModelTransformerFlexible

    if not isinstance(model, SeqModelTransformerFlexible):
        raise TypeError("probe_seq_model_flexible expects SeqModelTransformerFlexible")
    model.eval()
    _logits, attns = model.forward_collect_attentions(token_ids, src_key_padding_mask=pad_mask)
    _b, seq_len, d = model.embed(token_ids).shape  # type: ignore[union-attr]
    nhead = model.layers[0].self_attn.num_heads  # type: ignore[union-attr]
    head_dim = d // nhead
    scale = math.sqrt(float(head_dim))

    x = model.embed(token_ids)  # type: ignore[union-attr]
    if model.pos_embed is not None:
        pos = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        x = x + model.pos_embed(pos)
    elif model.pos_enc == "sinusoidal":
        x = x + model.pos_sin[:seq_len].to(device=x.device, dtype=x.dtype)  # type: ignore[union-attr]
    hiddens: list[torch.Tensor] = [x.detach()]
    qk_summaries: list[dict[str, Any]] = []
    for layer in model.layers:
        h = layer.norm1(x)
        logits_qk = qk_logits_per_head(h, layer.self_attn.in_proj, nhead, scale)
        qk_summaries.append(qk_singular_value_summary(logits_qk, top_k=8))
        attn_out, _ = layer.self_attn(
            h,
            h,
            h,
            key_padding_mask=pad_mask,
            need_weights=False,
        )
        if layer.post_attn_ln is not None:
            attn_out = layer.post_attn_ln(attn_out)
        x = x + layer.dropout1(attn_out)
        h2 = layer.norm2(x)
        ff = layer.linear2(layer.dropout2(layer.act(layer.linear1(h2))))
        x = x + layer.dropout2(ff)
        hiddens.append(x.detach())

    nuc_full, c_full, c_k = hidden_nuclear_chain(hiddens, k_trunc=min(8, d))
    entropies = [attention_row_entropy_mha(a) for a in attns]

    return {
        "nuclear_layers": nuc_full,
        "c_l_nuclear": [c for c in c_full if c is not None],
        "c_l_nuclear_k": [c for c in c_k if c is not None],
        "attn_entropy_layers": entropies,
        "qk_singular_top_per_layer": [q["singular_values_top"] for q in qk_summaries],
        "qk_spectral_norm_per_layer": [q["spectral_norm"] for q in qk_summaries],
    }


# Re-export for convenience
__all__ = [
    "attention_row_entropy",
    "attention_row_entropy_mha",
    "hidden_nuclear_chain",
    "qk_logits_per_head",
    "qk_singular_value_summary",
    "probe_seq_model_flexible",
    "singular_values_np",
]
