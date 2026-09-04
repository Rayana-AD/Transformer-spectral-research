"""TLB-Attention and related variants: custom multi-head attention for encoder-style (bidirectional) blocks.

Multi-head nuclear norm convention (used for TLB, layer-wise scalings, and diagnostics):

- **Per-head matrix**: For hidden layout ``(B, H, T, d_head)``, each head slice ``[b, h]`` is a
  ``(T, d_head)`` matrix. Its nuclear norm ``||·||_*`` is computed with ``dim=(-2, -1)``.

- **Layer scalar from hidden** ``x`` of shape ``(B, T, D)`` with ``D = num_heads * d_head``:
  reshape to ``(B, H, T, d_head)``, compute per-head nuclear norms, then take the **mean** over
  batch and heads. This single scalar matches neither "one giant matrix" nor vector norms; it is
  the project default for inter-layer ratios and for comparing to per-head TLB normalization.

- **Per-head scaling** (``kv_inner``, etc.): ratios use **per-(batch, head)** nuclear norms
  ``(B, H, 1, 1)`` broadcast over ``T, d_head``, so each head is self-consistent.

- **``nuc_inter_q``**: before softmax, scale **Q** by ``mean_h ||X||_* / mean_h ||Q||_*`` (layer input ``X`` vs projected ``Q``). No post-``out_proj`` ``nuc_inter`` scaling (contrast with ``nuc_inter`` on ``out``).

- **``v_jitter``** (training): add ``jitter_std * N(0,1)`` to **V** before ``attn @ V``; eval leaves V unchanged. Default std 1e-3 if ``jitter_std==0``.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

AttentionMode = Literal[
    "vanilla",
    "tlb",
    "frob",
    "l2",
    "sinkhorn",
    "nuc_inter",
    "nuc_inter_q",
    "kv_mul",
    "kv_inner",
    "kv_x_inter",
    "kv_inner_nuc_inter",
    "qv_inner",
    "randv_inner",
    "v_jitter",
]


def reference_original_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-head reference (batch, seq, d_k)."""
    d_k = q.size(-1)
    scale = math.sqrt(float(d_k))
    attn = (q @ k.transpose(-2, -1)) / scale
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    attn = F.softmax(attn, dim=-1)
    out = attn @ v
    return out, attn


def reference_tlb_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-head TLB: softmax(QK^T/sqrt(d)) @ (V / (||V||_* + eps))."""
    d_k = q.size(-1)
    scale = math.sqrt(float(d_k))
    attn = (q @ k.transpose(-2, -1)) / scale
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    attn = F.softmax(attn, dim=-1)
    nuclear_norm = torch.linalg.matrix_norm(v, ord="nuc", dim=(-2, -1), keepdim=True)
    v_norm = v / (nuclear_norm + eps)
    out = attn @ v_norm
    return out, attn


def _tlb_v_mix(v: torch.Tensor, eps: float, mix_ratio: float = 1.0) -> torch.Tensor:
    """Scale V by inverse nuclear norm; sanitize non-finite values for stable SVD.
    
    Args:
        v: Value tensor (batch, heads, seq, head_dim)
        eps: Minimum value for numerical stability
        mix_ratio: 0.0 = pure vanilla (no normalization), 1.0 = full TLB normalization
                   Values in (0,1) blend between vanilla and normalized V.
    """
    # Warmup phase: mix_ratio = 0.0 means pure vanilla (no normalization)
    if mix_ratio <= 0.0:
        return v
    
    v_safe = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    # MPS: nuclear norm on CPU float32 (small per-head matrices).
    if v_safe.device.type == "mps":
        nuc = torch.linalg.matrix_norm(
            v_safe.float().cpu(), ord="nuc", dim=(-2, -1), keepdim=True
        ).to(device=v_safe.device, dtype=v_safe.dtype)
    # CUDA bf16/fp16: matrix_norm can misbehave; compute norm in float32 then cast back.
    elif v_safe.device.type == "cuda" and v_safe.dtype in (torch.bfloat16, torch.float16):
        nuc = torch.linalg.matrix_norm(v_safe.float(), ord="nuc", dim=(-2, -1), keepdim=True)
    else:
        nuc = torch.linalg.matrix_norm(v_safe, ord="nuc", dim=(-2, -1), keepdim=True)
    nuc = torch.nan_to_num(nuc, nan=eps, posinf=1e6, neginf=eps).clamp(min=eps)
    v_norm = (v_safe / nuc).to(v_safe.dtype)

    # Blend: mix_ratio * normalized + (1 - mix_ratio) * original
    if mix_ratio >= 1.0:
        return v_norm
    return mix_ratio * v_norm + (1.0 - mix_ratio) * v_safe


def _matrix_nuclear_norm_bh(t: torch.Tensor) -> torch.Tensor:
    """Per-(batch, head) nuclear norm of (T, d_head) slices. t: (B, H, T, d_h) -> (B, H)."""
    ts = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    if ts.device.type == "mps":
        nuc = torch.linalg.matrix_norm(ts.float().cpu(), ord="nuc", dim=(-2, -1)).to(
            device=ts.device, dtype=ts.dtype
        )
    elif ts.device.type == "cuda" and ts.dtype in (torch.bfloat16, torch.float16):
        nuc = torch.linalg.matrix_norm(ts.float(), ord="nuc", dim=(-2, -1))
    else:
        nuc = torch.linalg.matrix_norm(ts, ord="nuc", dim=(-2, -1))
    return torch.nan_to_num(nuc, nan=0.0, posinf=1e6, neginf=0.0)


def _matrix_nuclear_norm_bh_keep(t: torch.Tensor) -> torch.Tensor:
    """Same as _matrix_nuclear_norm_bh but keepdim for broadcasting: (B, H, 1, 1)."""
    n = _matrix_nuclear_norm_bh(t)
    return n.unsqueeze(-1).unsqueeze(-1)


def layer_hidden_nuclear_mean(x: torch.Tensor, num_heads: int, eps: float = 1e-8) -> torch.Tensor:
    """Mean nuclear norm over batch and heads for hidden states (B, T, D) with D = num_heads * d_head.

    Reshapes to (B, H, T, d_head), computes per-head ||·||_*, returns a scalar tensor (mean).
    """
    b, t, d = x.shape
    if d % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    d_h = d // num_heads
    xh = x.view(b, t, num_heads, d_h).transpose(1, 2).contiguous()
    nuc = _matrix_nuclear_norm_bh(xh)
    return nuc.mean().clamp_min(eps)


def sinkhorn_normalize(
    attn: torch.Tensor,
    n_iters: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Alternate row/column normalization to a doubly-stochastic-like matrix (non-negative input)."""
    x = attn.clamp_min(0.0)
    for _ in range(max(1, n_iters)):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x


def copy_mha_weights_from_torch(dst: nn.Module, src: nn.MultiheadAttention) -> None:
    """Copy weights from nn.MultiheadAttention into CustomMultiheadAttention (same shapes)."""
    d = dst  # type: ignore[assignment]
    assert hasattr(d, "in_proj") and hasattr(d, "out_proj")
    with torch.no_grad():
        d.in_proj.weight.copy_(src.in_proj_weight)
        d.in_proj.bias.copy_(src.in_proj_bias)
        d.out_proj.weight.copy_(src.out_proj.weight)
        d.out_proj.bias.copy_(src.out_proj.bias)


class CustomMultiheadAttention(nn.Module):
    """Multi-head self-attention with optional TLB / Frobenius / L2 / Sinkhorn-on-weights variants."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        batch_first: bool = True,
        attention_mode: AttentionMode = "vanilla",
        tlb_eps: float = 1e-8,
        sinkhorn_iters: int = 5,
        tlb_warmup_epochs: int = 0,
        jitter_std: float = 0.0,
    ) -> None:
        super().__init__()
        if not batch_first:
            raise ValueError("Only batch_first=True is supported.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_mode = attention_mode
        self.tlb_eps = float(tlb_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.tlb_warmup_epochs = int(tlb_warmup_epochs)
        self.current_epoch = 0  # Will be set externally during training
        self.jitter_std = float(jitter_std)

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
    
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for adaptive TLB warmup."""
        self.current_epoch = epoch
    
    def _get_tlb_mix_ratio(self) -> float:
        """Calculate TLB mix ratio based on current epoch and warmup setting."""
        if self.attention_mode != "tlb" or self.tlb_warmup_epochs <= 0:
            return 1.0  # Full TLB or not in TLB mode
        # Linear warmup: epoch 0 -> 0.0, epoch warmup_epochs -> 1.0
        return min(1.0, self.current_epoch / self.tlb_warmup_epochs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query is not key or key is not value:
            raise NotImplementedError("Only self-attention (query=key=value) is implemented.")
        x = query
        b, t, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # Line A on Q: match layer-input vs Q nuclear "scale" before softmax (aligns query side, not V path).
        if self.attention_mode == "nuc_inter_q":
            nuc_in = layer_hidden_nuclear_mean(x, self.num_heads, self.tlb_eps)
            nuc_q = _matrix_nuclear_norm_bh(q).mean().clamp_min(self.tlb_eps)
            q = q * (nuc_in / nuc_q)

        scale = math.sqrt(float(self.head_dim))
        attn = (q @ k.transpose(-2, -1)) / scale
        if attn_mask is not None:
            # (L, L) or broadcastable
            attn = attn + attn_mask
        if key_padding_mask is not None:
            # True = ignore this key position
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        if self.attention_mode == "vanilla":
            v_mix = v
        elif self.attention_mode == "tlb":
            mix_ratio = self._get_tlb_mix_ratio()
            v_mix = _tlb_v_mix(v, self.tlb_eps, mix_ratio)
        elif self.attention_mode == "frob":
            fro = torch.linalg.vector_norm(v, ord="fro", dim=(-2, -1), keepdim=True)
            v_mix = v / (fro + self.tlb_eps)
        elif self.attention_mode == "l2":
            v_mix = v / (v.norm(dim=-1, keepdim=True) + self.tlb_eps)
        elif self.attention_mode == "sinkhorn":
            v_mix = v
            attn = sinkhorn_normalize(attn, n_iters=self.sinkhorn_iters, eps=self.tlb_eps)
        elif self.attention_mode == "kv_mul":
            v_mix = k * v
        elif self.attention_mode == "kv_inner":
            nuc_v = _matrix_nuclear_norm_bh_keep(v)
            nuc_kv = _matrix_nuclear_norm_bh_keep(k * v)
            c = nuc_v / (nuc_kv + self.tlb_eps)
            v_mix = c * (k * v)
        elif self.attention_mode == "kv_x_inter":
            nuc_prev = layer_hidden_nuclear_mean(x, self.num_heads, self.tlb_eps)
            KV = k * v
            nuc_kv = _matrix_nuclear_norm_bh(KV).mean().clamp_min(self.tlb_eps)
            c = nuc_prev / nuc_kv
            v_mix = c * KV
        elif self.attention_mode == "qv_inner":
            nuc_v = _matrix_nuclear_norm_bh_keep(v)
            nuc_qv = _matrix_nuclear_norm_bh_keep(q * v)
            c = nuc_v / (nuc_qv + self.tlb_eps)
            v_mix = c * (q * v)
        elif self.attention_mode == "randv_inner":
            rnd = torch.randn_like(v)
            nuc_v = _matrix_nuclear_norm_bh_keep(v)
            nuc_rv = _matrix_nuclear_norm_bh_keep(rnd * v)
            c = nuc_v / (nuc_rv + self.tlb_eps)
            v_mix = c * (rnd * v)
        elif self.attention_mode == "nuc_inter":
            v_mix = v
        elif self.attention_mode == "nuc_inter_q":
            v_mix = v
        elif self.attention_mode == "v_jitter":
            v_mix = v
            eff = self.jitter_std if self.jitter_std > 0 else 1e-3
            if self.training:
                v_mix = v + eff * torch.randn_like(v)
        elif self.attention_mode == "kv_inner_nuc_inter":
            nuc_v = _matrix_nuclear_norm_bh_keep(v)
            nuc_kv = _matrix_nuclear_norm_bh_keep(k * v)
            c = nuc_v / (nuc_kv + self.tlb_eps)
            v_mix = c * (k * v)
        else:
            raise ValueError(f"unknown attention_mode: {self.attention_mode!r}")

        out = attn @ v_mix
        out = out.transpose(1, 2).reshape(b, t, self.embed_dim)
        out = self.out_proj(out)
        if self.attention_mode in ("nuc_inter", "kv_inner_nuc_inter"):
            nuc_in = layer_hidden_nuclear_mean(x, self.num_heads, self.tlb_eps)
            nuc_o = layer_hidden_nuclear_mean(out, self.num_heads, self.tlb_eps)
            out = (nuc_in / nuc_o) * out
        return out, attn if need_weights else None


class CrossMultiheadAttention(nn.Module):
    """Cross-attention: Q from query sequence, K/V from memory (encoder output). Same V-normalization modes."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        batch_first: bool = True,
        attention_mode: AttentionMode = "vanilla",
        tlb_eps: float = 1e-8,
        sinkhorn_iters: int = 5,
        tlb_warmup_epochs: int = 0,
        jitter_std: float = 0.0,
    ) -> None:
        super().__init__()
        if not batch_first:
            raise ValueError("Only batch_first=True is supported.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_mode = attention_mode
        self.tlb_eps = float(tlb_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.tlb_warmup_epochs = int(tlb_warmup_epochs)
        self.current_epoch = 0
        self.jitter_std = float(jitter_std)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for adaptive TLB warmup."""
        self.current_epoch = epoch

    def _get_tlb_mix_ratio(self) -> float:
        """Calculate TLB mix ratio based on current epoch and warmup setting."""
        if self.attention_mode != "tlb" or self.tlb_warmup_epochs <= 0:
            return 1.0
        return min(1.0, self.current_epoch / self.tlb_warmup_epochs)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """query: (B, Tq, D); key_value: (B, Tk, D)."""
        b, tq, _ = query.shape
        tk = key_value.shape[1]
        q = self.q_proj(query).view(b, tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(b, tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(b, tk, self.num_heads, self.head_dim).transpose(1, 2)
        if self.attention_mode == "nuc_inter_q":
            nuc_in = layer_hidden_nuclear_mean(query, self.num_heads, self.tlb_eps)
            nuc_q = _matrix_nuclear_norm_bh(q).mean().clamp_min(self.tlb_eps)
            q = q * (nuc_in / nuc_q)
        scale = math.sqrt(float(self.head_dim))
        attn = (q @ k.transpose(-2, -1)) / scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        if self.attention_mode == "vanilla":
            v_mix = v
        elif self.attention_mode == "tlb":
            mix_ratio = self._get_tlb_mix_ratio()
            v_mix = _tlb_v_mix(v, self.tlb_eps, mix_ratio)
        elif self.attention_mode == "frob":
            fro = torch.linalg.vector_norm(v, ord="fro", dim=(-2, -1), keepdim=True)
            v_mix = v / (fro + self.tlb_eps)
        elif self.attention_mode == "l2":
            v_mix = v / (v.norm(dim=-1, keepdim=True) + self.tlb_eps)
        elif self.attention_mode == "sinkhorn":
            v_mix = v
            attn = sinkhorn_normalize(attn, n_iters=self.sinkhorn_iters, eps=self.tlb_eps)
        elif self.attention_mode == "kv_mul":
            v_mix = k * v
        elif self.attention_mode == "kv_inner":
            nuc_v = _matrix_nuclear_norm_bh_keep(v)
            nuc_kv = _matrix_nuclear_norm_bh_keep(k * v)
            c = nuc_v / (nuc_kv + self.tlb_eps)
            v_mix = c * (k * v)
        elif self.attention_mode == "kv_x_inter":
            nuc_prev = layer_hidden_nuclear_mean(query, self.num_heads, self.tlb_eps)
            KV = k * v
            nuc_kv = _matrix_nuclear_norm_bh(KV).mean().clamp_min(self.tlb_eps)
            c = nuc_prev / nuc_kv
            v_mix = c * KV
        elif self.attention_mode == "qv_inner":
            nuc_v = _matrix_nuclear_norm_bh_keep(v)
            nuc_qv = _matrix_nuclear_norm_bh_keep(q * v)
            c = nuc_v / (nuc_qv + self.tlb_eps)
            v_mix = c * (q * v)
        elif self.attention_mode == "randv_inner":
            rnd = torch.randn_like(v)
            nuc_v = _matrix_nuclear_norm_bh_keep(v)
            nuc_rv = _matrix_nuclear_norm_bh_keep(rnd * v)
            c = nuc_v / (nuc_rv + self.tlb_eps)
            v_mix = c * (rnd * v)
        elif self.attention_mode == "nuc_inter":
            v_mix = v
        elif self.attention_mode == "nuc_inter_q":
            v_mix = v
        elif self.attention_mode == "v_jitter":
            v_mix = v
            eff = self.jitter_std if self.jitter_std > 0 else 1e-3
            if self.training:
                v_mix = v + eff * torch.randn_like(v)
        elif self.attention_mode == "kv_inner_nuc_inter":
            nuc_v = _matrix_nuclear_norm_bh_keep(v)
            nuc_kv = _matrix_nuclear_norm_bh_keep(k * v)
            c = nuc_v / (nuc_kv + self.tlb_eps)
            v_mix = c * (k * v)
        else:
            raise ValueError(f"unknown attention_mode: {self.attention_mode!r}")
        out = attn @ v_mix
        out = out.transpose(1, 2).reshape(b, tq, self.embed_dim)
        out = self.out_proj(out)
        if self.attention_mode in ("nuc_inter", "kv_inner_nuc_inter"):
            nuc_in = layer_hidden_nuclear_mean(query, self.num_heads, self.tlb_eps)
            nuc_o = layer_hidden_nuclear_mean(out, self.num_heads, self.tlb_eps)
            out = (nuc_in / nuc_o) * out
        return out, attn if need_weights else None


class TransformerEncoderLayerCustom(nn.Module):
    """Post-LN encoder layer (same order as script10 TransformerEncoderLayerAttn)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "relu",
        *,
        attention_mode: AttentionMode = "vanilla",
        tlb_eps: float = 1e-8,
        sinkhorn_iters: int = 5,
        post_attn_layer_norm: bool = False,
        tlb_warmup_epochs: int = 0,
        jitter_std: float = 0.0,
    ) -> None:
        super().__init__()
        if activation not in ("relu", "gelu", "silu"):
            raise ValueError("activation must be relu, gelu, or silu")
        self.self_attn = CustomMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            attention_mode=attention_mode,
            tlb_eps=tlb_eps,
            sinkhorn_iters=sinkhorn_iters,
            tlb_warmup_epochs=tlb_warmup_epochs,
            jitter_std=jitter_std,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.post_attn_ln = nn.LayerNorm(d_model) if post_attn_layer_norm else None
        if activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        else:
            self.act = F.silu

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(src)
        attn_out, w = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=src_key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )
        if self.post_attn_ln is not None:
            attn_out = self.post_attn_ln(attn_out)
        src = src + self.dropout1(attn_out)
        h2 = self.norm2(src)
        ff = self.linear2(self.dropout2(self.act(self.linear1(h2))))
        src = src + self.dropout2(ff)
        if need_weights:
            return src, w
        return src


class SeqModelTransformerFlexible(nn.Module):
    """Transformer LM with CustomMultiheadAttention encoder stack."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        nhead: int,
        dim_feedforward: int | None = None,
        *,
        max_seq_len: int,
        pos_enc: str = "learned",
        ffn_activation: str = "relu",
        dropout: float = 0.0,
        attention_mode: AttentionMode = "vanilla",
        tlb_eps: float = 1e-8,
        sinkhorn_iters: int = 5,
        post_attn_layer_norm: bool = False,
        pos_sin_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * dim
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        if pos_enc not in ("learned", "none", "sinusoidal"):
            raise ValueError("pos_enc must be 'learned', 'none', or 'sinusoidal'")
        self.pos_enc = pos_enc
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim) if pos_enc == "learned" else None
        if pos_enc == "sinusoidal":
            if pos_sin_buffer is None:
                raise ValueError("pos_sin_buffer required when pos_enc=sinusoidal")
            self.register_buffer("pos_sin", pos_sin_buffer, persistent=False)
        else:
            self.pos_sin = None  # type: ignore[assignment]
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerCustom(
                    dim,
                    nhead,
                    dim_feedforward,
                    dropout=dropout,
                    activation=ffn_activation,
                    attention_mode=attention_mode,
                    tlb_eps=tlb_eps,
                    sinkhorn_iters=sinkhorn_iters,
                    post_attn_layer_norm=post_attn_layer_norm,
                    jitter_std=0.0,
                )
                for _ in range(depth)
            ]
        )
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _b, seq_len = token_ids.shape
        x = self.embed(token_ids)
        if self.pos_embed is not None:
            pos = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
            x = x + self.pos_embed(pos)
        elif self.pos_enc == "sinusoidal":
            x = x + self.pos_sin[:seq_len].to(device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)  # type: ignore[assignment]
        return self.lm_head(x)

    @torch.no_grad()
    def forward_collect_attentions(
        self,
        token_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns logits and list of attention weight tensors (B, H, L, L) per layer."""
        _b, seq_len = token_ids.shape
        x = self.embed(token_ids)
        if self.pos_embed is not None:
            pos = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
            x = x + self.pos_embed(pos)
        elif self.pos_enc == "sinusoidal":
            x = x + self.pos_sin[:seq_len].to(device=x.device, dtype=x.dtype)
        attns: list[torch.Tensor] = []
        for layer in self.layers:
            h = layer.norm1(x)
            _out, w = layer.self_attn(
                h,
                h,
                h,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
            )
            if layer.post_attn_ln is not None:
                _out = layer.post_attn_ln(_out)
            x = x + layer.dropout1(_out)
            h2 = layer.norm2(x)
            ff = layer.linear2(layer.dropout2(layer.act(layer.linear1(h2))))
            x = x + layer.dropout2(ff)
            if w is not None:
                attns.append(w.detach())
        logits = self.lm_head(x)
        return logits, attns


def _run_numerical_self_tests() -> None:
    torch.manual_seed(0)
    b, t, d_model, nh = 2, 7, 8, 1
    d_h = d_model // nh
    x = torch.randn(b, t, d_model)
    ref_mha = nn.MultiheadAttention(d_model, nh, dropout=0.0, batch_first=True)
    cust = CustomMultiheadAttention(d_model, nh, dropout=0.0, attention_mode="vanilla")
    copy_mha_weights_from_torch(cust, ref_mha)
    pad = torch.zeros(b, t, dtype=torch.bool)
    pad[:, -2:] = True
    y_ref, _ = ref_mha(x, x, x, key_padding_mask=pad, need_weights=False)
    y_cust, _ = cust(x, x, x, key_padding_mask=pad, need_weights=False)
    diff = (y_ref - y_cust).abs().max().item()
    assert diff < 1e-5, f"vanilla MHA mismatch max diff {diff}"

    # Single-head TLB vs reference
    q = torch.randn(b, t, d_h)
    k = torch.randn(b, t, d_h)
    v = torch.randn(b, t, d_h)
    eps = 1e-8
    o1, a1 = reference_tlb_attention(q, k, v, mask=None, eps=eps)
    lin = nn.Linear(d_h, 3 * d_h, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.cat([torch.eye(d_h), torch.eye(d_h), torch.eye(d_h)], dim=0))
    x1 = v  # identity for q,k,v projection test: use same v for all - bad
    # Instead: build one-head MHA with identity in_proj so q,k,v are independent
    cust1 = CustomMultiheadAttention(d_h, 1, dropout=0.0, attention_mode="tlb", tlb_eps=eps)
    with torch.no_grad():
        cust1.in_proj.weight.zero_()
        cust1.in_proj.bias.zero_()
        cust1.in_proj.weight[:d_h, :].copy_(torch.eye(d_h))
        cust1.in_proj.weight[d_h : 2 * d_h, :].copy_(torch.eye(d_h))
        cust1.in_proj.weight[2 * d_h :, :].copy_(torch.eye(d_h))
        cust1.out_proj.weight.copy_(torch.eye(d_h))
        cust1.out_proj.bias.zero_()
    inp = torch.randn(b, t, d_h)
    # q,k,v from same inp -> q=k=v=inp for identity blocks... actually each uses same x
    y2, _ = cust1(inp, inp, inp, key_padding_mask=None, need_weights=False)
    o2, _ = reference_tlb_attention(inp, inp, inp, mask=None, eps=eps)
    diff2 = (y2 - o2).abs().max().item()
    assert diff2 < 1e-4, f"TLB single-head mismatch {diff2}"
    print("tlb_attention_core self-tests OK (vanilla match PyTorch MHA; TLB match reference).")


if __name__ == "__main__":
    _run_numerical_self_tests()
