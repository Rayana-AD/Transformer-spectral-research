# Script8 — Experiment plan (implementation matches this):
# - Dual: After input LayerNorm, orthogonal projectors split x_n = x_mem + x_smt (complementary subspaces).
#   Memory path: Linear(x_mem) only (no activation). Smart path: Pre-LN + MHA + FFN (4*d) + ReLU on x_smt → s.
#   Merge: x_out = mem + s. Optional --global-dual-channel: shared ortho + linear_mem across layers.
#   Optional --dual-last-layer-no-mem: last block output = smart path s only (memory branch omitted from merge).
#
# Dual-channel loss (dual_ortho_mem; one backward):
#   L = dual_main_ce_weight * CE_main + Σ_aux
#   CE_main = CE( lm_head(h), labels ), h = final merged hidden (mem+s each layer).
#   Σ_aux (optional weights; × --depth by default via --dual-aux-scale-by-depth; use --no-dual-aux-scale-by-depth off):
#     • w_mem  · layer-mean MSE(mem_l, detach(target_l)) — default base 0.08 (use 0 to disable)
#     • w_orth · orthogonality mem vs smart path (cos2/dot2) — default base 0.08 (use 0 to disable)
#   Smart FFN: Linear1 → act → Linear2 (no extra readout head, no temporal aux).
# - Defaults favor longer copy sequences (n_min/n_max), more steps, and multi-seed summaries (see --seeds).
# - --task identity: native copy — tokens in [0, vocab_content), pad_id=vocab_content, vocab_size=vocab_content+1 (no DELIM).
#   Delimiter task: vocab_size=vocab_content+2; layout [content n | DELIM | tail n] → length 2n+1 (n parallel copy targets each need one input slot). tail = n i.i.d. content tokens.
# - Baseline: standard Transformer encoder on the full stream (same nhead / 4d FFN).
# - Default --depth 4 (was 2): deeper stack for copy/diagnostics; use --depth 6 if budget allows.
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from memory_tasks import build_dup_tail_copy_batch, build_gap_copy_batch
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paper_style import apply_paper_style

RANK_THRESHOLD = 1e-6
COND_EPS = 1e-9
IGNORE_LABEL = -100
DEFAULT_OUTPUT = Path("results_script8_dual_space_memory")


# ---------------------------------------------------------------------------
# Matrix metrics (full + top-k nuclear / structural norm)
# ---------------------------------------------------------------------------
def singular_values_np(M: torch.Tensor) -> np.ndarray:
    return np.linalg.svd(M.detach().float().cpu().numpy(), compute_uv=False)


def analyze_matrix(
    M: torch.Tensor, k_trunc: int | None = None
) -> tuple[int, float, float, float, float]:
    """秩、谱范数、全核范数、top-k 核范数 ‖·‖_*^(k)、条件数。"""
    s = singular_values_np(M)
    rank = int(np.sum(s > RANK_THRESHOLD))
    spectral = float(s[0]) if rank > 0 else 0.0
    nuclear_full = float(np.sum(s))
    kk = s.size if k_trunc is None else min(int(k_trunc), int(s.size))
    nuclear_k = float(np.sum(s[:kk])) if kk > 0 else 0.0
    cond = float(s[0] / s[-1]) if (rank > 0 and s[-1] > COND_EPS) else float("inf")
    return rank, spectral, nuclear_full, nuclear_k, cond


def batch_mean_nuclear(
    h: torch.Tensor, k_trunc: int | None = None
) -> tuple[float, float]:
    """h: (B, L, d)。返回 batch 平均的全核范数与 top-k 核范数。"""
    nuc_f, nuc_k = [], []
    for bi in range(h.shape[0]):
        _, _, nf, nk, _ = analyze_matrix(h[bi], k_trunc=k_trunc)
        nuc_f.append(nf)
        nuc_k.append(nk)
    return float(np.mean(nuc_f)), float(np.mean(nuc_k))


def nuclear_chain_from_hidden(
    hidden: list[torch.Tensor], k_trunc: int | None = None
) -> tuple[list[float], list[float | None], list[float | None]]:
    """Per-layer mean nuclear norms and c_l = N_{l+1}/N_l (length len(hidden)-1)."""
    nuc_full: list[float] = []
    nuc_k: list[float] = []
    for h in hidden:
        nf, nk = batch_mean_nuclear(h, k_trunc=k_trunc)
        nuc_full.append(nf)
        nuc_k.append(nk)

    def ratios(seq: list[float]) -> list[float | None]:
        out: list[float | None] = []
        for i in range(len(seq) - 1):
            if seq[i] > 1e-30:
                out.append(seq[i + 1] / seq[i])
            else:
                out.append(None)
        return out

    return nuc_full, ratios(nuc_full), ratios(nuc_k)


def parse_acc_hit_thresholds(s: str) -> list[float]:
    """Parse --acc-hit-thresholds into floats; empty string => []."""
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def first_steps_val_acc_ge(
    rows: list[dict], thresholds: list[float]
) -> dict[str, str]:
    """First logged step where val_acc_all >= thr; value '' if never reached."""
    result: dict[str, str] = {}
    for thr in thresholds:
        tag = format(thr, "g").replace(".", "p").replace("-", "neg")
        key = f"first_step_acc_ge_{tag}"
        hit = ""
        for r in rows:
            if float(r["val_acc_all"]) + 1e-12 >= thr:
                hit = str(int(r["step"]))
                break
        result[key] = hit
    return result


def nuclear_c_bar_geom(nuc_chain: list[float]) -> float:
    """(N_L/N_0)^(1/L)，L = len(chain)-1 段过渡；与 script7_vit_criticality.nuclear_c_bar_geom 一致。"""
    if len(nuc_chain) < 2 or nuc_chain[0] <= 1e-30:
        return float("nan")
    depth = len(nuc_chain) - 1
    ratio = nuc_chain[-1] / nuc_chain[0]
    if ratio <= 0 or not math.isfinite(ratio):
        return float("nan")
    return float(ratio ** (1.0 / depth))


def attention_row_entropy(attn: torch.Tensor, eps: float = 1e-10) -> float:
    """attn: (B, L, L) 行随机。返回 batch 平均 mean_i H(row_i)。"""
    # attn already softmax rows
    ent = -(attn * torch.log(attn.clamp_min(eps))).sum(dim=-1)  # (B, L)
    return float(ent.mean().item())


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class OrthogonalProjection(nn.Module):
    def __init__(self, dim: int, split_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.split_dim = split_dim
        self.W = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # MPS 尚无 linalg.qr；在 CPU 上对 d×d 做小矩阵 QR 再迁回（反传仍回到 W）。
        W = self.W
        if W.device.type == "mps":
            W_orth, _ = torch.linalg.qr(W.cpu())
            W_orth = W_orth.to(device=W.device, dtype=W.dtype)
        else:
            W_orth, _ = torch.linalg.qr(W)
        W_mem = W_orth[:, : self.split_dim]
        W_smt = W_orth[:, self.split_dim :]
        # x: (B, L, d)
        x_mem = x @ W_mem @ W_mem.T
        x_smt = x @ W_smt @ W_smt.T
        return x_mem, x_smt


def _coord_split_stream(x: torch.Tensor, mem_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed coordinate split: mem保留前 mem_dim 维，smt 保留后 d−mem_dim 维，互补位置填零。"""
    d = x.shape[-1]
    if not (0 < mem_dim < d):
        raise ValueError(f"coord split requires 0 < mem_dim < d; got mem_dim={mem_dim}, d={d}")
    a, b = x[..., :mem_dim], x[..., mem_dim:]
    x_mem = torch.cat([a, torch.zeros_like(b)], dim=-1)
    x_smt = torch.cat([torch.zeros_like(a), b], dim=-1)
    return x_mem, x_smt


class DualChannelTransformerBlock(nn.Module):
    """
    Pre-LN(x) -> ortho split -> mem: Linear(x_mem); smart: Pre-LN MHA + FFN on x_smt -> s.
    x_out = mem + s.
    If share_ortho_mem=True, pass ortho and linear_mem from SeqModelDualChannel (global basis + mem readout).
    """

    def __init__(
        self,
        dim: int,
        mem_dim: int,
        nhead: int,
        dim_feedforward: int,
        *,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        use_residual: bool = True,
        activation: str = "relu",
        share_ortho_mem: bool = False,
        mem_basis: str = "learned_ortho",
    ) -> None:
        super().__init__()
        self.share_ortho_mem = share_ortho_mem
        self.mem_basis = mem_basis
        self.mem_dim_coord = mem_dim
        if mem_basis not in ("learned_ortho", "coord_slice"):
            raise ValueError("mem_basis must be learned_ortho or coord_slice")
        self.norm_in = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        if share_ortho_mem:
            self.ortho = None  # supplied in forward (learned only)
            self.linear_mem = None
        elif mem_basis == "coord_slice":
            self.ortho = None
            self.linear_mem = nn.Linear(dim, dim)
        else:
            self.ortho = OrthogonalProjection(dim, mem_dim)
            self.linear_mem = nn.Linear(dim, dim)
        self.self_attn = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = F.relu if activation == "relu" else F.gelu
        self.use_residual = use_residual

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        *,
        ortho: OrthogonalProjection | None = None,
        linear_mem: nn.Linear | None = None,
        mem_identity: bool = False,
        no_mem_merge: bool = False,
        diag: bool = False,
        return_aux: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        if diag and return_aux:
            raise ValueError("diag and return_aux cannot both be True")
        if self.share_ortho_mem:
            assert linear_mem is not None
            lin_m = linear_mem
            ortho_m = ortho
            if self.mem_basis == "learned_ortho":
                assert ortho_m is not None
        else:
            ortho_m, lin_m = self.ortho, self.linear_mem  # type: ignore[assignment]
        x_n = self.norm_in(src)
        if self.mem_basis == "coord_slice":
            x_mem, x_smt = _coord_split_stream(x_n, self.mem_dim_coord)
        else:
            assert ortho_m is not None
            x_mem, x_smt = ortho_m(x_n)
        mem = x_mem if mem_identity else lin_m(x_mem)
        s = x_smt
        h = self.norm1(s)
        attn_out, attn_w = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        s = s + self.dropout1(attn_out)
        h2 = self.norm2(s)
        mid = self.linear1(h2)
        mid_act = self.act(mid)
        ff = self.linear2(self.dropout2(mid_act))
        s = s + self.dropout2(ff)
        if no_mem_merge:
            out = s
        elif self.use_residual:
            out = mem + s
        else:
            out = s
        if diag:
            return out, attn_w, mid, s
        if return_aux:
            aux = {
                "mem": mem,
                "s_smart": s,
                "src": src,
                "x_n": x_n,
                "x_mem": x_mem,
            }
            return out, attn_w, aux
        return out, attn_w


class SeqModelDualChannel(nn.Module):
    """Orthogonal mem + full Transformer-style smart path per layer; same nhead/ff as baseline Transformer."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        mem_dim: int,
        nhead: int,
        dim_feedforward: int | None = None,
        *,
        use_layernorm: bool = True,
        use_residual: bool = True,
        activation: str = "relu",
        global_dual_channel: bool = False,
        last_layer_mem_identity: bool = False,
        last_layer_no_mem: bool = False,
        mem_basis: str = "learned_ortho",
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * dim
        if mem_basis not in ("learned_ortho", "coord_slice"):
            raise ValueError("mem_basis must be learned_ortho or coord_slice")
        self.mem_basis = mem_basis
        self.global_dual_channel = global_dual_channel
        self.last_layer_mem_identity = last_layer_mem_identity
        self.last_layer_no_mem = last_layer_no_mem
        self.embed = nn.Embedding(vocab_size, dim)
        if global_dual_channel:
            if mem_basis == "learned_ortho":
                self.shared_ortho = OrthogonalProjection(dim, mem_dim)
            else:
                self.shared_ortho = None
            self.shared_linear_mem = nn.Linear(dim, dim)
        else:
            self.shared_ortho = None  # type: ignore[assignment]
            self.shared_linear_mem = None  # type: ignore[assignment]
        self.blocks = nn.ModuleList(
            [
                DualChannelTransformerBlock(
                    dim,
                    mem_dim,
                    nhead,
                    dim_feedforward,
                    dropout=0.0,
                    use_layernorm=use_layernorm,
                    use_residual=use_residual,
                    activation=activation,
                    share_ortho_mem=global_dual_channel,
                    mem_basis=mem_basis,
                )
                for _ in range(depth)
            ]
        )
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        *,
        return_dual_aux: bool = False,
        dual_aux_mem_target: str = "src",
        dual_aux_orthog_mode: str = "cos2",
    ) -> (
        tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]
    ):
        x = self.embed(token_ids)
        attns: list[torch.Tensor] = []
        n_blks = len(self.blocks)
        loss_mem = torch.zeros((), device=x.device, dtype=x.dtype)
        loss_orthog = torch.zeros((), device=x.device, dtype=x.dtype)
        mt = dual_aux_mem_target
        om = dual_aux_orthog_mode
        if self.global_dual_channel:
            o, lm = self.shared_ortho, self.shared_linear_mem
            for i, blk in enumerate(self.blocks):
                is_last = i == n_blks - 1
                no_mem = self.last_layer_no_mem and is_last
                if return_dual_aux:
                    x, aw, aux = blk(
                        x,
                        src_key_padding_mask,
                        ortho=o,
                        linear_mem=lm,
                        mem_identity=(
                            self.last_layer_mem_identity and is_last and not no_mem
                        ),
                        no_mem_merge=no_mem,
                        return_aux=True,
                    )
                    mem_b, s_b, src_b = aux["mem"], aux["s_smart"], aux["src"]
                    tgt = (
                        aux["x_n"].detach()
                        if mt == "x_n"
                        else src_b.detach()
                    )
                    loss_mem = loss_mem + _masked_mean_bt(
                        (mem_b - tgt).pow(2).mean(dim=-1), src_key_padding_mask
                    )
                    if om == "dot2":
                        loss_orthog = loss_orthog + _masked_mean_bt(
                            (mem_b * s_b).sum(dim=-1).pow(2),
                            src_key_padding_mask,
                        )
                    else:
                        dot = (mem_b * s_b).sum(dim=-1)
                        eps = 1e-8
                        nm = mem_b.pow(2).sum(dim=-1).clamp_min(eps)
                        ns = s_b.pow(2).sum(dim=-1).clamp_min(eps)
                        cos2 = dot.pow(2) / (nm * ns)
                        loss_orthog = loss_orthog + _masked_mean_bt(
                            cos2, src_key_padding_mask
                        )
                else:
                    x, aw = blk(
                        x,
                        src_key_padding_mask,
                        ortho=o,
                        linear_mem=lm,
                        mem_identity=(
                            self.last_layer_mem_identity and is_last and not no_mem
                        ),
                        no_mem_merge=no_mem,
                    )
                attns.append(aw)
        else:
            for i, blk in enumerate(self.blocks):
                is_last = i == n_blks - 1
                no_mem = self.last_layer_no_mem and is_last
                if return_dual_aux:
                    x, aw, aux = blk(
                        x,
                        src_key_padding_mask,
                        mem_identity=(
                            self.last_layer_mem_identity and is_last and not no_mem
                        ),
                        no_mem_merge=no_mem,
                        return_aux=True,
                    )
                    mem_b, s_b, src_b = aux["mem"], aux["s_smart"], aux["src"]
                    tgt = (
                        aux["x_n"].detach()
                        if mt == "x_n"
                        else src_b.detach()
                    )
                    loss_mem = loss_mem + _masked_mean_bt(
                        (mem_b - tgt).pow(2).mean(dim=-1), src_key_padding_mask
                    )
                    if om == "dot2":
                        loss_orthog = loss_orthog + _masked_mean_bt(
                            (mem_b * s_b).sum(dim=-1).pow(2),
                            src_key_padding_mask,
                        )
                    else:
                        dot = (mem_b * s_b).sum(dim=-1)
                        eps = 1e-8
                        nm = mem_b.pow(2).sum(dim=-1).clamp_min(eps)
                        ns = s_b.pow(2).sum(dim=-1).clamp_min(eps)
                        cos2 = dot.pow(2) / (nm * ns)
                        loss_orthog = loss_orthog + _masked_mean_bt(
                            cos2, src_key_padding_mask
                        )
                else:
                    x, aw = blk(
                        x,
                        src_key_padding_mask,
                        mem_identity=(
                            self.last_layer_mem_identity and is_last and not no_mem
                        ),
                        no_mem_merge=no_mem,
                    )
                attns.append(aw)
        logits = self.lm_head(x)
        if return_dual_aux:
            inv_l = 1.0 / float(n_blks)
            pack = {
                "loss_mem": loss_mem * inv_l,
                "loss_orthog": loss_orthog * inv_l,
            }
            return logits, attns, pack
        return logits, attns


class TransformerEncoderLayerAttn(nn.Module):
    """与 nn.TransformerEncoderLayer 同族，额外返回平均头注意力 (B,L,L)。"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(src)
        attn_out, attn_w = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        src = src + self.dropout1(attn_out)
        h2 = self.norm2(src)
        ff = self.linear2(self.dropout2(self.act(self.linear1(h2))))
        src = src + self.dropout2(ff)
        return src, attn_w


class SeqModelTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        nhead: int,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerAttn(
                    dim, nhead, dim_feedforward, dropout=0.0, activation="relu"
                )
                for _ in range(depth)
            ]
        )
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self, token_ids: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embed(token_ids)
        attns: list[torch.Tensor] = []
        for layer in self.layers:
            x, aw = layer(x, src_key_padding_mask=src_key_padding_mask)
            attns.append(aw)
        return self.lm_head(x), attns


class SeqModelDecayMemResidual(nn.Module):
    """
    衰减记忆残差：与标准 Transformer 同栈，每层块后对全维表征加 (decay_base**idx) * x_mem，
    x_mem 为嵌入输出 clone；无正交拆分、无末层 mem与 smart 分路相加，仅 lm_head(s)。
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        nhead: int,
        dim_feedforward: int | None = None,
        *,
        decay_base: float = 0.8,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * dim
        self.decay_base = float(decay_base)
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerAttn(
                    dim, nhead, dim_feedforward, dropout=0.0, activation=activation
                )
                for _ in range(depth)
            ]
        )
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embed(token_ids)
        x_mem = x.clone()
        s = x
        attns: list[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            s, aw = layer(s, src_key_padding_mask=src_key_padding_mask)
            s = s + (self.decay_base**idx) * x_mem
            attns.append(aw)
        return self.lm_head(s), attns


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def dual_mem_parameters(model: SeqModelDualChannel) -> list[nn.Parameter]:
    """OrthogonalProjection.W and linear_mem (shared or per-block) for Adam param groups."""
    params: list[nn.Parameter] = []
    if model.global_dual_channel:
        if (
            getattr(model, "mem_basis", "learned_ortho") == "learned_ortho"
            and model.shared_ortho is not None
        ):
            params.extend(list(model.shared_ortho.parameters()))
        if model.shared_linear_mem is not None:
            params.extend(list(model.shared_linear_mem.parameters()))
    else:
        for blk in model.blocks:
            if blk.ortho is not None:
                params.extend(list(blk.ortho.parameters()))
            if blk.linear_mem is not None:
                params.extend(list(blk.linear_mem.parameters()))
    return params


# ---------------------------------------------------------------------------
# Copy task data
# ---------------------------------------------------------------------------
def build_copy_batch(
    batch_size: int,
    n_min: int,
    n_max: int,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    max_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    *,
    fixed_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Delimiter copy: seq = [content_0..n-1 | DELIM | tail_0..n-1]，总长 2n+1（padding 前）。
    下标: 0..n-1 content；n = DELIM；n+1..2n = tail。监督 lab[n+1+i]=content[i]，切片 lab[n+1:2n+1]
    恰 n 项，与 tail 对齐（非整段错位）。content/tail ∈ [0,vocab_content)；pad_id=vocab_content, delim_id=vocab_content+1。
    """
    if vocab_content < 1:
        raise ValueError("vocab_content must be >= 1")
    if pad_id != vocab_content or delim_id != vocab_content + 1:
        raise ValueError(
            f"expect pad_id==vocab_content ({vocab_content}) and delim_id==vocab_content+1; "
            f"got pad_id={pad_id}, delim_id={delim_id}"
        )
    tokens_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for _ in range(batch_size):
        if fixed_n is not None:
            n = int(fixed_n)
        elif generator is not None:
            n = int(torch.randint(n_min, n_max + 1, (1,), generator=generator).item())
        else:
            n = int(torch.randint(n_min, n_max + 1, (1,)).item())
        if n < 1:
            raise ValueError("delimiter copy requires n >= 1")
        content = torch.randint(0, vocab_content, (n,), device=device)
        tail = torch.randint(0, vocab_content, (n,), device=device)
        seq = torch.cat(
            [
                content,
                torch.tensor([delim_id], device=device, dtype=torch.long),
                tail,
            ],
            dim=0,
        )
        assert int(seq.shape[0]) == 2 * n + 1
        lab = torch.full((seq.shape[0],), IGNORE_LABEL, dtype=torch.long, device=device)
        lab[n + 1 : 2 * n + 1] = content
        if __debug__:
            assert int((lab != IGNORE_LABEL).sum().item()) == n
            assert int(lab[n].item()) == IGNORE_LABEL
        # pad
        if seq.shape[0] < max_len:
            pad_n = max_len - seq.shape[0]
            seq = F.pad(seq, (0, pad_n), value=pad_id)
            lab = F.pad(lab, (0, pad_n), value=IGNORE_LABEL)
        tokens_list.append(seq)
        labels_list.append(lab)
    tokens = torch.stack(tokens_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    padding_mask = tokens == pad_id
    if __debug__:
        for b in range(tokens.shape[0]):
            m = labels[b] != IGNORE_LABEL
            assert m.any() and torch.all(tokens[b, m] < delim_id)
    return tokens, labels, padding_mask


def build_identity_copy_batch(
    batch_size: int,
    n_min: int,
    n_max: int,
    vocab_content: int,
    pad_id: int,
    max_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    *,
    fixed_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Native copy: each sample is length L ∈ [n_min, n_max] of i.i.d. tokens in [0, vocab_content).
    Labels equal tokens on content positions; padding uses IGNORE_LABEL.
    Padded to max_len (typically n_max)。
    """
    if vocab_content < 1:
        raise ValueError("vocab_content must be >= 1")
    if pad_id != vocab_content:
        raise ValueError(f"identity expects pad_id == vocab_content ({vocab_content})")
    tokens_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for _ in range(batch_size):
        if fixed_n is not None:
            L = int(fixed_n)
        elif generator is not None:
            L = int(torch.randint(n_min, n_max + 1, (1,), generator=generator).item())
        else:
            L = int(torch.randint(n_min, n_max + 1, (1,)).item())
        if L < 1:
            raise ValueError("identity copy requires L >= 1")
        seq = torch.randint(0, vocab_content, (L,), device=device)
        lab = seq.clone()
        if __debug__:
            assert int(seq.max().item()) < pad_id
        if L < max_len:
            pad_n = max_len - L
            seq = F.pad(seq, (0, pad_n), value=pad_id)
            lab = F.pad(lab, (0, pad_n), value=IGNORE_LABEL)
        tokens_list.append(seq)
        labels_list.append(lab)
    tokens = torch.stack(tokens_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    padding_mask = tokens == pad_id
    if __debug__:
        assert torch.all((labels != IGNORE_LABEL).sum(dim=1) > 0)
    return tokens, labels, padding_mask


def val_set_meta_dict(args: argparse.Namespace) -> dict[str, Any]:
    vc = int(args.vocab_content)
    vocab_size = vc + 1 if args.task == "identity" else vc + 2
    return {
        "version": 3,
        "task": args.task,
        "val_batch": args.val_batch,
        "val_n": "random" if args.val_random_n else args.val_n,
        "n_min": args.n_min,
        "n_max": args.n_max,
        "vocab_content": vc,
        "vocab_size": vocab_size,
        "val_set_seed": args.val_set_seed,
        "gap_min": int(args.gap_min) if args.task == "gap_copy" else None,
        "gap_max": int(args.gap_max) if args.task == "gap_copy" else None,
        "val_gap": (
            "random"
            if args.task == "gap_copy" and args.val_random_n
            else (int(args.val_gap) if args.task == "gap_copy" else None)
        ),
        "dup_tail": bool(args.task == "dup_tail_copy"),
    }


def meta_compatible(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    if stored.get("task", "delimiter") != current.get("task", "delimiter"):
        return False
    keys = (
        "version",
        "val_batch",
        "val_n",
        "n_min",
        "n_max",
        "vocab_content",
        "vocab_size",
        "val_set_seed",
        "gap_min",
        "gap_max",
        "val_gap",
        "dup_tail",
    )
    return all(stored.get(k) == current.get(k) for k in keys)


def save_fixed_val_set(
    path: Path,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    pad_mask: torch.Tensor,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tokens": tokens.cpu(),
            "labels": labels.cpu(),
            "pad_mask": pad_mask.cpu(),
            "meta": meta,
        },
        path,
    )


def load_fixed_val_set(
    path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    blob = torch.load(path, map_location=device)
    tokens = blob["tokens"].to(device)
    labels = blob["labels"].to(device)
    pad_mask = blob["pad_mask"].to(device)
    return tokens, labels, pad_mask, blob["meta"]


def ensure_fixed_val_set(
    args: argparse.Namespace,
    val_path: Path,
    max_len: int,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load cached val batch from val_path or build on CPU, save, then move to device."""
    meta = val_set_meta_dict(args)
    cpu = torch.device("cpu")
    if (
        val_path.is_file()
        and not args.regen_val_set
    ):
        t, lab, pm, old_meta = load_fixed_val_set(val_path, cpu)
        if meta_compatible(old_meta, meta):
            print(f"Loaded fixed val set from {val_path}")
            return t.to(device), lab.to(device), pm.to(device)
        print(f"Val set meta mismatch; regenerating {val_path}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.val_set_seed)
    if args.task == "identity":
        t, lab, pm = build_identity_copy_batch(
            args.val_batch,
            args.n_min,
            args.n_max,
            vocab_content,
            pad_id,
            max_len,
            cpu,
            generator=gen,
            fixed_n=None if args.val_random_n else args.val_n,
        )
    elif args.task == "gap_copy":
        t, lab, pm = build_gap_copy_batch(
            args.val_batch,
            args.n_min,
            args.n_max,
            int(args.gap_min),
            int(args.gap_max),
            vocab_content,
            pad_id,
            delim_id,
            max_len,
            cpu,
            gen,
            fixed_n=None if args.val_random_n else args.val_n,
            fixed_gap=None if args.val_random_n else int(args.val_gap),
        )
    elif args.task == "dup_tail_copy":
        t, lab, pm = build_dup_tail_copy_batch(
            args.val_batch,
            args.n_min,
            args.n_max,
            vocab_content,
            pad_id,
            delim_id,
            max_len,
            cpu,
            generator=gen,
            fixed_n=None if args.val_random_n else args.val_n,
        )
    else:
        t, lab, pm = build_copy_batch(
            args.val_batch,
            args.n_min,
            args.n_max,
            vocab_content,
            pad_id,
            delim_id,
            max_len,
            cpu,
            generator=gen,
            fixed_n=None if args.val_random_n else args.val_n,
        )
    save_fixed_val_set(val_path, t, lab, pm, meta)
    print(f"Saved fixed val set ({args.val_batch} seqs) to {val_path}")
    return t.to(device), lab.to(device), pm.to(device)


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits (B,L,V), labels (B,L) with IGNORE_LABEL skipped."""
    B, L, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * L, V),
        labels.reshape(B * L),
        ignore_index=IGNORE_LABEL,
        reduction="mean",
    )


def _masked_mean_bt(
    x_bt: torch.Tensor, pad_mask_bt: torch.Tensor | None
) -> torch.Tensor:
    """Mean over batch×length; pad_mask_bt True = padding position (ignored)."""
    if pad_mask_bt is None:
        return x_bt.mean()
    valid = (~pad_mask_bt).to(dtype=x_bt.dtype)
    return (x_bt * valid).sum() / valid.sum().clamp_min(1.0)


def accuracy_on_labeled(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float, float]:
    """整体、复制段前半、复制段后半（按每个样本复制段切半）。"""
    pred = logits.argmax(dim=-1)
    B = labels.shape[0]
    all_acc: list[float] = []
    first_acc: list[float] = []
    second_acc: list[float] = []
    for b in range(B):
        lab = labels[b]
        m = (lab != IGNORE_LABEL).nonzero(as_tuple=False).flatten()
        if m.numel() == 0:
            continue
        correct = pred[b, m] == lab[m]
        all_acc.append(float(correct.float().mean().item()))
        n = correct.shape[0]
        h = n // 2
        if h > 0:
            first_acc.append(float(correct[:h].float().mean().item()))
        if n > h:
            second_acc.append(float(correct[h:].float().mean().item()))

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return _mean(all_acc), _mean(first_acc), _mean(second_acc)


def levenshtein_distance(a: list[int], b: list[int]) -> int:
    """Classic edit distance; O(len(a)*len(b)), sufficient for copy length <= n_max."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[lb]


def copy_segment_em_and_norm_levenshtein(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    """delimiter/identity: labeled positions only. EM = mean(all positions match); norm_lev = mean(edit/n)."""
    pred = logits.argmax(dim=-1)
    B = labels.shape[0]
    ems: list[float] = []
    nlevs: list[float] = []
    for b in range(B):
        lab = labels[b]
        m = (lab != IGNORE_LABEL).nonzero(as_tuple=False).flatten()
        if m.numel() == 0:
            continue
        p = pred[b, m].detach().cpu().tolist()
        g = lab[m].detach().cpu().tolist()
        n = len(g)
        ems.append(1.0 if p == g else 0.0)
        nlevs.append(levenshtein_distance(p, g) / max(1, n))
    return (
        float(np.mean(ems)) if ems else float("nan"),
        float(np.mean(nlevs)) if nlevs else float("nan"),
    )


# ---------------------------------------------------------------------------
# Diagnostics: hidden chain, R_scat, spectra, attention entropy
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_dual_channel_diagnostics(
    model: SeqModelDualChannel,
    tokens: torch.Tensor,
    padding_mask: torch.Tensor,
    k_list: list[int],
) -> dict[str, Any]:
    model.eval()
    x = model.embed(tokens)
    hidden: list[torch.Tensor] = [x]
    aux: list[tuple[torch.Tensor, torch.Tensor]] = []
    layer_attns: list[torch.Tensor] = []
    h = x
    n_blks = len(model.blocks)
    for i, blk in enumerate(model.blocks):
        is_last = i == n_blks - 1
        no_mem = model.last_layer_no_mem and is_last
        mid_id = model.last_layer_mem_identity and is_last and not no_mem
        if model.global_dual_channel:
            out = blk(
                h,
                padding_mask,
                diag=True,
                ortho=model.shared_ortho,
                linear_mem=model.shared_linear_mem,
                mem_identity=mid_id,
                no_mem_merge=no_mem,
            )
        else:
            out = blk(
                h,
                padding_mask,
                diag=True,
                mem_identity=mid_id,
                no_mem_merge=no_mem,
            )
        assert isinstance(out, tuple) and len(out) == 4
        h, attn_w, mid, smt_out = out
        hidden.append(h)
        aux.append((mid, smt_out))
        layer_attns.append(attn_w)
    logits = model.lm_head(h)

    k_ref = k_list[0] if k_list else None
    nuc_full, c_full, c_k = nuclear_chain_from_hidden(hidden, k_trunc=k_ref)

    r_scat: dict[str, list[float]] = {f"k{k}": [] for k in k_list}
    for mid, smt in aux:
        for k in k_list:
            ratios = []
            for bi in range(mid.shape[0]):
                _, _, _, pre_k, _ = analyze_matrix(mid[bi], k_trunc=k)
                _, _, _, post_k, _ = analyze_matrix(smt[bi], k_trunc=k)
                ratios.append(post_k / (pre_k + 1e-12))
            r_scat[f"k{k}"].append(float(np.mean(ratios)))

    last = hidden[-1][0]
    s = singular_values_np(last)
    entropies = [attention_row_entropy(a) for a in layer_attns]
    attn_sample0 = [a[0].detach().float().cpu().numpy() for a in layer_attns]

    return {
        "nuclear_layers": nuc_full,
        "c_ell_full": [c for c in c_full if c is not None],
        "c_ell_k": [c for c in c_k if c is not None],
        "r_scat_by_layer": r_scat,
        "attn_entropy": entropies,
        "attn_matrices_sample0": attn_sample0,
        "singular_values_example": s[: min(32, s.size)].tolist(),
        "logits": logits,
    }


@torch.no_grad()
def collect_transformer_diagnostics(
    model: SeqModelTransformer,
    tokens: torch.Tensor,
    padding_mask: torch.Tensor,
    k_list: list[int],
) -> dict[str, Any]:
    model.eval()
    x = model.embed(tokens)
    hidden: list[torch.Tensor] = [x]
    h = x
    key_padding = padding_mask
    layer_attns: list[torch.Tensor] = []
    for layer in model.layers:
        h, aw = layer(h, src_key_padding_mask=key_padding)
        hidden.append(h)
        layer_attns.append(aw)
    logits = model.lm_head(h)

    k_ref = k_list[0] if k_list else None
    nuc_full, c_full, c_k = nuclear_chain_from_hidden(hidden, k_trunc=k_ref)

    entropies = [attention_row_entropy(a) for a in layer_attns]
    last = hidden[-1][0]
    s = singular_values_np(last)
    attn_sample0 = [a[0].detach().float().cpu().numpy() for a in layer_attns]

    return {
        "nuclear_layers": nuc_full,
        "c_ell_full": [c for c in c_full if c is not None],
        "c_ell_k": [c for c in c_k if c is not None],
        "attn_entropy": entropies,
        "attn_matrices_sample0": attn_sample0,
        "singular_values_example": s[: min(32, s.size)].tolist(),
        "logits": logits,
    }


@torch.no_grad()
def collect_decay_mem_diagnostics(
    model: SeqModelDecayMemResidual,
    tokens: torch.Tensor,
    padding_mask: torch.Tensor,
    k_list: list[int],
) -> dict[str, Any]:
    """与 Transformer 诊断同结构；hidden 含每层 Transformer 后再加衰减记忆残差后的状态。"""
    model.eval()
    x = model.embed(tokens)
    x_mem = x.clone()
    hidden: list[torch.Tensor] = [x]
    s = x
    key_padding = padding_mask
    layer_attns: list[torch.Tensor] = []
    for idx, layer in enumerate(model.layers):
        s, aw = layer(s, src_key_padding_mask=key_padding)
        s = s + (model.decay_base**idx) * x_mem
        hidden.append(s)
        layer_attns.append(aw)
    logits = model.lm_head(s)

    k_ref = k_list[0] if k_list else None
    nuc_full, c_full, c_k = nuclear_chain_from_hidden(hidden, k_trunc=k_ref)

    entropies = [attention_row_entropy(a) for a in layer_attns]
    last = hidden[-1][0]
    sv = singular_values_np(last)
    attn_sample0 = [a[0].detach().float().cpu().numpy() for a in layer_attns]

    return {
        "nuclear_layers": nuc_full,
        "c_ell_full": [c for c in c_full if c is not None],
        "c_ell_k": [c for c in c_k if c is not None],
        "r_scat_by_layer": {},
        "attn_entropy": entropies,
        "attn_matrices_sample0": attn_sample0,
        "singular_values_example": sv[: min(32, sv.size)].tolist(),
        "logits": logits,
    }


def empty_transformer_diagnostics() -> dict[str, Any]:
    """Placeholder when --dual-only skips Transformer train/eval."""
    return {
        "nuclear_layers": [],
        "c_ell_full": [],
        "attn_entropy": [],
        "singular_values_example": [],
        "attn_matrices_sample0": [],
        "r_scat_by_layer": {},
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Training / eval loop
# ---------------------------------------------------------------------------
def train_run(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    model_name: str,
    model: nn.Module,
    device: torch.device,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    max_len: int,
    val_tokens: torch.Tensor,
    val_labels: torch.Tensor,
    val_pad_mask: torch.Tensor,
) -> list[dict]:
    decay_base_cell: str | float = ""
    if model_name == "dual_decay_mem" and hasattr(model, "decay_base"):
        decay_base_cell = float(model.decay_base)  # type: ignore[arg-type]

    if model_name == "dual_ortho_mem" and isinstance(
        model, SeqModelDualChannel
    ) and float(args.dual_mem_lr_scale) != 1.0:
        mem_ps = dual_mem_parameters(model)
        mem_ids = {id(p) for p in mem_ps}
        base_ps = [p for p in model.parameters() if id(p) not in mem_ids]
        opt = torch.optim.Adam(
            [
                {"params": base_ps, "lr": args.lr},
                {
                    "params": mem_ps,
                    "lr": args.lr * float(args.dual_mem_lr_scale),
                },
            ]
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rows: list[dict] = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 777)

    aux_depth_mult = float(args.depth) if args.dual_aux_scale_by_depth else 1.0
    w_mem = float(args.dual_aux_mem_weight) * aux_depth_mult
    w_orth = float(args.dual_aux_orthog_weight) * aux_depth_mult
    use_dual_aux_const = (
        model_name == "dual_ortho_mem"
        and (
            w_mem > 0.0
            or w_orth > 0.0
        )
    )

    for step in range(args.steps + 1):
        model.train()
        if args.task == "identity":
            tokens, labels, pad_mask = build_identity_copy_batch(
                args.batch,
                args.n_min,
                args.n_max,
                vocab_content,
                pad_id,
                max_len,
                device,
                generator=gen,
            )
        elif args.task == "gap_copy":
            tokens, labels, pad_mask = build_gap_copy_batch(
                args.batch,
                args.n_min,
                args.n_max,
                int(args.gap_min),
                int(args.gap_max),
                vocab_content,
                pad_id,
                delim_id,
                max_len,
                device,
                gen,
            )
        elif args.task == "dup_tail_copy":
            tokens, labels, pad_mask = build_dup_tail_copy_batch(
                args.batch,
                args.n_min,
                args.n_max,
                vocab_content,
                pad_id,
                delim_id,
                max_len,
                device,
                generator=gen,
            )
        else:
            tokens, labels, pad_mask = build_copy_batch(
                args.batch,
                args.n_min,
                args.n_max,
                vocab_content,
                pad_id,
                delim_id,
                max_len,
                device,
                generator=gen,
            )
        use_dual_aux = use_dual_aux_const and isinstance(model, SeqModelDualChannel)
        main_ce_w = float(args.dual_main_ce_weight)

        opt.zero_grad()
        if use_dual_aux:
            assert isinstance(model, SeqModelDualChannel)
            logits, _, pack = model(
                tokens,
                src_key_padding_mask=pad_mask,
                return_dual_aux=True,
                dual_aux_mem_target=args.dual_aux_mem_target,
                dual_aux_orthog_mode=args.dual_aux_orthog_mode,
            )  # type: ignore[misc,assignment]
            ce = masked_ce_loss(logits, labels)
            aux_sum = torch.zeros((), device=ce.device, dtype=ce.dtype)
            if w_mem > 0.0:
                aux_sum = aux_sum + w_mem * pack["loss_mem"]
            if w_orth > 0.0:
                aux_sum = aux_sum + w_orth * pack["loss_orthog"]
            aux_mem_f = float(pack["loss_mem"].detach().item())
            aux_orth_f = float(pack["loss_orthog"].detach().item())
            loss = main_ce_w * ce + aux_sum
            loss.backward()
        else:
            logits, _ = model(tokens, src_key_padding_mask=pad_mask)  # type: ignore[misc]
            ce = masked_ce_loss(logits, labels)
            loss = main_ce_w * ce
            loss.backward()
            aux_mem_f = aux_orth_f = 0.0
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vt, vl, vpm = val_tokens, val_labels, val_pad_mask
                v_logits, _ = model(vt, src_key_padding_mask=vpm)  # type: ignore[misc]
                vloss = masked_ce_loss(v_logits, vl).item()
                acc, acc1, acc2 = accuracy_on_labeled(v_logits, vl)
                _, v_nlev = copy_segment_em_and_norm_levenshtein(v_logits, vl)
            ce_f = float(ce.item())
            tot_f = float(loss.item())
            row_tr: dict[str, str | float | int | bool] = {
                "model": model_name,
                "step": step,
                "train_loss": ce_f,
                "train_total": tot_f,
                "train_aux_mem": aux_mem_f,
                "train_aux_orthog": aux_orth_f,
                "val_loss": vloss,
                "val_acc_all": acc,
                "val_acc_copy_first_half": acc1,
                "val_acc_copy_second_half": acc2,
                "val_copy_norm_lev": v_nlev,
                "seed": args.seed,
                "mem_dim": args.mem_dim,
                "n_max": args.n_max,
                "val_batch": args.val_batch,
                "val_n": "random" if args.val_random_n else args.val_n,
                "global_dual_channel": (
                    bool(args.global_dual_channel)
                    if model_name == "dual_ortho_mem"
                    else False
                ),
                "dual_mem_lr_scale": (
                    float(args.dual_mem_lr_scale)
                    if model_name == "dual_ortho_mem"
                    else 1.0
                ),
                "dual_last_layer_mem_identity": (
                    bool(args.dual_last_layer_mem_identity)
                    if model_name == "dual_ortho_mem"
                    else False
                ),
                "dual_last_layer_no_mem": (
                    bool(args.dual_last_layer_no_mem)
                    if model_name == "dual_ortho_mem"
                    else False
                ),
                "dual_aux_mem_weight": (
                    float(getattr(args, "dual_aux_mem_weight", 0.0))
                    if model_name == "dual_ortho_mem"
                    else 0.0
                ),
                "dual_aux_orthog_weight": (
                    float(getattr(args, "dual_aux_orthog_weight", 0.0))
                    if model_name == "dual_ortho_mem"
                    else 0.0
                ),
                "dual_aux_scale_by_depth": (
                    bool(args.dual_aux_scale_by_depth)
                    if model_name == "dual_ortho_mem"
                    else False
                ),
                "dual_aux_depth_multiplier": (
                    float(aux_depth_mult) if model_name == "dual_ortho_mem" else 1.0
                ),
                "dual_aux_mem_effective": (
                    float(w_mem) if model_name == "dual_ortho_mem" else 0.0
                ),
                "dual_aux_orthog_effective": (
                    float(w_orth) if model_name == "dual_ortho_mem" else 0.0
                ),
                "dual_aux_mem_target": (
                    str(args.dual_aux_mem_target)
                    if model_name == "dual_ortho_mem"
                    else ""
                ),
                "dual_aux_orthog_mode": (
                    str(args.dual_aux_orthog_mode)
                    if model_name == "dual_ortho_mem"
                    else ""
                ),
                "dual_main_ce_weight": float(main_ce_w),
                "dual_mem_only_aux": False,
                "decay_mem_base": decay_base_cell,
            }
            rows.append(row_tr)
            if use_dual_aux:
                extra = (
                    f"(aux m={aux_mem_f:.4f}*{w_mem:.3g} o={aux_orth_f:.4f}*{w_orth:.3g}) "
                )
                print(
                    f"[{model_name}] step {step:5d} | train_ce {ce_f:.4f} | train_tot {tot_f:.4f} "
                    f"{extra}| val_ce {vloss:.4f} | "
                    f"acc {acc:.3f} |1st {acc1:.3f} | 2nd {acc2:.3f} | nLev {v_nlev:.3f}"
                )
            else:
                print(
                    f"[{model_name}] step {step:5d} | train_ce {ce_f:.4f} | val_ce {vloss:.4f} | "
                    f"acc {acc:.3f} |1st {acc1:.3f} | 2nd {acc2:.3f} | "
                    f"nLev {v_nlev:.3f}"
                )
    save_csv(out_dir / f"training_{model_name}_seed{args.seed}.csv", rows)
    return rows


def plot_loss_and_acc(rows_by_model: dict[str, list[dict]], out_dir: Path) -> None:
    """Train/Val CE 与准确率；准确率轴按数据范围缩放，避免 [0,1] 下早期 ~5% 看起来像空白。
    点数 =每 log_every 步记录一次 + 最后一步；若仅见 step 0..2，多半是 --steps 2 的 smoke 而非完整训练。"""
    apply_paper_style()
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    ax_tr, ax_val_lin, ax_val_log, ax_acc = axes
    n_max_pts = max(len(r) for r in rows_by_model.values())
    use_markers = n_max_pts <= 40
    mrk = "o" if use_markers else None
    ms = 4 if use_markers else 0

    for name, rows in sorted(rows_by_model.items()):
        xs = [r["step"] for r in rows]
        ax_tr.plot(
            xs,
            [r["train_loss"] for r in rows],
            label=name,
            marker=mrk,
            markersize=ms,
            lw=1.2,
        )
        ax_val_lin.plot(
            xs,
            [r["val_loss"] for r in rows],
            label=name,
            marker=mrk,
            markersize=ms,
            lw=1.2,
        )
        ax_val_log.plot(
            xs,
            [r["val_loss"] for r in rows],
            label=name,
            marker=mrk,
            markersize=ms,
            lw=1.2,
        )
        ax_acc.plot(
            xs,
            [r["val_acc_all"] for r in rows],
            label=name,
            marker=mrk,
            markersize=ms,
            lw=1.2,
        )

    ax_tr.set_ylabel("Train CE")
    ax_tr.set_title("Training / validation (single run, all logged steps)")
    ax_tr.legend(fontsize=7, ncol=2, loc="upper right")
    ax_tr.grid(True, ls=":", alpha=0.45)

    ax_val_lin.set_ylabel("Val CE (linear)")
    ax_val_lin.grid(True, ls=":", alpha=0.45)

    ax_val_log.set_ylabel("Val CE (log)")
    ax_val_log.set_yscale("log")
    ax_val_log.grid(True, which="both", ls=":", alpha=0.45)

    ax_acc.set_ylabel("Val acc (all copy positions)")
    ax_acc.set_xlabel("Step")
    ax_acc.legend(fontsize=7, loc="best")
    ax_acc.grid(True, ls=":", alpha=0.45)
    acc_vals: list[float] = []
    for rows in rows_by_model.values():
        for r in rows:
            acc_vals.append(float(r["val_acc_all"]))
    finite = [v for v in acc_vals if np.isfinite(v)]
    if finite:
        lo, hi = min(finite), max(finite)
        pad = max(0.02, (hi - lo) * 0.25, hi * 0.1)
        y_lo = max(0.0, lo - pad)
        y_hi = hi + pad
        # Plateau at acc=1.0 on ylim top=1.0 clips against the upper spine; add slight headroom.
        if y_hi > 1.0:
            y_hi = min(1.05, y_hi)
        else:
            y_hi = min(1.0, y_hi)
        ax_acc.set_ylim(y_lo, y_hi)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_script8_loss_acc.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure_script8_loss_acc.pdf", bbox_inches="tight")
    plt.close(fig)


def load_training_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_multi_seed_training_curves(
    root_out: Path,
    *,
    title_note: str = "",
) -> None:
    """
    Read seed_*/training_{dual_ortho_mem,transformer}_seed*.csv and plot mean ± std over seeds (same step grid). Writes figure_script8_loss_acc_multi_seed.{png,pdf} under root_out.
    """
    seed_dirs = sorted(
        p for p in root_out.iterdir() if p.is_dir() and p.name.startswith("seed_")
    )
    if not seed_dirs:
        raise FileNotFoundError(f"no seed_* subdirs under {root_out}")

    dual_candidates = ("dual_ortho_mem", "dual_decay_mem")
    dual_key: str | None = None
    for sd in seed_dirs:
        found: str | None = None
        for dk in dual_candidates:
            paths = sorted(sd.glob(f"training_{dk}_seed*.csv"))
            if paths:
                found = dk
                break
        if found is None:
            raise FileNotFoundError(
                f"no training_dual_ortho_mem_seed*.csv or training_dual_decay_mem_seed*.csv in {sd}"
            )
        if dual_key is None:
            dual_key = found
        elif dual_key != found:
            raise ValueError(
                "inconsistent dual architecture across seed_* (mix of ortho and decay-mem)"
            )

    assert dual_key is not None
    models = [dual_key, "transformer"]
    series: dict[str, list[list[dict[str, str]]]] = {m: [] for m in models}
    for sd in seed_dirs:
        for m in models:
            paths = sorted(sd.glob(f"training_{m}_seed*.csv"))
            if not paths:
                continue
            series[m].append(load_training_csv(paths[0]))

    if not series[dual_key]:
        raise FileNotFoundError(
            f"missing training_{dual_key}_seed*.csv under {root_out}/seed_*"
        )

    n_seeds = len(series[dual_key])
    models_plotted = [m for m in models if series[m]]
    for m in models_plotted:
        if len(series[m]) != n_seeds:
            raise ValueError(
                f"expected {n_seeds} seeds for {m}, got {len(series[m])}"
            )

    ref_steps = [int(r["step"]) for r in series[dual_key][0]]
    for m in models_plotted:
        for idx, rows in enumerate(series[m]):
            steps_i = [int(r["step"]) for r in rows]
            if steps_i != ref_steps:
                raise ValueError(
                    f"{m}: seed run {idx} has step list != first dual seed (check --steps / --log-every)"
                )

    def stack_metric(model: str, key: str) -> np.ndarray:
        return np.array(
            [[float(r[key]) for r in rows] for rows in series[model]], dtype=float
        )

    apply_paper_style()
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    ax_tr, ax_val_lin, ax_val_log, ax_acc = axes
    s_arr = np.array(ref_steps, dtype=float)

    colors = {
        "dual_ortho_mem": "C0",
        "dual_decay_mem": "C0",
        "transformer": "C1",
    }
    labels_short = {
        "dual_ortho_mem": "Dual",
        "dual_decay_mem": "Decay-dual",
        "transformer": "Transformer",
    }

    for m in models_plotted:
        Yt = stack_metric(m, "train_loss")
        Yv = stack_metric(m, "val_loss")
        Ya = stack_metric(m, "val_acc_all")
        ddof = 1 if Yt.shape[0] > 1 else 0
        mt, st = Yt.mean(0), Yt.std(0, ddof=ddof)
        mv, sv = Yv.mean(0), Yv.std(0, ddof=ddof)
        ma, sa = Ya.mean(0), Ya.std(0, ddof=ddof)
        c = colors[m]
        ls = labels_short[m]
        ax_tr.plot(s_arr, mt, label=f"{ls} (mean)", color=c, lw=1.5)
        ax_tr.fill_between(s_arr, mt - st, mt + st, color=c, alpha=0.22)
        ax_val_lin.plot(s_arr, mv, label=f"{ls} (mean)", color=c, lw=1.5)
        ax_val_lin.fill_between(s_arr, mv - sv, mv + sv, color=c, alpha=0.22)
        ax_val_log.plot(s_arr, mv, label=f"{ls} (mean)", color=c, lw=1.5)
        ax_val_log.fill_between(s_arr, mv - sv, mv + sv, color=c, alpha=0.22)
        ax_acc.plot(s_arr, ma, label=f"{ls} (mean)", color=c, lw=1.5)
        ax_acc.fill_between(s_arr, ma - sa, ma + sa, color=c, alpha=0.22)

    ax_tr.set_ylabel("Train CE")
    subt = f"Multi-seed mean ± std (N={n_seeds})"
    if title_note:
        subt = f"{subt} — {title_note}"
    ax_tr.set_title(subt)
    ax_tr.legend(fontsize=7, ncol=2, loc="upper right")
    ax_tr.grid(True, ls=":", alpha=0.45)

    ax_val_lin.set_ylabel("Val CE (linear)")
    ax_val_lin.grid(True, ls=":", alpha=0.45)

    ax_val_log.set_ylabel("Val CE (log)")
    ax_val_log.set_yscale("log")
    ax_val_log.grid(True, which="both", ls=":", alpha=0.45)

    ax_acc.set_ylabel("Val acc (labeled positions)")
    ax_acc.set_xlabel("Step")
    ax_acc.legend(fontsize=7, loc="best")
    ax_acc.grid(True, ls=":", alpha=0.45)
    acc_vals: list[float] = []
    for m in models_plotted:
        Ya = stack_metric(m, "val_acc_all")
        acc_vals.extend(Ya.mean(0).tolist())
    finite = [v for v in acc_vals if np.isfinite(v)]
    if finite:
        lo, hi = min(finite), max(finite)
        pad = max(0.02, (hi - lo) * 0.25, hi * 0.1)
        y_lo = max(0.0, lo - pad)
        y_hi = hi + pad
        # Plateau at acc=1.0 on ylim top=1.0 clips against the upper spine; add slight headroom.
        if y_hi > 1.0:
            y_hi = min(1.05, y_hi)
        else:
            y_hi = min(1.0, y_hi)
        ax_acc.set_ylim(y_lo, y_hi)

    ax_acc.set_xlim(float(s_arr.min()), float(s_arr.max()))
    fig.tight_layout()
    root_out.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        root_out / "figure_script8_loss_acc_multi_seed.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(root_out / "figure_script8_loss_acc_multi_seed.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_c_ell_bars(
    records: list[dict], out_dir: Path, title_suffix: str = ""
) -> None:
    apply_paper_style()
    names = sorted({r["model"] for r in records})
    ells = sorted({int(r["ell"]) for r in records})
    if not ells:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    width = min(0.28, 0.8 / max(1, len(names)))
    x_base = np.arange(len(ells), dtype=float)
    for i, name in enumerate(names):
        ys = []
        for ell in ells:
            row = next((r for r in records if r["model"] == name and int(r["ell"]) == ell), None)
            ys.append(float(row["c_ell_full"]) if row and row["c_ell_full"] is not None else float("nan"))
        ax.bar(x_base + i * width, ys, width=width, label=name)
    ax.set_xticks(x_base + width * (len(names) - 1) / 2)
    ax.set_xticklabels([str(e) for e in ells])
    ax.set_ylabel(r"$c_\ell = N_{\ell+1}/N_{\ell}$ (full nuclear)")
    ax.set_xlabel(r"Layer index $\ell$")
    ax.set_title("Inter-layer nuclear ratios (eval snapshot)" + title_suffix)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", ls=":", alpha=0.45)
    fig.savefig(out_dir / "figure_script8_c_ell.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_r_scat(r_scat_rows: list[dict], out_dir: Path) -> None:
    if not r_scat_rows:
        return
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
    layers = sorted({r["layer"] for r in r_scat_rows})
    ks = sorted({r["k"] for r in r_scat_rows})
    for k in ks:
        ys = [
            float(np.mean([r["r_scat_mean"] for r in r_scat_rows if r["layer"] == ell and r["k"] == k]))
            for ell in layers
        ]
        ax.plot(layers, ys, marker="o", label=rf"$k={k}$")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Block layer (smart path)")
    ax.set_ylabel(r"$R_{\mathrm{scat}}(k)$")
    ax.set_title(r"Smart path: post-FFN vs pre-act ($d\to 4d$) structural norm ratio")
    ax.legend(fontsize=7)
    ax.grid(True, ls=":", alpha=0.45)
    fig.savefig(out_dir / "figure_script8_r_scat.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attn_entropy(rows: list[dict], out_dir: Path) -> None:
    """Mean row entropy vs layer, one curve per model; values also in metrics_report."""
    if not rows:
        return
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    models = sorted({str(r["model"]) for r in rows})
    for mname in models:
        sub = [r for r in rows if str(r["model"]) == mname]
        by_layer: dict[int, list[float]] = {}
        for r in sub:
            by_layer.setdefault(int(r["layer"]), []).append(float(r["entropy"]))
        layers = sorted(by_layer)
        means = [float(np.mean(by_layer[l])) for l in layers]
        ax.plot(layers, means, marker="o", label=mname)
        for lx, my in zip(layers, means):
            ax.annotate(
                f"{my:.2f}",
                (lx, my),
                textcoords="offset points",
                xytext=(2, 4),
                fontsize=6,
            )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean row entropy (nats)")
    ax.set_title("Self-attention entropy (dual smart path vs full Transformer)")
    ax.grid(True, ls=":", alpha=0.45)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_script8_attn_entropy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attention_matrices_2d(
    attn_matrices: list[np.ndarray],
    out_dir: Path,
    *,
    titles: list[str] | None = None,
    fig_stem: str = "figure_script8_attn_heatmap",
    suptitle: str = "Self-attention weights (sample 0, averaged heads)",
) -> None:
    """imshow (L,L) attention for each layer; script1-style 2D panels."""
    if not attn_matrices:
        return
    apply_paper_style()
    n = len(attn_matrices)
    gmax = max(float(np.asarray(M, dtype=float).max()) for M in attn_matrices)
    vmax = max(1e-8, gmax)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 1, 3.8), squeeze=False)
    axes_flat = axes[0]
    im = None
    for i, ax in enumerate(axes_flat):
        M = np.asarray(attn_matrices[i], dtype=float)
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
        ttl = titles[i] if titles and i < len(titles) else f"Layer {i}"
        ax.set_title(ttl)
        ax.set_xlabel("key pos")
        ax.set_ylabel("query pos")
    if im is not None:
        fig.colorbar(im, ax=axes_flat.tolist(), shrink=0.82, label="Attention mass")
    fig.suptitle(suptitle, y=1.02)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.18, wspace=0.35)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{fig_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{fig_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_row_entropy_heatmap(
    attn_matrix: np.ndarray,
    out_dir: Path,
    *,
    fig_stem: str = "figure_script8_attn_row_entropy_map",
    title: str = "Per-row attention entropy H_i (layer 0, sample 0)",
) -> None:
    """Per-query attention entropy H_i as a 1-row heatmap."""
    attn_matrix = np.asarray(attn_matrix, dtype=float)
    eps = 1e-10
    row_ent = -(attn_matrix * np.log(np.clip(attn_matrix, eps, 1.0))).sum(axis=-1)
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8, 2.2))
    im = ax.imshow(row_ent[np.newaxis, :], aspect="auto", cmap="viridis")
    ax.set_yticks([])
    ax.set_xlabel("Query position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85, label="nats")
    fig.tight_layout()
    fig.savefig(out_dir / f"{fig_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_metrics_report(
    out_dir: Path,
    *,
    seed: int,
    task: str,
    rows_dual: list[dict],
    rows_tfmr: list[dict],
    dd: dict[str, Any],
    td: dict[str, Any],
    k_list: list[int],
    first_hits_dual: dict[str, str] | None = None,
    first_hits_tfmr: dict[str, str] | None = None,
) -> None:
    """终端打印并写入 metrics_report_seed.txt。"""
    lines: list[str] = []
    dual_model_key = (
        rows_dual[0].get("model", "dual_ortho_mem") if rows_dual else "dual_ortho_mem"
    )
    lines.append("=== Script8 metrics report ===")
    lines.append(
        f"Compare: {dual_model_key} vs transformer (same task, hyperparams, nhead,4d FFN)."
    )
    lines.append(f"seed={seed}")
    if task == "identity":
        lines.append(
            "task=identity (native copy): predict the same token at each non-pad position; "
            "L ~ Uniform[n_min,n_max], padded to n_max."
        )
    else:
        lines.append(
            "task=delimiter: [content n | DELIM | tail n] (len 2n+1), CE on tail positions = copy content."
        )
    if rows_dual:
        vr = rows_dual[0]
        lines.append(
            f"val: cached batch size={vr.get('val_batch')}, "
            f"n={vr.get('val_n')!r} (same tensors every eval step; shared dual vs transformer)."
        )
        dn = vr.get("model", "dual_ortho_mem")
        if dn == "dual_decay_mem":
            lines.append(
                "dual: decay memory residual — after each Transformer layer, "
                f"h += (decay_base**idx) * embed; decay_base={vr.get('decay_mem_base', '')}."
            )
        elif vr.get("global_dual_channel"):
            lines.append(
                "dual: GLOBAL channel — single OrthogonalProjection + linear_mem for all layers."
            )
        else:
            lines.append(
                "dual: per-layer OrthogonalProjection + linear_mem (default)."
            )
        if vr.get("dual_last_layer_no_mem"):
            lines.append(
                "dual: last layer — memory branch not merged (output from smart path only)."
            )
        lines.append("")

    def final_block(name: str, rows: list[dict]) -> None:
        last = rows[-1]
        lines.append(f"[{name}] final @ step {last['step']}")
        lines.append(f"  train_ce={last['train_loss']:.6f}")
        lines.append(f"  val_ce={last['val_loss']:.6f}")
        lines.append(f"  val_acc_all={last['val_acc_all']:.6f}")
        lines.append(f"  val_acc_copy_1st_half={last['val_acc_copy_first_half']:.6f}")
        lines.append(f"  val_acc_copy_2nd_half={last['val_acc_copy_second_half']:.6f}")
        if "val_copy_norm_lev" in last:
            lines.append(
                f"  val_copy_norm_lev={float(last['val_copy_norm_lev']):.6f}"
            )
        lines.append("")

    final_block(dual_model_key, rows_dual)
    if rows_tfmr:
        final_block("transformer", rows_tfmr)
    else:
        lines.append("[transformer] (not trained — --dual-only)")
        lines.append("")

    if first_hits_dual or first_hits_tfmr:
        lines.append(
            "[Time-to-accuracy] first logged step where val_acc_all >= threshold "
            "(— = never in this run)."
        )
        keys = sorted(set((first_hits_dual or {}).keys()) | set((first_hits_tfmr or {}).keys()))
        for key in keys:
            dv = (first_hits_dual or {}).get(key, "")
            tv = (first_hits_tfmr or {}).get(key, "")
            lines.append(
                f"  {key}: dual={dv or '—'}, transformer={tv or '—'}"
            )
        lines.append("")

    lines.append(f"[{dual_model_key}] nuclear norm N_l (batch-mean full ||H||_*)")
    for i, v in enumerate(dd.get("nuclear_layers", [])):
        lines.append(f"  layer {i}: {v:.6f}")
    lines.append("  c_ell (full): " + ", ".join(f"{c:.6f}" for c in dd.get("c_ell_full", [])))
    lines.append(
        f"  c_bar_geom: {nuclear_c_bar_geom(dd.get('nuclear_layers', [])):.6f}  "
        r"((N_L/N_0)^(1/L), same as script7)"
    )
    lines.append("")

    lines.append("[transformer] nuclear norm N_l")
    if td.get("nuclear_layers"):
        for i, v in enumerate(td["nuclear_layers"]):
            lines.append(f"  layer {i}: {v:.6f}")
        lines.append(
            "  c_ell (full): "
            + ", ".join(f"{c:.6f}" for c in td.get("c_ell_full", []))
        )
        lines.append(
            f"  c_bar_geom: {nuclear_c_bar_geom(td.get('nuclear_layers', [])):.6f}  "
            r"((N_L/N_0)^(1/L), same as script7)"
        )
    else:
        lines.append("  (not computed — --dual-only)")
    lines.append("")

    lines.append(
        f"[{dual_model_key}] R_scat(k) smart FFN (post-FFN / pre-act on 4d), by block"
    )
    rdict = dd.get("r_scat_by_layer", {})
    any_r = False
    for k in k_list:
        key = f"k{k}"
        if key in rdict:
            any_r = True
            vals = rdict[key]
            lines.append(f"  k={k}: " + ", ".join(f"{v:.6f}" for v in vals))
    if not any_r:
        lines.append("  (N/A — no R_scat hooks for this dual variant)")
    lines.append("")

    lines.append(
        f"[{dual_model_key}] attention mean row entropy (smart path, nats) by layer"
    )
    for i, e in enumerate(dd.get("attn_entropy", [])):
        lines.append(f"  layer {i}: {e:.6f}")
    lines.append("")

    lines.append("[transformer] attention mean row entropy (nats) by layer")
    if td.get("attn_entropy"):
        for i, e in enumerate(td["attn_entropy"]):
            lines.append(f"  layer {i}: {e:.6f}")
    else:
        lines.append("  (not computed — --dual-only)")
    lines.append("")

    lines.append("[singular values] top 8 (last hidden, sample 0)")
    ds = dd.get("singular_values_example", [])[:8]
    ts = td.get("singular_values_example", [])[:8]
    lines.append(
        f"  {dual_model_key}: " + ", ".join(f"{float(s):.6f}" for s in ds)
    )
    lines.append(
        "  transformer: "
        + (
            ", ".join(f"{float(s):.6f}" for s in ts)
            if ts
            else "(not computed — --dual-only)"
        )
    )
    lines.append("")

    text = "\n".join(lines) + "\n"
    out_path = out_dir / f"metrics_report_seed{seed}.txt"
    out_path.write_text(text, encoding="utf-8")
    print(text)


def plot_singular_spectrum(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    all_s: list[float] = []
    for r in rows:
        s = np.array(r["sigma"][:32], dtype=float)
        all_s.extend(s.tolist())
        ax.plot(np.arange(1, len(s) + 1), s, marker="o", markersize=3, label=r["model"])
    ax.set_xlabel(r"Index $i$")
    ax.set_ylabel(r"$\sigma_i$")
    ax.set_title("Top singular values (one batch example, last hidden)")
    ax.legend(fontsize=7)
    ax.grid(True, ls=":", alpha=0.45)
    if all_s:
        lo, hi = min(all_s), max(all_s)
        ax.set_ylim(0.0, hi * 1.08 +1e-9)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_script8_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def resolve_device(name: str) -> torch.device:
    """Prefer MPS on Apple Silicon when name is 'auto'; else pass through to torch.device."""
    key = name.strip().lower()
    if key == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def parse_k_list(s: str, mem_dim: int, dim: int) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        if p == "mem_dim":
            out.append(mem_dim)
        elif p == "sqrt_d":
            out.append(max(1, int(round(dim**0.5))))
        else:
            out.append(int(p))
    # de-dup sorted
    return sorted(set(out))


def run_one_seed(args: argparse.Namespace, subdir: str | None = None) -> dict[str, float]:
    device = resolve_device(args.device)
    print(f"Device: {device} (requested {args.device!r})")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mem_dim <= 0 or args.mem_dim >= args.dim:
        raise ValueError("require0 < mem_dim < dim for orthogonal split (smart subspace non-empty).")
    if args.dim % args.nhead != 0:
        raise ValueError("dim must be divisible by nhead (MultiheadAttention).")
    if not args.val_random_n:
        if args.val_n < args.n_min or args.val_n > args.n_max:
            raise ValueError("val_n must satisfy n_min <= val_n <= n_max (or use --val-random-n).")
    if args.task == "gap_copy":
        if int(args.gap_min) < 0 or int(args.gap_max) < int(args.gap_min):
            raise ValueError("gap_copy requires 0 <= gap-min <= gap-max")
        if not args.val_random_n:
            if int(args.val_gap) < int(args.gap_min) or int(args.val_gap) > int(
                args.gap_max
            ):
                raise ValueError("val-gap must lie in [gap-min, gap-max] for gap_copy")

    vocab_content = args.vocab_content
    pad_id = vocab_content
    if args.task == "identity":
        # Only content tokens [0, vocab_content) and pad_id = vocab_content (no DELIM in stream).
        vocab_size = vocab_content + 1
        delim_id = vocab_content + 1  # unused by identity batches; kept for API symmetry
    else:
        delim_id = vocab_content + 1
        vocab_size = vocab_content + 2  # content + pad + DELIM
    if args.task == "identity":
        max_len = args.n_max
    elif args.task == "gap_copy":
        max_len = int(args.n_max) + int(args.gap_max) + 1 + int(args.n_max)
    elif args.task == "dup_tail_copy":
        max_len = 3 * int(args.n_max) + 1
    else:
        max_len = 2 * args.n_max + 1
    dim_ff = 4 * args.dim

    k_list = parse_k_list(args.k_list, args.mem_dim, args.dim)

    root_out = Path(args.output_dir)
    out_dir = root_out / subdir if subdir else root_out
    out_dir.mkdir(parents=True, exist_ok=True)

    vsp = Path(args.val_set_file)
    val_path = vsp if vsp.is_absolute() else root_out / vsp
    val_tokens, val_labels, val_pad_mask = ensure_fixed_val_set(
        args,
        val_path,
        max_len,
        vocab_content,
        pad_id,
        delim_id,
        device,
    )

    if args.decay_mem_dual and args.global_dual_channel:
        raise ValueError("--decay-mem-dual is incompatible with --global-dual-channel.")
    if args.decay_mem_dual and args.dual_last_layer_no_mem:
        raise ValueError(
            "--dual-last-layer-no-mem applies to ortho dual only (not --decay-mem-dual)."
        )
    dual_label = "dual_decay_mem" if args.decay_mem_dual else "dual_ortho_mem"
    if args.decay_mem_dual:
        dual = SeqModelDecayMemResidual(
            vocab_size,
            args.dim,
            args.depth,
            args.nhead,
            dim_feedforward=dim_ff,
            decay_base=args.decay_mem_base,
            activation="relu",
        ).to(device)
    else:
        dual = SeqModelDualChannel(
            vocab_size,
            args.dim,
            args.depth,
            args.mem_dim,
            args.nhead,
            dim_feedforward=dim_ff,
            use_layernorm=not args.no_layernorm,
            use_residual=not args.no_residual,
            activation="relu",
            global_dual_channel=args.global_dual_channel,
            last_layer_mem_identity=args.dual_last_layer_mem_identity,
            last_layer_no_mem=args.dual_last_layer_no_mem,
            mem_basis=args.mem_basis,
        ).to(device)
    tfmr: SeqModelTransformer | None = None
    if not args.dual_only:
        tfmr = SeqModelTransformer(
            vocab_size, args.dim, args.depth, args.nhead, dim_feedforward=dim_ff
        ).to(device)

    if not args.decay_mem_dual:
        _aux_depth_mult = float(args.depth) if args.dual_aux_scale_by_depth else 1.0
        _aux_eff = (
            float(args.dual_aux_mem_weight) * _aux_depth_mult,
            float(args.dual_aux_orthog_weight) * _aux_depth_mult,
        )
    else:
        _aux_depth_mult = ""
        _aux_eff = ("", "")

    param_rows = [
        {
            "model": dual_label,
            "params": count_parameters(dual),
            "dim": args.dim,
            "depth": args.depth,
            "mem_dim": args.mem_dim if not args.decay_mem_dual else "",
            "dim_feedforward": dim_ff,
            "nhead": args.nhead,
            "global_dual_channel": False if args.decay_mem_dual else args.global_dual_channel,
            "decay_mem_base": args.decay_mem_base if args.decay_mem_dual else "",
            "dual_last_layer_mem_identity": (
                bool(args.dual_last_layer_mem_identity)
                if not args.decay_mem_dual
                else ""
            ),
            "dual_last_layer_no_mem": (
                bool(args.dual_last_layer_no_mem) if not args.decay_mem_dual else ""
            ),
            "dual_aux_mem_weight": (
                float(args.dual_aux_mem_weight) if not args.decay_mem_dual else ""
            ),
            "dual_aux_orthog_weight": (
                float(args.dual_aux_orthog_weight) if not args.decay_mem_dual else ""
            ),
            "dual_aux_scale_by_depth": (
                bool(args.dual_aux_scale_by_depth) if not args.decay_mem_dual else ""
            ),
            "dual_aux_depth_multiplier": _aux_depth_mult,
            "dual_aux_mem_effective": _aux_eff[0] if not args.decay_mem_dual else "",
            "dual_aux_orthog_effective": _aux_eff[1] if not args.decay_mem_dual else "",
            "dual_aux_mem_target": (
                str(args.dual_aux_mem_target) if not args.decay_mem_dual else ""
            ),
            "dual_aux_orthog_mode": (
                str(args.dual_aux_orthog_mode) if not args.decay_mem_dual else ""
            ),
            "dual_main_ce_weight": (
                float(args.dual_main_ce_weight) if not args.decay_mem_dual else ""
            ),
            "dual_mem_only_aux": (False if not args.decay_mem_dual else ""),
            "mem_basis": args.mem_basis if not args.decay_mem_dual else "",
        },
    ]
    if tfmr is not None:
        param_rows.append(
            {
                "model": "transformer",
                "params": count_parameters(tfmr),
                "dim": args.dim,
                "depth": args.depth,
                "mem_dim": "",
                "dim_feedforward": dim_ff,
                "nhead": args.nhead,
                "global_dual_channel": "",
                "decay_mem_base": "",
                "dual_last_layer_mem_identity": "",
                "dual_last_layer_no_mem": "",
                "dual_aux_mem_weight": "",
                "dual_aux_orthog_weight": "",
                "dual_aux_scale_by_depth": "",
                "dual_aux_depth_multiplier": "",
                "dual_aux_mem_effective": "",
                "dual_aux_orthog_effective": "",
                "dual_aux_mem_target": "",
                "dual_aux_orthog_mode": "",
                "dual_main_ce_weight": float(args.dual_main_ce_weight),
                "dual_mem_only_aux": "",
                "mem_basis": "",
            }
        )
    save_csv(out_dir / f"param_counts_seed{args.seed}.csv", param_rows)
    print("Parameter counts:", param_rows)
    if args.decay_mem_dual:
        print(
            f"Decay-mem dual: after each Transformer layer add "
            f"({args.decay_mem_base}**idx) * embed (no ortho split)."
        )
    elif args.global_dual_channel:
        if getattr(args, "mem_basis", "learned_ortho") == "coord_slice":
            print(
                "Dual: global coord_slice + shared linear_mem (no learnable ortho W).",
                flush=True,
            )
        else:
            print(
                "Dual: global shared OrthogonalProjection + linear_mem (one basis for all layers).",
                flush=True,
            )
    if args.dual_last_layer_no_mem and not args.decay_mem_dual:
        print("Dual: last block output = smart path only (memory branch not merged).")
    if args.dual_only:
        print("Dual-only: skipping Transformer training and Transformer diagnostics.")

    rows_dual = train_run(
        args,
        out_dir,
        model_name=dual_label,
        model=dual,
        device=device,
        vocab_content=vocab_content,
        pad_id=pad_id,
        delim_id=delim_id,
        max_len=max_len,
        val_tokens=val_tokens,
        val_labels=val_labels,
        val_pad_mask=val_pad_mask,
    )
    rows_tfmr: list[dict] = []
    if tfmr is not None:
        rows_tfmr = train_run(
            args,
            out_dir,
            model_name="transformer",
            model=tfmr,
            device=device,
            vocab_content=vocab_content,
            pad_id=pad_id,
            delim_id=delim_id,
            max_len=max_len,
            val_tokens=val_tokens,
            val_labels=val_labels,
            val_pad_mask=val_pad_mask,
        )

    rows_for_plot: dict[str, list[dict]] = {dual_label: rows_dual}
    if rows_tfmr:
        rows_for_plot["transformer"] = rows_tfmr
    plot_loss_and_acc(rows_for_plot, out_dir)

    # Post-hoc diagnostics: same fixed val batch as training eval
    tokens, labels, pad_mask = val_tokens, val_labels, val_pad_mask

    if isinstance(dual, SeqModelDecayMemResidual):
        dd = collect_decay_mem_diagnostics(dual, tokens, pad_mask, k_list)
    else:
        dd = collect_dual_channel_diagnostics(dual, tokens, pad_mask, k_list)
    td: dict[str, Any]
    if tfmr is not None:
        td = collect_transformer_diagnostics(tfmr, tokens, pad_mask, k_list)
    else:
        td = empty_transformer_diagnostics()

    dual_cgeom = nuclear_c_bar_geom(dd["nuclear_layers"])
    tfmr_cgeom = nuclear_c_bar_geom(td["nuclear_layers"])

    c_records: list[dict] = []
    for i, c in enumerate(dd["c_ell_full"]):
        c_records.append(
            {
                "model": dual_label,
                "ell": i,
                "c_ell_full": c,
                "c_bar_geom": dual_cgeom,
            }
        )
    for i, c in enumerate(td["c_ell_full"]):
        c_records.append(
            {
                "model": "transformer",
                "ell": i,
                "c_ell_full": c,
                "c_bar_geom": tfmr_cgeom,
            }
        )
    save_csv(out_dir / f"diagnostics_c_ell_seed{args.seed}.csv", c_records)
    plot_c_ell_bars(c_records, out_dir)

    r_scat_rows: list[dict] = []
    aux_keys = dd["r_scat_by_layer"]
    n_layers = len(aux_keys[list(aux_keys.keys())[0]]) if aux_keys else 0
    for ell in range(n_layers):
        for k in k_list:
            key = f"k{k}"
            if key in dd["r_scat_by_layer"]:
                r_scat_rows.append(
                    {
                        "model": dual_label,
                        "layer": ell,
                        "k": k,
                        "r_scat_mean": dd["r_scat_by_layer"][key][ell],
                    }
                )
    save_csv(out_dir / f"diagnostics_r_scat_seed{args.seed}.csv", r_scat_rows)
    plot_r_scat(r_scat_rows, out_dir)

    attn_rows = [
        {"model": dual_label, "layer": i, "entropy": e}
        for i, e in enumerate(dd["attn_entropy"])
    ]
    if td.get("attn_entropy"):
        attn_rows.extend(
            [
                {"model": "transformer", "layer": i, "entropy": e}
                for i, e in enumerate(td["attn_entropy"])
            ]
        )
    save_csv(out_dir / f"diagnostics_attn_entropy_seed{args.seed}.csv", attn_rows)
    plot_attn_entropy(attn_rows, out_dir)
    mats_t = td.get("attn_matrices_sample0", [])
    mats_d = dd.get("attn_matrices_sample0", [])
    if args.task == "identity":
        val_n_note = (
            f"random L in [{args.n_min},{args.n_max}]"
            if args.val_random_n
            else f"fixed seq length L={args.val_n}"
        )
    else:
        val_n_note = (
            f"random n in [{args.n_min},{args.n_max}]"
            if args.val_random_n
            else f"fixed val n={args.val_n}"
        )
    if mats_t:
        plot_attention_matrices_2d(
            mats_t,
            out_dir,
            titles=[f"Transformer L{i}" for i in range(len(mats_t))],
            fig_stem="figure_script8_attn_heatmap",
            suptitle=f"Full Transformer: self-attention (sample 0, {val_n_note})",
        )
    if mats_d:
        _dual_attn_title = (
            "Decay-mem dual: self-attention"
            if args.decay_mem_dual
            else "Dual model: smart-path self-attention"
        )
        plot_attention_matrices_2d(
            mats_d,
            out_dir,
            titles=[f"{dual_label} L{i}" for i in range(len(mats_d))],
            fig_stem="figure_script8_attn_heatmap_dual",
            suptitle=f"{_dual_attn_title} (sample 0, {val_n_note})",
        )
    if mats_t:
        plot_row_entropy_heatmap(
            mats_t[0],
            out_dir,
            fig_stem="figure_script8_attn_row_entropy_map_transformer",
            title="Transformer layer 0: per-row H_i (sample 0)",
        )
    if mats_d:
        plot_row_entropy_heatmap(
            mats_d[0],
            out_dir,
            fig_stem="figure_script8_attn_row_entropy_map_dual",
            title=f"{dual_label} layer 0: per-row H_i (sample 0)",
        )

    spec_rows = [
        {
            "model": dual_label,
            "sigma_json": json.dumps(dd["singular_values_example"]),
        },
    ]
    if td.get("singular_values_example"):
        spec_rows.append(
            {
                "model": "transformer",
                "sigma_json": json.dumps(td["singular_values_example"]),
            }
        )
    save_csv(out_dir / f"diagnostics_spectrum_seed{args.seed}.csv", spec_rows)
    spec_plot_rows: list[dict[str, Any]] = [
        {"model": dual_label, "sigma": dd["singular_values_example"]},
    ]
    if td.get("singular_values_example"):
        spec_plot_rows.append(
            {"model": "transformer", "sigma": td["singular_values_example"]}
        )
    plot_singular_spectrum(spec_plot_rows, out_dir)

    thrs = parse_acc_hit_thresholds(args.acc_hit_thresholds)
    fh_dual = first_steps_val_acc_ge(rows_dual, thrs) if thrs else {}
    fh_tfmr = (
        first_steps_val_acc_ge(rows_tfmr, thrs) if (thrs and rows_tfmr) else {}
    )

    write_metrics_report(
        out_dir,
        seed=args.seed,
        task=args.task,
        rows_dual=rows_dual,
        rows_tfmr=rows_tfmr,
        dd=dd,
        td=td,
        k_list=k_list,
        first_hits_dual=fh_dual or None,
        first_hits_tfmr=fh_tfmr or None,
    )

    ld = rows_dual[-1]
    summary: dict[str, Any] = {
        "seed": args.seed,
        "dual_final_val": ld["val_loss"],
        "tfmr_final_val": rows_tfmr[-1]["val_loss"] if rows_tfmr else "",
        "dual_final_acc": ld["val_acc_all"],
        "tfmr_final_acc": rows_tfmr[-1]["val_acc_all"] if rows_tfmr else "",
        "dual_final_norm_lev": ld.get("val_copy_norm_lev", ""),
        "tfmr_final_norm_lev": (
            rows_tfmr[-1].get("val_copy_norm_lev", "") if rows_tfmr else ""
        ),
        "acc_hit_thresholds": args.acc_hit_thresholds,
        "subdir": subdir or ".",
    }
    for k, v in fh_dual.items():
        summary[f"dual_{k}"] = v
    for k, v in fh_tfmr.items():
        summary[f"tfmr_{k}"] = v
    save_csv(out_dir / f"summary_seed{args.seed}.csv", [summary])
    return {
        "dual_final_val": float(summary["dual_final_val"]),
        "tfmr_final_val": (
            float(summary["tfmr_final_val"])
            if summary["tfmr_final_val"] != ""
            else float("nan")
        ),
        "dual_final_acc": float(summary["dual_final_acc"]),
        "tfmr_final_acc": (
            float(summary["tfmr_final_acc"])
            if summary["tfmr_final_acc"] != ""
            else float("nan")
        ),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Script 8: dual-space memory vs transformer")
    p.add_argument(
        "--aggregate-plots-only",
        action="store_true",
        help="Only read output-dir/seed_*/training_*.csv and write figure_script8_loss_acc_multi_seed.{png,pdf} (mean±std).",
    )
    p.add_argument(
        "--task",
        type=str,
        choices=("delimiter", "identity", "gap_copy", "dup_tail_copy"),
        default="delimiter",
        help="delimiter / identity；gap_copy / dup_tail_copy 见 memory_tasks（与 script9 同规格）。",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="",
        choices=("", "mem-smoke-gap", "mem-smoke-dup", "mem-dup-1k"),
        help="mem-smoke-*：短烟测；mem-dup-1k：dup_tail_copy n=8 vocab=32 val_batch=256 steps=1000（更适合多 seed / 方差估计）。",
    )
    p.add_argument(
        "--gap-min",
        type=int,
        default=0,
        help="gap_copy：filler 长度下界。",
    )
    p.add_argument(
        "--gap-max",
        type=int,
        default=0,
        help="gap_copy：filler 长度上界。",
    )
    p.add_argument(
        "--val-gap",
        type=int,
        default=0,
        help="gap_copy 且固定 val：filler 长度。",
    )
    p.add_argument(
        "--mem-basis",
        type=str,
        choices=("learned_ortho", "coord_slice"),
        default="learned_ortho",
        help="coord_slice：无 OrthogonalProjection.W，固定坐标 mem/smt（参数量近对齐 Transformer）。",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto: MPS if available, else CUDA, else CPU. Or e.g. cpu, mps, cuda:0.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45",
        help="Comma-separated training seeds. Default runs 4 seeds, subdirs seed_*/ + multi_seed_aggregate.csv (mean/std). Use --seeds '' for single --seed only.",
    )
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--dim", type=int, default=128)
    p.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of dual/transformer blocks. Default 4; use 6 for deeper runs (may need more steps or smaller batch).",
    )
    p.add_argument(
        "--mem-dim",
        type=int,
        default=32,
        help="Memory subspace rank; must satisfy 0 < mem_dim < dim.",
    )
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--vocab-content", type=int, default=24)
    p.add_argument(
        "--n-min",
        type=int,
        default=16,
        help="Delimiter task: min content length n (padded len 2*n_max+1). Identity task: min sequence length L.",
    )
    p.add_argument(
        "--n-max",
        type=int,
        default=96,
        help="Delimiter: max n; identity: max L; padded width is n_max (identity) or 2*n_max+1 (delimiter).",
    )
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--val-batch",
        type=int,
        default=128,
        help="Fixed validation batch size (cached to val_set_file).",
    )
    p.add_argument(
        "--val-n",
        type=int,
        default=48,
        help="Fixed n (delimiter: content length) or L (identity: seq length) for val (unless --val-random-n).",
    )
    p.add_argument(
        "--val-random-n",
        action="store_true",
        help="If set, validation sequences use random n in [n_min,n_max] (still cached; no longer fixed length).",
    )
    p.add_argument(
        "--val-set-file",
        type=str,
        default="fixed_val_set.pt",
        help="Saved under --output-dir; reused across seeds when meta matches.",
    )
    p.add_argument(
        "--val-set-seed",
        type=int,
        default=999_001,
        help="RNG seed for generating the fixed val batch (content tokens).",
    )
    p.add_argument(
        "--regen-val-set",
        action="store_true",
        help="Ignore cached val file and regenerate.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="Training steps per model (dual then transformer).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam LR (default 1e-3; was 3e-4 in older runs).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Validation / CSV / print interval (smaller => smoother curves, slightly more val work).",
    )
    p.add_argument(
        "--acc-hit-thresholds",
        type=str,
        default="0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated val_acc_all targets; summary + metrics_report record the first logged "
        "step reaching each (dual_*/tfmr_* columns). Use empty string to disable.",
    )
    p.add_argument(
        "--k-list",
        type=str,
        default="mem_dim,sqrt_d,8",
        help="Comma-separated k values, or mem_dim, sqrt_d",
    )
    p.add_argument(
        "--no-layernorm",
        action="store_true",
        help="Disable LayerNorm+pre-mix in dual-space blocks (not recommended)",
    )
    p.add_argument(
        "--no-residual",
        action="store_true",
        help="Disable residual connections in dual-space blocks",
    )
    p.add_argument(
        "--global-dual-channel",
        action="store_true",
        help="Share one OrthogonalProjection and one linear_mem across all dual layers (fewer params).",
    )
    p.add_argument(
        "--dual-only",
        action="store_true",
        help="Train and diagnose dual only (skip Transformer). Multi-seed plot shows Dual only; "
        "tfmr columns in summary/aggregate CSVs are left blank.",
    )
    p.add_argument(
        "--dual-mem-lr-scale",
        type=float,
        default=1.0,
        help="dual_ortho_mem only: multiply LR for OrthogonalProjection + linear_mem params (1.0 = same as base).",
    )
    p.add_argument(
        "--dual-aux-mem-weight",
        type=float,
        default=0.08,
        help="dual_ortho_mem only: base weight for per-layer mean MSE(mem, detached target); "
        "target from --dual-aux-mem-target (0 = off). Default 0.08; effective weight also × --depth "
        "when --dual-aux-scale-by-depth (default on).",
    )
    p.add_argument(
        "--dual-aux-orthog-weight",
        type=float,
        default=0.08,
        help="dual_ortho_mem only: base weight for orthogonality penalty (see --dual-aux-orthog-mode; 0 = off). "
        "Default 0.08 (same scale as --dual-aux-mem-weight); × --depth when --dual-aux-scale-by-depth (default on).",
    )
    p.add_argument(
        "--dual-aux-mem-target",
        type=str,
        choices=("src", "x_n"),
        default="src",
        help="dual_ortho_mem only: mem aux target — src=block input, x_n=after norm_in (both detached in loss).",
    )
    p.add_argument(
        "--dual-aux-orthog-mode",
        type=str,
        choices=("cos2", "dot2"),
        default="cos2",
        help="dual_ortho_mem only: cos2=(mem·s)²/(‖mem‖²‖s‖²+ε); dot2=raw (mem·s)².",
    )
    p.add_argument(
        "--dual-aux-scale-by-depth",
        dest="dual_aux_scale_by_depth",
        action="store_true",
        help="dual_ortho_mem only: multiply --dual-aux-mem-weight and --dual-aux-orthog-weight "
        "by --depth before adding to loss (this is the default; use --no-dual-aux-scale-by-depth to disable). "
        "Aux scalars are layer-meaned inside the model.",
    )
    p.add_argument(
        "--no-dual-aux-scale-by-depth",
        dest="dual_aux_scale_by_depth",
        action="store_false",
        help="dual_ortho_mem only: do not multiply aux weights by --depth.",
    )
    p.set_defaults(dual_aux_scale_by_depth=True)
    p.add_argument(
        "--dual-main-ce-weight",
        type=float,
        default=1.0,
        help="Multiply main CE (full hidden lm_head logits) before adding aux terms; default 1.0 keeps primary loss scale fixed.",
    )
    p.add_argument(
        "--dual-last-layer-mem-identity",
        action="store_true",
        help="Last dual block: mem branch uses x_mem directly (skip linear_mem) before mem + s merge.",
    )
    p.add_argument(
        "--dual-last-layer-no-mem",
        action="store_true",
        help="Last dual block: do not add memory branch to output (y = smart path s only). "
        "Ortho dual only; overrides --dual-last-layer-mem-identity on that layer.",
    )
    p.add_argument(
        "--decay-mem-dual",
        action="store_true",
        help="Replace ortho dual with decay-mem residual: each layer is TransformerEncoderLayerAttn, "
        "then h += (decay_mem_base**idx) * embed; output lm_head(h) only. Incompatible with --global-dual-channel.",
    )
    p.add_argument(
        "--decay-mem-base",
        type=float,
        default=0.8,
        help="Base for (base**layer_idx) additive memory term (--decay-mem-dual only).",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run grid over mem_dims and n_max list with multiple seeds",
    )
    p.add_argument(
        "--mem-dims",
        type=str,
        default="16,32,64",
        help="Used with --sweep: comma-separated mem_dim values",
    )
    p.add_argument(
        "--n-max-list",
        type=str,
        default="32,64",
        help="Used with --sweep: comma-separated n_max values",
    )
    p.add_argument(
        "--sweep-seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds for sweep",
    )
    return p


def apply_mem_smoke_preset(args: argparse.Namespace) -> None:
    if args.preset == "mem-smoke-gap":
        args.task = "gap_copy"
        args.n_min = args.n_max = 6
        args.val_n = 6
        args.gap_min = args.gap_max = 12
        args.val_gap = 12
        args.vocab_content = 16
        args.val_batch = 128
        args.batch = 32
        args.steps = 800
        args.log_every = 50
        args.dual_aux_mem_weight = 0.0
        args.dual_aux_orthog_weight = 0.0
        print(
            "preset mem-smoke-gap: gap_copy n=6 gap=12 vocab=16 steps=800 (dual aux off)",
            flush=True,
        )
    elif args.preset == "mem-smoke-dup":
        args.task = "dup_tail_copy"
        args.n_min = args.n_max = 5
        args.val_n = 5
        args.vocab_content = 16
        args.val_batch = 128
        args.batch = 32
        args.steps = 800
        args.log_every = 50
        args.dual_aux_mem_weight = 0.0
        args.dual_aux_orthog_weight = 0.0
        print(
            "preset mem-smoke-dup: dup_tail_copy n=5 vocab=16 steps=800 (dual aux off)",
            flush=True,
        )
    elif args.preset == "mem-dup-1k":
        args.task = "dup_tail_copy"
        args.n_min = args.n_max = 8
        args.val_n = 8
        args.vocab_content = 32
        args.val_batch = 256
        args.batch = 48
        args.steps = 1000
        args.log_every = 50
        args.dual_aux_mem_weight = 0.0
        args.dual_aux_orthog_weight = 0.0
        print(
            "preset mem-dup-1k: dup_tail_copy n=8 vocab=32 seq_len=25 val_batch=256 "
            "batch=48 steps=1000 (dual aux off); CE on tail_B only (8 positions)",
            flush=True,
        )


def _parse_training_seeds(args: argparse.Namespace) -> list[int]:
    s = args.seeds.strip()
    if s:
        return [int(x) for x in s.split(",") if x.strip()]
    return [args.seed]


def main() -> None:
    args = build_argparser().parse_args()
    apply_mem_smoke_preset(args)
    if args.aggregate_plots_only:
        root = Path(args.output_dir)
        note = ""
        agg_path = root / "multi_seed_aggregate.csv"
        if agg_path.is_file():
            with agg_path.open(newline="", encoding="utf-8") as f:
                agg_rows = list(csv.DictReader(f))
            if agg_rows:
                r0 = agg_rows[0]
                note = (
                    f"task={r0.get('task', '')}, steps={r0.get('steps', '')}, "
                    f"n in [{r0.get('n_min', '')},{r0.get('n_max', '')}]"
                )
        plot_multi_seed_training_curves(root, title_note=note)
        print(
            f"Wrote figure_script8_loss_acc_multi_seed.png and .pdf under {root}"
        )
        return
    if args.sweep:
        mem_dims = [int(x) for x in args.mem_dims.split(",") if x.strip()]
        n_maxs = [int(x) for x in args.n_max_list.split(",") if x.strip()]
        seeds = [int(x) for x in args.sweep_seeds.split(",") if x.strip()]
        sweep_rows: list[dict] = []
        for md in mem_dims:
            for nmax in n_maxs:
                for sd in seeds:
                    sub = f"sweep_mem{md}_nmax{nmax}_seed{sd}"
                    args.mem_dim = md
                    args.n_max = nmax
                    args.seed = sd
                    finals = run_one_seed(args, subdir=sub)
                    sweep_rows.append(
                        {
                            "mem_dim": md,
                            "n_max": nmax,
                            "seed": sd,
                            "subdir": sub,
                            "dual_final_val_ce": finals["dual_final_val"],
                            "transformer_final_val_ce": finals["tfmr_final_val"],
                        }
                    )
        out_root = Path(args.output_dir)
        save_csv(out_root / "sweep_index.csv", sweep_rows)
        # 按 (mem_dim, n_max) 聚合并跨 seed 平均
        agg: dict[tuple[int, int], list[dict]] = {}
        for r in sweep_rows:
            key = (int(r["mem_dim"]), int(r["n_max"]))
            agg.setdefault(key, []).append(r)
        summary_rows = []
        for (md, nmax), rs in sorted(agg.items()):
            summary_rows.append(
                {
                    "mem_dim": md,
                    "n_max": nmax,
                    "n_seeds": len(rs),
                    "dual_val_ce_mean": float(np.mean([x["dual_final_val_ce"] for x in rs])),
                    "tfmr_val_ce_mean": float(np.mean([x["transformer_final_val_ce"] for x in rs])),
                }
            )
        save_csv(out_root / "sweep_summary.csv", summary_rows)
    else:
        seeds = _parse_training_seeds(args)
        root_out = Path(args.output_dir)
        if len(seeds) > 1:
            multi: list[dict[str, Any]] = []
            for i, sd in enumerate(seeds):
                args.seed = sd
                if i > 0:
                    args.regen_val_set = False
                fin = run_one_seed(args, subdir=f"seed_{sd}")
                row_ms: dict[str, Any] = {
                    "seed": sd,
                    "dual_final_val_ce": fin["dual_final_val"],
                    "dual_final_val_acc": fin["dual_final_acc"],
                }
                if args.dual_only:
                    row_ms["tfmr_final_val_ce"] = ""
                    row_ms["tfmr_final_val_acc"] = ""
                else:
                    row_ms["tfmr_final_val_ce"] = fin["tfmr_final_val"]
                    row_ms["tfmr_final_val_acc"] = fin["tfmr_final_acc"]
                multi.append(row_ms)
            save_csv(root_out / "multi_seed_summary.csv", multi)
            dual_ce = [float(r["dual_final_val_ce"]) for r in multi]
            dual_acc = [float(r["dual_final_val_acc"]) for r in multi]

            def _std(xs: list[float]) -> float:
                return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0

            agg_row: dict[str, Any] = {
                "task": args.task,
                "n_seeds": len(seeds),
                "steps": args.steps,
                "n_min": args.n_min,
                "n_max": args.n_max,
                "val_n": "random" if args.val_random_n else args.val_n,
                "global_dual_channel": args.global_dual_channel,
                "dual_only": args.dual_only,
                "dual_val_ce_mean": float(np.mean(dual_ce)),
                "dual_val_ce_std": _std(dual_ce),
                "dual_val_acc_mean": float(np.mean(dual_acc)),
                "dual_val_acc_std": _std(dual_acc),
            }
            if args.dual_only:
                agg_row["tfmr_val_ce_mean"] = ""
                agg_row["tfmr_val_ce_std"] = ""
                agg_row["tfmr_val_acc_mean"] = ""
                agg_row["tfmr_val_acc_std"] = ""
            else:
                tf_ce = [float(r["tfmr_final_val_ce"]) for r in multi]
                tf_acc = [float(r["tfmr_final_val_acc"]) for r in multi]
                agg_row["tfmr_val_ce_mean"] = float(np.mean(tf_ce))
                agg_row["tfmr_val_ce_std"] = _std(tf_ce)
                agg_row["tfmr_val_acc_mean"] = float(np.mean(tf_acc))
                agg_row["tfmr_val_acc_std"] = _std(tf_acc)

            save_csv(root_out / "multi_seed_aggregate.csv", [agg_row])
            print(
                f"Wrote multi_seed_summary.csv and multi_seed_aggregate.csv under {root_out} "
                f"(per-seed artifacts in seed_<seed>/; aggregate row includes mean±std over seeds)."
            )
            plot_multi_seed_training_curves(
                root_out,
                title_note=(
                    f"task={args.task}, steps={args.steps}, "
                    f"n in [{args.n_min},{args.n_max}]"
                ),
            )
            print(
                f"Wrote figure_script8_loss_acc_multi_seed.png and .pdf under {root_out}"
            )
        else:
            run_one_seed(args)


if __name__ == "__main__":
    main()
