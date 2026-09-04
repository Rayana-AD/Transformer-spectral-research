# Script 9 — FFN-bottleneck dual (W1 后拆分，mem / smt 各自 W2 收口)
# - After linear1 (W1): split expanded mid into mem / smt (learnable ortho or fixed slice).
# - ff = W2_mem(mid_mem) + W2_smt(σ(mid_smt)) → d; CE 反传经两路独立投影。MHA sees full d after norm_in.
# - Training: CE only (no script8 MSE/orth aux). Baseline: same Pre-LN Transformer as script8.
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_tasks import build_dup_tail_copy_batch, build_gap_copy_batch

IGNORE_LABEL = -100
DEFAULT_OUTPUT = Path("results_script9_ffn_bottleneck_dual")
VAL_META_VERSION = 5

# Token layout (delimiter): content ∈ [0, vocab_content), pad_id = vocab_content,
# delim_id = vocab_content+1 — 与 content 无数值重叠；IGNORE_LABEL=-100 仅用于 label 张量。


# ---------------------------------------------------------------------------
# Bottleneck orthogonal split (same logic as script8 OrthogonalProjection, dim = r)
# ---------------------------------------------------------------------------
class OrthogonalProjection1d(nn.Module):
    """Complementary projectors on ℝ^r; mid_mem + mid_smt == mid (both shapes (B,L,r))."""

    def __init__(self, dim: int, split_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.split_dim = split_dim
        self.W = nn.Parameter(torch.randn(dim, dim))

    def forward(self, mid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W = self.W
        if W.device.type == "mps":
            W_orth, _ = torch.linalg.qr(W.cpu())
            W_orth = W_orth.to(device=W.device, dtype=W.dtype)
        else:
            W_orth, _ = torch.linalg.qr(W)
        W_mem = W_orth[:, : self.split_dim]
        W_smt = W_orth[:, self.split_dim :]
        mid_mem = mid @ W_mem @ W_mem.T
        mid_smt = mid @ W_smt @ W_smt.T
        return mid_mem, mid_smt


# ---------------------------------------------------------------------------
# Transformer baseline (script8-compatible)
# ---------------------------------------------------------------------------
class TransformerEncoderLayerAttn(nn.Module):
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
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * dim
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
        self, token_ids: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embed(token_ids)
        attns: list[torch.Tensor] = []
        for layer in self.layers:
            x, aw = layer(x, src_key_padding_mask=src_key_padding_mask)
            attns.append(aw)
        return self.lm_head(x), attns


# ---------------------------------------------------------------------------
# FFN-bottleneck dual block
# ---------------------------------------------------------------------------
SplitMode = Literal["learnable_ortho", "slice"]
FFReadoutMode = Literal["dual", "fused"]


class BottleneckDualTransformerBlock(nn.Module):
    """
    Pre-LN: norm_in(src) → full-d stream s → norm1 → MHA → residual;
    norm2 → W1 → split(mid) → FFN 收口：
    - dual: W2_mem(mid_mem) + W2_smt(σ(mid_smt))
    - fused: W2(cat(mid_mem, σ(mid_smt)))，单矩阵读回 d（与 baseline FFN 更接近的优化形态）
    - slice 时：坐标划分为 [0:r_mem) 与 [r_mem:dim_ff)，恒有 r_mem + r_smt = dim_ff，无 learnable 正交矩阵。
    """

    def __init__(
        self,
        dim: int,
        nhead: int,
        dim_feedforward: int,
        r_mem: int,
        *,
        split_mode: SplitMode = "learnable_ortho",
        ff_readout: FFReadoutMode = "dual",
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if not (0 < r_mem < dim_feedforward):
            raise ValueError(
                f"r_mem must satisfy 0 < r_mem < dim_feedforward; got r_mem={r_mem}, r={dim_feedforward}"
            )
        self.dim = dim
        self.dim_feedforward = dim_feedforward
        self.r_mem = r_mem
        self.split_mode = split_mode
        self.ff_readout = ff_readout
        r_smt = dim_feedforward - r_mem
        self.r_smt = r_smt
        if self.r_mem + self.r_smt != dim_feedforward:
            raise ValueError(
                f"slice partition broken: r_mem + r_smt must equal dim_ff; "
                f"got {self.r_mem}+{self.r_smt} vs {dim_feedforward}"
            )
        self.norm_in = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        if ff_readout == "fused":
            fused_in = (
                2 * dim_feedforward
                if split_mode == "learnable_ortho"
                else dim_feedforward
            )
            self.linear2_fused = nn.Linear(fused_in, dim)
            self.linear2_mem = None
            self.linear2_smt = None
        else:
            self.linear2_fused = None
            if split_mode == "learnable_ortho":
                self.linear2_mem = nn.Linear(dim_feedforward, dim)
                self.linear2_smt = nn.Linear(dim_feedforward, dim)
            else:
                self.linear2_mem = nn.Linear(r_mem, dim)
                self.linear2_smt = nn.Linear(r_smt, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = F.relu if activation == "relu" else F.gelu
        if split_mode == "learnable_ortho":
            self.ortho_r = OrthogonalProjection1d(dim_feedforward, r_mem)
        else:
            self.ortho_r = None

    def _split_mid(self, mid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.split_mode == "learnable_ortho":
            assert self.ortho_r is not None
            return self.ortho_r(mid)
        # slice: first r_mem coords = mem, rest = smt
        mid_mem = mid[..., : self.r_mem]
        mid_smt = mid[..., self.r_mem :]
        return mid_mem, mid_smt

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.norm_in(src)
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
        mid_mem, mid_smt = self._split_mid(mid)
        u_smt = self.act(mid_smt)
        if self.ff_readout == "fused":
            assert self.linear2_fused is not None
            z = torch.cat([mid_mem, u_smt], dim=-1)
            if __debug__ and self.split_mode == "slice":
                assert z.shape[-1] == self.dim_feedforward
            ff = self.linear2_fused(z)
        else:
            assert self.linear2_mem is not None and self.linear2_smt is not None
            ff = self.linear2_mem(mid_mem) + self.linear2_smt(u_smt)
        ff = self.dropout2(ff)
        s = s + ff
        return s, attn_w


class SeqModelBottleneckDual(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        nhead: int,
        dim_feedforward: int | None,
        r_mem: int,
        *,
        split_mode: SplitMode = "learnable_ortho",
        ff_readout: FFReadoutMode = "dual",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [
                BottleneckDualTransformerBlock(
                    dim,
                    nhead,
                    dim_feedforward,
                    r_mem,
                    split_mode=split_mode,
                    ff_readout=ff_readout,
                    dropout=0.0,
                    activation=activation,
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
        for blk in self.blocks:
            x, aw = blk(x, src_key_padding_mask=src_key_padding_mask)
            attns.append(aw)
        return self.lm_head(x), attns


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Copy batches (script8-compatible)
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
    Delimiter delayed copy: seq = [content_0..n-1 | DELIM | tail_0..n-1]，总长 2n+1（padding 前）。

    下标（0-based）: 0..n-1 为 content；n 为 DELIM；n+1..2n 为 tail。
    监督：在 **tail 位置** 预测对应 content[i]，即 lab[n+1+i] = content[i]。
    切片 lab[n+1 : 2*n+1] 恰为 n 个位置，与 content 等长 — **无 +1 偏移错误**。

    掩码：content与 DELIM 位置 lab=-100；仅 tail 有类号 ∈ [0, vocab_content)；pad 段 lab=-100。
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
            raise ValueError("delimiter copy requires n >= 1 (set n_min >= 1)")
        content = torch.randint(0, vocab_content, (n,), device=device)
        tail = torch.randint(0, vocab_content, (n,), device=device)
        seq = torch.cat(
            [content, torch.tensor([delim_id], device=device, dtype=torch.long), tail],
            dim=0,
        )
        assert int(seq.shape[0]) == 2 * n + 1
        lab = torch.full((seq.shape[0],), IGNORE_LABEL, dtype=torch.long, device=device)
        lab[n + 1 : 2 * n + 1] = content
        if __debug__:
            n_sup = int((lab != IGNORE_LABEL).sum().item())
            assert n_sup == n, (n_sup, n)
            assert int(lab[n].item()) == IGNORE_LABEL
            assert torch.all(lab[n + 1 : 2 * n + 1] < vocab_content)
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


def build_copy_mixed_batch(
    batch_size: int,
    n_min: int,
    n_max: int,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    max_len: int,
    device: torch.device,
    generator: torch.Generator | None,
    tail_keep_p: float,
    *,
    fixed_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    介于 identity 与全随机 tail 之间：tail[i] 以概率 tail_keep_p 等于 content[i]，
    否则在 [0,vocab_content) 上均匀随机。监督仍为 tail 段预测 content。
    tail_keep_p=1 -> 退化为「tail=content」（较易）；=0 -> 标准 delimiter。
    """
    if not (0.0 <= tail_keep_p <= 1.0):
        raise ValueError("tail_keep_p must be in [0,1]")
    if vocab_content < 1:
        raise ValueError("vocab_content must be >= 1")
    if pad_id != vocab_content or delim_id != vocab_content + 1:
        raise ValueError("pad_id / delim_id mismatch for mixed delimiter")
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
            raise ValueError("delimiter_mixed requires n >= 1")
        content = torch.randint(0, vocab_content, (n,), device=device)
        tail = content.clone()
        u = torch.rand(n, device=device)
        replace = u >= tail_keep_p
        n_rep = int(replace.sum().item())
        if n_rep > 0:
            tail[replace] = torch.randint(
                0, vocab_content, (n_rep,), device=device
            )
        seq = torch.cat(
            [content, torch.tensor([delim_id], device=device, dtype=torch.long), tail],
            dim=0,
        )
        lab = torch.full((seq.shape[0],), IGNORE_LABEL, dtype=torch.long, device=device)
        lab[n + 1 : 2 * n + 1] = content
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
    """逐位置复制：lab[i]=seq[i]；仅 padding 段 lab=-100。content token 可为 0..vocab_content-1，pad_id=vocab_content 不重叠。"""
    if vocab_content < 1:
        raise ValueError("vocab_content must be >= 1")
    if pad_id != vocab_content:
        raise ValueError(f"identity task expects pad_id == vocab_content ({vocab_content}), got {pad_id}")
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
        n_labeled = (labels != IGNORE_LABEL).sum(dim=1)
        assert torch.all(n_labeled > 0)
    return tokens, labels, padding_mask


def val_set_meta_dict(args: argparse.Namespace) -> dict[str, Any]:
    vc = int(args.vocab_content)
    vocab_size = vc + 1 if args.task == "identity" else vc + 2
    meta: dict[str, Any] = {
        "version": VAL_META_VERSION,
        "exp_family": "script9_ffn_bottleneck_dual",
        "task": args.task,
        "val_batch": args.val_batch,
        "val_n": "random" if args.val_random_n else args.val_n,
        "n_min": args.n_min,
        "n_max": args.n_max,
        "vocab_content": vc,
        "vocab_size": vocab_size,
        "val_set_seed": args.val_set_seed,
        "tail_keep_p": (
            float(args.tail_keep_p) if args.task == "delimiter_mixed" else None
        ),
        "gap_min": int(args.gap_min) if args.task == "gap_copy" else None,
        "gap_max": int(args.gap_max) if args.task == "gap_copy" else None,
        "val_gap": (
            "random"
            if args.task == "gap_copy" and args.val_random_n
            else (int(args.val_gap) if args.task == "gap_copy" else None)
        ),
        "dup_tail": bool(args.task == "dup_tail_copy"),
    }
    return meta


def meta_compatible(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    if stored.get("exp_family") != current.get("exp_family"):
        return False
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
        "tail_keep_p",
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
    meta = val_set_meta_dict(args)
    cpu = torch.device("cpu")
    if val_path.is_file() and not args.regen_val_set:
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
    elif args.task == "delimiter_mixed":
        t, lab, pm = build_copy_mixed_batch(
            args.val_batch,
            args.n_min,
            args.n_max,
            vocab_content,
            pad_id,
            delim_id,
            max_len,
            cpu,
            gen,
            float(args.tail_keep_p),
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
    B, L, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * L, V),
        labels.reshape(B * L),
        ignore_index=IGNORE_LABEL,
        reduction="mean",
    )


def accuracy_on_labeled(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float, float]:
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


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


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
    n_params: int,
    split_mode: str,
    r_mem: int,
    dim_ff: int,
    ff_readout: str,
) -> tuple[list[dict], float]:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rows: list[dict] = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 777)
    best_val_acc = float("-inf")
    prog = int(args.progress_every)
    n_steps = int(args.steps)
    lr_sched = getattr(args, "lr_schedule", "constant")

    print(
        f"[{model_name}] training {n_steps} steps | val every {args.log_every} | "
        f"train progress every {prog if prog > 0 else 'off'} | lr_schedule={lr_sched}",
        flush=True,
    )

    for step in range(n_steps + 1):
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
        elif args.task == "delimiter_mixed":
            tokens, labels, pad_mask = build_copy_mixed_batch(
                args.batch,
                args.n_min,
                args.n_max,
                vocab_content,
                pad_id,
                delim_id,
                max_len,
                device,
                gen,
                float(args.tail_keep_p),
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
        opt.zero_grad()
        logits, _ = model(tokens, src_key_padding_mask=pad_mask)
        loss = masked_ce_loss(logits, labels)
        loss.backward()
        opt.step()
        if lr_sched == "cosine":
            t = min(1.0, float(step + 1) / max(1, n_steps))
            lr_t = float(args.lr_min) + (float(args.lr) - float(args.lr_min)) * 0.5 * (
                1.0 + math.cos(math.pi * t)
            )
            for pg in opt.param_groups:
                pg["lr"] = lr_t

        ce_f = float(loss.item())
        if step == 0:
            print(
                f"[{model_name}] step 0/{n_steps} train_ce {ce_f:.4f} (running val...)",
                flush=True,
            )
        elif (
            prog > 0
            and step % prog == 0
            and step % args.log_every != 0
            and step != n_steps
        ):
            print(
                f"[{model_name}] step {step:5d}/{n_steps} | train_ce {ce_f:.4f}",
                flush=True,
            )

        if step % args.log_every == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                v_logits, _ = model(val_tokens, src_key_padding_mask=val_pad_mask)
                vloss = masked_ce_loss(v_logits, val_labels).item()
                acc, acc1, acc2 = accuracy_on_labeled(v_logits, val_labels)
                _, v_nlev = copy_segment_em_and_norm_levenshtein(v_logits, val_labels)
            if acc > best_val_acc:
                best_val_acc = acc
            # CSV names aligned with script8_dual_space_memory train_run (CE-only: train_total = train_loss)
            val_n_cell: str | int = "random" if args.val_random_n else args.val_n
            row: dict[str, str | float | int] = {
                "model": model_name,
                "step": step,
                "train_loss": ce_f,
                "train_total": ce_f,
                "train_aux_mem": 0.0,
                "train_aux_orthog": 0.0,
                "val_loss": vloss,
                "val_acc_all": acc,
                "val_acc_copy_first_half": acc1,
                "val_acc_copy_second_half": acc2,
                "val_copy_norm_lev": v_nlev,
                "best_val_acc_so_far": best_val_acc,
                "seed": args.seed,
                "mem_dim": r_mem if split_mode else "",
                "n_max": args.n_max,
                "val_batch": args.val_batch,
                "val_n": val_n_cell,
                "split_mode": split_mode,
                "r_mem": r_mem,
                "dim_ff": dim_ff,
                "n_params": n_params,
                "depth": args.depth,
                "dim": args.dim,
                "nhead": args.nhead,
                "ff_mult": args.ff_mult,
                "ff_readout": ff_readout,
                "lr_schedule": lr_sched,
            }
            rows.append(row)
            # Same metric print pattern as script8 when aux is off
            print(
                f"[{model_name}] step {step:5d} | train_ce {ce_f:.4f} | val_ce {vloss:.4f} | "
                f"acc {acc:.3f} |1st {acc1:.3f} | 2nd {acc2:.3f} | nLev {v_nlev:.3f} | "
                f"best {best_val_acc:.3f}",
                flush=True,
            )
    save_csv(out_dir / f"training_{model_name}_seed{args.seed}.csv", rows)
    return rows, best_val_acc


def resolve_device(name: str) -> torch.device:
    key = name.strip().lower()
    if key == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def write_note_experiments(path: Path) -> None:
    text = """# Script9 experiment matrix (FFN-bottleneck dual)

## vs Script8
- Script8: orthogonal split in **d** before MHA; mem bypasses FFN; merge **mem + s** in **d** after FFN.
- Script9: split after **W1** in **r = dim_ff**; **MHA on full d**; **ff = W2_mem(mid_mem) + W2_smt(σ(mid_smt))**（slice 时两路维数分别为 r_mem 与 r−r_mem）。

## A. Smoke / correctness
- `steps<=50`, small `dim`, `depth=2`, both `--model transformer` and `--model bottleneck_dual`.
- Expect finite loss, no NaN.

## B. Main comparison
- **`--preset benchmark`**：**delimiter_mixed**（tail 以0.5 概率抄 content[i]）固定 **n=4**、**vocab=16**，默认 **256×4**；identity 仅用 `--preset micro-identity` 冒烟。
- **`--preset bench-compare`**：**n=8**、vocab=24、val_batch=256、**6000 steps**、**cosine lr**（1e-3→1e-5），验证更稳、任务更难，便于和 **参数量对齐 dual**（`--split-mode slice --ff-readout fused`，建议 `--r-mem` = dim_ff/2）对照。
- **`--small-model`** 仅容量消融，不用于解决「学不会」。更难用 **`--preset hard-delimiter`**。
- Seeds: at least 2--3 (`--seed` per run or shell loop).
- Models: `transformer` vs `bottleneck_dual`.
- Readout: `--ff-readout dual`（双 W2）vs `fused`（单 W2 融合 cat，优化更接近 baseline FFN）。
- Splits: `slice` (param-fair vs extra ortho) vs `learnable_ortho`.
- Logged CSV: `val_ce`, `val_acc_*`, `best_val_acc_so_far`, `n_params`, `split_mode`, `r_mem`.

## C. Ablations (optional)
- Sweep `r_mem`: e.g. `r/8`, `r/4`, `r/2` for `--split-mode slice`, or corresponding `--r-mem` for ortho.

## Example commands
Do not append literal `...` in the shell (it is parsed as an extra argument).

```bash
# Transformer baseline (val every 25 steps; defaults)
python script9_ffn_bottleneck_dual.py --model transformer --seed 42 --steps 2500 --output-dir results_script9_demo

# Bottleneck dual, slice
python script9_ffn_bottleneck_dual.py --model bottleneck_dual --split-mode slice --r-mem 128 --seed 42 --steps 2500 --output-dir results_script9_demo

# Bottleneck dual, learnable ortho
python script9_ffn_bottleneck_dual.py --model bottleneck_dual --split-mode learnable_ortho --r-mem 128 --seed 42 --steps 2500 --output-dir results_script9_demo
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Script 9: FFN-bottleneck dual vs Transformer (CE only)",
        epilog=(
            "Tip: do not type literal ... on the command line (shell passes it as an argument and you get "
            "'unrecognized arguments: ...'). Use a complete flag list, or end the command after the last value.\n\n"
            "Example (val + metrics every 25 steps, default):\n"
            "  python script9_ffn_bottleneck_dual.py --model transformer --output-dir results_script9_run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        type=str,
        choices=("transformer", "bottleneck_dual"),
        default="bottleneck_dual",
    )
    p.add_argument(
        "--split-mode",
        type=str,
        choices=("learnable_ortho", "slice"),
        default="learnable_ortho",
        help="bottleneck_dual only: learnable ortho vs coordinate slice on mid.",
    )
    p.add_argument(
        "--ff-readout",
        type=str,
        choices=("dual", "fused"),
        default="dual",
        help="bottleneck_dual only: dual = W2_mem+W2_smt；fused = 单 W2(cat(mid_mem,σ(mid_smt)))，参数量与 dual 的 W2 部分同阶。",
    )
    p.add_argument(
        "--r-mem",
        type=int,
        default=32,
        help="Bottleneck mem width: ortho split_dim (learnable_ortho) or first slice length (slice); must be in (0, dim_ff).",
    )
    p.add_argument(
        "--task",
        type=str,
        choices=(
            "delimiter",
            "delimiter_mixed",
            "identity",
            "gap_copy",
            "dup_tail_copy",
        ),
        default="delimiter",
        help="delimiter / delimiter_mixed / identity；gap_copy=[content|filler|DELIM|tail]；"
        "dup_tail_copy=[content|DELIM|tail_A|tail_B] 仅 tail_B 监督。",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=4, help="dim_feedforward = ff_mult * dim")
    p.add_argument("--vocab-content", type=int, default=24)
    p.add_argument(
        "--tail-keep-p",
        type=float,
        default=0.5,
        help="仅 delimiter_mixed：每个 tail[i] 保留为 content[i] 的概率 ∈[0,1]；0=全随机 delimiter，1=tail=content。",
    )
    p.add_argument(
        "--gap-min",
        type=int,
        default=0,
        help="仅 gap_copy：filler 段长度下界（含）。",
    )
    p.add_argument(
        "--gap-max",
        type=int,
        default=0,
        help="仅 gap_copy：filler 段长度上界（含）。",
    )
    p.add_argument(
        "--val-gap",
        type=int,
        default=0,
        help="仅 gap_copy 且非 --val-random-n：固定验证集 filler 长度。",
    )
    p.add_argument("--n-min", type=int, default=16)
    p.add_argument("--n-max", type=int, default=96)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--val-batch", type=int, default=128)
    p.add_argument("--val-n", type=int, default=48)
    p.add_argument("--val-random-n", action="store_true")
    p.add_argument("--val-set-file", type=str, default="fixed_val_set_script9.pt")
    p.add_argument("--val-set-seed", type=int, default=999_001)
    p.add_argument("--regen-val-set", action="store_true")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--lr-schedule",
        type=str,
        choices=("constant", "cosine"),
        default="constant",
        help="cosine: 训练过程中 lr 从 --lr 余弦降至 --lr-min（利于长训程拉开差异、减轻平台期）。",
    )
    p.add_argument(
        "--lr-min",
        type=float,
        default=1e-5,
        help="与 --lr-schedule cosine 配合使用。",
    )
    p.add_argument(
        "--small-model",
        action="store_true",
        help="可选：dim=96 depth=3 nhead=4，仅作容量消融。**主实验请保持默认 256×4**；学不会应减轻任务（preset benchmark），而非靠缩小模型。",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Run validation and print acc / val_ce every N steps (and step 0, last step).",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print train_ce every N steps without val; 0 disables. Default 25 (same as --log-every: "
        "only full val lines). Use 1 for a train_ce heartbeat every step.",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="",
        choices=(
            "",
            "micro-identity",
            "benchmark",
            "bench-compare",
            "mem-smoke-gap",
            "mem-smoke-dup",
            "hard-delimiter",
        ),
        help="micro-identity / benchmark / bench-compare / hard-delimiter 见上；"
        "mem-smoke-gap: gap_copy n=6 gap=12 vocab=16，≤~1000 步烟测；"
        "mem-smoke-dup: dup_tail_copy n=5 vocab=16。",
    )
    return p


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "micro-identity":
        args.task = "identity"
        args.n_min = 8
        args.n_max = 8
        args.val_n = 8
        args.vocab_content = 32
        print(
            "preset micro-identity: 固定 L=8 vocab=32（identity+双向注意极易快饱和，仅冒烟）",
            flush=True,
        )
    elif args.preset == "benchmark":
        args.task = "delimiter_mixed"
        args.n_min = 4
        args.n_max = 4
        args.val_n = 4
        args.vocab_content = 16
        args.tail_keep_p = 0.5
        print(
            "preset benchmark: delimiter_mixed 固定 n=4, vocab=16, tail_keep_p=0.5, val_n=4 — "
            "难度在 identity 与全随机 tail 之间；默认 dim=256 depth=4",
            flush=True,
        )
    elif args.preset == "bench-compare":
        args.task = "delimiter_mixed"
        args.n_min = 8
        args.n_max = 8
        args.val_n = 8
        args.vocab_content = 24
        args.tail_keep_p = 0.5
        args.val_batch = 256
        args.steps = 6000
        args.lr_schedule = "cosine"
        args.lr_min = 1e-5
        args.log_every = 50
        args.progress_every = 50
        print(
            "preset bench-compare: delimiter_mixed 固定 n=8, vocab=24, tail_keep_p=0.5, val_n=8, "
            "val_batch=256, steps=6000, lr cosine 1e-3→1e-5, val/log每50步 — "
            "更长 copy + 更低方差验证 + 长训程；dual 请用 --split-mode slice --ff-readout fused --r-mem 与 dim_ff/2 对齐",
            flush=True,
        )
    elif args.preset == "mem-smoke-gap":
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
        args.progress_every = 50
        args.lr_schedule = "constant"
        print(
            "preset mem-smoke-gap: gap_copy n=6 gap=12 vocab=16 val_batch=128 steps=800",
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
        args.progress_every = 50
        args.lr_schedule = "constant"
        print(
            "preset mem-smoke-dup: dup_tail_copy n=5 vocab=16 val_batch=128 steps=800",
            flush=True,
        )
    elif args.preset == "hard-delimiter":
        args.task = "delimiter"
        args.n_min = 14
        args.n_max = 22
        args.val_n = 18
        args.vocab_content = 40
        print(
            "preset hard-delimiter: delimiter n in [14,22], val_n=18, vocab=40",
            flush=True,
        )


def main() -> None:
    args = build_argparser().parse_args()
    apply_preset(args)
    if args.small_model:
        args.dim = 96
        args.depth = 3
        args.nhead = 4
        print("small-model: dim=96 depth=3 nhead=4", flush=True)
    print("script9: parsing args, resolving device...", flush=True)
    device = resolve_device(args.device)
    print(f"Device: {device} (requested {args.device!r})", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dim % args.nhead != 0:
        raise ValueError("dim must be divisible by nhead")
    if args.vocab_content < 1:
        raise ValueError("vocab_content must be >= 1")
    if args.n_min < 1 or args.n_max < 1:
        raise ValueError("n_min and n_max must be >= 1")
    if args.n_min > args.n_max:
        raise ValueError("n_min must be <= n_max")
    if not args.val_random_n:
        if args.val_n < args.n_min or args.val_n > args.n_max:
            raise ValueError("val_n must satisfy n_min <= val_n <= n_max (or --val-random-n)")

    vocab_content = args.vocab_content
    pad_id = vocab_content
    if args.task == "identity":
        vocab_size = vocab_content + 1
        delim_id = vocab_content + 1
    else:
        delim_id = vocab_content + 1
        vocab_size = vocab_content + 2
    if args.task == "delimiter_mixed" and not (
        0.0 <= float(args.tail_keep_p) <= 1.0
    ):
        raise ValueError("--tail-keep-p must be in [0,1]")
    if args.task == "gap_copy":
        if int(args.gap_min) < 0 or int(args.gap_max) < int(args.gap_min):
            raise ValueError("gap_copy requires 0 <= gap-min <= gap-max")
        if not args.val_random_n:
            if int(args.val_gap) < int(args.gap_min) or int(args.val_gap) > int(
                args.gap_max
            ):
                raise ValueError("val-gap must lie in [gap-min, gap-max] for gap_copy")
    if args.lr_schedule == "cosine" and float(args.lr_min) >= float(args.lr):
        raise ValueError("cosine lr requires --lr-min < --lr")
    if args.task == "identity":
        max_len = args.n_max
    elif args.task == "gap_copy":
        max_len = int(args.n_max) + int(args.gap_max) + 1 + int(args.n_max)
    elif args.task == "dup_tail_copy":
        max_len = 3 * int(args.n_max) + 1
    else:
        max_len = 2 * args.n_max + 1
    dim_ff = int(args.ff_mult) * args.dim

    if args.model == "bottleneck_dual":
        if not (0 < args.r_mem < dim_ff):
            raise ValueError(
                f"For bottleneck_dual require 0 < r_mem < dim_ff; got r_mem={args.r_mem}, dim_ff={dim_ff}"
            )
    elif args.ff_readout != "dual":
        raise ValueError("--ff-readout fused only applies to --model bottleneck_dual")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_path = Path(args.val_set_file)
    if not val_path.is_absolute():
        val_path = out_dir / val_path
    print(f"Loading or building fixed val set -> {val_path} ...", flush=True)
    val_tokens, val_labels, val_pad_mask = ensure_fixed_val_set(
        args,
        val_path,
        max_len,
        vocab_content,
        pad_id,
        delim_id,
        device,
    )
    print("Val set ready.", flush=True)
    meta_path = out_dir / "val_meta.json"
    meta_path.write_text(
        json.dumps(val_set_meta_dict(args), indent=2), encoding="utf-8"
    )

    print(f"Building model ({args.model}) and moving to {device} ...", flush=True)
    if args.model == "transformer":
        model = SeqModelTransformer(
            vocab_size,
            args.dim,
            args.depth,
            args.nhead,
            dim_feedforward=dim_ff,
            activation="relu",
        ).to(device)
        model_name = "transformer"
        split_mode = ""
        r_mem_cell: int | str = ""
    else:
        sm: SplitMode = "learnable_ortho" if args.split_mode == "learnable_ortho" else "slice"
        fro: FFReadoutMode = "fused" if args.ff_readout == "fused" else "dual"
        model = SeqModelBottleneckDual(
            vocab_size,
            args.dim,
            args.depth,
            args.nhead,
            dim_ff,
            args.r_mem,
            split_mode=sm,
            ff_readout=fro,
            activation="relu",
        ).to(device)
        model_name = f"bottleneck_dual_{args.split_mode}_{args.ff_readout}"
        split_mode = args.split_mode
        r_mem_cell = args.r_mem

    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,} | dim_ff={dim_ff}", flush=True)

    _, best_acc = train_run(
        args,
        out_dir,
        model_name=model_name,
        model=model,
        device=device,
        vocab_content=vocab_content,
        pad_id=pad_id,
        delim_id=delim_id,
        max_len=max_len,
        val_tokens=val_tokens,
        val_labels=val_labels,
        val_pad_mask=val_pad_mask,
        n_params=n_params,
        split_mode=split_mode,
        r_mem=int(r_mem_cell) if r_mem_cell != "" else 0,
        dim_ff=dim_ff,
        ff_readout=args.ff_readout if args.model == "bottleneck_dual" else "n/a",
    )
    print(f"Done. best_val_acc={best_acc:.4f} | CSV under {out_dir}")

    write_note_experiments(out_dir / "note.md")


if __name__ == "__main__":
    main()
