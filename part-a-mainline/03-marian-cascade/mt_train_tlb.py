#!/usr/bin/env python3
"""IWSLT-style EN↔DE training with **自建** ``EncDecTransformerTLB`` + BLEU/PPL CSV.

默认 ``--tokenizer Helsinki-NLP/opus-mt-de-en`` 只复用 **词表**；**不**加载 OPUS-MT Marian 预训练权重。
若要在 **Helsinki 全量预训练模型** 上微调，请用 ``marian_mt_finetune_tlb.py``（``AutoModelForSeq2SeqLM``）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mt_transformer_tlb import EncDecTransformerTLB

DEFAULT_OUT = Path("results_mt_tlb")


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[str, str]:
        return self.pairs[i]


def load_iwslt2017_de_en(*, max_train: int | None, max_val: int | None) -> tuple[list, list]:
    from datasets import load_dataset

    ds = load_dataset("iwslt2017", "iwslt2017-de-en")
    train = ds["train"]
    val = ds["validation"]
    pairs_tr: list[tuple[str, str]] = []
    pairs_va: list[tuple[str, str]] = []
    for i, ex in enumerate(train):
        if max_train is not None and i >= max_train:
            break
        pairs_tr.append((ex["translation"]["de"], ex["translation"]["en"]))
    for i, ex in enumerate(val):
        if max_val is not None and i >= max_val:
            break
        pairs_va.append((ex["translation"]["de"], ex["translation"]["en"]))
    return pairs_tr, pairs_va


def load_opus_books_de_en(*, max_train: int | None, max_val: int | None, seed: int) -> tuple[list, list]:
    """Helsinki-NLP/opus_books de-en (~51k pairs); Parquet-based, works with current ``datasets``."""
    from datasets import load_dataset

    ds = load_dataset("Helsinki-NLP/opus_books", "de-en")
    split = ds["train"].train_test_split(test_size=0.02, seed=seed)
    train, val = split["train"], split["test"]
    pairs_tr: list[tuple[str, str]] = []
    pairs_va: list[tuple[str, str]] = []
    for i, ex in enumerate(train):
        if max_train is not None and i >= max_train:
            break
        pairs_tr.append((ex["translation"]["de"], ex["translation"]["en"]))
    for i, ex in enumerate(val):
        if max_val is not None and i >= max_val:
            break
        pairs_va.append((ex["translation"]["de"], ex["translation"]["en"]))
    return pairs_tr, pairs_va


def _load_jsonl_pairs(path: str) -> list[tuple[str, str]]:
    import json
    pairs: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            pairs.append((obj["de"], obj["en"]))
    return pairs


def load_local_fixed_pairs() -> tuple[list, list]:
    """Load pre-exported local OPUS de-en data (no HF Hub required)."""
    data_dir = Path("data/opus_de_en")
    train_path = data_dir / "train_5k.jsonl"
    val_path = data_dir / "val_500.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Local data not found: {train_path} or {val_path}. "
            "Run dataset export first or use --dataset opus_books to download."
        )
    train_pairs = _load_jsonl_pairs(str(train_path))
    val_pairs = _load_jsonl_pairs(str(val_path))
    return train_pairs, val_pairs


def load_synthetic_pairs(n: int, seed: int) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    de_words = ["das", "ist", "ein", "test", "haus", "kind", "gut", "tag"]
    en_words = ["the", "is", "a", "test", "house", "child", "good", "day"]
    pairs: list[tuple[str, str]] = []
    for _ in range(n):
        k = int(rng.integers(3, 8))
        di = " ".join(rng.choice(de_words, size=k).tolist())
        ei = " ".join(rng.choice(en_words, size=k).tolist())
        pairs.append((di, ei))
    split = max(1, int(0.9 * n))
    return pairs[:split], pairs[split:]


def collate_batch(
    batch: list[tuple[str, str]],
    tokenizer,
    src_lang_first: bool,
    max_src_len: int,
    max_tgt_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns src, tgt_in, tgt_out, pad_mask for decoder."""
    src_texts = []
    tgt_texts = []
    for s, t in batch:
        if src_lang_first:
            src_texts.append(s)
            tgt_texts.append(t)
        else:
            src_texts.append(t)
            tgt_texts.append(s)
    enc = tokenizer(
        src_texts,
        padding=True,
        truncation=True,
        max_length=max_src_len,
        return_tensors="pt",
    )
    dec = tokenizer(
        tgt_texts,
        padding=True,
        truncation=True,
        max_length=max_tgt_len,
        return_tensors="pt",
    )
    src = enc["input_ids"].to(device)
    tgt_full = dec["input_ids"].to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    # teacher forcing: decoder input = full[:-1], labels = full[1:]
    tgt_in = tgt_full[:, :-1].contiguous()
    tgt_out = tgt_full[:, 1:].contiguous()
    tgt_pad = tgt_in == pad_id
    return src, tgt_in, tgt_out, tgt_pad


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    ntok = 0
    for batch in loader:
        src, tgt_in, tgt_out, tgt_pad = collate_batch(
            batch,
            tokenizer,
            args.src_lang_de,
            args.max_src_len,
            args.max_tgt_len,
            device,
        )
        pad_id = tokenizer.pad_token_id or 0
        src_key_padding_mask = src == pad_id
        tgt_key_padding_mask = tgt_in == pad_id
        opt.zero_grad()
        logits = model(
            src,
            tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat = tgt_out.reshape(-1)
        loss = nn.functional.cross_entropy(logits_flat, tgt_flat, ignore_index=pad_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        opt.step()
        total_loss += loss.item() * tgt_flat.numel()
        ntok += tgt_flat.numel()
    return total_loss / max(1, ntok)


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    ntok = 0
    pad_id = tokenizer.pad_token_id or 0
    for batch in loader:
        src, tgt_in, tgt_out, _ = collate_batch(
            batch,
            tokenizer,
            args.src_lang_de,
            args.max_src_len,
            args.max_tgt_len,
            device,
        )
        src_key_padding_mask = src == pad_id
        tgt_key_padding_mask = tgt_in == pad_id
        logits = model(
            src,
            tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat = tgt_out.reshape(-1)
        loss = nn.functional.cross_entropy(logits_flat, tgt_flat, ignore_index=pad_id)
        total_loss += loss.item() * tgt_flat.numel()
        ntok += tgt_flat.numel()
    ce = total_loss / max(1, ntok)
    ppl = math.exp(min(20.0, ce))
    return ce, ppl


def _decoder_start_token_id(tokenizer) -> int:
    """Prefer seq2seq decoder start (Marian/T5-style); fallback BOS/CLS/PAD."""
    ds = getattr(tokenizer, "decoder_start_token_id", None)
    if ds is not None:
        return int(ds)
    for cand in (tokenizer.bos_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id):
        if cand is not None:
            return int(cand)
    return 0


def _normalize_text_for_bleu(tokenizer, text: str) -> str:
    """Encode+decode so references match tokenizer whitespace/special handling vs hyps."""
    enc = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(enc, skip_special_tokens=True).strip()


@torch.no_grad()
def eval_token_accuracy(
    model: nn.Module,
    loader: DataLoader,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """Teacher-forcing: fraction of non-pad target tokens predicted correctly (closer to 'accuracy' than BLEU)."""
    model.eval()
    pad_id = tokenizer.pad_token_id or 0
    correct = 0
    total = 0
    for batch in loader:
        src, tgt_in, tgt_out, _ = collate_batch(
            batch,
            tokenizer,
            args.src_lang_de,
            args.max_src_len,
            args.max_tgt_len,
            device,
        )
        src_key_padding_mask = src == pad_id
        tgt_key_padding_mask = tgt_in == pad_id
        logits = model(
            src,
            tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        pred = logits.argmax(dim=-1)
        m = tgt_out != pad_id
        correct += int((pred == tgt_out).masked_select(m).sum().item())
        total += int(m.sum().item())
    return correct / max(1, total)


@torch.no_grad()
def eval_bleu(
    model: nn.Module,
    loader: DataLoader,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """
    Corpus BLEU on greedy decoding vs references. Not classification accuracy: it rewards n-gram
    overlap only. Often stays 0 with (1) random synthetic pairs, (2) few epochs, (3) greedy search.
    See val_token_acc for teacher-forcing overlap.
    """
    from sacrebleu import corpus_bleu

    model.eval()
    hyps: list[str] = []
    refs: list[str] = []
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.sep_token_id or pad_id
    max_gen = args.max_tgt_len
    start_id = _decoder_start_token_id(tokenizer)
    for batch in loader:
        src, _, _, _ = collate_batch(
            batch,
            tokenizer,
            args.src_lang_de,
            args.max_src_len,
            args.max_tgt_len,
            device,
        )
        src_key_padding_mask = src == pad_id
        mem = model.encode(src, src_key_padding_mask=src_key_padding_mask)
        b = src.size(0)
        tgt = torch.full((b, 1), start_id, dtype=torch.long, device=device)
        for _ in range(max_gen - 1):
            tpm = tgt == pad_id
            logits = model.decode(
                tgt,
                mem,
                tgt_key_padding_mask=tpm,
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        for bi in range(b):
            seq = tgt[bi].tolist()
            if eos_id is not None and eos_id in seq:
                seq = seq[: seq.index(eos_id)]
            hyps.append(tokenizer.decode(seq, skip_special_tokens=True))
        for s, t in batch:
            raw = t if args.src_lang_de else s
            refs.append(_normalize_text_for_bleu(tokenizer, raw))
    return float(corpus_bleu(hyps, [refs]).score)


def main() -> None:
    p = argparse.ArgumentParser(description="MT training: IWSLT2017 de-en (or synthetic) with TLB attention")
    p.add_argument(
        "--attention",
        type=str,
        default="tlb",
        choices=(
            "vanilla",
            "tlb",
            "frob",
            "l2",
            "sinkhorn",
            "nuc_inter",
            "kv_mul",
            "kv_inner",
            "kv_x_inter",
            "kv_inner_nuc_inter",
            "qv_inner",
            "randv_inner",
            "nuc_inter_q",
            "v_jitter",
        ),
    )
    p.add_argument(
        "--jitter-std",
        type=float,
        default=0.0,
        help="For --attention v_jitter: noise std on V (if 0, uses 1e-3).",
    )
    p.add_argument("--tlb-eps", type=float, default=1e-8)
    p.add_argument("--sinkhorn-iters", type=int, default=5)
    p.add_argument("--post-attn-ln", action="store_true")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-encoder-layers", type=int, default=3)
    p.add_argument("--num-decoder-layers", type=int, default=3)
    p.add_argument("--dim-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--max-src-len", type=int, default=96)
    p.add_argument("--max-tgt-len", type=int, default=96)
    p.add_argument("--max-train", type=int, default=None, help="Cap training pairs (debug).")
    p.add_argument("--max-val", type=int, default=None, help="Cap val pairs.")
    p.add_argument(
        "--dataset",
        type=str,
        default="opus_books",
        choices=("opus_books", "iwslt2017", "synthetic", "local"),
        help="Parallel de→en data: opus_books (default, HF Parquet, ~51k train split), "
        "iwslt2017 (legacy loader; may fail on datasets>=4 with script datasets disabled), "
        "synthetic (debug only), local (pre-exported data/opus_de_en/, no HF Hub).",
    )
    p.add_argument("--synthetic-samples", type=int, default=5000)
    p.add_argument(
        "--src-lang-de",
        action="store_true",
        default=True,
        help="Source=German, target=English (IWSLT de-en).",
    )
    p.add_argument("--src-lang-en", action="store_true", help="Swap: source=English, target=German.")
    p.add_argument("--tokenizer", type=str, default="Helsinki-NLP/opus-mt-de-en")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--bleu-every", type=int, default=1, help="Compute BLEU every N epochs (greedy decode; slow).")
    p.add_argument("--tlb-warmup-epochs", type=int, default=0, help="Epochs to linearly warmup TLB normalization (0=full TLB from start).")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training: load optional --checkpoint, run val CE/PPL, token acc, and BLEU on val set (forward-only).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to EncDecTransformerTLB state_dict (.pt). Used with --eval-only or to init before training.",
    )
    p.add_argument(
        "--save-checkpoint",
        type=str,
        default="",
        help="If set, save model.state_dict() to this path after training completes.",
    )
    p.add_argument(
        "--skip-bleu-eval",
        action="store_true",
        help="With --eval-only: skip greedy BLEU (only CE/PPL/token acc).",
    )
    args = p.parse_args()
    if args.src_lang_en:
        args.src_lang_de = False
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except ValueError:
        # Newer transformers may not map Marian to AutoTokenizer
        from transformers import MarianTokenizer

        tokenizer = MarianTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if args.dataset == "local":
        train_pairs, val_pairs = load_local_fixed_pairs()
        if args.max_train is not None:
            train_pairs = train_pairs[: args.max_train]
        if args.max_val is not None:
            val_pairs = val_pairs[: args.max_val]
    elif args.dataset == "opus_books":
        train_pairs, val_pairs = load_opus_books_de_en(
            max_train=args.max_train,
            max_val=args.max_val,
            seed=args.seed,
        )
    elif args.dataset == "iwslt2017":
        try:
            train_pairs, val_pairs = load_iwslt2017_de_en(
                max_train=args.max_train,
                max_val=args.max_val,
            )
        except Exception as e:
            print(
                f"[mt_train_tlb] IWSLT load failed ({e}). "
                "Use --dataset opus_books (recommended) or pin an older ``datasets`` with script support.",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
    else:
        train_pairs, val_pairs = load_synthetic_pairs(args.synthetic_samples, args.seed)

    train_ds = PairDataset(train_pairs)
    val_ds = PairDataset(val_pairs)
    def _collate_pairs(batch):
        return batch

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=_collate_pairs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_pairs,
    )

    src_vocab = tokenizer.vocab_size
    tgt_vocab = tokenizer.vocab_size
    model = EncDecTransformerTLB(
        src_vocab,
        tgt_vocab,
        args.d_model,
        args.nhead,
        args.num_encoder_layers,
        args.num_decoder_layers,
        args.dim_ff,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        dropout=args.dropout,
        attention_mode=args.attention,  # type: ignore[arg-type]
        tlb_eps=args.tlb_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        post_attn_layer_norm=args.post_attn_ln,
        tlb_warmup_epochs=args.tlb_warmup_epochs,
        jitter_std=args.jitter_std,
    ).to(device)

    ckpt_in = args.checkpoint.strip()
    if ckpt_in:
        try:
            blob = torch.load(ckpt_in, map_location=device, weights_only=True)
        except TypeError:
            blob = torch.load(ckpt_in, map_location=device)
        model.load_state_dict(blob, strict=True)
        print(f"[mt_train_tlb] Loaded checkpoint {ckpt_in}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"mt_metrics_{args.attention}_seed{args.seed}.csv"
    rows: list[dict[str, float | str | int]] = []

    if args.eval_only:
        if not ckpt_in:
            print(
                "[mt_train_tlb] --eval-only without --checkpoint: metrics are for a **randomly initialized** model "
                "(useful only for pipeline/debug).",
                flush=True,
            )
        va_ce, va_ppl = eval_loss(model, val_loader, tokenizer, args, device)
        val_tok_acc = eval_token_accuracy(model, val_loader, tokenizer, args, device)
        bleu = float("nan")
        if not args.skip_bleu_eval:
            bleu = eval_bleu(model, val_loader, tokenizer, args, device)
        eval_payload = {
            "mode": "eval_only",
            "dataset": args.dataset,
            "attention": args.attention,
            "seed": args.seed,
            "checkpoint": ckpt_in or None,
            "val_ce_token": va_ce,
            "val_ppl": va_ppl,
            "val_token_acc": val_tok_acc,
            "bleu": bleu,
        }
        jpath = out / f"eval_only_{args.attention}_seed{args.seed}.json"
        jpath.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
        print(
            f"[eval-only] val_ce {va_ce:.4f} | val_ppl {va_ppl:.2f} | val_tok_acc {val_tok_acc:.4f} | bleu {bleu}",
            flush=True,
        )
        print(f"Wrote {jpath}", flush=True)
        return

    for epoch in range(args.epochs):
        # Set current epoch for TLB warmup scheduling
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        tr_ce = train_epoch(model, train_loader, opt, tokenizer, args, device)
        va_ce, va_ppl = eval_loss(model, val_loader, tokenizer, args, device)
        val_tok_acc = eval_token_accuracy(model, val_loader, tokenizer, args, device)
        bleu = float("nan")
        if args.bleu_every > 0 and (epoch + 1) % args.bleu_every == 0:
            bleu = eval_bleu(model, val_loader, tokenizer, args, device)
        row = {
            "epoch": epoch,
            "train_ce_token": tr_ce,
            "val_ce_token": va_ce,
            "val_ppl": va_ppl,
            "val_token_acc": val_tok_acc,
            "bleu": bleu,
            "attention": args.attention,
        }
        rows.append(row)
        print(
            f"epoch {epoch} | train_ce {tr_ce:.4f} | val_ce {va_ce:.4f} | val_ppl {va_ppl:.2f} | "
            f"val_tok_acc {val_tok_acc:.4f} | bleu {bleu}",
            flush=True,
        )

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {csv_path}", flush=True)

    save_ckpt = args.save_checkpoint.strip()
    if save_ckpt:
        torch.save(model.state_dict(), save_ckpt)
        print(f"[mt_train_tlb] Saved checkpoint {save_ckpt}", flush=True)


if __name__ == "__main__":
    main()
