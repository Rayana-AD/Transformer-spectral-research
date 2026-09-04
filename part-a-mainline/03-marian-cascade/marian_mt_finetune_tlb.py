#!/usr/bin/env python3
"""Helsinki OPUS-MT Marian: 加载预训练 ``AutoModelForSeq2SeqLM``，微调与 BLEU。

与 ``mt_train_tlb``（**自建** ``EncDecTransformerTLB`` + 仅复用 **tokenizer=Helsinki**）**不同**：
本脚本在 **HuggingFace Marian 全权重** 上工作；``--attention`` 非 ``vanilla`` 时需
``attn_implementation='eager'`` + ``marian_attn_tlb_patch``。
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from marian_attn_tlb_patch import apply_marian_attention, uninstall_marian_tlb
from mt_train_tlb import (
    load_iwslt2017_de_en,
    load_local_fixed_pairs,
    load_opus_books_de_en,
    load_synthetic_pairs,
)


def _get_device(s: str) -> torch.device:
    if s == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if s == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Pairs(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[str, str]:
        return self.pairs[i]


def _encode_batch(
    tokenizer, batch: list[tuple[str, str]], max_src: int, max_tgt: int, device: torch.device, pad_id: int
) -> dict:
    srcs, tgts = [b[0] for b in batch], [b[1] for b in batch]
    enc = tokenizer(
        srcs, padding=True, truncation=True, max_length=max_src, return_tensors="pt"
    )
    _cm = (
        tokenizer.as_target_tokenizer()
        if hasattr(tokenizer, "as_target_tokenizer")
        else contextlib.nullcontext()
    )
    with _cm:
        tenc = tokenizer(
            tgts, padding=True, truncation=True, max_length=max_tgt, return_tensors="pt"
        )
    lab = tenc["input_ids"].clone()
    lab[lab == pad_id] = -100
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "labels": lab.to(device),
    }


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=str, default="Helsinki-NLP/opus-mt-de-en")
    p.add_argument(
        "--attention",
        type=str,
        default="vanilla",
        choices=("vanilla", "kv_inner", "nuc_inter", "nuc_inter_q", "nuc_inter_q_learned", "nuc_inter_bias", "v_jitter", "v_smooth", "v_smooth_mean", "v_smooth_roll", "v_smooth_hybrid", "v_smooth_cascade", "v_smooth_cascade_rev", "v_smooth_post", "v_smooth_feature", "v_smooth_both", "line_c"),
    )
    p.add_argument("--tlb-eps", type=float, default=1e-8)
    p.add_argument("--jitter-std", type=float, default=0.0)
    p.add_argument("--smooth-alpha", type=float, default=0.1, help="V smoothing factor for v_smooth (default 0.1)")
    p.add_argument("--alpha-temporal", type=float, default=0.1, help="Temporal (roll) component weight for v_smooth_hybrid/cascade (default 0.1)")
    p.add_argument("--alpha-global", type=float, default=0.1, help="Global (mean) component weight for v_smooth_hybrid (default 0.1)")
    p.add_argument("--alpha-spatial", type=float, default=0.1, help="Spatial (mean-first) weight for v_smooth_cascade (default 0.1)")
    p.add_argument("--nuc-bias", type=float, default=0.9, help="Intentional bias factor for nuc_inter_bias (default 0.9)")
    p.add_argument("--dataset", type=str, default="local", choices=("local", "opus_books", "iwslt2017", "synthetic"))
    p.add_argument("--synthetic-samples", type=int, default=5000)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-src-len", type=int, default=128)
    p.add_argument("--max-tgt-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--output-dir", type=str, default="results_marian_tlb")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="HF 目录 (save_pretrained) 以加载已微调；留空=仅用预训练或配合训练保存路径",
    )
    p.add_argument("--skip-bleu-eval", action="store_true")
    p.add_argument("--bleu-every", type=int, default=1, help="训练时每 N epoch 打 BLEU (0=不打)")
    p.add_argument("--log-every-steps", type=int, default=0)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


@torch.no_grad()
def _eval_val(
    model: nn.Module, loader, tokenizer, max_src: int, max_tgt: int, device: torch.device, do_bleu: bool
) -> tuple[float, float, float | None]:
    from sacrebleu import corpus_bleu

    model.eval()
    pad_id = tokenizer.pad_token_id or 0
    n_loss, d_loss = 0.0, 0
    for batch in loader:
        enc = _encode_batch(tokenizer, batch, max_src, max_tgt, device, pad_id)
        o = model(**enc)
        nt = (enc["labels"] != -100).sum().item()
        n_loss += float(o.loss) * max(1, nt)
        d_loss += max(1, nt)
    ce = n_loss / d_loss
    ppl = math.exp(min(20.0, ce))
    if not do_bleu:
        return ce, ppl, None
    hyps, refs = [], []
    for batch in loader:
        srcs = [b[0] for b in batch]
        enc = tokenizer(
            srcs, padding=True, truncation=True, max_length=max_src, return_tensors="pt"
        )
        g = model.generate(
            enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            max_length=max_tgt,
            num_beams=1,
            do_sample=False,
        )
        for i in range(g.size(0)):
            hyps.append(tokenizer.decode(g[i], skip_special_tokens=True).strip())
        for _, t in batch:
            refs.append(t.strip())
    bl = float(corpus_bleu(hyps, [refs]).score) if hyps else 0.0
    return ce, ppl, bl


def _train_epoch(
    model, loader, tok, max_src, max_tgt, dev, opt, pad_id, max_norm, log_every: int
) -> float:
    model.train()
    s, c = 0.0, 0
    step = 0
    for batch in loader:
        enc = _encode_batch(tok, batch, max_src, max_tgt, dev, pad_id)
        opt.zero_grad()
        o = model(**enc)
        o.loss.backward()
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        opt.step()
        s += float(o.loss.item())
        c += 1
        step += 1
        if log_every and step % log_every == 0:
            print(f"[step {step}] loss={o.loss.item():.4f}", flush=True)
    return s / max(1, c)


def _load_model(
    model_id: str, device: torch.device, attention: str, local_only: bool, ckpt_dir: str | None
):
    from transformers import AutoModelForSeq2SeqLM

    use_e = attention != "vanilla"
    if ckpt_dir and Path(ckpt_dir).is_dir():
        m = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir, local_files_only=local_only, attn_implementation="eager" if use_e else "sdpa"
        )
    else:
        m = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, local_files_only=local_only, attn_implementation="eager" if use_e else "sdpa"
        )
    m.to(device)
    return m


def main() -> None:
    args = _parse()
    if args.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = _get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoTokenizer, MarianTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    except Exception:
        tok = MarianTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    pad_id = tok.pad_token_id or 0
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.dataset == "local":
        tr, va = load_local_fixed_pairs()
    elif args.dataset == "opus_books":
        tr, va = load_opus_books_de_en(
            max_train=args.max_train, max_val=args.max_val, seed=args.seed
        )
    elif args.dataset == "iwslt2017":
        tr, va = load_iwslt2017_de_en(max_train=args.max_train, max_val=args.max_val)
    else:
        tr, va = load_synthetic_pairs(args.synthetic_samples, args.seed)
    if args.max_train is not None:
        tr = tr[: args.max_train]
    if args.max_val is not None:
        va = va[: args.max_val]

    tr_ld = DataLoader(
        _Pairs(tr), batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: b
    )
    va_ld = DataLoader(_Pairs(va), batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: b)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    att = args.attention
    save_dir = out / f"hf_finetuned_{att}_seed{seed}"
    metrics_csv = out / f"marian_metrics_{att}_seed{seed}.csv"
    eval_out = out / f"marian_eval_{att}_seed{seed}.json"

    ck = args.checkpoint.strip() or None
    if args.eval_only and ck and Path(ck).is_dir():
        model = _load_model(
            args.model_id, dev, att, args.local_files_only, ck
        )
    else:
        model = _load_model(
            args.model_id, dev, att, args.local_files_only, None
        )
        if ck and Path(ck).is_file() and str(ck).endswith(".pt"):
            blob = torch.load(ck, map_location=dev, weights_only=True)
            model.load_state_dict(blob, strict=True)
            print(f"[marian] loaded .pt {ck}", flush=True)

    apply_marian_attention(att, tlb_eps=args.tlb_eps, jitter_std=args.jitter_std, smooth_alpha=args.smooth_alpha, alpha_temporal=args.alpha_temporal, alpha_global=args.alpha_global, alpha_spatial=args.alpha_spatial, nuc_bias=args.nuc_bias)

    if args.eval_only:
        ce, ppl, bleu = _eval_val(
            model, va_ld, tok, args.max_src_len, args.max_tgt_len, dev, not args.skip_bleu_eval
        )
        d = {
            "val_ce": ce,
            "val_ppl": ppl,
            "bleu": bleu,
            "attention": att,
            "seed": seed,
            "checkpoint": ck,
        }
        eval_out.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[eval-only] {d}", flush=True)
        uninstall_marian_tlb()
        return

    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    rows: list[dict] = []
    for ep in range(args.epochs):
        tl = _train_epoch(
            model, tr_ld, tok, args.max_src_len, args.max_tgt_len, dev, opt, pad_id, args.clip_norm, args.log_every_steps
        )
        want_bleu = (args.bleu_every > 0) and (ep + 1) % args.bleu_every == 0 and (not args.skip_bleu_eval)
        ce, ppl, bl = _eval_val(
            model, va_ld, tok, args.max_src_len, args.max_tgt_len, dev, want_bleu
        )
        if not want_bleu:
            bl = float("nan")
        rows.append(
            {
                "epoch": ep,
                "train_loss_mean": tl,
                "val_ce": ce,
                "val_ppl": ppl,
                "bleu": bl,
            }
        )
        print(
            f"epoch {ep} train_loss {tl:.4f} val_ce {ce:.4f} ppl {ppl:.2f} bleu {bl}",
            flush=True,
        )
    if rows:
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {metrics_csv}", flush=True)
    model.save_pretrained(str(save_dir))
    tok.save_pretrained(str(save_dir))
    if not args.skip_bleu_eval:
        _, _, bleu = _eval_val(
            model, va_ld, tok, args.max_src_len, args.max_tgt_len, dev, True
        )
        eval_out.write_text(
            json.dumps(
                {
                    "bleu": bleu,
                    "hf_dir": str(save_dir),
                    "attention": att,
                    "seed": seed,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {eval_out} bleu={bleu}", flush=True)
    uninstall_marian_tlb()


if __name__ == "__main__":
    main()
