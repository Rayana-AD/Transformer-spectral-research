# Shared synthetic "memory stress" sequences for script8 / script9.
# M1 gap_copy: [content | filler(gap) | DELIM | tail] — CE on tail = content.
# M2 dup_tail_copy: [content | DELIM | tail_A | tail_B] — CE only on tail_B = content.
# M3 niah (fixed layout): [left_hay | needle | right_hay | DELIM | junk] — CE on last |needle| = needle.
# M4 niah_prefix: [needle | filler | junk] — needle at positions 0..K-1, CE only on last K = needle (no DELIM in seq).
# M5 kv_assoc: shuffled [(key ARROW value) x P] (+ optional gaps) | DELIM | distractor | query_key ARROW.
# M6 story_assoc: random "story" prefix | shuffled fact triples | DELIM | distractor | content query (same CE).
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_LABEL = -100


def build_gap_copy_batch(
    batch_size: int,
    n_min: int,
    n_max: int,
    gap_min: int,
    gap_max: int,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    max_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    *,
    fixed_n: int | None = None,
    fixed_gap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if vocab_content < 1:
        raise ValueError("vocab_content must be >= 1")
    if pad_id != vocab_content or delim_id != vocab_content + 1:
        raise ValueError("pad_id / delim_id mismatch for gap_copy")
    if gap_min < 0 or gap_max < gap_min:
        raise ValueError("require 0 <= gap_min <= gap_max")
    tokens_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for _ in range(batch_size):
        if fixed_n is not None:
            n = int(fixed_n)
        elif generator is not None:
            n = int(torch.randint(n_min, n_max + 1, (1,), generator=generator).item())
        else:
            n = int(torch.randint(n_min, n_max + 1, (1,)).item())
        if fixed_gap is not None:
            gap = int(fixed_gap)
        elif generator is not None:
            gap = int(torch.randint(gap_min, gap_max + 1, (1,), generator=generator).item())
        else:
            gap = int(torch.randint(gap_min, gap_max + 1, (1,)).item())
        if n < 1:
            raise ValueError("gap_copy requires n >= 1")
        content = torch.randint(0, vocab_content, (n,), device=device)
        filler = torch.randint(0, vocab_content, (gap,), device=device)
        tail = torch.randint(0, vocab_content, (n,), device=device)
        seq = torch.cat(
            [
                content,
                filler,
                torch.tensor([delim_id], device=device, dtype=torch.long),
                tail,
            ],
            dim=0,
        )
        dpos = n + gap  # index of DELIM
        lab = torch.full((seq.shape[0],), IGNORE_LABEL, dtype=torch.long, device=device)
        lab[dpos + 1 : dpos + 1 + n] = content
        if __debug__:
            assert int(seq.shape[0]) == n + gap + 1 + n
            assert int((lab != IGNORE_LABEL).sum().item()) == n
            assert int(lab[dpos].item()) == IGNORE_LABEL
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


def build_dup_tail_copy_batch(
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
    if vocab_content < 1:
        raise ValueError("vocab_content must be >= 1")
    if pad_id != vocab_content or delim_id != vocab_content + 1:
        raise ValueError("pad_id / delim_id mismatch for dup_tail_copy")
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
            raise ValueError("dup_tail_copy requires n >= 1")
        content = torch.randint(0, vocab_content, (n,), device=device)
        tail_a = torch.randint(0, vocab_content, (n,), device=device)
        tail_b = torch.randint(0, vocab_content, (n,), device=device)
        seq = torch.cat(
            [
                content,
                torch.tensor([delim_id], device=device, dtype=torch.long),
                tail_a,
                tail_b,
            ],
            dim=0,
        )
        assert int(seq.shape[0]) == 3 * n + 1
        lab = torch.full((seq.shape[0],), IGNORE_LABEL, dtype=torch.long, device=device)
        # tail_B starts at index 2n+1, length n
        lab[2 * n + 1 : 3 * n + 1] = content
        if __debug__:
            assert int((lab != IGNORE_LABEL).sum().item()) == n
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


def build_niah_fixed_batch(
    batch_size: int,
    hay_each: int,
    needle_len: int,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    max_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimal fixed Needle-in-a-Haystack for encoder training (bidirectional attention).

    Layout per row: left_hay (hay_each) | needle (needle_len) | right_hay (hay_each)
    | DELIM | junk (needle_len random content tokens). Labels on the last needle_len
    positions = needle (must retrieve from context; junk differs from needle w.h.p.).
    """
    if vocab_content < 1:
        raise ValueError("vocab_content must be >= 1")
    if pad_id != vocab_content or delim_id != vocab_content + 1:
        raise ValueError("pad_id / delim_id mismatch for niah")
    if hay_each < 0 or needle_len < 1:
        raise ValueError("niah requires hay_each >= 0 and needle_len >= 1")
    core = 2 * hay_each + needle_len + 1 + needle_len
    if core > max_len:
        raise ValueError(f"niah core length {core} exceeds max_len {max_len}")

    cpu = torch.device("cpu")
    tokens_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for _ in range(batch_size):
        if generator is not None:
            left = torch.randint(
                0,
                vocab_content,
                (hay_each,),
                device=cpu,
                dtype=torch.long,
                generator=generator,
            ).to(device)
            needle = torch.randint(
                0,
                vocab_content,
                (needle_len,),
                device=cpu,
                dtype=torch.long,
                generator=generator,
            ).to(device)
            right = torch.randint(
                0,
                vocab_content,
                (hay_each,),
                device=cpu,
                dtype=torch.long,
                generator=generator,
            ).to(device)
            junk = torch.randint(
                0,
                vocab_content,
                (needle_len,),
                device=cpu,
                dtype=torch.long,
                generator=generator,
            ).to(device)
        else:
            left = torch.randint(0, vocab_content, (hay_each,), device=device)
            needle = torch.randint(0, vocab_content, (needle_len,), device=device)
            right = torch.randint(0, vocab_content, (hay_each,), device=device)
            junk = torch.randint(0, vocab_content, (needle_len,), device=device)
        seq = torch.cat(
            [
                left,
                needle,
                right,
                torch.tensor([delim_id], device=device, dtype=torch.long),
                junk,
            ],
            dim=0,
        )
        lab = torch.full((seq.shape[0],), IGNORE_LABEL, dtype=torch.long, device=device)
        lab[-needle_len:] = needle
        if seq.shape[0] < max_len:
            pad_n = max_len - seq.shape[0]
            seq = F.pad(seq, (0, pad_n), value=pad_id)
            lab = F.pad(lab, (0, pad_n), value=IGNORE_LABEL)
        tokens_list.append(seq)
        labels_list.append(lab)
    tokens = torch.stack(tokens_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    padding_mask = tokens == pad_id
    return tokens, labels, padding_mask


def build_kv_assoc_batch(
    batch_size: int,
    num_pairs: int,
    dist_min: int,
    dist_max: int,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    arrow_id: int,
    max_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    *,
    pair_gap_min: int = 0,
    pair_gap_max: int = 0,
    fixed_dist_len: int | None = None,
    fixed_query_pair_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Associative key→value recall with a distractor gap (encoder / bidirectional).

    Content tokens are 0..vocab_content-1, split into key pool [0, key_hi) and value pool
    [key_hi, vocab_content). Specials: pad_id, delim_id, arrow_id (must be consecutive after
    vocab_content).

    Per sequence: P disjoint (key, value) pairs. Presentation order is a random permutation of
    pairs (not slot-aligned with PE). Optional ``pair_gap`` random content tokens between
    consecutive triples. Then DELIM, ``d`` distractor tokens, query [q_key, ARROW]. CE only on
    the query ARROW predicts the value paired with q_key (content, not position).
    """
    if vocab_content < 4:
        raise ValueError("kv_assoc needs vocab_content >= 4")
    if int(pad_id) != int(vocab_content) or int(delim_id) != int(vocab_content) + 1:
        raise ValueError("kv_assoc: require pad_id=vocab_content, delim_id=vocab_content+1")
    if int(arrow_id) != int(vocab_content) + 2:
        raise ValueError("kv_assoc: require arrow_id=vocab_content+2")
    if num_pairs < 1:
        raise ValueError("kv_assoc: num_pairs >= 1")
    if dist_min < 0 or dist_max < dist_min:
        raise ValueError("kv_assoc: need 0 <= dist_min <= dist_max")
    pair_gap_min = int(pair_gap_min)
    pair_gap_max = int(pair_gap_max)
    if pair_gap_min < 0 or pair_gap_max < pair_gap_min:
        raise ValueError("kv_assoc: need 0 <= pair_gap_min <= pair_gap_max")
    key_hi = vocab_content // 2
    if key_hi < num_pairs or (vocab_content - key_hi) < num_pairs:
        raise ValueError(
            f"kv_assoc: vocab_content={vocab_content} too small for num_pairs={num_pairs} "
            f"(need key pool >= P and value pool >= P)"
        )
    gap_slot_max = max(0, num_pairs - 1) * pair_gap_max
    core_max = 3 * num_pairs + gap_slot_max + 1 + dist_max + 2
    if core_max > max_len:
        raise ValueError(f"kv_assoc core max length {core_max} exceeds max_len {max_len}")

    cpu = torch.device("cpu")
    tokens_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    key_pool = torch.arange(0, key_hi, device=cpu, dtype=torch.long)
    val_pool = torch.arange(key_hi, vocab_content, device=cpu, dtype=torch.long)
    for _ in range(batch_size):
        if fixed_dist_len is not None:
            d = int(fixed_dist_len)
        elif generator is not None:
            d = int(
                torch.randint(
                    dist_min, dist_max + 1, (1,), device=cpu, generator=generator
                ).item()
            )
        else:
            d = int(torch.randint(dist_min, dist_max + 1, (1,), device=cpu).item())
        if fixed_query_pair_idx is not None:
            q_idx = int(fixed_query_pair_idx) % num_pairs
        elif generator is not None:
            q_idx = int(
                torch.randint(0, num_pairs, (1,), device=cpu, generator=generator).item()
            )
        else:
            q_idx = int(torch.randint(0, num_pairs, (1,), device=cpu).item())

        if generator is not None:
            k_ix = torch.randperm(key_hi, generator=generator, device=cpu)[:num_pairs]
            v_ix = torch.randperm(
                vocab_content - key_hi, generator=generator, device=cpu
            )[:num_pairs]
        else:
            k_ix = torch.randperm(key_hi, device=cpu)[:num_pairs]
            v_ix = torch.randperm(vocab_content - key_hi, device=cpu)[:num_pairs]
        keys = key_pool[k_ix].to(device)
        vals = val_pool[v_ix].to(device)

        if generator is not None:
            perm = torch.randperm(num_pairs, device=cpu, generator=generator)
        else:
            perm = torch.randperm(num_pairs, device=cpu)
        chunks: list[torch.Tensor] = []
        for j in range(num_pairs):
            i = int(perm[j].item())
            chunks.append(keys[i].view(1))
            chunks.append(
                torch.tensor([arrow_id], device=device, dtype=torch.long)
            )
            chunks.append(vals[i].view(1))
            if j < num_pairs - 1 and pair_gap_max > 0:
                if generator is not None:
                    g = int(
                        torch.randint(
                            pair_gap_min,
                            pair_gap_max + 1,
                            (1,),
                            device=cpu,
                            generator=generator,
                        ).item()
                    )
                else:
                    g = int(
                        torch.randint(
                            pair_gap_min, pair_gap_max + 1, (1,), device=cpu
                        ).item()
                    )
                if g > 0:
                    if generator is not None:
                        gap_t = torch.randint(
                            0,
                            vocab_content,
                            (g,),
                            device=cpu,
                            dtype=torch.long,
                            generator=generator,
                        ).to(device)
                    else:
                        gap_t = torch.randint(
                            0, vocab_content, (g,), device=device, dtype=torch.long
                        )
                    chunks.append(gap_t)
        study = torch.cat(chunks, dim=0)
        delim_t = torch.tensor([delim_id], device=device, dtype=torch.long)
        if generator is not None:
            distr = torch.randint(
                0,
                vocab_content,
                (d,),
                device=cpu,
                dtype=torch.long,
                generator=generator,
            ).to(device)
        else:
            distr = torch.randint(0, vocab_content, (d,), device=device, dtype=torch.long)
        q_key = keys[q_idx].view(1)
        arrow_t = torch.tensor([arrow_id], device=device, dtype=torch.long)
        seq = torch.cat([study, delim_t, distr, q_key, arrow_t], dim=0)
        lab = torch.full((seq.shape[0],), IGNORE_LABEL, dtype=torch.long, device=device)
        lab[-1] = vals[q_idx]
        if seq.shape[0] < max_len:
            pad_n = max_len - seq.shape[0]
            seq = F.pad(seq, (0, pad_n), value=pad_id)
            lab = F.pad(lab, (0, pad_n), value=IGNORE_LABEL)
        tokens_list.append(seq)
        labels_list.append(lab)
    tokens = torch.stack(tokens_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    padding_mask = tokens == pad_id
    return tokens, labels, padding_mask


def build_story_assoc_batch(
    batch_size: int,
    num_pairs: int,
    story_open_min: int,
    story_open_max: int,
    pair_gap_min: int,
    pair_gap_max: int,
    dist_min: int,
    dist_max: int,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    arrow_id: int,
    max_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    *,
    fixed_open_len: int | None = None,
    fixed_dist_len: int | None = None,
    fixed_query_pair_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pseudo story: random opening tokens, then shuffled (entity ARROW detail) facts, DELIM,
    distractor, query [entity, ARROW]. CE on query ARROW predicts the detail (content QA, not
    ``which position''). Same key/value pools as kv_assoc."""
    if vocab_content < 4:
        raise ValueError("story_assoc needs vocab_content >= 4")
    if int(pad_id) != int(vocab_content) or int(delim_id) != int(vocab_content) + 1:
        raise ValueError("story_assoc: require pad_id=vocab_content, delim_id=vocab_content+1")
    if int(arrow_id) != int(vocab_content) + 2:
        raise ValueError("story_assoc: require arrow_id=vocab_content+2")
    if num_pairs < 1:
        raise ValueError("story_assoc: num_pairs >= 1")
    if story_open_min < 0 or story_open_max < story_open_min:
        raise ValueError("story_assoc: need 0 <= story_open_min <= story_open_max")
    pair_gap_min = int(pair_gap_min)
    pair_gap_max = int(pair_gap_max)
    if pair_gap_min < 0 or pair_gap_max < pair_gap_min:
        raise ValueError("story_assoc: need 0 <= pair_gap_min <= pair_gap_max")
    if dist_min < 0 or dist_max < dist_min:
        raise ValueError("story_assoc: need 0 <= dist_min <= dist_max")
    key_hi = vocab_content // 2
    if key_hi < num_pairs or (vocab_content - key_hi) < num_pairs:
        raise ValueError(
            f"story_assoc: vocab_content={vocab_content} too small for num_pairs={num_pairs}"
        )
    gap_slot_max = max(0, num_pairs - 1) * pair_gap_max
    core_max = (
        story_open_max
        + 3 * num_pairs
        + gap_slot_max
        + 1
        + dist_max
        + 2
    )
    if core_max > max_len:
        raise ValueError(
            f"story_assoc core max length {core_max} exceeds max_len {max_len}"
        )

    cpu = torch.device("cpu")
    tokens_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    key_pool = torch.arange(0, key_hi, device=cpu, dtype=torch.long)
    val_pool = torch.arange(key_hi, vocab_content, device=cpu, dtype=torch.long)
    for _ in range(batch_size):
        if fixed_open_len is not None:
            so = int(fixed_open_len)
        elif generator is not None:
            so = int(
                torch.randint(
                    story_open_min,
                    story_open_max + 1,
                    (1,),
                    device=cpu,
                    generator=generator,
                ).item()
            )
        else:
            so = int(
                torch.randint(
                    story_open_min, story_open_max + 1, (1,), device=cpu
                ).item()
            )
        if fixed_dist_len is not None:
            d = int(fixed_dist_len)
        elif generator is not None:
            d = int(
                torch.randint(
                    dist_min, dist_max + 1, (1,), device=cpu, generator=generator
                ).item()
            )
        else:
            d = int(torch.randint(dist_min, dist_max + 1, (1,), device=cpu).item())
        if fixed_query_pair_idx is not None:
            q_idx = int(fixed_query_pair_idx) % num_pairs
        elif generator is not None:
            q_idx = int(
                torch.randint(0, num_pairs, (1,), device=cpu, generator=generator).item()
            )
        else:
            q_idx = int(torch.randint(0, num_pairs, (1,), device=cpu).item())

        if generator is not None:
            k_ix = torch.randperm(key_hi, generator=generator, device=cpu)[:num_pairs]
            v_ix = torch.randperm(
                vocab_content - key_hi, generator=generator, device=cpu
            )[:num_pairs]
        else:
            k_ix = torch.randperm(key_hi, device=cpu)[:num_pairs]
            v_ix = torch.randperm(vocab_content - key_hi, device=cpu)[:num_pairs]
        keys = key_pool[k_ix].to(device)
        vals = val_pool[v_ix].to(device)

        if so > 0:
            if generator is not None:
                open_t = torch.randint(
                    0,
                    vocab_content,
                    (so,),
                    device=cpu,
                    dtype=torch.long,
                    generator=generator,
                ).to(device)
            else:
                open_t = torch.randint(
                    0, vocab_content, (so,), device=device, dtype=torch.long
                )
        else:
            open_t = torch.empty((0,), dtype=torch.long, device=device)

        if generator is not None:
            perm = torch.randperm(num_pairs, device=cpu, generator=generator)
        else:
            perm = torch.randperm(num_pairs, device=cpu)
        body_chunks: list[torch.Tensor] = []
        for j in range(num_pairs):
            i = int(perm[j].item())
            body_chunks.append(keys[i].view(1))
            body_chunks.append(
                torch.tensor([arrow_id], device=device, dtype=torch.long)
            )
            body_chunks.append(vals[i].view(1))
            if j < num_pairs - 1 and pair_gap_max > 0:
                if generator is not None:
                    g = int(
                        torch.randint(
                            pair_gap_min,
                            pair_gap_max + 1,
                            (1,),
                            device=cpu,
                            generator=generator,
                        ).item()
                    )
                else:
                    g = int(
                        torch.randint(
                            pair_gap_min, pair_gap_max + 1, (1,), device=cpu
                        ).item()
                    )
                if g > 0:
                    if generator is not None:
                        gap_t = torch.randint(
                            0,
                            vocab_content,
                            (g,),
                            device=cpu,
                            dtype=torch.long,
                            generator=generator,
                        ).to(device)
                    else:
                        gap_t = torch.randint(
                            0, vocab_content, (g,), device=device, dtype=torch.long
                        )
                    body_chunks.append(gap_t)
        body = torch.cat(body_chunks, dim=0)
        if so > 0:
            study = torch.cat([open_t, body], dim=0)
        else:
            study = body
        delim_t = torch.tensor([delim_id], device=device, dtype=torch.long)
        if generator is not None:
            distr = torch.randint(
                0,
                vocab_content,
                (d,),
                device=cpu,
                dtype=torch.long,
                generator=generator,
            ).to(device)
        else:
            distr = torch.randint(0, vocab_content, (d,), device=device, dtype=torch.long)
        q_key = keys[q_idx].view(1)
        arrow_t = torch.tensor([arrow_id], device=device, dtype=torch.long)
        seq = torch.cat([study, delim_t, distr, q_key, arrow_t], dim=0)
        lab = torch.full((seq.shape[0],), IGNORE_LABEL, dtype=torch.long, device=device)
        lab[-1] = vals[q_idx]
        if seq.shape[0] < max_len:
            pad_n = max_len - seq.shape[0]
            seq = F.pad(seq, (0, pad_n), value=pad_id)
            lab = F.pad(lab, (0, pad_n), value=IGNORE_LABEL)
        tokens_list.append(seq)
        labels_list.append(lab)
    tokens = torch.stack(tokens_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    padding_mask = tokens == pad_id
    return tokens, labels, padding_mask


def build_niah_prefix_batch(
    batch_size: int,
    needle_len: int,
    seq_len_min: int,
    seq_len_max: int,
    vocab_content: int,
    pad_id: int,
    delim_id: int,
    max_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    *,
    fixed_total_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefix needle: seq = needle (K tok) | filler (gap) | junk (K tok); labels only on last K = needle.

    filler and junk are random content tokens; junk is not supervised to match needle in the input
    (must retrieve needle from the prefix). delim_id is unused (kept for API parity with other builders).
    """
    del delim_id  # no special token in sequence; vocab still includes pad/delim for script10
    if vocab_content < 1:
        raise ValueError("vocab_content must be >= 1")
    if pad_id != vocab_content:
        raise ValueError("pad_id mismatch for niah_prefix")
    if needle_len < 1:
        raise ValueError("needle_len must be >= 1")
    lo = int(seq_len_min)
    hi = int(seq_len_max)
    if lo > hi or lo < 2 * needle_len + 1:
        raise ValueError("need seq_len_min <= seq_len_max and seq_len_min >= 2*needle_len+1")
    if hi > max_len:
        raise ValueError(f"seq_len_max {hi} exceeds max_len {max_len}")

    cpu = torch.device("cpu")
    tokens_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for _ in range(batch_size):
        if fixed_total_len is not None:
            core_len = int(fixed_total_len)
        elif generator is not None:
            core_len = int(
                torch.randint(lo, hi + 1, (1,), device=cpu, generator=generator).item()
            )
        else:
            core_len = int(torch.randint(lo, hi + 1, (1,), device=cpu).item())
        gap = core_len - 2 * needle_len
        if gap < 1:
            raise ValueError(f"core_len {core_len} too small for needle_len {needle_len}")
        if generator is not None:
            needle = torch.randint(
                0,
                vocab_content,
                (needle_len,),
                device=cpu,
                dtype=torch.long,
                generator=generator,
            ).to(device)
            filler = torch.randint(
                0,
                vocab_content,
                (gap,),
                device=cpu,
                dtype=torch.long,
                generator=generator,
            ).to(device)
            junk = torch.randint(
                0,
                vocab_content,
                (needle_len,),
                device=cpu,
                dtype=torch.long,
                generator=generator,
            ).to(device)
        else:
            needle = torch.randint(0, vocab_content, (needle_len,), device=device)
            filler = torch.randint(0, vocab_content, (gap,), device=device)
            junk = torch.randint(0, vocab_content, (needle_len,), device=device)
        seq = torch.cat([needle, filler, junk], dim=0)
        if int(seq.shape[0]) != core_len:
            raise RuntimeError("niah_prefix length mismatch")
        lab = torch.full((core_len,), IGNORE_LABEL, dtype=torch.long, device=device)
        lab[-needle_len:] = needle
        if seq.shape[0] < max_len:
            pad_n = max_len - seq.shape[0]
            seq = F.pad(seq, (0, pad_n), value=pad_id)
            lab = F.pad(lab, (0, pad_n), value=IGNORE_LABEL)
        tokens_list.append(seq)
        labels_list.append(lab)
    tokens = torch.stack(tokens_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    padding_mask = tokens == pad_id
    return tokens, labels, padding_mask
