#!/usr/bin/env bash
# Minimal regularization baselines vs cascade (stable decoder config, seed 42, 32 train batches).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/results_probe_independent/rebuttal_baselines_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
PY="${PY:-$ROOT/.venv_minepy/bin/python}"
if [ ! -x "$PY" ]; then
  PY="python3"
fi
DEVICE="${DEVICE:-auto}"
MAX_TRAIN="${MAX_TRAIN:-32}"
MAX_EVAL="${MAX_EVAL:-8}"

common=(
  run_probe_independent.py
  --probe-target decoder
  --seed 42
  --output-dir "$OUT"
  --max-train-batches "$MAX_TRAIN"
  --max-eval-batches "$MAX_EVAL"
  --device "$DEVICE"
  --temporal-mode causal_prev
  --spatial-mode prefix_mean
)

echo "========== Cascade stable (0.1, 0.2) =========="
"$PY" "${common[@]}" --alpha-spatial 0.1 --alpha-temporal 0.2 --exp-suffix cascade_stable

echo "========== Vanilla finetune (no cascade) =========="
"$PY" "${common[@]}" --no-cascade --exp-suffix vanilla_finetune

echo "========== Vanilla + label smoothing 0.1 =========="
"$PY" "${common[@]}" --no-cascade --label-smoothing 0.1 --exp-suffix label_smooth_0.1

echo "========== Vanilla + attention dropout 0.2 =========="
"$PY" "${common[@]}" --no-cascade --attention-dropout 0.2 --exp-suffix attn_dropout_0.2

echo "Done. Results in $OUT"
"$PY" - <<PY
import json
from pathlib import Path
out = Path("$OUT")
rows = []
for p in sorted(out.glob("*/result.json")):
    d = json.loads(p.read_text())
    rows.append({
        "exp": d.get("exp_name", p.parent.name),
        "cascade": d.get("enable_cascade", True),
        "label_smoothing": d.get("label_smoothing", 0),
        "attn_dropout": d.get("attention_dropout"),
        "bleu": round(float(d.get("bleu", 0)), 2),
        "train_ppl": round(float(d.get("train_ppl", 0)), 1),
        "eval_ppl": round(float(d.get("eval_ppl", 0)), 1) if d.get("eval_ppl") else None,
    })
md = out / "baseline_summary.md"
lines = ["# Regularization baseline summary", "", "| exp | cascade | label_smooth | attn_dropout | BLEU | train_ppl | eval_ppl |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
for r in rows:
    lines.append(
        f"| {r['exp']} | {r['cascade']} | {r['label_smoothing']} | {r['attn_dropout']} | {r['bleu']} | {r['train_ppl']} | {r['eval_ppl']} |"
    )
md.write_text("\\n".join(lines))
print(f"Wrote {md}")
for r in rows:
    print(r)
PY
