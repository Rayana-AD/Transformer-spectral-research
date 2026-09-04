#!/usr/bin/env bash
# Minimal Marian decoder causal ablation: stable/spike configs × circular/causal-safe (seed 42 only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/results_probe_independent/schemeA_dec_causal_ablation_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
PY="${PY:-python}"
DEVICE="${DEVICE:-auto}"
MAX_TRAIN="${MAX_TRAIN:-32}"
MAX_EVAL="${MAX_EVAL:-8}"

run_one() {
  local rev="$1"
  local s="$2"
  local t="$3"
  local tm="$4"
  local sm="$5"
  local tag="$6"
  echo "========== $tag s=$s t=$t rev=$rev temporal=$tm spatial=$sm =========="
  ARGS=(
    run_probe_independent.py
    --probe-target decoder
    --alpha-spatial "$s"
    --alpha-temporal "$t"
    --seed 42
    --output-dir "$OUT"
    --exp-suffix "$tag"
    --max-train-batches "$MAX_TRAIN"
    --max-eval-batches "$MAX_EVAL"
    --device "$DEVICE"
    --temporal-mode "$tm"
    --spatial-mode "$sm"
    --export-probe-csv
  )
  if [ "$rev" = "1" ]; then
    ARGS+=(--reverse)
  fi
  "$PY" "${ARGS[@]}"
}

# Stable config (0.1, 0.2) spatial_first
run_one 0 0.1 0.2 circular_roll spatial_global "stable_circular"
run_one 0 0.1 0.2 causal_prev prefix_mean "stable_causal"

# Spike-prone config (0.2, 0.1) temporal_first
run_one 1 0.2 0.1 circular_roll spatial_global "spike_circular"
run_one 1 0.2 0.1 causal_prev prefix_mean "spike_causal"

echo "Done. Results in $OUT"
"$PY" experiments/summarize_decoder_causal_ablation.py "$OUT"
