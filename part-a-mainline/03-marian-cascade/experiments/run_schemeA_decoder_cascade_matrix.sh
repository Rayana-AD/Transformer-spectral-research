#!/usr/bin/env bash
# Decoder 方案 A：级联探针 18 次（与 Encoder 矩阵相同：3 组 α × 2 种顺序 × 3 seed）
# 级联仅作用于 Decoder **自注意**；Cross-Attn 不修改。
# 额外：训练步梯度范数、Sparse Linear CKA（默认 dec_2/dec_5）、
#       generate 上历史窗口 BLEU（4 / 16 / 64 / full）。
#
# 用法：./experiments/run_schemeA_decoder_cascade_matrix.sh
#  可选：OUT_ROOT=... SEEDS=... MAX_TRAIN=32 MAX_EVAL=8 DEVICE=auto

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-tlb_mine}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results_probe_independent/schemeA_dec_3x2x3_$(date +%Y%m%d_%H%M)}"
SEEDS="${SEEDS:-42 43 44}"
MAX_TRAIN="${MAX_TRAIN:-32}"
MAX_EVAL="${MAX_EVAL:-8}"
VENV_MINEPY="$ROOT/.venv_minepy/bin/python"
VENV_DEFAULT="$ROOT/.venv/bin/python"

if [ -x "$VENV_MINEPY" ]; then
  RUN_CMD=( "$VENV_MINEPY" )
  DEVICE="${DEVICE:-auto}"
  export TLB_DEVICE="${TLB_DEVICE:-$DEVICE}"
  RUN_LABEL=".venv_minepy"
elif [ -x "$VENV_DEFAULT" ]; then
  RUN_CMD=( "$VENV_DEFAULT" )
  DEVICE="${DEVICE:-auto}"
  export TLB_DEVICE="${TLB_DEVICE:-$DEVICE}"
  RUN_LABEL=".venv"
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "未找到 $VENV_MINEPY、$VENV_DEFAULT 且无 conda" >&2
    exit 1
  fi
  if ! conda run -n "$CONDA_ENV" true 2>/dev/null; then
    echo "请创建 $CONDA_ENV 或本仓库的 .venv / .venv_minepy" >&2
    exit 1
  fi
  RUN_CMD=( conda run -n "$CONDA_ENV" --no-capture-output python )
  DEVICE="${DEVICE:-cpu}"
  export TLB_DEVICE="${TLB_DEVICE:-$DEVICE}"
  RUN_LABEL="conda:$CONDA_ENV"
fi

mkdir -p "$OUT_ROOT"
TS="$(date "+%Y-%m-%dT%H:%M:%S%z" 2>/dev/null || date)"
{
  echo "LOG: $TS OUT_ROOT=$OUT_ROOT MAX_TRAIN=$MAX_TRAIN MAX_EVAL=$MAX_EVAL TLB_DEVICE=$TLB_DEVICE RUN=$RUN_LABEL SCHEME=decoder"
} | tee -a "$OUT_ROOT/batch.log"

ALPHAS=(
  "0.2 0.2"
  "0.1 0.2"
  "0.2 0.1"
)
REVERSE_FLAGS=( "" "--reverse" )
ORDER_LABELS=( "spatial_first" "temporal_first" )

SEED_ARR=($SEEDS)
run=0
total=$(( ${#ALPHAS[@]} * ${#REVERSE_FLAGS[@]} * ${#SEED_ARR[@]} ))
for a_pair in "${ALPHAS[@]}"; do
  read -r S T <<<"$a_pair"
  for r_idx in 0 1; do
    RFLAG="${REVERSE_FLAGS[$r_idx]}"
    OL="${ORDER_LABELS[$r_idx]}"
    for seed in $SEEDS; do
      run=$((run + 1))
      echo "========== [$run/$total] [DEC] s=$S t=$T order=$OL seed=$seed ==========" | tee -a "$OUT_ROOT/batch.log"
      PROBE_ARGS=(
        run_probe_independent.py
        --probe-target decoder
        --alpha-spatial "$S"
        --alpha-temporal "$T"
      )
      if [ -n "$RFLAG" ]; then
        PROBE_ARGS+=( "$RFLAG" )
      fi
      PROBE_ARGS+=(
        --seed "$seed"
        --output-dir "$OUT_ROOT"
        --max-train-batches "$MAX_TRAIN"
        --max-eval-batches "$MAX_EVAL"
        --device "$DEVICE"
        --plot-probes
        --export-probe-csv
        --save-eval-probe-json
      )
      "${RUN_CMD[@]}" "${PROBE_ARGS[@]}" 2>&1 | tee -a "$OUT_ROOT/batch.log"
    done
  done
done

echo "========== 完成 $(date) 汇总 ==========" | tee -a "$OUT_ROOT/batch.log"
"${RUN_CMD[@]}" "$ROOT/experiments/aggregate_schemeA_results.py" "$OUT_ROOT" | tee -a "$OUT_ROOT/batch.log"

echo "所有结果在: $OUT_ROOT"
