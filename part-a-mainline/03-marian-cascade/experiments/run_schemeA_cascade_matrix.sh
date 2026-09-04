#!/usr/bin/env bash
# 方案 A：级联探针 完整矩阵（18 次）
#  优先 .venv_minepy（Python3.10 + minepy + arm64 MPS）；
#  其次本仓库 .venv（日常成功环境，默认 --device auto）；
#  最后 conda run -n tlb_mine（默认 CPU，防 x86/Rosetta + MPS SIGSEGV）。
#
# 用法：./experiments/run_schemeA_cascade_matrix.sh
#  可选：OUT_ROOT=... SEEDS=... MAX_TRAIN=32 MAX_EVAL=8 DEVICE=auto

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-tlb_mine}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results_probe_independent/schemeA_3x2x3_$(date +%Y%m%d_%H%M)}"
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
# macOS 无 GNU date -I 时用近似 ISO
TS="$(date "+%Y-%m-%dT%H:%M:%S%z" 2>/dev/null || date)"
{
  echo "LOG: $TS OUT_ROOT=$OUT_ROOT MAX_TRAIN=$MAX_TRAIN MAX_EVAL=$MAX_EVAL TLB_DEVICE=$TLB_DEVICE RUN=$RUN_LABEL"
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
      echo "========== [$run/$total] s=$S t=$T order=$OL seed=$seed ==========" | tee -a "$OUT_ROOT/batch.log"
      # 用数组拼参数，避免行尾 \ 丢失时把 --output-dir 当成独立命令
      PROBE_ARGS=(
        run_probe_independent.py
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
