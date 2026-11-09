#!/usr/bin/env bash
# Copy Hessian-FFN prototype slice from TLB-Algorithm (no large checkpoints / paper result dumps).
set -euo pipefail
SRC="${SRC:-$HOME/Desktop/TLB-Algorithm}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  "$SRC/tlb/ffn/" "$ROOT/tlb/ffn/"
rsync -a --exclude='__pycache__' \
  "$SRC/tlb/models/causal_lm.py" \
  "$SRC/tlb/models/gpt2_ctxsplit_init.py" \
  "$SRC/tlb/models/gpt2_ffn_transfer.py" \
  "$SRC/tlb/models/vit_ffn_transfer.py" \
  "$SRC/tlb/models/vit_torchvision_ft.py" \
  "$ROOT/tlb/models/"
rsync -a --exclude='__pycache__' \
  "$SRC/tlb/training/memory_track_probe.py" \
  "$SRC/tlb/training/hessian_geometry_diag.py" \
  "$SRC/tlb/training/utils.py" \
  "$ROOT/tlb/training/"
mkdir -p "$ROOT/tlb/kernels" "$ROOT/tlb/blocks" "$ROOT/tlb/data"
rsync -a --exclude='__pycache__' \
  "$SRC/tlb/kernels/hessian_ffn_fused.py" \
  "$SRC/tlb/kernels/hessian_dispatch.py" \
  "$SRC/tlb/kernels/__init__.py" \
  "$ROOT/tlb/kernels/" 2>/dev/null || true
rsync -a --exclude='__pycache__' \
  "$SRC/tlb/blocks/transformer_layer.py" \
  "$SRC/tlb/blocks/transformer_decoder_layer.py" \
  "$ROOT/tlb/blocks/" 2>/dev/null || true

PROTOS=(
  l0_smoke.py
  l4_causal_wt103_single.py
  run_wt103_d256_lr_smoke.py
  run_memory_capacity_sweep.py
  plot_memory_track_geometry.py
  plot_memory_track_2d_native.py
  l_vit_cifar_smoke.py
  vit_forward_smoke.py
)
mkdir -p "$ROOT/experiments"
for f in "${PROTOS[@]}"; do
  [[ -f "$SRC/experiments/$f" ]] && cp -a "$SRC/experiments/$f" "$ROOT/experiments/"
done

mkdir -p "$ROOT/configs/wt103_scale_smoke"
rsync -a "$SRC/configs/wt103_scale_smoke/" "$ROOT/configs/wt103_scale_smoke/" 2>/dev/null || true

mkdir -p "$ROOT/figures/memory_track_geom"
rsync -a "$SRC/figures/memory_track_geom/"*.png "$ROOT/figures/memory_track_geom/" 2>/dev/null || true

echo "Synced prototype from $SRC → $ROOT"
