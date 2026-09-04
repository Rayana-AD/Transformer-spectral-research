#!/usr/bin/env bash
# Supplement staging with figures, eval manifests, and one finetune checkpoint.
# Target: full zip < 1GB (Marian safetensors do not compress in zip).
set -euo pipefail

SRC="${SRC:-/Users/rayana/Desktop/TLB-Experiments}"
DST="${DST:-/Users/rayana/Desktop/github-research-staging}"

copy() {
  local from="$SRC/$1"
  local to="$DST/$2"
  if [[ ! -e "$from" ]]; then
    echo "SKIP missing: $1"
    return 0
  fi
  mkdir -p "$(dirname "$to")"
  if [[ -d "$from" ]]; then
    rsync -a "$from/" "$to/"
  else
    cp -a "$from" "$to"
  fi
  echo "OK: $1"
}

echo "=== Paper figures (paper_assets) ==="
rsync -a --include='*.png' --include='*.pdf' --include='*.tex' --include='*.md' \
  --exclude='*' "$SRC/paper_assets/" "$DST/part-a-mainline/05-results/paper_assets/"

echo "=== Scheme A paper figures ==="
rsync -a "$SRC/results_probe_independent/paper_figures/" \
  "$DST/part-a-mainline/05-results/results_probe_independent/paper_figures/"

echo "=== ViT / Marian geometry figures ==="
for f in \
  results_random_init_activation_geometry/fig_random_pretrained_activation_geometry_overlay_zoom.png \
  results_random_init_activation_geometry/fig_random_pretrained_activation_geometry_overlay_zoom.pdf \
  results_random_init_activation_geometry/fig_random_pretrained_activation_geometry_overlay.png \
  results_random_init_activation_geometry/fig_random_init_activation_geometry_vit_marian.png; do
  copy "$f" "part-a-mainline/05-results/$f"
done

echo "=== Script3 paper figure ==="
copy "results_script3/depth8_condition_grid_paper.png" \
  "part-a-mainline/05-results/results_script3/depth8_condition_grid_paper.png"

echo "=== Script7 (CSV + main figures only, no epoch_weights) ==="
mkdir -p "$DST/part-a-mainline/05-results/results_script7/exp3_vit_cifar"
for f in \
  results_script7/exp3_vit_cifar/exp3_vit_per_layer_summary.csv \
  results_script7/exp3_vit_cifar/exp3_vit_per_layer_layer_raw.csv \
  results_script7/exp3_vit_cifar/exp3_vit_depth_summary.csv \
  results_script7/exp3_vit_cifar/fig_exp3_vit_per_layer_acc_vs_layer.png \
  results_script7/exp3_vit_cifar/fig_exp3_vit_per_layer_acc_vs_layer.pdf \
  results_script7/exp3_vit_cifar/fig_exp3_vit_per_layer_note.md \
  results_script7/exp2_mlp_vit_combined/fig_exp2_combined_mlp_vit_depth8.png \
  results_script7/exp2_mlp_vit_combined/fig_exp2_combined_mlp_vit_depth8.pdf; do
  copy "$f" "part-a-mainline/05-results/$f"
done

echo "=== Finetune eval manifests (all seeds, no extra checkpoints) ==="
EVAL_DST="$DST/part-a-mainline/06-checkpoints/eval_manifests"
mkdir -p "$EVAL_DST"
FINAL="$SRC/results_final_1epoch_3seed_20260423_180017"
for d in eval_vanilla_s42 eval_vanilla_s43 eval_vanilla_s44; do
  rsync -a "$FINAL/$d/" "$EVAL_DST/$d/" 2>/dev/null || true
done
for f in log_vanilla_s42.txt log_vanilla_s43.txt log_vanilla_s44.txt \
  log_vanilla_s42_eval.txt log_vanilla_s43_eval.txt log_vanilla_s44_eval.txt; do
  copy "results_final_1epoch_3seed_20260423_180017/$f" "part-a-mainline/06-checkpoints/eval_manifests/$f"
done

echo "=== Scheme A archived numbers ==="
copy "results_probe_independent/schemeA_3x2x3_20260424_0137/schemeA_detailed_runs.csv" \
  "part-a-mainline/05-results/results_probe_independent/schemeA_3x2x3_20260424_0137/schemeA_detailed_runs.csv"
copy "results_probe_independent/schemeA_dec_3x2x3_20260425_1943/schemeA_dec_detailed_runs.csv" \
  "part-a-mainline/05-results/results_probe_independent/schemeA_dec_3x2x3_20260425_1943/schemeA_dec_detailed_runs.csv" 2>/dev/null || true

echo "=== One finetune checkpoint (vanilla seed42, ~512MB) ==="
CKPT_SRC="$FINAL/train_vanilla_s42/hf_finetuned_vanilla_seed42"
CKPT_DST="$DST/part-a-mainline/06-checkpoints/vanilla_finetune_seed42"
if [[ -d "$CKPT_SRC" ]]; then
  rm -rf "$CKPT_DST"
  mkdir -p "$(dirname "$CKPT_DST")"
  cp -RL "$CKPT_SRC" "$CKPT_DST"
  echo "OK: vanilla_finetune_seed42 (self-contained, symlinks dereferenced)"
else
  echo "WARN: missing $CKPT_SRC"
fi

echo "=== Curated gallery (copies) ==="
GAL="$DST/figures/gallery"
mkdir -p "$GAL"
copy_gallery() {
  local name="$1" rel="$2"
  if [[ -f "$SRC/$rel" ]]; then
    cp -a "$SRC/$rel" "$GAL/$name"
  fi
}
copy_gallery "01_activation_triptych.png" "paper_assets/figure_activation_triptych.png"
copy_gallery "02_script3_depth_conditioning.png" "paper_assets/figure_script3_depth8_condition_grid.png"
copy_gallery "03_script4_val_loss.png" "paper_assets/figure_script4_val_loss.png"
copy_gallery "04_script4_final_val_bar.png" "paper_assets/figure_script4_final_val_bar.png"
copy_gallery "05_vit_marian_c_geometry.png" "results_random_init_activation_geometry/fig_random_pretrained_activation_geometry_overlay_zoom.png"
copy_gallery "06_vit_per_layer_probe.png" "results_script7/exp3_vit_cifar/fig_exp3_vit_per_layer_acc_vs_layer.png"
copy_gallery "07_schemeA_depth_nuclear_norm.png" "results_probe_independent/paper_figures/schemeA_train_depth_nuc_seedavg_alpha_s01_t02_pub.png"
copy_gallery "08_schemeA_encoder_eval_heatmap.png" "results_probe_independent/paper_figures/schemeA_corr_heatmap_encoder_eval_seedavg_3x4_mainrows.png"
copy_gallery "09_schemeA_decoder_eval_heatmap.png" "results_probe_independent/paper_figures/schemeA_corr_heatmap_decoder_eval_seedavg_3x4_mainrows.png"
copy_gallery "10_schemeA_loss_ppl_closeup.png" "results_probe_independent/paper_figures/schemeA_train_nuclear_loss_ppl_closeup_enc2_dec3_dec5_seedavg.png"
ls -la "$GAL"

echo "=== Done ==="
du -sh "$DST" "$DST/part-a-mainline/06-checkpoints" "$DST/figures" 2>/dev/null
