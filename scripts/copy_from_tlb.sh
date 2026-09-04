#!/usr/bin/env bash
# Lightweight copy from TLB-Experiments → github-research-staging per MANIFEST.yaml
# Skips checkpoints, venv, results_v_smooth_*, *.pt
set -euo pipefail

SRC="${SRC:-/Users/rayana/Desktop/TLB-Experiments}"
DST="${DST:-/Users/rayana/Desktop/github-research-staging}"
MANIFEST="$DST/MANIFEST.yaml"

if [[ ! -d "$SRC" ]]; then
  echo "Missing source: $SRC" >&2
  exit 1
fi

copy_path() {
  local rel="$1"
  local dest_sub="$2"
  local from="$SRC/$rel"
  local to="$DST/$dest_sub/$rel"
  if [[ ! -e "$from" ]]; then
    echo "SKIP (missing): $rel"
    return 0
  fi
  mkdir -p "$(dirname "$to")"
  if [[ -d "$from" ]]; then
    rsync -a --exclude='hf_finetuned_*' --exclude='*.pt' --exclude='*.pth' \
      --exclude='.venv*' --exclude='__pycache__' "$from/" "$to/"
  else
    cp -a "$from" "$to"
  fi
  echo "OK: $rel → $dest_sub/"
}

echo "=== Part A theory ==="
for f in script1_activation_visualization_invariants.py script2_universal_validation.py \
  script3_depth_scattering.py script4_scattering_transformer_training.py \
  activation_extras.py paper_style.py; do
  copy_path "$f" "part-a-mainline/01-theory-scripts"
done

echo "=== Part A vit ==="
for f in script5_vit_representation_svd.py script5_vit_norms_combined_panel.py \
  script7_vit_criticality.py experiments/pretrained_random_activation_geometry_overlay.py; do
  copy_path "$f" "part-a-mainline/02-vit-bridge"
done

echo "=== Part A marian ==="
for f in tlb_value_smooth.py marian_attn_tlb_patch.py marian_mt_finetune_tlb.py \
  run_probe_independent.py marian_mechanism_probe.py marian_enhanced_probe.py \
  probe_mech_plots.py probe_visualization.py mt_train_tlb.py \
  experiments/run_schemeA_cascade_matrix.sh experiments/run_schemeA_decoder_cascade_matrix.sh \
  experiments/run_schemeA_decoder_causal_ablation.sh experiments/run_regularization_baselines.sh \
  experiments/circular_shift_leakage_demo.py experiments/verify_cascade_spectral_operators.py \
  experiments/aggregate_schemeA_results.py experiments/SCHEME_A_CASCADE_EXPERIMENT.md; do
  copy_path "$f" "part-a-mainline/03-marian-cascade"
done

echo "=== Part A data ==="
copy_path "data/opus_de_en/train_5k.jsonl" "part-a-mainline/04-data"
copy_path "data/opus_de_en/val_500.jsonl" "part-a-mainline/04-data"

echo "=== Part A results (light) ==="
for r in paper_assets results_script2/universal_summary.csv results_script2/universal_raw.csv \
  results_script3/depth_summary.csv results_script4/training_loss_summary.csv \
  results_script4/training_loss_raw.csv results_probe_independent/paper_figures \
  results_probe_independent/schemeA_3x2x3_20260424_0137/schemeA_full_report.md \
  results_probe_independent/schemeA_dec_3x2x3_20260425_1943 \
  results_probe_independent/schemeA_dec_causal_ablation_20260630_1812 \
  results_probe_independent/rebuttal_baselines_20260701_1646/baseline_summary.md \
  results_marian_ffn_frozen_eval results_prefix_invariance results_cascade_spectral_verify \
  results_random_init_activation_geometry \
  results_ppl_spike_analysis/ppl_spike_bad_cases.md \
  results_ppl_spike_analysis/ppl_spike_report.md; do
  copy_path "$r" "part-a-mainline/05-results"
done

echo "=== Part A docs ==="
mkdir -p "$DST/part-a-mainline/docs"
cp -a "$SRC/docs/EXPERIMENT_RESULTS_PAPER_FIGURES.md" "$SRC/docs/PROBE_EXPERIMENT_AND_PLOTS.md" \
  "$SRC/docs/RESEARCH_GITHUB_PLAN.md" "$DST/part-a-mainline/docs/" 2>/dev/null || true

echo "=== Part B code ==="
for f in script6_scattering_criticality.py script8_dual_space_memory.py script9_ffn_bottleneck_dual.py \
  script10_token_mirror.py script11_tlb_attention.py tlb_attention_core.py memory_tasks.py \
  transformer_mechanics.py; do
  copy_path "$f" "part-b-explorations/code"
done

echo "=== Part B summaries ==="
for r in results_marian_qkv_proj_nuc/summary.json results_marian_bleu_quick/summary.json \
  experiments/v_smooth_experiments_summary.md EXPERIMENT_ALIGNMENT_GUIDE.md \
  Discussion_Conclusion.md docs/VIT_MARIAN_ROOT_CAUSE.md; do
  copy_path "$r" "part-b-explorations/summaries"
done

# requirements
cp -a "$SRC/requirements.txt" "$DST/" 2>/dev/null || echo "no requirements.txt"

echo "=== Done ==="
du -sh "$DST"/* 2>/dev/null | head -20
