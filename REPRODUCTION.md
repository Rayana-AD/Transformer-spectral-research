# Reproduction Guide

This archive is sized for **GitHub / zip under ~1GB**. Marian `safetensors` checkpoints **do not compress** in zip (~512MB each).

## What you can verify without GPU

| Claim | Location |
|-------|----------|
| Table 1 (synthetic nuclear / κ) | `part-a-mainline/05-results/results_script2/universal_summary.csv` |
| Depth conditioning (Fig 2a) | `results_script3/depth_summary.csv` + gallery `02_script3_depth_conditioning.png` |
| Pretrained BLEU 19.916 | `results_marian_ffn_frozen_eval/summary.json` |
| Vanilla finetune BLEU ~15.1–15.5 | `06-checkpoints/eval_manifests/eval_vanilla_s42/` |
| Scheme A 18× BLEU matrix | `schemeA_full_report.md`, `schemeA_detailed_runs.csv`, per-run `result.json` |
| Collapse (QKV / FFN swap) | Part B summaries + `results_marian_ffn_frozen_eval/` |

## Two finetune protocols (important)

| Protocol | Script | Train steps | Typical vanilla BLEU |
|----------|--------|-------------|----------------------|
| **Stress-test full** | `marian_mt_finetune_tlb.py` | ~100 (800 pairs, batch 8) | ~15.5 |
| **Scheme A probe** | `run_probe_independent.py` | default **32 batches** | ~18.1 (less degradation) |

Paper tables mix both: collapse / pretrained use full protocol; Scheme A matrix uses probe protocol unless you override `MAX_TRAIN`.

## Re-run synthetic block (CPU, ~minutes)

```bash
cd part-a-mainline/01-theory-scripts
python script2_universal_validation.py
python script3_depth_scattering.py
python script4_scattering_transformer_training.py
# Compare outputs to 05-results/results_script2–4/*.csv
```

## Re-eval vanilla finetune checkpoint (GPU, ~minutes)

Bundled checkpoint: `part-a-mainline/06-checkpoints/vanilla_finetune_seed42/`

```bash
cd part-a-mainline/03-marian-cascade
python marian_mt_finetune_tlb.py \
  --checkpoint ../06-checkpoints/vanilla_finetune_seed42 \
  --eval-only --max-val 128 --seed 42
# Expect BLEU ~15.1 (see CHECKPOINT_MANIFEST.json)
```

Pretrained baseline: omit `--checkpoint` or use HuggingFace `Helsinki-NLP/opus-mt-de-en` → BLEU ~19.9.

## Re-run Scheme A matrix (GPU, ~hours)

```bash
cd part-a-mainline/03-marian-cascade
# Default 32 batches (matches archived schemeA_* runs):
bash experiments/run_schemeA_cascade_matrix.sh
# Full 800-pair stress-test:
MAX_TRAIN=100 bash experiments/run_schemeA_cascade_matrix.sh
```

Compare to `05-results/results_probe_independent/schemeA_3x2x3_20260424_0137/`.

## ViT block (optional)

CIFAR-10 **not bundled** (~340MB). Download:

```python
from torchvision.datasets import CIFAR10
CIFAR10(root="part-a-mainline/04-data", download=True)
```

Then run `part-a-mainline/02-vit-bridge/script7_vit_criticality.py`. Figures archived in `05-results/results_script7/`.

## Figures

- **Gallery (README-ready)**: `figures/gallery/01–10_*.png`
- **Full paper assets**: `part-a-mainline/05-results/paper_assets/`
- **Scheme A**: `05-results/results_probe_independent/paper_figures/`

Regenerate Scheme A plots from probe CSV:

```bash
cd part-a-mainline/03-marian-cascade
python probe_visualization.py --help  # point to schemeA run dirs
```

## Size budget

See `SIZE_AUDIT.md` for excluded directories (>20GB early v_smooth grids).
