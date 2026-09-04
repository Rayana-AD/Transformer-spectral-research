# Size Audit — TLB-Experiments

Survey date: 2026-04. **Total project >>30GB**; this staging targets **<1GB zipped**.

## Largest directories (source: `TLB-Experiments/`)

| Path | Size | Shipped? | Reason |
|------|------|----------|--------|
| `results_v_smooth_hybrid/` | 8.5 GB | No | V-smooth grid; summary in Part B |
| `results_v_smooth_grid_20260423_171442/` | 3.0 GB | No | Duplicate grid epoch |
| `results_v_smooth_cascade/` | 3.0 GB | No | 6×512MB checkpoints; Scheme A uses probe protocol |
| `results_final_1epoch_3seed_*` | 2.0 GB | **Partial** | 1× vanilla checkpoint + eval JSON only |
| `results_v_smooth_variants_compare/` | 1.5 GB | No | Ablation grid |
| `results_marian_bleu_m1run/` | 1.5 GB | No | Early BLEU sweeps |
| `results_marian_reverse_experiments_*` | 1.0 GB | No | Negative Line-C runs |
| `results_marian_qkv_proj_nuc/` | 512 MB | Summary only | Collapse demo; JSON in Part B |
| `results_probe_independent/` | 88 MB | **Yes** | Scheme A CSV/JSON + figures |
| `results_script7/exp3_vit_cifar/epoch_weights/` | 443 MB | No | ViT probe intermediates |
| `data/cifar-10-*` | 342 MB | No | Use `torchvision` download |
| `paper_assets/` + figures | ~4 MB | **Yes** | Paper PNG/PDF |
| `data/opus_de_en/` | 1.4 MB | **Yes** | MT finetune subset |

## Staging composition (after supplement)

| Component | ~Size |
|-----------|-------|
| Code + CSV + probe data | ~100 MB |
| Figures (gallery + paper) | ~15 MB |
| `vanilla_finetune_seed42` checkpoint | **512 MB** |
| **Total uncompressed** | **~630 MB** |
| **Zip (safetensors incompressible)** | **~620–650 MB** |

Adding a second Marian checkpoint would push zip **>1 GB** — not included by default.

## If you need more checkpoints locally

```bash
# Best single cascade from v_smooth search (different protocol than Scheme A probe):
# TLB-Experiments/results_v_smooth_cascade/cascade_01_02/hf_finetuned_v_smooth_cascade_seed42/

# Full Scheme A re-run (no saved weights in probe runs):
bash part-a-mainline/03-marian-cascade/experiments/run_schemeA_cascade_matrix.sh
```
