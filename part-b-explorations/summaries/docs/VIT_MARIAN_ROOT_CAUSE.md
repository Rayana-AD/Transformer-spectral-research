# ViT vs Marian: Why Results Differ

This note supports rebuttal responses on depth-confounded ViT readability and cross-modal comparison.

## Summary Table

| Dimension | Marian Encoder | Marian Decoder | ViT (CIFAR probe) |
| --- | --- | --- | --- |
| Attention mask | Bidirectional | Causal (autoregressive) | Bidirectional (patch self-attn) |
| Downstream cross-attn | Yes (to decoder) | Yes (to encoder memory) | No |
| Task in main MT block | Machine translation | Machine translation | Frozen linear probe on CIFAR-10 |
| Native depth profile (value / hidden geometry) | Arch-shaped; raw V peaks ~layer 2 | U-shaped; high shallow norm, mid compression | Monotonic probe accuracy vs layer |
| Cascade causal issue | Circular roll acceptable | Circular roll leaks via V smooth | N/A (no cascade intervention in ViT block) |
| Claim strength in paper | Controlled intervention | Controlled intervention | **Observational** diagnostic only |

## Three Root Causes for ViT Looking Different

### 1. Temporal boundary (weaker X-axis coupling)

ViT is encoder-only with bidirectional patch attention. There is no autoregressive decoder history requirement and no encoder→decoder global caching schedule. Marian translation **encoder** develops a mid-layer capacity expansion (arch peak) tied to preparing bidirectional representations for a downstream decoder; Marian **decoder** keeps high shallow entropy for history and compresses mid-layers. ViT lacks this cross-module scheduling, so it does not show the same arch/U profiles.

### 2. Depth confound in Exp3 per-layer probes

On frozen ViT-B/16 with independent linear probes per layer (`results_script7/exp3_vit_cifar/exp3_vit_per_layer_summary.csv`):

- Test accuracy rises monotonically from layer 0 (0.10) to layer 12 (0.865).
- Spearman(acc, layer_index) = **1.0**.
- Raw Spearman(acc, c_bar_geom) ≈ **−0.96**.
- Partial Pearson(acc, c_bar_geom | layer_index) ≈ **−0.40** (`results_script7/vit_correlation_baseline/exp3_partial_correlation_control_depth.csv`).

Interpretation: deeper layers are more readable **because depth correlates perfectly with probe position**, not because a particular nuclear ratio causally improves accuracy. Rebuttal should **not** claim ViT proves cascade causality.

### 3. Pretraining reshapes geometry (not activation-only)

Random-init vs pretrained overlay (`results_random_init_activation_geometry/random_pretrained_activation_geometry_overlay_note.md`):

- Random GeLU ViT: early expansion then relaxation toward 1 (c_l ≈ 1.18 → 1.03).
- Pretrained GeVL ViT: early **compression** then late rise — qualitatively reversed trajectory.
- Pretrained Marian decoder: extreme final expansion (c_l ≈ 4.2 at last layer).

Geometry tracks **learned scheduling under pretraining**, not a universal activation law.

## Recommended Rebuttal Wording

> The ViT experiments motivate spectral probes as cross-modal diagnostics. Because accuracy and layer index are fully confounded in frozen per-layer probes, we report partial correlations controlling for depth and treat ViT as observational evidence. Mechanistic and causal claims rest on Marian value-path cascade interventions and the new decoder-only prefix-invariance / causal-safe ablations.

## Related Assets

- Per-layer figure: `results_script7/exp3_vit_cifar/fig_exp3_vit_per_layer_acc_vs_layer.pdf`
- Partial corr CSV: `results_script7/vit_correlation_baseline/exp3_partial_correlation_control_depth.csv`
- Random/pretrained overlay: `results_random_init_activation_geometry/fig_random_pretrained_activation_geometry_overlay_zoom.pdf`
- Scheme A decoder geometry: `results_probe_independent/paper_figures/schemeA_train_depth_nuc_seedavg_alpha_s01_t02_pub.pdf`
