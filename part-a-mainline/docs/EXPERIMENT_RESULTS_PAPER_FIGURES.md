# Experiment Results and Paper Figure Map

This document consolidates the paper-facing experimental assets for three result blocks:

1. Activation-function spectral scattering.
2. ViT criticality / nuclear-norm geometry.
3. Scheme A cascade experiments, including Encoder and Decoder variants.

The goal is to make the paper figures, tables, data sources, and main interpretations easy to locate.

---

## 1. Activation-Function Spectral Scattering

### Role in the paper

This block is the motivation section before the cascade experiments. It establishes that pointwise activations reshape matrix spectra, and that these effects accumulate with depth. The emphasis should be on:

- Smooth activations mostly preserve numerical rank in the single-step setting.
- Activations attenuate nuclear mass relative to a linear baseline.
- Activations regularize conditioning, but ReLU can concentrate energy in the leading singular direction.
- Depth turns these local spectral effects into layerwise conditioning/stability phenomena.

### Main Table: Singular-Value Statistics of Activated Feature Matrices

**Paper table title:** Singular-value statistics of the activated feature matrix  
**Mathematical object:**  

```latex
\mathbf{Z}=W\mathbf{X}, \qquad \mathbf{M}_{\sigma}=\sigma(\mathbf{Z})
```

Use this notation in the activation section, then bridge to Scheme A by stating that later Transformer probes apply the same nuclear-norm and conditioning metrics to actual value matrices \(\mathbf{V}_{\ell}\).

**Table file:**

- TeX: `paper_assets/table_script2_universal_booktabs.tex`
- Source CSV: `results_script2/universal_summary.csv`

**Representative numbers in the current table:**

| Activation | \(n=16\) nuclear | \(n=16\) med. \(\kappa\) | \(n=64\) nuclear | \(n=64\) med. \(\kappa\) |
| --- | ---: | ---: | ---: | ---: |
| Linear | 27.68 | 553.79 | 221.66 | 6730.68 |
| ReLU | 19.26 | 168.03 | 149.81 | 736.09 |
| GELU | 16.86 | 122.13 | 135.24 | 605.56 |
| SiLU | 15.54 | 92.73 | 124.66 | 553.04 |
| Mish | 18.24 | 91.80 | 146.41 | 456.91 |
| ELU | 23.92 | 229.89 | 192.07 | 518.34 |

### Main Figure: Depth-Wise Accumulation of Conditioning

**Paper figure title:** Depth-wise accumulation of conditioning  
**Use in paper:** Main motivation figure after the singular-value table.

**Figure files:**

- PNG: `paper_assets/figure_script3_depth8_condition_grid.png`
- PDF: `paper_assets/figure_script3_depth8_condition_grid.pdf`
- Regenerated copy: `results_script3/depth8_condition_grid_paper.png`
- Source CSV: `results_script3/depth_summary.csv`
- Script: `script3_depth_scattering.py`

**Current caption candidate:**

```latex
\caption{Depth-wise accumulation of conditioning under repeated linear--activation scattering blocks.
Each panel reports one activation at depth $N=8$.
Orange curves show the median finite condition number $\kappa$ on a logarithmic scale, while gray dashed curves show the fraction of random seeds for which $\kappa$ remains finite.
A drop in the gray curve indicates that some runs have become singular or numerically singular at that layer, revealing activation-dependent differences in deep spectral stability.}
```

### Secondary / Optional Figures

These can be kept in the paper only if space allows, otherwise move to appendix.

| Item | Role | PNG | PDF / TeX |
| --- | --- | --- | --- |
| Activation heatmap | Visual intuition only | `paper_assets/figure_activation_triptych.png` | `paper_assets/figure_activation_triptych.pdf` |
| Low-rank denoising loss | Task-level sanity check | `paper_assets/figure_script4_val_loss.png` | `paper_assets/figure_script4_val_loss.pdf` |
| Final validation bar | Compact ranking of Script4 final validation loss | `paper_assets/figure_script4_final_val_bar.png` | `paper_assets/figure_script4_final_val_bar.pdf` |
| Script4 table | Task-level summary table | N/A | `paper_assets/table_script4_val_loss_booktabs.tex` |

**Script4 source data:**

- Raw training rows: `results_script4/training_loss_raw.csv`
- Summary table source: `results_script4/training_loss_summary.csv`

---

## 2. ViT Criticality and Nuclear-Norm Geometry

### Role in the paper

This block is a bridge between synthetic activation geometry and Transformer-scale representation geometry. It should be framed as observational, not causal:

- ViT layerwise geometry shows that nuclear-norm transition ratios \(c_{\ell}\) evolve with depth.
- Frozen ViT linear probes show that task readability increases with layer index.
- Accuracy, depth, and nuclear geometry are strongly entangled, so this block motivates Scheme A rather than proving causality.

### Main Figure: \(c_{\ell}\)-Depth Comparison

**Paper figure title:** Layer-transition nuclear-norm ratio \(c_{\ell}\) across depth  
**Recommended figure:** random-initialized versus pretrained activation-geometry overlay. This is the paper-facing \(c_{\ell}\)-depth comparison figure.

**Figure files:**

- PNG: `results_random_init_activation_geometry/fig_random_pretrained_activation_geometry_overlay_zoom.png`
- PDF: `results_random_init_activation_geometry/fig_random_pretrained_activation_geometry_overlay_zoom.pdf`
- Full-range PNG: `results_random_init_activation_geometry/fig_random_pretrained_activation_geometry_overlay.png`
- Full-range PDF: `results_random_init_activation_geometry/fig_random_pretrained_activation_geometry_overlay.pdf`
- Notes: `results_random_init_activation_geometry/random_pretrained_activation_geometry_overlay_note.md`

**Source data:**

- Merged random/pretrained rows: `results_random_init_activation_geometry/random_pretrained_activation_c_raw.csv`
- Per-transition summary: `results_random_init_activation_geometry/random_pretrained_activation_c_summary.csv`
- Pretrained-only raw rows: `results_random_init_activation_geometry/pretrained_activation_c_raw.csv`

**Key reading:** The figure compares \(c_{\ell}=N(h_{\ell})/N(h_{\ell-1})\) across ViT, Marian encoder, and Marian decoder under random initialization and pretrained checkpoints. Random ViT shows early expansion followed by relaxation toward 1, while pretrained ViT changes the layerwise trajectory and Light-depth can create a late high transition. Marian encoder and decoder display distinct pretrained regimes, including strong final encoder compression and strong final decoder expansion. This supports the claim that pretrained Transformers reshape activation-induced geometry in a module-dependent way.

### Main Figure: \(N\) / Accuracy Layer Profile

**Paper figure title:** Layer-wise test accuracy and nuclear-norm geometry in frozen ViT  
**Current figure:** plots CIFAR-10 linear-probe accuracy and \(\bar c_{\mathrm{geom}}\) from \(N_0,\ldots,N_{\ell}\).

**Figure files:**

- PNG: `results_script7/exp3_vit_cifar/fig_exp3_vit_per_layer_acc_vs_layer.png`
- PDF: `results_script7/exp3_vit_cifar/fig_exp3_vit_per_layer_acc_vs_layer.pdf`
- Notes: `results_script7/exp3_vit_cifar/fig_exp3_vit_per_layer_note.md`

**Source data:**

- Per-layer summary: `results_script7/exp3_vit_cifar/exp3_vit_per_layer_summary.csv`
- Per-layer raw rows: `results_script7/exp3_vit_cifar/exp3_vit_per_layer_layer_raw.csv`

**Key numbers from current CSV:**

| Layer \(\ell\) | Test accuracy | \(\bar c_{\mathrm{geom}}\) | Nuclear norm \(N_{\ell}\) |
| ---: | ---: | ---: | ---: |
| 0 | 0.1000 | N/A | 1508.61 |
| 1 | 0.2882 | 2.1779 | 3285.64 |
| 2 | 0.3247 | 1.6779 | 4247.29 |
| 8 | 0.6951 | 1.3140 | 13407.06 |
| 12 | 0.8652 | 1.2572 | 23517.01 |

**Interpretation:** Accuracy increases monotonically with layer depth, while \(\bar c_{\mathrm{geom}}\) relaxes from a high early value toward a lower range. This supports the use of layerwise nuclear geometry as a representation diagnostic, but depth is a strong confound.

### Secondary ViT Figure

| Item | Role | PNG | PDF |
| --- | --- | --- | --- |
| Truncated-depth summary | Single-depth-point snapshot; less useful as a main figure because current CSV has only one depth row | `results_script7/exp3_vit_cifar/fig_exp3_vit_acc_vs_depth.png` | `results_script7/exp3_vit_cifar/fig_exp3_vit_acc_vs_depth.pdf` |
| Combined MLP/ViT activation panel | Supplementary contrast between synthetic ScatteringLayer stacks and truncated pretrained ViT | `results_script7/exp2_mlp_vit_combined/fig_exp2_combined_mlp_vit_depth8.png` | `results_script7/exp2_mlp_vit_combined/fig_exp2_combined_mlp_vit_depth8.pdf` |

---

## 3. Scheme A Cascade Experiments

### Role in the paper

Scheme A is the main experimental block. It moves from synthetic or frozen representation measurements to direct interventions on Marian attention value tensors.

The main story should emphasize:

- The cascade acts as a controlled compression ladder on value matrices:
  \[
  \|V^{\mathrm{raw}}\|_{\ast} > \|V^{(1)}\|_{\ast} > \|V^{(2)}\|_{\ast}.
  \]
- Encoder and Decoder have different depth profiles.
- Encoder shows a mid-layer expansion pattern around layer 2.
- Decoder has high shallow-layer value norm and a stronger shallow-to-middle compression valley.
- Evaluation BLEU correlations are more interpretable than train-loss correlations.

### Experiment Definition

**Design document:**

- `experiments/SCHEME_A_CASCADE_EXPERIMENT.md`

**Encoder full report:**

- `results_probe_independent/schemeA_3x2x3_20260424_0137/schemeA_full_report.md`

**Encoder data:**

- Runs: `results_probe_independent/schemeA_3x2x3_20260424_0137/schemeA_runs.csv`
- Config summary: `results_probe_independent/schemeA_3x2x3_20260424_0137/schemeA_summary_by_config.csv`
- Detailed runs: `results_probe_independent/schemeA_3x2x3_20260424_0137/schemeA_detailed_runs.csv`
- Detailed summary: `results_probe_independent/schemeA_3x2x3_20260424_0137/schemeA_detailed_summary.csv`

**Decoder data:**

- Runs: `results_probe_independent/schemeA_dec_3x2x3_20260425_1943/schemeA_runs.csv`
- Config summary: `results_probe_independent/schemeA_dec_3x2x3_20260425_1943/schemeA_summary_by_config.csv`

---

## 4. Scheme A Paper Figures

### Main Figure 1: Encoder/Decoder Nuclear Norm vs Layer Index

**Paper figure title:** Train-time depth profile of cascade nuclear norm in Encoder and Decoder self-attention  
**Recommended file:** use the publication version if available.

**Figure files:**

- PNG: `results_probe_independent/paper_figures/schemeA_train_depth_nuc_seedavg_alpha_s01_t02_pub.png`
- PDF: `results_probe_independent/paper_figures/schemeA_train_depth_nuc_seedavg_alpha_s01_t02_pub.pdf`
- Alternate PNG: `results_probe_independent/paper_figures/schemeA_train_depth_nuc_seedavg_alpha_s01_t02.png`
- Alternate PDF: `results_probe_independent/paper_figures/schemeA_train_depth_nuc_seedavg_alpha_s01_t02.pdf`
- Notes: `results_probe_independent/paper_figures/schemeA_train_depth_nuc_seedavg_notes.md`

**Source configuration:**

- Encoder source: `results_probe_independent/schemeA_3x2x3_20260424_0137`
- Decoder source: `results_probe_independent/schemeA_dec_3x2x3_20260425_1943`
- Configuration: \(\alpha_s=0.1,\alpha_t=0.2\), seeds 42, 43, 44.

**Core data from notes:**

| Target | Pattern |
| --- | --- |
| Encoder | Mid-layer expansion; raw \(V\) peaks around layer 2. |
| Decoder | High shallow value norm, then compression valley toward middle layers. |
| Cascade stage | \(V^{\mathrm{raw}}>V^{(1)}>V^{(2)}\) across almost all layers. |
| Order effect | Smaller than Encoder-vs-Decoder target-module effect at this setting. |

### Main Figure 2: Training Dynamics of \(N\) at Layers 2, 3, and 5

**Paper figure title:** Training dynamics of value nuclear norm for representative Encoder and Decoder layers  
**Recommended figure:** three-column close-up: Encoder layer 2, Decoder layer 3, Decoder layer 5.

**Figure files:**

- PNG: `results_probe_independent/paper_figures/schemeA_train_nuclear_loss_ppl_closeup_enc2_dec3_dec5_seedavg.png`
- PDF: `results_probe_independent/paper_figures/schemeA_train_nuclear_loss_ppl_closeup_enc2_dec3_dec5_seedavg.pdf`
- Alternate smaller figure: `results_probe_independent/paper_figures/schemeA_train_nuclear_loss_ppl_closeup_enc2_dec5_seedavg.png`
- Alternate smaller PDF: `results_probe_independent/paper_figures/schemeA_train_nuclear_loss_ppl_closeup_enc2_dec5_seedavg.pdf`
- Notes: `results_probe_independent/paper_figures/schemeA_train_depth_nuc_seedavg_notes.md`

**Layer roles:**

| Panel | Meaning |
| --- | --- |
| Encoder layer 2 | Mid-layer expansion / high-capacity bidirectional representation. |
| Decoder layer 3 | Compression valley in Decoder self-attention. |
| Decoder layer 5 | Late output-facing reconstruction regime. |

### Main Figure 3: Encoder and Decoder Summary Heatmaps

Use the `mainrows` versions in the paper because their metric rows are aligned with the method formulas and omit less central rows.

**Encoder eval heatmap:**

- PNG: `results_probe_independent/paper_figures/schemeA_corr_heatmap_encoder_eval_seedavg_3x4_mainrows.png`
- PDF: `results_probe_independent/paper_figures/schemeA_corr_heatmap_encoder_eval_seedavg_3x4_mainrows.pdf`

**Decoder eval heatmap:**

- PNG: `results_probe_independent/paper_figures/schemeA_corr_heatmap_decoder_eval_seedavg_3x4_mainrows.png`
- PDF: `results_probe_independent/paper_figures/schemeA_corr_heatmap_decoder_eval_seedavg_3x4_mainrows.pdf`

**Appendix / diagnostic heatmaps:**

- Encoder train PNG: `results_probe_independent/paper_figures/schemeA_corr_heatmap_encoder_train_seedavg_3x4.png`
- Encoder train PDF: `results_probe_independent/paper_figures/schemeA_corr_heatmap_encoder_train_seedavg_3x4.pdf`
- Decoder train PNG: `results_probe_independent/paper_figures/schemeA_corr_heatmap_decoder_train_seedavg_3x4.png`
- Decoder train PDF: `results_probe_independent/paper_figures/schemeA_corr_heatmap_decoder_train_seedavg_3x4.pdf`
- Full-row Encoder eval PNG: `results_probe_independent/paper_figures/schemeA_corr_heatmap_encoder_eval_seedavg_3x4.png`
- Full-row Encoder eval PDF: `results_probe_independent/paper_figures/schemeA_corr_heatmap_encoder_eval_seedavg_3x4.pdf`
- Full-row Decoder eval PNG: `results_probe_independent/paper_figures/schemeA_corr_heatmap_decoder_eval_seedavg_3x4.png`
- Full-row Decoder eval PDF: `results_probe_independent/paper_figures/schemeA_corr_heatmap_decoder_eval_seedavg_3x4.pdf`
- Notes: `results_probe_independent/paper_figures/schemeA_corr_heatmap_seedavg_notes.md`

**Quick numeric checks from notes:**

| Target | \(V^{\mathrm{raw}}\) nuclear Spearman | Temporal cosine Spearman | Temporal difference Spearman | MIC temporal cosine |
| --- | ---: | ---: | ---: | ---: |
| Encoder eval | -0.74 | +0.65 | -0.66 | +0.85 |
| Decoder eval | -0.77 | +0.73 | -0.73 | +0.85 |

### Additional Scheme A Figure/Table: Gradient and CKA

The gradient/CKA information is currently a table in the notes rather than a separate PNG/PDF.

**Source notes:**

- `results_probe_independent/paper_figures/schemeA_corr_heatmap_seedavg_notes.md`

**Table rows from current notes:**

| Target | Best stable regime | Observation |
| --- | --- | --- |
| Decoder gradient \(G_{\mathrm{dec}}\) | Stable configs have \(G_{\mathrm{dec}}\approx 33\) | High-spatial-strength unstable settings can exceed 250--450. |
| Decoder CKA | \(\mathrm{CKA}_{dec5}>\mathrm{CKA}_{dec2}\) | Late Decoder FFN is more representation-preserving than middle Decoder FFN. |

---

## 5. Scheme A Paper Tables

### Main Table 1: Preliminary Comparison of Direct Pretrained Inference

**Purpose:** show that direct activation replacement in frozen pretrained Marian is not a fair substitute for retraining or cascade intervention; pretrained weights are highly activation-specific.

**Source file:**

- JSON: `results_marian_ffn_frozen_eval/summary.json`

**Current values:**

| Setting | FFN activation | Frozen weights | BLEU | Val CE | Val PPL | Token acc |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Pretrained baseline | original / null | yes | 19.916 | 5.471 | 237.805 | 0.248 |
| Frozen + Light-depth \(t=2\) | `light_depth_t2.0` | yes | 0.348 | 6.604 | 737.876 | 0.119 |
| Frozen + physical \(\tau\) | `phys_tau_softplus_z_b1.0` | yes | 0.028 | 10.107 | 24522.493 | 0.016 |

**Related one-epoch FFN replacement check:**

- JSON: `results_marian_ffn_act_cmp/summary.json`
- Baseline Swish BLEU: 15.460
- Light-depth \(t=2\) BLEU: 8.915
- Physical \(\tau\) BLEU: 0.000

### Main Table 2: Full 12-Configuration Comparison (Encoder 6 + Decoder 6)

**Purpose:** compare all Scheme A Encoder and Decoder configurations across cascade order and \((\alpha_s,\alpha_t)\).

**Source files:**

- Encoder summary CSV: `results_probe_independent/schemeA_3x2x3_20260424_0137/schemeA_summary_by_config.csv`
- Decoder summary CSV: `results_probe_independent/schemeA_dec_3x2x3_20260425_1943/schemeA_summary_by_config.csv`

**Current values:**

| Target | Order | \(\alpha_s\) | \(\alpha_t\) | Runs | BLEU mean | BLEU std | Train PPL mean | Train PPL std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Encoder | Spatial first | 0.1 | 0.2 | 3 | 17.812 | 0.078 | 13825.011 | 18420.818 |
| Encoder | Spatial first | 0.2 | 0.1 | 3 | 17.853 | 0.154 | 23.020 | 9.265 |
| Encoder | Spatial first | 0.2 | 0.2 | 3 | 17.785 | 0.251 | 267.256 | 358.595 |
| Encoder | Temporal first | 0.1 | 0.2 | 3 | 18.492 | 0.278 | 59.764 | 62.435 |
| Encoder | Temporal first | 0.2 | 0.1 | 3 | 17.833 | 0.110 | 367.313 | 373.970 |
| Encoder | Temporal first | 0.2 | 0.2 | 3 | 17.908 | 0.202 | 746.038 | 448.129 |
| Decoder | Spatial first | 0.1 | 0.2 | 3 | 18.244 | 0.128 | 19.921 | 1.871 |
| Decoder | Spatial first | 0.2 | 0.1 | 3 | 17.968 | 0.368 | 468.605 | 628.829 |
| Decoder | Spatial first | 0.2 | 0.2 | 3 | 18.234 | 0.092 | 22.151 | 2.415 |
| Decoder | Temporal first | 0.1 | 0.2 | 3 | 18.244 | 0.128 | 19.921 | 1.871 |
| Decoder | Temporal first | 0.2 | 0.1 | 3 | 18.030 | 0.415 | 892.057 | 616.054 |
| Decoder | Temporal first | 0.2 | 0.2 | 3 | 18.156 | 0.181 | 682.930 | 933.780 |

**Main reading:**

- Encoder best BLEU: Temporal first, \((\alpha_s,\alpha_t)=(0.1,0.2)\), BLEU \(18.492\pm0.278\).
- Decoder best BLEU: Spatial first or Temporal first at \((0.1,0.2)\), BLEU \(18.244\pm0.128\); both orders coincide in current summary.
- Decoder is more stable at \((0.1,0.2)\) and \((0.2,0.2)\) under Spatial first, with train PPL near 20--22.
- Stronger spatial setting \((0.2,0.1)\) tends to produce high PPL variance, especially in Decoder.

---

## 6. Suggested Paper Placement

### Main text

1. Activation singular-value table: `paper_assets/table_script2_universal_booktabs.tex`
2. Activation depth-conditioning figure: `paper_assets/figure_script3_depth8_condition_grid.pdf`
3. ViT / Marian \(c_{\ell}\)-depth figure: `results_random_init_activation_geometry/fig_random_pretrained_activation_geometry_overlay_zoom.pdf`
4. ViT layerwise accuracy / geometry figure: `results_script7/exp3_vit_cifar/fig_exp3_vit_per_layer_acc_vs_layer.pdf`
5. Scheme A nuclear norm vs layer index: `results_probe_independent/paper_figures/schemeA_train_depth_nuc_seedavg_alpha_s01_t02_pub.pdf`
6. Scheme A close-up dynamics: `results_probe_independent/paper_figures/schemeA_train_nuclear_loss_ppl_closeup_enc2_dec3_dec5_seedavg.pdf`
7. Scheme A Encoder/Decoder eval heatmaps: `*_mainrows.pdf`
8. Main Table 1: direct pretrained inference, source `results_marian_ffn_frozen_eval/summary.json`
9. Main Table 2: full 12-configuration comparison, source Encoder/Decoder `schemeA_summary_by_config.csv`

### Appendix / Supplement

- Activation heatmap: `paper_assets/figure_activation_triptych.pdf`
- Low-rank denoising curve and final bar: `paper_assets/figure_script4_val_loss.pdf`, `paper_assets/figure_script4_final_val_bar.pdf`
- Full-row heatmaps and train heatmaps: `results_probe_independent/paper_figures/schemeA_corr_heatmap_*_3x4.pdf`
- Scheme A full report: `results_probe_independent/schemeA_3x2x3_20260424_0137/schemeA_full_report.md`
- Combined synthetic MLP/ViT activation contrast: `results_script7/exp2_mlp_vit_combined/fig_exp2_combined_mlp_vit_depth8.pdf`

