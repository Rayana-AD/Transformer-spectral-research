# Seed-Averaged Four-Coefficient Correlation Heatmaps

Generated figures:

- `schemeA_corr_heatmap_encoder_eval_seedavg_3x4.png`
- `schemeA_corr_heatmap_encoder_eval_seedavg_3x4.pdf`
- `schemeA_corr_heatmap_encoder_train_seedavg_3x4.png`
- `schemeA_corr_heatmap_encoder_train_seedavg_3x4.pdf`
- `schemeA_corr_heatmap_decoder_eval_seedavg_3x4.png`
- `schemeA_corr_heatmap_decoder_eval_seedavg_3x4.pdf`
- `schemeA_corr_heatmap_decoder_train_seedavg_3x4.png`
- `schemeA_corr_heatmap_decoder_train_seedavg_3x4.pdf`

## Source Data and Aggregation

- Encoder uses `schemeA_3x2x3_20260424_0137/cascade_rev_s0.1_t0.2_seed*/result.json`, i.e. Temporal first at `(alpha_s, alpha_t)=(0.1,0.2)`.
- Decoder uses `schemeA_dec_3x2x3_20260425_1943/cascade_dec_s0.1_t0.2_seed*/result.json`, i.e. Spatial first at `(alpha_s, alpha_t)=(0.1,0.2)`.
- Each heatmap cell is averaged over seeds 42, 43, and 44.
- Eval figures correlate probe metrics with validation batch BLEU. Train figures correlate probe metrics with training loss.
- Rows inside each heatmap are probe metrics; columns are layer indices 0--5. The outer grid is cascade stage (3 rows) by association method (4 columns).
- Numeric annotations inside each cell are rounded to one decimal place.

## Suggested Caption: Eval Encoder

**Seed-averaged four-coefficient association maps for Encoder evaluation.** Each subplot correlates a probe metric with validation batch BLEU across layers. Rows correspond to probe metrics and columns to Encoder layer indices. The outer grid is organized by cascade stage and association method. The Encoder map shows that BLEU is positively associated with temporal cosine and negatively associated with nuclear norm and temporal difference, indicating that validation quality tracks coherent temporal geometry more strongly than raw value capacity.

## Suggested Caption: Eval Decoder

**Seed-averaged four-coefficient association maps for Decoder evaluation.** Each subplot correlates Decoder self-attention value probes with validation batch BLEU. Compared with Encoder, Decoder maps show stronger negative associations for nuclear norm and temporal difference, while temporal cosine remains positively associated with BLEU. This supports the interpretation that decoding benefits from smoother directional evolution rather than larger value-space capacity.

## Suggested Caption: Train Encoder

**Seed-averaged four-coefficient association maps for Encoder training.** Each subplot correlates probe metrics with training loss across layers and cascade stages. Compared with eval maps, train associations are weaker and less monotonic, showing that optimization loss alone does not fully explain the validation geometry revealed by BLEU correlations.

## Suggested Caption: Train Decoder

**Seed-averaged four-coefficient association maps for Decoder training.** Each subplot correlates Decoder probe metrics with training loss. The train maps serve as optimization diagnostics and should be interpreted together with eval BLEU maps, because PPL/loss calibration can be unstable even when decoded BLEU remains competitive.

## Paper Placement Recommendation

Use the two eval maps (Encoder eval and Decoder eval) in the main paper if space allows, because they directly connect representation geometry to translation quality. Keep the two train maps in the appendix or supplementary material as optimization diagnostics. The train maps are useful, but they are less directly tied to the final translation metric than eval BLEU associations.

## Quick Numeric Checks

### Encoder eval

- Mean Spearman for `V_raw.nuclear`: -0.74
- Mean Spearman for `V_raw.temporal_cosine_mean`: +0.65
- Mean Spearman for `V_raw.temporal_diff_mean`: -0.66
- Mean MIC for `V_raw.temporal_cosine_mean`: +0.85

### Decoder eval

- Mean Spearman for `V_raw.nuclear`: -0.77
- Mean Spearman for `V_raw.temporal_cosine_mean`: +0.73
- Mean Spearman for `V_raw.temporal_diff_mean`: -0.73
- Mean MIC for `V_raw.temporal_cosine_mean`: +0.85

### Encoder train

- Mean Spearman for `V_raw.nuclear`: +0.13
- Mean Spearman for `V_raw.temporal_cosine_mean`: -0.14
- Mean Spearman for `V_raw.temporal_diff_mean`: +0.12
- Mean MIC for `V_raw.temporal_cosine_mean`: +0.30

### Decoder train

- Mean Spearman for `V_raw.nuclear`: -0.01
- Mean Spearman for `V_raw.temporal_cosine_mean`: -0.21
- Mean Spearman for `V_raw.temporal_diff_mean`: +0.06
- Mean MIC for `V_raw.temporal_cosine_mean`: +0.31


---

## Revised Main-Text Eval Heatmaps (Rows Aligned With Formulas)

Generated figures:

- `schemeA_corr_heatmap_encoder_eval_seedavg_3x4_mainrows.png`
- `schemeA_corr_heatmap_encoder_eval_seedavg_3x4_mainrows.pdf`
- `schemeA_corr_heatmap_decoder_eval_seedavg_3x4_mainrows.png`
- `schemeA_corr_heatmap_decoder_eval_seedavg_3x4_mainrows.pdf`

Row set used in these main-text heatmaps:

1. Nuclear norm
2. Element mean
3. L2/Frobenius norm
4. Temporal cosine
5. Temporal difference
6. Spatial variance

`Temp. var.` was removed from the main-text heatmap row set because it is less directly tied to the formulas emphasized in the method section and showed weaker layer-specific interpretability than temporal difference in the current figures. Full heatmaps with temporal variance remain available in the earlier generated appendix versions.

Quick Vraw/Spearman layer-mean checks:

| Target | Nuclear | Elem. mean | L2 norm | Temp. cosine | Temp. diff | Spat. var. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Encoder eval | -0.74 | -0.52 | -0.10 | +0.65 | -0.66 | -0.08 |
| Decoder eval | -0.77 | +0.13 | -0.54 | +0.73 | -0.73 | -0.54 |

## Decoder Gradient and CKA Summary

These quantities are not merged into the value-geometry heatmaps because they have different granularity: gradient is optimization-level and CKA is FFN representation-level, while the heatmaps are stage-by-layer value-geometry associations. They are better reported as a separate Decoder-only summary table.

| Order | $\alpha_s$ | $\alpha_t$ | BLEU | PPL | $G_{\mathrm{dec}}$ | CKA$_{\mathrm{dec2}}$ | CKA$_{\mathrm{dec5}}$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial first | 0.1 | 0.2 | 18.24±0.13 | 19.92±1.87 | 32.78±0.62 | 0.9192±0.0015 | 0.9767±0.0012 |
| Spatial first | 0.2 | 0.1 | 17.97±0.37 | 468.61±628.83 | 248.99±306.42 | 0.9184±0.0010 | 0.9794±0.0028 |
| Spatial first | 0.2 | 0.2 | 18.23±0.09 | 22.15±2.41 | 32.70±0.65 | 0.9195±0.0014 | 0.9780±0.0014 |
| Temporal first | 0.1 | 0.2 | 18.24±0.13 | 19.92±1.87 | 32.78±0.62 | 0.9192±0.0015 | 0.9767±0.0012 |
| Temporal first | 0.2 | 0.1 | 18.03±0.41 | 892.06±616.06 | 456.42±299.90 | 0.9178±0.0016 | 0.9814±0.0040 |
| Temporal first | 0.2 | 0.2 | 18.16±0.18 | 682.93±933.78 | 272.12±338.39 | 0.9192±0.0015 | 0.9807±0.0025 |

Interpretation: stable Decoder configurations have $G_{\mathrm{dec}}\approx 33$, whereas unstable high-spatial-strength settings can exceed $250$--$450$ on average. CKA remains high, with dec5 consistently higher than dec2, suggesting that the late Decoder FFN is more representation-preserving than the middle Decoder FFN even when optimization calibration changes strongly.
