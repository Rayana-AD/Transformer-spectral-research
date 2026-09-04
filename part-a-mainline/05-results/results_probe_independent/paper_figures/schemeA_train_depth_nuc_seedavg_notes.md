# Scheme A Cascade Depth Figure Notes

Generated figure files:

- `schemeA_train_depth_nuc_seedavg_alpha_s01_t02.png`
- `schemeA_train_depth_nuc_seedavg_alpha_s01_t02.pdf`

Source data:

- Encoder: `results_probe_independent/schemeA_3x2x3_20260424_0137`
- Decoder: `results_probe_independent/schemeA_dec_3x2x3_20260425_1943`
- Configuration: `alpha_s=0.1`, `alpha_t=0.2`
- Seeds: 42, 43, 44
- Quantity: train-time mean nuclear norm from `probe_summaries` in each `result.json`

## Suggested Caption

**Train-time depth profile of cascade nuclear norm in Encoder and Decoder self-attention.**
For each target module, we plot the mean nuclear norm of the attention value tensor across three seeds for Spatial first and Temporal first cascade orders. Each order contributes three curves corresponding to raw value states, first-stage cascade states, and second-stage cascade states. Shaded bands show seed-level standard deviation. The Encoder exhibits a mid-layer expansion pattern, whereas the Decoder keeps a higher shallow-layer value norm and shows a more pronounced compression valley toward the middle layers.

## How To Read The Axes

- **x-axis:** Transformer layer index, from shallow layer 0 to deep layer 5.
- **y-axis:** Nuclear norm of the value tensor, computed after reshaping `V` to a two-dimensional matrix for Schatten-1 norm evaluation.
- **Color:** Cascade order. Orange is Spatial first; blue is Temporal first.
- **Line style / marker:** Cascade stage. Solid/circle is raw `V`; dashed/square is first stage; dotted/triangle is second stage.
- **Shaded region:** Standard deviation across three seeds.

## Related Data: Encoder

| Order | Stage | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Spatial first | V^{\mathrm{raw}} | 3205.1 | 2890.8 | 3360.7 | 3199.9 | 3035.5 | 3161.1 |
| Spatial first | V^{(1)} | 2961.0 | 2667.9 | 3112.5 | 2954.1 | 2814.2 | 2931.7 |
| Spatial first | V^{(2)} | 2639.6 | 2401.3 | 2819.7 | 2654.2 | 2568.9 | 2685.6 |
| Temporal first | V^{\mathrm{raw}} | 3205.2 | 2889.9 | 3358.7 | 3204.6 | 3037.9 | 3156.1 |
| Temporal first | V^{(1)} | 2846.6 | 2591.3 | 3030.2 | 2867.6 | 2761.6 | 2878.5 |
| Temporal first | V^{(2)} | 2639.6 | 2400.4 | 2817.7 | 2658.9 | 2571.0 | 2680.3 |

## Related Data: Decoder

| Order | Stage | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Spatial first | V^{\mathrm{raw}} | 3836.7 | 2666.7 | 2340.9 | 2275.5 | 2619.6 | 2472.7 |
| Spatial first | V^{(1)} | 3501.6 | 2416.4 | 2121.0 | 2064.7 | 2386.7 | 2317.3 |
| Spatial first | V^{(2)} | 3045.5 | 2060.2 | 1809.7 | 1787.0 | 2106.3 | 2149.1 |
| Temporal first | V^{\mathrm{raw}} | 3836.7 | 2666.7 | 2340.9 | 2275.5 | 2619.6 | 2472.7 |
| Temporal first | V^{(1)} | 3327.7 | 2269.1 | 1993.2 | 1964.9 | 2305.5 | 2282.3 |
| Temporal first | V^{(2)} | 3045.5 | 2060.2 | 1809.7 | 1787.0 | 2106.3 | 2149.1 |

## Interpretation

1. **Cascade stages are consistently contractive.** Across both Encoder and Decoder, the order `V raw > V first > V second` is preserved for nearly all layers, showing that the cascade acts as a controlled compression operator on value representations.

2. **Encoder has a mid-layer expansion profile.** Encoder curves peak around layer 2 and remain relatively high in deeper layers. This supports the view that the Encoder builds a high-capacity bidirectional representation in the middle depth before passing information to downstream decoding.

3. **Decoder has a stronger shallow-to-middle compression profile.** Decoder layer 0 starts with the largest value nuclear norm, then drops sharply through middle layers and partially recovers toward layer 5. This is consistent with autoregressive decoding: shallow self-attention must preserve rich generated-history information, while middle layers filter and consolidate it before output-facing reconstruction.

4. **Order effects are smaller than target-module effects at this setting.** For `alpha_s=0.1`, `alpha_t=0.2`, Spatial first and Temporal first are close within each target, especially for Decoder. The dominant qualitative difference is therefore Encoder versus Decoder, not order alone.

5. **Connection to performance tables.** This figure corresponds to the best or near-best Scheme A setting in the summary tables. Encoder favors Temporal first at this setting, while Decoder performs similarly for both orders, matching the visual observation that Decoder order curves nearly overlap after seed averaging.


---

## Figure 32 Close-up: Nuclear Norm vs Loss/PPL

Generated figure files:

- `schemeA_train_nuclear_loss_ppl_closeup_enc2_dec3_dec5_seedavg.png`
- `schemeA_train_nuclear_loss_ppl_closeup_enc2_dec3_dec5_seedavg.pdf`

Source data:

- Encoder layer 2: `cascade_rev_s0.1_t0.2_seed*/probe_data/probe_train_enc_2.csv`
- Decoder layer 3: `cascade_dec_s0.1_t0.2_seed*/probe_data/probe_train_dec_3.csv`
- Decoder layer 5: `cascade_dec_s0.1_t0.2_seed*/probe_data/probe_train_dec_5.csv`
- Seeds: 42, 43, 44
- Each trajectory averages the same training step across seeds; `x` marks the first step and `*` marks the last step.
- PPL is plotted on a logarithmic scale to keep high-PPL steps visible without flattening the low-PPL regime.

### Suggested Caption

**Training dynamics of value nuclear norm for representative Encoder and Decoder layers.** Left: Encoder layer 2 under Temporal-first cascade, the layer highlighted by the depth-profile peak. Middle: Decoder layer 3 under Spatial-first cascade, the compression valley in the Decoder depth profile. Right: Decoder layer 5 under Spatial-first cascade, a late output-facing layer after the shallow-to-middle compression valley. Each curve is averaged over three seeds and follows the time order of training steps for the three cascade stages. The top row relates nuclear norm to training loss, and the bottom row relates nuclear norm to PPL on a logarithmic scale.

### How To Read

- **Columns:** representative modules/layers: Encoder layer 2, Decoder layer 3, and Decoder layer 5.
- **Rows:** loss versus nuclear norm (top), PPL versus nuclear norm (bottom).
- **Colors:** blue/orange/green correspond to raw, first-stage, and second-stage value states.
- **Trajectory direction:** `x` is the first averaged training step; `*` is the final averaged training step.
- **Interpretation:** horizontal separation between stages shows the cascade contraction; vertical movement shows optimization/calibration changes during training.

### Related Data

### Encoder layer 2

- Steps averaged: 32
- Mean loss range: 2.086--11.045
- Mean PPL range: 8.0--62639.7

| Stage | Nuclear norm range | Mean nuclear norm |
| --- | --- | --- |
| V^{\mathrm{raw}} | 3220.3--3524.8 | 3358.7 |
| V^{(1)} | 2925.5--3148.3 | 3030.2 |
| V^{(2)} | 2732.1--2910.6 | 2817.7 |

### Decoder layer 3

- Steps averaged: 32
- Mean loss range: 1.567--10.859
- Mean PPL range: 4.8--51992.5

| Stage | Nuclear norm range | Mean nuclear norm |
| --- | --- | --- |
| V^{\mathrm{raw}} | 1941.0--2695.7 | 2275.5 |
| V^{(1)} | 1758.5--2451.4 | 2064.7 |
| V^{(2)} | 1514.8--2128.9 | 1787.0 |

### Decoder layer 5

- Steps averaged: 32
- Mean loss range: 1.567--10.859
- Mean PPL range: 4.8--51992.5

| Stage | Nuclear norm range | Mean nuclear norm |
| --- | --- | --- |
| V^{\mathrm{raw}} | 1969.7--2874.2 | 2472.7 |
| V^{(1)} | 1833.2--2698.4 | 2317.3 |
| V^{(2)} | 1683.9--2500.7 | 2149.1 |

### Interpretation

The Encoder layer 2 close-up corresponds to the mid-layer expansion observed in the depth-profile figure. Its three cascade stages are well separated in nuclear norm, showing that the probe creates a stable compression ladder while training loss evolves along the trajectory. Decoder layer 3 captures the compression valley: it has the lowest mean nuclear norms among the selected panels and makes the history-filtering interpretation visually explicit. Decoder layer 5 then shows a later, output-facing reconstruction regime after the valley. This figure is most useful as a supplementary or small main-text close-up: it explains how the layerwise depth profile connects to training-time loss/PPL dynamics, but it should not replace the main depth-profile figure.
