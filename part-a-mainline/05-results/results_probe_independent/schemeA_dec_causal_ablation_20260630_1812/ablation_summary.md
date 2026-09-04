# Marian decoder causal ablation summary

Source: `/Users/rayana/Desktop/TLB-Experiments/results_probe_independent/schemeA_dec_causal_ablation_20260630_1812`

| exp | temporal | spatial | BLEU | train_ppl | grad_decoder peak | eval_ce_p99 | eval_ppl |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cascade_dec_s0.1_t0.2_seed42_stable_causal | causal_prev | prefix_mean | 18.39 | 76.0 | 32.1 | 14.24 | 71.6 |
| cascade_dec_s0.1_t0.2_seed42_stable_circular | circular_roll | spatial_global | 17.97 | 264.2 | 79.2 | 14.30 | 429.0 |
| cascade_rev_dec_s0.2_t0.1_seed42_spike_causal | causal_prev | prefix_mean | 17.66 | 2566.5 | 1599.8 | 14.58 | 19550.1 |
| cascade_rev_dec_s0.2_t0.1_seed42_spike_circular | circular_roll | spatial_global | 17.66 | 5313.1 | 1914.7 | 14.49 | 20602.1 |

Generated from 4 runs.
