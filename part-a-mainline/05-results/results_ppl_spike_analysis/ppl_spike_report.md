# Scheme A PPL Spike Analysis

Train PPL from JSON is mean over **last 20 batches**; step-wise CSV reveals spikes.

| Run | BLEU | train_ppl (json) | ppl_max_step | ppl_p90_step | grad_dec_max | V_raw mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cascade_dec_s0.1_t0.2_seed42 | 18.07 | 19.3 | 61859.6 | 11515.7 | 51.8 | 3830 |
| cascade_rev_dec_s0.2_t0.1_seed43 | 18.39 | 1296.7 | 149704.6 | 7831.0 | 2190.1 | 3858 |
| cascade_rev_dec_s0.2_t0.1_seed44 | 17.45 | 1357.9 | 31552.0 | 7756.6 | 1931.1 | 3825 |
| cascade_s0.1_t0.2_seed44 | 17.77 | 39860.0 | 167151.7 | 105225.3 | nan | 3191 |

## Case studies

- `cascade_dec_s0.1_t0.2_seed42`: see `cascade_dec_s0.1_t0.2_seed42_ppl_spike.png`
- `cascade_rev_dec_s0.2_t0.1_seed43`: see `cascade_rev_dec_s0.2_t0.1_seed43_ppl_spike.png`
- `cascade_rev_dec_s0.2_t0.1_seed44`: see `cascade_rev_dec_s0.2_t0.1_seed44_ppl_spike.png`
- `cascade_s0.1_t0.2_seed44`: see `cascade_s0.1_t0.2_seed44_ppl_spike.png`