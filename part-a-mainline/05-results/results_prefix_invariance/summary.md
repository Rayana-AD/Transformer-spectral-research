# Prefix-Invariance Test Summary

Compare logits on a shared core span when **future** tokens differ. Strict causal value smooth should yield near-zero diff; circular/global smooth may not.

## qwen3

| Condition | temporal | spatial | mean max logit diff | mean KL | n |
| --- | --- | --- | ---: | ---: | ---: |
| vanilla | None | None | 0.0000 | 0.000000 | 8 |
| cascade_circular_global | circular_roll | spatial_global | 3.8047 | 0.072747 | 8 |
| cascade_causal_prefix | causal_prev | prefix_mean | 0.0000 | 0.000000 | 8 |
