# Cascade spectral operator verification

Reproducible via `experiments/verify_cascade_spectral_operators.py`.

## 1. Torch vs explicit matrix operators

- circular_roll+spatial_global: temporal err **2.53e-07**, spatial err **2.79e-07**
- causal_prev+prefix_mean: temporal err **2.53e-07**, spatial err **2.65e-07**

## 2. Commutativity (α_s=0.1, α_t=0.2)

- S=64, circular_roll+spatial_global: ||[S_spat,S_temp]||_2=**0.0000**
- S=64, causal_prev+prefix_mean: ||[S_spat,S_temp]||_2=**0.0099**
- Encoder (circular+global): commutator ≈ 0 ⇒ final V is order-invariant; order only relabels intermediate probe stages.

## 3. Bound ||O||_* ≤ ||A||_2 ||V'||_* (numeric)
- max ratio over 50 trials: **0.0911**

## 4. Temporal low-pass |λ(ω)| at S=64
- DC k=0: **1.0000**; Nyquist k=32: **0.6000**

## 5. Activation vs cascade (S=64,D=64, 20 seeds; parallels script2)

| stage | nuclear (mean) | med κ |
|---|---:|---:|
| act:linear | 3049.6 | 6274.0 |
| act:gelu | 2070.6 | 1023.5 |
| act:relu | 2064.8 | 1012.5 |
| act:silu | 2078.3 | 990.2 |
| cascade:raw | 433.2 | 205.3 |
| cascade:spatial_only | 390.6 | 205.1 |
| cascade:temporal_only | 353.2 | 227.0 |
| cascade:spatial_first | 318.6 | 229.9 |

## 6. Marian enc_0 probe vs synthetic 128×64
- nuclear_ratio_first_to_raw: probe **0.9238**, synthetic **0.9008**
- nuclear_ratio_second_to_raw: probe **0.8234**, synthetic **0.7380**