# Random-init activation geometry: ViT vs Marian encoder/decoder

Figure: `fig_random_init_activation_geometry_vit_marian.png` / `.pdf`.

## Setup

- **No pretrained checkpoints loaded**: all weights are random initialization from config.
- **No truncation**: ViT uses the full configured encoder stack; Marian uses full configured encoder and decoder stacks.
- **Columns** are ordered left-to-right: **ReLU**, **GeLU**, **Light depth $t=2$**.
- **x-axis** is `Block/layer transition ell`: transition `ell=1` means hidden states `h_0 -> h_1`, not an absolute block label.
- Seeds: `6`; batch: `2`.
- ViT config: layers=12, hidden=768, heads=12, ffn=3072, image_size=224, patch=16.
- Marian config: encoder_layers=6, decoder_layers=6, d_model=512, heads=8, ffn=2048, seq_len=32, decoder_seq_len=32.

## Data

- Raw c rows: `random_init_activation_c_raw.csv` (432 rows).
- Per-transition summaries: `random_init_activation_c_summary.csv` (72 rows).

## Median $c_\ell$ by transition

### vit_random

| activation | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `relu` | 1.2160 | 1.1323 | 1.0971 | 1.0752 | 1.0622 | 1.0529 | 1.0461 | 1.0400 | 1.0372 | 1.0333 | 1.0296 | 1.0278 |
| `gelu` | 1.1809 | 1.1257 | 1.0981 | 1.0791 | 1.0672 | 1.0578 | 1.0512 | 1.0450 | 1.0415 | 1.0377 | 1.0338 | 1.0318 |
| `light_depth_t2.0` | 1.1411 | 1.1053 | 1.0857 | 1.0719 | 1.0620 | 1.0545 | 1.0488 | 1.0431 | 1.0401 | 1.0367 | 1.0333 | 1.0314 |

### marian_encoder_random

| activation | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `relu` | 1.5749 | 0.9939 | 0.9970 | 0.9958 | 0.9920 | 0.9960 |
| `gelu` | 1.5720 | 0.9934 | 0.9977 | 0.9969 | 0.9961 | 0.9960 |
| `light_depth_t2.0` | 1.5689 | 0.9921 | 0.9961 | 0.9953 | 0.9957 | 0.9938 |

### marian_decoder_random

| activation | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `relu` | 1.5659 | 0.9874 | 0.9868 | 0.9845 | 0.9857 | 0.9854 |
| `gelu` | 1.5663 | 0.9880 | 0.9893 | 0.9854 | 0.9871 | 0.9892 |
| `light_depth_t2.0` | 1.5641 | 0.9871 | 0.9898 | 0.9857 | 0.9865 | 0.9887 |
