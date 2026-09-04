# Regularization baseline summary

| exp | cascade | label_smooth | attn_dropout | BLEU | train_ppl | eval_ppl |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cascade_dec_s0.1_t0.2_seed42_cascade_stable | True | 0.0 | None | 17.61 | 3533.0 | 19897.0 |
| vanilla_dec_seed42_attn_dropout_0.2 | False | 0.0 | 0.2 | 17.68 | 17.5 | 19.1 |
| vanilla_dec_seed42_label_smooth_0.1 | False | 0.1 | None | 17.79 | 213467.4 | 433460.4 |
| vanilla_dec_seed42_vanilla_finetune | False | 0.0 | None | 18.16 | 19.3 | 17.9 |