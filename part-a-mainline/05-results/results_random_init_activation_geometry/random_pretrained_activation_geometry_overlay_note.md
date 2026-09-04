# Random vs pretrained activation geometry overlay

Figure: `fig_random_pretrained_activation_geometry_overlay.png` / `.pdf`.
Zoomed trend figure: `fig_random_pretrained_activation_geometry_overlay_zoom.png` / `.pdf`.

## Setup

- Pretrained ViT: `google/vit-base-patch16-224`.
- Pretrained Marian: `Helsinki-NLP/opus-mt-de-en`.
- No truncation: full available stacks are used.
- Columns: ReLU, GeLU, Light depth `t=2`.
- Legend: blue/white = random initialization; orange = pretrained checkpoint loaded, then activation swapped.
- x-axis is transition index: `ell=1` means `h_0 -> h_1`, not absolute block id. The plotting code disables Matplotlib boxplot auto-ticks and explicitly labels `1..L`.
- The zoomed figure uses balanced row-wise y-ranges, so the three activation columns in the same model row remain directly comparable. Triangles mark medians outside the visible range and are annotated with the clipped median value.
- Pretrained Marian input kind: `tokenized_sentences`.

## Experiment Settings

- **Random initialization.** The random baseline comes from `experiments/random_init_activation_geometry_compare.py`: a randomly initialized ViT with 12 encoder blocks, and a randomly initialized Marian-style encoder-decoder with 6 encoder layers and 6 decoder layers. No checkpoint is loaded.
- **Pretrained checkpoints.** The pretrained ViT is `google/vit-base-patch16-224`; the pretrained Marian model is `Helsinki-NLP/opus-mt-de-en`. Both are loaded from the Hugging Face cache when `--local_files_only` is used.
- **No truncation.** All available layers/blocks are kept: 12 ViT encoder blocks, 6 Marian encoder layers, and 6 Marian decoder layers.
- **Activation intervention.** For ViT, each encoder block MLP uses `layer.intermediate.intermediate_act_fn`. For Marian, both encoder and decoder FFN layers use `layer.activation_fn`. We compare ReLU, GeLU, and `LightDepthAct(2.0)`.
- **Inputs.** ViT uses Gaussian images with ImageNet normalization. Marian pretrained runs use short tokenized German/English sentence pairs for encoder and decoder inputs; random Marian runs use random token IDs from the configured vocabulary.
- **Metric.** For every hidden state `h_ell`, the token-by-hidden matrix is SVD-decomposed per sample; nuclear norms are averaged across the batch. The plotted quantity is `c_ell = N(h_ell) / N(h_{ell-1})`, so `ell=1` means the transition `h_0 -> h_1`.
- **Replicates.** Each box is over six seeds (`42..47`). The pretrained weights are fixed; seed changes only the input batch.

## Conference-style Caption

**Figure 41. Random-initialized versus pretrained layerwise geometric response under activation interventions.** Rows show ViT, Marian encoder, and Marian decoder; columns show ReLU, GeLU, and Light depth $t=2$. Each boxplot reports the layer-transition ratio $c_\ell=N(h_\ell)/N(h_{\ell-1})$, where $N(\cdot)$ is the batch-averaged nuclear norm of the token-by-hidden representation matrix. Blue/white boxes are random initialization; orange boxes load pretrained checkpoints (`google/vit-base-patch16-224` and `Helsinki-NLP/opus-mt-de-en`) and then replace the FFN/MLP activation while keeping weights fixed. No layers are truncated. The zoomed version uses balanced row-wise y-axis ranges; triangles denote clipped medians and print their values, so isolated extreme transitions do not obscure the main layerwise trend while preserving column-wise comparability within each row.

## How to Read the Figure

- The x-axis is **transition index**, not the absolute block label. `1` means `h_0 -> h_1`; ViT has transitions `1..12`, Marian encoder/decoder have `1..6`.
- The dashed horizontal line is `c_ell=1`. Values above 1 indicate nuclear-norm expansion across the transition; values below 1 indicate compression.
- Compare **blue/white vs orange within the same panel** to see how pretraining reshapes the geometry for a fixed architecture and activation.
- Compare **columns within a row** to see how the activation changes the geometry under the same model family.
- In the zoomed figure, each row shares the same y-axis range across its three activation columns. Upward/downward triangles mark medians outside the displayed y-range and print the clipped median. In particular, the Marian encoder's sixth transition is very low under pretraining (`0.36`, `0.37`, `0.47` for ReLU/GeLU/Light depth), the Marian decoder's sixth transition is very high (`4.16`, `4.26`, `4.12`), and pretrained ViT Light depth has a late high transition near `1.65`.

## Main Observations

### Random initialization

- **Random ViT:** all three activations have `c_ell > 1`, and the ratios gradually relax toward 1 with depth. ReLU has the strongest initial expansion, while Light depth `t=2` has the weakest.
- **Random Marian Encoder:** the first transition `h_0 -> h_1` strongly expands the nuclear norm (about `1.57`), while later transitions stay close to `0.99-1.00`.
- **Random Marian Decoder:** the first transition also strongly expands (about `1.56`), while later transitions are mostly slightly below 1 (about `0.985-0.989`).
- These random-init trends isolate the **initial geometric bias** of the architecture and activation before task training. They show that residual Transformer blocks are not geometrically neutral at initialization: the first transition can dominate the nuclear-norm scale, and activation choice changes the strength of this initial expansion.

### Pretrained geometric response

- Under pretrained checkpoints, the observed activation-swap response should be interpreted as `observed c_ell change = activation-intrinsic geometric effect + mismatch/interactions with GeLU-pretrained weights`.
- In other words, this panel does **not** claim that ReLU/GeLU/Light depth have a standalone universal ranking. It measures how sensitive an already trained GeLU representation flow is to activation perturbations.
- **ViT GeLU:** random initialization has the trend `1.18 -> 1.13 -> 1.10 -> ... -> 1.03`, i.e. early expansion followed by relaxation toward 1. The pretrained GeLU trajectory instead compresses in early blocks (`~0.97 -> 0.95 -> 0.97`) and then rises toward or above 1 in later blocks. This means pretraining reshapes the layerwise geometry rather than merely smoothing the random-init curve.
- **ViT Light depth:** pretrained weights are especially sensitive in the late ViT stack; around transition 10 the median `c_ell` is about `1.65`, far above the random-init Light depth curve.
- **Marian Encoder:** after pretraining, the encoder is no longer the random pattern of first-step expansion followed by near-identity transitions. It shows a learned layer structure, including mid-stack compression and a strong final compression.
- **Marian Decoder:** after pretraining, the final decoder transition expands extremely strongly (`c_ell` about `4.2`). This is qualitatively different from ViT and from the encoder, and is consistent with the decoder/generation side having a distinct geometric regime.

## LaTeX Body Text

```latex
\paragraph{Activation geometry before and after pretraining.}
Figure~\ref{fig41} compares layerwise geometric ratios under random initialization and pretrained checkpoints for three Transformer families: ViT, the Marian encoder, and the Marian decoder.  For each hidden state $h_\ell$, we compute the batch-averaged nuclear norm $N(h_\ell)$ of the token-by-hidden representation matrix and report the transition ratio
\begin{equation}
c_\ell = \frac{N(h_\ell)}{N(h_{\ell-1})},
\end{equation}
where $\ell=1$ denotes the transition $h_0\to h_1$.  The ViT row uses the full 12-block encoder, while Marian uses all 6 encoder layers and all 6 decoder layers; no stack is truncated.  The pretrained runs load \texttt{google/vit-base-patch16-224} and \texttt{Helsinki-NLP/opus-mt-de-en}, after which only the FFN/MLP activation is replaced by ReLU, GeLU, or Light depth $t=2$.

Random initialization exposes the initial geometric bias induced by architecture and activation.  In the ViT row, all three activations yield $c_\ell>1$ and gradually relax toward one as depth increases.  ReLU has the largest initial expansion, whereas Light depth $t=2$ has the weakest.  In the randomly initialized Marian encoder, the first transition expands strongly ($c_\ell\approx1.57$), but subsequent transitions remain near $0.99$--$1.00$.  The randomly initialized Marian decoder behaves similarly at the first transition ($c_\ell\approx1.56$), while later transitions are slightly contractive, around $0.985$--$0.989$.  Thus, before learning, the dominant geometric effect is an early residual/FFN expansion followed by near-critical propagation.

Pretraining qualitatively changes these trajectories.  For GeLU ViT, the random-initialized curve roughly follows $1.18\to1.13\to1.10\to\cdots\to1.03$, i.e. early expansion followed by relaxation toward one.  The pretrained GeLU curve instead compresses in the early blocks ($0.97\to0.95\to0.97$) and rises in the later blocks toward or above one.  This reversal indicates that pretraining does not merely smooth the initial geometry; it learns a task-adapted layerwise geometric trajectory in which early layers filter or compress representations and later layers re-expand or reorganize them.

Activation swaps under pretrained weights should therefore be interpreted as perturbations of a learned GeLU geometry rather than as pure activation-only effects.  The measured change in $c_\ell$ combines the intrinsic geometric bias of the activation with its interaction or mismatch with weights optimized under GeLU.  This interaction is especially visible for Light depth in pretrained ViT, where late transitions are strongly amplified, with transition 10 reaching a median $c_\ell\approx1.65$.  Marian shows a different pretrained response: the encoder develops mid-stack compression and a strong final compression, while the decoder exhibits an extreme final expansion ($c_\ell\approx4.2$).  The decoder behavior suggests that autoregressive generation induces a separate geometric regime from encoder-only visual recognition and from the translation encoder.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{fig_random_pretrained_activation_geometry_overlay_zoom.pdf}
  \caption{Random-initialized versus pretrained layerwise geometric response under activation interventions. Rows show ViT, Marian encoder, and Marian decoder; columns show ReLU, GeLU, and Light depth $t=2$. Each boxplot reports $c_\ell=N(h_\ell)/N(h_{\ell-1})$. Blue/white boxes are random initialization; orange boxes load pretrained checkpoints and then replace the FFN/MLP activation while keeping weights fixed. No layers are truncated. The displayed version uses balanced row-wise y-axis ranges; triangles mark medians outside the visible range and annotate their clipped values.}
  \label{fig41}
\end{figure}
```

## Files

- `pretrained_activation_c_raw.csv`: pretrained-only raw c rows.
- `random_pretrained_activation_c_raw.csv`: merged random + pretrained raw c rows.
- `random_pretrained_activation_c_summary.csv`: per transition summaries.

## Median snapshots

### ViT

| variant / activation | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `random` / `relu` | 1.2160 | 1.1323 | 1.0971 | 1.0752 | 1.0622 | 1.0529 | 1.0461 | 1.0400 | 1.0372 | 1.0333 | 1.0296 | 1.0278 |
| `random` / `gelu` | 1.1809 | 1.1257 | 1.0981 | 1.0791 | 1.0672 | 1.0578 | 1.0512 | 1.0450 | 1.0415 | 1.0377 | 1.0338 | 1.0318 |
| `random` / `light_depth_t2.0` | 1.1411 | 1.1053 | 1.0857 | 1.0719 | 1.0620 | 1.0545 | 1.0488 | 1.0431 | 1.0401 | 1.0367 | 1.0333 | 1.0314 |
| `pretrained` / `relu` | 0.9587 | 0.9508 | 0.9723 | 0.9792 | 0.9815 | 0.9880 | 1.0066 | 0.9964 | 1.0035 | 1.0219 | 1.0490 | 1.0923 |
| `pretrained` / `gelu` | 0.9730 | 0.9541 | 0.9726 | 0.9758 | 0.9792 | 0.9892 | 1.0048 | 1.0109 | 1.0255 | 1.0784 | 1.0747 | 1.0681 |
| `pretrained` / `light_depth_t2.0` | 0.9254 | 0.9421 | 0.9592 | 0.9656 | 0.9876 | 1.0263 | 1.0485 | 1.0779 | 1.1071 | 1.6501 | 1.0009 | 0.9835 |

### Marian encoder

| variant / activation | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `random` / `relu` | 1.5749 | 0.9939 | 0.9970 | 0.9958 | 0.9920 | 0.9960 |
| `random` / `gelu` | 1.5720 | 0.9934 | 0.9977 | 0.9969 | 0.9961 | 0.9960 |
| `random` / `light_depth_t2.0` | 1.5689 | 0.9921 | 0.9961 | 0.9953 | 0.9957 | 0.9938 |
| `pretrained` / `relu` | 1.0276 | 0.9851 | 0.8262 | 1.1338 | 1.0705 | 0.3558 |
| `pretrained` / `gelu` | 1.0309 | 0.9945 | 0.8386 | 1.1333 | 1.0614 | 0.3679 |
| `pretrained` / `light_depth_t2.0` | 1.0282 | 1.0190 | 0.8551 | 1.1401 | 1.0568 | 0.4721 |

### Marian decoder

| variant / activation | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `random` / `relu` | 1.5659 | 0.9874 | 0.9868 | 0.9845 | 0.9857 | 0.9854 |
| `random` / `gelu` | 1.5663 | 0.9880 | 0.9893 | 0.9854 | 0.9871 | 0.9892 |
| `random` / `light_depth_t2.0` | 1.5641 | 0.9871 | 0.9898 | 0.9857 | 0.9865 | 0.9887 |
| `pretrained` / `relu` | 0.6620 | 0.7898 | 0.9610 | 0.8972 | 1.2537 | 4.1591 |
| `pretrained` / `gelu` | 0.6691 | 0.8152 | 0.9927 | 0.9143 | 1.2772 | 4.2561 |
| `pretrained` / `light_depth_t2.0` | 0.6761 | 0.9361 | 1.0008 | 0.9529 | 1.0980 | 4.1211 |
