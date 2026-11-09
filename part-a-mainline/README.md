# Part A — 方向 1 主线

本目录归档 **value-path 谱诊断与 Scheme A 级联探针**：在 **数据极少、微调步数短、AdamW 未进入稳定 regime** 时，观察 **Encoder/Decoder 训练期几何** 与 **loss/PPL 稳定性** 的关联。级联主要用于 **可度量的探针**，而非充足数据下的默认增益模块。

仓库根目录 **图 07**（`figures/gallery/07_schemeA_depth_nuclear_norm.png`）展示 **Enc 中层膨胀 vs Dec 浅高–中谷–深回升** 的训练期核范数深度曲线。

## 阅读顺序

1. **`05-results/paper_assets/`** — 合成段主图（激活热力图、深度 conditioning、script4 曲线）
2. **`05-results/results_probe_independent/schemeA_*`** — Scheme A 完整矩阵报告与 `result.json`
3. **`05-results/results_marian_ffn_frozen_eval/`** — 预训练 19.9 vs 换 FFN 激活崩溃
4. **`06-checkpoints/`** — vanilla finetune seed42（重 eval）+ 三 seed eval manifest

## 子目录

| 目录 | 内容 | 对应论文 |
|------|------|----------|
| `01-theory-scripts/` | script1–4 | §2 Activation spectral scattering |
| `02-vit-bridge/` | script5/7, geometry overlay | §3 / appendix ViT bridge |
| `03-marian-cascade/` | `tlb_value_smooth`, probes, Scheme A shells | §4–5 Cascade diagnostic |
| `04-data/` | `opus_de_en` train_5k / val_500 | MT stress-test 数据 |
| `05-results/` | CSV、JSON、PDF/PNG | Tables & figures |
| `06-checkpoints/` | 1× HF finetune + eval JSON | Baseline 复现 |
| `docs/` | 图–表映射、探针说明 | 内部索引 |

## 关键数字（archived）

| 指标 | 值 | 来源 |
|------|-----|------|
| Pretrained BLEU | 19.916 | `results_marian_ffn_frozen_eval/summary.json` |
| Vanilla finetune（旧 stress / 标签协议） | ~15.1–15.5 | `06-checkpoints/eval_manifests/` |
| Vanilla finetune（matched `text_target`） | ~20.8–21 | 见根目录 `REPRODUCTION.md` |
| Scheme A 探针（32 batch，Enc） | ~17.5–18.5 | `schemeA_full_report.md` |
| QKV nuc rescale | ~0 | Part B summary |

## 最小复现

```bash
# 合成（CPU）
cd 01-theory-scripts && python script2_universal_validation.py

# Marian eval（GPU）
cd ../03-marian-cascade
python marian_mt_finetune_tlb.py \
  --checkpoint ../06-checkpoints/vanilla_finetune_seed42 \
  --eval-only --max-val 128 --seed 42
```

Scheme A 全矩阵：`bash 03-marian-cascade/experiments/run_schemeA_cascade_matrix.sh`  
协议说明见仓库根目录 [REPRODUCTION.md](../REPRODUCTION.md)。
