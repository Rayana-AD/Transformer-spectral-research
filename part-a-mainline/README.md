# Part A — 论文主线

本目录包含 **可复现论文核心 claim** 的代码、数据与归档结果：合成谱理论 → ViT 桥 → Marian value 级联。

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
| Vanilla finetune | ~15.1–15.5 | `06-checkpoints/eval_manifests/` |
| Scheme A best (Enc) | ~18.49 | `schemeA_full_report.md` |
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
