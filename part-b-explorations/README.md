# Part B — 探索性与负面结果

本目录记录 **未成为主线方法** 的假设与实验。它们与 Part A 同样重要：说明「为什么最后只编辑 value 路径、不碰 QK 路由」。

---

## 时间线

### 2026-04 初 — 合成谱（已在 Part A 成立）

假设 H1：点式激活 + 深度 → 谱病态。script1–4 验证 ✅ → 进入正文。

---

### 2026-04-22 — 直接改 Q/K/V 谱 ❌

**假设**：对 Q、K、V 投影做核范数规整，或用 K⊙V、Q-scaling 等「几何正确」的改法，可在低资源 MT 上增益。

**实验**：`tlb_attention_core.py`、`marian_attn_tlb_patch.py`、`script11_tlb_attention.py`

**结果**：

| 方法 | BLEU（量级） |
|------|-------------|
| Pretrained | ~19.9 |
| Vanilla finetune | ~15.5 |
| `qkv_proj_nuc` | **~0** |
| `kv_inner` finetune | **~0.15** |

**结论**：动 QK 路由或硬改投影谱 → 灾难性崩溃。引向 **value-only** 干预。

**阅读**：`summaries/EXPERIMENT_ALIGNMENT_GUIDE.md`、`summaries/results_marian_qkv_proj_nuc/summary.json`

---

### 2026-04-23 — V-smooth 网格搜索 ⚠️

**假设**：只对 value 做 temporal/spatial 平滑（α 网格），不碰 QK。

**结果**：部分 α 有效（如 temporal α=0.2 → BLEU ~17.5），但无单一简单公式；过程摘要见 `summaries/v_smooth_experiments_summary.md`。

**说明**：完整 checkpoint 网格 (~16GB) 未打包；见根目录 `SIZE_AUDIT.md`。

---

### 2026-04-29+ — 架构分叉 ❌ / 探索

| 脚本 | 假设 | 结果 |
|------|------|------|
| **script8** | 正交 mem/smart 双流保记忆 | dual ≈ baseline |
| **script9** | FFN expand 后拆双路 | toy 无 MT 增益 |
| **script10** | Token mirror 并行支路 | 合成 LM only |
| **script11** | TLB attention 模式族 | 与 Q/K 实验同族，Marian 无 win |

代码：`code/script8_dual_space_memory.py` … `code/script11_tlb_attention.py`

---

## 开放问题（实验有、理论浅）

1. **Spatial vs temporal 顺序**：Encoder 对顺序更敏感；Decoder 在 Table 4 中顺序不敏感 — 优化路径 vs 算子交换性，未单独做 FFN 因果分解。  
2. **Decoder fragility**：梯度/CKA 与 PPL spike 支持「Decoder 更 fragile」的 **诊断叙事**；最佳 BLEU 上 Encoder 仍可略高 — 非单一 ablation 可锁定。  
3. **两套 finetune 协议**：800-step stress-test vs 32-batch probe — 读 BLEU 时必须对照 [REPRODUCTION.md](../REPRODUCTION.md)。

---

## 建议阅读

1. `summaries/EXPERIMENT_ALIGNMENT_GUIDE.md` — 探针 vs 正式实验  
2. `summaries/Discussion_Conclusion.md` — 作者向 synthesis  
3. `summaries/docs/VIT_MARIAN_ROOT_CAUSE.md` — ViT vs Marian 几何讨论  

---

## 代码入口

```bash
# 负面 QKV（勿期望高 BLEU）
python code/script11_tlb_attention.py --help

# Dual memory toy
python code/script8_dual_space_memory.py --help
```

完整负面 run 的 checkpoint 在本地 `TLB-Experiments/results_marian_*`（未全部打包）。
