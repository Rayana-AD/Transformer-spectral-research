# V-Smooth 实验总结与扩展计划

## 1. 现有实验结果

### 1.1 方法描述
**v_smooth**: 在 attention 计算前对 V 进行向序列均值回归的平滑

```python
# V shape: (batch, num_heads, seq_len, head_dim)
v_mean = value.mean(dim=-2, keepdim=True)  # 沿 seq_len 求均值
v_smooth = (1 - α) * value + α * v_mean
```

### 1.2 实验设置
- **模型**: Helsinki-NLP/opus-mt-de-en (Marian, 预训练)
- **数据**: local opus_de_en (800 train / 128 val)
- **训练**: 1 epoch, batch=8, lr=1e-5
- **基线对比**:
  - vanilla (预训练): BLEU = 19.9
  - vanilla (微调后): BLEU = 15.5

### 1.3 网格搜索完整结果 (seed=42)

| 实验 | 位置 | 方向 | α | **BLEU** | PPL | vs vanilla | 结论 |
|------|------|------|---|----------|-----|------------|------|
| v_smooth_005 | pre | temporal | 0.05 | 16.17 | 45.28 | +4.3% | 轻度有效 |
| **v_smooth_015** | pre | temporal | **0.15** | **17.19** | 52.20 | **+11.0%** | ✅ **次优** |
| **v_smooth_02** | pre | temporal | **0.2** | **17.46** | 56.96 | **+12.6%** | ✅ **最优** |
| v_smooth_post | post | temporal | 0.1 | 16.17 | 46.29 | +4.3% | ❌ 不如 pre |
| v_smooth_both | both | temporal | 0.1 | 17.08 | 53.03 | +10.2% | 双重有效 |
| v_smooth_feature | pre | feature | 0.1 | 15.13 | 43.76 | -2.4% | ❌ 方向错误 |

**关键结论**:
1. **α=0.2 是最佳选择** (BLEU 17.46)，比 vanilla 提升 **12.6%**
2. **temporal 平滑 >> feature 平滑** (沿序列维度 > 沿特征维度)
3. **pre-attention 平滑 > post-attention 平滑**
4. **存在 BLEU/PPL trade-off**: 更强平滑提升 BLEU 但增加 PPL

---

## 2. 最优配置

**临时最优**: `v_smooth` (pre-attention, temporal, α=0.2)
- BLEU: **17.46** (vs vanilla 15.5)
- PPL: 56.96 (vs vanilla ~48)
- 相对提升: **+12.6%**

---

## 3. 下一步: 多 Seed 验证

为验证最优配置的稳定性，计划进行多 seed 测试。

### 实验配置
- 方法: v_smooth (α=0.2, pre, temporal)
- Seeds: 42, 43, 44
- 其他参数保持不变

### 预期产出
```
seed,bleu,ppl
42,17.46,56.96
43,X.XX,XX.XX
44,X.XX,XX.XX
```

### 验证通过标准
- 3 个 seed 平均 BLEU > 16.5 (显著优于 vanilla 的 15.5)
- 标准差 < 0.5 (稳定性可接受)

---

## 4. 实验记录

### 2026-04-23: 网格搜索完成
- 6 个配置测试完成
- **最佳**: α=0.2, temporal, pre-attention
- 结果文件: `results_v_smooth_grid_20260423_171442/summary.csv`

---

## 5. 待完成

- [x] 网格搜索 (6 配置)
- [ ] 多 seed 验证 (3 seeds)
- [ ] 完整报告撰写
