# TLB 实验对齐指南

## 1. 本次对话的教训

### 1.1 需求理解错误

**用户意图（来自 cursor_new_method_and_experimental_plan.md）**:
- 线A：层间核范数约束 `c_inter = ‖x‖_*/‖out‖_*`，直接约束注意力输出
- 线B：层内缩放 `c_l = ‖V‖_*/‖K⊙V‖_*`，实际替换 value 路径为 `c_l · (K⊙V)`
- 消融：Only K⊙V、Q⊙V、Rand⊙V 等变体
- **关键要求**：测量实际指标（loss, PPL, BLEU），不只是计算范数比例

**我的错误**:
- 默认使用现有 `qwen_forward_benchmark.py`（诊断探针，只测不改）
- 未第一时间检查示例代码中的 `Attn_KV_InnerCl` 实现
- 混淆了"测量范数"与"替换 attention 路径"两个完全不同的任务

### 1.2 关键区分

| 类型 | 诊断探针 (Probe) | 实验实现 (Experiment) |
|------|------------------|----------------------|
| 目的 | 测量现有模型的范数分布 | 验证新机制的效果 |
| 做法 | Hook 捕获 Q/K/V，计算范数 | 替换 `eager_attention_forward` |
| 输出 | ratio、nuclear norm 统计 | PPL、BLEU、loss |
| 是否改模型行为 | ❌ 否 | ✅ 是 |

**下次必须问清的问题**:
> "你要的是 A) 测量并报告现有模型的范数比例，还是 B) 实际替换 attention value 路径并测量翻译指标？"

## 2. 当前实现状态

### 2.1 已实现（猴子补丁方案）

**线A (`nuc_inter`)**: 在 Qwen3 `eager_attention_forward` 后应用 `c_inter` 缩放
- 位置：输出端（整个 attention block 后）
- 与 draft 差异：draft 是在 attention 内部构造，当前是外部 wrapper
- 状态：⚠️ 部分对齐

**线B (`kv_inner`)**: 替换 `value_states = c_l · (key_states * value_states)`
- 位置：`repeat_kv` 之后，softmax 之前
- 与 draft 对齐：✅ 是
- 与 `tlb_attention_core` 一致：✅ 是

### 2.2 实现方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **猴子补丁** (当前) | 快速、不改源码、可逆 | 脆弱、依赖 HF 内部 API、难审计 | 快速验证假设 |
| **继承重写** | 显式、可审计、可移植 | 需要理解 HF 结构、代码量大 | 论文/长期维护 |
| **Hook-based** | 比 monkey-patch 干净 | 需要找准确的 tensor 替换时机 | 中间路线 |
| **CustomMultiheadAttention** | 完全可控、可移植 | 不能直接用预训练权重 | 从头训练/LoRA |

## 3. 实验设计检查清单

### 3.1 启动前必须确认

- [ ] **实验类型**：诊断探针 / 实际替换 / 训练
- [ ] **指标需求**：是否需要 PPL/BLEU / 仅范数统计
- [ ] **样本量**：128 / 500 / 全量
- [ ] **统计显著性**：单次运行 / 多 seed 平均
- [ ] **对照组**：基线 vanilla 是否已跑

### 3.2 代码实现检查

- [ ] 是否真正替换了 attention 路径（不只是 hook 测量）
- [ ] 公式实现是否与 draft 一致（`c_l = ‖V‖_*/‖K⊙V‖_*`）
- [ ] 是否在模型 load 之前打补丁（否则补丁不生效）
- [ ] 输出 JSON 是否包含 `eager_attention_forward` 字段标记当前模式

### 3.3 运行时检查

- [ ] 核几何阶段是否有进度输出（`[nuclear] layer X/28`）
- [ ] PPL 阶段是否有 batch 进度（`[qwen_opus] PPL batch k/N`）
- [ ] BLEU 阶段是否有逐条进度（`[qwen_opus] BLEU decode k/N`）
- [ ] 是否用了正确的 Python 解释器（`python3` 而非 conda `python`）

## 4. 后续实验建议

### 4.1 当前三条线（基线/A/B）

- 等结果出来后，对比 PPL/BLEU 趋势
- 若趋势符合预期（线B < 线A < 基线 或类似），可继续消融
- 若趋势异常，检查补丁是否真正生效（看日志中的 `[Qwen3 patch]` 行）

### 4.2 消融实验（下一步）

| 消融组 | 实现 | 预期作用 |
|--------|------|---------|
| Only K⊙V (no scaling) | `kv_mul` | 验证 scaling 的必要性 |
| Q⊙V × c_l | `qv_inner` | 验证 K 的特异性 |
| Rand⊙V × c_l | `randv_inner` | 验证 K 的语义信息必要性 |
| 层间 c (线A混合) | `kv_inner_nuc_inter` | 层内+层间双重约束 |

### 4.3 更科学的实现（长期）

1. **方案1：继承重写**
   - 创建 `Qwen3TLB(Qwen3ForCausalLM)`
   - 显式替换 attention 实现
   - 优点：可审计、可移植、不依赖 monkey-patch

2. **方案2：CustomMultiheadAttention + LoRA**
   - 用仓库已有的 `tlb_attention_core`
   - 在 MT 或小型模型上训练 adapter
   - 优点：完全可控，不依赖 Qwen3 内部结构

## 5. 沟通协议

### 5.1 用户指令解析

当用户说"跑实验"时，立即确认：
1. **实验类型**：探针 / 替换 / 训练
2. **指标**：PPL/BLEU / 范数 / 两者都要
3. **样本量**：快速趋势验证 / 正式统计
4. **对照**：是否需要基线对比

### 5.2 代码审查清单

- 查看用户的示例代码（如果有），理解意图
- 对比现有实现与用户意图的差异
- 明确说明实现与意图的 gap，让用户确认

### 5.3 进度报告

- 实验启动时：报告 PID、预期耗时、查看方式
- 实验进行中：定期报告进度（tail -f 日志位置）
- 实验完成：汇总 JSON 结果，制作对比表格

---

## 7. Colab：与本对话计划对齐的前向对比（GPU）

| 文件 | 内容 |
|------|------|
| `colab/colab_qwen3_forward_plan.ipynb` | **一站自包含**：内联 `CONFIG` JSON、核范数（多头 `(T,d_h)`，无 flatten 大矩阵）、线 A/B 补丁、HF OPUS-100 de-en 或本地 jsonl、多 seed、PPL、BLEU(13a/char)、chrF++、最后一层 attention 熵、每层核范数比与奇异值摘要、可视化、zip 导出。 |
| `colab/colab_qwen3_train_plan.ipynb` | **自包含 LoRA 训练**：与 forward 同套 eager 线 A/B 补丁、`peft`、HF OPUS train/val、指标 CSV、可选训练末 BLEU、adapter 保存、汇总图与 zip。 |
| `colab/colab_qwen3_tlb_train_lora.ipynb` | 可训练多 seed 循环（需上传/clone 仓库调用 `qwen3_lora_tlb_train.py`，`vanilla`/`tlb`）。 |

**与本地一致**：默认基座 `Qwen/Qwen3-1.7B`（与本地 Qwen3-1.7B 同族）；可将 `CONFIG["model_id"]` 改为 Drive 路径。

**线 A**：若 HF 未向 `eager_attention_forward` 传入 `hidden_states`，线 A 可能与 vanilla 相同——Notebook 文首已说明，需对照 PPL 或升级 transformers。

---

## 附录：关键文件索引

| 文件 | 作用 | 注意事项 |
|------|------|---------|
| `qwen3_attn_tlb_patch.py` | 猴子补丁实现 | 需在模型 load 前调用 install |
| `qwen_forward_benchmark.py` | 主入口 | 支持 `--line-a`、`--line-b` |
| `tlb_attention_core.py` | 标准 attention 实现 | `CustomMultiheadAttention` 有更完整实现 |
| `qwen_opus_forward_eval.py` | OPUS 前向评估 | PPL + BLEU |
| `run_serial_phased.py` | 本地三线串行（先基线 → 暂停读统计 → line_a → line_b） | 默认 64 条；`--no-pause` 连跑 |
| `mt_train_tlb.py` | MT 训练 | 需要训练时才用 |
| `colab/colab_qwen3_forward_plan.ipynb` | Colab GPU 前向全流程 | 见 §7 |
| `colab/colab_qwen3_train_plan.ipynb` | Colab LoRA + 线 A/B | 见 §7、`colab/COMPUTE_AND_PARAMS.md` |

---

**最后更新**: 2026-04-21
**下次实验前必读**: 是
