#!/usr/bin/env python3
"""
独立式级联探针 - 不修改 marian_attn_tlb_patch.py。
- ``transformers`` 有 ``eager_attention_forward`` 时：全局 eager 调度到 ``_cascade_probe_forward``。
- 否则：monkey-patch ``MarianAttention.forward``，在 ``(B,H,S,D)`` 上注入级联并读 ``_cascade_probe``。

训练：按步记录各层 V 阶段指标；loss 与各探针标量 的 **Spearman / Pearson / Kendall τ / MIC**
（见 ``train_assoc_loss_vs_metrics_by_layer``；旧版仅 Spearman 键仍保留）。
Eval：多 batch teacher-forcing 探针 + 同批 corpus BLEU；**逐批 BLEU** 与各标量 上四类关联系数
（``eval_assoc_bleu_vs_metrics_by_layer``）。相关表建议 **train/eval 各至少 4 step/batch**；
看 **loss/PPL 曲线** 需更大 ``--max-train-batches``（如 10–32+，800/8 约 100 steps 封顶）。

长序列可视化：``--plot-probes`` / ``--export-probe-csv`` 写出 ``probe_data/`` 与 ``probe_plots/``
（级联三阶段、分层多线、ρ 热力图等，见 ``probe_visualization`` + ``probe_mech_plots``）。

**Decoder 模式**（``--probe-target decoder``）：级联仅加在 Decoder **自注意**；可记录梯度 L2 范数、
稀疏 **Linear CKA**（FFN 前 / 子层后）、``generate`` 上 **自注意历史窗口** BLEU（4/16/64/full，见
``result.json`` 的 ``bleu_by_decoder_window``）。``experiments/run_schemeA_decoder_cascade_matrix.sh`` 为 18 组
Decoder 与 Encoder 同矩阵的批跑入口。
"""

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import MarianMTModel, MarianTokenizer
import sacrebleu

from mt_train_tlb import load_local_fixed_pairs
from tlb_value_smooth import (
    SpatialMode,
    TemporalMode,
    apply_value_cascade,
    decoder_safe_modes,
    encoder_legacy_modes,
)


def spearman_rho(x, y) -> float:
    """与 script6_scattering_criticality.spearman_rho 一致：优先 scipy，否则秩相关近似。"""
    try:
        from scipy.stats import spearmanr

        r, _ = spearmanr(x, y)
        return float(r) if math.isfinite(r) else float("nan")
    except ImportError:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean()
        ry -= ry.mean()
        sx, sy = rx.std(), ry.std()
        if sx < 1e-12 or sy < 1e-12:
            return float("nan")
        return float((rx * ry).mean() / (sx * sy))


# 与作图/TSV/热力图 对齐的关联系数名（除 Spearman 外需 scipy；MIC 需 minepy，缺失则为 NaN）
ASSOC_METHOD_KEYS: Tuple[str, ...] = (
    "spearman",
    "pearson",
    "kendall_tau",
    "mic",
)


def association_metrics(
    x: np.ndarray, y: np.ndarray, min_n: int = 4
) -> Dict[str, float]:
    """
    同一对序列上：Spearman、Pearson、Kendall :math:`\\tau`、MIC。
    样本不足或方差为 0 时各量为 NaN；MIC 在 minepy 不可用时为 NaN。
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    base = {k: float("nan") for k in ASSOC_METHOD_KEYS}
    if n < min_n or y.size != n:
        return base
    if float(np.nanstd(x)) < 1e-12 or float(np.nanstd(y)) < 1e-12:
        return base
    out = dict(base)
    s = spearman_rho(x, y)
    out["spearman"] = s if math.isfinite(s) else float("nan")
    try:
        from scipy.stats import kendalltau, pearsonr

        pr = pearsonr(x, y)
        out["pearson"] = float(
            pr.statistic if hasattr(pr, "statistic") else pr[0]
        )
        kt = kendalltau(x, y)
        out["kendall_tau"] = float(
            kt.statistic if hasattr(kt, "statistic") else kt[0]
        )
    except Exception:
        pass
    try:
        from minepy import MINE

        mine = MINE()
        mine.compute_score(x, y)
        m = float(mine.mic())
        out["mic"] = m if math.isfinite(m) else float("nan")
    except Exception:
        pass
    return out


def linear_cka_numpy(A: np.ndarray, B: np.ndarray) -> float:
    """同一样本行上 Linear CKA：A (n, p), B (n, q)。"""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] != B.shape[0] or A.shape[0] < 2:
        return float("nan")
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    ata = A.T @ A
    btb = B.T @ B
    atb = A.T @ B
    num = float(np.linalg.norm(atb, ord="fro") ** 2)
    den = float(
        (np.linalg.norm(ata, ord="fro") + 1e-12) * (np.linalg.norm(btb, ord="fro") + 1e-12)
    )
    return num / max(den, 1e-12)


def total_grad_l2_by_name_substr(model: torch.nn.Module, name_substr: Optional[str] = None) -> float:
    s = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if name_substr is not None and name_substr not in n:
            continue
        s += float(p.grad.data.float().norm(2).item() ** 2)
    return float(math.sqrt(s + 1e-12))


def set_decoder_historical_window_all(
    model: "MarianMTModel", w: Union[int, str, None]
) -> None:
    for layer in model.model.decoder.layers:
        if w is None or w == "full":
            if hasattr(layer.self_attn, "_decoder_historical_window"):
                delattr(layer.self_attn, "_decoder_historical_window")
        else:
            layer.self_attn._decoder_historical_window = w


class _DecoderCkaHooks:
    """在 Decoder 子层上挂 fc1 前、final_layer_norm 后，用稀疏 token 对算 Linear CKA。"""

    def __init__(self, model: Any, layer_indices: Tuple[int, ...]) -> None:
        self.handles: List[Any] = []
        self.pre: Dict[int, torch.Tensor] = {}
        self.post: Dict[int, torch.Tensor] = {}
        layers = model.model.decoder.layers
        self.valid_indices: Tuple[int, ...] = tuple(
            li for li in layer_indices if 0 <= li < len(layers)
        )
        for li in self.valid_indices:
            layer = layers[li]

            def pre_hook(mod: Any, ins: Any, i: int = li) -> None:
                if not ins or ins[0] is None:
                    return
                self.pre[i] = ins[0].detach()

            def post_hook(
                mod: Any, ins: Any, out: Any, i: int = li
            ) -> None:
                if out is not None:
                    self.post[i] = out.detach()

            self.handles.append(layer.fc1.register_forward_pre_hook(pre_hook))
            self.handles.append(
                layer.final_layer_norm.register_forward_hook(post_hook)
            )

    def clear(self) -> None:
        self.pre.clear()
        self.post.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.clear()

    def cka_on_last_forward(
        self, rng: np.random.Generator, max_tokens: int
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for li in self.valid_indices:
            if li not in self.pre or li not in self.post:
                continue
            a = self.pre[li].reshape(-1, self.pre[li].shape[-1])
            b = self.post[li].reshape(-1, self.post[li].shape[-1])
            n = int(min(a.shape[0], b.shape[0]))
            if n < 8:
                continue
            m = int(min(n, max_tokens))
            perm = rng.permutation(n)[:m]
            aa = a[perm].float().cpu().numpy()
            bb = b[perm].float().cpu().numpy()
            out[f"dec_{li}"] = linear_cka_numpy(aa, bb)
        return out


# 探针挂接：旧版 ``modeling_marian.eager_attention_forward``（module 级函数）；
# 新版 transformers 已移除，改为 **替换** ``MarianAttention.forward``，在 4D value 上注入级联。
# 需 ``attn_implementation='eager'``（Marian 仍走 `MarianAttention`）。

_EAGER_DISPATCH_BASE = None
_MARIAN_ATTN_FORWARD_BASE = None
_MARIAN_PROBE_MODE = "none"  # "eager_mod" | "class_forward" | "none"


def _decoder_historical_window_mask_bias(
    query: torch.Tensor,
    key: torch.Tensor,
    window: int,
    min_val: float = -1e4,
) -> torch.Tensor:
    """
    在已有 causal mask 上 **追加**：Decoder 自注意中 query 位 i 只能看到 key 位
    j ∈ [max(0, i - (W-1)), i]（与因果一致前提下的最近 W 个 token，含自身）。
    ``attention_mask`` 与 attention scores 同形广播相加。
    """
    T = query.shape[-2]
    S = key.shape[-2]
    w = int(window)
    if w <= 0 or w >= S:
        return torch.zeros(
            1, 1, T, S, device=query.device, dtype=query.dtype
        )
    i = torch.arange(T, device=query.device)[:, None]
    j = torch.arange(S, device=query.device)[None, :]
    j_min = (i - (w - 1)).clamp_min(0)
    block = (j < j_min).to(dtype=query.dtype) * min_val
    return block.unsqueeze(0).unsqueeze(0)


def _apply_decoder_window_to_mask(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """仅当 ``module._decoder_historical_window`` 为 int 时，向 ``attention_mask`` 叠加历史窗口偏置。"""
    w = getattr(module, "_decoder_historical_window", None)
    if w is None or w == "full":
        return attention_mask
    if not isinstance(w, (int, float)):
        return attention_mask
    wi = int(w)
    if wi <= 0 or wi > key.shape[-2]:
        return attention_mask
    b = _decoder_historical_window_mask_bias(query, key, wi)
    if attention_mask is None:
        return b
    return attention_mask + b.to(dtype=attention_mask.dtype, device=attention_mask.device)


def _install_marian_eager_probe_dispatch() -> None:
    global _EAGER_DISPATCH_BASE, _MARIAN_ATTN_FORWARD_BASE, _MARIAN_PROBE_MODE
    if _MARIAN_PROBE_MODE != "none":
        return
    import transformers.models.marian.modeling_marian as mm

    if hasattr(mm, "eager_attention_forward"):
        _EAGER_DISPATCH_BASE = mm.eager_attention_forward

        def _dispatch_eager(
            module,
            query,
            key,
            value,
            attention_mask=None,
            *,
            dropout=0.0,
            scaling=None,
            **kwargs,
        ):
            attention_mask = _apply_decoder_window_to_mask(
                module, query, key, attention_mask
            )
            custom = getattr(module, "_cascade_probe_forward", None)
            if custom is not None:
                return custom(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    dropout=dropout,
                    scaling=scaling,
                    **kwargs,
                )
            return _EAGER_DISPATCH_BASE(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                **kwargs,
            )

        mm.eager_attention_forward = _dispatch_eager
        _MARIAN_PROBE_MODE = "eager_mod"
    else:
        # transformers ≥~4.40：Marian 无 module 级 eager，直接 patch 类
        _MARIAN_ATTN_FORWARD_BASE = mm.MarianAttention.forward
        mm.MarianAttention.forward = _marian_attention_forward_inject_value_cascade
        _MARIAN_PROBE_MODE = "class_forward"


def _uninstall_marian_eager_probe_dispatch() -> None:
    global _EAGER_DISPATCH_BASE, _MARIAN_ATTN_FORWARD_BASE, _MARIAN_PROBE_MODE
    if _MARIAN_PROBE_MODE == "eager_mod" and _EAGER_DISPATCH_BASE is not None:
        import transformers.models.marian.modeling_marian as mm

        mm.eager_attention_forward = _EAGER_DISPATCH_BASE
        _EAGER_DISPATCH_BASE = None
    elif _MARIAN_PROBE_MODE == "class_forward" and _MARIAN_ATTN_FORWARD_BASE is not None:
        import transformers.models.marian.modeling_marian as mm

        mm.MarianAttention.forward = _MARIAN_ATTN_FORWARD_BASE
        _MARIAN_ATTN_FORWARD_BASE = None
    _MARIAN_PROBE_MODE = "none"


def _marian_attention_forward_inject_value_cascade(
    self: Any,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> tuple:
    """
    与当前 ``transformers`` 的 ``MarianAttention.forward`` 一致，仅在
    自注意力（非 cross）、无 past 时，在 ``value`` 的 4D 上注入级联。
    新版 ``transformers`` 无 ``eager_attention_forward`` 时，encoder/decoder 均走此路径。
    """
    probe = getattr(self, "_cascade_probe", None)
    is_cross_attention = key_value_states is not None
    if (
        probe is not None
        and not is_cross_attention
        and past_key_value is None
    ):
        return _marian_self_attn_cascade(self, hidden_states, attention_mask, layer_head_mask, output_attentions)

    return _MARIAN_ATTN_FORWARD_BASE(
        self,
        hidden_states,
        key_value_states,
        past_key_value,
        attention_mask,
        layer_head_mask,
        output_attentions,
    )


def _marian_self_attn_cascade(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    layer_head_mask: Optional[torch.Tensor],
    output_attentions: bool,
) -> tuple:
    """Encoder/decoder 自注意：对 ``value`` 4D 注入级联，其余与 HF ``MarianAttention.forward`` 相同。"""
    bsz, tgt_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states) * self.scaling
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    past_key_value_out = None
    if getattr(self, "is_decoder", False):
        past_key_value_out = (key_states, value_states)
    value_states = self._cascade_probe.apply_mixed_4d(value_states, self)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.reshape(*proj_shape)
    value_states = value_states.reshape(*proj_shape)
    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
        )
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    if output_attentions:
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None
    attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.bmm(attn_probs, value_states)
    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
        )
    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)
    return attn_output, attn_weights_reshaped, past_key_value_out


# ============ Token级指标计算 ============

def compute_token_metrics(V: torch.Tensor) -> Dict:
    """计算Token级指标"""
    stats = {}
    
    # 基础统计
    stats['mean'] = V.mean().item()
    stats['std'] = V.std().item()
    stats['l2_norm'] = torch.norm(V, p=2).item()
    
    # 核范数 (快速版) — MPS 无 SVD，需转 CPU
    if V.dim() == 4:
        b, h, s, d = V.shape
        matrix = V.reshape(b * h, s * d)
    else:
        matrix = V.reshape(-1, V.shape[-1])
    if matrix.device.type == "mps":
        matrix = matrix.detach().float().cpu()
    stats['nuclear'] = float(torch.norm(matrix, p='nuc'))
    
    # 时序方差 (沿seq_len dim=-2)
    var_seq = V.var(dim=-2, unbiased=False)
    stats['temporal_var_mean'] = var_seq.mean().item()
    stats['temporal_var_std'] = var_seq.std().item()
    
    # 特征方差 (沿head_dim dim=-1)
    var_feat = V.var(dim=-1, unbiased=False)
    stats['spatial_var_mean'] = var_feat.mean().item()
    stats['spatial_var_std'] = var_feat.std().item()
    
    # 时序差分 ΔV_t = V_{t+1} - V_t
    if V.shape[-2] > 1:
        delta = V[..., 1:, :] - V[..., :-1, :]
        stats['temporal_diff_mean'] = delta.abs().mean().item()
        stats['temporal_diff_std'] = delta.abs().std().item()
    else:
        stats['temporal_diff_mean'] = 0.0
        stats['temporal_diff_std'] = 0.0

    # 相邻时间步归一化向量的余弦相似度（沿 seq 的局部平滑度）
    if V.shape[-2] > 1:
        v0 = F.normalize(V[..., :-1, :], dim=-1, eps=1e-6)
        v1 = F.normalize(V[..., 1:, :], dim=-1, eps=1e-6)
        cos_t = (v0 * v1).sum(dim=-1)
        stats['temporal_cosine_mean'] = cos_t.mean().item()
    else:
        stats['temporal_cosine_mean'] = 1.0

    return stats


# ============ 带探针的级联Forward ============

class CascadeWithProbe:
    """包装类：创建带探针的级联forward函数"""
    
    def __init__(
        self,
        alpha_spatial: float,
        alpha_temporal: float,
        layer_name: str,
        is_reverse: bool = False,
        temporal_mode: TemporalMode = "circular_roll",
        spatial_mode: SpatialMode = "spatial_global",
    ):
        self.alpha_s = alpha_spatial
        self.alpha_t = alpha_temporal
        self.layer_name = layer_name
        self.is_reverse = is_reverse
        self.temporal_mode = temporal_mode
        self.spatial_mode = spatial_mode
        self.measurements = []
        self.eval_measurements = []
        self.step = 0
        self.eval_step = 0
        self.current_loss = None
        self.record_eval = False  # 在 eval 单 batch 诊断时置 True
        self.force_apply = False  # when True, apply cascade even in eval mode

    def apply_mixed_4d(self, value: torch.Tensor, module: Any) -> torch.Tensor:
        """
        在 ``(B,H,S,D)`` 的 V 上施级联，写 ``measurements``/``eval``，返回前向要用的 ``v_mix``。
        供新版 ``MarianAttention.forward`` 使用（`transformers` 无 ``eager_attention_forward`` 时）。
        """
        alpha_s, alpha_t = self.alpha_s, self.alpha_t
        order = "temporal_first" if self.is_reverse else "spatial_first"
        V_raw, V_first, V_second = apply_value_cascade(
            value,
            alpha_s,
            alpha_t,
            order=order,
            temporal_mode=self.temporal_mode,
            spatial_mode=self.spatial_mode,
        )
        V_raw = V_raw.detach()
        V_first = V_first.detach()
        V_second = V_second.detach()
        v_smooth = V_second
        log_train = module.training
        log_eval = (not module.training) and self.record_eval
        if log_train or log_eval:
            raw_stats = compute_token_metrics(V_raw)
            first_stats = compute_token_metrics(V_first)
            second_stats = compute_token_metrics(V_second)
            record = {
                "layer": self.layer_name,
                "order": "spatial_first" if not self.is_reverse else "temporal_first",
                "alpha_spatial": alpha_s,
                "alpha_temporal": alpha_t,
                "temporal_mode": self.temporal_mode,
                "spatial_mode": self.spatial_mode,
                "V_raw": raw_stats,
                "V_first": first_stats,
                "V_second": second_stats,
                "deltas": {
                    "first_to_raw": {
                        "nuclear_ratio": first_stats["nuclear"] / (raw_stats["nuclear"] + 1e-8),
                        "temporal_var_ratio": first_stats["temporal_var_mean"] / (raw_stats["temporal_var_mean"] + 1e-8),
                        "spatial_var_ratio": first_stats["spatial_var_mean"] / (raw_stats["spatial_var_mean"] + 1e-8),
                    },
                    "second_to_first": {
                        "nuclear_ratio": second_stats["nuclear"] / (first_stats["nuclear"] + 1e-8),
                        "temporal_var_ratio": second_stats["temporal_var_mean"] / (first_stats["temporal_var_mean"] + 1e-8),
                        "spatial_var_ratio": second_stats["spatial_var_mean"] / (first_stats["spatial_var_mean"] + 1e-8),
                    },
                },
            }
            if log_train:
                self.step += 1
                record["step"] = self.step
                record["loss"] = self.current_loss
                record["phase"] = "train"
                self.measurements.append(record)
            else:
                self.eval_step += 1
                record["eval_step"] = self.eval_step
                record["loss"] = None
                record["phase"] = "eval"
                self.eval_measurements.append(record)
        return v_smooth if (module.training or self.force_apply) else value

    def make_forward(self):
        """创建可替换的forward函数（旧版 ``eager_attention_forward`` 路径）。"""
        probe = self  # 在closure中引用

        def cascade_forward(module, query, key, value, attention_mask, *,
                           dropout=0.0, scaling=None, **kwargs):
            if scaling is None:
                scaling = float(query.size(-1) ** -0.5)
            v_mix = probe.apply_mixed_4d(value, module)
            # 标准 attention（与 ``apply_mixed_4d`` 中级联、记录逻辑一致）
            attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
            attn_output = torch.matmul(attn_weights, v_mix)
            attn_output = attn_output.transpose(1, 2).contiguous()
            
            return attn_output, attn_weights
        
        return cascade_forward
    
    def get_summary(self, rows: Optional[List[Dict]] = None) -> Dict:
        """汇总测量结果；rows 默认 self.measurements，可传 eval_measurements。"""
        data = self.measurements if rows is None else rows
        if not data:
            return {}

        summary = {
            'layer_name': self.layer_name,
            'total_records': len(data),
            'alpha_config': {'spatial': self.alpha_s, 'temporal': self.alpha_t},
            'order': 'spatial_first' if not self.is_reverse else 'temporal_first',
        }

        for stage in ['V_raw', 'V_first', 'V_second']:
            all_metrics = defaultdict(list)
            for m in data:
                if stage in m:
                    for k, v in m[stage].items():
                        all_metrics[k].append(v)

            summary[stage] = {
                k: {'mean': np.mean(v), 'std': np.std(v), 'min': np.min(v), 'max': np.max(v)}
                for k, v in all_metrics.items()
            }

        delta_metrics = defaultdict(list)
        for m in data:
            if 'deltas' in m:
                for transition, metrics in m['deltas'].items():
                    for k, v in metrics.items():
                        delta_metrics[f'{transition}_{k}'].append(v)

        summary['deltas'] = {
            k: {'mean': np.mean(v), 'std': np.std(v)}
            for k, v in delta_metrics.items()
        }

        return summary


def loss_assoc_metrics_for_probe(
    probe: "CascadeWithProbe", min_n: int = 4
) -> Dict[str, Dict[str, float]]:
    """
    训练步序列上，loss 与各探针标量 的关联系数字典（每键含 spearman, pearson, kendall_tau, mic）。
    解释：观测共变，不蕴含因果。``min_n`` 与作图/热力图最小步数 一致（默认 4）。
    """
    mlist = probe.measurements
    n = len(mlist)
    if n < min_n:
        return {}
    try:
        loss = np.array([float(m["loss"]) for m in mlist], dtype=float)
    except (KeyError, TypeError):
        return {}
    out: Dict[str, Dict[str, float]] = {}

    def one(name: str, arr: np.ndarray) -> None:
        if arr.shape[0] != n:
            return
        if np.std(arr) < 1e-12 or np.std(loss) < 1e-12:
            return
        mets = association_metrics(loss, arr, min_n=min_n)
        if any(
            mets.get(k) is not None and math.isfinite(mets.get(k, float("nan")))
            for k in ASSOC_METHOD_KEYS
        ):
            out[name] = mets

    for stage in ("V_raw", "V_first", "V_second"):
        if stage not in mlist[0]:
            continue
        for k in mlist[0][stage].keys():
            arr = np.array([m[stage][k] for m in mlist], dtype=float)
            one(f"{stage}.{k}", arr)
    d0 = mlist[0].get("deltas") or {}
    for tr_name, dsub in d0.items():
        for mk in dsub:
            arr = np.array([m["deltas"][tr_name][mk] for m in mlist], dtype=float)
            one(f"deltas.{tr_name}.{mk}", arr)
    return out


def loss_assoc_vs_all_layers(
    probes: List["CascadeWithProbe"], min_n: int = 4
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """训练：各 encoder 层 × 各标量 名 → 关联系数字典。"""
    return {
        p.layer_name: loss_assoc_metrics_for_probe(p, min_n=min_n)
        for p in probes
        if len(getattr(p, "measurements", []) or []) >= min_n
    }


def _assoc_layer_dict_to_spearman_only(
    assoc: Optional[Dict[str, Dict[str, Dict[str, float]]]],
) -> Optional[Dict[str, Dict[str, float]]]:
    if not assoc:
        return None
    flat: Dict[str, Dict[str, float]] = {}
    for layer, d0 in assoc.items():
        d_out: Dict[str, float] = {}
        for mname, mets in (d0 or {}).items():
            s = mets.get("spearman") if mets else None
            if s is not None and np.isfinite(s):
                d_out[mname] = float(s)
        if d_out:
            flat[layer] = d_out
    return flat if flat else None


def batch_corpus_bleu(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    batch: Dict,
    device: Union[str, torch.device],
) -> float:
    """当前 batch 上 generate + corpus BLEU（与 eval_bleu 单 batch 一致）。"""
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=4,
    )
    hyps, refs = [], []
    for j in range(len(generated)):
        hyps.append(tokenizer.decode(generated[j], skip_special_tokens=True))
        refs.append(tokenizer.decode(batch["labels"][j], skip_special_tokens=True))
    if not hyps:
        return 0.0
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)


@torch.no_grad()
def run_multi_batch_eval_probe(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    val_loader: torch.utils.data.DataLoader,
    probes: List["CascadeWithProbe"],
    device: Union[str, torch.device],
    max_eval_batches: int,
) -> List[Dict[str, Any]]:
    """
    多 batch：teacher-forcing 前向收探针 + 同 batch 上 beam generate 得逐批 BLEU。
    每个 batch 后各层 probe.eval_measurements 仅保留本步一条，深拷贝到 rows。
    """
    rows: List[Dict[str, Any]] = []
    for bi, batch in enumerate(val_loader):
        if bi >= max_eval_batches:
            break
        for p in probes:
            p.eval_measurements.clear()
            p.eval_step = 0
            p.record_eval = True
        model.eval()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        for p in probes:
            p.record_eval = False
        layer_snap: Dict[str, Any] = {}
        for p in probes:
            if not p.eval_measurements:
                continue
            layer_snap[p.layer_name] = deepcopy(p.eval_measurements[-1])
        bleu_b = batch_corpus_bleu(model, tokenizer, batch, device)
        rows.append({"batch_idx": int(bi), "bleu": float(bleu_b), "layers": layer_snap})
    return rows


def bleu_assoc_vs_layer_metrics(
    batch_rows: List[Dict[str, Any]], min_k: int = 4
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    逐批 corpus BLEU 与各层、各标量 的关联系数字典（每层：标量名 → 四指标）。
    需 ``len(batch_rows) >= min_k`` 且 BLEU 跨 batch 非常数；``min_k`` 与 Spearman/热力图 一致，默认 4。
    """
    K = len(batch_rows)
    if K < min_k:
        return {}
    bleu = np.array([r["bleu"] for r in batch_rows], dtype=float)
    if float(np.std(bleu)) < 1e-9:
        return {}
    if not batch_rows[0].get("layers"):
        return {}

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    layer_names = sorted(batch_rows[0]["layers"].keys())

    for layer in layer_names:
        out[layer] = {}
        m0 = batch_rows[0]["layers"][layer]
        for stage in ("V_raw", "V_first", "V_second"):
            if stage not in m0:
                continue
            for k in m0[stage]:
                try:
                    arr = np.array(
                        [r["layers"][layer][stage][k] for r in batch_rows], dtype=float
                    )
                except (KeyError, TypeError):
                    continue
                if float(np.std(arr)) < 1e-12:
                    continue
                mets = association_metrics(bleu, arr, min_n=min_k)
                if any(
                    mets.get(k) is not None
                    and math.isfinite(mets.get(k, float("nan")))
                    for k in ASSOC_METHOD_KEYS
                ):
                    out[layer][f"{stage}.{k}"] = mets
        d0 = m0.get("deltas") or {}
        for tr_name, dsub in d0.items():
            for mk in dsub:
                try:
                    arr = np.array(
                        [r["layers"][layer]["deltas"][tr_name][mk] for r in batch_rows],
                        dtype=float,
                    )
                except (KeyError, TypeError):
                    continue
                if float(np.std(arr)) < 1e-12:
                    continue
                mets = association_metrics(bleu, arr, min_n=min_k)
                if any(
                    mets.get(k) is not None
                    and math.isfinite(mets.get(k, float("nan")))
                    for k in ASSOC_METHOD_KEYS
                ):
                    out[layer][f"deltas.{tr_name}.{mk}"] = mets
    return {k: v for k, v in out.items() if v}


def eval_rows_to_layer_summaries(
    batch_rows: List[Dict[str, Any]], probes: List["CascadeWithProbe"]
) -> Dict[str, Dict]:
    """将多 batch 的 layer 记录串成每层的列表，复用 get_summary 聚合。"""
    if not batch_rows or not probes:
        return {}
    by_name = {p.layer_name: p for p in probes}
    lists: Dict[str, List[Dict]] = {p.layer_name: [] for p in probes}
    for r in batch_rows:
        for name, m in r.get("layers", {}).items():
            if name in lists:
                lists[name].append(m)
    out: Dict[str, Dict] = {}
    for name, lst in lists.items():
        if not lst or name not in by_name:
            continue
        out[name] = by_name[name].get_summary(rows=lst)
    return out


def eval_metrics_reasonableness(
    batch_rows: List[Dict[str, Any]],
    train_summaries: Dict[str, Dict],
    shallow_layer: str = "enc_0",
) -> Dict[str, Any]:
    """
    粗检：数值域、跨 batch 是否退化、与 train 汇总是否同量级。
    非严格单元测试，仅作实验健康度提示。
    """
    checks: List[Dict[str, Any]] = []
    ok = True

    if not batch_rows:
        return {"ok": False, "checks": [{"name": "eval_batches", "pass": False, "note": "no rows"}]}

    nbatch = len(batch_rows)
    if nbatch < 4:
        checks.append(
            {
                "name": "enough_batches_for_spearman",
                "pass": False,
                "note": f"need >=4 eval batches for BLEU–metric Spearman, got {nbatch}",
            }
        )
        ok = False

    bleus = [r["bleu"] for r in batch_rows]
    checks.append(
        {
            "name": "bleu_range",
            "pass": all(0.0 <= b <= 100.0 for b in bleus),
            "note": f"min={min(bleus):.2f} max={max(bleus):.2f}",
        }
    )
    checks.append(
        {
            "name": "bleu_variation",
            "pass": float(np.std(bleus)) > 1e-6,
            "note": f"std={float(np.std(bleus)):.4f} (need >0 for cross-batch Spearman)",
        }
    )
    if not checks[-1]["pass"]:
        ok = False

    # 抽样最浅层 的 nuclear、cosine
    s0 = batch_rows[0].get("layers", {}).get(shallow_layer)
    if s0 and "V_raw" in s0:
        nuc0 = s0["V_raw"].get("nuclear", 0)
        cos0 = s0["V_raw"].get("temporal_cosine_mean", 0)
        checks.append(
            {
                "name": f"{shallow_layer}_Vraw_nuclear_positive",
                "pass": nuc0 > 0 and math.isfinite(nuc0),
                "note": f"nuclear={nuc0:.2f}",
            }
        )
        checks.append(
            {
                "name": f"{shallow_layer}_temporal_cosine_in_-1_1",
                "pass": -1.0 - 1e-3 <= cos0 <= 1.0 + 1e-3,
                "note": f"cos={cos0:.4f}",
            }
        )
        if not checks[-2]["pass"] or not checks[-1]["pass"]:
            ok = False

    # train vs eval 最浅层 核范数数量级
    t_mean = (train_summaries.get(shallow_layer) or {}).get("V_raw", {}).get("nuclear", {}).get("mean")
    e_list = [
        r["layers"][shallow_layer]["V_raw"]["nuclear"]
        for r in batch_rows
        if shallow_layer in r.get("layers", {}) and "V_raw" in r["layers"].get(shallow_layer, {})
    ]
    if t_mean is not None and e_list:
        e_mean = float(np.mean(e_list))
        ratio = e_mean / (float(t_mean) + 1e-8)
        sane = 0.1 <= ratio <= 10.0
        checks.append(
            {
                "name": f"{shallow_layer}_nuclear_train_eval_magnitude",
                "pass": sane,
                "note": f"train_mean_nuclear={t_mean:.1f} eval_batch_mean_nuclear≈{e_mean:.1f} ratio={ratio:.2f} (expect O(1))",
            }
        )
        if not sane:
            ok = False

    return {"ok": ok, "checks": checks}


def compute_eval_ce_quantiles(
    model: Any,
    val_loader: Any,
    device: str,
    max_batches: int = 8,
) -> Dict[str, float]:
    """Teacher-forcing token CE distribution on validation (labeled positions only)."""
    model.eval()
    ce_vals: List[float] = []
    n_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
            n_batches += 1
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            per_tok = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            )
            mask = shift_labels.view(-1) != -100
            if mask.any():
                ce_vals.extend(per_tok[mask].float().cpu().tolist())

    if not ce_vals:
        return {
            "eval_ce_mean": float("nan"),
            "eval_ce_p50": float("nan"),
            "eval_ce_p90": float("nan"),
            "eval_ce_p99": float("nan"),
            "eval_ce_max": float("nan"),
            "eval_ppl": float("nan"),
        }

    arr = np.asarray(ce_vals, dtype=float)
    mean_ce = float(np.mean(arr))
    return {
        "eval_ce_mean": mean_ce,
        "eval_ce_p50": float(np.percentile(arr, 50)),
        "eval_ce_p90": float(np.percentile(arr, 90)),
        "eval_ce_p99": float(np.percentile(arr, 99)),
        "eval_ce_max": float(np.max(arr)),
        "eval_ppl": float(np.exp(mean_ce)),
        "eval_ce_n_tokens": float(arr.size),
        "eval_ce_batches": float(n_batches),
    }


def _resolve_smooth_modes(
    probe_target: str,
    temporal_mode: str,
    spatial_mode: str,
) -> Tuple[TemporalMode, SpatialMode]:
    if temporal_mode == "auto":
        tm = decoder_safe_modes()[0] if probe_target == "decoder" else encoder_legacy_modes()[0]
    elif temporal_mode in ("circular_roll", "causal_prev"):
        tm = temporal_mode  # type: ignore[assignment]
    else:
        raise ValueError(f"unknown temporal_mode: {temporal_mode!r}")

    if spatial_mode == "auto":
        sm = decoder_safe_modes()[1] if probe_target == "decoder" else encoder_legacy_modes()[1]
    elif spatial_mode in ("spatial_global", "prefix_mean"):
        sm = spatial_mode  # type: ignore[assignment]
    else:
        raise ValueError(f"unknown spatial_mode: {spatial_mode!r}")
    return tm, sm


# ============ 数据加载 ============

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tokenizer, max_length=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        de, en = self.pairs[idx]
        model_inputs = self.tokenizer(
            de, max_length=self.max_length, truncation=True, 
            return_tensors="pt", padding="max_length"
        )
        labels = self.tokenizer(
            en, max_length=self.max_length, truncation=True,
            return_tensors="pt", padding="max_length"
        )
        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }


def create_dataloader(pairs, tokenizer, batch_size=8, shuffle=True):
    dataset = SimpleDataset(pairs, tokenizer)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============ 评估 ============

@torch.no_grad()
def eval_bleu(model, tokenizer, dataloader, device, max_samples=64):
    model.eval()
    hyps, refs = [], []
    
    for i, batch in enumerate(dataloader):
        if i * batch["input_ids"].size(0) >= max_samples:
            break
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=4,
        )
        
        for j in range(len(generated)):
            hyp = tokenizer.decode(generated[j], skip_special_tokens=True)
            ref = tokenizer.decode(batch["labels"][j], skip_special_tokens=True)
            hyps.append(hyp)
            refs.append(ref)
    
    if not hyps:
        return 0.0, 0
    
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score, len(hyps)


# ============ 主流程 ============

def _resolve_experiment_device(explicit: Optional[str] = None) -> str:
    """
    选设备。环境变量：``TLB_DEVICE`` / ``TLB_EXPERIMENT_DEVICE`` = cpu|mps|cuda；
    ``TLB_FORCE_CPU=1`` 强制 CPU（x86/Rosetta conda + MPS 上常见 **SIGSEGV 11** 时必开）。
    """
    import os

    ex = (explicit or "").strip().lower()
    if ex in ("auto", ""):
        ex = ""
    if ex in ("cpu", "mps", "cuda"):
        if ex == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS 不可用，改用 cpu", flush=True)
            return "cpu"
        if ex == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA 不可用，改用 cpu", flush=True)
            return "cpu"
        return ex
    v = (os.environ.get("TLB_DEVICE") or os.environ.get("TLB_EXPERIMENT_DEVICE") or "").strip().lower()
    if v in ("cpu", "mps", "cuda"):
        return _resolve_experiment_device(v)
    if os.environ.get("TLB_FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "y"):
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _compute_label_smoothing_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        label_smoothing=float(label_smoothing),
    )


def _set_marian_attention_dropout(model: MarianMTModel, dropout: float) -> None:
    p = float(dropout)
    for layer in list(model.model.encoder.layers) + list(model.model.decoder.layers):
        attn = layer.self_attn
        if hasattr(attn, "dropout"):
            attn.dropout = p


def run_probe_experiment(
    alpha_spatial: float,
    alpha_temporal: float,
    is_reverse: bool,
    seed: int,
    output_dir: str = "results_probe_independent",
    max_eval_batches: int = 4,
    max_train_batches: int = 32,
    export_probe_csv: bool = False,
    plot_probes: bool = False,
    save_eval_probe_json: bool = False,
    probe_plot_rolling: int = 0,
    train_plot_stride: int = 1,
    include_legacy_step_plots: bool = False,
    include_eval_batch_axis_plots: bool = False,
    device: str = "auto",
    probe_target: str = "encoder",
    record_gradient_norm: bool = True,
    record_linear_cka: bool = True,
    cka_layer_indices: Tuple[int, ...] = (2, 5),
    cka_interval: int = 4,
    cka_max_tokens: int = 2048,
    decoder_window_bleu: bool = True,
    temporal_mode: str = "auto",
    spatial_mode: str = "auto",
    exp_suffix: str = "",
    enable_cascade: bool = True,
    label_smoothing: float = 0.0,
    attention_dropout: Optional[float] = None,
) -> Dict:
    """
    运行独立式探针实验。

    ``probe_target="encoder"`` | ``"decoder"``：级联与探针只挂在该侧的 **自注意力**（Decoder 不碰 Cross-Attn）。

    **Decoder 增进**：可选训练步上 Decoder 子模块梯度 L2 范数、选定层的 **Linear CKA**（FFN 前 / 子层后），
    及 ``generate`` 上 **自注意历史窗口** BLEU（4 / 16 / 64，``full`` 为无额外面具）。
    """
    import os
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    if (device or "auto").strip().lower() in ("auto",):
        dev = _resolve_experiment_device()
    else:
        dev = _resolve_experiment_device(device)
    device = dev

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if enable_cascade:
        mode_name = 'cascade_rev' if is_reverse else 'cascade'
        pfx = f"{mode_name}_"
        if probe_target == "decoder":
            pfx = f"{mode_name}_dec_"
        suffix = (exp_suffix or "").strip()
        if suffix and not suffix.startswith("_"):
            suffix = f"_{suffix}"
        exp_name = f"{pfx}s{alpha_spatial}_t{alpha_temporal}_seed{seed}{suffix}"
    else:
        base = "vanilla" if probe_target == "encoder" else "vanilla_dec"
        suffix = (exp_suffix or "").strip()
        if suffix and not suffix.startswith("_"):
            suffix = f"_{suffix}"
        exp_name = f"{base}_seed{seed}{suffix}"
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    shallow = "dec_0" if probe_target == "decoder" else "enc_0"
    tm, sm = ("circular_roll", "spatial_global")
    if enable_cascade:
        tm, sm = _resolve_smooth_modes(probe_target, temporal_mode, spatial_mode)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name} (probe_target={probe_target})")
    print(f"Device: {device}")
    print(f"Smooth modes: temporal={tm}, spatial={sm}")
    print(f"Cascade enabled: {enable_cascade}, label_smoothing={label_smoothing}, attention_dropout={attention_dropout}")
    print(f"{'='*60}")
    
    # 加载数据
    train_pairs, val_pairs = load_local_fixed_pairs()
    train_pairs = train_pairs[:800]
    val_pairs = val_pairs[:128]
    
    # 加载模型（eager 才会走 ``eager_attention_forward``，探针才生效）
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    model = MarianMTModel.from_pretrained(
        "Helsinki-NLP/opus-mt-de-en",
        attn_implementation="eager",
    ).to(device)
    if attention_dropout is not None:
        _set_marian_attention_dropout(model, attention_dropout)
    
    train_loader = create_dataloader(train_pairs, tokenizer, batch_size=8, shuffle=True)
    val_loader = create_dataloader(val_pairs, tokenizer, batch_size=8, shuffle=False)
    
    _install_marian_eager_probe_dispatch()

    def _clear_probe_forwards() -> None:
        for group in (model.model.encoder.layers, model.model.decoder.layers):
            for layer in group:
                if not hasattr(layer, "self_attn"):
                    continue
                for attr in ("_cascade_probe_forward", "_cascade_probe"):
                    if hasattr(layer.self_attn, attr):
                        delattr(layer.self_attn, attr)

    cka_hooks: Optional[_DecoderCkaHooks] = None
    cka_by_step: List[Dict[str, Any]] = []
    rng_cka = np.random.default_rng(seed)
    if (
        probe_target == "decoder"
        and record_linear_cka
        and cka_layer_indices
    ):
        cka_hooks = _DecoderCkaHooks(model, cka_layer_indices)

    try:
        # 每层：旧版用 ``_cascade_probe_forward``；新版 patch ``MarianAttention.forward`` 用 ``_cascade_probe``。
        probes: List["CascadeWithProbe"] = []
        if enable_cascade:
            if probe_target == "encoder":
                for i, layer in enumerate(model.model.encoder.layers):
                    probe = CascadeWithProbe(
                        alpha_spatial=alpha_spatial,
                        alpha_temporal=alpha_temporal,
                        layer_name=f"enc_{i}",
                        is_reverse=is_reverse,
                        temporal_mode=tm,
                        spatial_mode=sm,
                    )
                    probes.append(probe)
                    layer.self_attn._cascade_probe = probe
                    layer.self_attn._cascade_probe_forward = probe.make_forward()
            else:
                for i, layer in enumerate(model.model.decoder.layers):
                    probe = CascadeWithProbe(
                        alpha_spatial=alpha_spatial,
                        alpha_temporal=alpha_temporal,
                        layer_name=f"dec_{i}",
                        is_reverse=is_reverse,
                        temporal_mode=tm,
                        spatial_mode=sm,
                    )
                    probes.append(probe)
                    layer.self_attn._cascade_probe = probe
                    layer.self_attn._cascade_probe_forward = probe.make_forward()

        # 训练
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        num_batches = 0
        max_batches = max_train_batches
        losses = []
        grad_log: List[Dict[str, float]] = []

        print(f"\nTraining with probe recording (max {max_batches} batches)...")
        for batch in train_loader:
            if num_batches >= max_batches:
                break
            if cka_hooks is not None:
                cka_hooks.clear()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            if label_smoothing > 0.0:
                loss = _compute_label_smoothing_loss(
                    outputs.logits, labels, label_smoothing=label_smoothing
                )
            else:
                loss = outputs.loss
            if (
                cka_hooks is not None
                and (num_batches + 1) % max(1, cka_interval) == 0
            ):
                row = cka_hooks.cka_on_last_forward(
                    rng_cka, cka_max_tokens
                )
                cka_by_step.append(
                    {"train_batch": num_batches + 1, "cka": row}
                )
            loss_item = float(loss.item())

            # 探针在 forward 里写 ``loss``，此时需用**本步** loss（上一步的 ``current_loss`` 在第一步为 None）
            for probe in probes:
                probe.current_loss = loss_item
                if probe.measurements:
                    probe.measurements[-1]["loss"] = loss_item

            losses.append(loss.item())

            loss.backward()
            if probe_target == "decoder" and record_gradient_norm:
                grad_log.append(
                    {
                        "batch": float(num_batches + 1),
                        "grad_total": total_grad_l2_by_name_substr(
                            model, None
                        ),
                        "grad_decoder": total_grad_l2_by_name_substr(
                            model, "model.decoder"
                        ),
                    }
                )
            optimizer.step()

            num_batches += 1
            if num_batches == 1:
                n0 = len(probes[0].measurements) if probes else 0
                print(
                    f"  [probe check] first batch: {shallow} records={n0} (expect >0 if eager+hook ok)"
                )
            if num_batches % 20 == 0:
                avg_loss = np.mean(losses[-20:])
                ppl = np.exp(avg_loss)
                print(f"  Batch {num_batches}/{max_batches}: loss={avg_loss:.4f}, ppl={ppl:.2f}")

        # 多 batch 验证集：逐批 teacher-forcing 探针 + 同批 BLEU，供与层指标做关联表
        print(
            f"\nMulti-batch eval probe (max {max_eval_batches} batches) + per-batch BLEU..."
        )
        eval_batch_rows = run_multi_batch_eval_probe(
            model, tokenizer, val_loader, probes, device, max_eval_batches
        )
        train_assoc_by_layer = loss_assoc_vs_all_layers(probes) if probes else {}
        train_spearman_by_layer = _assoc_layer_dict_to_spearman_only(train_assoc_by_layer) or {}
        train_spearman_enc0 = (
            train_spearman_by_layer.get(shallow, {}) if train_spearman_by_layer else {}
        )

        # 训练集探针汇总
        all_summaries = {}
        for probe in probes:
            all_summaries[probe.layer_name] = probe.get_summary()

        eval_assoc_by_layer = bleu_assoc_vs_layer_metrics(eval_batch_rows)
        spearman_bleu_by_layer = _assoc_layer_dict_to_spearman_only(
            eval_assoc_by_layer
        ) or {}
        eval_multi_summaries = eval_rows_to_layer_summaries(eval_batch_rows, probes)
        eval_sanity = (
            eval_metrics_reasonableness(
                eval_batch_rows, all_summaries, shallow_layer=shallow
            )
            if eval_batch_rows
            else {"ok": False, "checks": []}
        )

        # 评估
        print("\nEvaluating BLEU (full val sample cap, same as before)...")
        set_decoder_historical_window_all(model, None)
        bleu_score, num_eval = eval_bleu(model, tokenizer, val_loader, device, max_samples=64)
        print(f"BLEU: {bleu_score:.2f}")

        eval_ce_stats = compute_eval_ce_quantiles(
            model, val_loader, device, max_batches=max_eval_batches
        )
        print(
            "  Eval CE quantiles | "
            f"p50={eval_ce_stats.get('eval_ce_p50', float('nan')):.3f}, "
            f"p90={eval_ce_stats.get('eval_ce_p90', float('nan')):.3f}, "
            f"p99={eval_ce_stats.get('eval_ce_p99', float('nan')):.3f}, "
            f"max={eval_ce_stats.get('eval_ce_max', float('nan')):.3f}, "
            f"PPL={eval_ce_stats.get('eval_ppl', float('nan')):.2f}"
        )

        bleu_by_decoder_window: Dict[str, float] = {}
        if probe_target == "decoder" and decoder_window_bleu:
            for wv in (4, 16, 64):
                set_decoder_historical_window_all(model, wv)
                b_w, _n = eval_bleu(
                    model, tokenizer, val_loader, device, max_samples=64
                )
                bleu_by_decoder_window[str(wv)] = float(b_w)
            set_decoder_historical_window_all(model, None)
            bleu_by_decoder_window["full"] = float(bleu_score)
            order = ("4", "16", "64", "full")
            wparts = [f"{k}={bleu_by_decoder_window[k]:.2f}" for k in order if k in bleu_by_decoder_window]
            print("  BLEU (decoder self history window) | " + ", ".join(wparts))

        cka_mean_by_layer: Dict[str, float] = {}
        if cka_by_step:
            for name in cka_by_step[0].get("cka", {}):
                vals: List[float] = []
                for s in cka_by_step:
                    c = s.get("cka", {}).get(name)
                    if c is not None and math.isfinite(c):
                        vals.append(float(c))
                if vals:
                    cka_mean_by_layer[name] = float(np.mean(vals))
        grad_stats: Dict[str, Dict[str, float]] = {}
        if grad_log:
            for gk in ("grad_total", "grad_decoder"):
                vals2 = [float(x[gk]) for x in grad_log if gk in x]
                if vals2:
                    grad_stats[gk] = {
                        "mean": float(np.mean(vals2)),
                        "std": float(np.std(vals2)),
                    }

        # 保存
        result = {
            'exp_name': exp_name,
            'probe_target': probe_target,
            'first_probe_layer': shallow,
            'alpha_spatial': alpha_spatial,
            'alpha_temporal': alpha_temporal,
            'is_reverse': is_reverse,
            'enable_cascade': enable_cascade,
            'label_smoothing': float(label_smoothing),
            'attention_dropout': attention_dropout,
            'temporal_mode': tm,
            'spatial_mode': sm,
            'seed': seed,
            'max_eval_batches': max_eval_batches,
            'max_train_batches': max_train_batches,
            'train_plot_stride': train_plot_stride,
            'bleu': bleu_score,
            'num_eval_bleu_samples': num_eval,
            'train_loss': np.mean(losses[-20:]),
            'train_ppl': np.exp(np.mean(losses[-20:])),
            **eval_ce_stats,
            'probe_summaries': all_summaries,
            'eval_multi_batch_summaries': eval_multi_summaries,
            'eval_multi_batch_count': len(eval_batch_rows),
            'eval_per_batch_bleu': [r['bleu'] for r in eval_batch_rows],
            'train_assoc_loss_vs_metrics_by_layer': train_assoc_by_layer,
            'eval_assoc_bleu_vs_metrics_by_layer': eval_assoc_by_layer,
            'assoc_method_keys': list(ASSOC_METHOD_KEYS),
            'train_spearman_loss_vs_metrics_enc0': train_spearman_enc0,
            'train_spearman_loss_vs_first_probe_layer': train_spearman_enc0,
            'train_spearman_loss_vs_metrics_by_layer': train_spearman_by_layer,
            'eval_spearman_bleu_vs_metrics_by_layer': spearman_bleu_by_layer,
            'eval_metrics_reasonableness': eval_sanity,
            'cka_by_train_step': cka_by_step,
            'cka_mean_by_layer': cka_mean_by_layer,
            'gradient_norm_by_batch': grad_log,
            'gradient_norm_summary': grad_stats,
            'bleu_by_decoder_window': bleu_by_decoder_window
            if bleu_by_decoder_window
            else None,
        }

        with open(exp_dir / "result.json", 'w') as f:
            json.dump(result, f, indent=2, default=float)

        if export_probe_csv or plot_probes or save_eval_probe_json:
            from probe_visualization import export_and_plot

            probe_art = export_and_plot(
                probes,
                eval_batch_rows,
                exp_dir,
                plot_train=plot_probes,
                plot_eval=plot_probes,
                save_csv=export_probe_csv or plot_probes,
                save_eval_json=save_eval_probe_json,
                rolling_window=probe_plot_rolling,
                train_assoc_by_layer=train_assoc_by_layer,
                eval_assoc_by_layer=eval_assoc_by_layer,
                train_spearman_by_layer=train_spearman_by_layer,
                eval_spearman_by_layer=spearman_bleu_by_layer,
                include_legacy_step_plots=include_legacy_step_plots,
                include_eval_batch_axis_plots=include_eval_batch_axis_plots,
                train_plot_stride=train_plot_stride,
            )
            result["probe_artifacts"] = probe_art
            print(f"\nProbe artifacts: data -> {probe_art.get('data_dir')}, plots -> {probe_art.get('plot_dir')}")

        print(f"\nResults saved to: {exp_dir}")

        # 打印关键发现
        print("\n--- Probe Summary (train) ---")
        for layer_name, summary in sorted(all_summaries.items())[:3]:  # 只打印前3层
            if 'V_raw' in summary and 'nuclear' in summary['V_raw']:
                raw_nuc = summary['V_raw']['nuclear']['mean']
                print(f"{layer_name}: raw_nuclear={raw_nuc:.1f}")
                if 'V_raw' in summary and 'temporal_cosine_mean' in summary['V_raw']:
                    tc = summary['V_raw']['temporal_cosine_mean']['mean']
                    print(f"  temporal_cosine_mean (adjacent tokens): {tc:.4f}")
                if 'deltas' in summary and 'first_to_raw_nuclear_ratio' in summary['deltas']:
                    ratio = summary['deltas']['first_to_raw_nuclear_ratio']['mean']
                    print(f"  first stage nuclear ratio: {ratio:.3f}")

        if train_spearman_enc0:
            print(
                f"\n--- Spearman rho(loss, metric) | {shallow} | train steps (observational) ---"
            )
            top = sorted(train_spearman_enc0.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
            for name, r in top:
                print(f"  {name}: {r:+.3f}")

        if eval_multi_summaries.get(shallow):
            es = eval_multi_summaries[shallow]
            print(
                f"\n--- Eval multi-batch probe | {shallow} | batches={len(eval_batch_rows)} ---"
            )
            if es.get("V_raw", {}).get("nuclear"):
                print(
                    f"  V_raw nuclear mean: {es['V_raw']['nuclear']['mean']:.1f} "
                    f"(vs train {all_summaries[shallow]['V_raw']['nuclear']['mean']:.1f})"
                )
            if es.get("V_raw", {}).get("temporal_cosine_mean"):
                print(f"  V_raw temporal_cosine_mean: {es['V_raw']['temporal_cosine_mean']['mean']:.4f}")

        if cka_mean_by_layer:
            print("\n--- Linear CKA (FFN pre vs sublayer out) | decoder layers (train, sparse) ---")
            for k, v in sorted(cka_mean_by_layer.items()):
                print(f"  {k}: mean={v:.4f}")
        if grad_stats:
            print("\n--- Gradient L2 norms (train) ---")
            for k, st in grad_stats.items():
                print(
                    f"  {k}: mean={st['mean']:.4f} std={st['std']:.4f}"
                )

        print("\n--- Eval reasonableness ---")
        for c in eval_sanity.get("checks", []):
            mark = "OK" if c.get("pass") else "!!"
            print(f"  [{mark}] {c.get('name')}: {c.get('note', '')}")
        print(f"  overall_ok={eval_sanity.get('ok')}")

        if spearman_bleu_by_layer:
            print(
                "\n--- Spearman rho(batch BLEU, metric) by layer (top |ρ|; full 4: see result.json assoc) ---"
            )
            for layer in sorted(spearman_bleu_by_layer.keys())[:6]:
                items = spearman_bleu_by_layer[layer]
                top = sorted(
                    items.items(), key=lambda kv: abs(kv[1]), reverse=True
                )[:5]
                if top:
                    print(
                        f"  {layer}: "
                        + ", ".join(f"{k}={v:+.2f}" for k, v in top)
                    )

        return result
    finally:
        if cka_hooks is not None:
            cka_hooks.remove()
        _clear_probe_forwards()
        _uninstall_marian_eager_probe_dispatch()


def main():
    parser = argparse.ArgumentParser(description="Independent Cascade Probe")
    parser.add_argument("--alpha-spatial", type=float, default=0.2)
    parser.add_argument("--alpha-temporal", type=float, default=0.2)
    parser.add_argument("--reverse", action="store_true", help="Use temporal->spatial order")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results_probe_independent")
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=4,
        help="val 上多 batch 探针 + 逐批 BLEU 的上限；4 即满足相关表最低 batch 数；全 val(bs8) 用 16",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=32,
        help="训练循环最大 batch 数（约 8 条/ batch 时 32≈256 条）；关联系数需步数≥4；看平滑 loss/探针 曲线建议 16+，全 800 条约 100",
    )
    parser.add_argument(
        "--export-probe-csv",
        action="store_true",
        help="导出每层训练步探针 CSV 与 eval 批次 CSV（上千点便于外部分析）",
    )
    parser.add_argument(
        "--plot-probes",
        action="store_true",
        help="生成探针图：沿深度/三阶段、池化相关散点、每层的 nuc–L hexbin、ρ 热力图等（会同时写 CSV）",
    )
    parser.add_argument(
        "--probe-legacy-step-plots",
        action="store_true",
        help="额外画以训练 step 为横轴 的时间序列、loss–PPL 步进图",
    )
    parser.add_argument(
        "--probe-legacy-eval-batch-plots",
        action="store_true",
        help="额外画以 val batch 序号为横轴 的各层线、per-layer 三阶段、BLEU 折线等",
    )
    parser.add_argument(
        "--save-eval-probe-json",
        action="store_true",
        help="将完整多 batch eval 行（各层）写入 JSON，体量大时慎用",
    )
    parser.add_argument(
        "--probe-plot-rolling",
        type=int,
        default=0,
        help="训练曲线滚动平均窗口，0 表示长序列时自动",
    )
    parser.add_argument(
        "--probe-plot-train-stride",
        type=int,
        default=1,
        help="仅作图：每隔几个训练步取一点（1=全画）。不减少训练/CSV；例 32 步+stride=2 图上约 16 点",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="计算设备。x86/Rosetta 的 conda + MPS 易崩溃，请用 cpu 或设环境变量 TLB_FORCE_CPU=1",
    )
    parser.add_argument(
        "--probe-target",
        type=str,
        default="encoder",
        choices=("encoder", "decoder"),
        help="级联与探针挂在 Encoder 或 Decoder 自注意（不修改 Cross-Attn）。",
    )
    parser.add_argument(
        "--no-gradient-norm",
        action="store_true",
        help="Decoder 实验：不记录各步梯度 L2 范数",
    )
    parser.add_argument(
        "--no-linear-cka",
        action="store_true",
        help="Decoder 实验：不挂 FFN 前/后 Linear CKA 钩子",
    )
    parser.add_argument(
        "--cka-layers",
        type=str,
        default="2,5",
        help="Decoder CKA 的 decoder 层下标，逗号分隔，默认 2,5",
    )
    parser.add_argument(
        "--cka-interval",
        type=int,
        default=4,
        help="每多少个 train batch 记一次 CKA",
    )
    parser.add_argument(
        "--cka-max-tokens",
        type=int,
        default=2048,
        help="CKA 每步在 (B×T) 中随机抽样的位置数上限",
    )
    parser.add_argument(
        "--no-decoder-window-bleu",
        action="store_true",
        help="Decoder 实验：不在 generate 上评估 4/16/64/ full 历史窗口 BLEU",
    )
    parser.add_argument(
        "--temporal-mode",
        type=str,
        default="auto",
        choices=("auto", "circular_roll", "causal_prev"),
        help="Temporal smooth on V: auto=causal_prev for decoder, circular_roll for encoder.",
    )
    parser.add_argument(
        "--spatial-mode",
        type=str,
        default="auto",
        choices=("auto", "spatial_global", "prefix_mean"),
        help="Spatial smooth on V: auto=prefix_mean for decoder, spatial_global for encoder.",
    )
    parser.add_argument(
        "--exp-suffix",
        type=str,
        default="",
        help="Optional suffix appended to exp_name (e.g. circ or causal) to avoid overwrites.",
    )
    parser.add_argument(
        "--no-cascade",
        action="store_true",
        help="Vanilla finetune baseline: do not install value-path cascade on self-attention.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing coefficient for training CE (0 = HF default).",
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=None,
        help="Override Marian self-attention dropout on all encoder/decoder layers.",
    )

    args = parser.parse_args()
    cka_tup: Tuple[int, ...] = tuple(
        int(x.strip()) for x in (args.cka_layers or "2,5").split(",") if x.strip() != ""
    )
    if not cka_tup:
        cka_tup = (2, 5)
    
    result = run_probe_experiment(
        alpha_spatial=args.alpha_spatial,
        alpha_temporal=args.alpha_temporal,
        is_reverse=args.reverse,
        seed=args.seed,
        output_dir=args.output_dir,
        max_eval_batches=args.max_eval_batches,
        max_train_batches=args.max_train_batches,
        export_probe_csv=args.export_probe_csv,
        plot_probes=args.plot_probes,
        save_eval_probe_json=args.save_eval_probe_json,
        probe_plot_rolling=args.probe_plot_rolling,
        train_plot_stride=args.probe_plot_train_stride,
        include_legacy_step_plots=args.probe_legacy_step_plots,
        include_eval_batch_axis_plots=args.probe_legacy_eval_batch_plots,
        device=args.device,
        probe_target=args.probe_target,
        record_gradient_norm=not args.no_gradient_norm,
        record_linear_cka=not args.no_linear_cka,
        cka_layer_indices=cka_tup,
        cka_interval=args.cka_interval,
        cka_max_tokens=args.cka_max_tokens,
        decoder_window_bleu=not args.no_decoder_window_bleu,
        temporal_mode=args.temporal_mode,
        spatial_mode=args.spatial_mode,
        exp_suffix=args.exp_suffix,
        enable_cascade=not args.no_cascade,
        label_smoothing=float(args.label_smoothing),
        attention_dropout=args.attention_dropout,
    )
    
    print(f"\n{'='*60}")
    print(f"Final BLEU: {result['bleu']:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
