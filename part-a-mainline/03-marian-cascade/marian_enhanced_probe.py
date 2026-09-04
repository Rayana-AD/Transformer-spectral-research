#!/usr/bin/env python3
"""
增强版 Marian 机理探针
支持：级联双阶段记录、Token级指标、训练时强制应用平滑
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import MarianMTModel, MarianTokenizer


# ============ Token级指标计算 ============

def compute_temporal_difference(V: torch.Tensor) -> torch.Tensor:
    """
    计算时序差分: ΔV_t = V_{t+1} - V_t
    V: (batch, heads, seq_len, head_dim)
    返回: (batch, heads, seq_len-1, head_dim)
    """
    if V.shape[-2] <= 1:
        return torch.zeros_like(V[..., :-1, :])
    return V[..., 1:, :] - V[..., :-1, :]


def compute_temporal_variance(V: torch.Tensor) -> torch.Tensor:
    """
    时序衰减度: TimeVar = Var(V, dim=seq_len)
    V: (batch, heads, seq_len, head_dim)
    返回: (batch, heads, head_dim)
    """
    return V.var(dim=-2, unbiased=False)


def compute_spatial_variance(V: torch.Tensor, window_size: int = 3) -> torch.Tensor:
    """
    空间尖锐度: 局部Token邻域方差
    V: (batch, heads, seq_len, head_dim)
    对每个位置t，计算其邻域[t-w//2, t+w//2]的方差
    """
    b, h, s, d = V.shape
    
    # 使用unfold创建滑动窗口
    # V_padded: (b, h, s+w-1, d)
    pad = window_size // 2
    V_padded = F.pad(V, (0, 0, pad, pad), mode='reflect')
    
    # unfold: (b, h, s, w, d)
    windows = V_padded.unfold(dimension=-2, size=window_size, step=1)
    # 现在 windows: (b, h, s, d, w)
    windows = windows.permute(0, 1, 2, 4, 3)
    
    # 计算窗口方差 (沿窗口维度)
    spatial_var = windows.var(dim=-2, unbiased=False)  # (b, h, s, d)
    
    return spatial_var.mean(dim=-1)  # (b, h, s) - 每个位置的空间方差


def compute_window_similarity(V: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    窗口余弦相似度: TimeSim_t = CosSim(V_t, mean(V_{t-k:t}))
    V: (batch, heads, seq_len, head_dim)
    返回: (batch, heads, seq_len)
    """
    b, h, s, d = V.shape
    
    if s <= k:
        return torch.ones(b, h, s, device=V.device)
    
    # 累积平均窗口
    V_normalized = F.normalize(V, p=2, dim=-1)  # (b, h, s, d)
    
    similarities = []
    for t in range(s):
        if t < k:
            # 前k个token，用前面所有可用的
            window_end = max(1, t)
            V_curr = V_normalized[..., t, :]
            V_window = V_normalized[..., :window_end, :].mean(dim=-2)
        else:
            V_curr = V_normalized[..., t, :]
            V_window = V_normalized[..., t-k:t, :].mean(dim=-2)
        
        sim = (V_curr * V_window).sum(dim=-1)  # (b, h)
        similarities.append(sim)
    
    return torch.stack(similarities, dim=-1)  # (b, h, s)


def compute_token_level_stats(V: torch.Tensor) -> Dict[str, float]:
    """计算所有Token级指标"""
    stats = {}
    
    # 1. 时序差分均值
    delta_V = compute_temporal_difference(V)
    stats['temporal_diff_mean'] = delta_V.abs().mean().item()
    stats['temporal_diff_std'] = delta_V.abs().std().item()
    
    # 2. 时序方差
    time_var = compute_temporal_variance(V)
    stats['temporal_var_mean'] = time_var.mean().item()
    stats['temporal_var_std'] = time_var.std().item()
    
    # 3. 空间尖锐度（局部方差）
    spatial_var = compute_spatial_variance(V, window_size=3)
    stats['spatial_var_mean'] = spatial_var.mean().item()
    stats['spatial_var_std'] = spatial_var.std().item()
    
    # 4. 窗口相似度
    window_sim = compute_window_similarity(V, k=3)
    stats['window_sim_mean'] = window_sim.mean().item()
    stats['window_sim_std'] = window_sim.std().item()
    
    # 5. 自注意力风格的全局指标
    V_flat = V.reshape(-1, V.shape[-1])
    stats['global_l2_mean'] = torch.norm(V_flat, p=2, dim=-1).mean().item()
    stats['global_variance'] = V_flat.var().item()
    
    return stats


# ============ 核范数与层间指标 ============

def analyze_nuclear_norm(tensor: torch.Tensor) -> Tuple[float, float, int]:
    """分析核范数: (spectral_norm, nuclear_norm, rank)"""
    if tensor.dim() == 4:
        b, h, s, d = tensor.shape
        matrix = tensor.reshape(b * h, s * d)
    elif tensor.dim() == 3:
        b, s, d = tensor.shape
        matrix = tensor.reshape(b, s * d)
    else:
        matrix = tensor
    
    try:
        svals = torch.linalg.svdvals(matrix.float())
        svals_np = svals.cpu().numpy()
        rank = int(np.sum(svals_np > 1e-6))
        spectral = float(svals_np[0]) if rank > 0 else 0.0
        nuclear = float(np.sum(svals_np))
    except:
        nuclear = float(torch.norm(matrix, p='nuc').cpu())
        spectral = float(torch.norm(matrix, p=2).cpu())
        rank = -1
    
    return spectral, nuclear, rank


def compute_layer_similarity(V_curr: torch.Tensor, V_prev: Optional[torch.Tensor]) -> Dict[str, float]:
    """计算层间相似度"""
    if V_prev is None:
        return {'cosine': 1.0, 'cka': 1.0}
    
    # 展平
    curr_flat = V_curr.reshape(-1, V_curr.shape[-1])
    prev_flat = V_prev.reshape(-1, V_prev.shape[-1])
    
    # 余弦相似度
    curr_norm = F.normalize(curr_flat, p=2, dim=-1)
    prev_norm = F.normalize(prev_flat, p=2, dim=-1)
    cosine = (curr_norm * prev_norm).sum(dim=-1).mean().item()
    
    # CKA
    curr_c = curr_flat - curr_flat.mean(dim=0, keepdim=True)
    prev_c = prev_flat - prev_flat.mean(dim=0, keepdim=True)
    
    cka_num = torch.trace(curr_c @ curr_c.T @ prev_c @ prev_c.T).item()
    cka_den = (torch.trace(curr_c @ curr_c.T @ curr_c @ curr_c.T).item() * 
               torch.trace(prev_c @ prev_c.T @ prev_c @ prev_c.T).item()) ** 0.5
    cka = cka_num / (cka_den + 1e-8)
    
    return {'cosine': cosine, 'cka': cka}


# ============ 增强版探针类 ============

class EnhancedCascadeProbe:
    """
    专为级联平滑设计的增强探针
    记录：原始V、第一次平滑后V_mid、第二次平滑后V_smooth、注意力输出
    """
    
    def __init__(self, record_during_eval: bool = False):
        self.record_during_eval = record_during_eval
        self.measurements = []  # 所有测量点
        self.layer_v_history = {}  # 每层V的历史
        self.hooks = []
        self.step_count = 0
        
    def register_cascade_hooks(self, model, mode: str = "cascade"):
        """
        注册钩子来捕获级联平滑的中间值
        注意：需要修改attention forward来暴露中间值
        """
        # 这里我们采用另一种方式：在训练循环中手动调用
        pass
    
    def manual_record(self, layer_name: str, step: int, 
                      V_raw: torch.Tensor, 
                      V_after_first: Optional[torch.Tensor] = None,
                      V_after_second: Optional[torch.Tensor] = None,
                      attn_output: Optional[torch.Tensor] = None,
                      loss: Optional[float] = None,
                      is_training: bool = True):
        """
        手动记录测量值（在训练循环中调用）
        
        参数:
            V_raw: 原始Value (cascade前)
            V_after_first: 第一次平滑后 (spatial或temporal)
            V_after_second: 第二次平滑后 (完整cascade结果)
            attn_output: Attention输出
            loss: 当前batch的loss
        """
        record = {
            'layer': layer_name,
            'step': step,
            'is_training': is_training,
            'loss': loss,
        }
        
        # 1. 原始V的统计
        raw_stats = compute_token_level_stats(V_raw)
        raw_stats['nuclear'] = analyze_nuclear_norm(V_raw)[1]
        record['V_raw'] = raw_stats
        
        # 2. 第一次平滑后的统计
        if V_after_first is not None:
            first_stats = compute_token_level_stats(V_after_first)
            first_stats['nuclear'] = analyze_nuclear_norm(V_after_first)[1]
            record['V_first'] = first_stats
            
            # 计算从raw到first的变化
            record['delta_first'] = {
                'nuclear_ratio': first_stats['nuclear'] / (raw_stats['nuclear'] + 1e-8),
                'temporal_var_ratio': first_stats['temporal_var_mean'] / (raw_stats['temporal_var_mean'] + 1e-8),
                'spatial_var_ratio': first_stats['spatial_var_mean'] / (raw_stats['spatial_var_mean'] + 1e-8),
            }
        
        # 3. 第二次平滑后的统计
        if V_after_second is not None:
            second_stats = compute_token_level_stats(V_after_second)
            second_stats['nuclear'] = analyze_nuclear_norm(V_after_second)[1]
            record['V_second'] = second_stats
            
            if V_after_first is not None:
                record['delta_second'] = {
                    'nuclear_ratio': second_stats['nuclear'] / (first_stats['nuclear'] + 1e-8),
                    'temporal_var_ratio': second_stats['temporal_var_mean'] / (first_stats['temporal_var_mean'] + 1e-8),
                    'spatial_var_ratio': second_stats['spatial_var_mean'] / (first_stats['spatial_var_mean'] + 1e-8),
                }
        
        # 4. Attention输出的统计
        if attn_output is not None:
            out_stats = compute_token_level_stats(attn_output)
            out_stats['nuclear'] = analyze_nuclear_norm(attn_output)[1]
            record['attn_output'] = out_stats
        
        self.measurements.append(record)
        self.step_count += 1
    
    def get_summary(self) -> Dict:
        """汇总所有测量结果"""
        if not self.measurements:
            return {}
        
        summary = {
            'num_records': len(self.measurements),
            'by_layer': {},
            'global_stats': {},
        }
        
        # 按层分组
        by_layer = defaultdict(list)
        for m in self.measurements:
            by_layer[m['layer']].append(m)
        
        # 每层统计
        for layer_name, records in sorted(by_layer.items()):
            layer_summary = {
                'num_records': len(records),
                'V_raw_stats': self._aggregate_records(records, 'V_raw'),
            }
            
            # 如果有cascade阶段的数据
            if any('V_first' in r for r in records):
                layer_summary['V_first_stats'] = self._aggregate_records(records, 'V_first')
            if any('V_second' in r for r in records):
                layer_summary['V_second_stats'] = self._aggregate_records(records, 'V_second')
            
            # 变化率统计
            if any('delta_first' in r for r in records):
                layer_summary['delta_first'] = self._aggregate_records(records, 'delta_first')
            if any('delta_second' in r for r in records):
                layer_summary['delta_second'] = self._aggregate_records(records, 'delta_second')
            
            summary['by_layer'][layer_name] = layer_summary
        
        # 全局统计：计算所有层的平均
        all_raw_nuclear = [r['V_raw']['nuclear'] for r in self.measurements]
        summary['global_stats']['mean_nuclear'] = np.mean(all_raw_nuclear)
        summary['global_stats']['std_nuclear'] = np.std(all_raw_nuclear)
        
        # 计算c链 (层间核范数比)
        if len(summary['by_layer']) > 1:
            layer_names = sorted(summary['by_layer'].keys())
            c_values = []
            for i in range(len(layer_names) - 1):
                curr_n = summary['by_layer'][layer_names[i]].get('V_raw_stats', {}).get('nuclear', {}).get('mean', 0)
                next_n = summary['by_layer'][layer_names[i+1]].get('V_raw_stats', {}).get('nuclear', {}).get('mean', 0)
                if curr_n > 1e-8:
                    c_values.append(next_n / curr_n)
            
            if c_values:
                summary['global_stats']['c_chain'] = {
                    'c_values': c_values,
                    'c_mean': np.mean(c_values),
                    'c_std': np.std(c_values),
                    'c_geom_mean': np.exp(np.mean(np.log(c_values))) if all(c > 0 for c in c_values) else 0,
                }
        
        return summary
    
    def _aggregate_records(self, records: List[Dict], key: str) -> Dict:
        """聚合多条记录的统计值"""
        values = [r[key] for r in records if key in r]
        if not values:
            return {}
        
        # 收集所有子键
        all_subkeys = set()
        for v in values:
            if isinstance(v, dict):
                all_subkeys.update(v.keys())
        
        aggregated = {}
        for subkey in all_subkeys:
            subvalues = [v[subkey] for v in values if isinstance(v, dict) and subkey in v]
            if subvalues:
                aggregated[subkey] = {
                    'mean': np.mean(subvalues),
                    'std': np.std(subvalues),
                    'min': np.min(subvalues),
                    'max': np.max(subvalues),
                }
        
        return aggregated
    
    def save(self, output_path: Path):
        """保存测量结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_summary()
        with open(output_path, 'w') as f:
            json.dump({
                'summary': summary,
                'raw_measurements': self.measurements,  # 可选：保存原始数据
            }, f, indent=2, default=float)
        
        print(f"Probe results saved to: {output_path}")
        return summary


# ============ 用于修改attention forward的工具 ============

def make_cascade_forward_with_probe(base_forward, probe: EnhancedCascadeProbe, layer_name: str, 
                                     alpha_spatial: float, alpha_temporal: float, 
                                     is_reverse: bool = False):
    """
    创建一个带有探针记录的cascade forward函数
    
    注意：这个函数假设 base_forward 是标准的 Marian attention forward
    我们需要在V被使用的地方插入探针记录
    """
    # 这是一个包装器，需要在实际修改 attention 模块时使用
    pass


if __name__ == "__main__":
    # 测试代码
    print("Enhanced Cascade Probe module loaded.")
    
    # 创建一个测试tensor
    test_V = torch.randn(2, 8, 64, 64)  # (batch, heads, seq, dim)
    
    # 测试token级指标
    stats = compute_token_level_stats(test_V)
    print("\nToken-level stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")
