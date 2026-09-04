"""
Marian MT 注意力机理诊断工具
测量: 核范数链、Value方差、层间相似度
参考: script6_scattering_criticality.py
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

from marian_attn_tlb_patch import apply_marian_attention, uninstall_marian_tlb
from mt_train_tlb import load_local_fixed_pairs


def analyze_nuclear_norm(tensor: torch.Tensor) -> Tuple[float, float, float]:
    """
    分析张量的核范数相关指标
    输入: (batch, num_heads, seq_len, head_dim) 或 (batch, seq_len, dim)
    返回: (spectral_norm, nuclear_norm, rank)
    """
    # 将张量 reshape 为 2D 矩阵进行分析
    if tensor.dim() == 4:
        # (batch, num_heads, seq_len, head_dim) -> (batch*num_heads, seq_len*head_dim)
        b, h, s, d = tensor.shape
        matrix = tensor.reshape(b * h, s * d)
    elif tensor.dim() == 3:
        # (batch, seq_len, dim) -> (batch, seq_len*dim)
        b, s, d = tensor.shape
        matrix = tensor.reshape(b, s * d)
    else:
        matrix = tensor
    
    # 使用 numpy SVD
    s = np.linalg.svd(matrix.detach().float().cpu().numpy(), compute_uv=False)
    
    rank = int(np.sum(s > 1e-6))
    spectral = float(s[0]) if rank > 0 else 0.0
    nuclear = float(np.sum(s))
    
    return spectral, nuclear, rank


def measure_v_variance(value: torch.Tensor) -> Dict[str, float]:
    """
    测量 Value 张量的多维度方差
    value shape: (batch, num_heads, seq_len, head_dim)
    """
    var_stats = {}
    
    # 1. 序列维度方差 (沿 seq_len)
    var_seq = value.var(dim=-2, unbiased=False).mean().item()
    var_stats['var_seq'] = var_seq
    
    # 2. 特征维度方差 (沿 head_dim)
    var_feat = value.var(dim=-1, unbiased=False).mean().item()
    var_stats['var_feat'] = var_feat
    
    # 3. 头间方差 (各 head 的均值之间的方差)
    v_per_head_mean = value.mean(dim=[-2, -1])  # (batch, num_heads)
    var_head = v_per_head_mean.var(dim=-1).mean().item()
    var_stats['var_head'] = var_head
    
    # 4. 批次方差
    var_batch = value.mean(dim=[-2, -1]).var(dim=0).mean().item()
    var_stats['var_batch'] = var_batch
    
    # 5. 总方差
    var_total = value.var().item()
    var_stats['var_total'] = var_total
    
    # 6. L2 范数均值
    l2_mean = torch.norm(value, p=2, dim=[-2, -1]).mean().item()
    var_stats['l2_mean'] = l2_mean
    
    return var_stats


def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    计算 Centered Kernel Alignment (CKA) 相似度
    X, Y: (batch, seq_len, dim) 或 flatten 后的 (batch*seq_len, dim)
    """
    # Flatten if needed
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
        Y = Y.reshape(-1, Y.shape[-1])
    
    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute CKA
    XXT = X @ X.T
    YYT = Y @ Y.T
    
    hsic = torch.trace(XXT @ YYT)
    norm_x = torch.trace(XXT @ XXT)
    norm_y = torch.trace(YYT @ YYT)
    
    cka = (hsic / (torch.sqrt(norm_x * norm_y) + 1e-8)).item()
    return cka


def compute_cosine_similarity(X: torch.Tensor, Y: torch.Tensor) -> float:
    """计算两个张量的平均余弦相似度"""
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
        Y = Y.reshape(-1, Y.shape[-1])
    
    X = F.normalize(X, p=2, dim=-1)
    Y = F.normalize(Y, p=2, dim=-1)
    
    sim = (X * Y).sum(dim=-1).mean().item()
    return sim


class MarianMechanismProbe:
    """Marian 模型机理探针"""
    
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-de-en", device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # 存储每层测量的钩子
        self.layer_values = {}
        self.layer_outputs = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册前向钩子来捕获每层注意力输入输出"""
        def make_hook(layer_idx, storage_dict):
            def hook(module, input, output):
                # output 是 (attn_output, attn_weights) 元组
                if isinstance(output, tuple):
                    attn_output = output[0]  # (batch, seq_len, num_heads, head_dim)
                    storage_dict[layer_idx] = attn_output.detach()
            return hook
        
        # 遍历 encoder 和 decoder 的注意力层
        layer_idx = 0
        for layer in self.model.model.encoder.layers:
            h = layer.self_attn.register_forward_hook(make_hook(f"enc_{layer_idx}", self.layer_outputs))
            self.hooks.append(h)
            layer_idx += 1
        
        layer_idx = 0
        for layer in self.model.model.decoder.layers:
            # Self attention
            h = layer.self_attn.register_forward_hook(make_hook(f"dec_self_{layer_idx}", self.layer_outputs))
            self.hooks.append(h)
            # Cross attention
            h = layer.encoder_attn.register_forward_hook(make_hook(f"dec_cross_{layer_idx}", self.layer_outputs))
            self.hooks.append(h)
            layer_idx += 1
    
    def remove_hooks(self):
        """移除所有钩子"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def probe_batch(self, batch, attention_mode: str = "vanilla", **attn_kwargs) -> Dict:
        """对单个 batch 进行机理测量"""
        # 应用注意力模式
        uninstall_marian_tlb()
        apply_marian_attention(attention_mode, **attn_kwargs)
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # 清空存储
        self.layer_outputs = {}
        self.register_hooks()
        
        with torch.no_grad():
            # Encoder only (测量 encoder 注意力)
            encoder_outputs = self.model.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        self.remove_hooks()
        
        # 分析每层输出
        results = {
            'nuclear_chain': [],
            'variance_stats': [],
            'layer_sims': [],
            'c_values': []
        }
        
        prev_output = None
        prev_nuclear = None
        
        for key in sorted(self.layer_outputs.keys()):
            if not key.startswith("enc_"):
                continue
            
            output = self.layer_outputs[key]  # (batch, seq_len, num_heads, head_dim)
            
            # 1. 核范数分析
            spectral, nuclear, rank = analyze_nuclear_norm(output)
            results['nuclear_chain'].append({
                'layer': key,
                'spectral': spectral,
                'nuclear': nuclear,
                'rank': rank
            })
            
            # 2. Value 方差 (这里分析的是输出，对于 V-smooth 方法有意义)
            var_stats = measure_v_variance(output)
            var_stats['layer'] = key
            results['variance_stats'].append(var_stats)
            
            # 3. 层间相似度
            if prev_output is not None:
                cos_sim = compute_cosine_similarity(prev_output, output)
                cka_sim = compute_cka(prev_output, output)
                results['layer_sims'].append({
                    'layer_pair': f"{prev_key}->{key}",
                    'cosine': cos_sim,
                    'cka': cka_sim
                })
            
            prev_output = output
            prev_key = key
            
            # 4. 计算 c = N_{t+1} / N_t
            if prev_nuclear is not None:
                c = nuclear / prev_nuclear if prev_nuclear > 1e-8 else float('nan')
                results['c_values'].append({
                    'transition': f"{prev_key}->{key}",
                    'c': c
                })
            prev_nuclear = nuclear
        
        return results
    
    def run_probe(self, dataloader, attention_mode: str = "vanilla", max_batches: int = 10, **attn_kwargs) -> Dict:
        """在多个 batch 上运行探针并汇总"""
        all_results = []
        
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            
            results = self.probe_batch(batch, attention_mode, **attn_kwargs)
            all_results.append(results)
        
        # 汇总统计
        summary = self._summarize_results(all_results)
        return summary
    
    def _summarize_results(self, all_results: List[Dict]) -> Dict:
        """汇总多个 batch 的结果"""
        summary = {
            'nuclear_chain_mean': [],
            'variance_mean': [],
            'layer_sim_mean': [],
            'c_stats': {}
        }
        
        # 按层汇总核范数
        layer_nuc_groups = {}
        for r in all_results:
            for nuc in r['nuclear_chain']:
                layer = nuc['layer']
                if layer not in layer_nuc_groups:
                    layer_nuc_groups[layer] = {'nuclear': [], 'spectral': [], 'rank': []}
                layer_nuc_groups[layer]['nuclear'].append(nuc['nuclear'])
                layer_nuc_groups[layer]['spectral'].append(nuc['spectral'])
                layer_nuc_groups[layer]['rank'].append(nuc['rank'])
        
        for layer, stats in sorted(layer_nuc_groups.items()):
            summary['nuclear_chain_mean'].append({
                'layer': layer,
                'nuclear_mean': np.mean(stats['nuclear']),
                'nuclear_std': np.std(stats['nuclear']),
                'ln_nuclear': np.log(np.mean(stats['nuclear']))
            })
        
        # 汇总 c 值
        all_c = []
        for r in all_results:
            for c_info in r['c_values']:
                if not math.isnan(c_info['c']):
                    all_c.append(c_info['c'])
        
        if all_c:
            summary['c_stats'] = {
                'c_mean': np.mean(all_c),
                'c_std': np.std(all_c),
                'c_median': np.median(all_c),
                'c_bar_geom': np.exp(np.mean(np.log(all_c))) if all(c > 0 for c in all_c) else float('nan')
            }
        
        # 汇总方差
        var_keys = ['var_seq', 'var_feat', 'var_head', 'var_total']
        for key in var_keys:
            values = []
            for r in all_results:
                for v in r['variance_stats']:
                    if key in v:
                        values.append(v[key])
            if values:
                summary['variance_mean'].append({
                    'metric': key,
                    'mean': np.mean(values),
                    'std': np.std(values)
                })
        
        return summary


# 简单的数据集类
class SimpleDataset:
    def __init__(self, pairs, tokenizer, max_length=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        de, en = self.pairs[idx]
        model_inputs = self.tokenizer(
            de, max_length=self.max_length, truncation=True, return_tensors="pt", padding="max_length"
        )
        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
        }


def create_dataloader(pairs, tokenizer, batch_size=4, shuffle=False):
    dataset = SimpleDataset(pairs, tokenizer)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(description="Marian MT Mechanism Probe")
    parser.add_argument("--model-id", type=str, default="Helsinki-NLP/opus-mt-de-en")
    parser.add_argument("--attention", type=str, default="vanilla")
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results_mechanism_probe")
    parser.add_argument("--device", type=str, default="auto")
    # 平滑参数
    parser.add_argument("--smooth-alpha", type=float, default=0.1)
    parser.add_argument("--alpha-temporal", type=float, default=0.1)
    parser.add_argument("--alpha-global", type=float, default=0.1)
    parser.add_argument("--alpha-spatial", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / f"{args.attention}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据 (使用 val 集的少量数据)
    print("Loading data...")
    train_pairs, val_pairs = load_local_fixed_pairs()
    
    # 初始化tokenizer
    tokenizer = MarianTokenizer.from_pretrained(args.model_id)
    val_loader = create_dataloader(val_pairs, tokenizer, batch_size=4, shuffle=False)
    
    # 初始化探针
    print(f"Initializing probe with {args.model_id}...")
    probe = MarianMechanismProbe(args.model_id, device=args.device)
    
    # 准备注意力参数
    attn_kwargs = {
        'smooth_alpha': args.smooth_alpha,
        'alpha_temporal': args.alpha_temporal,
        'alpha_global': args.alpha_global,
        'alpha_spatial': args.alpha_spatial,
    }
    
    # 运行探针
    print(f"Running probe for attention mode: {args.attention}...")
    summary = probe.run_probe(
        val_loader, 
        attention_mode=args.attention,
        max_batches=args.max_batches,
        **attn_kwargs
    )
    
    # 保存结果
    output_file = output_dir / "probe_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Probe Summary for {args.attention} ===")
    print(f"\nNuclear Chain (ln N):")
    for item in summary['nuclear_chain_mean']:
        print(f"  {item['layer']}: ln(N) = {item['ln_nuclear']:.4f}")
    
    print(f"\nc value stats:")
    if summary['c_stats']:
        for k, v in summary['c_stats'].items():
            print(f"  {k}: {v:.4f}")
    
    print(f"\nVariance stats:")
    for item in summary['variance_mean']:
        print(f"  {item['metric']}: {item['mean']:.6f} ± {item['std']:.6f}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
