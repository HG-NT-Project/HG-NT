#!/usr/bin/env python
# coding: utf-8
"""
单碱基分辨率序列ISM分析脚本 - 完整版
================================================================================
功能:
1. 加载训练好的M3模型
2. 对测试集中预测最准的200个基因进行分析
3. 滑动窗口遮挡序列，计算每个碱基的重要性
4. 输出: 重要性曲线(6000维)、热力图、Motif(含位置信息)
5. 累计贡献可视化: 高表达vs低表达组对比

分辨率: 单碱基级别 (滑动窗口，步长=1)
================================================================================
"""

import os
import random
import gc
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph
from pyfaidx import Fasta
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# =================================================================================
# 配置类
# =================================================================================
class Config:
    # ========== 路径配置 ==========
    M3_MODEL_PATH = "M3_Training_NeighborISM/models/m3_seed42_best.pth"
    EMBEDDINGS_FILE = "crayfish_embeddings/crayfish_embeddings.pt"
    PREDICTIONS_CSV = "M3_Training_NeighborISM/predictions/m3_seed42_predictions.csv"

    TF_PATH = "processed_tf/crayfish_tf_edge_index.pt"
    GCN_PATH = "processed_gcn/crayfish_gcn_network.pt"

    GENOME_FA = "ref.fa"
    ANNO_FILE = "anno.summary.xls"

    OUTPUT_DIR = "Sequence_ISM_Results"
    HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")
    MOTIF_DIR = os.path.join(OUTPUT_DIR, "motifs")
    CURVE_DIR = os.path.join(OUTPUT_DIR, "importance_curves")

    # ========== 分析参数 ==========
    N_HIGH_GENES = 50
    N_LOW_GENES = 50
    EXPR_THRESHOLD = 3.0
    USE_ONLY_TEST_SET = True

    # 序列参数
    SEQ_LENGTH = 6000
    WINDOW_SIZE = 6
    STRIDE = 3  # 单碱基分辨率

    # NT模型参数
    NT_MODEL_PATH = "../pretrain_model/Nucleotide-Transformer"
    NT_MAX_LENGTH = 1000
    BP_PER_TOKEN = SEQ_LENGTH / NT_MAX_LENGTH

    # 邻居参数
    MAX_NEIGHBORS = 32

    # Motif提取参数
    IMPORTANCE_PERCENTILE = 90
    MOTIF_MIN_LENGTH = 6
    MOTIF_MAX_LENGTH = 20

    # 可视化参数
    SMOOTH_SIGMA = 1.0
    HEATMAP_DPI = 150
    SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =================================================================================
# M3模型定义（与训练完全一致）
# =================================================================================
class DeepRegressor(nn.Module):
    def __init__(self, in_dim, dropout=0.3):
        super(DeepRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


class ModelM3_MultiGraphConcat(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3_MultiGraphConcat, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = None
        self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub = tf_sub
        self.gcn_sub = gcn_sub

    def forward(self, x, edge_index=None):
        s_feat = self.seq_proj(x)

        if self.tf_sub is not None and self.tf_sub.numel() > 0:
            t_info = F.elu(self.tf_conv(s_feat, self.tf_sub))
        else:
            t_info = s_feat

        if self.gcn_sub is not None and self.gcn_sub.numel() > 0:
            g_info = F.elu(self.gcn_conv(s_feat, self.gcn_sub))
        else:
            g_info = s_feat

        concat_feat = torch.cat([t_info, g_info], dim=-1)
        graph_feat = self.fusion(concat_feat)
        final_feat = s_feat + graph_feat
        return self.regressor(final_feat).squeeze(-1), {}


# =================================================================================
# 工具函数
# =================================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model():
    print("🤖 加载M3模型...")
    model = ModelM3_MultiGraphConcat(input_dim=2560).to(Config.DEVICE)
    checkpoint = torch.load(Config.M3_MODEL_PATH, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✅ 模型加载完成，设备: {Config.DEVICE}")
    return model


def load_nt_model():
    print("🤖 加载NT模型...")
    tokenizer = AutoTokenizer.from_pretrained(Config.NT_MODEL_PATH)
    model = AutoModelForMaskedLM.from_pretrained(
        Config.NT_MODEL_PATH,
        torch_dtype=torch.float16
    ).to(Config.DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"   ✅ NT模型加载完成")
    return tokenizer, model


def load_network_and_embeddings():
    """加载网络和Embeddings - 与训练代码完全一致"""
    print("🔗 加载网络和Embeddings...")

    embed_data = torch.load(Config.EMBEDDINGS_FILE, map_location='cpu', weights_only=False)
    gene_ids = embed_data['gene_ids']
    all_embeddings = embed_data['x']
    gene_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
    idx_to_gene = {i: gid for i, gid in enumerate(gene_ids)}
    num_nodes = len(gene_ids)

    print(f"   Embedding基因数: {num_nodes}")

    def load_and_map_network(path, network_name):
        if not os.path.exists(path):
            print(f"   ⚠️ {network_name}网络不存在")
            return torch.zeros((2, 0), dtype=torch.long)

        data = torch.load(path, map_location='cpu', weights_only=False)
        edge_index = data.get('edge_index', data.get('edges'))
        network_gene_list = data.get('gene_list', [])

        if not network_gene_list:
            return torch.zeros((2, 0), dtype=torch.long)

        network_idx_to_new_idx = {}
        for net_idx, gid in enumerate(network_gene_list):
            if gid in gene_to_idx:
                network_idx_to_new_idx[net_idx] = gene_to_idx[gid]

        if not network_idx_to_new_idx:
            return torch.zeros((2, 0), dtype=torch.long)

        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        src_mask = torch.isin(edge_index[0], torch.tensor(list(network_idx_to_new_idx.keys()), dtype=torch.long))
        dst_mask = torch.isin(edge_index[1], torch.tensor(list(network_idx_to_new_idx.keys()), dtype=torch.long))
        valid_mask = src_mask & dst_mask

        if valid_mask.sum() == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        valid_src = edge_index[0][valid_mask]
        valid_dst = edge_index[1][valid_mask]

        new_src = torch.tensor([network_idx_to_new_idx[s.item()] for s in valid_src], dtype=torch.long)
        new_dst = torch.tensor([network_idx_to_new_idx[d.item()] for d in valid_dst], dtype=torch.long)

        new_edge_index = torch.stack([new_src, new_dst])
        print(f"   {network_name}网络: {edge_index.shape[1]} → {new_edge_index.shape[1]} 条边")
        return new_edge_index

    tf_edge_index = load_and_map_network(Config.TF_PATH, "TF")
    gcn_edge_index = load_and_map_network(Config.GCN_PATH, "GCN")

    # 构建邻居字典
    tf_neighbors = defaultdict(list)
    gcn_neighbors = defaultdict(list)

    for i in range(tf_edge_index.shape[1]):
        s, d = tf_edge_index[0, i].item(), tf_edge_index[1, i].item()
        if s < num_nodes and d < num_nodes:
            tf_neighbors[s].append(d)
            tf_neighbors[d].append(s)

    for i in range(gcn_edge_index.shape[1]):
        s, d = gcn_edge_index[0, i].item(), gcn_edge_index[1, i].item()
        if s < num_nodes and d < num_nodes:
            gcn_neighbors[s].append(d)
            gcn_neighbors[d].append(s)

    print(f"   ✅ 全数据对齐完成，共 {num_nodes} 个基因")
    return (gene_to_idx, idx_to_gene, all_embeddings, num_nodes,
            tf_neighbors, gcn_neighbors, tf_edge_index, gcn_edge_index)


def parse_annotations(anno_file):
    """解析注释文件，获取染色体位置信息"""
    print("📖 解析注释文件...")
    df = pd.read_csv(anno_file, sep='\t')
    gene_info = {}

    for _, row in df.iterrows():
        gene_id_raw = str(row['GeneID'])
        parts = gene_id_raw.split(':')

        if len(parts) >= 5:
            gene_id = parts[0]
            chrom = parts[2]
            try:
                if '..' in parts[3]:
                    start = int(parts[3].split('..')[0])
                    end = int(parts[3].split('..')[1])
                    strand = parts[4] if len(parts) > 4 else '+'
                else:
                    start = int(parts[3])
                    end = int(parts[4])
                    strand = parts[5] if len(parts) > 5 else '+'
            except (ValueError, IndexError):
                continue
            gene_info[gene_id] = {
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': strand
            }

    print(f"   ✅ 解析 {len(gene_info)} 个基因")
    return gene_info


def extract_gene_sequence(genome, gene_info):
    """提取6000bp序列 (启动子2000bp + 基因体2000bp + 终止子2000bp)"""
    chrom = gene_info['chrom']
    start = gene_info['start']
    end = gene_info['end']
    strand = gene_info['strand']

    if chrom not in genome:
        for avail_chrom in genome.keys():
            if chrom in avail_chrom or avail_chrom in chrom:
                chrom = avail_chrom
                break
        else:
            return 'N' * Config.SEQ_LENGTH

    try:
        if strand == '+':
            tss_seq = str(genome[chrom][max(0, start - 2000):start + 1000])
            tts_seq = str(genome[chrom][max(0, end - 1000):end + 2000])
            seq = tss_seq + tts_seq
        else:
            def rc(s):
                return s[::-1].translate(str.maketrans('ATGCatgc', 'TACGtacg'))

            tss_seq = rc(str(genome[chrom][max(0, end - 1000):end + 2000]))
            tts_seq = rc(str(genome[chrom][max(0, start - 2000):start + 1000]))
            seq = tss_seq + tts_seq

        seq = seq.ljust(Config.SEQ_LENGTH, 'N')[:Config.SEQ_LENGTH].upper()
        return seq
    except Exception:
        return 'N' * Config.SEQ_LENGTH


def select_target_genes(predictions_csv, gene_to_idx):
    """选择目标基因：测试集中预测最准的高表达100个 + 低表达100个"""
    print("\n🎯 选择分析目标基因...")

    predictions = pd.read_csv(predictions_csv)
    print(f"   原始预测文件: {len(predictions)} 条记录")

    if 'true_expression' not in predictions.columns:
        for col in ['expression', 'label']:
            if col in predictions.columns:
                predictions = predictions.rename(columns={col: 'true_expression'})
                break

    if 'predicted_expression' not in predictions.columns:
        for col in ['prediction', 'pred']:
            if col in predictions.columns:
                predictions = predictions.rename(columns={col: 'predicted_expression'})
                break

    if Config.USE_ONLY_TEST_SET and 'set' in predictions.columns:
        original_count = len(predictions)
        predictions = predictions[predictions['set'] == 'test']
        print(f"   ✅ 只使用测试集: {original_count} → {len(predictions)} 个基因")

    predictions = predictions[predictions['gene_id'].isin(gene_to_idx.keys())]
    print(f"   有效基因: {len(predictions)} 个")

    predictions['abs_error'] = abs(predictions['true_expression'] - predictions['predicted_expression'])

    high_candidates = predictions[predictions['true_expression'] >= Config.EXPR_THRESHOLD]
    high_genes = high_candidates.nsmallest(Config.N_HIGH_GENES, 'abs_error')

    low_candidates = predictions[predictions['true_expression'] < Config.EXPR_THRESHOLD]
    low_genes = low_candidates.nsmallest(Config.N_LOW_GENES, 'abs_error')

    print(f"\n   📊 基因选择:")
    print(f"      高表达候选: {len(high_candidates)} → 选中 {len(high_genes)}")
    print(f"      低表达候选: {len(low_candidates)} → 选中 {len(low_genes)}")

    return pd.concat([high_genes, low_genes])


def get_neighbor_indices(gene_idx, tf_neighbors, gcn_neighbors, seed=42):
    """获取邻居索引（与训练一致）"""
    tf_neigh = tf_neighbors.get(gene_idx, [])
    gcn_neigh = gcn_neighbors.get(gene_idx, [])
    all_neigh_idx = list(set(tf_neigh + gcn_neigh))

    if len(all_neigh_idx) == 0:
        all_neigh_idx = [gene_idx]

    if len(all_neigh_idx) > Config.MAX_NEIGHBORS:
        random.seed(seed + gene_idx)
        all_neigh_idx = random.sample(all_neigh_idx, Config.MAX_NEIGHBORS)

    return all_neigh_idx


# =================================================================================
# 序列ISM分析器
# =================================================================================
class SequenceISMAnalyzer:
    def __init__(self, model, nt_tokenizer, nt_model, all_embeddings, num_nodes,
                 tf_edge_index, gcn_edge_index):
        self.model = model
        self.nt_tokenizer = nt_tokenizer
        self.nt_model = nt_model
        self.all_embeddings = all_embeddings
        self.num_nodes = num_nodes
        self.tf_edge_index = tf_edge_index.to(Config.DEVICE)
        self.gcn_edge_index = gcn_edge_index.to(Config.DEVICE)
        self.device = Config.DEVICE
        self._subgraph_cache = {}

    def _extract_local_subgraph(self, edge_index, selected_nodes):
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        if not isinstance(selected_nodes, torch.Tensor):
            selected_nodes = torch.tensor(selected_nodes, dtype=torch.long)

        selected_nodes = selected_nodes.to(edge_index.device)

        edge_sub, _ = subgraph(selected_nodes, edge_index, relabel_nodes=True, num_nodes=self.num_nodes)
        return edge_sub.to(self.device)

    def _get_or_build_subgraph(self, center_idx, neighbor_indices):
        cache_key = (center_idx, tuple(sorted(neighbor_indices)))
        if cache_key in self._subgraph_cache:
            return self._subgraph_cache[cache_key]

        all_nodes = [center_idx] + [int(n) for n in neighbor_indices if int(n) != int(center_idx)]
        selected_nodes = list(set(all_nodes))
        selected_nodes.sort()

        tf_sub = self._extract_local_subgraph(self.tf_edge_index, selected_nodes)
        gcn_sub = self._extract_local_subgraph(self.gcn_edge_index, selected_nodes)

        node_mapping = {old: new for new, old in enumerate(selected_nodes)}
        center_local = node_mapping[center_idx]

        cached = (tf_sub, gcn_sub, selected_nodes, node_mapping, center_local)
        self._subgraph_cache[cache_key] = cached
        return cached

    def _get_nt_features(self, sequence):
        """获取NT模型的embedding"""
        inputs = self.nt_tokenizer(
            sequence,
            return_tensors="pt",
            max_length=Config.NT_MAX_LENGTH,
            padding='max_length',
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nt_model(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            attention_mask = inputs['attention_mask']

        # Mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        center_embedding = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return center_embedding

    def _perturb_sequence(self, sequence, start, end):
        """将指定区域替换为N"""
        seq_list = list(sequence)
        for i in range(start, min(end, len(sequence))):
            seq_list[i] = 'N'
        return ''.join(seq_list)

    def predict_with_sequence(self, center_idx, neighbor_indices, sequence):
        """使用给定序列预测表达值"""
        # 获取NT编码
        center_embedding = self._get_nt_features(sequence)

        # 获取子图
        tf_sub, gcn_sub, selected_nodes, node_mapping, center_local = self._get_or_build_subgraph(
            center_idx, neighbor_indices
        )

        # 构建特征矩阵（邻居使用预训练embedding）
        x = self.all_embeddings[selected_nodes].clone().to(self.device)
        x[center_local] = center_embedding.squeeze(0)

        self.model.set_subgraphs(tf_sub, gcn_sub)

        with torch.no_grad():
            pred, _ = self.model(x)
            return pred[center_local].item()

    def compute_importance_curve(self, center_idx, neighbor_indices, sequence, baseline_pred):
        """计算重要性曲线（单碱基分辨率）"""
        seq_len = Config.SEQ_LENGTH
        window_size = Config.WINDOW_SIZE
        stride = Config.STRIDE

        # 生成窗口起始位置
        starts = list(range(0, seq_len - window_size + 1, stride))
        n_windows = len(starts)

        # 存储每个窗口的重要性
        window_importance = np.zeros(n_windows, dtype=np.float32)

        # 逐个窗口扰动
        for idx, start_bp in enumerate(tqdm(starts, desc="  扰动窗口", leave=False)):
            end_bp = start_bp + window_size
            perturbed_seq = self._perturb_sequence(sequence, start_bp, end_bp)
            perturbed_pred = self.predict_with_sequence(center_idx, neighbor_indices, perturbed_seq)
            window_importance[idx] = abs(baseline_pred - perturbed_pred)

        # 映射回单碱基分辨率
        importance_curve = np.zeros(seq_len, dtype=np.float32)
        counts = np.zeros(seq_len, dtype=np.float32)

        for idx, start_bp in enumerate(starts):
            score = window_importance[idx]
            importance_curve[start_bp:start_bp + window_size] += score
            counts[start_bp:start_bp + window_size] += 1

        avg_importance = np.divide(importance_curve, counts, out=np.zeros_like(importance_curve), where=counts != 0)

        # 平滑
        if Config.SMOOTH_SIGMA > 0:
            avg_importance = gaussian_filter1d(avg_importance, sigma=Config.SMOOTH_SIGMA)

        return avg_importance.astype(np.float32)

    def clear_cache(self):
        self._subgraph_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()


# =================================================================================
# 可视化函数
# =================================================================================
def plot_individual_heatmap(importance, gene_id, group, chrom, start, end, save_path):
    """绘制单基因热力图"""
    fig, ax = plt.subplots(figsize=(16, 4), dpi=150)

    data_2d = importance.reshape(1, -1)
    vmax = np.percentile(importance, 99) if importance.max() > 0 else 1

    im = ax.imshow(data_2d, cmap='Reds', aspect='auto', interpolation='nearest', vmin=0, vmax=vmax, origin='upper')

    ax.set_xticks([])
    ax.set_yticks([])

    # TSS (2000bp) 和 TTS (4000bp) 边界
    ax.axvline(x=2000 - 0.5, color='black', linewidth=2, alpha=0.9, linestyle='-')
    ax.axvline(x=4000 - 0.5, color='black', linewidth=2, alpha=0.9, linestyle='-')

    title = f"{gene_id} ({group}) | {chrom}:{start}-{end}"
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
    cbar.set_label('Importance Score', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=Config.HEATMAP_DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_cumulative_importance(all_results, output_path):
    """绘制累计重要性曲线（高表达 vs 低表达）"""
    high_curves = [r['importance_curve'] for r in all_results if r['group'] == 'High']
    low_curves = [r['importance_curve'] for r in all_results if r['group'] == 'Low']

    if not high_curves or not low_curves:
        print("   ⚠️ 没有足够数据绘制累计曲线")
        return

    high_mean = np.mean(high_curves, axis=0)
    low_mean = np.mean(low_curves, axis=0)

    # 计算累计贡献
    high_cumulative = np.cumsum(high_mean) / np.sum(high_mean)
    low_cumulative = np.cumsum(low_mean) / np.sum(low_mean)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    x = np.arange(Config.SEQ_LENGTH)

    # 子图1：平均重要性曲线
    ax1 = axes[0]
    ax1.plot(x, high_mean, 'r-', linewidth=1.5, label=f'High Expression (n={len(high_curves)})')
    ax1.plot(x, low_mean, 'b-', linewidth=1.5, label=f'Low Expression (n={len(low_curves)})')
    ax1.axvline(x=2000, color='black', linestyle='--', linewidth=1, alpha=0.7, label='TSS')
    ax1.axvline(x=4000, color='black', linestyle='--', linewidth=1, alpha=0.7, label='TTS')
    ax1.fill_betweenx([0, ax1.get_ylim()[1]], 0, 2000, alpha=0.05, color='orange')
    ax1.fill_betweenx([0, ax1.get_ylim()[1]], 2000, 4000, alpha=0.05, color='lightgreen')
    ax1.fill_betweenx([0, ax1.get_ylim()[1]], 4000, 6000, alpha=0.05, color='lightblue')
    ax1.set_xlabel('Position (bp)')
    ax1.set_ylabel('Mean Importance')
    ax1.set_title('Average Importance Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：累计贡献曲线
    ax2 = axes[1]
    ax2.step(x, high_cumulative, 'r-', linewidth=1.5, where='mid', label='High Expression')
    ax2.step(x, low_cumulative, 'b-', linewidth=1.5, where='mid', label='Low Expression')
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='50% threshold')
    ax2.axvline(x=2000, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=4000, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_betweenx([0, 1], 0, 2000, alpha=0.05, color='orange')
    ax2.fill_betweenx([0, 1], 2000, 4000, alpha=0.05, color='lightgreen')
    ax2.fill_betweenx([0, 1], 4000, 6000, alpha=0.05, color='lightblue')
    ax2.set_xlabel('Position (bp)')
    ax2.set_ylabel('Cumulative Proportion')
    ax2.set_title('Cumulative Importance Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 累计曲线已保存: {output_path}")


def extract_motifs(importance_curve, sequence, gene_info, gene_id, group):
    """提取Motif，包含染色体位置信息"""
    threshold = np.percentile(importance_curve, Config.IMPORTANCE_PERCENTILE)

    motifs = []
    i = 0
    while i < len(importance_curve):
        if importance_curve[i] > threshold:
            start = i
            while i < len(importance_curve) and importance_curve[i] > threshold:
                i += 1
            end = i
            length = end - start

            if Config.MOTIF_MIN_LENGTH <= length <= Config.MOTIF_MAX_LENGTH:
                motif_seq = sequence[start:end]

                # 确定区域
                if start < 2000:
                    region = "Promoter_Upstream"
                    genomic_start = gene_info['start'] - (2000 - start) if gene_info['strand'] == '+' else gene_info[
                                                                                                               'end'] + start
                    genomic_end = gene_info['start'] - (2000 - end) if gene_info['strand'] == '+' else gene_info[
                                                                                                           'end'] + end
                elif start < 4000:
                    region = "Gene_Body"
                    if gene_info['strand'] == '+':
                        genomic_start = gene_info['start'] + (start - 2000)
                        genomic_end = gene_info['start'] + (end - 2000)
                    else:
                        genomic_start = gene_info['end'] - (end - 2000)
                        genomic_end = gene_info['end'] - (start - 2000)
                else:
                    region = "Terminator"
                    genomic_start = gene_info['end'] + (start - 4000) if gene_info['strand'] == '+' else gene_info[
                                                                                                             'start'] - (
                                                                                                                     start - 4000)
                    genomic_end = gene_info['end'] + (end - 4000) if gene_info['strand'] == '+' else gene_info[
                                                                                                         'start'] - (
                                                                                                                 end - 4000)

                motifs.append({
                    'gene_id': gene_id,
                    'group': group,
                    'chrom': gene_info['chrom'],
                    'strand': gene_info['strand'],
                    'genomic_start': max(0, genomic_start),
                    'genomic_end': max(0, genomic_end),
                    'region': region,
                    'seq_position_start': start,
                    'seq_position_end': end,
                    'sequence': motif_seq,
                    'mean_importance': float(importance_curve[start:end].mean()),
                    'length': length
                })
        else:
            i += 1

    return motifs


def save_motif_fastas(motif_list, output_dir):
    """保存Motif FASTA（包含位置信息）"""
    if not motif_list:
        return

    motif_df = pd.DataFrame(motif_list)

    # 全部Motif
    with open(os.path.join(output_dir, 'all_motifs.fa'), 'w') as f:
        for _, row in motif_df.iterrows():
            header = f">{row['gene_id']}|{row['chrom']}:{row['genomic_start']}-{row['genomic_end']}({row['strand']})|{row['region']}|imp={row['mean_importance']:.4f}"
            f.write(f"{header}\n{row['sequence']}\n")

    # 高表达组
    high_df = motif_df[motif_df['group'] == 'High']
    if len(high_df) > 0:
        with open(os.path.join(output_dir, 'high_expression_motifs.fa'), 'w') as f:
            for _, row in high_df.iterrows():
                header = f">{row['gene_id']}|{row['chrom']}:{row['genomic_start']}-{row['genomic_end']}({row['strand']})|{row['region']}|imp={row['mean_importance']:.4f}"
                f.write(f"{header}\n{row['sequence']}\n")

    # 低表达组
    low_df = motif_df[motif_df['group'] == 'Low']
    if len(low_df) > 0:
        with open(os.path.join(output_dir, 'low_expression_motifs.fa'), 'w') as f:
            for _, row in low_df.iterrows():
                header = f">{row['gene_id']}|{row['chrom']}:{row['genomic_start']}-{row['genomic_end']}({row['strand']})|{row['region']}|imp={row['mean_importance']:.4f}"
                f.write(f"{header}\n{row['sequence']}\n")

    print(f"   ✅ Motif统计: 全部={len(motif_df)}, 高表达={len(high_df)}, 低表达={len(low_df)}")


# =================================================================================
# 主函数
# =================================================================================
def main():
    start_time = time.time()

    print("=" * 80)
    print("🧬 单碱基分辨率序列ISM分析")
    print(f"   窗口大小: {Config.WINDOW_SIZE}bp, 步长: {Config.STRIDE}bp")
    print(f"   高表达: {Config.N_HIGH_GENES}, 低表达: {Config.N_LOW_GENES}")
    print(f"   Motif阈值: {Config.IMPORTANCE_PERCENTILE}%")
    print("=" * 80)

    set_seed(Config.SEED)

    # 创建输出目录
    for d in [Config.OUTPUT_DIR, Config.HEATMAP_DIR, Config.MOTIF_DIR, Config.CURVE_DIR]:
        os.makedirs(d, exist_ok=True)

    # 1. 加载数据
    print("\n1️⃣ 加载数据...")
    gene_info = parse_annotations(Config.ANNO_FILE)
    genome = Fasta(Config.GENOME_FA, as_raw=True, sequence_always_upper=True)

    (gene_to_idx, idx_to_gene, all_embeddings, num_nodes,
     tf_neighbors, gcn_neighbors, tf_edge_index, gcn_edge_index) = load_network_and_embeddings()

    # 2. 选择目标基因
    print("\n2️⃣ 选择目标基因...")
    target_genes_df = select_target_genes(Config.PREDICTIONS_CSV, gene_to_idx)

    if len(target_genes_df) == 0:
        print("❌ 没有找到目标基因")
        return

    # 3. 加载模型
    print("\n3️⃣ 加载模型...")
    model = load_model()
    nt_tokenizer, nt_model = load_nt_model()

    # 4. 初始化分析器
    print("\n4️⃣ 初始化序列ISM分析器...")
    analyzer = SequenceISMAnalyzer(
        model, nt_tokenizer, nt_model, all_embeddings, num_nodes,
        tf_edge_index, gcn_edge_index
    )

    # 5. 分析每个基因
    print(f"\n5️⃣ 开始分析 {len(target_genes_df)} 个基因...")
    print(f"   ⚠️ 注意: 每个基因需要 {Config.SEQ_LENGTH - Config.WINDOW_SIZE + 1} 次NT编码和M3推理")
    print(f"   预计耗时较长，请耐心等待...")

    all_results = []
    all_motifs = []

    for idx, (_, row) in enumerate(tqdm(target_genes_df.iterrows(), total=len(target_genes_df), desc="基因分析")):
        gene_id = row['gene_id']
        group = 'High' if row['true_expression'] >= Config.EXPR_THRESHOLD else 'Low'
        true_expr = row['true_expression']
        baseline_pred = row['predicted_expression']

        if gene_id not in gene_info:
            print(f"   ⚠️ {gene_id} 不在注释中，跳过")
            continue

        if gene_id not in gene_to_idx:
            print(f"   ⚠️ {gene_id} 不在embedding索引中，跳过")
            continue

        center_idx = gene_to_idx[gene_id]

        # 获取序列
        sequence = extract_gene_sequence(genome, gene_info[gene_id])
        if sequence.count('N') > 3000:
            print(f"   ⚠️ {gene_id} N含量过高，跳过")
            continue

        # 获取邻居
        neighbor_indices = get_neighbor_indices(center_idx, tf_neighbors, gcn_neighbors, Config.SEED)

        print(f"\n   📊 分析 {gene_id} ({group})")
        print(f"      真实值: {true_expr:.4f}, 预测值: {baseline_pred:.4f}")
        print(f"      邻居数: {len(neighbor_indices)}")

        # 计算重要性曲线
        importance_curve = analyzer.compute_importance_curve(
            center_idx, neighbor_indices, sequence, baseline_pred
        )

        # 保存重要性曲线
        curve_df = pd.DataFrame({
            'position': range(Config.SEQ_LENGTH),
            'importance': importance_curve,
            'region': ['Promoter' if p < 2000 else 'Gene_Body' if p < 4000 else 'Terminator' for p in
                       range(Config.SEQ_LENGTH)]
        })
        curve_df.to_csv(os.path.join(Config.CURVE_DIR, f"{gene_id}.csv"), index=False)

        # 绘制热力图
        chrom = gene_info[gene_id]['chrom']
        start_pos = gene_info[gene_id]['start']
        end_pos = gene_info[gene_id]['end']
        heatmap_path = os.path.join(Config.HEATMAP_DIR, f"{gene_id}.png")
        plot_individual_heatmap(importance_curve, gene_id, group, chrom, start_pos, end_pos, heatmap_path)

        # 提取Motif
        motifs = extract_motifs(importance_curve, sequence, gene_info[gene_id], gene_id, group)
        all_motifs.extend(motifs)

        # 保存结果
        all_results.append({
            'gene_id': gene_id,
            'group': group,
            'true_expression': true_expr,
            'baseline_prediction': baseline_pred,
            'importance_curve': importance_curve,
            'n_neighbors': len(neighbor_indices),
            'chrom': chrom,
            'start': start_pos,
            'end': end_pos
        })

        # 定期清理缓存
        if (idx + 1) % 10 == 0:
            analyzer.clear_cache()
            print(f"\n   🧹 已清理缓存")

    print(f"\n   ✅ 成功分析 {len(all_results)} 个基因")
    print(f"   🔍 发现 {len(all_motifs)} 个候选Motif")

    # 6. 保存Motif
    print("\n6️⃣ 保存Motif FASTA...")
    save_motif_fastas(all_motifs, Config.MOTIF_DIR)

    # 7. 绘制累计贡献曲线
    print("\n7️⃣ 绘制累计贡献曲线...")
    plot_cumulative_importance(all_results, os.path.join(Config.OUTPUT_DIR, 'cumulative_importance.png'))

    # 8. 汇总统计
    summary_df = pd.DataFrame([{
        'gene_id': r['gene_id'],
        'group': r['group'],
        'chrom': r['chrom'],
        'start': r['start'],
        'end': r['end'],
        'true_expression': r['true_expression'],
        'baseline_prediction': r['baseline_prediction'],
        'mean_importance': r['importance_curve'].mean(),
        'max_importance': r['importance_curve'].max(),
        'promoter_importance': r['importance_curve'][0:2000].mean(),
        'gene_body_importance': r['importance_curve'][2000:4000].mean(),
        'terminator_importance': r['importance_curve'][4000:6000].mean(),
        'n_neighbors': r['n_neighbors']
    } for r in all_results])
    summary_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'summary.csv'), index=False)

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("✨ 序列ISM分析完成！")
    print(f"⏱️  总耗时: {elapsed_time:.2f} 秒 ({elapsed_time / 60:.2f} 分钟)")
    print(f"📁 输出目录: {Config.OUTPUT_DIR}")
    print(f"   - 热力图: {Config.HEATMAP_DIR}")
    print(f"   - Motif: {Config.MOTIF_DIR}")
    print(f"   - 重要性曲线: {Config.CURVE_DIR}")
    print(f"   - 累计曲线: cumulative_importance.png")
    print(f"   - 汇总表: summary.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()