#!/usr/bin/env python
# coding: utf-8
"""
M3模型训练 + 邻居ISM分析（一体化脚本）
================================================================================
流程:
1. 数据加载与预处理（与原始训练完全一致）
2. M3模型训练
3. 训练完成后立即进行邻居ISM分析
4. 输出邻居重要性结果

优势: 训练和分析使用完全相同的配置，避免任何不一致
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import warnings
import json
from datetime import datetime
from tqdm import tqdm
import random
import gc
from collections import defaultdict
from sklearn.model_selection import train_test_split

# PyTorch Geometric
try:
    from torch_geometric.utils import subgraph
    from torch_geometric.nn import GCNConv
    print("✅ PyTorch Geometric loaded successfully")
except ImportError:
    print("❌ Error: PyTorch Geometric not installed.")
    print("请安装: pip install torch-geometric")
    exit(1)

warnings.filterwarnings('ignore')

# =================================================================
# 固定参数
# =================================================================
MAX_NEIGHBORS = 32
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 文件路径配置
INDEX_FILE = "gene_id_index.txt"
LABELS_FILE = "crayfish_labels.csv"
EMBEDDING_FILE = "crayfish_embeddings/crayfish_embeddings.pt"
TF_PATH = "processed_tf/crayfish_tf_edge_index.pt"
GCN_PATH = "processed_gcn/crayfish_gcn_network.pt"

# 输出目录
OUTPUT_DIR = "M3_Training_NeighborISM"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
NEIGHBOR_ISM_DIR = os.path.join(OUTPUT_DIR, "neighbor_importance")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ISM分析参数
N_HIGH_GENES = 100
N_LOW_GENES = 100
EXPR_THRESHOLD = 3.0
USE_ONLY_TEST_SET = True
ISM_METHOD = 'mean_fill'  # 'mean_fill' 或 'remove'
FILL_STRATEGY = 'global_mean'  # 'global_mean' 或 'local_mean'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =================================================================
# 深度回归头
# =================================================================
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


# =================================================================
# M3模型: Multi-Graph Concat (分图静态拼接)
# =================================================================
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


# =================================================================
# 工具函数
# =================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 随机种子已设置为: {seed}")


def load_gene_index(index_file):
    if not os.path.exists(index_file):
        print(f"❌ 索引文件不存在: {index_file}")
        return None, None
    with open(index_file, 'r') as f:
        gene_ids = [line.strip() for line in f if line.strip()]
    gene_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
    print(f"✅ 加载索引文件成功: {len(gene_ids)} 个基因")
    return gene_ids, gene_to_idx


def load_nt_embeddings(embedding_file):
    if not os.path.exists(embedding_file):
        print(f"❌ NT embeddings文件不存在: {embedding_file}")
        return None
    data = torch.load(embedding_file, map_location='cpu', weights_only=False)
    print(f"✅ 成功加载NT embeddings: {embedding_file}")
    print(f"   embeddings形状: {data['x'].shape}")
    print(f"   基因数: {len(data['gene_ids'])}")
    return {'embeddings': data['x'], 'gene_ids': data['gene_ids']}


def load_expression_data(labels_file, valid_gene_ids):
    if not os.path.exists(labels_file):
        print(f"❌ 标签文件不存在: {labels_file}")
        return None, None
    df = pd.read_csv(labels_file)
    print(f"✅ 成功加载标签文件: {labels_file}")

    if 'gene_id' not in df.columns:
        for col in ['GeneID', 'gene', 'Gene', 'id', 'ID']:
            if col in df.columns:
                df = df.rename(columns={col: 'gene_id'})
                break
    if 'label' not in df.columns:
        for col in ['mean_expression', 'expression', 'log2_expression', 'tpm']:
            if col in df.columns:
                df = df.rename(columns={col: 'label'})
                break

    expr_dict = {}
    for _, row in df.iterrows():
        if not pd.isna(row['label']):
            expr_dict[row['gene_id']] = row['label']

    expression_values = []
    for gene_id in valid_gene_ids:
        expression_values.append(expr_dict.get(gene_id, float('nan')))
    expression_tensor = torch.tensor(expression_values, dtype=torch.float32)
    found_count = sum(1 for v in expression_values if not np.isnan(v))
    print(f"✅ 标签对齐完成: {found_count}/{len(valid_gene_ids)} 个基因有表达值")
    return expression_tensor, [gid for gid in valid_gene_ids if gid in expr_dict]


def load_network_filtered(network_path, valid_gene_to_idx, network_name):
    if not os.path.exists(network_path):
        print(f"⚠️ {network_name}网络不存在: {network_path}")
        return torch.zeros((2, 0), dtype=torch.long)

    data = torch.load(network_path, map_location='cpu', weights_only=False)
    edge_index = data.get('edge_index', data.get('edges'))
    if edge_index is None:
        print(f"⚠️ {network_name}网络缺少edge_index")
        return torch.zeros((2, 0), dtype=torch.long)

    network_gene_list = data.get('gene_list', [])
    if not network_gene_list:
        print(f"⚠️ {network_name}网络缺少gene_list")
        return torch.zeros((2, 0), dtype=torch.long)

    network_idx_to_new_idx = {}
    matched_count = 0
    for net_idx, gid in enumerate(network_gene_list):
        if gid in valid_gene_to_idx:
            network_idx_to_new_idx[net_idx] = valid_gene_to_idx[gid]
            matched_count += 1

    if matched_count == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    if edge_index.dim() == 2 and edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()

    original_src = edge_index[0]
    original_dst = edge_index[1]

    key_list = list(network_idx_to_new_idx.keys())
    src_mask = torch.isin(original_src, torch.tensor(key_list, dtype=torch.long))
    dst_mask = torch.isin(original_dst, torch.tensor(key_list, dtype=torch.long))
    valid_mask = src_mask & dst_mask

    if valid_mask.sum() == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    valid_src = original_src[valid_mask]
    valid_dst = original_dst[valid_mask]

    new_src = torch.tensor([network_idx_to_new_idx[s.item()] for s in valid_src], dtype=torch.long)
    new_dst = torch.tensor([network_idx_to_new_idx[d.item()] for d in valid_dst], dtype=torch.long)

    new_edge_index = torch.stack([new_src, new_dst])
    print(f"   {network_name}网络: 原始边数 {edge_index.shape[1]} → 过滤后 {new_edge_index.shape[1]} 条边")
    return new_edge_index


# =================================================================
# 数据集类
# =================================================================
class M3Dataset(Dataset):
    def __init__(self, embeddings, tf_edge_index, gcn_edge_index,
                 expression_values, gene_ids, target_genes, seed=42):
        self.seed = seed
        self.embeddings = embeddings
        self.tf_edge_index = tf_edge_index
        self.gcn_edge_index = gcn_edge_index
        self.gene_ids = gene_ids
        self.num_nodes = len(gene_ids)
        self.gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
        self.expression_tensor = expression_values

        self.valid_genes = []
        self.valid_gene_indices = []
        self.valid_expression_mask = []
        for gene_id in target_genes:
            if gene_id in self.gene_to_idx:
                idx = self.gene_to_idx[gene_id]
                self.valid_genes.append(gene_id)
                self.valid_gene_indices.append(idx)
                has_expr = not torch.isnan(self.expression_tensor[idx])
                self.valid_expression_mask.append(has_expr)

        print(f"\n📊 数据集构建: {len(self.valid_genes)}/{len(target_genes)} 个基因")
        self.all_neighbors = self._precompute_all_neighbors()

    def _precompute_all_neighbors(self):
        print("   🔍 预计算邻居...")
        adj_dict = {'tf': defaultdict(list), 'gcn': defaultdict(list)}

        if self.tf_edge_index is not None and self.tf_edge_index.numel() > 0:
            for i in range(self.tf_edge_index.shape[1]):
                s, d = self.tf_edge_index[0, i].item(), self.tf_edge_index[1, i].item()
                if s < self.num_nodes and d < self.num_nodes:
                    adj_dict['tf'][s].append(d)
                    adj_dict['tf'][d].append(s)

        if self.gcn_edge_index is not None and self.gcn_edge_index.numel() > 0:
            for i in range(self.gcn_edge_index.shape[1]):
                s, d = self.gcn_edge_index[0, i].item(), self.gcn_edge_index[1, i].item()
                if s < self.num_nodes and d < self.num_nodes:
                    adj_dict['gcn'][s].append(d)
                    adj_dict['gcn'][d].append(s)

        all_neighbors = {}
        for node_idx in range(self.num_nodes):
            neighbors_dict = {}
            for net_name in ['tf', 'gcn']:
                neighbors = adj_dict[net_name].get(node_idx, [])
                if neighbors:
                    if len(neighbors) > MAX_NEIGHBORS:
                        random.seed(self.seed + node_idx)
                        neighbors = random.sample(neighbors, MAX_NEIGHBORS)
                    neighbors_dict[net_name] = torch.tensor(neighbors, dtype=torch.long)
                else:
                    neighbors_dict[net_name] = torch.tensor([node_idx], dtype=torch.long)
            all_neighbors[node_idx] = neighbors_dict
        return all_neighbors

    def __len__(self):
        return len(self.valid_genes)

    def __getitem__(self, idx):
        gene_id = self.valid_genes[idx]
        gene_idx = self.valid_gene_indices[idx]
        expression = self.expression_tensor[gene_idx]
        return {
            'gene_id': gene_id,
            'gene_idx': gene_idx,
            'neighbor_indices': self.all_neighbors[gene_idx],
            'expression': torch.tensor(expression, dtype=torch.float32),
            'has_expression': not torch.isnan(expression)
        }


def collate_fn(batch):
    return {
        'gene_ids': [item['gene_id'] for item in batch],
        'gene_indices': torch.LongTensor([item['gene_idx'] for item in batch]),
        'neighbor_indices': [item['neighbor_indices'] for item in batch],
        'expressions': torch.stack([item['expression'] for item in batch]),
        'has_expression': torch.BoolTensor([item['has_expression'] for item in batch])
    }


# =================================================================
# 训练器
# =================================================================
class M3Trainer:
    def __init__(self, model, device='cpu', learning_rate=5e-5, patience=15, seed=42):
        self.model = model.to(device)
        self.device = device
        self.seed = seed
        self.all_embeddings = None
        self.tf_edge_index = None
        self.gcn_edge_index = None
        self.num_nodes = 0

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        self.criterion = nn.HuberLoss(reduction='none', delta=1.0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def set_graph_data(self, all_embeddings, tf_edge_index, gcn_edge_index):
        self.all_embeddings = all_embeddings
        self.num_nodes = len(all_embeddings)
        self.tf_edge_index = tf_edge_index
        self.gcn_edge_index = gcn_edge_index

    def _extract_subgraphs(self, unique_nodes):
        unique_nodes_cpu = unique_nodes.cpu()
        if self.tf_edge_index is not None and self.tf_edge_index.numel() > 0:
            tf_sub, _ = subgraph(unique_nodes_cpu, self.tf_edge_index.cpu(), relabel_nodes=True, num_nodes=self.num_nodes)
            tf_sub = tf_sub.to(self.device)
        else:
            tf_sub = torch.zeros((2, 0), device=self.device, dtype=torch.long)

        if self.gcn_edge_index is not None and self.gcn_edge_index.numel() > 0:
            gcn_sub, _ = subgraph(unique_nodes_cpu, self.gcn_edge_index.cpu(), relabel_nodes=True, num_nodes=self.num_nodes)
            gcn_sub = gcn_sub.to(self.device)
        else:
            gcn_sub = torch.zeros((2, 0), device=self.device, dtype=torch.long)
        return tf_sub, gcn_sub

    def _prepare_batch(self, batch):
        gene_indices = batch['gene_indices'].to(self.device)
        neighbor_indices = batch['neighbor_indices']
        expressions = batch['expressions'].to(self.device)
        has_expression = batch['has_expression'].to(self.device)

        all_nodes = gene_indices.clone()
        for n_dict in neighbor_indices:
            for net_name in ['tf', 'gcn']:
                if net_name in n_dict:
                    all_nodes = torch.cat([all_nodes, n_dict[net_name].to(self.device)])

        unique_nodes = torch.unique(all_nodes)
        unique_nodes = unique_nodes[unique_nodes < self.num_nodes]

        if len(unique_nodes) == 0:
            return None

        x = self.all_embeddings[unique_nodes.cpu()].to(self.device)
        node_mapping = {node.item(): i for i, node in enumerate(unique_nodes)}

        tf_sub, gcn_sub = self._extract_subgraphs(unique_nodes)
        self.model.set_subgraphs(tf_sub, gcn_sub)

        target_local_indices = torch.tensor([node_mapping[idx.item()] for idx in gene_indices if idx.item() in node_mapping], device=self.device)
        valid_mask = torch.tensor([idx.item() in node_mapping for idx in gene_indices], device=self.device)

        return {
            'x': x,
            'target_local_indices': target_local_indices,
            'expressions': expressions[valid_mask],
            'has_expression': has_expression[valid_mask]
        }

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        num_valid = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch in pbar:
            prepared = self._prepare_batch(batch)
            if prepared is None or not prepared['has_expression'].any():
                continue

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs, _ = self.model(prepared['x'])
                outputs = outputs[prepared['target_local_indices']]
                loss = self.criterion(outputs[prepared['has_expression']], prepared['expressions'][prepared['has_expression']]).mean()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * prepared['has_expression'].sum().item()
            num_valid += prepared['has_expression'].sum().item()
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / num_valid if num_valid > 0 else float('inf')

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_valid = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                prepared = self._prepare_batch(batch)
                if prepared is None or not prepared['has_expression'].any():
                    continue

                outputs, _ = self.model(prepared['x'])
                outputs = outputs[prepared['target_local_indices']]
                valid_outputs = outputs[prepared['has_expression']]
                valid_targets = prepared['expressions'][prepared['has_expression']]

                loss = self.criterion(valid_outputs, valid_targets).mean()
                total_loss += loss.item() * prepared['has_expression'].sum().item()
                num_valid += prepared['has_expression'].sum().item()
                all_preds.extend(valid_outputs.cpu().numpy())
                all_targets.extend(valid_targets.cpu().numpy())

        avg_loss = total_loss / num_valid if num_valid > 0 else float('inf')
        pearson_corr = pearsonr(all_preds, all_targets)[0] if len(all_preds) > 1 else 0.0
        return avg_loss, pearson_corr, np.array(all_preds), np.array(all_targets)

    def train(self, train_loader, val_loader, epochs=100):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_pearson, _, _ = self.validate(val_loader)

            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
                self.best_model_state = self.model.state_dict().copy()
                print(f"  ✅ Epoch {epoch+1}: 最佳验证损失 {val_loss:.6f}, Pearson: {val_pearson:.4f}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"  🚨 Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 5 == 0:
                print(f"  📊 Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Pearson={val_pearson:.4f}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.best_loss


# =================================================================
# 邻居ISM分析器（与训练后直接使用）
# =================================================================
class NeighborISMAnalyzer:
    def __init__(self, model, all_embeddings, num_nodes, tf_edge_index, gcn_edge_index):
        self.model = model
        self.all_embeddings = all_embeddings
        self.num_nodes = num_nodes
        self.tf_edge_index = tf_edge_index.to(DEVICE)
        self.gcn_edge_index = gcn_edge_index.to(DEVICE)
        self.device = DEVICE
        self.global_mean_embedding = all_embeddings.mean(dim=0).to(DEVICE)
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

    def predict_full(self, center_idx, neighbor_indices):
        tf_sub, gcn_sub, selected_nodes, node_mapping, center_local = self._get_or_build_subgraph(
            center_idx, neighbor_indices)
        x = self.all_embeddings[selected_nodes].clone().to(self.device)
        self.model.set_subgraphs(tf_sub, gcn_sub)
        with torch.no_grad():
            pred, _ = self.model(x)
            return pred[center_local].item()

    def compute_neighbor_importance(self, center_idx, neighbor_indices, baseline_pred):
        k = len(neighbor_indices)
        fill_embedding = self.global_mean_embedding if FILL_STRATEGY == 'global_mean' else self.all_embeddings[neighbor_indices].mean(dim=0).to(self.device)

        importance_scores = []
        for i in range(k):
            if ISM_METHOD == 'mean_fill':
                tf_sub, gcn_sub, selected_nodes, node_mapping, center_local = self._get_or_build_subgraph(
                    center_idx, neighbor_indices)
                x = self.all_embeddings[selected_nodes].clone().to(self.device)
                removed_node = neighbor_indices[i]
                if removed_node != center_idx and removed_node in node_mapping:
                    x[node_mapping[removed_node]] = fill_embedding
                self.model.set_subgraphs(tf_sub, gcn_sub)
                with torch.no_grad():
                    pred, _ = self.model(x)
                    perturbed_pred = pred[center_local].item()
            else:
                masked = neighbor_indices[:i] + neighbor_indices[i+1:]
                if len(masked) == 0:
                    masked = [center_idx]
                perturbed_pred = self.predict_full(center_idx, masked)

            importance_scores.append(abs(baseline_pred - perturbed_pred))
        return importance_scores

    def clear_cache(self):
        self._subgraph_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()


# =================================================================
# 主函数
# =================================================================
def main():
    parser = argparse.ArgumentParser(description='M3训练 + 邻居ISM分析')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    args = parser.parse_args()

    print("=" * 80)
    print("🦞 M3模型训练 + 邻居ISM分析（一体化脚本）")
    print(f"   种子: {args.seed}")
    print(f"   设备: {DEVICE}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print("=" * 80)

    set_seed(args.seed)

    # 创建目录
    for d in [OUTPUT_DIR, MODEL_DIR, PRED_DIR, NEIGHBOR_ISM_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    # ========== 1. 加载数据 ==========
    print("\n1️⃣ 加载数据...")

    index_gene_ids, _ = load_gene_index(INDEX_FILE)
    embed_data = load_nt_embeddings(EMBEDDING_FILE)

    # 对齐基因
    embed_gene_set = set(embed_data['gene_ids'])
    index_gene_set = set(index_gene_ids)
    common_genes = list(index_gene_set.intersection(embed_gene_set))
    print(f"   共同基因数: {len(common_genes)}")

    embed_idx_map = {gid: i for i, gid in enumerate(embed_data['gene_ids'])}
    new_embeddings_list = []
    new_gene_ids_list = []
    for gid in index_gene_ids:
        if gid in embed_idx_map:
            new_embeddings_list.append(embed_data['embeddings'][embed_idx_map[gid]])
            new_gene_ids_list.append(gid)

    all_embeddings = torch.stack(new_embeddings_list)
    gene_ids = new_gene_ids_list
    gene_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
    num_nodes = len(gene_ids)
    print(f"   Embeddings形状: {all_embeddings.shape}")

    # 加载表达量
    expression_tensor, _ = load_expression_data(LABELS_FILE, gene_ids)

    # 加载网络
    valid_gene_to_idx = gene_to_idx
    tf_edge_index = load_network_filtered(TF_PATH, valid_gene_to_idx, "TF")
    gcn_edge_index = load_network_filtered(GCN_PATH, valid_gene_to_idx, "GCN")

    # 划分数据集
    indices = list(range(num_nodes))
    train_idx, temp_idx = train_test_split(indices, train_size=TRAIN_RATIO, random_state=args.seed, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, train_size=VAL_RATIO/(VAL_RATIO+TEST_RATIO), random_state=args.seed, shuffle=True)

    train_genes = [gene_ids[i] for i in train_idx]
    val_genes = [gene_ids[i] for i in val_idx]
    test_genes = [gene_ids[i] for i in test_idx]

    print(f"\n   训练集: {len(train_genes)} 基因")
    print(f"   验证集: {len(val_genes)} 基因")
    print(f"   测试集: {len(test_genes)} 基因")

    # 创建数据集
    train_dataset = M3Dataset(all_embeddings, tf_edge_index, gcn_edge_index, expression_tensor, gene_ids, train_genes, args.seed)
    val_dataset = M3Dataset(all_embeddings, tf_edge_index, gcn_edge_index, expression_tensor, gene_ids, val_genes, args.seed+1)
    test_dataset = M3Dataset(all_embeddings, tf_edge_index, gcn_edge_index, expression_tensor, gene_ids, test_genes, args.seed+2)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size*2, shuffle=False, collate_fn=collate_fn)

    # ========== 2. 训练M3模型 ==========
    print("\n2️⃣ 训练M3模型...")
    model = ModelM3_MultiGraphConcat(input_dim=2560)
    trainer = M3Trainer(model, device=DEVICE, learning_rate=args.learning_rate, seed=args.seed)
    trainer.set_graph_data(all_embeddings, tf_edge_index, gcn_edge_index)

    best_val_loss = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 保存模型
    model_path = os.path.join(MODEL_DIR, f'm3_seed{args.seed}_best.pth')
    torch.save({'model_state_dict': model.state_dict()}, model_path)
    print(f"   💾 模型已保存: {model_path}")

    # 测试集评估
    _, test_pearson, test_preds, test_targets = trainer.validate(test_loader)
    print(f"\n   📊 测试集 Pearson: {test_pearson:.4f}")

    # 保存预测结果
    test_gene_ids_list = []
    all_test_preds = []
    all_test_targets = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            gene_indices = batch['gene_indices']
            prepared = trainer._prepare_batch(batch)
            if prepared is not None:
                outputs, _ = model(prepared['x'])
                outputs = outputs[prepared['target_local_indices']]
                for i, idx in enumerate(gene_indices):
                    if idx.item() < len(gene_ids):
                        test_gene_ids_list.append(gene_ids[idx.item()])
                        all_test_preds.append(outputs[i].item())
                        all_test_targets.append(batch['expressions'][i].item())

    pred_df = pd.DataFrame({
        'gene_id': test_gene_ids_list,
        'true_expression': all_test_targets,
        'predicted_expression': all_test_preds,
        'set': 'test'
    })
    pred_df.to_csv(os.path.join(PRED_DIR, f'm3_seed{args.seed}_predictions.csv'), index=False)
    print(f"   💾 预测结果已保存")

    # ========== 3. 邻居ISM分析 ==========
    print("\n3️⃣ 邻居ISM分析...")

    # 选择目标基因（测试集中预测最准的）
    pred_df['abs_error'] = abs(pred_df['true_expression'] - pred_df['predicted_expression'])
    high_candidates = pred_df[pred_df['true_expression'] >= EXPR_THRESHOLD]
    low_candidates = pred_df[pred_df['true_expression'] < EXPR_THRESHOLD]

    high_genes = high_candidates.nsmallest(N_HIGH_GENES, 'abs_error')
    low_genes = low_candidates.nsmallest(N_LOW_GENES, 'abs_error')
    target_genes_df = pd.concat([high_genes, low_genes])
    print(f"   选中 {len(target_genes_df)} 个基因 (高表达{N_HIGH_GENES}+低表达{N_LOW_GENES})")

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

    def get_neighbors(gene_idx):
        neigh = list(set(tf_neighbors.get(gene_idx, []) + gcn_neighbors.get(gene_idx, [])))
        if len(neigh) == 0:
            neigh = [gene_idx]
        if len(neigh) > MAX_NEIGHBORS:
            random.seed(args.seed + gene_idx)
            neigh = random.sample(neigh, MAX_NEIGHBORS)
        return neigh

    analyzer = NeighborISMAnalyzer(model, all_embeddings, num_nodes, tf_edge_index, gcn_edge_index)

    for _, row in tqdm(target_genes_df.iterrows(), total=len(target_genes_df)):
        gene_id = row['gene_id']
        baseline_pred = row['predicted_expression']
        center_idx = gene_to_idx[gene_id]

        neighbor_indices = get_neighbors(center_idx)
        neighbor_ids = [gene_ids[i] for i in neighbor_indices]
        importance_scores = analyzer.compute_neighbor_importance(center_idx, neighbor_indices, baseline_pred)

        neighbor_df = pd.DataFrame({
            'neighbor_id': neighbor_ids,
            'importance_score': importance_scores,
            'rank': range(1, len(importance_scores)+1)
        }).sort_values('importance_score', ascending=False)
        neighbor_df.to_csv(os.path.join(NEIGHBOR_ISM_DIR, f"{gene_id}.csv"), index=False)

    analyzer.clear_cache()

    # 汇总统计
    print("\n4️⃣ 生成汇总...")
    all_neighbors = []
    for _, row in target_genes_df.iterrows():
        gene_id = row['gene_id']
        df = pd.read_csv(os.path.join(NEIGHBOR_ISM_DIR, f"{gene_id}.csv"))
        for _, nrow in df.iterrows():
            all_neighbors.append({
                'target_gene': gene_id,
                'target_group': 'High' if row['true_expression'] >= EXPR_THRESHOLD else 'Low',
                'neighbor_id': nrow['neighbor_id'],
                'importance': nrow['importance_score']
            })

    if all_neighbors:
        neighbor_df = pd.DataFrame(all_neighbors)
        top_neighbors = neighbor_df.groupby('neighbor_id')['importance'].agg(['mean', 'count']).round(4)
        top_neighbors.columns = ['mean_importance', 'frequency']
        top_neighbors = top_neighbors.sort_values('frequency', ascending=False).head(50)
        top_neighbors.to_csv(os.path.join(OUTPUT_DIR, 'top_regulatory_neighbors.csv'))

    summary_df = pd.DataFrame([{
        'gene_id': row['gene_id'],
        'group': 'High' if row['true_expression'] >= EXPR_THRESHOLD else 'Low',
        'true_expression': row['true_expression'],
        'predicted_expression': row['predicted_expression'],
    } for _, row in target_genes_df.iterrows()])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index=False)

    print("\n" + "=" * 80)
    print("✨ 完成！")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"   - 模型: {MODEL_DIR}")
    print(f"   - 预测: {PRED_DIR}")
    print(f"   - 邻居ISM: {NEIGHBOR_ISM_DIR}")
    print(f"   - Top调控因子: top_regulatory_neighbors.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()