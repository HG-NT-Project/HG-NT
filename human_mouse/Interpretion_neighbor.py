#!/usr/bin/env python
# coding: utf-8
"""
Human M3模型训练 + 邻居ISM分析（一体化脚本）
================================================================================
基于M3模型（Multi-Graph Weighted Sum - 可学习权重融合）
流程:
1. 数据加载与预处理
2. M3模型训练
3. 训练完成后立即进行邻居ISM分析（仅对高表达中预测最准的前200个基因）
4. 输出邻居重要性结果

特点:
- 使用可学习权重的M3模型
- 集成ISM分析识别重要调控邻居
- 只分析高表达基因中预测最准的200个基因
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
    from torch_geometric.utils import subgraph, coalesce
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

# 文件路径配置（人类数据）
PPI_PATH = "processed_ppi/human_ppi_edge_index.pt"
TF_PATH = "processed_tf/human_tf_edge_index.pt"
GCN_PATH = "processed_gcn/human_gcn_network_aligned.pt"
EMBEDDING_FILE = "processed_features/human_nt_embeddings.pt"
LABELS_FILE = "processed_labels/human_labels.pt"

# 输出目录
OUTPUT_DIR = "Human_M3_NeighborISM"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
NEIGHBOR_ISM_DIR = os.path.join(OUTPUT_DIR, "neighbor_importance")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ISM分析参数 - 只分析高表达中预测最准的前200个基因
N_TOP_GENES = 200  # 分析200个高表达基因
EXPR_THRESHOLD = 3.0  # 高表达阈值（log2 scale），只分析表达量>=此值的基因
ISM_METHOD = 'mean_fill'  # 'mean_fill' 或 'remove'
FILL_STRATEGY = 'global_mean'  # 'global_mean' 或 'local_mean'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =================================================================
# M3模型: Multi-Graph Weighted Sum (可学习权重融合)
# =================================================================
class ModelM3_MultiGraphWeightedSum(nn.Module):
    """M3: 分图加权和 - 可学习权重融合 (Softmax归一化)"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3_MultiGraphWeightedSum, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 三路GCN
        self.ppi_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)

        # 可学习的logits，用于softmax计算融合权重 (3个网络)
        self.fusion_logits = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))

        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.ppi_sub = self.tf_sub = self.gcn_sub = None
        self.ppi_weight = self.tf_weight = self.gcn_weight = None

    def set_subgraphs(self, ppi_sub, tf_sub, gcn_sub):
        """设置子图并使用coalesce规范化"""
        if ppi_sub is not None and ppi_sub.numel() > 0:
            self.ppi_sub, self.ppi_weight = coalesce(ppi_sub, None, reduce='mean')
        else:
            self.ppi_sub = torch.zeros((2, 0), dtype=torch.long)
            self.ppi_weight = None

        if tf_sub is not None and tf_sub.numel() > 0:
            self.tf_sub, self.tf_weight = coalesce(tf_sub, None, reduce='mean')
        else:
            self.tf_sub = torch.zeros((2, 0), dtype=torch.long)
            self.tf_weight = None

        if gcn_sub is not None and gcn_sub.numel() > 0:
            self.gcn_sub, self.gcn_weight = coalesce(gcn_sub, None, reduce='mean')
        else:
            self.gcn_sub = torch.zeros((2, 0), dtype=torch.long)
            self.gcn_weight = None

    def forward(self, x, edge_index=None):
        s_feat = self.seq_proj(x)

        # 三路图卷积
        if self.ppi_sub.numel() > 0:
            if self.ppi_weight is not None:
                p_info = F.elu(self.ppi_conv(s_feat, self.ppi_sub, self.ppi_weight))
            else:
                p_info = F.elu(self.ppi_conv(s_feat, self.ppi_sub))
        else:
            p_info = s_feat

        if self.tf_sub.numel() > 0:
            if self.tf_weight is not None:
                t_info = F.elu(self.tf_conv(s_feat, self.tf_sub, self.tf_weight))
            else:
                t_info = F.elu(self.tf_conv(s_feat, self.tf_sub))
        else:
            t_info = s_feat

        if self.gcn_sub.numel() > 0:
            if self.gcn_weight is not None:
                g_info = F.elu(self.gcn_conv(s_feat, self.gcn_sub, self.gcn_weight))
            else:
                g_info = F.elu(self.gcn_conv(s_feat, self.gcn_sub))
        else:
            g_info = s_feat

        # 使用softmax计算融合权重（确保权重和为1）
        fusion_weights = F.softmax(self.fusion_logits, dim=0)
        ppi_w = fusion_weights[0]
        tf_w = fusion_weights[1]
        gcn_w = fusion_weights[2]

        # 加权和融合
        graph_feat = ppi_w * p_info + tf_w * t_info + gcn_w * g_info

        # 残差连接
        final_feat = s_feat + graph_feat

        weights = {
            "ppi_weight": ppi_w.item(),
            "tf_weight": tf_w.item(),
            "gcn_weight": gcn_w.item(),
        }

        return self.regressor(final_feat).squeeze(-1), weights


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


def load_nt_embeddings(embedding_file):
    if not os.path.exists(embedding_file):
        print(f"❌ NT embeddings文件不存在: {embedding_file}")
        return None

    data = torch.load(embedding_file, map_location='cpu', weights_only=False)
    print(f"✅ 成功加载NT embeddings: {embedding_file}")
    print(f"   embeddings形状: {data['x'].shape}")
    print(f"   基因数: {len(data['gene_ids'])}")
    return {'embeddings': data['x'], 'gene_ids': data['gene_ids']}


def load_expression_data(labels_file, gene_ids):
    if not os.path.exists(labels_file):
        print(f"❌ 标签文件不存在: {labels_file}")
        return None, None

    data = torch.load(labels_file, map_location='cpu', weights_only=False)

    # 构建表达量字典
    labels = data['labels']
    label_gene_ids = data.get('gene_id', data.get('gene_ids'))

    if label_gene_ids is None:
        print(f"❌ 标签文件中没有基因ID信息")
        return None, None

    expr_dict = {}
    for i, gid in enumerate(label_gene_ids):
        expr_dict[gid] = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]

    # 对齐到基因列表
    expression_values = []
    for gid in gene_ids:
        expression_values.append(expr_dict.get(gid, float('nan')))

    expression_tensor = torch.tensor(expression_values, dtype=torch.float32)
    found_count = sum(1 for v in expression_values if not np.isnan(v))
    print(f"✅ 标签对齐完成: {found_count}/{len(gene_ids)} 个基因有表达值")

    return expression_tensor, expr_dict


def load_network_filtered(network_path, valid_gene_to_idx, network_name):
    if not os.path.exists(network_path):
        print(f"⚠️ {network_name}网络不存在: {network_path}")
        return None

    data = torch.load(network_path, map_location='cpu', weights_only=False)

    if isinstance(data, dict):
        edge_index = data.get('edge_index', data.get('edges'))
        network_gene_list = data.get('gene_list', data.get('gene_ids', []))
    else:
        edge_index = data
        network_gene_list = []

    if edge_index is None:
        print(f"⚠️ {network_name}网络缺少edge_index")
        return None

    # 如果没有gene_list，假设索引已经对齐
    if not network_gene_list:
        print(f"⚠️ {network_name}网络缺少gene_list，假设索引已对齐")
        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        max_idx = edge_index.max().item()
        if max_idx >= len(valid_gene_to_idx):
            valid_mask = (edge_index[0] < len(valid_gene_to_idx)) & (edge_index[1] < len(valid_gene_to_idx))
            edge_index = edge_index[:, valid_mask]
            print(f"   {network_name}网络: 过滤后 {edge_index.shape[1]} 条边")
        return edge_index

    # 构建映射
    network_idx_to_new_idx = {}
    matched_count = 0
    for net_idx, gid in enumerate(network_gene_list):
        if gid in valid_gene_to_idx:
            network_idx_to_new_idx[net_idx] = valid_gene_to_idx[gid]
            matched_count += 1

    if matched_count == 0:
        print(f"⚠️ {network_name}网络没有匹配的基因")
        return None

    if edge_index.dim() == 2 and edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()

    original_src = edge_index[0]
    original_dst = edge_index[1]

    key_list = list(network_idx_to_new_idx.keys())
    src_mask = torch.isin(original_src, torch.tensor(key_list, dtype=torch.long))
    dst_mask = torch.isin(original_dst, torch.tensor(key_list, dtype=torch.long))
    valid_mask = src_mask & dst_mask

    if valid_mask.sum() == 0:
        print(f"⚠️ {network_name}网络没有有效边")
        return None

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
    def __init__(self, embeddings, ppi_edge_index, tf_edge_index, gcn_edge_index,
                 expression_values, gene_ids, target_genes, seed=42):
        self.seed = seed
        self.embeddings = embeddings
        self.ppi_edge_index = ppi_edge_index
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
        print(f"   有表达值: {sum(self.valid_expression_mask)}")

        self.all_neighbors = self._precompute_all_neighbors()

    def _precompute_all_neighbors(self):
        print("   🔍 预计算邻居...")
        adj_dict = {'ppi': defaultdict(list), 'tf': defaultdict(list), 'gcn': defaultdict(list)}

        if self.ppi_edge_index is not None and self.ppi_edge_index.numel() > 0:
            for i in range(self.ppi_edge_index.shape[1]):
                s, d = self.ppi_edge_index[0, i].item(), self.ppi_edge_index[1, i].item()
                if s < self.num_nodes and d < self.num_nodes:
                    adj_dict['ppi'][s].append(d)
                    adj_dict['ppi'][d].append(s)

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

        # 使用固定种子确保一致性
        random.seed(self.seed)

        all_neighbors = {}
        for node_idx in range(self.num_nodes):
            neighbors_dict = {}
            for net_name in ['ppi', 'tf', 'gcn']:
                neighbors = adj_dict[net_name].get(node_idx, [])
                if neighbors:
                    if len(neighbors) > MAX_NEIGHBORS:
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
# M3训练器
# =================================================================
class M3Trainer:
    def __init__(self, model, device='cpu', learning_rate=5e-5, patience=15, seed=42):
        self.model = model.to(device)
        self.device = device
        self.seed = seed
        self.all_embeddings = None
        self.ppi_edge_index = None
        self.tf_edge_index = None
        self.gcn_edge_index = None
        self.num_nodes = 0

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        self.criterion = nn.HuberLoss(reduction='none', delta=1.0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def set_graph_data(self, all_embeddings, ppi_edge_index, tf_edge_index, gcn_edge_index):
        self.all_embeddings = all_embeddings
        self.num_nodes = len(all_embeddings)
        self.ppi_edge_index = ppi_edge_index
        self.tf_edge_index = tf_edge_index
        self.gcn_edge_index = gcn_edge_index

    def _extract_subgraphs(self, unique_nodes):
        unique_nodes_cpu = unique_nodes.cpu()

        if self.ppi_edge_index is not None and self.ppi_edge_index.numel() > 0:
            ppi_sub, _ = subgraph(unique_nodes_cpu, self.ppi_edge_index.cpu(),
                                  relabel_nodes=True, num_nodes=self.num_nodes)
            ppi_sub = ppi_sub.to(self.device)
        else:
            ppi_sub = torch.zeros((2, 0), device=self.device, dtype=torch.long)

        if self.tf_edge_index is not None and self.tf_edge_index.numel() > 0:
            tf_sub, _ = subgraph(unique_nodes_cpu, self.tf_edge_index.cpu(),
                                 relabel_nodes=True, num_nodes=self.num_nodes)
            tf_sub = tf_sub.to(self.device)
        else:
            tf_sub = torch.zeros((2, 0), device=self.device, dtype=torch.long)

        if self.gcn_edge_index is not None and self.gcn_edge_index.numel() > 0:
            gcn_sub, _ = subgraph(unique_nodes_cpu, self.gcn_edge_index.cpu(),
                                  relabel_nodes=True, num_nodes=self.num_nodes)
            gcn_sub = gcn_sub.to(self.device)
        else:
            gcn_sub = torch.zeros((2, 0), device=self.device, dtype=torch.long)

        return ppi_sub, tf_sub, gcn_sub

    def _prepare_batch(self, batch):
        gene_indices = batch['gene_indices'].to(self.device)
        neighbor_indices = batch['neighbor_indices']
        expressions = batch['expressions'].to(self.device)
        has_expression = batch['has_expression'].to(self.device)

        all_nodes = gene_indices.clone()
        for n_dict in neighbor_indices:
            for net_name in ['ppi', 'tf', 'gcn']:
                if net_name in n_dict:
                    all_nodes = torch.cat([all_nodes, n_dict[net_name].to(self.device)])

        unique_nodes = torch.unique(all_nodes)
        unique_nodes = unique_nodes[unique_nodes < self.num_nodes]

        if len(unique_nodes) == 0:
            return None

        x = self.all_embeddings[unique_nodes.cpu()].to(self.device)
        node_mapping = {node.item(): i for i, node in enumerate(unique_nodes)}

        ppi_sub, tf_sub, gcn_sub = self._extract_subgraphs(unique_nodes)
        self.model.set_subgraphs(ppi_sub, tf_sub, gcn_sub)

        target_local_indices = torch.tensor(
            [node_mapping[idx.item()] for idx in gene_indices if idx.item() in node_mapping],
            device=self.device
        )
        valid_mask = torch.tensor(
            [idx.item() in node_mapping for idx in gene_indices],
            device=self.device
        )

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
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)

        for batch in pbar:
            prepared = self._prepare_batch(batch)
            if prepared is None or not prepared['has_expression'].any():
                continue

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs, weights = self.model(prepared['x'])
                outputs = outputs[prepared['target_local_indices']]
                loss = self.criterion(
                    outputs[prepared['has_expression']],
                    prepared['expressions'][prepared['has_expression']]
                ).mean()

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

                outputs, weights = self.model(prepared['x'])
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
                print(f"  ✅ Epoch {epoch + 1}: 最佳验证损失 {val_loss:.6f}, Pearson: {val_pearson:.4f}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"  🚨 Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0:
                print(
                    f"  📊 Epoch {epoch + 1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Pearson={val_pearson:.4f}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.best_loss


# =================================================================
# 邻居ISM分析器
# =================================================================
class NeighborISMAnalyzer:
    def __init__(self, model, all_embeddings, num_nodes,
                 ppi_edge_index, tf_edge_index, gcn_edge_index):
        self.model = model
        self.all_embeddings = all_embeddings
        self.num_nodes = num_nodes
        self.ppi_edge_index = ppi_edge_index.to(DEVICE) if ppi_edge_index is not None else None
        self.tf_edge_index = tf_edge_index.to(DEVICE) if tf_edge_index is not None else None
        self.gcn_edge_index = gcn_edge_index.to(DEVICE) if gcn_edge_index is not None else None
        self.device = DEVICE
        self.global_mean_embedding = all_embeddings.mean(dim=0).to(DEVICE)
        self._subgraph_cache = {}

    def _extract_local_subgraph(self, edge_index, selected_nodes):
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        if not isinstance(selected_nodes, torch.Tensor):
            selected_nodes = torch.tensor(selected_nodes, dtype=torch.long)

        selected_nodes = selected_nodes.to(edge_index.device)

        edge_sub, _ = subgraph(selected_nodes, edge_index, relabel_nodes=True,
                               num_nodes=self.num_nodes)
        return edge_sub.to(self.device)

    def _get_or_build_subgraph(self, center_idx, neighbor_indices):
        cache_key = (center_idx, tuple(sorted(neighbor_indices)))
        if cache_key in self._subgraph_cache:
            return self._subgraph_cache[cache_key]

        all_nodes = [center_idx] + [int(n) for n in neighbor_indices if int(n) != int(center_idx)]
        selected_nodes = list(set(all_nodes))
        selected_nodes.sort()

        ppi_sub = self._extract_local_subgraph(self.ppi_edge_index, selected_nodes)
        tf_sub = self._extract_local_subgraph(self.tf_edge_index, selected_nodes)
        gcn_sub = self._extract_local_subgraph(self.gcn_edge_index, selected_nodes)

        node_mapping = {old: new for new, old in enumerate(selected_nodes)}
        center_local = node_mapping[center_idx]

        cached = (ppi_sub, tf_sub, gcn_sub, selected_nodes, node_mapping, center_local)
        self._subgraph_cache[cache_key] = cached
        return cached

    def predict_full(self, center_idx, neighbor_indices):
        ppi_sub, tf_sub, gcn_sub, selected_nodes, node_mapping, center_local = self._get_or_build_subgraph(
            center_idx, neighbor_indices)
        x = self.all_embeddings[selected_nodes].clone().to(self.device)
        self.model.set_subgraphs(ppi_sub, tf_sub, gcn_sub)
        with torch.no_grad():
            pred, _ = self.model(x)
            return pred[center_local].item()

    def compute_neighbor_importance(self, center_idx, neighbor_indices, baseline_pred):
        k = len(neighbor_indices)

        if FILL_STRATEGY == 'global_mean':
            fill_embedding = self.global_mean_embedding
        else:
            fill_embedding = self.all_embeddings[neighbor_indices].mean(dim=0).to(self.device)

        importance_scores = []
        for i in range(k):
            if ISM_METHOD == 'mean_fill':
                ppi_sub, tf_sub, gcn_sub, selected_nodes, node_mapping, center_local = self._get_or_build_subgraph(
                    center_idx, neighbor_indices)
                x = self.all_embeddings[selected_nodes].clone().to(self.device)
                removed_node = neighbor_indices[i]
                if removed_node != center_idx and removed_node in node_mapping:
                    x[node_mapping[removed_node]] = fill_embedding
                self.model.set_subgraphs(ppi_sub, tf_sub, gcn_sub)
                with torch.no_grad():
                    pred, _ = self.model(x)
                    perturbed_pred = pred[center_local].item()
            else:
                masked = neighbor_indices[:i] + neighbor_indices[i + 1:]
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
    parser = argparse.ArgumentParser(description='Human M3训练 + 邻居ISM分析')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()

    print("=" * 80)
    print("🧬 Human M3模型训练 + 邻居ISM分析")
    print(f"   种子: {args.seed}")
    print(f"   设备: {DEVICE}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   ISM分析: 高表达基因中预测最准的前{N_TOP_GENES}个 (表达量≥{EXPR_THRESHOLD})")
    print("=" * 80)

    set_seed(args.seed)

    # 创建目录
    for d in [OUTPUT_DIR, MODEL_DIR, PRED_DIR, NEIGHBOR_ISM_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    # ========== 1. 加载数据 ==========
    print("\n1️⃣ 加载数据...")

    # 加载embeddings
    embed_data = load_nt_embeddings(EMBEDDING_FILE)
    if embed_data is None:
        print("❌ 加载embeddings失败")
        return

    gene_ids = embed_data['gene_ids']
    all_embeddings = embed_data['embeddings']
    num_nodes = len(gene_ids)
    gene_to_idx = {gid: i for i, gid in enumerate(gene_ids)}

    # 加载表达量
    expression_tensor, expr_dict = load_expression_data(LABELS_FILE, gene_ids)
    if expression_tensor is None:
        print("❌ 加载表达量数据失败")
        return

    # 加载网络
    print("\n2️⃣ 加载网络...")
    ppi_edge_index = load_network_filtered(PPI_PATH, gene_to_idx, "PPI")
    tf_edge_index = load_network_filtered(TF_PATH, gene_to_idx, "TF")
    gcn_edge_index = load_network_filtered(GCN_PATH, gene_to_idx, "GCN")

    # 过滤没有表达值的基因
    has_expr_mask = ~torch.isnan(expression_tensor)
    valid_indices = torch.where(has_expr_mask)[0].tolist()
    valid_genes = [gene_ids[i] for i in valid_indices]
    print(f"\n   有表达值的基因数: {len(valid_genes)}/{num_nodes}")

    # 划分数据集（只用有表达值的基因）
    indices = valid_indices
    train_idx, temp_idx = train_test_split(
        indices, train_size=TRAIN_RATIO, random_state=args.seed, shuffle=True
    )
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_ratio_adjusted, random_state=args.seed, shuffle=True
    )

    train_genes = [gene_ids[i] for i in train_idx]
    val_genes = [gene_ids[i] for i in val_idx]
    test_genes = [gene_ids[i] for i in test_idx]

    print(f"\n   训练集: {len(train_genes)} 基因")
    print(f"   验证集: {len(val_genes)} 基因")
    print(f"   测试集: {len(test_genes)} 基因")

    # 创建数据集
    train_dataset = M3Dataset(
        all_embeddings, ppi_edge_index, tf_edge_index, gcn_edge_index,
        expression_tensor, gene_ids, train_genes, args.seed
    )
    val_dataset = M3Dataset(
        all_embeddings, ppi_edge_index, tf_edge_index, gcn_edge_index,
        expression_tensor, gene_ids, val_genes, args.seed + 1
    )
    test_dataset = M3Dataset(
        all_embeddings, ppi_edge_index, tf_edge_index, gcn_edge_index,
        expression_tensor, gene_ids, test_genes, args.seed + 2
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn
    )

    # ========== 2. 训练M3模型 ==========
    print("\n3️⃣ 训练M3模型...")
    model = ModelM3_MultiGraphWeightedSum(
        input_dim=all_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )

    trainer = M3Trainer(
        model, device=DEVICE, learning_rate=args.learning_rate,
        patience=15, seed=args.seed
    )
    trainer.set_graph_data(all_embeddings, ppi_edge_index, tf_edge_index, gcn_edge_index)

    best_val_loss = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 保存模型
    model_path = os.path.join(MODEL_DIR, f'm3_seed{args.seed}_best.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'fusion_logits': model.fusion_logits.data.cpu().numpy(),
    }, model_path)
    print(f"   💾 模型已保存: {model_path}")

    # 打印融合权重
    fusion_weights = F.softmax(model.fusion_logits, dim=0)
    print(f"\n   🔮 学习到的融合权重:")
    print(f"      PPI权重: {fusion_weights[0].item():.4f}")
    print(f"      TF权重:  {fusion_weights[1].item():.4f}")
    print(f"      GCN权重: {fusion_weights[2].item():.4f}")

    # 测试集评估
    _, test_pearson, test_preds, test_targets = trainer.validate(test_loader)
    test_r2 = r2_score(test_targets, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_spearman = spearmanr(test_targets, test_preds)[0]

    print(f"\n   📊 测试集结果:")
    print(f"      R²: {test_r2:.6f}")
    print(f"      Pearson: {test_pearson:.6f}")
    print(f"      Spearman: {test_spearman:.6f}")
    print(f"      RMSE: {test_rmse:.6f}")

    # 保存所有测试集预测结果
    test_gene_ids_list = []
    all_test_preds = []
    all_test_targets = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            prepared = trainer._prepare_batch(batch)
            if prepared is not None:
                outputs, _ = model(prepared['x'])
                outputs = outputs[prepared['target_local_indices']]
                for i, idx in enumerate(batch['gene_indices']):
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
    pred_df['abs_error'] = abs(pred_df['true_expression'] - pred_df['predicted_expression'])
    pred_df.to_csv(os.path.join(PRED_DIR, f'm3_seed{args.seed}_predictions.csv'), index=False)
    print(f"   💾 所有测试集预测结果已保存 ({len(pred_df)} 个基因)")

    # ========== 3. 邻居ISM分析 - 只对高表达中预测最准的基因 ==========
    print("\n4️⃣ 邻居ISM分析 - 只对高表达中预测最准的基因...")

    # 筛选高表达基因（表达量 >= 阈值）
    high_expr_df = pred_df[pred_df['true_expression'] >= EXPR_THRESHOLD]

    if len(high_expr_df) == 0:
        print(f"❌ 没有找到表达量≥{EXPR_THRESHOLD}的高表达基因")
        return

    print(f"   测试集中高表达基因数: {len(high_expr_df)}")

    # 按预测误差排序，选取最准的前N_TOP_GENES个
    high_expr_sorted = high_expr_df.sort_values('abs_error')
    target_genes_df = high_expr_sorted.head(N_TOP_GENES)

    print(f"\n   ✅ 选中 {len(target_genes_df)} 个高表达基因进行ISM分析:")
    print(
        f"      表达量范围: [{target_genes_df['true_expression'].min():.2f}, {target_genes_df['true_expression'].max():.2f}]")
    print(f"      预测误差范围: [{target_genes_df['abs_error'].min():.6f}, {target_genes_df['abs_error'].max():.6f}]")
    print(f"      平均预测误差: {target_genes_df['abs_error'].mean():.6f}")
    print(f"      中位数预测误差: {target_genes_df['abs_error'].median():.6f}")

    # 保存选中的基因列表
    target_genes_df[['gene_id', 'true_expression', 'predicted_expression', 'abs_error']].to_csv(
        os.path.join(OUTPUT_DIR, 'selected_high_expr_genes_for_ism.csv'), index=False
    )

    # 构建邻居字典（使用三个网络的并集）
    all_neighbors_dict = defaultdict(set)

    for edge_idx, net_name in [(ppi_edge_index, 'ppi'), (tf_edge_index, 'tf'), (gcn_edge_index, 'gcn')]:
        if edge_idx is not None and edge_idx.numel() > 0:
            for i in range(edge_idx.shape[1]):
                s, d = edge_idx[0, i].item(), edge_idx[1, i].item()
                if s < num_nodes and d < num_nodes:
                    all_neighbors_dict[s].add(d)
                    all_neighbors_dict[d].add(s)

    def get_combined_neighbors(gene_idx):
        neighbors = list(all_neighbors_dict.get(gene_idx, []))
        if len(neighbors) == 0:
            neighbors = [gene_idx]
        if len(neighbors) > MAX_NEIGHBORS * 2:  # 允许更多邻居
            random.seed(args.seed + gene_idx)
            neighbors = random.sample(neighbors, MAX_NEIGHBORS * 2)
        return neighbors

    # 初始化分析器
    analyzer = NeighborISMAnalyzer(
        model, all_embeddings, num_nodes,
        ppi_edge_index, tf_edge_index, gcn_edge_index
    )

    # 分析每个目标基因
    all_neighbor_info = []
    successful_analysis = 0
    failed_genes = []

    for _, row in tqdm(target_genes_df.iterrows(), total=len(target_genes_df), desc="ISM分析"):
        gene_id = row['gene_id']
        baseline_pred = row['predicted_expression']
        true_expr = row['true_expression']
        center_idx = gene_to_idx.get(gene_id)

        if center_idx is None:
            failed_genes.append((gene_id, "索引不存在"))
            continue

        neighbor_indices = get_combined_neighbors(center_idx)
        neighbor_ids = [gene_ids[i] for i in neighbor_indices if i < len(gene_ids)]
        neighbor_indices = neighbor_indices[:len(neighbor_ids)]

        if len(neighbor_indices) == 0:
            failed_genes.append((gene_id, "无邻居"))
            continue

        try:
            importance_scores = analyzer.compute_neighbor_importance(
                center_idx, neighbor_indices, baseline_pred
            )

            # 保存结果
            neighbor_df = pd.DataFrame({
                'neighbor_id': neighbor_ids,
                'importance_score': importance_scores,
                'rank': range(1, len(importance_scores) + 1)
            }).sort_values('importance_score', ascending=False)

            # 添加元信息
            neighbor_df['target_expr'] = true_expr
            neighbor_df['target_pred'] = baseline_pred
            neighbor_df['target_error'] = row['abs_error']

            neighbor_df.to_csv(
                os.path.join(NEIGHBOR_ISM_DIR, f"{gene_id}.csv"),
                index=False
            )

            # 记录信息
            for nid, score in zip(neighbor_ids, importance_scores):
                all_neighbor_info.append({
                    'target_gene': gene_id,
                    'target_expr': true_expr,
                    'target_pred': baseline_pred,
                    'target_error': row['abs_error'],
                    'neighbor_id': nid,
                    'importance': score
                })

            successful_analysis += 1

        except Exception as e:
            failed_genes.append((gene_id, str(e)))
            continue

    analyzer.clear_cache()

    print(f"\n   ✅ 成功分析 {successful_analysis}/{len(target_genes_df)} 个基因")
    if failed_genes:
        print(f"   ⚠️ 失败基因: {len(failed_genes)}个")
        with open(os.path.join(OUTPUT_DIR, 'failed_genes.txt'), 'w') as f:
            for gid, reason in failed_genes:
                f.write(f"{gid}\t{reason}\n")

    # ========== 4. 汇总统计 ==========
    print("\n5️⃣ 生成汇总统计...")

    if all_neighbor_info:
        neighbor_df = pd.DataFrame(all_neighbor_info)

        # 总体重要邻居排名
        top_neighbors = neighbor_df.groupby('neighbor_id')['importance'].agg(['mean', 'count', 'std']).round(6)
        top_neighbors.columns = ['mean_importance', 'frequency', 'std_importance']
        top_neighbors = top_neighbors.sort_values('frequency', ascending=False).head(50)
        top_neighbors.to_csv(os.path.join(OUTPUT_DIR, 'top_regulatory_neighbors.csv'))
        print(f"   💾 保存了Top 50调控邻居")

        # 按重要性分数排序的Top邻居
        top_by_importance = neighbor_df.groupby('neighbor_id')['importance'].mean().sort_values(ascending=False).head(
            50)
        top_by_importance.to_csv(os.path.join(OUTPUT_DIR, 'top_neighbors_by_importance.csv'))
        print(f"   💾 保存了按重要性分数排序的Top 50邻居")

        # 统计分析
        stats = {
            'total_edges_analyzed': len(neighbor_df),
            'unique_neighbors': neighbor_df['neighbor_id'].nunique(),
            'unique_targets': neighbor_df['target_gene'].nunique(),
            'avg_importance': neighbor_df['importance'].mean(),
            'std_importance': neighbor_df['importance'].std(),
            'max_importance': neighbor_df['importance'].max(),
            'min_importance': neighbor_df['importance'].min(),
            'median_importance': neighbor_df['importance'].median(),
        }

        with open(os.path.join(OUTPUT_DIR, 'ism_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"   💾 保存了ISM统计信息")

    # 保存分析汇总表
    summary_df = pd.DataFrame([{
        'gene_id': row['gene_id'],
        'true_expression': row['true_expression'],
        'predicted_expression': row['predicted_expression'],
        'prediction_error': row['abs_error'],
        'rank': idx + 1
    } for idx, (_, row) in enumerate(target_genes_df.iterrows())])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'analysis_summary.csv'), index=False)

    # 保存配置
    config = vars(args)
    config['device'] = DEVICE
    config['n_top_genes'] = N_TOP_GENES
    config['expr_threshold'] = EXPR_THRESHOLD
    config['ism_method'] = ISM_METHOD
    config['fill_strategy'] = FILL_STRATEGY
    config['max_neighbors'] = MAX_NEIGHBORS
    config['ppi_path'] = PPI_PATH
    config['tf_path'] = TF_PATH
    config['gcn_path'] = GCN_PATH
    config['embedding_file'] = EMBEDDING_FILE
    config['labels_file'] = LABELS_FILE

    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 保存测试集性能
    test_results = {
        'seed': args.seed,
        'r2': float(test_r2),
        'pearson': float(test_pearson),
        'spearman': float(test_spearman),
        'rmse': float(test_rmse),
        'fusion_weights': {
            'ppi': float(fusion_weights[0].item()),
            'tf': float(fusion_weights[1].item()),
            'gcn': float(fusion_weights[2].item())
        },
        'num_train': len(train_genes),
        'num_val': len(val_genes),
        'num_test': len(test_genes),
        'num_high_expr_test': len(high_expr_df),
        'num_ism_analyzed': successful_analysis
    }
    with open(os.path.join(OUTPUT_DIR, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)

    print("\n" + "=" * 80)
    print("✨ 分析完成！")
    print(f"\n📊 分析统计:")
    print(f"   - 测试集基因总数: {len(pred_df)}")
    print(f"   - 测试集高表达基因数: {len(high_expr_df)} (表达量≥{EXPR_THRESHOLD})")
    print(f"   - ISM分析基因数: {successful_analysis} (预测最准的前{N_TOP_GENES}个高表达基因)")
    print(f"\n📁 输出目录: {OUTPUT_DIR}")
    print(f"   - 模型: {MODEL_DIR}")
    print(f"   - 预测: {PRED_DIR}")
    print(f"   - 邻居ISM: {NEIGHBOR_ISM_DIR} (每个基因单独文件)")
    print(f"   - Top调控邻居: top_regulatory_neighbors.csv")
    print(f"   - 按重要性排序: top_neighbors_by_importance.csv")
    print(f"   - 选中基因列表: selected_high_expr_genes_for_ism.csv")
    print(f"   - 分析汇总: analysis_summary.csv")
    print(f"   - 测试结果: test_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()