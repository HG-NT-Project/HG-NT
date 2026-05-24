"""
多种子实验版本 (Human/Mouse)
保留模型: M1(Base MLP), M2(Unified Graph), M3(Multi-Graph Weighted Sum - 可学习权重融合)
支持5次随机种子训练，输出均值和标准差
修复版本：修复M2模型逻辑错误、数据泄露问题、边去重、数据划分和GCN归一化
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

# 默认5个随机种子
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# 文件路径配置
PPI_PATH_TEMPLATE = "processed_ppi/{species}_ppi_edge_index.pt"
TF_PATH_TEMPLATE = "processed_tf/{species}_tf_edge_index.pt"
GCN_PATH_TEMPLATE = "processed_gcn/{species}_gcn_network_aligned.pt"
EMBEDDING_DIR = "processed_features"
LABELS_DIR = "processed_labels"


# =================================================================
# 深度回归头 (统一使用)
# =================================================================

class DeepRegressor(nn.Module):
    """通用深度回归头 (in -> 512 -> 256 -> 128 -> 1)"""

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
# M1: Base MLP (纯序列基准)
# =================================================================

class ModelM1_MLP(nn.Module):
    """M1: 纯MLP基准模型 - 不引入任何图信息"""

    def __init__(self, input_dim=2560, dropout=0.3):
        super(ModelM1_MLP, self).__init__()
        self.regressor = DeepRegressor(input_dim, dropout)

    def forward(self, x, edge_index=None):
        return self.regressor(x).squeeze(-1), {}

    def set_subgraphs(self, *args, **kwargs):
        """兼容性方法 - M1不使用子图"""
        pass


# =================================================================
# M2: Unified Graph (整图融合) - 修复版本
# =================================================================

class ModelM2_UnifiedGraph(nn.Module):
    """M2: 整图融合 - 验证引入图信息的价值"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM2_UnifiedGraph, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 修复3: 设置normalize=True，启用对称归一化
        self.conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.unified_edge_index = None
        self.edge_weight = None

    def set_unified_graph(self, unified_edge_index):
        """M2: 直接设置合并图，并使用coalesce规范化"""
        if unified_edge_index is not None and unified_edge_index.numel() > 0:
            # 使用coalesce处理重复边和双向边规范化
            self.unified_edge_index, self.edge_weight = coalesce(
                unified_edge_index,
                None,
                reduce='mean'
            )
        else:
            self.unified_edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_weight = None

    def set_subgraphs(self, ppi_sub, tf_sub, gcn_sub):
        """M2: 兼容旧接口，但实际使用合并图"""
        if ppi_sub is not None and ppi_sub.numel() > 0:
            self.set_unified_graph(ppi_sub)
        elif tf_sub is not None and tf_sub.numel() > 0:
            self.set_unified_graph(tf_sub)
        elif gcn_sub is not None and gcn_sub.numel() > 0:
            self.set_unified_graph(gcn_sub)
        else:
            self.unified_edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_weight = None

    def forward(self, x, edge_index=None):
        s_feat = self.seq_proj(x)

        if self.unified_edge_index is not None and self.unified_edge_index.numel() > 0:
            if self.edge_weight is not None:
                graph_feat = F.elu(self.conv(s_feat, self.unified_edge_index, self.edge_weight))
            else:
                graph_feat = F.elu(self.conv(s_feat, self.unified_edge_index))
        else:
            graph_feat = s_feat

        # 残差连接
        final_feat = s_feat + graph_feat
        return self.regressor(final_feat).squeeze(-1), {}


# =================================================================
# M3: Multi-Graph Weighted Sum (可学习权重融合)
# =================================================================

class ModelM3_MultiGraphConcat(nn.Module):
    """M3: 分图加权和 - 可学习权重融合 (Softmax归一化)"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3_MultiGraphConcat, self).__init__()
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
        if self.ppi_sub is not None and self.ppi_sub.numel() > 0:
            if self.ppi_weight is not None:
                p_info = F.elu(self.ppi_conv(s_feat, self.ppi_sub, self.ppi_weight))
            else:
                p_info = F.elu(self.ppi_conv(s_feat, self.ppi_sub))
        else:
            p_info = s_feat

        if self.tf_sub is not None and self.tf_sub.numel() > 0:
            if self.tf_weight is not None:
                t_info = F.elu(self.tf_conv(s_feat, self.tf_sub, self.tf_weight))
            else:
                t_info = F.elu(self.tf_conv(s_feat, self.tf_sub))
        else:
            t_info = s_feat

        if self.gcn_sub is not None and self.gcn_sub.numel() > 0:
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
            "aggregation_type": "GCN_weighted_sum",
            "fusion_type": "learnable_weighted_sum_softmax",
            "ppi_weight": ppi_w.item(),
            "tf_weight": tf_w.item(),
            "gcn_weight": gcn_w.item(),
            "fusion_gate": ppi_w.item()  # 用PPI的权重作为融合门控指标
        }

        return self.regressor(final_feat).squeeze(-1), weights


# =================================================================
# 模型工厂函数
# =================================================================

def build_model(model_name, input_dim=2560, dropout=0.3, **kwargs):
    """模型工厂 - 支持M1, M2, M3"""

    if model_name == 'm1':
        return ModelM1_MLP(input_dim=input_dim, dropout=dropout)
    elif model_name == 'm2':
        return ModelM2_UnifiedGraph(input_dim=input_dim, dropout=dropout)
    elif model_name == 'm3':
        return ModelM3_MultiGraphConcat(input_dim=input_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}. 可选: m1, m2, m3")


# =================================================================
# 固定随机种子函数
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


# =================================================================
# 加载预提取的NT embeddings
# =================================================================

def load_nt_embeddings(species):
    filename = f"{species}_nt_embeddings.pt"
    file_path = os.path.join(EMBEDDING_DIR, filename)

    if not os.path.exists(file_path):
        print(f"❌ NT embeddings文件不存在: {file_path}")
        return None

    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        print(f"✅ 成功加载NT embeddings: {file_path}")
        print(f"   embeddings形状: {data['x'].shape}")
        print(f"   基因数: {len(data['gene_ids'])}")

        return {
            'embeddings': data['x'],
            'gene_ids': data['gene_ids'],
            'species': data.get('species', species)
        }
    except Exception as e:
        print(f"❌ 加载NT embeddings失败: {e}")
        return None


# =================================================================
# 加载表达量数据
# =================================================================

def load_expression_data(species):
    filename = f"{species}_labels.pt"
    file_path = os.path.join(LABELS_DIR, filename)

    if not os.path.exists(file_path):
        print(f"❌ 标签文件不存在: {file_path}")
        return None, None

    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        labels = data['labels']
        gene_ids = data['gene_id']

        print(f"✅ 成功加载标签文件: {file_path}")
        print(f"   标签形状: {labels.shape}")
        print(f"   基因数: {len(gene_ids)}")

        expr_dict = {}
        for i, gene_id in enumerate(gene_ids):
            expr_dict[gene_id] = labels[i].item()

        return expr_dict, set(gene_ids)

    except Exception as e:
        print(f"❌ 加载标签文件失败: {e}")
        return None, None


# =================================================================
# 加载网络文件 - 修复版
# =================================================================

def load_network(species, network_type, num_nodes=None):
    templates = {'ppi': PPI_PATH_TEMPLATE, 'tf': TF_PATH_TEMPLATE, 'gcn': GCN_PATH_TEMPLATE}
    path = templates[network_type].format(species=species)

    if not os.path.exists(path):
        print(f"⚠️ {network_type.upper()} 网络未发现: {path}")
        return torch.zeros((2, 0), dtype=torch.long)

    try:
        data = torch.load(path, map_location='cpu', weights_only=False)

        # 修复：正确处理Tensor的布尔判断
        if isinstance(data, dict):
            if 'edge_index' in data:
                edge_index = data['edge_index']
            elif 'edges' in data:
                edge_index = data['edges']
            else:
                print(f"⚠️ {network_type.upper()} 网络字典格式不支持，可用键: {list(data.keys())}")
                return torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = data

        # 确保edge_index是2维且第一维为2
        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        # 确保是LongTensor
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        # 过滤超出范围的节点
        if num_nodes is not None and edge_index.numel() > 0:
            max_idx = edge_index.max().item()
            if max_idx >= num_nodes:
                valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                original_edges = edge_index.shape[1]
                edge_index = edge_index[:, valid_mask]
                if original_edges - edge_index.shape[1] > 0:
                    print(f"     过滤了 {original_edges - edge_index.shape[1]} 条非法边")

        print(f"   {network_type.upper()} 网络成功加载. 边数: {edge_index.shape[1]}")
        return edge_index

    except Exception as e:
        print(f"❌ 加载 {network_type.upper()} 网络失败: {e}")
        return torch.zeros((2, 0), dtype=torch.long)


# =================================================================
# 构建表达值张量
# =================================================================

def build_expression_tensor(gene_ids, expr_dict):
    expression_values = []
    for gene_id in gene_ids:
        expression_values.append(expr_dict.get(gene_id, float('nan')))
    return torch.tensor(expression_values, dtype=torch.float32)


# =================================================================
# 数据集类 - 修复数据泄露问题
# =================================================================

class GATDeepCREDataset(Dataset):
    def __init__(self, embeddings, ppi_edge_index, tf_edge_index, gcn_edge_index,
                 expression_values, gene_ids, target_genes, fixed_neighbors=None, fixed_seed=42):
        """
        fixed_neighbors: 预计算的固定邻居字典，如果不提供则重新计算
        fixed_seed: 固定种子确保一致性
        """
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
                expr_val = self.expression_tensor[idx]
                has_expr = not torch.isnan(expr_val)
                self.valid_expression_mask.append(has_expr)

        print(f"\n📊 数据集构建统计:")
        print(f"   有效基因数量: {len(self.valid_genes)}/{len(target_genes)}")
        print(f"   有表达值的有效基因: {sum(self.valid_expression_mask)}")

        # 如果提供了预计算的邻居，直接使用；否则计算
        if fixed_neighbors is not None:
            print("   ✅ 使用预计算的固定邻居集合")
            self.all_neighbors = fixed_neighbors
        else:
            print("   🆕 首次计算邻居集合（将用于整个实验）")
            self.all_neighbors = self._precompute_all_neighbors(fixed_seed)

    def _precompute_all_neighbors(self, fixed_seed=42):
        """预计算所有节点的邻居 - 使用固定种子确保一致性"""
        print("\n   🔍 预计算所有节点的多网络邻居...")

        adj_dict = {
            'ppi': defaultdict(list),
            'tf': defaultdict(list),
            'gcn': defaultdict(list)
        }

        if self.ppi_edge_index is not None and self.ppi_edge_index.numel() > 0:
            for i in range(self.ppi_edge_index.shape[1]):
                src = self.ppi_edge_index[0, i].item()
                tgt = self.ppi_edge_index[1, i].item()
                if src < self.num_nodes and tgt < self.num_nodes:
                    adj_dict['ppi'][src].append(tgt)
                    adj_dict['ppi'][tgt].append(src)

        if self.tf_edge_index is not None and self.tf_edge_index.numel() > 0:
            for i in range(self.tf_edge_index.shape[1]):
                src = self.tf_edge_index[0, i].item()
                tgt = self.tf_edge_index[1, i].item()
                if src < self.num_nodes and tgt < self.num_nodes:
                    adj_dict['tf'][src].append(tgt)
                    adj_dict['tf'][tgt].append(src)

        if self.gcn_edge_index is not None and self.gcn_edge_index.numel() > 0:
            for i in range(self.gcn_edge_index.shape[1]):
                src = self.gcn_edge_index[0, i].item()
                tgt = self.gcn_edge_index[1, i].item()
                if src < self.num_nodes and tgt < self.num_nodes:
                    adj_dict['gcn'][src].append(tgt)
                    adj_dict['gcn'][tgt].append(src)

        all_neighbors = {}
        random.seed(fixed_seed)

        for node_idx in range(self.num_nodes):
            neighbors_dict = {}

            for net_name in ['ppi', 'tf', 'gcn']:
                neighbors = adj_dict[net_name].get(node_idx, [])
                if neighbors:
                    if len(neighbors) > MAX_NEIGHBORS:
                        sampled_neighbors = random.sample(neighbors, MAX_NEIGHBORS)
                        neighbors_dict[net_name] = torch.tensor(sampled_neighbors, dtype=torch.long)
                    else:
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

        neighbor_dict = self.all_neighbors[gene_idx]

        return {
            'gene_id': gene_id,
            'gene_idx': gene_idx,
            'neighbor_indices': neighbor_dict,
            'expression': torch.tensor(expression, dtype=torch.float32),
            'has_expression': not torch.isnan(expression)
        }


def gat_collate_fn(batch):
    gene_ids = [item['gene_id'] for item in batch]
    gene_indices = torch.LongTensor([item['gene_idx'] for item in batch])
    has_expression = torch.BoolTensor([item['has_expression'] for item in batch])

    neighbor_indices = [item['neighbor_indices'] for item in batch]
    expressions = torch.stack([item['expression'] for item in batch])

    return {
        'gene_ids': gene_ids,
        'gene_indices': gene_indices,
        'neighbor_indices': neighbor_indices,
        'expressions': expressions,
        'has_expression': has_expression
    }


# =================================================================
# GAT训练器
# =================================================================

class GATDeepCRETrainer:
    def __init__(self, model, model_name, device='cpu', learning_rate=1e-4,
                 patience=15, min_lr=1e-6, seed=42):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.seed = seed

        self.all_embeddings = None
        self.ppi_edge_index = None
        self.tf_edge_index = None
        self.gcn_edge_index = None
        self.num_nodes = 0

        set_seed(seed)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-3,
            betas=(0.9, 0.999)
        )

        self.criterion = nn.HuberLoss(reduction='none', delta=1.0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=min_lr
        )

        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        self.best_epoch = 0

        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        self.fusion_gate_history = []

        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def set_graph_data(self, all_embeddings, ppi_edge_index=None, tf_edge_index=None, gcn_edge_index=None):
        self.all_embeddings = all_embeddings
        self.num_nodes = len(all_embeddings) if all_embeddings is not None else 0

        if ppi_edge_index is not None and self.num_nodes > 0:
            valid_mask = (ppi_edge_index[0] < self.num_nodes) & (ppi_edge_index[1] < self.num_nodes)
            self.ppi_edge_index = ppi_edge_index[:, valid_mask]
        else:
            self.ppi_edge_index = ppi_edge_index

        if tf_edge_index is not None and self.num_nodes > 0:
            valid_mask = (tf_edge_index[0] < self.num_nodes) & (tf_edge_index[1] < self.num_nodes)
            self.tf_edge_index = tf_edge_index[:, valid_mask]
        else:
            self.tf_edge_index = tf_edge_index

        if gcn_edge_index is not None and self.num_nodes > 0:
            valid_mask = (gcn_edge_index[0] < self.num_nodes) & (gcn_edge_index[1] < self.num_nodes)
            self.gcn_edge_index = gcn_edge_index[:, valid_mask]
        else:
            self.gcn_edge_index = gcn_edge_index

        print(f"   📊 最终图数据状态:")
        print(f"      PPI 边数: {self.ppi_edge_index.shape[1] if self.ppi_edge_index is not None else 0}")
        print(f"      TF 边数: {self.tf_edge_index.shape[1] if self.tf_edge_index is not None else 0}")
        print(f"      GCN 边数: {self.gcn_edge_index.shape[1] if self.gcn_edge_index is not None else 0}")

    def _extract_subgraphs(self, unique_nodes, device):
        """提取三个分离的子图"""
        unique_nodes_cpu = unique_nodes.cpu()

        if self.ppi_edge_index is not None and self.ppi_edge_index.numel() > 0:
            ppi_sub, _ = subgraph(
                unique_nodes_cpu,
                self.ppi_edge_index.cpu(),
                relabel_nodes=False,
                num_nodes=self.num_nodes
            )
            ppi_sub = ppi_sub.to(device)
        else:
            ppi_sub = torch.zeros((2, 0), device=device, dtype=torch.long)

        if self.tf_edge_index is not None and self.tf_edge_index.numel() > 0:
            tf_sub, _ = subgraph(
                unique_nodes_cpu,
                self.tf_edge_index.cpu(),
                relabel_nodes=False,
                num_nodes=self.num_nodes
            )
            tf_sub = tf_sub.to(device)
        else:
            tf_sub = torch.zeros((2, 0), device=device, dtype=torch.long)

        if self.gcn_edge_index is not None and self.gcn_edge_index.numel() > 0:
            gcn_sub, _ = subgraph(
                unique_nodes_cpu,
                self.gcn_edge_index.cpu(),
                relabel_nodes=False,
                num_nodes=self.num_nodes
            )
            gcn_sub = gcn_sub.to(device)
        else:
            gcn_sub = torch.zeros((2, 0), device=device, dtype=torch.long)

        return ppi_sub, tf_sub, gcn_sub

    def _extract_unified_subgraph(self, unique_nodes, device):
        """提取合并后的子图（用于M2模型）"""
        unique_nodes_cpu = unique_nodes.cpu()

        # 合并所有网络
        unified_full = None
        if self.ppi_edge_index is not None and self.ppi_edge_index.numel() > 0:
            unified_full = self.ppi_edge_index
        if self.tf_edge_index is not None and self.tf_edge_index.numel() > 0:
            if unified_full is None:
                unified_full = self.tf_edge_index
            else:
                unified_full = torch.cat([unified_full, self.tf_edge_index], dim=1)
        if self.gcn_edge_index is not None and self.gcn_edge_index.numel() > 0:
            if unified_full is None:
                unified_full = self.gcn_edge_index
            else:
                unified_full = torch.cat([unified_full, self.gcn_edge_index], dim=1)

        if unified_full is None:
            return torch.zeros((2, 0), device=device, dtype=torch.long)

        if unified_full.numel() > 0:
            unified_full, _ = coalesce(unified_full, None, reduce='mean')
            unified_sub, _ = subgraph(
                unique_nodes_cpu,
                unified_full.cpu(),
                relabel_nodes=False,
                num_nodes=self.num_nodes
            )
            return unified_sub.to(device)
        else:
            return torch.zeros((2, 0), device=device, dtype=torch.long)

    def _prepare_batch_for_model(self, batch):
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
        valid_mask = unique_nodes < self.num_nodes
        unique_nodes = unique_nodes[valid_mask]

        if len(unique_nodes) == 0:
            return {
                'x': torch.zeros((1, self.all_embeddings.size(1)), device=self.device),
                'target_local_indices': torch.tensor([], device=self.device, dtype=torch.long),
                'expressions': expressions,
                'has_expression': torch.zeros_like(has_expression)
            }

        x = self.all_embeddings[unique_nodes.cpu()].to(self.device)

        # 建立严格的局部映射表
        node_mapping = {node.item(): i for i, node in enumerate(unique_nodes)}

        # 根据模型类型提取不同的子图
        if self.model_name == 'm2':
            unified_sub_raw = self._extract_unified_subgraph(unique_nodes, self.device)

            if unified_sub_raw.numel() > 0:
                src_local = torch.tensor([node_mapping[n.item()] for n in unified_sub_raw[0]], device=self.device)
                tgt_local = torch.tensor([node_mapping[n.item()] for n in unified_sub_raw[1]], device=self.device)
                unified_sub = torch.stack([src_local, tgt_local], dim=0)
            else:
                unified_sub = torch.zeros((2, 0), device=self.device, dtype=torch.long)

            if hasattr(self.model, 'set_unified_graph'):
                self.model.set_unified_graph(unified_sub)
            else:
                self.model.set_subgraphs(unified_sub, unified_sub, unified_sub)
        else:
            ppi_sub_raw, tf_sub_raw, gcn_sub_raw = self._extract_subgraphs(unique_nodes, self.device)

            # 手动映射三个子图
            if ppi_sub_raw.numel() > 0:
                src = torch.tensor([node_mapping[n.item()] for n in ppi_sub_raw[0]], device=self.device)
                tgt = torch.tensor([node_mapping[n.item()] for n in ppi_sub_raw[1]], device=self.device)
                ppi_sub = torch.stack([src, tgt], dim=0)
            else:
                ppi_sub = torch.zeros((2, 0), device=self.device, dtype=torch.long)

            if tf_sub_raw.numel() > 0:
                src = torch.tensor([node_mapping[n.item()] for n in tf_sub_raw[0]], device=self.device)
                tgt = torch.tensor([node_mapping[n.item()] for n in tf_sub_raw[1]], device=self.device)
                tf_sub = torch.stack([src, tgt], dim=0)
            else:
                tf_sub = torch.zeros((2, 0), device=self.device, dtype=torch.long)

            if gcn_sub_raw.numel() > 0:
                src = torch.tensor([node_mapping[n.item()] for n in gcn_sub_raw[0]], device=self.device)
                tgt = torch.tensor([node_mapping[n.item()] for n in gcn_sub_raw[1]], device=self.device)
                gcn_sub = torch.stack([src, tgt], dim=0)
            else:
                gcn_sub = torch.zeros((2, 0), device=self.device, dtype=torch.long)

            if hasattr(self.model, 'set_subgraphs'):
                self.model.set_subgraphs(ppi_sub, tf_sub, gcn_sub)

        # 构建目标节点局部索引
        target_local_indices = []
        valid_indices_mask = []
        for idx in gene_indices:
            idx_item = idx.item()
            if idx_item in node_mapping:
                target_local_indices.append(node_mapping[idx_item])
                valid_indices_mask.append(True)
            else:
                target_local_indices.append(0)
                valid_indices_mask.append(False)

        target_local_indices = torch.tensor(target_local_indices, device=self.device)
        valid_indices_mask = torch.tensor(valid_indices_mask, device=self.device)

        return {
            'x': x,
            'target_local_indices': target_local_indices,
            'expressions': expressions,
            'has_expression': has_expression & valid_indices_mask
        }

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        num_valid_samples = 0
        all_preds = []
        all_targets = []

        epoch_gate_means = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)
        for batch_idx, batch in enumerate(pbar):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                prepared = self._prepare_batch_for_model(batch)

                x = prepared['x']
                target_local_indices = prepared['target_local_indices']
                expressions = prepared['expressions']
                has_expression = prepared['has_expression']

                if not has_expression.any():
                    continue

                if target_local_indices.numel() == 0:
                    continue

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs, weights = self.model(x)

                    if target_local_indices.numel() > 0:
                        outputs = outputs[target_local_indices]
                    outputs = outputs.squeeze()

                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)

                    valid_outputs = outputs[has_expression]
                    valid_targets = expressions[has_expression]

                    if len(valid_outputs) == 0:
                        continue

                    loss_values = self.criterion(valid_outputs, valid_targets)
                    loss = loss_values.mean()

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item() * has_expression.sum().item()
                num_valid_samples += has_expression.sum().item()

                all_preds.extend(valid_outputs.detach().cpu().numpy())
                all_targets.extend(valid_targets.cpu().numpy())

                if weights and 'fusion_gate' in weights:
                    gate_val = weights['fusion_gate']
                    if isinstance(gate_val, torch.Tensor):
                        if gate_val.numel() == 1:
                            epoch_gate_means.append(gate_val.item())
                        else:
                            epoch_gate_means.append(gate_val.mean().item())
                    else:
                        epoch_gate_means.append(gate_val)

                pbar.set_postfix({'loss': loss.item(), 'valid': has_expression.sum().item()})

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    GPU OOM at batch {batch_idx}, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e

        avg_loss = total_loss / num_valid_samples if num_valid_samples > 0 else float('inf')
        train_pearson = pearsonr(all_preds, all_targets)[0] if len(all_preds) > 1 else 0.0

        if epoch_gate_means:
            self.fusion_gate_history.append(np.mean(epoch_gate_means))

        return avg_loss, train_pearson

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_valid_samples = 0
        all_preds = []
        all_targets = []
        all_gene_ids = []

        val_gate_means = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in pbar:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                prepared = self._prepare_batch_for_model(batch)

                x = prepared['x']
                target_local_indices = prepared['target_local_indices']
                expressions = prepared['expressions']
                has_expression = prepared['has_expression']

                if not has_expression.any():
                    continue

                if target_local_indices.numel() == 0:
                    continue

                outputs, weights = self.model(x)

                if target_local_indices.numel() > 0:
                    outputs = outputs[target_local_indices]
                outputs = outputs.squeeze()

                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)

                valid_outputs = outputs[has_expression]
                valid_targets = expressions[has_expression]

                if len(valid_outputs) == 0:
                    continue

                loss_values = self.criterion(valid_outputs, valid_targets)
                loss = loss_values.mean()

                total_loss += loss.item() * has_expression.sum().item()
                num_valid_samples += has_expression.sum().item()

                all_preds.extend(valid_outputs.cpu().numpy())
                all_targets.extend(valid_targets.cpu().numpy())

                if weights and 'fusion_gate' in weights:
                    gate_val = weights['fusion_gate']
                    if isinstance(gate_val, torch.Tensor):
                        if gate_val.numel() == 1:
                            val_gate_means.append(gate_val.item())
                        else:
                            val_gate_means.append(gate_val.mean().item())
                    else:
                        val_gate_means.append(gate_val)

                batch_gene_ids = batch['gene_ids']
                for i, valid in enumerate(has_expression.cpu().numpy()):
                    if valid:
                        all_gene_ids.append(batch_gene_ids[i])

        avg_loss = total_loss / num_valid_samples if num_valid_samples > 0 else float('inf')

        if val_gate_means and self.model_name in ['m3']:
            print(f"   🚪 门控平均值: {np.mean(val_gate_means):.4f}")

        return avg_loss, np.array(all_preds), np.array(all_targets), all_gene_ids

    def train(self, train_loader, val_loader, epochs=100):
        for epoch in range(epochs):
            train_loss, train_pearson = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            val_loss, val_preds, val_targets, val_gene_ids = self.validate(val_loader)
            self.val_losses.append(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.counter = 0
                self.best_model_state = self.model.state_dict().copy()
                print(f"  ✅ Epoch {epoch + 1}: 新的最佳验证损失 {val_loss:.6f}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"  🚨 Early stopping at epoch {epoch + 1}")
                    print(f"     Best val loss: {self.best_loss:.6f} at epoch {self.best_epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  📊 Epoch {epoch + 1}/{epochs}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.6f}, "
                      f"Train Pearson: {train_pearson:.4f}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_loss,
            'seed': self.seed,
            'fusion_gate_history': self.fusion_gate_history
        }


# =================================================================
# 评估函数
# =================================================================

def evaluate_regression(y_true, y_pred):
    results = {}

    mse = mean_squared_error(y_true, y_pred)
    results['mse'] = float(mse)
    results['rmse'] = float(np.sqrt(mse))
    results['mae'] = float(mean_absolute_error(y_true, y_pred))
    results['r2'] = float(r2_score(y_true, y_pred))

    if len(y_true) > 1:
        try:
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
            results['pearson_corr'] = float(pearson_corr)
            results['pearson_p'] = float(pearson_p)
        except:
            results['pearson_corr'] = 0.0
            results['pearson_p'] = 1.0

        try:
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)
            results['spearman_corr'] = float(spearman_corr)
            results['spearman_p'] = float(spearman_p)
        except:
            results['spearman_corr'] = 0.0
            results['spearman_p'] = 1.0

        residuals = y_pred - y_true
        results['residual_mean'] = float(np.mean(residuals))
        results['residual_std'] = float(np.std(residuals))

        residual_var = float(np.var(residuals))
        y_var = float(np.var(y_true))
        results['explained_variance'] = float(1 - residual_var / y_var) if y_var > 0 else 0.0
    else:
        results['pearson_corr'] = 0.0
        results['spearman_corr'] = 0.0

    results['num_samples'] = int(len(y_true))
    return results


# =================================================================
# 单次训练运行（固定种子）
# =================================================================

def run_single_seed(species, model_name, data_dict, args, seed, fixed_neighbors):
    """使用固定种子进行单次训练 - 共享相同的邻居集合"""
    print(f"\n{'=' * 50}")
    print(f"🎲 种子: {seed} - {model_name.upper()}")
    print(f"{'=' * 50}")

    set_seed(seed)

    gene_ids = data_dict['gene_ids']
    total_genes = len(gene_ids)
    indices = list(range(total_genes))

    # 标准化的数据划分
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=TRAIN_RATIO,
        random_state=seed,
        shuffle=True
    )

    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio_adjusted,
        random_state=seed,
        shuffle=True
    )

    train_genes = [gene_ids[i] for i in train_indices]
    val_genes = [gene_ids[i] for i in val_indices]
    test_genes = [gene_ids[i] for i in test_indices]

    print(f"训练集: {len(train_genes)} 个基因")
    print(f"验证集: {len(val_genes)} 个基因")
    print(f"测试集: {len(test_genes)} 个基因")

    # 创建数据集
    train_dataset = GATDeepCREDataset(
        data_dict['embeddings'],
        data_dict['ppi_edge_index'], data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
        data_dict['expression_values'],
        data_dict['gene_ids'],
        train_genes,
        fixed_neighbors=fixed_neighbors
    )

    val_dataset = GATDeepCREDataset(
        data_dict['embeddings'],
        data_dict['ppi_edge_index'], data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
        data_dict['expression_values'],
        data_dict['gene_ids'],
        val_genes,
        fixed_neighbors=fixed_neighbors
    )

    test_dataset = GATDeepCREDataset(
        data_dict['embeddings'],
        data_dict['ppi_edge_index'], data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
        data_dict['expression_values'],
        data_dict['gene_ids'],
        test_genes,
        fixed_neighbors=fixed_neighbors
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=gat_collate_fn,
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=gat_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=gat_collate_fn
    )

    input_dim = data_dict['embeddings'].size(1)
    model = build_model(model_name, input_dim=input_dim)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    trainer = GATDeepCRETrainer(
        model, model_name, device=device,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=seed
    )

    trainer.set_graph_data(
        all_embeddings=data_dict['embeddings'],
        ppi_edge_index=data_dict['ppi_edge_index'],
        tf_edge_index=data_dict['tf_edge_index'],
        gcn_edge_index=data_dict['gcn_edge_index']
    )

    print(f"\n🚀 开始训练...")
    training_history = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 测试集评估
    test_loss, test_preds, test_targets, test_gene_ids = trainer.validate(test_loader)
    test_evaluation = evaluate_regression(test_targets, test_preds)

    print(f"\n  📈 测试结果:")
    print(f"     R²: {test_evaluation['r2']:.6f}")
    print(f"     Pearson: {test_evaluation['pearson_corr']:.6f}")
    print(f"     Spearman: {test_evaluation['spearman_corr']:.6f}")
    print(f"     RMSE: {test_evaluation['rmse']:.6f}")

    result = {
        'seed': seed,
        'model_name': model_name,
        'best_epoch': training_history['best_epoch'] + 1,
        'best_val_loss': float(training_history['best_val_loss']),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'test_r2': test_evaluation['r2'],
        'test_pearson': test_evaluation['pearson_corr'],
        'test_spearman': test_evaluation['spearman_corr'],
        'test_rmse': test_evaluation['rmse'],
        'test_mae': test_evaluation['mae'],
        'model_params': sum(p.numel() for p in model.parameters())
    }

    return result


# =================================================================
# 多种子训练
# =================================================================

def train_multi_seed(species, data_dict, args):
    """使用多个种子训练，每个种子依次训练所有模型"""
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    print(f"\n{'=' * 70}")
    print(f"🚀 多种子训练 - {species.upper()}")
    print(f"种子列表: {seeds}")
    print(f"模型列表: {args.models}")
    print(f"{'=' * 70}")

    # 预计算固定邻居集合
    print("\n🔧 预计算固定邻居集合（用于所有数据集）...")
    temp_dataset = GATDeepCREDataset(
        data_dict['embeddings'],
        data_dict['ppi_edge_index'], data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
        data_dict['expression_values'],
        data_dict['gene_ids'],
        data_dict['gene_ids'][:100],
        fixed_neighbors=None,
        fixed_seed=42
    )
    fixed_neighbors = temp_dataset.all_neighbors
    print("✅ 固定邻居集合计算完成")

    all_results = {model_name: [] for model_name in args.models}

    for seed in seeds:
        print(f"\n{'#' * 60}")
        print(f"# 当前种子: {seed}")
        print(f"{'#' * 60}")

        for model_name in args.models:
            result = run_single_seed(species, model_name, data_dict, args, seed, fixed_neighbors)
            if result:
                all_results[model_name].append(result)

        print(f"\n📊 种子 {seed} 模型比较:")
        print(f"{'模型':<10} {'R²':<12} {'Pearson':<12} {'Spearman':<12} {'RMSE':<12}")
        print(f"{'-' * 58}")
        for model_name in args.models:
            if all_results[model_name]:
                latest_result = all_results[model_name][-1]
                print(f"{model_name.upper():<10} {latest_result['test_r2']:<12.6f} "
                      f"{latest_result['test_pearson']:<12.6f} {latest_result['test_spearman']:<12.6f} "
                      f"{latest_result['test_rmse']:<12.6f}")

    # 汇总统计
    print(f"\n{'=' * 70}")
    print(f"📊 {species.upper()} 多种子汇总结果")
    print(f"{'=' * 70}")

    metrics_keys = ['test_r2', 'test_pearson', 'test_spearman', 'test_rmse']

    for model_name in args.models:
        if not all_results[model_name]:
            continue

        print(f"\n🔬 {model_name.upper()} 模型 ({len(all_results[model_name])} seeds):")
        print(f"{'指标':<12} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
        print(f"{'-' * 60}")

        metrics_summary = {}
        for key in metrics_keys:
            values = [r[key] for r in all_results[model_name]]
            metrics_summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }
            print(f"{key:<12} {np.mean(values):<12.6f} {metrics_summary[key]['std']:<12.6f} "
                  f"{np.min(values):<12.6f} {np.max(values):<12.6f}")

        # 保存结果
        results_df = pd.DataFrame(all_results[model_name])
        results_file = os.path.join(args.output_dir, f'{species}_{model_name}_seed_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 各种子指标已保存: {results_file}")

        summary = {
            'species': species,
            'model_name': model_name,
            'seeds': seeds,
            'num_seeds': len(all_results[model_name]),
            'metrics_summary': metrics_summary
        }

        with open(os.path.join(args.output_dir, f'{species}_{model_name}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"💾 汇总统计已保存: {args.output_dir}/{species}_{model_name}_summary.json")

        print(f"\n📈 {species.upper()} - {model_name.upper()} 性能指标 (均值 ± 标准差):")
        print(f"   R²:       {metrics_summary['test_r2']['mean']:.6f} ± {metrics_summary['test_r2']['std']:.6f}")
        print(
            f"   Pearson:  {metrics_summary['test_pearson']['mean']:.6f} ± {metrics_summary['test_pearson']['std']:.6f}")
        print(
            f"   Spearman: {metrics_summary['test_spearman']['mean']:.6f} ± {metrics_summary['test_spearman']['std']:.6f}")
        print(f"   RMSE:     {metrics_summary['test_rmse']['mean']:.6f} ± {metrics_summary['test_rmse']['std']:.6f}")

    # 模型比较表
    print(f"\n{'=' * 70}")
    print(f"🏆 {species.upper()} 模型性能比较 (均值 ± 标准差)")
    print(f"{'=' * 70}")
    print(f"{'模型':<10} {'R²':<20} {'Pearson':<20} {'Spearman':<20} {'RMSE':<20}")
    print(f"{'-' * 90}")

    for model_name in args.models:
        if not all_results[model_name]:
            continue
        values_r2 = [r['test_r2'] for r in all_results[model_name]]
        values_pearson = [r['test_pearson'] for r in all_results[model_name]]
        values_spearman = [r['test_spearman'] for r in all_results[model_name]]
        values_rmse = [r['test_rmse'] for r in all_results[model_name]]

        print(f"{model_name.upper():<10} {np.mean(values_r2):.6f}±{np.std(values_r2, ddof=1):.6f}   "
              f"{np.mean(values_pearson):.6f}±{np.std(values_pearson, ddof=1):.6f}   "
              f"{np.mean(values_spearman):.6f}±{np.std(values_spearman, ddof=1):.6f}   "
              f"{np.mean(values_rmse):.6f}±{np.std(values_rmse, ddof=1):.6f}")

    return all_results


# =================================================================
# 辅助类
# =================================================================

class DeepGATAblationHelper:
    @staticmethod
    def load_species_data(species):
        print(f"\n🔍 加载 {species} 数据...")

        embed_data = load_nt_embeddings(species)
        if embed_data is None:
            return None

        num_nodes = len(embed_data['gene_ids'])

        expr_dict, _ = load_expression_data(species)
        if expr_dict is None:
            return None

        ppi_edge_index = load_network(species, 'ppi', num_nodes)
        tf_edge_index = load_network(species, 'tf', num_nodes)
        gcn_edge_index = load_network(species, 'gcn', num_nodes)

        expression_values = build_expression_tensor(embed_data['gene_ids'], expr_dict)

        data_dict = {
            'embeddings': embed_data['embeddings'],
            'expression_values': expression_values,
            'gene_ids': embed_data['gene_ids'],
            'ppi_edge_index': ppi_edge_index,
            'tf_edge_index': tf_edge_index,
            'gcn_edge_index': gcn_edge_index,
            'num_genes': num_nodes,
            'num_genes_with_expression': int((~torch.isnan(expression_values)).sum().item())
        }

        print(f"\n📊 数据统计:")
        print(f"   embeddings形状: {data_dict['embeddings'].shape}")
        print(f"   基因总数: {data_dict['num_genes']}")
        print(f"   有表达值基因: {data_dict['num_genes_with_expression']}")

        return data_dict


# =================================================================
# 主函数
# =================================================================

def main():
    parser = argparse.ArgumentParser(description='DeepGAT多种子实验 - Human/Mouse (M1, M2, M3)')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--models', type=str, nargs='+',
                        default=['m1', 'm2', 'm3'],
                        choices=['m1', 'm2', 'm3'])
    parser.add_argument('--output_dir', type=str, default='Results_xr')
    parser.add_argument('--species', type=str, default='all', choices=['human', 'mouse', 'all'])
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)

    args = parser.parse_args()

    print("=" * 80)
    print("🔬 DeepGAT 多种子实验 - Human/Mouse (M1, M2, M3)")
    print("=" * 80)
    print(f"\n🔧 训练配置:")
    print(f"  物种: {args.species}")
    print(f"  模型: {args.models}")
    print(f"  随机种子: {args.seeds}")
    print(f"  输出目录: {args.output_dir}")
    print("=" * 80)

    if not os.path.exists(EMBEDDING_DIR):
        print(f"❌ embeddings目录不存在: {EMBEDDING_DIR}")
        return

    if not os.path.exists(LABELS_DIR):
        print(f"❌ 标签目录不存在: {LABELS_DIR}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    config = {
        'seeds': args.seeds,
        'models': args.models,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'timestamp': datetime.now().isoformat()
    }

    config_file = os.path.join(args.output_dir, 'experiment_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # 确定物种列表
    if args.species == 'all':
        species_list = ['human', 'mouse']
    else:
        species_list = [args.species]

    # 对每个物种进行多种子训练
    for species in species_list:
        print(f"\n{'=' * 60}")
        print(f"🌿 处理物种: {species.upper()}")
        print(f"{'=' * 60}")

        data_dict = DeepGATAblationHelper.load_species_data(species)
        if data_dict is None:
            print(f"❌ 数据加载失败，跳过 {species}")
            continue

        train_multi_seed(species, data_dict, args)

    print(f"\n{'=' * 80}")
    print("✅ 全部训练完成!")
    print(f"   结果保存在: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()