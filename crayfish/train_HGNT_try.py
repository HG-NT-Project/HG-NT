"""
小龙虾基因表达预测 - 多模型消融实验
支持4种模型架构:
- M1: HG-NT_Concat (双路GCN + 拼接融合)
- M2: HG-NT_WeightSum (全局可学习权重 + 加权求和)
- M3: HG-NT_Decoupled (特征空间解耦 + 独立投影层)
- M4: HG-NT_Dynamic (实例级动态权重 + 门控网络)

特性:
- 邻居预计算固定机制 (防止数据泄露)
- 5种随机种子循环
- 自动输出4种模型性能对比表
- 融合权重实时监控
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

# 默认5个随机种子
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# 小龙虾文件路径配置
INDEX_FILE = "gene_id_index.txt"
LABELS_FILE = "crayfish_labels.csv"
EMBEDDING_FILE = "crayfish_embeddings/crayfish_embeddings.pt"

# 网络文件路径
TF_PATH = "processed_tf/crayfish_tf_edge_index.pt"
GCN_PATH = "processed_gcn/crayfish_gcn_network.pt"


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
# 模型架构定义
# =================================================================

class ModelM1_Concat(nn.Module):
    """
    M1: HG-NT_Concat
    双路GCN(开启归一化) + 拼接融合
    """

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM1_Concat, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 双路GCN - 开启归一化
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=True)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=True)
        # 拼接融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

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

        output = self.regressor(final_feat).squeeze(-1)

        # 统一返回格式: (output, weights_dict) - M1使用默认权重
        weights_dict = {'tf_weights': 0.5, 'gcn_weights': 0.5, 'tf_mean': 0.5, 'gcn_mean': 0.5}
        return output, weights_dict


class ModelM2_WeightSum(nn.Module):
    """
    M2: HG-NT_WeightSum
    全局可学习权重 + Softmax + 加权求和
    """

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM2_WeightSum, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 双路GCN - 开启归一化
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=True)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=True)

        # 全局可学习权重参数
        self.logits = nn.Parameter(torch.ones(2) * 0.5)

        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

    def get_weights(self):
        """返回当前融合权重 (TF权重, GCN权重)"""
        weights = F.softmax(self.logits, dim=0)
        return weights[0].item(), weights[1].item()

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

        weights = F.softmax(self.logits, dim=0)
        combined = weights[0] * t_info + weights[1] * g_info
        final_feat = s_feat + combined

        output = self.regressor(final_feat).squeeze(-1)

        # 统一返回格式: (output, weights_dict)
        weights_dict = {
            'tf_weights': weights[0].item(),
            'gcn_weights': weights[1].item(),
            'tf_mean': weights[0].item(),
            'gcn_mean': weights[1].item()
        }
        return output, weights_dict


class ModelM3_Decoupled(nn.Module):
    """
    M3: HG-NT_Decoupled
    特征空间解耦: 独立投影层 + GCN
    """

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3_Decoupled, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 分支独立投影层
        self.tf_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gcn_proj = nn.Linear(hidden_dim, hidden_dim)

        # 双路GCN - 开启归一化
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=True)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=True)

        # 拼接融合 (与M1相同)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

    def forward(self, x, edge_index=None):
        s_feat = self.seq_proj(x)

        tf_feat = self.tf_proj(s_feat)
        gcn_feat = self.gcn_proj(s_feat)

        if self.tf_sub is not None and self.tf_sub.numel() > 0:
            t_info = F.elu(self.tf_conv(tf_feat, self.tf_sub))
        else:
            t_info = tf_feat

        if self.gcn_sub is not None and self.gcn_sub.numel() > 0:
            g_info = F.elu(self.gcn_conv(gcn_feat, self.gcn_sub))
        else:
            g_info = gcn_feat

        concat_feat = torch.cat([t_info, g_info], dim=-1)
        graph_feat = self.fusion(concat_feat)
        final_feat = s_feat + graph_feat

        output = self.regressor(final_feat).squeeze(-1)

        # 统一返回格式: (output, weights_dict) - 使用默认权重
        weights_dict = {'tf_weights': 0.5, 'gcn_weights': 0.5, 'tf_mean': 0.5, 'gcn_mean': 0.5}
        return output, weights_dict


class ModelM4_Dynamic(nn.Module):
    """
    M4: HG-NT_Dynamic
    实例级动态权重: 门控网络生成每个样本的专属权重
    """

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM4_Dynamic, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 双路GCN - 开启归一化
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=True)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=True)

        # 门控网络 (轻量级)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Softmax(dim=-1)
        )

        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

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

        # 动态权重生成
        weights = self.gate(s_feat)  # [N, 2]
        combined = weights[:, 0:1] * t_info + weights[:, 1:2] * g_info
        final_feat = s_feat + combined

        output = self.regressor(final_feat).squeeze(-1)

        # 统一返回格式: (output, weights_dict)
        weights_dict = {
            'tf_weights': weights[:, 0].detach().cpu().numpy(),
            'gcn_weights': weights[:, 1].detach().cpu().numpy(),
            'tf_mean': weights[:, 0].mean().item(),
            'gcn_mean': weights[:, 1].mean().item()
        }
        return output, weights_dict


# =================================================================
# 模型工厂
# =================================================================

MODEL_REGISTRY = {
    'M1': ModelM1_Concat,
    'M2': ModelM2_WeightSum,
    'M3': ModelM3_Decoupled,
    'M4': ModelM4_Dynamic,
}


def create_model(model_name, input_dim=2560, hidden_dim=512, dropout=0.3):
    """模型工厂函数"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)


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


# =================================================================
# 数据加载函数 (与原代码相同)
# =================================================================

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

    try:
        data = torch.load(embedding_file, map_location='cpu', weights_only=False)
        print(f"✅ 成功加载NT embeddings: {embedding_file}")
        print(f"   embeddings形状: {data['x'].shape}")
        print(f"   基因数: {len(data['gene_ids'])}")
        return {
            'embeddings': data['x'],
            'gene_ids': data['gene_ids'],
            'species': data.get('species', 'crayfish')
        }
    except Exception as e:
        print(f"❌ 加载NT embeddings失败: {e}")
        return None


def load_expression_data(labels_file, valid_gene_ids):
    if not os.path.exists(labels_file):
        print(f"❌ 标签文件不存在: {labels_file}")
        return None, None

    df = pd.read_csv(labels_file)
    print(f"✅ 成功加载标签文件: {labels_file}")
    print(f"   标签数据形状: {df.shape}")

    if 'gene_id' not in df.columns:
        possible_id_cols = ['GeneID', 'gene', 'Gene', 'id', 'ID']
        found = False
        for col in possible_id_cols:
            if col in df.columns:
                df = df.rename(columns={col: 'gene_id'})
                found = True
                print(f"🔄 已将 '{col}' 列重命名为 'gene_id'")
                break
        if not found:
            print(f"❌ 未找到基因ID列，可用的列: {list(df.columns)}")
            return None, None

    if 'label' not in df.columns:
        possible_label_cols = ['mean_expression', 'expression', 'log2_expression', 'tpm']
        found = False
        for col in possible_label_cols:
            if col in df.columns:
                df = df.rename(columns={col: 'label'})
                found = True
                print(f"🔄 已将 '{col}' 列重命名为 'label'")
                break
        if not found:
            print(f"❌ 未找到标签列，可用的列: {list(df.columns)}")
            return None, None

    expr_dict = {}
    for _, row in df.iterrows():
        if not pd.isna(row['label']):
            expr_dict[row['gene_id']] = row['label']

    expression_values = []
    found_count = 0
    for gene_id in valid_gene_ids:
        if gene_id in expr_dict:
            expression_values.append(expr_dict[gene_id])
            found_count += 1
        else:
            expression_values.append(float('nan'))

    expression_tensor = torch.tensor(expression_values, dtype=torch.float32)
    print(f"✅ 标签对齐完成: {found_count}/{len(valid_gene_ids)} 个基因有表达值")

    return expression_tensor, [gid for gid in valid_gene_ids if gid in expr_dict]


def load_network_filtered(network_path, valid_gene_to_idx, network_name):
    if not os.path.exists(network_path):
        print(f"⚠️ {network_name}网络不存在: {network_path}")
        return None

    try:
        data = torch.load(network_path, map_location='cpu', weights_only=False)

        if isinstance(data, dict):
            if 'edge_index' in data:
                edge_index = data['edge_index']
            elif 'edges' in data:
                edge_index = data['edges']
            else:
                print(f"⚠️ {network_name}网络缺少edge_index")
                return None

            if 'gene_list' in data:
                network_gene_list = data['gene_list']
            else:
                print(f"   ⚠️ {network_name}网络缺少gene_list")
                return None
        else:
            print(f"⚠️ {network_name}网络格式不支持")
            return None

        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        network_idx_to_new_idx = {}
        matched_count = 0

        for net_idx, gene_id in enumerate(network_gene_list):
            if gene_id in valid_gene_to_idx:
                network_idx_to_new_idx[net_idx] = valid_gene_to_idx[gene_id]
                matched_count += 1

        print(f"   {network_name}网络: {matched_count}/{len(network_gene_list)} 个基因在有效基因中")

        if matched_count == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        original_src = edge_index[0]
        original_dst = edge_index[1]

        src_mask = torch.isin(original_src, torch.tensor(list(network_idx_to_new_idx.keys()), dtype=torch.long))
        dst_mask = torch.isin(original_dst, torch.tensor(list(network_idx_to_new_idx.keys()), dtype=torch.long))
        valid_mask = src_mask & dst_mask

        if valid_mask.sum() == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        valid_src = original_src[valid_mask]
        valid_dst = original_dst[valid_mask]

        new_src = torch.tensor([network_idx_to_new_idx[idx.item()] for idx in valid_src], dtype=torch.long)
        new_dst = torch.tensor([network_idx_to_new_idx[idx.item()] for idx in valid_dst], dtype=torch.long)

        new_edge_index = torch.stack([new_src, new_dst])
        print(f"   {network_name}网络: 原始边数 {edge_index.shape[1]} → 过滤后 {new_edge_index.shape[1]} 条边")

        return new_edge_index

    except Exception as e:
        print(f"❌ 加载{network_name}网络失败: {e}")
        return None


# =================================================================
# 数据集类 (增强版邻居预计算)
# =================================================================

class HGNTDataset(Dataset):
    def __init__(self, embeddings, tf_edge_index, gcn_edge_index,
                 expression_values, gene_ids, target_genes,
                 precomputed_neighbors=None, seed=42):

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
                expr_val = self.expression_tensor[idx]
                has_expr = not torch.isnan(expr_val)
                self.valid_expression_mask.append(has_expr)

        print(f"\n📊 数据集构建统计:")
        print(f"   有效基因数量: {len(self.valid_genes)}/{len(target_genes)}")
        print(f"   有表达值的有效基因: {sum(self.valid_expression_mask)}")

        # 使用预计算的邻居或重新计算
        if precomputed_neighbors is not None:
            self.all_neighbors = precomputed_neighbors
            print(f"   使用预计算邻居字典")
        else:
            self.all_neighbors = self._precompute_all_neighbors()

    def _precompute_all_neighbors(self):
        """预计算所有节点的网络邻居 (使用固定种子)"""
        print("\n   🔍 预计算所有节点的网络邻居...")

        adj_dict = {
            'tf': defaultdict(list),
            'gcn': defaultdict(list)
        }

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
        FIXED_SEED = 42  # 固定种子确保所有模型一致

        for node_idx in range(self.num_nodes):
            neighbors_dict = {}

            for net_name in ['tf', 'gcn']:
                neighbors = adj_dict[net_name].get(node_idx, [])
                if neighbors:
                    if len(neighbors) > MAX_NEIGHBORS:
                        random.seed(FIXED_SEED + node_idx)
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

        neighbor_dict = self.all_neighbors[gene_idx]

        return {
            'gene_id': gene_id,
            'gene_idx': gene_idx,
            'neighbor_indices': neighbor_dict,
            'expression': torch.tensor(expression, dtype=torch.float32),
            'has_expression': not torch.isnan(expression)
        }


def hgnt_collate_fn(batch):
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
# 通用训练器
# =================================================================

class HGNTTrainer:
    def __init__(self, model, device='cpu', learning_rate=1e-4,
                 patience=15, min_lr=1e-6, seed=42, model_name="M1"):
        self.model = model.to(device)
        self.device = device
        self.seed = seed
        self.model_name = model_name

        self.all_embeddings = None
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

        # 权重监控 (M2, M3, M4)
        self.weight_history = []

        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def set_graph_data(self, all_embeddings, tf_edge_index=None, gcn_edge_index=None):
        self.all_embeddings = all_embeddings
        self.num_nodes = len(all_embeddings) if all_embeddings is not None else 0

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

    def _extract_subgraphs(self, unique_nodes, device):
        unique_nodes_cpu = unique_nodes.cpu()

        if self.tf_edge_index is not None and self.tf_edge_index.numel() > 0:
            tf_sub, _ = subgraph(
                unique_nodes_cpu,
                self.tf_edge_index.cpu(),
                relabel_nodes=True,
                num_nodes=self.num_nodes
            )
            tf_sub = tf_sub.to(device)
        else:
            tf_sub = torch.zeros((2, 0), device=device, dtype=torch.long)

        if self.gcn_edge_index is not None and self.gcn_edge_index.numel() > 0:
            gcn_sub, _ = subgraph(
                unique_nodes_cpu,
                self.gcn_edge_index.cpu(),
                relabel_nodes=True,
                num_nodes=self.num_nodes
            )
            gcn_sub = gcn_sub.to(device)
        else:
            gcn_sub = torch.zeros((2, 0), device=device, dtype=torch.long)

        return tf_sub, gcn_sub

    def _prepare_batch_for_model(self, batch):
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
        node_mapping = {node.item(): i for i, node in enumerate(unique_nodes)}

        tf_sub, gcn_sub = self._extract_subgraphs(unique_nodes, self.device)

        if hasattr(self.model, 'set_subgraphs'):
            self.model.set_subgraphs(tf_sub, gcn_sub)

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

        # 权重监控
        epoch_weights = {'tf': [], 'gcn': []}

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

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    # 统一调用方式 - 所有模型都返回 (output, weights_dict)
                    outputs, weights_dict = self.model(x)

                    if target_local_indices.numel() > 0:
                        outputs = outputs[target_local_indices]
                    outputs = outputs.squeeze()

                    valid_outputs = outputs[has_expression]
                    valid_targets = expressions[has_expression]

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

                # 记录权重
                if 'tf_mean' in weights_dict:
                    epoch_weights['tf'].append(weights_dict['tf_mean'])
                if 'gcn_mean' in weights_dict:
                    epoch_weights['gcn'].append(weights_dict['gcn_mean'])

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

        # 记录权重均值
        if epoch_weights['tf']:
            self.weight_history.append({
                'epoch': epoch,
                'tf_mean': np.mean(epoch_weights['tf']),
                'gcn_mean': np.mean(epoch_weights['gcn'])
            })

        return avg_loss, train_pearson

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_valid_samples = 0
        all_preds = []
        all_targets = []

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

                # 统一调用方式 - 所有模型都返回 (output, weights_dict)
                outputs, _ = self.model(x)  # 验证时不需要权重信息

                if target_local_indices.numel() > 0:
                    outputs = outputs[target_local_indices]
                outputs = outputs.squeeze()

                valid_outputs = outputs[has_expression]
                valid_targets = expressions[has_expression]

                loss_values = self.criterion(valid_outputs, valid_targets)
                loss = loss_values.mean()

                total_loss += loss.item() * has_expression.sum().item()
                num_valid_samples += has_expression.sum().item()

                all_preds.extend(valid_outputs.cpu().numpy())
                all_targets.extend(valid_targets.cpu().numpy())

        avg_loss = total_loss / num_valid_samples if num_valid_samples > 0 else float('inf')

        return avg_loss, np.array(all_preds), np.array(all_targets)

    def train(self, train_loader, val_loader, epochs=100):
        for epoch in range(epochs):
            train_loss, train_pearson = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            val_loss, val_preds, val_targets = self.validate(val_loader)
            self.val_losses.append(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.counter = 0
                self.best_model_state = self.model.state_dict().copy()
                if (epoch + 1) % 5 == 0 or epoch == 0:
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
            'weight_history': self.weight_history
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
# 单模型多种子训练
# =================================================================

def train_model_across_seeds(data_dict, model_name, args, global_precomputed_neighbors=None):
    """对单个模型进行多种子训练"""
    print(f"\n{'=' * 70}")
    print(f"🚀 训练模型: {model_name}")
    print(f"{'=' * 70}")

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    all_results = []

    for seed in seeds:
        print(f"\n{'=' * 50}")
        print(f"🎲 种子: {seed}")
        print(f"{'=' * 50}")

        set_seed(seed)

        # 划分数据
        gene_ids = data_dict['gene_ids']
        total_genes = len(gene_ids)
        indices = list(range(total_genes))

        train_indices, temp_indices = train_test_split(
            indices, train_size=TRAIN_RATIO, random_state=seed, shuffle=True
        )
        val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_indices, test_indices = train_test_split(
            temp_indices, train_size=val_ratio_adjusted, random_state=seed, shuffle=True
        )

        train_genes = [gene_ids[i] for i in train_indices]
        val_genes = [gene_ids[i] for i in val_indices]
        test_genes = [gene_ids[i] for i in test_indices]

        print(f"训练集: {len(train_genes)}, 验证集: {len(val_genes)}, 测试集: {len(test_genes)}")

        # 创建数据集 (使用预计算的邻居)
        train_dataset = HGNTDataset(
            data_dict['embeddings'],
            data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
            data_dict['expression_values'],
            data_dict['gene_ids'],
            train_genes,
            precomputed_neighbors=global_precomputed_neighbors,
            seed=seed
        )

        val_dataset = HGNTDataset(
            data_dict['embeddings'],
            data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
            data_dict['expression_values'],
            data_dict['gene_ids'],
            val_genes,
            precomputed_neighbors=global_precomputed_neighbors,
            seed=seed + 1
        )

        test_dataset = HGNTDataset(
            data_dict['embeddings'],
            data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
            data_dict['expression_values'],
            data_dict['gene_ids'],
            test_genes,
            precomputed_neighbors=global_precomputed_neighbors,
            seed=seed + 2
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            collate_fn=hgnt_collate_fn,
            generator=torch.Generator().manual_seed(seed)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=hgnt_collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=hgnt_collate_fn
        )

        input_dim = data_dict['embeddings'].size(1)
        model = create_model(model_name, input_dim=input_dim)
        print(f"  ✅ 成功构建模型: {model_name}, 输入维度: {input_dim}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  📊 模型参数: {total_params:,}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  🔧 使用设备: {device}")

        trainer = HGNTTrainer(
            model, device=device,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=seed,
            model_name=model_name
        )

        trainer.set_graph_data(
            all_embeddings=data_dict['embeddings'],
            tf_edge_index=data_dict['tf_edge_index'],
            gcn_edge_index=data_dict['gcn_edge_index']
        )

        print(f"  🚀 开始训练...")

        training_history = trainer.train(
            train_loader, val_loader,
            epochs=args.epochs
        )

        # 测试集评估
        test_loss, test_preds, test_targets = trainer.validate(test_loader)
        test_evaluation = evaluate_regression(test_targets, test_preds)

        print(f"\n  📈 Seed {seed} 测试结果:")
        print(f"     R²: {test_evaluation['r2']:.6f}")
        print(f"     Pearson: {test_evaluation['pearson_corr']:.6f}")
        print(f"     Spearman: {test_evaluation['spearman_corr']:.6f}")
        print(f"     RMSE: {test_evaluation['rmse']:.6f}")

        # 打印权重监控信息 (M2, M3, M4)
        if model_name in ['M2', 'M3', 'M4']:
            if model_name == 'M2' and hasattr(model, 'get_weights'):
                tf_w, gcn_w = model.get_weights()
                print(f"     🔍 融合权重: TF={tf_w:.4f}, GCN={gcn_w:.4f}")
            elif training_history['weight_history']:
                last_weights = training_history['weight_history'][-1]
                print(f"     🔍 动态权重 (最后epoch): TF={last_weights['tf_mean']:.4f}, GCN={last_weights['gcn_mean']:.4f}")

        result = {
            'seed': seed,
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
            'model_params': total_params
        }

        all_results.append(result)

    if not all_results:
        return None

    # 计算统计量
    metrics_keys = ['test_r2', 'test_pearson', 'test_spearman', 'test_rmse']
    metrics_summary = {}

    for key in metrics_keys:
        values = [r[key] for r in all_results]
        metrics_summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values
        }

    return {
        'model_name': model_name,
        'seed_results': all_results,
        'metrics_summary': metrics_summary
    }


# =================================================================
# 多模型消融实验
# =================================================================

def run_ablation_experiment(data_dict, args):
    """运行所有4种模型的消融实验"""

    print("\n" + "=" * 80)
    print("🔬 小龙虾基因表达预测 - 多模型消融实验")
    print(f"模型列表: {list(MODEL_REGISTRY.keys())}")
    print("=" * 80)

    # 预先计算邻居字典 (固定种子42，确保所有模型一致)
    print("\n📦 预计算邻居字典 (固定种子=42)...")
    temp_dataset = HGNTDataset(
        data_dict['embeddings'],
        data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
        data_dict['expression_values'],
        data_dict['gene_ids'],
        data_dict['gene_ids'][:100],  # 只用少量基因触发预计算
        seed=42
    )
    global_precomputed_neighbors = temp_dataset.all_neighbors
    print(f"✅ 邻居字典预计算完成，节点数: {len(global_precomputed_neighbors)}")

    all_model_results = {}

    for model_name in args.models:  # 使用 args.models
        model_result = train_model_across_seeds(
            data_dict, model_name, args, global_precomputed_neighbors
        )
        if model_result:
            all_model_results[model_name] = model_result

        # 保存中间结果
        with open(os.path.join(args.output_dir, f'{model_name}_results.json'), 'w') as f:
            json.dump(model_result, f, indent=2, default=str)

    # 打印对比表格 - 传入 output_dir
    print_comparison_table(all_model_results, args.output_dir)

    # 保存汇总结果
    save_ablation_results(all_model_results, args)

    return all_model_results


def print_comparison_table(all_model_results, output_dir):
    """打印4种模型的性能对比表"""

    print("\n" + "=" * 90)
    print("📊 模型性能对比表 (均值 ± 标准差)")
    print("=" * 90)

    # 表头
    print(f"{'Model':<15} {'R²':<20} {'Pearson':<20} {'Spearman':<20} {'RMSE':<20}")
    print("-" * 95)

    for model_name in ['M1', 'M2', 'M3', 'M4']:
        if model_name not in all_model_results:
            continue

        metrics = all_model_results[model_name]['metrics_summary']

        r2_str = f"{metrics['test_r2']['mean']:.6f} ± {metrics['test_r2']['std']:.6f}"
        pearson_str = f"{metrics['test_pearson']['mean']:.6f} ± {metrics['test_pearson']['std']:.6f}"
        spearman_str = f"{metrics['test_spearman']['mean']:.6f} ± {metrics['test_spearman']['std']:.6f}"
        rmse_str = f"{metrics['test_rmse']['mean']:.6f} ± {metrics['test_rmse']['std']:.6f}"

        print(f"{model_name:<15} {r2_str:<20} {pearson_str:<20} {spearman_str:<20} {rmse_str:<20}")

    print("=" * 90)

    # 打印最佳模型
    print("\n🏆 最佳模型 (按Pearson相关系数):")
    best_model = max(all_model_results.items(),
                     key=lambda x: x[1]['metrics_summary']['test_pearson']['mean'])
    print(f"   最佳模型: {best_model[0]}")
    print(
        f"   Pearson: {best_model[1]['metrics_summary']['test_pearson']['mean']:.6f} ± {best_model[1]['metrics_summary']['test_pearson']['std']:.6f}")

    # 打印权重偏好分析
    print("\n🔍 融合权重偏好分析:")
    for model_name in ['M2', 'M3', 'M4']:
        if model_name in all_model_results:
            # 使用传入的 output_dir 参数
            weight_file = os.path.join(output_dir, f'{model_name}_results.json')
            if os.path.exists(weight_file):
                with open(weight_file, 'r') as f:
                    data = json.load(f)
                if 'weight_summary' in data:
                    ws = data['weight_summary']
                    print(f"   {model_name}: TF权重均值={ws['tf_mean']:.4f}, GCN权重均值={ws['gcn_mean']:.4f}")


def save_ablation_results(all_model_results, args):
    """保存消融实验结果"""

    # 构建汇总DataFrame
    summary_data = []
    for model_name, results in all_model_results.items():
        metrics = results['metrics_summary']
        summary_data.append({
            'model': model_name,
            'params': results['seed_results'][0]['model_params'] if results['seed_results'] else 0,
            'r2_mean': metrics['test_r2']['mean'],
            'r2_std': metrics['test_r2']['std'],
            'pearson_mean': metrics['test_pearson']['mean'],
            'pearson_std': metrics['test_pearson']['std'],
            'spearman_mean': metrics['test_spearman']['mean'],
            'spearman_std': metrics['test_spearman']['std'],
            'rmse_mean': metrics['test_rmse']['mean'],
            'rmse_std': metrics['test_rmse']['std']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(args.output_dir, 'ablation_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n💾 消融实验汇总已保存: {summary_file}")

    # 保存完整结果
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'seeds': args.seeds,
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'patience': args.patience,
            'max_neighbors': MAX_NEIGHBORS
        },
        'models': all_model_results
    }

    full_file = os.path.join(args.output_dir, 'full_ablation_results.json')
    with open(full_file, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"💾 完整结果已保存: {full_file}")


# =================================================================
# 主函数
# =================================================================

def main():
    parser = argparse.ArgumentParser(description='小龙虾基因表达预测 - 多模型消融实验')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--output_dir', type=str, default='Results_try')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'随机种子列表 (默认: {DEFAULT_SEEDS})')
    parser.add_argument('--models', type=str, nargs='+', default=['M1', 'M2', 'M3', 'M4'],
                        help='要测试的模型列表 (默认: 全部)')

    args = parser.parse_args()

    print("=" * 80)
    print("🔬 小龙虾基因表达预测 - 多模型消融实验")
    print("=" * 80)
    print("\n🔧 训练配置:")
    print(f"  物种: 小龙虾")
    print(f"  模型列表: {args.models}")
    print(f"  随机种子: {args.seeds}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  索引文件: {INDEX_FILE}")
    print(f"  标签文件: {LABELS_FILE}")
    print(f"  Embedding文件: {EMBEDDING_FILE}")
    print(f"  TF网络: {TF_PATH}")
    print(f"  GCN网络: {GCN_PATH}")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    config = {
        'seeds': args.seeds,
        'models': args.models,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'max_neighbors': MAX_NEIGHBORS,
        'index_file': INDEX_FILE,
        'labels_file': LABELS_FILE,
        'embedding_file': EMBEDDING_FILE,
        'tf_path': TF_PATH,
        'gcn_path': GCN_PATH,
        'timestamp': datetime.now().isoformat()
    }

    config_file = os.path.join(args.output_dir, 'experiment_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # 加载数据
    print(f"\n🔍 加载小龙虾数据...")

    # 1. 加载基因索引文件
    index_gene_ids, gene_to_idx = load_gene_index(INDEX_FILE)
    if index_gene_ids is None:
        return
    print(f"   索引文件基因数: {len(index_gene_ids)}")

    # 2. 加载NT embeddings
    embed_data = load_nt_embeddings(EMBEDDING_FILE)
    if embed_data is None:
        return
    print(f"   Embeddings基因数: {len(embed_data['gene_ids'])}")

    # 3. 找到共同基因
    embed_gene_set = set(embed_data['gene_ids'])
    index_gene_set = set(index_gene_ids)
    common_genes = index_gene_set.intersection(embed_gene_set)

    if len(common_genes) == 0:
        print(f"❌ 索引文件和embeddings没有共同基因")
        return

    print(f"   共同基因数: {len(common_genes)}")

    # 4. 按索引顺序重新排列embeddings
    embed_idx_map = {gid: i for i, gid in enumerate(embed_data['gene_ids'])}

    new_embeddings_list = []
    new_gene_ids_list = []

    for gid in index_gene_ids:
        if gid in embed_idx_map:
            new_embeddings_list.append(embed_data['embeddings'][embed_idx_map[gid]])
            new_gene_ids_list.append(gid)

    if not new_embeddings_list:
        print(f"❌ 没有找到有效的基因")
        return

    new_embeddings = torch.stack(new_embeddings_list)
    new_gene_ids = new_gene_ids_list
    print(f"✅ Embeddings已对齐，形状: {new_embeddings.shape}")
    print(f"   有效基因数: {len(new_gene_ids)}")

    # 5. 创建有效基因到新索引的映射
    valid_gene_to_idx = {gid: i for i, gid in enumerate(new_gene_ids)}
    num_valid_nodes = len(new_gene_ids)

    # 6. 加载表达量数据
    expression_tensor, valid_expr_genes = load_expression_data(LABELS_FILE, new_gene_ids)
    if expression_tensor is None:
        return

    # 7. 加载网络文件
    print(f"\n📖 加载网络文件...")
    tf_edge_index = load_network_filtered(TF_PATH, valid_gene_to_idx, "TF")
    gcn_edge_index = load_network_filtered(GCN_PATH, valid_gene_to_idx, "GCN")

    data_dict = {
        'embeddings': new_embeddings,
        'expression_values': expression_tensor,
        'gene_ids': new_gene_ids,
        'tf_edge_index': tf_edge_index,
        'gcn_edge_index': gcn_edge_index,
        'num_genes': num_valid_nodes,
        'num_genes_with_expression': int((~torch.isnan(expression_tensor)).sum().item())
    }

    print(f"\n📊 数据统计:")
    print(f"   embeddings形状: {data_dict['embeddings'].shape}")
    print(f"   有效基因总数: {data_dict['num_genes']}")
    print(f"   有表达值基因: {data_dict['num_genes_with_expression']}")
    print(f"   TF网络边数: {tf_edge_index.shape[1] if tf_edge_index is not None else 0}")
    print(f"   GCN网络边数: {gcn_edge_index.shape[1] if gcn_edge_index is not None else 0}")

    # 运行消融实验
    run_ablation_experiment(data_dict, args)

    print(f"\n{'=' * 80}")
    print("✅ 全部消融实验完成!")
    print(f"   结果保存在: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()