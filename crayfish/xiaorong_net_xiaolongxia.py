"""
小龙虾基因表达预测 - 网络来源消融实验 V3 (支持多种子重复实验)
对比不同网络来源对预测性能的影响

M3a: 可学习权重融合 (TF + GCN双路) - 继承V2 M3架构
M3b: 单路GCN (仅TF网络) - 无融合参数
M3c: 单路GCN (仅GCN网络) - 无融合参数
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
# M3a: 可学习权重融合 (TF + GCN双路) - 继承V2 M3架构
# =================================================================

class ModelM3a_WeightedFusion(nn.Module):
    """M3a: 双路GCN + 可学习权重融合 (Softmax归一化)"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3a_WeightedFusion, self).__init__()

        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 双路GCN
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)

        # 可学习融合权重 (Softmax归一化)
        self.fusion_logits = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))

        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

    def forward(self, x, edge_index=None):
        s_feat = self.seq_proj(x)

        # 双路图卷积
        if self.tf_sub is not None and self.tf_sub.numel() > 0:
            t_info = F.elu(self.tf_conv(s_feat, self.tf_sub))
        else:
            t_info = s_feat

        if self.gcn_sub is not None and self.gcn_sub.numel() > 0:
            g_info = F.elu(self.gcn_conv(s_feat, self.gcn_sub))
        else:
            g_info = s_feat

        # Softmax加权融合
        fusion_weights = F.softmax(self.fusion_logits, dim=0)
        tf_weight = fusion_weights[0]
        gcn_weight = fusion_weights[1]

        graph_feat = tf_weight * t_info + gcn_weight * g_info

        # 残差连接
        final_feat = s_feat + graph_feat

        weights = {
            "aggregation_type": "Weighted_Fusion_TF_GCN",
            "fusion_type": "learnable_softmax",
            "tf_weight": tf_weight.item(),
            "gcn_weight": gcn_weight.item(),
            "fusion_gate": tf_weight.item()
        }

        return self.regressor(final_feat).squeeze(-1), weights


# =================================================================
# M3b: 单路GCN (仅TF网络) - 无融合参数
# =================================================================

class ModelM3b_OnlyTF(nn.Module):
    """M3b: 单路图 - 仅使用TF网络"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3b_OnlyTF, self).__init__()

        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 单路GCN
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)

        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub=None):
        self.tf_sub = tf_sub

    def forward(self, x, edge_index=None):
        s_feat = self.seq_proj(x)

        # 单路图卷积
        if self.tf_sub is not None and self.tf_sub.numel() > 0:
            graph_feat = F.elu(self.tf_conv(s_feat, self.tf_sub))
        else:
            graph_feat = s_feat

        # 残差连接
        final_feat = s_feat + graph_feat

        weights = {
            "aggregation_type": "Single_TF",
            "fusion_type": "single_path"
        }

        return self.regressor(final_feat).squeeze(-1), weights


# =================================================================
# M3c: 单路GCN (仅GCN网络) - 无融合参数
# =================================================================

class ModelM3c_OnlyGCN(nn.Module):
    """M3c: 单路图 - 仅使用GCN网络"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3c_OnlyGCN, self).__init__()

        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 单路GCN
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)

        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.gcn_sub = None

    def set_subgraphs(self, tf_sub=None, gcn_sub=None):
        self.gcn_sub = gcn_sub

    def forward(self, x, edge_index=None):
        s_feat = self.seq_proj(x)

        # 单路图卷积
        if self.gcn_sub is not None and self.gcn_sub.numel() > 0:
            graph_feat = F.elu(self.gcn_conv(s_feat, self.gcn_sub))
        else:
            graph_feat = s_feat

        # 残差连接
        final_feat = s_feat + graph_feat

        weights = {
            "aggregation_type": "Single_GCN",
            "fusion_type": "single_path"
        }

        return self.regressor(final_feat).squeeze(-1), weights


# =================================================================
# 模型工厂函数
# =================================================================

def build_model(model_name, input_dim=2560, dropout=0.3, **kwargs):
    """模型工厂 - 支持M3a, M3b, M3c"""

    if model_name == 'm3a':
        return ModelM3a_WeightedFusion(input_dim=input_dim, dropout=dropout)
    elif model_name == 'm3b':
        return ModelM3b_OnlyTF(input_dim=input_dim, dropout=dropout)
    elif model_name == 'm3c':
        return ModelM3c_OnlyGCN(input_dim=input_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}. 可选: m3a, m3b, m3c")


# =================================================================
# 固定随机种子函数
# =================================================================
def set_seed(seed=42, verbose=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if verbose:
        print(f"✅ 随机种子已设置为: {seed}")


# =================================================================
# 加载基因索引文件
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


# =================================================================
# 加载NT embeddings
# =================================================================
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


# =================================================================
# 加载表达量数据（从CSV）
# =================================================================
def load_expression_data(labels_file, valid_gene_ids):
    if not os.path.exists(labels_file):
        print(f"❌ 标签文件不存在: {labels_file}")
        return None, None

    df = pd.read_csv(labels_file)
    print(f"✅ 成功加载标签文件: {labels_file}")
    print(f"   标签数据形状: {df.shape}")

    # 检查gene_id列
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

    # 检查label列
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

    # 创建标签字典
    expr_dict = {}
    for _, row in df.iterrows():
        if not pd.isna(row['label']):
            expr_dict[row['gene_id']] = row['label']

    # 按有效基因顺序构建标签张量
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


# =================================================================
# 加载网络文件 - 使用gene_list快速映射
# =================================================================
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
                print(f"   {network_name}网络包含基因列表，共 {len(network_gene_list)} 个基因")
            else:
                print(f"   ⚠️ {network_name}网络缺少gene_list")
                return None
        else:
            print(f"⚠️ {network_name}网络格式不支持")
            return None

        # 确保 edge_index 是 2 x E 格式
        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        # 创建网络索引到新索引的映射
        network_idx_to_new_idx = {}
        matched_count = 0

        for net_idx, gene_id in enumerate(network_gene_list):
            if gene_id in valid_gene_to_idx:
                network_idx_to_new_idx[net_idx] = valid_gene_to_idx[gene_id]
                matched_count += 1

        print(f"   {network_name}网络: {matched_count}/{len(network_gene_list)} 个基因在有效基因中")

        if matched_count == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        # 过滤和映射边
        original_src = edge_index[0]
        original_dst = edge_index[1]

        # 使用向量化操作加速
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
# 数据集类
# =================================================================
class GATDeepCREDataset(Dataset):
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
                expr_val = self.expression_tensor[idx]
                has_expr = not torch.isnan(expr_val)
                self.valid_expression_mask.append(has_expr)

        print(f"\n📊 数据集构建统计:")
        print(f"   有效基因数量: {len(self.valid_genes)}/{len(target_genes)}")
        print(f"   有表达值的有效基因: {sum(self.valid_expression_mask)}")

        self.all_neighbors = self._precompute_all_neighbors()

    def _precompute_all_neighbors(self):
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
        self.tf_edge_index = None
        self.gcn_edge_index = None
        self.num_nodes = 0

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

        self.network_importance_history = []
        self.fusion_gate_history = []

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

        print(f"   📊 最终图数据状态:")
        print(f"      TF边数: {self.tf_edge_index.shape[1] if self.tf_edge_index is not None else 0}")
        print(f"      GCN边数: {self.gcn_edge_index.shape[1] if self.gcn_edge_index is not None else 0}")

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
        """批次准备函数，支持不同模型的子图提取策略"""
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

        # 提取子图
        tf_sub, gcn_sub = self._extract_subgraphs(unique_nodes, self.device)

        if hasattr(self.model, 'set_subgraphs'):
            # 根据模型类型传递正确的子图参数
            if self.model_name == 'm3b':
                self.model.set_subgraphs(tf_sub, None)
            elif self.model_name == 'm3c':
                self.model.set_subgraphs(None, gcn_sub)
            else:  # m3a 或其他
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

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs, weights = self.model(x)

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

                # 记录融合权重（仅M3a）
                if weights and 'fusion_gate' in weights:
                    gate_val = weights['fusion_gate']
                    if isinstance(gate_val, torch.Tensor):
                        gate_val = gate_val.item() if gate_val.numel() == 1 else gate_val.mean().item()
                    epoch_gate_means.append(gate_val)

                if weights and 'aggregation_type' in weights and epoch == 0 and batch_idx == 0:
                    print(f"   📌 模型聚合类型: {weights['aggregation_type']}")

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

                outputs, _ = self.model(x)

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

                batch_gene_ids = batch['gene_ids']
                for i, valid in enumerate(has_expression.cpu().numpy()):
                    if valid:
                        all_gene_ids.append(batch_gene_ids[i])

        avg_loss = total_loss / num_valid_samples if num_valid_samples > 0 else float('inf')

        return avg_loss, np.array(all_preds), np.array(all_targets), all_gene_ids

    def train(self, train_loader, val_loader, epochs=100):
        """完整训练循环"""
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
# 多种子实验结果汇总
# =================================================================
def aggregate_seed_results(all_seed_results):
    """汇总多个种子的实验结果，计算均值、标准差等统计量"""

    if not all_seed_results:
        return None

    # 收集各指标
    metrics_keys = ['r2', 'pearson_corr', 'spearman_corr', 'rmse', 'mae', 'mse']

    aggregated = {}

    for split in ['validation', 'test']:
        aggregated[split] = {}
        for metric in metrics_keys:
            values = []
            for seed_result in all_seed_results:
                if split in seed_result and metric in seed_result[split]:
                    values.append(seed_result[split][metric])

            if values:
                aggregated[split][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }

    # 记录融合权重（如果是M3a）
    if 'final_tf_weight' in all_seed_results[0]:
        tf_weights = [r['final_tf_weight'] for r in all_seed_results if 'final_tf_weight' in r]
        gcn_weights = [r['final_gcn_weight'] for r in all_seed_results if 'final_gcn_weight' in r]
        if tf_weights:
            aggregated['final_tf_weight'] = {
                'mean': np.mean(tf_weights),
                'std': np.std(tf_weights),
                'values': tf_weights
            }
            aggregated['final_gcn_weight'] = {
                'mean': np.mean(gcn_weights),
                'std': np.std(gcn_weights),
                'values': gcn_weights
            }

    # 记录模型信息和种子列表
    aggregated['model_name'] = all_seed_results[0]['model_name']
    aggregated['model_description'] = all_seed_results[0]['model_description']
    aggregated['seeds'] = [r['random_seed'] for r in all_seed_results]
    aggregated['num_seeds'] = len(all_seed_results)

    return aggregated


# =================================================================
# 主训练类 (支持多随机种子)
# =================================================================
class CrayfishDeepGATAblation:
    def __init__(self,
                 output_dir='Results_Crayfish_Ablation_V3',
                 batch_size=32,
                 epochs=100,
                 learning_rate=1e-4,
                 patience=15,
                 seeds=[42],
                 analyze_only=False):

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.seeds = seeds if isinstance(seeds, list) else [seeds]
        self.analyze_only = analyze_only

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

        self.all_results = []  # 存储所有模型的所有种子结果
        self.model_seed_results = {}  # {model_name: {seed: result}}

    def convert_for_json(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def load_species_data(self):
        print(f"\n🔍 加载小龙虾数据...")

        # 1. 加载基因索引文件
        index_gene_ids, gene_to_idx = load_gene_index(INDEX_FILE)
        if index_gene_ids is None:
            return None
        print(f"   索引文件基因数: {len(index_gene_ids)}")

        # 2. 加载NT embeddings
        embed_data = load_nt_embeddings(EMBEDDING_FILE)
        if embed_data is None:
            return None
        print(f"   Embeddings基因数: {len(embed_data['gene_ids'])}")

        # 3. 找到共同基因
        embed_gene_set = set(embed_data['gene_ids'])
        index_gene_set = set(index_gene_ids)
        common_genes = index_gene_set.intersection(embed_gene_set)

        if len(common_genes) == 0:
            print(f"❌ 索引文件和embeddings没有共同基因")
            return None

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
            return None

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
            return None

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

        return data_dict

    def prepare_datasets(self, data_dict, seed):
        print(f"\n  准备训练/验证/测试数据集 (seed={seed})...")

        gene_ids = data_dict['gene_ids']
        total_genes = len(gene_ids)

        indices = list(range(total_genes))

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

        print(f"    训练集: {len(train_genes)} 个基因 ({len(train_genes) / total_genes:.1%})")
        print(f"    验证集: {len(val_genes)} 个基因 ({len(val_genes) / total_genes:.1%})")
        print(f"    测试集: {len(test_genes)} 个基因 ({len(test_genes) / total_genes:.1%})")

        train_dataset = GATDeepCREDataset(
            data_dict['embeddings'],
            data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
            data_dict['expression_values'],
            data_dict['gene_ids'],
            train_genes, seed=seed
        )

        val_dataset = GATDeepCREDataset(
            data_dict['embeddings'],
            data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
            data_dict['expression_values'],
            data_dict['gene_ids'],
            val_genes, seed=seed + 1
        )

        test_dataset = GATDeepCREDataset(
            data_dict['embeddings'],
            data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
            data_dict['expression_values'],
            data_dict['gene_ids'],
            test_genes, seed=seed + 2
        )

        return train_dataset, val_dataset, test_dataset

    def _get_model_description(self, model_name):
        descriptions = {
            'm3a': '可学习权重融合 (TF + GCN双路) - 动态学习两网络重要性',
            'm3b': '单路GCN (仅TF网络) - 验证转录因子网络单独效果',
            'm3c': '单路GCN (仅GCN网络) - 验证共表达网络单独效果'
        }
        return descriptions.get(model_name, 'Unknown')

    def train_model(self, train_dataset, val_dataset, test_dataset, data_dict, model_name, seed):
        print(f"\n  🔬 训练小龙虾 - 模型 {model_name.upper()} (seed={seed})")
        print(f"  {'-' * 50}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            collate_fn=gat_collate_fn,
            generator=torch.Generator().manual_seed(seed)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=gat_collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=gat_collate_fn
        )

        input_dim = data_dict['embeddings'].size(1)
        model = build_model(model_name, input_dim=input_dim)
        print(f"  ✅ 成功构建模型: {model_name.upper()}, 输入维度: {input_dim}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  📊 模型参数: {total_params:,}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  🔧 使用设备: {device}")

        trainer = GATDeepCRETrainer(
            model, model_name, device=device,
            learning_rate=self.learning_rate,
            patience=self.patience,
            seed=seed
        )

        trainer.set_graph_data(
            all_embeddings=data_dict['embeddings'],
            tf_edge_index=data_dict['tf_edge_index'],
            gcn_edge_index=data_dict['gcn_edge_index']
        )

        print(f"  🚀 开始训练...")

        training_history = trainer.train(
            train_loader, val_loader,
            epochs=self.epochs
        )

        val_loss, val_preds, val_targets, val_gene_ids = trainer.validate(val_loader)
        test_loss, test_preds, test_targets, test_gene_ids = trainer.validate(test_loader)

        val_evaluation = evaluate_regression(val_targets, val_preds)
        test_evaluation = evaluate_regression(test_targets, test_preds)

        results = {
            'species': 'crayfish',
            'model_name': model_name,
            'model_description': self._get_model_description(model_name),
            'random_seed': seed,
            'best_epoch': training_history['best_epoch'] + 1,
            'best_val_loss': float(training_history['best_val_loss']),
            'num_train_genes': len(train_dataset),
            'num_val_genes': len(val_dataset),
            'num_test_genes': len(test_dataset),
            'model_params': total_params,
            'max_neighbors': MAX_NEIGHBORS,
            'validation': val_evaluation,
            'test': test_evaluation,
            'fusion_gate_history': self.convert_for_json(training_history.get('fusion_gate_history', []))
        }

        # 如果是M3a，记录最终的融合权重
        if model_name == 'm3a' and hasattr(model, 'fusion_logits'):
            with torch.no_grad():
                weights = F.softmax(model.fusion_logits, dim=0)
                results['final_tf_weight'] = float(weights[0].cpu().item())
                results['final_gcn_weight'] = float(weights[1].cpu().item())
            print(f"  📊 最终融合权重 - TF: {results['final_tf_weight']:.4f}, GCN: {results['final_gcn_weight']:.4f}")

        self._print_evaluation_summary(results)
        self._save_predictions(val_gene_ids, val_targets, val_preds,
                               test_gene_ids, test_targets, test_preds, results, seed)
        self._save_model(model_name, model, trainer, results, seed)
        self._save_training_history(model_name, training_history, seed)

        return results

    def _print_evaluation_summary(self, results):
        print("\n" + "=" * 70)
        print(f"📈 模型性能评估 - {results['model_name'].upper()} (seed={results['random_seed']})")
        print("=" * 70)

        print("\n📊 验证集性能:")
        val = results['validation']
        print(f"  R²: {val['r2']:.6f}")
        print(f"  Pearson: {val['pearson_corr']:.6f}")
        print(f"  Spearman: {val['spearman_corr']:.6f}")
        print(f"  RMSE: {val['rmse']:.6f}")

        print("\n📊 测试集性能:")
        test = results['test']
        print(f"  R²: {test['r2']:.6f}")
        print(f"  Pearson: {test['pearson_corr']:.6f}")
        print(f"  Spearman: {test['spearman_corr']:.6f}")
        print(f"  RMSE: {test['rmse']:.6f}")

    def _save_predictions(self, val_gene_ids, val_targets, val_preds,
                          test_gene_ids, test_targets, test_preds, metrics, seed):
        pred_dir = os.path.join(self.output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)

        val_pred_df = pd.DataFrame({
            'gene_id': val_gene_ids,
            'true_expression': val_targets,
            'predicted_expression': val_preds,
            'set': 'validation'
        })

        test_pred_df = pd.DataFrame({
            'gene_id': test_gene_ids,
            'true_expression': test_targets,
            'predicted_expression': test_preds,
            'set': 'test'
        })

        all_pred_df = pd.concat([val_pred_df, test_pred_df], ignore_index=True)

        pred_file = os.path.join(pred_dir, f'crayfish_{metrics["model_name"]}_seed{seed}_predictions.csv')
        all_pred_df.to_csv(pred_file, index=False)

        metrics_file = os.path.join(pred_dir, f'crayfish_{metrics["model_name"]}_seed{seed}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.convert_for_json(metrics), f, indent=2)

        print(f"  💾 预测结果已保存: {pred_file}")

    def _save_model(self, model_name, model, trainer, metrics, seed):
        model_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)

        model_file = os.path.join(model_dir, f'crayfish_{model_name}_seed{seed}_best.pth')

        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': metrics,
            'species': 'crayfish',
            'model_name': model_name,
            'seed': seed
        }

        # 保存融合权重（如果是M3a）
        if model_name == 'm3a' and hasattr(model, 'fusion_logits'):
            save_dict['fusion_logits'] = model.fusion_logits.detach().cpu()

        torch.save(save_dict, model_file)
        print(f"  💾 模型已保存: {model_file}")

    def _save_training_history(self, model_name, history, seed):
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        history_file = os.path.join(log_dir, f'crayfish_{model_name}_seed{seed}_history.json')

        with open(history_file, 'w') as f:
            json.dump(self.convert_for_json(history), f, indent=2)

    def _save_model_aggregated_results(self, model_name, seed_results):
        """保存单个模型的多种子汇总结果"""
        agg_results = aggregate_seed_results(seed_results)

        if agg_results is None:
            return

        # 保存汇总结果
        agg_dir = os.path.join(self.output_dir, 'aggregated')
        os.makedirs(agg_dir, exist_ok=True)

        agg_file = os.path.join(agg_dir, f'{model_name}_aggregated_seeds.json')
        with open(agg_file, 'w') as f:
            json.dump(self.convert_for_json(agg_results), f, indent=2)

        # 保存为CSV格式便于查看
        rows = []
        for metric in ['r2', 'pearson_corr', 'spearman_corr', 'rmse', 'mae']:
            for split in ['validation', 'test']:
                if metric in agg_results[split]:
                    rows.append({
                        'model': model_name,
                        'split': split,
                        'metric': metric,
                        'mean': agg_results[split][metric]['mean'],
                        'std': agg_results[split][metric]['std'],
                        'min': agg_results[split][metric]['min'],
                        'max': agg_results[split][metric]['max']
                    })

        # 添加融合权重统计（如果是M3a）
        if 'final_tf_weight' in agg_results:
            rows.append({
                'model': model_name,
                'split': 'fusion_weights',
                'metric': 'tf_weight_mean',
                'mean': agg_results['final_tf_weight']['mean'],
                'std': agg_results['final_tf_weight']['std'],
                'min': np.min(agg_results['final_tf_weight']['values']),
                'max': np.max(agg_results['final_tf_weight']['values'])
            })
            rows.append({
                'model': model_name,
                'split': 'fusion_weights',
                'metric': 'gcn_weight_mean',
                'mean': agg_results['final_gcn_weight']['mean'],
                'std': agg_results['final_gcn_weight']['std'],
                'min': np.min(agg_results['final_gcn_weight']['values']),
                'max': np.max(agg_results['final_gcn_weight']['values'])
            })

        df = pd.DataFrame(rows)
        csv_file = os.path.join(agg_dir, f'{model_name}_aggregated_stats.csv')
        df.to_csv(csv_file, index=False)

        print(f"\n📊 {model_name.upper()} 多种子统计已保存:")
        print(f"   均值文件: {agg_file}")
        print(f"   统计文件: {csv_file}")

    def train(self, model_names=['m3a', 'm3b', 'm3c']):
        print(f"\n{'=' * 80}")
        print(f"🦞 开始训练小龙虾基因表达预测模型 - 网络来源消融实验 V3 (多种子重复实验)")
        print(f"随机种子列表: {self.seeds}")
        print(f"模型: {model_names}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'=' * 80}")
        print(f"\n📋 消融实验模型架构说明:")
        print(f"  {'=' * 70}")
        print(f"  M3a: 可学习权重融合 (TF + GCN双路) - 继承V2 M3架构")
        print(f"        → 可自动学习两个网络的重要性权重")
        print(f"  M3b: 单路GCN (仅TF网络) - 验证转录因子网络单独效果")
        print(f"  M3c: 单路GCN (仅GCN网络) - 验证共表达网络单独效果")
        print(f"  {'=' * 70}")
        print(f"\n💡 实验目的: 对比不同网络来源对基因表达预测的贡献")
        print(f"   - 对比 M3a 的融合权重是否倾向于某一网络")
        print(f"   - 对比单路网络 vs 双路融合的性能差异")
        print(f"   - 多随机种子验证结果的稳定性")
        print(f"{'=' * 80}")

        config = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'max_neighbors': MAX_NEIGHBORS,
            'seeds': self.seeds,
            'model_names': model_names,
            'index_file': INDEX_FILE,
            'labels_file': LABELS_FILE,
            'embedding_file': EMBEDDING_FILE,
            'tf_path': TF_PATH,
            'gcn_path': GCN_PATH,
            'timestamp': datetime.now().isoformat(),
            'version': 'V3_NetworkSourceAblation_MultiSeed'
        }

        config_file = os.path.join(self.output_dir, 'logs', 'experiment_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        data_dict = self.load_species_data()
        if data_dict is None:
            print("❌ 数据加载失败")
            return

        if self.analyze_only:
            return

        # 存储每个模型的所有种子结果
        for model_name in model_names:
            print(f"\n{'=' * 60}")
            print(f"🤖 训练模型: {model_name.upper()} - {self._get_model_description(model_name)}")
            print(f"{'=' * 60}")

            model_seed_results = []

            for seed_idx, seed in enumerate(self.seeds):
                print(f"\n{'=' * 50}")
                print(f"🔄 种子 {seed_idx + 1}/{len(self.seeds)}: seed={seed}")
                print(f"{'=' * 50}")

                # 设置随机种子（静默模式，避免重复打印）
                set_seed(seed, verbose=False)
                print(f"✅ 随机种子已设置为: {seed}")

                # 为每个种子独立划分数据集
                train_dataset, val_dataset, test_dataset = self.prepare_datasets(data_dict, seed)

                if train_dataset is None:
                    continue

                result = self.train_model(
                    train_dataset, val_dataset, test_dataset, data_dict, model_name, seed
                )

                if result is not None:
                    model_seed_results.append(result)
                    self.all_results.append(result)

            # 汇总该模型的多种子结果
            if model_seed_results:
                self.model_seed_results[model_name] = model_seed_results
                self._save_model_aggregated_results(model_name, model_seed_results)

        if self.all_results:
            self._generate_final_report()

    def _generate_final_report(self):
        """生成包含所有模型多种子统计的最终报告"""
        if not self.all_results:
            return

        print(f"\n{'=' * 80}")
        print(f"📊 最终实验报告 - 小龙虾消融实验 V3 (网络来源对比 + 多种子重复)")
        print(f"种子列表: {self.seeds}")
        print(f"{'=' * 80}")

        # 按模型汇总
        all_aggregated = {}
        for model_name, seed_results in self.model_seed_results.items():
            agg = aggregate_seed_results(seed_results)
            if agg:
                all_aggregated[model_name] = agg

        # 打印汇总表格
        print(f"\n📈 模型性能对比 (均值 ± 标准差):")
        print(f"{'-' * 120}")
        print(f"{'模型':<12} {'融合方式':<35} {'Test R²':<16} {'Test Pearson':<16} {'Test RMSE':<16}")
        print(f"{'-' * 120}")

        order = ['m3a', 'm3b', 'm3c']
        agg_names = {
            'm3a': '可学习权重融合 (TF+GCN双路)',
            'm3b': '单路GCN (仅TF网络)',
            'm3c': '单路GCN (仅GCN网络)'
        }

        comparison_data = []
        for model in order:
            if model in all_aggregated:
                agg = all_aggregated[model]
                test_r2 = agg['test']['r2']
                test_pearson = agg['test']['pearson_corr']
                test_rmse = agg['test']['rmse']

                print(f"{model.upper():<12} {agg_names[model]:<35} "
                      f"{test_r2['mean']:+.4f}±{test_r2['std']:.4f}   "
                      f"{test_pearson['mean']:+.4f}±{test_pearson['std']:.4f}   "
                      f"{test_rmse['mean']:.4f}±{test_rmse['std']:.4f}")

                comparison_data.append({
                    'model': model.upper(),
                    'aggregation': agg_names[model],
                    'test_r2_mean': test_r2['mean'],
                    'test_r2_std': test_r2['std'],
                    'test_pearson_mean': test_pearson['mean'],
                    'test_pearson_std': test_pearson['std'],
                    'test_rmse_mean': test_rmse['mean'],
                    'test_rmse_std': test_rmse['std']
                })

        # 打印融合权重信息（M3a）
        if 'm3a' in all_aggregated and 'final_tf_weight' in all_aggregated['m3a']:
            tf_w = all_aggregated['m3a']['final_tf_weight']
            gcn_w = all_aggregated['m3a']['final_gcn_weight']
            print(f"\n📊 M3a 可学习融合权重统计 (TF/GCN):")
            print(f"   TF权重: {tf_w['mean']:.4f} ± {tf_w['std']:.4f}")
            print(f"   GCN权重: {gcn_w['mean']:.4f} ± {gcn_w['std']:.4f}")

        print(f"\n{'=' * 80}")

        # 找出最佳模型
        if comparison_data:
            best_pearson_mean = -1
            best_model = None
            for model_data in comparison_data:
                if model_data['test_pearson_mean'] > best_pearson_mean:
                    best_pearson_mean = model_data['test_pearson_mean']
                    best_model = model_data['model']

            print(f"🏆 最佳模型 (基于测试集 Pearson 均值): {best_model}")

            # 计算相对增益
            baseline = next((m for m in comparison_data if m['model'] == 'M3A'), None)
            if baseline:
                baseline_pearson = baseline['test_pearson_mean']
                print(f"\n📊 相对于基线 (M3A - 可学习双路融合) 的性能变化:")
                for model_data in comparison_data:
                    if model_data['model'] != 'M3A':
                        gain = (model_data['test_pearson_mean'] - baseline_pearson) / abs(baseline_pearson) * 100
                        print(f"   {model_data['model']}: {gain:+.2f}%")

            print(f"\n💡 关键发现:")
            if 'm3a' in all_aggregated and 'final_tf_weight' in all_aggregated['m3a']:
                tf_w = all_aggregated['m3a']['final_tf_weight']
                gcn_w = all_aggregated['m3a']['final_gcn_weight']
                if tf_w['mean'] > gcn_w['mean']:
                    print(f"   ✓ 可学习权重更倾向于 TF 网络 ({tf_w['mean']:.3f} > {gcn_w['mean']:.3f})")
                else:
                    print(f"   ✓ 可学习权重更倾向于 GCN 网络 ({gcn_w['mean']:.3f} > {tf_w['mean']:.3f})")

        print(f"{'=' * 80}")

        # 保存最终报告
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            seeds_str = "_".join(map(str, self.seeds))
            report_file = os.path.join(self.output_dir, f'final_report_seeds_{seeds_str}.csv')
            comparison_df.to_csv(report_file, index=False)
            print(f"\n📄 完整报告已保存: {report_file}")

        # 保存所有种子的原始结果
        if self.all_results:
            all_results_df = pd.DataFrame([{
                'model': r['model_name'],
                'seed': r['random_seed'],
                'test_r2': r['test']['r2'],
                'test_pearson': r['test']['pearson_corr'],
                'test_spearman': r['test']['spearman_corr'],
                'test_rmse': r['test']['rmse'],
                'val_r2': r['validation']['r2'],
                'val_pearson': r['validation']['pearson_corr'],
                'best_epoch': r['best_epoch'],
                'final_tf_weight': r.get('final_tf_weight', None),
                'final_gcn_weight': r.get('final_gcn_weight', None)
            } for r in self.all_results])

            all_results_file = os.path.join(self.output_dir, f'all_seed_results.csv')
            all_results_df.to_csv(all_results_file, index=False)
            print(f"📄 所有种子原始结果已保存: {all_results_file}")


# =================================================================
# 主函数
# =================================================================
def main():
    parser = argparse.ArgumentParser(description='小龙虾GAT模型 - 网络来源消融实验 V3 (多种子重复实验)')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 789],
                        help='随机种子列表，例如: --seeds 42 123 456')
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--models', type=str, nargs='+',
                        default=['m3a', 'm3b', 'm3c'],
                        choices=['m3a', 'm3b', 'm3c'])
    parser.add_argument('--output_dir', type=str, default='Results_xr_net')
    parser.add_argument('--analyze_only', action='store_true')

    args = parser.parse_args()

    print("🔧 训练配置:")
    print(f"  物种: 小龙虾")
    print(f"  模型: {args.models}")
    print(f"  随机种子: {args.seeds}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  索引文件: {INDEX_FILE}")
    print(f"  标签文件: {LABELS_FILE}")
    print(f"  Embedding文件: {EMBEDDING_FILE}")
    print(f"  TF网络: {TF_PATH}")
    print(f"  GCN网络: {GCN_PATH}")
    print(f"\n🎯 消融实验 V3 核心改进:")
    print(f"  ✅ M3a: 可学习权重融合架构 (继承V2 M3)")
    print(f"  ✅ M3b: 单路TF网络 - 简化版")
    print(f"  ✅ M3c: 单路GCN网络 - 简化版")
    print(f"  ✅ 对比双路融合 vs 单路网络性能")
    print(f"  ✅ 观察可学习权重的收敛倾向")
    print(f"  ✅ 多随机种子重复实验: {len(args.seeds)} 个种子")

    trainer = CrayfishDeepGATAblation(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seeds=args.seeds,
        analyze_only=args.analyze_only
    )

    trainer.train(model_names=args.models)


if __name__ == "__main__":
    main()