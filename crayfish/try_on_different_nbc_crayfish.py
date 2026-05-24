"""
小龙虾基因表达预测 - 不确定性量化集成模型套件
统一架构：分图不同聚合方式 + 静态拼接融合

M3: Multi-Graph Concat (分图GCN + 静态拼接) - 基线
M4: Multi-GATv2 (分图GATv2 + 静态拼接)
M5: Edge-Gated DeepNT (显式边门控 + 静态拼接)
RGCN: 关系图卷积网络 + 静态拼接
GraphSAGE: SAGE聚合 + 静态拼接
GIN: GIN聚合 + 静态拼接
APPNP: APPNP传播 + 静态拼接
HAN: 分图GATv2 + 语义注意力 (融合方式不同，单独对比)

支持 Deep Ensemble 不确定性量化
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
    from torch_geometric.nn import GCNConv, GATv2Conv, RGCNConv, SAGEConv, GINConv, APPNP
    print("✅ PyTorch Geometric loaded successfully")
except ImportError as e:
    print(f"❌ Error: PyTorch Geometric not installed or import error: {e}")
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
INDEX_FILE = "processed_tf/gene_id_index.txt"
LABELS_FILE = "crayfish_labels.csv"
EMBEDDING_FILE = "crayfish_embeddings/crayfish_embeddings.pt"

# 网络文件路径
TF_PATH = "processed_tf/crayfish_tf_edge_index.pt"
GCN_PATH = "processed_gcn/crayfish_gcn_network.pt"

# Deep Ensemble 配置
ENSEMBLE_SEEDS = [42, 123, 2026, 888, 777]


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
# 统一融合模块 (所有静态拼接模型共享)
# =================================================================

class UnifiedFusion(nn.Module):
    """统一的静态拼接融合模块"""
    def __init__(self, hidden_dim=512, dropout=0.3):
        super(UnifiedFusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, t_info, g_info):
        concat_feat = torch.cat([t_info, g_info], dim=-1)
        return self.fusion(concat_feat)


# =================================================================
# M3: 分图GCN (均等聚合) - 基线
# =================================================================

class ModelM3_GCN(nn.Module):
    """M3: 分图GCN + 静态拼接融合 - 基线"""
    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3_GCN, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 双路GCN
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)
        # 统一融合
        self.fusion = UnifiedFusion(hidden_dim, dropout)
        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

    def forward(self, x, edge_index=None):
        s_feat = self.seq_proj(x)

        t_info = F.elu(self.tf_conv(s_feat, self.tf_sub)) if self.tf_sub is not None and self.tf_sub.numel() > 0 else s_feat
        g_info = F.elu(self.gcn_conv(s_feat, self.gcn_sub)) if self.gcn_sub is not None and self.gcn_sub.numel() > 0 else s_feat

        graph_feat = self.fusion(t_info, g_info)
        final_feat = s_feat + graph_feat

        return self.regressor(final_feat).squeeze(-1), {"model": "M3_GCN"}


# =================================================================
# M4: 分图GATv2 (分图注意力)
# =================================================================

class ModelM4_GATv2(nn.Module):
    """M4: 分图GATv2 + 静态拼接融合"""
    def __init__(self, input_dim=2560, head_dim=128, heads=4, dropout=0.3):
        super(ModelM4_GATv2, self).__init__()
        hidden_dim = head_dim * heads

        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 双路GATv2
        self.tf_gat = GATv2Conv(hidden_dim, head_dim, heads=heads, dropout=dropout,
                                add_self_loops=False, concat=True)
        self.gcn_gat = GATv2Conv(hidden_dim, head_dim, heads=heads, dropout=dropout,
                                 add_self_loops=False, concat=True)
        # 统一融合
        self.fusion = UnifiedFusion(hidden_dim, dropout)
        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

    def forward(self, x, edge_index=None):
        s_feat = self.seq_proj(x)

        t_info = F.elu(self.tf_gat(s_feat, self.tf_sub)) if self.tf_sub is not None and self.tf_sub.numel() > 0 else s_feat
        g_info = F.elu(self.gcn_gat(s_feat, self.gcn_sub)) if self.gcn_sub is not None and self.gcn_sub.numel() > 0 else s_feat

        graph_feat = self.fusion(t_info, g_info)
        final_feat = s_feat + graph_feat

        return self.regressor(final_feat).squeeze(-1), {"model": "M4_GATv2"}


# =================================================================
# M5: 边门控GCN (显式边门控)
# =================================================================

class ModelM5_EdgeGated(nn.Module):
    """M5: 边门控GCN + 静态拼接融合"""
    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM5_EdgeGated, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 边门控网络
        self.edge_gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # 双路GCN
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)
        # 统一融合
        self.fusion = UnifiedFusion(hidden_dim, dropout)
        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

    def _get_dynamic_weight(self, x, edge_index):
        if edge_index is None or edge_index.numel() == 0:
            return None
        src_x = x[edge_index[0]]
        dst_x = x[edge_index[1]]
        edge_feat = torch.cat([src_x, dst_x], dim=-1)
        return self.edge_gate_net(edge_feat).squeeze(-1)

    def forward(self, x, edge_index=None):
        s_feat = self.seq_proj(x)

        w_t = self._get_dynamic_weight(s_feat, self.tf_sub)
        w_g = self._get_dynamic_weight(s_feat, self.gcn_sub)

        t_info = F.elu(self.tf_conv(s_feat, self.tf_sub, edge_weight=w_t)) if self.tf_sub is not None and self.tf_sub.numel() > 0 else s_feat
        g_info = F.elu(self.gcn_conv(s_feat, self.gcn_sub, edge_weight=w_g)) if self.gcn_sub is not None and self.gcn_sub.numel() > 0 else s_feat

        graph_feat = self.fusion(t_info, g_info)
        final_feat = s_feat + graph_feat

        return self.regressor(final_feat).squeeze(-1), {"model": "M5_EdgeGated"}


# =================================================================
# RGCN: 关系图卷积网络
# =================================================================

class Model_RGCN(nn.Module):
    """RGCN: 关系图卷积 + 静态拼接融合"""
    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(Model_RGCN, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 2个关系：0为TF，1为GCN
        self.conv = RGCNConv(hidden_dim, hidden_dim, num_relations=2)
        # 统一融合
        self.fusion = UnifiedFusion(hidden_dim, dropout)
        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None
        self._edge_type = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub
        self._build_edge_type()

    def _build_edge_type(self):
        if self.tf_sub is not None and self.gcn_sub is not None:
            device = self.tf_sub.device if self.tf_sub.numel() > 0 else self.gcn_sub.device
            tf_edges = self.tf_sub.shape[1] if self.tf_sub.numel() > 0 else 0
            gcn_edges = self.gcn_sub.shape[1] if self.gcn_sub.numel() > 0 else 0

            rel_tf = torch.zeros(tf_edges, dtype=torch.long, device=device)
            rel_gcn = torch.ones(gcn_edges, dtype=torch.long, device=device)
            self._edge_type = torch.cat([rel_tf, rel_gcn])
        else:
            self._edge_type = None

    def _get_separate_embeddings(self, h, edge_index, edge_type):
        """分别获取两个关系下的节点表示"""
        t_info, g_info = h.clone(), h.clone()

        if (edge_type == 0).any():
            tf_mask = (edge_type == 0)
            tf_edges = edge_index[:, tf_mask]
            t_info = F.elu(self.conv(h, tf_edges, edge_type[tf_mask]))

        if (edge_type == 1).any():
            gcn_mask = (edge_type == 1)
            gcn_edges = edge_index[:, gcn_mask]
            g_info = F.elu(self.conv(h, gcn_edges, edge_type[gcn_mask]))

        return t_info, g_info

    def forward(self, x, edge_index=None):
        h = self.seq_proj(x)

        if self.tf_sub is not None and self.gcn_sub is not None:
            if self.tf_sub.numel() > 0 and self.gcn_sub.numel() > 0:
                edge_index = torch.cat([self.tf_sub, self.gcn_sub], dim=1)
                self._build_edge_type()
            elif self.tf_sub.numel() > 0:
                edge_index = self.tf_sub
                self._edge_type = torch.zeros(self.tf_sub.shape[1], dtype=torch.long, device=x.device)
            elif self.gcn_sub.numel() > 0:
                edge_index = self.gcn_sub
                self._edge_type = torch.ones(self.gcn_sub.shape[1], dtype=torch.long, device=x.device)
            else:
                edge_index = torch.zeros((2, 0), device=x.device, dtype=torch.long)
                self._edge_type = torch.zeros(0, dtype=torch.long, device=x.device)

            if edge_index.numel() > 0:
                t_info, g_info = self._get_separate_embeddings(h, edge_index, self._edge_type)
            else:
                t_info = g_info = h
        else:
            t_info = g_info = h

        graph_feat = self.fusion(t_info, g_info)
        final_feat = h + graph_feat

        return self.regressor(final_feat).squeeze(-1), {"model": "RGCN"}


# =================================================================
# GraphSAGE: 归纳采样
# =================================================================

class Model_GraphSAGE(nn.Module):
    """GraphSAGE: SAGE聚合 + 静态拼接融合"""
    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(Model_GraphSAGE, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.tf_sage = SAGEConv(hidden_dim, hidden_dim)
        self.gcn_sage = SAGEConv(hidden_dim, hidden_dim)
        self.fusion = UnifiedFusion(hidden_dim, dropout)
        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

    def forward(self, x, edge_index=None):
        h = self.seq_proj(x)

        t_info = F.elu(self.tf_sage(h, self.tf_sub)) if self.tf_sub is not None and self.tf_sub.numel() > 0 else h
        g_info = F.elu(self.gcn_sage(h, self.gcn_sub)) if self.gcn_sub is not None and self.gcn_sub.numel() > 0 else h

        graph_feat = self.fusion(t_info, g_info)
        final_feat = h + graph_feat

        return self.regressor(final_feat).squeeze(-1), {"model": "GraphSAGE"}


# =================================================================
# GIN: 图同构网络
# =================================================================

class Model_GIN(nn.Module):
    """GIN: GIN聚合 + 静态拼接融合"""
    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(Model_GIN, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.tf_gin = GINConv(mlp)
        self.gcn_gin = GINConv(mlp)
        self.fusion = UnifiedFusion(hidden_dim, dropout)
        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

    def forward(self, x, edge_index=None):
        h = self.seq_proj(x)

        t_info = F.elu(self.tf_gin(h, self.tf_sub)) if self.tf_sub is not None and self.tf_sub.numel() > 0 else h
        g_info = F.elu(self.gcn_gin(h, self.gcn_sub)) if self.gcn_sub is not None and self.gcn_sub.numel() > 0 else h

        graph_feat = self.fusion(t_info, g_info)
        final_feat = h + graph_feat

        return self.regressor(final_feat).squeeze(-1), {"model": "GIN"}


# =================================================================
# APPNP: PPR传播增强
# =================================================================

class Model_APPNP(nn.Module):
    """APPNP: APPNP传播 + 静态拼接融合"""
    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3, appnp_k=10, appnp_alpha=0.1):
        super(Model_APPNP, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.tf_transform = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.gcn_transform = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.tf_appnp = APPNP(K=appnp_k, alpha=appnp_alpha)
        self.gcn_appnp = APPNP(K=appnp_k, alpha=appnp_alpha)
        self.fusion = UnifiedFusion(hidden_dim, dropout)
        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

    def forward(self, x, edge_index=None):
        h = self.seq_proj(x)

        tf_h0 = self.tf_transform(h)
        gcn_h0 = self.gcn_transform(h)

        t_info = self.tf_appnp(tf_h0, self.tf_sub) if self.tf_sub is not None and self.tf_sub.numel() > 0 else tf_h0
        g_info = self.gcn_appnp(gcn_h0, self.gcn_sub) if self.gcn_sub is not None and self.gcn_sub.numel() > 0 else gcn_h0

        graph_feat = self.fusion(t_info, g_info)
        final_feat = h + graph_feat

        return self.regressor(final_feat).squeeze(-1), {"model": "APPNP"}


# =================================================================
# HAN: 真正的异质图注意力网络 (分图GAT + 语义注意力) - 融合方式不同
# =================================================================

class Model_HAN_True(nn.Module):
    """真正的HAN: 节点级GATv2 + 语义级注意力融合 (融合方式与上述模型不同)"""
    def __init__(self, input_dim=2560, head_dim=128, heads=4, dropout=0.3):
        super(Model_HAN_True, self).__init__()
        hidden_dim = head_dim * heads

        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 节点级：双路独立的GATv2（分图注意力）
        self.tf_gat = GATv2Conv(hidden_dim, head_dim, heads=heads, dropout=dropout,
                                add_self_loops=False, concat=True)
        self.gcn_gat = GATv2Conv(hidden_dim, head_dim, heads=heads, dropout=dropout,
                                 add_self_loops=False, concat=True)

        # 语义级：注意力融合
        self.semantic_attn = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )

        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, tf_sub, gcn_sub):
        self.tf_sub, self.gcn_sub = tf_sub, gcn_sub

    def forward(self, x, edge_index=None):
        h = self.seq_proj(x)

        # 节点级：分图GATv2聚合
        z_tf = F.elu(self.tf_gat(h, self.tf_sub)) if self.tf_sub is not None and self.tf_sub.numel() > 0 else h
        z_gcn = F.elu(self.gcn_gat(h, self.gcn_sub)) if self.gcn_sub is not None and self.gcn_sub.numel() > 0 else h

        # 堆叠两个语义表示
        stack_z = torch.stack([z_tf, z_gcn], dim=1)  # [N, 2, Dim]

        # 语义级注意力
        attn_scores = self.semantic_attn(stack_z).squeeze(-1)  # [N, 2]
        semantic_weights = F.softmax(attn_scores, dim=1)  # [N, 2]

        # 加权融合
        fused = (stack_z * semantic_weights.unsqueeze(-1)).sum(dim=1)

        weights_info = {
            "model": "HAN_True",
            "semantic_weights": semantic_weights.mean(dim=0).detach().cpu(),
            "tf_weight": semantic_weights[:, 0].mean().item(),
            "gcn_weight": semantic_weights[:, 1].mean().item()
        }

        return self.regressor(fused).squeeze(-1), weights_info


# =================================================================
# 模型工厂函数
# =================================================================

def build_model(model_name, input_dim=2560, dropout=0.3, **kwargs):
    """模型工厂 - 所有模型统一使用分图架构 + 静态拼接融合（HAN除外）"""
    if model_name == 'm3':
        return ModelM3_GCN(input_dim=input_dim, dropout=dropout)
    elif model_name == 'm4':
        return ModelM4_GATv2(input_dim=input_dim, dropout=dropout)
    elif model_name == 'm5':
        return ModelM5_EdgeGated(input_dim=input_dim, dropout=dropout)
    elif model_name == 'rgcn':
        return Model_RGCN(input_dim=input_dim, dropout=dropout)
    elif model_name == 'sage':
        return Model_GraphSAGE(input_dim=input_dim, dropout=dropout)
    elif model_name == 'gin':
        return Model_GIN(input_dim=input_dim, dropout=dropout)
    elif model_name == 'appnp':
        return Model_APPNP(input_dim=input_dim, dropout=dropout)
    elif model_name == 'han':
        return Model_HAN_True(input_dim=input_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}. 可选: m3, m4, m5, rgcn, sage, gin, appnp, han")


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
                if len(network_gene_list) != len(set(network_gene_list)):
                    print(f"⚠️ {network_name}网络的gene_list存在重复ID，可能引入索引歧义")
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
        coverage = matched_count / max(1, len(network_gene_list))
        if coverage < 0.95:
            print(f"❌ {network_name}网络覆盖率过低: {coverage:.2%} (<95%)，请检查索引文件是否一致")
            return None

        if matched_count == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        # 过滤和映射边
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
        adj_dict = {'tf': defaultdict(list), 'gcn': defaultdict(list)}

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

        set_seed(seed)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                           weight_decay=1e-3, betas=(0.9, 0.999))
        self.criterion = nn.HuberLoss(reduction='none', delta=1.0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=min_lr
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

        # 存储最后一次测试预测结果
        self.last_test_preds = None
        self.last_test_targets = None
        self.last_test_gene_ids = None

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
            tf_sub, _ = subgraph(unique_nodes_cpu, self.tf_edge_index.cpu(),
                                 relabel_nodes=True, num_nodes=self.num_nodes)
            tf_sub = tf_sub.to(device)
        else:
            tf_sub = torch.zeros((2, 0), device=device, dtype=torch.long)

        if self.gcn_edge_index is not None and self.gcn_edge_index.numel() > 0:
            gcn_sub, _ = subgraph(unique_nodes_cpu, self.gcn_edge_index.cpu(),
                                  relabel_nodes=True, num_nodes=self.num_nodes)
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

        epoch_gate_means = []
        epoch_net_weights = []

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

                if weights and 'network_importance' in weights:
                    epoch_net_weights.append(weights['network_importance'].cpu().numpy())

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

        if epoch_net_weights:
            self.network_importance_history.append(np.mean(epoch_net_weights, axis=0))

        return avg_loss, train_pearson

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_valid_samples = 0
        all_preds = []
        all_targets = []
        all_gene_ids = []

        val_net_weights = []

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

                outputs, weights = self.model(x)
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

                if weights and 'semantic_weights' in weights:
                    val_net_weights.append(weights['semantic_weights'].cpu().numpy())
                elif weights and 'network_importance' in weights:
                    val_net_weights.append(weights['network_importance'].cpu().numpy())

                batch_gene_ids = batch['gene_ids']
                for i, valid in enumerate(has_expression.cpu().numpy()):
                    if valid:
                        all_gene_ids.append(batch_gene_ids[i])

        avg_loss = total_loss / num_valid_samples if num_valid_samples > 0 else float('inf')

        if val_net_weights:
            avg_net_weights = np.mean(val_net_weights, axis=0)
            if len(avg_net_weights) == 2:
                print(f"\n   🎯 网络重要性 - TF: {avg_net_weights[0]:.4f}, GCN: {avg_net_weights[1]:.4f}")

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
                print(f"  📊 Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}, Train Pearson: {train_pearson:.4f}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_loss,
            'seed': self.seed,
            'network_importance_history': self.network_importance_history,
            'fusion_gate_history': self.fusion_gate_history
        }

    def predict_test(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_gene_ids = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Test Prediction", leave=False)
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

                all_preds.extend(valid_outputs.cpu().numpy())
                all_targets.extend(valid_targets.cpu().numpy())

                batch_gene_ids = batch['gene_ids']
                for i, valid in enumerate(has_expression.cpu().numpy()):
                    if valid:
                        all_gene_ids.append(batch_gene_ids[i])

        self.last_test_preds = np.array(all_preds)
        self.last_test_targets = np.array(all_targets)
        self.last_test_gene_ids = all_gene_ids
        return self.last_test_preds, self.last_test_targets, self.last_test_gene_ids


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
# 主训练类 - 支持 Deep Ensemble
# =================================================================

class CrayfishDeepGATTrainer:
    def __init__(self,
                 output_dir='Results_Crayfish_DeepEnsemble',
                 batch_size=32,
                 epochs=100,
                 learning_rate=1e-4,
                 patience=15,
                 seed=42,
                 analyze_only=False):

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed
        self.analyze_only = analyze_only

        set_seed(seed)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'ensemble'), exist_ok=True)

        self.all_results = []
        self.ensemble_results = {}

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

        index_gene_ids, gene_to_idx = load_gene_index(INDEX_FILE)
        if index_gene_ids is None:
            return None
        print(f"   索引文件基因数: {len(index_gene_ids)}")

        embed_data = load_nt_embeddings(EMBEDDING_FILE)
        if embed_data is None:
            return None
        print(f"   Embeddings基因数: {len(embed_data['gene_ids'])}")

        embed_gene_set = set(embed_data['gene_ids'])
        index_gene_set = set(index_gene_ids)
        common_genes = index_gene_set.intersection(embed_gene_set)

        if len(common_genes) == 0:
            print(f"❌ 索引文件和embeddings没有共同基因")
            return None

        print(f"   共同基因数: {len(common_genes)}")

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

        valid_gene_to_idx = {gid: i for i, gid in enumerate(new_gene_ids)}
        num_valid_nodes = len(new_gene_ids)

        expression_tensor, valid_expr_genes = load_expression_data(LABELS_FILE, new_gene_ids)
        if expression_tensor is None:
            return None

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

    def prepare_datasets(self, data_dict, split_seed=None):
        print(f"\n  准备训练/验证/测试数据集...")

        gene_ids = data_dict['gene_ids']
        total_genes = len(gene_ids)
        indices = list(range(total_genes))

        current_seed = self.seed if split_seed is None else split_seed
        set_seed(current_seed)

        train_indices, temp_indices = train_test_split(
            indices, train_size=TRAIN_RATIO, random_state=current_seed, shuffle=True
        )
        val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_indices, test_indices = train_test_split(
            temp_indices, train_size=val_ratio_adjusted, random_state=current_seed, shuffle=True
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
            data_dict['expression_values'], data_dict['gene_ids'], train_genes, seed=current_seed
        )
        val_dataset = GATDeepCREDataset(
            data_dict['embeddings'],
            data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
            data_dict['expression_values'], data_dict['gene_ids'], val_genes, seed=current_seed + 1
        )
        test_dataset = GATDeepCREDataset(
            data_dict['embeddings'],
            data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
            data_dict['expression_values'], data_dict['gene_ids'], test_genes, seed=current_seed + 2
        )

        return train_dataset, val_dataset, test_dataset

    def _get_model_description(self, model_name):
        descriptions = {
            'm3': 'M3: 分图GCN (均等聚合) - 基线',
            'm4': 'M4: 分图GATv2 (分图注意力)',
            'm5': 'M5: 边门控GCN (显式边门控)',
            'rgcn': 'RGCN: 关系图卷积网络',
            'sage': 'GraphSAGE: 归纳采样',
            'gin': 'GIN: 图同构网络',
            'appnp': 'APPNP: PPR传播增强',
            'han': 'HAN: 分图GAT + 语义注意力 (融合方式不同)'
        }
        return descriptions.get(model_name, 'Unknown')

    def train_single_run(self, data_dict, model_name, run_seed, run_idx):
        print(f"\n  🔬 训练运行 {run_idx + 1} - 模型 {model_name.upper()}, 种子: {run_seed}")
        print(f"  {'-' * 50}")

        train_dataset, val_dataset, test_dataset = self.prepare_datasets(data_dict, split_seed=self.seed)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False, drop_last=True,
                                  collate_fn=gat_collate_fn,
                                  generator=torch.Generator().manual_seed(run_seed))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False,
                                num_workers=0, pin_memory=False, collate_fn=gat_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size * 2, shuffle=False,
                                 num_workers=0, pin_memory=False, collate_fn=gat_collate_fn)

        input_dim = data_dict['embeddings'].size(1)
        model = build_model(model_name, input_dim=input_dim)
        print(f"  ✅ 成功构建模型: {model_name.upper()}, 输入维度: {input_dim}")
        print(f"  📊 模型参数: {sum(p.numel() for p in model.parameters()):,}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  🔧 使用设备: {device}")

        trainer = GATDeepCRETrainer(model, model_name, device=device,
                                    learning_rate=self.learning_rate, patience=self.patience, seed=run_seed)
        trainer.set_graph_data(data_dict['embeddings'],
                               data_dict['tf_edge_index'], data_dict['gcn_edge_index'])

        print(f"  🚀 开始训练...")
        training_history = trainer.train(train_loader, val_loader, epochs=self.epochs)

        test_preds, test_targets, test_gene_ids = trainer.predict_test(test_loader)

        val_loss, val_preds, val_targets, val_gene_ids = trainer.validate(val_loader)
        test_loss, _, _, _ = trainer.validate(test_loader)

        val_evaluation = evaluate_regression(val_targets, val_preds)
        test_evaluation = evaluate_regression(test_targets, test_preds)

        results = {
            'run_idx': run_idx,
            'run_seed': run_seed,
            'species': 'crayfish',
            'model_name': model_name,
            'model_description': self._get_model_description(model_name),
            'best_epoch': training_history['best_epoch'] + 1,
            'best_val_loss': float(training_history['best_val_loss']),
            'num_train_genes': len(train_dataset),
            'num_val_genes': len(val_dataset),
            'num_test_genes': len(test_dataset),
            'model_params': sum(p.numel() for p in model.parameters()),
            'max_neighbors': MAX_NEIGHBORS,
            'validation': val_evaluation,
            'test': test_evaluation,
            'test_predictions': test_preds.tolist(),
            'test_targets': test_targets.tolist(),
            'test_gene_ids': test_gene_ids,
            'network_importance_history': self.convert_for_json(training_history.get('network_importance_history', [])),
            'fusion_gate_history': self.convert_for_json(training_history.get('fusion_gate_history', []))
        }

        return results, test_preds, test_targets, test_gene_ids

    def run_ensemble(self, model_names=['m3', 'm4', 'm5', 'rgcn', 'sage', 'gin', 'appnp', 'han'], num_seeds=5):
        print(f"\n{'=' * 80}")
        print(f"🦞 开始 Deep Ensemble 不确定性量化分析")
        print(f"种子数量: {num_seeds}")
        print(f"模型: {model_names}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'=' * 80}")

        print(f"\n📋 模型架构说明:")
        print(f"  {'=' * 65}")
        print(f"  M3:     分图GCN (均等聚合) + 静态拼接 - 基线")
        print(f"  M4:     分图GATv2 (分图注意力) + 静态拼接")
        print(f"  M5:     边门控GCN (显式边门控) + 静态拼接")
        print(f"  RGCN:   关系图卷积 + 静态拼接")
        print(f"  SAGE:   GraphSAGE + 静态拼接")
        print(f"  GIN:    图同构网络 + 静态拼接")
        print(f"  APPNP:  PPR传播增强 + 静态拼接")
        print(f"  HAN:    分图GAT + 语义注意力 (融合方式不同，单独对比)")
        print(f"  {'=' * 65}")

        data_dict = self.load_species_data()
        if data_dict is None:
            print("❌ 数据加载失败")
            return

        if self.analyze_only:
            return

        ensemble_seeds = ENSEMBLE_SEEDS[:num_seeds]

        for model_name in model_names:
            print(f"\n{'=' * 60}")
            print(f"🚀 开始对模型 {model_name.upper()} 进行 {num_seeds} 次集成训练...")
            print(f"{'=' * 60}")

            all_test_preds = []
            all_test_targets = None
            all_test_gene_ids = None
            all_run_results = []

            for i, current_seed in enumerate(ensemble_seeds):
                print(f"\n--- [Run {i + 1}/{num_seeds}] Seed: {current_seed} ---")
                result, test_preds, test_targets, test_gene_ids = self.train_single_run(
                    data_dict, model_name, current_seed, i
                )

                all_run_results.append(result)
                all_test_preds.append(test_preds)

                if all_test_targets is None:
                    all_test_targets = test_targets
                    all_test_gene_ids = test_gene_ids
                else:
                    assert all_test_gene_ids == test_gene_ids, "测试集基因顺序不一致"

                self._save_single_run_predictions(model_name, i, current_seed,
                                                  test_gene_ids, test_targets, test_preds, result)

            all_test_preds = np.array(all_test_preds)
            final_mean = np.mean(all_test_preds, axis=0)
            final_std = np.std(all_test_preds, axis=0)

            ensemble_metrics = evaluate_regression(all_test_targets, final_mean)

            uncertainty_stats = {
                'mean_std': float(np.mean(final_std)),
                'median_std': float(np.median(final_std)),
                'min_std': float(np.min(final_std)),
                'max_std': float(np.max(final_std)),
                'std_of_stds': float(np.std(final_std))
            }

            uq_df = pd.DataFrame({
                'gene_id': all_test_gene_ids,
                'true_value': all_test_targets,
                'ensemble_mean': final_mean,
                'uncertainty_std': final_std,
                'cv_ratio': final_std / (np.abs(final_mean) + 1e-8)
            })
            for i in range(num_seeds):
                uq_df[f'run_{i + 1}_pred'] = all_test_preds[i]

            uq_file = os.path.join(self.output_dir, 'ensemble', f'{model_name}_ensemble_uq.csv')
            uq_df.to_csv(uq_file, index=False)

            ensemble_summary = {
                'model_name': model_name,
                'num_seeds': num_seeds,
                'seeds_used': ensemble_seeds,
                'ensemble_metrics': ensemble_metrics,
                'uncertainty_statistics': uncertainty_stats,
                'individual_run_metrics': [
                    {
                        'run_idx': r['run_idx'],
                        'run_seed': r['run_seed'],
                        'test_pearson': r['test']['pearson_corr'],
                        'test_spearman': r['test'].get('spearman_corr', 0.0),
                        'test_r2': r['test']['r2'],
                        'test_rmse': r['test']['rmse'],
                        'test_mae': r['test']['mae']
                    } for r in all_run_results
                ]
            }

            summary_file = os.path.join(self.output_dir, 'ensemble', f'{model_name}_ensemble_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(self.convert_for_json(ensemble_summary), f, indent=2)

            print(f"\n✅ {model_name.upper()} 集成完成！")
            print(f"📊 集成性能 (Pearson): {ensemble_metrics['pearson_corr']:.6f}")
            print(f"📊 集成性能 (R²): {ensemble_metrics['r2']:.6f}")
            print(f"📊 平均不确定性 (Mean STD): {uncertainty_stats['mean_std']:.6f}")

            self.ensemble_results[model_name] = ensemble_summary
            self.all_results.extend(all_run_results)

        self._generate_ensemble_report()

    def _save_single_run_predictions(self, model_name, run_idx, seed, gene_ids, targets, preds, metrics):
        pred_dir = os.path.join(self.output_dir, 'predictions', model_name)
        os.makedirs(pred_dir, exist_ok=True)

        pred_df = pd.DataFrame({
            'gene_id': gene_ids,
            'true_expression': targets,
            'predicted_expression': preds,
            'run_idx': run_idx,
            'seed': seed
        })
        pred_file = os.path.join(pred_dir, f'{model_name}_run{run_idx}_seed{seed}_predictions.csv')
        pred_df.to_csv(pred_file, index=False)

        metrics_file = os.path.join(pred_dir, f'{model_name}_run{run_idx}_seed{seed}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.convert_for_json(metrics), f, indent=2)

    def _generate_ensemble_report(self):
        if not self.ensemble_results:
            return

        print(f"\n{'=' * 80}")
        print(f"📊 最终 Deep Ensemble 不确定性量化报告")
        print(f"{'=' * 80}")

        report_data = []
        for model_name, result in self.ensemble_results.items():
            run_pearsons = [r['test_pearson'] for r in result['individual_run_metrics']]
            run_spearmans = [r.get('test_spearman', 0.0) for r in result['individual_run_metrics']]
            run_r2s = [r['test_r2'] for r in result['individual_run_metrics']]
            run_rmses = [r['test_rmse'] for r in result['individual_run_metrics']]
            run_maes = [r['test_mae'] for r in result['individual_run_metrics']]

            report_data.append({
                'model_name': model_name.upper(),
                'model_description': self._get_model_description(model_name),
                'ensemble_pearson': result['ensemble_metrics']['pearson_corr'],
                'ensemble_spearman': result['ensemble_metrics'].get('spearman_corr', 0.0),
                'ensemble_r2': result['ensemble_metrics']['r2'],
                'ensemble_rmse': result['ensemble_metrics']['rmse'],
                'ensemble_mae': result['ensemble_metrics']['mae'],
                'pearson_mean': np.mean(run_pearsons),
                'pearson_std': np.std(run_pearsons),
                'spearman_mean': np.mean(run_spearmans),
                'spearman_std': np.std(run_spearmans),
                'r2_mean': np.mean(run_r2s),
                'r2_std': np.std(run_r2s),
                'rmse_mean': np.mean(run_rmses),
                'rmse_std': np.std(run_rmses),
                'mae_mean': np.mean(run_maes),
                'mae_std': np.std(run_maes),
                'mean_uncertainty': result['uncertainty_statistics']['mean_std'],
                'std_uncertainty': result['uncertainty_statistics']['std_of_stds'],
                'num_seeds': result['num_seeds'],
            })

        summary_df = pd.DataFrame(report_data)
        summary_df = summary_df.sort_values('ensemble_pearson', ascending=False)

        print(f"\n📈 模型集成性能对比 (按集成Pearson相关系数排序):")
        print(f"{'=' * 130}")
        print(f"{'模型':<10} {'集成Pearson':<14} {'集成R²':<12} {'单次Pearson':<16} {'单次R²':<12} {'平均不确定性':<14}")
        print(f"{'-' * 130}")

        for _, row in summary_df.iterrows():
            print(f"{row['model_name']:<10} "
                  f"{row['ensemble_pearson']:<14.6f} "
                  f"{row['ensemble_r2']:<12.6f} "
                  f"{row['pearson_mean']:.4f}±{row['pearson_std']:.4f}   "
                  f"{row['r2_mean']:.4f}±{row['r2_std']:.4f}   "
                  f"{row['mean_uncertainty']:<14.6f}")

        print(f"\n{'=' * 80}")
        print(f"🏆 最佳集成模型: {summary_df.iloc[0]['model_name']}")
        print(f"   ├─ 集成 Pearson:  {summary_df.iloc[0]['ensemble_pearson']:.6f}")
        print(f"   ├─ 集成 R²:       {summary_df.iloc[0]['ensemble_r2']:.6f}")
        print(f"   ├─ 单次 Pearson:  {summary_df.iloc[0]['pearson_mean']:.4f} ± {summary_df.iloc[0]['pearson_std']:.4f}")
        print(f"   └─ 平均不确定性:  {summary_df.iloc[0]['mean_uncertainty']:.6f}")

        report_file = os.path.join(self.output_dir, 'ensemble', 'ensemble_final_report.csv')
        summary_df.to_csv(report_file, index=False)
        print(f"\n📄 详细报告已保存: {report_file}")

    def train(self, model_names=['m3', 'm4', 'm5', 'rgcn', 'sage', 'gin', 'appnp', 'han'], num_seeds=5):
        self.run_ensemble(model_names=model_names, num_seeds=num_seeds)


# =================================================================
# 主函数
# =================================================================

def main():
    parser = argparse.ArgumentParser(description='小龙虾Deep Ensemble - 不确定性量化分析')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--models', type=str, nargs='+',
                        default=['m3', 'm4', 'm5', 'rgcn', 'sage', 'gin', 'appnp', 'han'],
                        choices=['m3', 'm4', 'm5', 'rgcn', 'sage', 'gin', 'appnp', 'han'])
    parser.add_argument('--output_dir', type=str, default='Results_Crayfish_DeepEnsemble')
    parser.add_argument('--analyze_only', action='store_true')

    args = parser.parse_args()

    print("🔧 Deep Ensemble 训练配置:")
    print(f"  物种: 小龙虾")
    print(f"  模型: {args.models}")
    print(f"  Ensemble种子数量: {args.num_seeds}")
    print(f"  基础种子: {args.seed}")
    print(f"  输出目录: {args.output_dir}")

    trainer = CrayfishDeepGATTrainer(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        analyze_only=args.analyze_only
    )

    trainer.train(model_names=args.models, num_seeds=args.num_seeds)


if __name__ == "__main__":
    main()