"""
HG-NT (Human/Mouse Graph Network with Transformer)
基于 Multi-Graph Concat 架构 - GCN均等聚合
支持多种子重复实验，输出性能指标的均值和标准差
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

# 文件路径配置
PPI_PATH_TEMPLATE = "processed_ppi/{species}_ppi_edge_index.pt"
TF_PATH_TEMPLATE = "processed_tf/{species}_tf_edge_index.pt"
GCN_PATH_TEMPLATE = "processed_gcn/{species}_gcn_network_aligned.pt"
EMBEDDING_DIR = "processed_features"
LABELS_DIR = "processed_labels"

# 5个随机种子
SEEDS = [42, 123, 456, 789, 1024]


# =================================================================
# 深度回归头
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
# HG-NT: Multi-Graph Concat (分图架构 - 静态拼接融合)
# =================================================================

class ModelHGNT(nn.Module):
    """HG-NT: 分图静态拼接 - GCN均等聚合"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelHGNT, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 三路GCN - 均等聚合（边权重固定为1）
        self.ppi_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False)
        # 静态拼接融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.regressor = DeepRegressor(hidden_dim, dropout)
        self.ppi_sub = self.tf_sub = self.gcn_sub = None

    def set_subgraphs(self, ppi_sub, tf_sub, gcn_sub):
        self.ppi_sub, self.tf_sub, self.gcn_sub = ppi_sub, tf_sub, gcn_sub

    def forward(self, x, edge_index=None):
        s_feat = self.seq_proj(x)

        # 三路图卷积 - 均等聚合
        if self.ppi_sub is not None and self.ppi_sub.numel() > 0:
            p_info = F.elu(self.ppi_conv(s_feat, self.ppi_sub))
        else:
            p_info = s_feat

        if self.tf_sub is not None and self.tf_sub.numel() > 0:
            t_info = F.elu(self.tf_conv(s_feat, self.tf_sub))
        else:
            t_info = s_feat

        if self.gcn_sub is not None and self.gcn_sub.numel() > 0:
            g_info = F.elu(self.gcn_conv(s_feat, self.gcn_sub))
        else:
            g_info = s_feat

        # 静态拼接融合
        concat_feat = torch.cat([p_info, t_info, g_info], dim=-1)
        graph_feat = self.fusion(concat_feat)

        # 残差连接
        final_feat = s_feat + graph_feat

        return self.regressor(final_feat).squeeze(-1)


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
# 加载网络文件
# =================================================================

def load_network(species, network_type, num_nodes=None):
    if network_type == 'ppi':
        path = PPI_PATH_TEMPLATE.format(species=species)
    elif network_type == 'tf':
        path = TF_PATH_TEMPLATE.format(species=species)
    elif network_type == 'gcn':
        path = GCN_PATH_TEMPLATE.format(species=species)
    else:
        return None

    if not os.path.exists(path):
        print(f"⚠️ {network_type.upper()}网络不存在: {path}")
        return None

    try:
        data = torch.load(path, map_location='cpu', weights_only=False)

        if isinstance(data, dict):
            if 'edge_index' in data:
                edge_index = data['edge_index']
            elif 'edges' in data:
                edge_index = data['edges']
            else:
                print(f"⚠️ {network_type.upper()}网络字典格式不支持")
                return None
        else:
            edge_index = data

        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        if num_nodes is not None:
            max_idx = edge_index.max().item()
            if max_idx >= num_nodes:
                valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                edge_index = edge_index[:, valid_mask]

        print(f"   {network_type.upper()}网络: {edge_index.shape[1]} 条边")
        return edge_index

    except Exception as e:
        print(f"❌ 加载{network_type}网络失败: {e}")
        return None


# =================================================================
# 构建表达值张量
# =================================================================

def build_expression_tensor(gene_ids, expr_dict):
    expression_values = []
    for gene_id in gene_ids:
        expression_values.append(expr_dict.get(gene_id, float('nan')))
    return torch.tensor(expression_values, dtype=torch.float32)


# =================================================================
# 数据集类
# =================================================================

class HGNTDataset(Dataset):
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
                expr_val = self.expression_tensor[idx]
                has_expr = not torch.isnan(expr_val)
                self.valid_expression_mask.append(has_expr)

        print(f"\n📊 数据集构建统计:")
        print(f"   有效基因数量: {len(self.valid_genes)}/{len(target_genes)}")
        print(f"   有表达值的有效基因: {sum(self.valid_expression_mask)}")

        self.all_neighbors = self._precompute_all_neighbors()

    def _precompute_all_neighbors(self):
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

        for node_idx in range(self.num_nodes):
            neighbors_dict = {}

            for net_name in ['ppi', 'tf', 'gcn']:
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
# HG-NT训练器
# =================================================================

class HGNTTrainer:
    def __init__(self, model, device='cpu', learning_rate=1e-4,
                 patience=15, min_lr=1e-6, seed=42):
        self.model = model.to(device)
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
        unique_nodes_cpu = unique_nodes.cpu()

        if self.ppi_edge_index is not None and self.ppi_edge_index.numel() > 0:
            ppi_sub, _ = subgraph(
                unique_nodes_cpu,
                self.ppi_edge_index.cpu(),
                relabel_nodes=True,
                num_nodes=self.num_nodes
            )
            ppi_sub = ppi_sub.to(device)
        else:
            ppi_sub = torch.zeros((2, 0), device=device, dtype=torch.long)

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

        return ppi_sub, tf_sub, gcn_sub

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
        node_mapping = {node.item(): i for i, node in enumerate(unique_nodes)}

        ppi_sub, tf_sub, gcn_sub = self._extract_subgraphs(unique_nodes, self.device)

        if hasattr(self.model, 'set_subgraphs'):
            self.model.set_subgraphs(ppi_sub, tf_sub, gcn_sub)

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
                    outputs = self.model(x)

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

                outputs = self.model(x)

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

            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.counter = 0
                self.best_model_state = self.model.state_dict().copy()
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
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_loss,
            'seed': self.seed
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
    else:
        results['pearson_corr'] = 0.0
        results['spearman_corr'] = 0.0

    results['num_samples'] = int(len(y_true))
    return results


# =================================================================
# HG-NT主训练类
# =================================================================

class HGNTExperiment:
    def __init__(self,
                 output_dir='Results_HGNT',
                 batch_size=32,
                 epochs=100,
                 learning_rate=1e-4,
                 patience=15,
                 train_ratio=0.7,
                 val_ratio=0.15,
                 test_ratio=0.15):

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

        self.all_results = []

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

    def load_species_data(self, species):
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

    def prepare_datasets(self, data_dict, species, seed):
        print(f"\n  准备训练/验证/测试数据集 (seed={seed})...")

        gene_ids = data_dict['gene_ids']
        total_genes = len(gene_ids)

        indices = list(range(total_genes))

        train_indices, temp_indices = train_test_split(
            indices,
            train_size=self.train_ratio,
            random_state=seed,
            shuffle=True
        )

        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_ratio_adjusted,
            random_state=seed,
            shuffle=True
        )

        train_genes = [gene_ids[i] for i in train_indices]
        val_genes = [gene_ids[i] for i in val_indices]
        test_genes = [gene_ids[i] for i in test_indices]

        print(f"    训练集: {len(train_genes)} 个基因 ({len(train_genes)/total_genes:.1%})")
        print(f"    验证集: {len(val_genes)} 个基因 ({len(val_genes)/total_genes:.1%})")
        print(f"    测试集: {len(test_genes)} 个基因 ({len(test_genes)/total_genes:.1%})")

        if len(train_genes) == 0 or len(val_genes) == 0 or len(test_genes) == 0:
            return None, None, None

        train_dataset = HGNTDataset(
            data_dict['embeddings'],
            data_dict['ppi_edge_index'], data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
            data_dict['expression_values'],
            data_dict['gene_ids'],
            train_genes, seed=seed
        )

        val_dataset = HGNTDataset(
            data_dict['embeddings'],
            data_dict['ppi_edge_index'], data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
            data_dict['expression_values'],
            data_dict['gene_ids'],
            val_genes, seed=seed + 1
        )

        test_dataset = HGNTDataset(
            data_dict['embeddings'],
            data_dict['ppi_edge_index'], data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
            data_dict['expression_values'],
            data_dict['gene_ids'],
            test_genes, seed=seed + 2
        )

        return train_dataset, val_dataset, test_dataset

    def train_single_seed(self, species, seed, data_dict):
        print(f"\n  🔬 训练 {species.upper()} - seed={seed}")
        print(f"  {'-' * 40}")

        train_dataset, val_dataset, test_dataset = self.prepare_datasets(data_dict, species, seed)

        if train_dataset is None:
            return None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            collate_fn=hgnt_collate_fn,
            generator=torch.Generator().manual_seed(seed)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=hgnt_collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=hgnt_collate_fn
        )

        input_dim = data_dict['embeddings'].size(1)
        model = ModelHGNT(input_dim=input_dim)
        print(f"  ✅ 成功构建HG-NT模型, 输入维度: {input_dim}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  📊 模型参数: {total_params:,}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  🔧 使用设备: {device}")

        trainer = HGNTTrainer(
            model, device=device,
            learning_rate=self.learning_rate,
            patience=self.patience,
            seed=seed
        )

        trainer.set_graph_data(
            all_embeddings=data_dict['embeddings'],
            ppi_edge_index=data_dict['ppi_edge_index'],
            tf_edge_index=data_dict['tf_edge_index'],
            gcn_edge_index=data_dict['gcn_edge_index']
        )

        print(f"  🚀 开始训练...")

        training_history = trainer.train(
            train_loader, val_loader,
            epochs=self.epochs
        )

        # 测试集评估
        test_loss, test_preds, test_targets = trainer.validate(test_loader)
        test_evaluation = evaluate_regression(test_targets, test_preds)

        # 保存模型
        model_file = os.path.join(self.output_dir, 'models', f'{species}_seed{seed}_best.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'test_metrics': test_evaluation,
            'best_epoch': training_history['best_epoch'],
            'seed': seed
        }, model_file)

        result = {
            'species': species,
            'seed': seed,
            'test_r2': test_evaluation['r2'],
            'test_pearson': test_evaluation['pearson_corr'],
            'test_spearman': test_evaluation['spearman_corr'],
            'test_rmse': test_evaluation['rmse'],
            'test_mae': test_evaluation['mae'],
            'best_epoch': training_history['best_epoch'],
            'best_val_loss': training_history['best_val_loss'],
            'model_params': total_params,
            'num_test_genes': len(test_dataset)
        }

        print(f"\n  📈 Seed {seed} 测试结果:")
        print(f"     R²: {test_evaluation['r2']:.6f}")
        print(f"     Pearson: {test_evaluation['pearson_corr']:.6f}")
        print(f"     Spearman: {test_evaluation['spearman_corr']:.6f}")
        print(f"     RMSE: {test_evaluation['rmse']:.6f}")

        return result

    def run_experiment(self, species, seeds=SEEDS):
        print(f"\n{'=' * 60}")
        print(f"🌿 HG-NT 实验: {species.upper()}")
        print(f"   使用 {len(seeds)} 个随机种子: {seeds}")
        print(f"{'=' * 60}")

        data_dict = self.load_species_data(species)
        if data_dict is None:
            print(f"⚠️ 跳过 {species}: 数据加载失败")
            return None

        seed_results = []
        for seed in seeds:
            result = self.train_single_seed(species, seed, data_dict)
            if result is not None:
                seed_results.append(result)

        if not seed_results:
            return None

        # 计算统计量
        metrics = ['test_r2', 'test_pearson', 'test_spearman', 'test_rmse']
        summary = {
            'species': species,
            'num_seeds': len(seed_results),
            'seeds': seeds
        }

        for metric in metrics:
            values = [r[metric] for r in seed_results]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values, ddof=1)
            summary[f'{metric}_values'] = values

        # 获取最佳种子结果
        best_idx = np.argmax([r['test_pearson'] for r in seed_results])
        summary['best_seed'] = seed_results[best_idx]['seed']
        summary['best_pearson'] = seed_results[best_idx]['test_pearson']

        # 保存详细结果（各种子的指标）
        results_df = pd.DataFrame(seed_results)
        results_file = os.path.join(self.output_dir, f'{species}_seed_results.csv')
        results_df.to_csv(results_file, index=False)

        # 保存统计摘要
        summary_file = os.path.join(self.output_dir, f'{species}_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(self.convert_for_json(summary), f, indent=2)

        self.all_results.append(summary)

        return summary

    def print_final_report(self):
        """打印最终报告，包含均值和标准差"""
        if not self.all_results:
            return

        print(f"\n{'=' * 80}")
        print(f"📊 HG-NT 最终实验报告")
        print(f"{'=' * 80}")

        report_data = []
        for summary in self.all_results:
            species = summary['species']
            print(f"\n{'=' * 60}")
            print(f"物种: {species.upper()}")
            print(f"{'=' * 60}")
            print(f"  基于 {summary['num_seeds']} 个随机种子")
            print(f"\n  📈 性能指标 (均值 ± 标准差):")
            print(f"    R²:       {summary['test_r2_mean']:.6f} ± {summary['test_r2_std']:.6f}")
            print(f"    Pearson:  {summary['test_pearson_mean']:.6f} ± {summary['test_pearson_std']:.6f}")
            print(f"    Spearman: {summary['test_spearman_mean']:.6f} ± {summary['test_spearman_std']:.6f}")
            print(f"    RMSE:     {summary['test_rmse_mean']:.6f} ± {summary['test_rmse_std']:.6f}")
            print(f"\n  🏆 最佳种子: seed={summary['best_seed']}, Pearson={summary['best_pearson']:.6f}")

            report_data.append({
                'species': species,
                'r2_mean': summary['test_r2_mean'],
                'r2_std': summary['test_r2_std'],
                'pearson_mean': summary['test_pearson_mean'],
                'pearson_std': summary['test_pearson_std'],
                'spearman_mean': summary['test_spearman_mean'],
                'spearman_std': summary['test_spearman_std'],
                'rmse_mean': summary['test_rmse_mean'],
                'rmse_std': summary['test_rmse_std'],
                'best_seed': summary['best_seed'],
                'best_pearson': summary['best_pearson'],
                'num_seeds': summary['num_seeds']
            })

        # 保存汇总报告
        report_df = pd.DataFrame(report_data)
        report_file = os.path.join(self.output_dir, 'final_summary_report.csv')
        report_df.to_csv(report_file, index=False)
        print(f"\n📄 汇总报告已保存: {report_file}")


# =================================================================
# 主函数
# =================================================================

def main():
    parser = argparse.ArgumentParser(description='HG-NT: Multi-Graph Concat with GCN - Human/Mouse')

    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--output_dir', type=str, default='Results_HGNT', help='输出目录')
    parser.add_argument('--species', type=str, default='all', choices=['human', 'mouse', 'all'],
                        help='要训练的物种')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help=f'随机种子列表 (默认: {SEEDS})')

    args = parser.parse_args()

    print("=" * 80)
    print("🔬 HG-NT (Human/Mouse Graph Network with Transformer)")
    print("   基于 Multi-Graph Concat 架构 - GCN均等聚合")
    print("=" * 80)
    print("\n🔧 训练配置:")
    print(f"  物种: {args.species}")
    print(f"  随机种子: {args.seeds}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  早停耐心: {args.patience}")
    print(f"\n📊 数据划分: {TRAIN_RATIO} / {VAL_RATIO} / {TEST_RATIO}")
    print("=" * 80)

    # 检查数据目录
    if not os.path.exists(EMBEDDING_DIR):
        print(f"❌ embeddings目录不存在: {EMBEDDING_DIR}")
        return

    if not os.path.exists(LABELS_DIR):
        print(f"❌ 标签目录不存在: {LABELS_DIR}")
        return

    # 创建实验对象
    experiment = HGNTExperiment(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience
    )

    # 运行实验
    if args.species == 'all':
        for species in ['human', 'mouse']:
            experiment.run_experiment(species, seeds=args.seeds)
    else:
        experiment.run_experiment(args.species, seeds=args.seeds)

    # 打印最终报告
    experiment.print_final_report()

    print("\n✅ 实验完成!")
    print(f"   结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()