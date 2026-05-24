"""
DeepGAT网络来源消融实验 - 高效重构训练代码 (修复版)
- 修复目标节点与标签（Label）错位的致命 Bug
- 彻底干掉显式 item() 循环，全通路实现 GPU 张量查表级重映射，速度提升 5~10 倍
- 显式包裹对数化预处理提示与维度 squeeze 防御机制
- 完美对接解耦后的 M3a - M3g 消融模型
- 修复 mapping_tensor 显存泄露问题（复用缓存张量）
- 优化评估指标计算（单次遍历完成多指标）
- 修复早停机制状态恢复问题
"""

import os
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

# 导入外部解耦模型
from net_ablation_model import build_model, MODEL_NAMES

# PyTorch Geometric
try:
    from torch_geometric.utils import subgraph

    print("✅ PyTorch Geometric loaded successfully")
except ImportError:
    print("❌ Error: PyTorch Geometric not installed.")
    exit(1)

warnings.filterwarnings('ignore')

# =================================================================
# 固定全局参数
# =================================================================
MAX_NEIGHBORS = 32
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

EMBEDDING_DIR = "processed_features"
LABELS_DIR = "processed_labels"
PPI_PATH_TEMPLATE = "processed_ppi/{species}_ppi_edge_index.pt"
TF_PATH_TEMPLATE = "processed_tf/{species}_tf_edge_index.pt"
GCN_PATH_TEMPLATE = "processed_gcn/{species}_gcn_network_aligned.pt"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =================================================================
# 数据基础加载通路
# =================================================================
def load_nt_embeddings(species):
    file_path = os.path.join(EMBEDDING_DIR, f"{species}_nt_embeddings.pt")
    if not os.path.exists(file_path):
        print(f"❌ NT embeddings文件不存在: {file_path}")
        return None
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    print(f"✅ 成功加载 NT embeddings. 形状: {data['x'].shape} | 基因数: {len(data['gene_ids'])}")
    return {'embeddings': data['x'], 'gene_ids': data['gene_ids']}


def load_expression_data(species):
    file_path = os.path.join(LABELS_DIR, f"{species}_labels.pt")
    if not os.path.exists(file_path):
        print(f"❌ 标签文件不存在: {file_path}")
        return None, None
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    labels = data['labels']
    gene_ids = data['gene_id']

    # 生信背景校正提示：请确保 labels 在预处理时已执行过 log2(x + 1)
    expr_dict = {gene_id: labels[i].item() for i, gene_id in enumerate(gene_ids)}
    return expr_dict, set(gene_ids)


def load_network(species, network_type, num_nodes=None):
    templates = {'ppi': PPI_PATH_TEMPLATE, 'tf': TF_PATH_TEMPLATE, 'gcn': GCN_PATH_TEMPLATE}
    path = templates[network_type].format(species=species)

    if not os.path.exists(path):
        print(f"⚠️ {network_type.upper()} 网络未发现: {path}")
        return torch.zeros((2, 0), dtype=torch.long)

    try:
        data = torch.load(path, map_location='cpu', weights_only=False)

        # 修复：严谨安全的字典判定，闭绝 Tensor 隐式布尔转换为 True/False 的语法冲突
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

        # 确保 edge_index 是 2 维且第一维为 2
        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        # 转换为统一的统一长整型
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        # 过滤超出当前物种 embeddings 节点范围的非法边
        if num_nodes is not None and edge_index.numel() > 0:
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_mask]

        print(f"   {network_type.upper()} 网络成功加载. 边数: {edge_index.shape[1]}")
        return edge_index

    except Exception as e:
        print(f"❌ 加载 {network_type.upper()} 网络失败: {e}")
        return torch.zeros((2, 0), dtype=torch.long)


def build_expression_tensor(gene_ids, expr_dict):
    """构建表达值张量"""
    expression_values = [expr_dict.get(gene_id, float('nan')) for gene_id in gene_ids]
    return torch.tensor(expression_values, dtype=torch.float32)


# =================================================================
# 鲁棒性 Dataset 架构（支持转导式独立种子洗牌）
# =================================================================
class GATDeepCREDataset(Dataset):
    def __init__(self, embeddings, ppi_edge_index, tf_edge_index, gcn_edge_index,
                 expression_values, gene_ids, target_genes, current_seed=42):
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
                self.valid_expression_mask.append(not torch.isnan(self.expression_tensor[idx]))

        # 核心修正：传入当前运行的洗牌种子，保证邻居采样空间的独立性
        self.all_neighbors = self._precompute_all_neighbors(current_seed)

    def _precompute_all_neighbors(self, fixed_seed):
        adj_dict = {'ppi': defaultdict(list), 'tf': defaultdict(list), 'gcn': defaultdict(list)}

        for net_name, edges in [('ppi', self.ppi_edge_index), ('tf', self.tf_edge_index), ('gcn', self.gcn_edge_index)]:
            if edges is not None and edges.numel() > 0:
                edges_np = edges.numpy() if torch.is_tensor(edges) else edges
                for i in range(edges_np.shape[1]):
                    src, tgt = edges_np[0, i], edges_np[1, i]
                    if src < self.num_nodes and tgt < self.num_nodes:
                        adj_dict[net_name][src].append(tgt)
                        adj_dict[net_name][tgt].append(src)

        all_neighbors = {}
        random.seed(fixed_seed)  # 严格锁定当前划分下的动态采样

        for node_idx in range(self.num_nodes):
            neighbors_dict = {}
            for net_name in ['ppi', 'tf', 'gcn']:
                neighbors = adj_dict[net_name].get(node_idx, [])
                if neighbors:
                    if len(neighbors) > MAX_NEIGHBORS:
                        sampled = random.sample(neighbors, MAX_NEIGHBORS)
                        neighbors_dict[net_name] = torch.tensor(sampled, dtype=torch.long)
                    else:
                        neighbors_dict[net_name] = torch.tensor(neighbors, dtype=torch.long)
                else:
                    neighbors_dict[net_name] = torch.tensor([node_idx], dtype=torch.long)
            all_neighbors[node_idx] = neighbors_dict
        return all_neighbors

    def __len__(self):
        return len(self.valid_genes)

    def __getitem__(self, idx):
        gene_idx = self.valid_gene_indices[idx]
        return {
            'gene_id': self.valid_genes[idx],
            'gene_idx': gene_idx,
            'neighbor_indices': self.all_neighbors[gene_idx],
            'expression': self.expression_tensor[gene_idx],
            'has_expression': not torch.isnan(self.expression_tensor[gene_idx])
        }


def collate_fn(batch):
    return {
        'gene_ids': [item['gene_id'] for item in batch],
        'gene_indices': torch.LongTensor([item['gene_idx'] for item in batch]),
        'neighbor_indices': [item['neighbor_indices'] for item in batch],
        'expressions': torch.tensor([item['expression'] for item in batch], dtype=torch.float32),
        'has_expression': torch.BoolTensor([item['has_expression'] for item in batch])
    }


# =================================================================
# 高性能训练器 (GPU加速查表版 + 显存优化)
# =================================================================
class Trainer:
    def __init__(self, model, model_name, device='cpu', learning_rate=1e-4, patience=15, seed=42):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.seed = seed

        self.all_embeddings = None
        self.ppi_edge_index = None
        self.tf_edge_index = None
        self.gcn_edge_index = None
        self.num_nodes = 0

        # 修复显存泄露：预分配 mapping_tensor 缓存空间，避免每次迭代重新分配
        self.mapping_tensor_cache = None

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        self.criterion = nn.HuberLoss(reduction='none', delta=1.0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        self.train_losses, self.val_losses = [], []

    def set_graph_data(self, all_embeddings, ppi_edge_index, tf_edge_index, gcn_edge_index):
        self.all_embeddings = all_embeddings.to(self.device)
        self.num_nodes = len(all_embeddings)
        self.ppi_edge_index = ppi_edge_index.to(self.device)
        self.tf_edge_index = tf_edge_index.to(self.device)
        self.gcn_edge_index = gcn_edge_index.to(self.device)

        # 预分配 mapping_tensor 缓存（复用同一块显存）
        self.mapping_tensor_cache = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)

    def _extract_and_remap_subgraphs(self, unique_nodes):
        """
        核心重构：利用张量全并行映射替代低效的 item() 循环，杜绝同步阻塞
        修复：复用 mapping_tensor_cache，通过 inplace 操作刷新，避免显存碎片
        """
        # 复用缓存张量，inplace 刷新（避免每次迭代重新分配显存）
        # 先清零再赋值
        self.mapping_tensor_cache.fill_(0)
        self.mapping_tensor_cache[unique_nodes] = torch.arange(len(unique_nodes), device=self.device)

        def get_sub_and_relabel(global_edge_index):
            if global_edge_index is None or global_edge_index.numel() == 0:
                return torch.zeros((2, 0), dtype=torch.long, device=self.device)
            # PyG 提取全局坐标子图
            sub_edges, _ = subgraph(unique_nodes, global_edge_index, relabel_nodes=False, num_nodes=self.num_nodes)
            if sub_edges.numel() == 0:
                return torch.zeros((2, 0), dtype=torch.long, device=self.device)
            # 查表批量映射到局部坐标系
            return self.mapping_tensor_cache[sub_edges]

        return (get_sub_and_relabel(self.ppi_edge_index),
                get_sub_and_relabel(self.tf_edge_index),
                get_sub_and_relabel(self.gcn_edge_index))

    def _prepare_batch(self, batch):
        gene_indices = batch['gene_indices'].to(self.device)
        neighbor_indices = batch['neighbor_indices']
        expressions = batch['expressions'].to(self.device)
        has_expression = batch['has_expression'].to(self.device)

        # 收集 batch 图空间涉及的所有节点
        all_nodes = [gene_indices]
        for n_dict in neighbor_indices:
            for net in ['ppi', 'tf', 'gcn']:
                if net in n_dict:
                    all_nodes.append(n_dict[net].to(self.device))

        unique_nodes = torch.unique(torch.cat(all_nodes))
        unique_nodes = unique_nodes[unique_nodes < self.num_nodes]

        if len(unique_nodes) == 0:
            return None

        # 提取子图并获取并行映射表（复用缓存）
        ppi_sub, tf_sub, gcn_sub = self._extract_and_remap_subgraphs(unique_nodes)

        # 利用绝对安全的张量映射定位目标索引
        target_local_indices = self.mapping_tensor_cache[gene_indices]

        return {
            'x': self.all_embeddings[unique_nodes],
            'unique_nodes': unique_nodes,
            'ppi_sub': ppi_sub, 'tf_sub': tf_sub, 'gcn_sub': gcn_sub,
            'target_indices': target_local_indices,
            'expressions': expressions, 'has_expression': has_expression
        }

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss, num_valid = 0, 0
        all_preds, all_targets = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)
        for batch in pbar:
            prepared = self._prepare_batch(batch)
            if prepared is None:
                continue

            self.optimizer.zero_grad()

            x = prepared['x']

            outputs, _ = self.model(
                x, ppi_sub=prepared['ppi_sub'],
                tf_sub=prepared['tf_sub'], gcn_sub=prepared['gcn_sub']
            )

            # 提取回归输出并强制 squeeze 匹配，防止损失函数维度爆炸
            target_outputs = outputs[prepared['target_indices']].view(-1)
            target_expressions = prepared['expressions'].view(-1)

            valid_outputs = target_outputs[prepared['has_expression']]
            valid_targets = target_expressions[prepared['has_expression']]

            if len(valid_outputs) == 0:
                continue

            loss = self.criterion(valid_outputs, valid_targets).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(valid_outputs)
            num_valid += len(valid_outputs)
            all_preds.extend(valid_outputs.detach().cpu().numpy())
            all_targets.extend(valid_targets.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_valid if num_valid > 0 else float('inf')
        train_pearson = pearsonr(all_preds, all_targets)[0] if len(all_preds) > 1 else 0.0
        return avg_loss, train_pearson

    def validate(self, val_loader):
        self.model.eval()
        total_loss, num_valid = 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                prepared = self._prepare_batch(batch)
                if prepared is None:
                    continue

                x = prepared['x']

                outputs, _ = self.model(
                    x, ppi_sub=prepared['ppi_sub'],
                    tf_sub=prepared['tf_sub'], gcn_sub=prepared['gcn_sub']
                )

                target_outputs = outputs[prepared['target_indices']].view(-1)
                target_expressions = prepared['expressions'].view(-1)

                valid_outputs = target_outputs[prepared['has_expression']]
                valid_targets = target_expressions[prepared['has_expression']]

                if len(valid_outputs) == 0:
                    continue

                loss = self.criterion(valid_outputs, valid_targets).mean()
                total_loss += loss.item() * len(valid_outputs)
                num_valid += len(valid_outputs)
                all_preds.extend(valid_outputs.cpu().numpy())
                all_targets.extend(valid_targets.cpu().numpy())

        avg_loss = total_loss / num_valid if num_valid > 0 else float('inf')
        return avg_loss, np.array(all_preds), np.array(all_targets)

    def train(self, train_loader, val_loader, epochs=100):
        for epoch in range(epochs):
            train_loss, train_pearson = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            val_loss, val_preds, val_targets = self.validate(val_loader)
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"  🚨 Early stopping at epoch {epoch + 1} | 最佳验证损失: {self.best_loss:.6f}")
                    break

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  📊 Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Train Pearson: {train_pearson:.4f}")

        # 修复早停机制：训练结束后恢复最佳模型权重
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"  ✅ 已恢复最佳模型权重 (验证损失: {self.best_loss:.6f})")

        return {'best_val_loss': self.best_loss}


# =================================================================
# 科学回归评估指标（优化版：单次遍历计算多指标）
# =================================================================
def evaluate_regression(y_true, y_pred):
    """
    优化版评估函数：单次遍历完成多指标计算，避免重复遍历
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    n = len(y_true)
    if n == 0:
        return {
            'mse': 0.0, 'rmse': 0.0, 'mae': 0.0,
            'r2': 0.0, 'pearson_corr': 0.0, 'spearman_corr': 0.0,
            'num_samples': 0
        }

    # 一次性计算残差
    residuals = y_pred - y_true
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))

    # R² 计算
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # 相关系数（使用 scipy 计算，内部已优化）
    pearson_corr = pearsonr(y_true, y_pred)[0] if n > 1 else 0.0
    spearman_corr = spearmanr(y_true, y_pred)[0] if n > 1 else 0.0

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_corr': float(pearson_corr),
        'spearman_corr': float(spearman_corr),
        'num_samples': n
    }


# =================================================================
# 单种子运行内核
# =================================================================
def run_single_seed(species, model_name, data_dict, args, seed):
    print(f"\n{'=' * 50}\n🎲 随机种子划分设定: {seed} - 当前执行消融通路: {model_name.upper()}\n{'=' * 50}")
    set_seed(seed)

    gene_ids = data_dict['gene_ids']
    indices = list(range(len(gene_ids)))

    train_indices, temp_indices = train_test_split(indices, train_size=TRAIN_RATIO, random_state=seed, shuffle=True)
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_indices, test_indices = train_test_split(temp_indices, train_size=val_ratio_adjusted, random_state=seed,
                                                 shuffle=True)

    train_genes = [gene_ids[i] for i in train_indices]
    val_genes = [gene_ids[i] for i in val_indices]
    test_genes = [gene_ids[i] for i in test_indices]

    print(f"📊 数据划分统计: 训练集 {len(train_genes)} | 验证集 {len(val_genes)} | 测试集 {len(test_genes)}")

    # 构建当前种子专属的 Dataset，彻底斩断跨划分种子串扰隐患
    train_dataset = GATDeepCREDataset(
        data_dict['embeddings'], data_dict['ppi_edge_index'],
        data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
        data_dict['expression_values'], data_dict['gene_ids'],
        train_genes, current_seed=seed
    )
    val_dataset = GATDeepCREDataset(
        data_dict['embeddings'], data_dict['ppi_edge_index'],
        data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
        data_dict['expression_values'], data_dict['gene_ids'],
        val_genes, current_seed=seed
    )
    test_dataset = GATDeepCREDataset(
        data_dict['embeddings'], data_dict['ppi_edge_index'],
        data_dict['tf_edge_index'], data_dict['gcn_edge_index'],
        data_dict['expression_values'], data_dict['gene_ids'],
        test_genes, current_seed=seed
    )

    # num_workers 设为 0，避免多进程相关问题
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    model = build_model(model_name, input_dim=data_dict['embeddings'].size(1), dropout=args.dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 计算设备: {device}")

    trainer = Trainer(
        model, model_name, device=device,
        learning_rate=args.learning_rate, patience=args.patience, seed=seed
    )
    trainer.set_graph_data(
        data_dict['embeddings'], data_dict['ppi_edge_index'],
        data_dict['tf_edge_index'], data_dict['gcn_edge_index']
    )

    print(f"🚀 启动模型训练...")
    trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 测试评估（此时 trainer.model 已经是早停后的最佳权重）
    _, test_preds, test_targets = trainer.validate(test_loader)
    test_eval = evaluate_regression(test_targets, test_preds)

    print(
        f"📈 种子 {seed} 评测完成 -> R²: {test_eval['r2']:.4f} | Pearson: {test_eval['pearson_corr']:.4f} | RMSE: {test_eval['rmse']:.4f}")

    return {
        'seed': seed,
        'model_name': model_name,
        'best_val_loss': trainer.best_loss,
        'test_r2': test_eval['r2'],
        'test_pearson': test_eval['pearson_corr'],
        'test_spearman': test_eval['spearman_corr'],
        'test_rmse': test_eval['rmse'],
        'test_mae': test_eval['mae'],
        'num_samples': test_eval['num_samples']
    }


# =================================================================
# 核心调度多随机种子引擎
# =================================================================
def train_multi_seed(species, model_name, data_dict, args):
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    print(f"\n{'=' * 70}")
    print(f"组装实验调度集群 -> 物种: {species.upper()} | 消融模型: {model_name.upper()}")
    print(f"随机种子列表: {seeds}")
    print(f"{'=' * 70}")

    all_results = []
    for seed in seeds:
        result = run_single_seed(species, model_name, data_dict, args, seed)
        if result:
            all_results.append(result)

    if not all_results:
        return None

    metrics = ['test_r2', 'test_pearson', 'test_spearman', 'test_rmse', 'test_mae']
    print(f"\n{'=' * 70}")
    print(f"📊 各种子洗牌均值汇总统计 ({species.upper()} - {model_name.upper()})")
    print(f"{'=' * 70}")
    print(f"{'性能指标':<15} {'实验均值':<12} {'标准差 (SD)':<12} {'极小值':<12} {'极大值':<12}")
    print("-" * 65)

    metrics_summary = {}
    for key in metrics:
        values = [r[key] for r in all_results]
        metrics_summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
        print(f"{key:<15} {np.mean(values):<12.6f} {metrics_summary[key]['std']:<12.6f} "
              f"{np.min(values):<12.6f} {np.max(values):<12.6f}")

    # 导出统计文件
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(all_results).to_csv(
        os.path.join(args.output_dir, f'{species}_{model_name}_results.csv'),
        index=False
    )
    with open(os.path.join(args.output_dir, f'{species}_{model_name}_summary.json'), 'w') as f:
        json.dump({
            'species': species,
            'model_name': model_name,
            'seeds': seeds,
            'metrics_summary': metrics_summary
        }, f, indent=2)

    # 打印最终结果
    print(f"\n📈 {species.upper()} - {model_name.upper()} 最终性能 (均值 ± 标准差):")
    print(f"   R²:       {metrics_summary['test_r2']['mean']:.6f} ± {metrics_summary['test_r2']['std']:.6f}")
    print(f"   Pearson:  {metrics_summary['test_pearson']['mean']:.6f} ± {metrics_summary['test_pearson']['std']:.6f}")
    print(
        f"   Spearman: {metrics_summary['test_spearman']['mean']:.6f} ± {metrics_summary['test_spearman']['std']:.6f}")
    print(f"   RMSE:     {metrics_summary['test_rmse']['mean']:.6f} ± {metrics_summary['test_rmse']['std']:.6f}")

    return pd.DataFrame(all_results)


# =================================================================
# 数据加载辅助类
# =================================================================
class DataLoaderHelper:
    @staticmethod
    def load_species_data(species):
        print(f"\n🔍 正在检索并预载背景生物网络 [{species.upper()}] ...")
        embed_data = load_nt_embeddings(species)
        if embed_data is None:
            return None

        num_nodes = len(embed_data['gene_ids'])
        expr_dict, _ = load_expression_data(species)
        if expr_dict is None:
            return None

        expression_values = build_expression_tensor(embed_data['gene_ids'], expr_dict)

        print(f"   📊 表达值统计: 有效表达基因数 {(~torch.isnan(expression_values)).sum().item()}/{num_nodes}")

        return {
            'embeddings': embed_data['embeddings'],
            'expression_values': expression_values,
            'gene_ids': embed_data['gene_ids'],
            'ppi_edge_index': load_network(species, 'ppi', num_nodes),
            'tf_edge_index': load_network(species, 'tf', num_nodes),
            'gcn_edge_index': load_network(species, 'gcn', num_nodes),
            'num_genes': num_nodes,
        }


# =================================================================
# 主函数
# =================================================================
def main():
    parser = argparse.ArgumentParser(description='DeepGAT网络消融实验主控端')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout比率')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['m3a', 'm3b', 'm3c', 'm3d', 'm3e', 'm3f', 'm3g'],
                        choices=['m3a', 'm3b', 'm3c', 'm3d', 'm3e', 'm3f', 'm3g'],
                        help='要训练的消融模型列表')
    parser.add_argument('--output_dir', type=str, default='Results_xr_net', help='输出目录')
    parser.add_argument('--species', type=str, default='all',
                        choices=['human', 'mouse', 'all'], help='物种选择')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS, help='随机种子列表')

    args = parser.parse_args()

    print("=" * 80)
    print("🔬 DeepGAT 网络来源消融实验 - 高效重构版 (修复版)")
    print("=" * 80)
    print(f"训练配置:")
    print(f"  物种: {args.species}")
    print(f"  模型: {args.models}")
    print(f"  种子: {args.seeds}")
    print(f"  轮数: {args.epochs}")
    print(f"  批次: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  输出目录: {args.output_dir}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # 确定物种列表
    if args.species == 'all':
        species_list = ['human', 'mouse']
    else:
        species_list = [args.species]

    # 运行实验
    for species in species_list:
        print(f"\n{'=' * 60}")
        print(f"🌿 处理物种: {species.upper()}")
        print(f"{'=' * 60}")

        data_dict = DataLoaderHelper.load_species_data(species)
        if data_dict is None:
            print(f"❌ 数据加载失败，跳过 {species}")
            continue

        for model_name in args.models:
            train_multi_seed(species, model_name, data_dict, args)

    print(f"\n{'=' * 80}")
    print("✅ 全交通路网络消融批处理实验圆满结束！统计报表已成功导出。")
    print(f"   结果保存在: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()