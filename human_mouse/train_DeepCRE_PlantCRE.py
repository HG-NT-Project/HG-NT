"""
人类和小鼠基因表达预测模型训练 - 整合CNN和Basenji2模型
支持两种模型架构对比，使用标准的训练/验证/测试集划分
支持多种子重复实验，输出性能指标的均值和标准差
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import warnings
import json
from datetime import datetime
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')

# 默认5个随机种子
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


# =================================================================
# 固定随机种子函数
# =================================================================
def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✅ 随机种子已设置为: {seed}")


# =================================================================
# 物种配置
# =================================================================
SPECIES_CONFIG = {
    'human': {
        'name': 'human',
        'full_name': 'homo_sapiens',
        'sequences_file': 'precomputed_sequences/human_sequences.pt',
        'labels_file': 'processed_labels/human_labels.pt',
        'max_nuclear_chrom': 22,
    },
    'mouse': {
        'name': 'mouse',
        'full_name': 'mus_musculus',
        'sequences_file': 'precomputed_sequences/mouse_sequences.pt',
        'labels_file': 'processed_labels/mouse_labels.pt',
        'max_nuclear_chrom': 19,
    }
}

SEQUENCE_LENGTH = 3000


# =================================================================
# GELU激活函数
# =================================================================
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
        self.alpha = 1.702

    def forward(self, x):
        return torch.sigmoid(self.alpha * x) * x


# =================================================================
# CNN模型
# =================================================================
class Conv1DRegression(nn.Module):
    """用于回归的1D CNN模型"""

    def __init__(self, input_channels=4, seq_length=3000):
        super(Conv1DRegression, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=8, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=8, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=8, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=8, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=8, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
            nn.Dropout(0.25)
        )

        self._to_linear = None
        self._calculate_flatten_size(input_channels, seq_length)

        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def _calculate_flatten_size(self, channels, length):
        with torch.no_grad():
            x = torch.randn(1, channels, length)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =================================================================
# Basenji2模型
# =================================================================
def exponential_linspace_int(initial_value, target_value, num_layers):
    factor = (target_value / initial_value) ** (1 / num_layers)
    values = []
    value = initial_value
    for _ in range(num_layers + 1):
        values.append(np.round(value))
        value *= factor
    return values[1:]


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            bias=False
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels_num, dilation=1, dropout_rate=0.05):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv1DBlock(
            channels_num, int(0.5 * channels_num),
            kernel_size=3, dilation=dilation
        )
        self.conv2 = Conv1DBlock(
            int(0.5 * channels_num), channels_num,
            kernel_size=1, dilation=1
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x + identity


class PlantCREBasenji2(nn.Module):
    def __init__(self, input_channels=4, seq_length=3000,
                 channels_num=720, L=3, W=15):
        super(PlantCREBasenji2, self).__init__()

        self.channels_num = channels_num
        self.seq_length = seq_length

        self.input_conv = nn.Sequential(
            nn.Conv1d(
                input_channels, int(0.375 * channels_num),
                kernel_size=W, padding=W // 2, dilation=1
            ),
            nn.BatchNorm1d(int(0.375 * channels_num)),
            GELU()
        )

        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        Ci_steps = exponential_linspace_int(0.5 * channels_num, channels_num, 6)

        for i, Ci in enumerate(Ci_steps):
            if i == 0:
                in_channels = int(0.375 * channels_num)
            else:
                in_channels = int(Ci_steps[i - 1])

            self.conv_blocks.append(
                Conv1DBlock(
                    in_channels=in_channels,
                    out_channels=int(Ci),
                    kernel_size=5,
                    dilation=1
                )
            )
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.residual_blocks = nn.ModuleList()
        Di = [1, 2, 3, 4]

        for i in range(len(Di)):
            self.residual_blocks.append(
                ResidualBlock(channels_num, dilation=Di[i], dropout_rate=0.05)
            )

        self.output_conv = nn.Sequential(
            Conv1DBlock(channels_num, channels_num // 2, kernel_size=1),
            nn.Dropout(0.05),
            GELU(),
            nn.Conv1d(channels_num // 2, 1, kernel_size=1)
        )

        self._to_linear = None
        self._calculate_flatten_size(input_channels, seq_length)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 1)
        )

    def _calculate_flatten_size(self, channels, length):
        with torch.no_grad():
            x = torch.randn(1, channels, length)
            x = self.input_conv(x)
            x = self.pool1(x)
            for conv_block, pool in zip(self.conv_blocks, self.pools):
                x = conv_block(x)
                x = pool(x)
            for residual_block in self.residual_blocks:
                x = residual_block(x)
            x = self.output_conv(x)
            self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.pool1(x)
        for conv_block, pool in zip(self.conv_blocks, self.pools):
            x = conv_block(x)
            x = pool(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        x = self.output_conv(x)
        x = self.fc(x)
        return x


# =================================================================
# 数据增强
# =================================================================
class SequenceAugmentation:
    def __init__(self, augment_prob=0.3):
        self.augment_prob = augment_prob

    def reverse_complement(self, sequence):
        reversed_seq = torch.flip(sequence, dims=[-1])
        complemented = torch.zeros_like(reversed_seq)
        complemented[0] = reversed_seq[3]
        complemented[1] = reversed_seq[2]
        complemented[2] = reversed_seq[1]
        complemented[3] = reversed_seq[0]
        return complemented

    def __call__(self, sequence):
        augmented = sequence.clone()
        if torch.rand(1).item() < 0.5:
            augmented = self.reverse_complement(augmented)
        return augmented


# =================================================================
# 数据集类
# =================================================================
class GeneSequenceDataset(Dataset):
    def __init__(self, sequences_tensor, labels_tensor, gene_ids,
                 transform=None, augment=False):
        self.sequences = sequences_tensor
        self.labels = labels_tensor
        self.gene_ids = gene_ids
        self.transform = transform
        self.augment = augment

        if len(self.labels.shape) == 1:
            self.labels = self.labels.unsqueeze(1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        gene_id = self.gene_ids[idx]

        if self.augment and self.transform:
            sequence = self.transform(sequence)

        return {
            'sequence': sequence,
            'label': label,
            'gene_id': gene_id
        }


# =================================================================
# 训练器类
# =================================================================
class ModelTrainer:
    def __init__(self, model, model_name, device='cpu',
                 learning_rate=1e-4, weight_decay=1e-4,
                 patience=15, min_lr=1e-6):

        self.model = model.to(device)
        self.model_name = model_name
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

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

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)
        for batch in pbar:
            sequences = batch['sequence'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches if num_batches > 0 else float('inf')

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_gene_ids = []
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in pbar:
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)
                gene_ids = batch['gene_id']

                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_gene_ids.extend(gene_ids)

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss, np.array(all_preds), np.array(all_labels), all_gene_ids

    def train(self, train_loader, val_loader, epochs=100):
        train_losses = []
        val_losses = []
        learning_rates = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            train_losses.append(train_loss)

            val_loss, val_preds, val_labels, val_gene_ids = self.validate(val_loader)
            val_losses.append(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            old_lr = current_lr
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < old_lr and epoch >= 5:
                print(f"  📉 学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.counter = 0
                self.best_model_state = self.model.state_dict().copy()
                print(f"  ✅ Epoch {epoch + 1}: 新的最佳验证损失 {val_loss:.6f}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"  🚨 早停触发于第 {epoch + 1} 个epoch")
                    print(f"     最佳验证损失: {self.best_loss:.6f} (第 {self.best_epoch + 1} 个epoch)")
                    break

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  📊 Epoch {epoch + 1}/{epochs}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.6f}, "
                      f"Patience: {self.counter}/{self.patience}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"  ✅ 恢复第 {self.best_epoch + 1} 个epoch的最佳模型")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_loss
        }


# =================================================================
# 评估函数
# =================================================================
def evaluate_regression(y_true, y_pred, gene_ids=None):
    results = {}

    mse = mean_squared_error(y_true, y_pred)
    results['mse'] = float(mse)
    results['rmse'] = float(np.sqrt(mse))
    results['mae'] = float(mean_absolute_error(y_true, y_pred))
    results['r2'] = float(r2_score(y_true, y_pred))

    if len(y_true) > 1:
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        results['pearson_corr'] = float(pearson_corr)
        results['pearson_p'] = float(pearson_p)
        results['pearson_significant'] = bool(pearson_p < 0.05)

        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        results['spearman_corr'] = float(spearman_corr)
        results['spearman_p'] = float(spearman_p)
        results['spearman_significant'] = bool(spearman_p < 0.05)

        residuals = y_pred - y_true
        results['residual_mean'] = float(np.mean(residuals))
        results['residual_std'] = float(np.std(residuals))

        residual_var = float(np.var(residuals))
        y_var = float(np.var(y_true))
        results['explained_variance'] = float(1 - residual_var / y_var) if y_var > 0 else 0.0
    else:
        results['pearson_corr'] = 0.0
        results['pearson_p'] = 1.0
        results['pearson_significant'] = False
        results['spearman_corr'] = 0.0
        results['spearman_p'] = 1.0
        results['spearman_significant'] = False
        results['residual_mean'] = 0.0
        results['residual_std'] = 0.0
        results['explained_variance'] = 0.0

    results['num_samples'] = int(len(y_true))

    return results


# =================================================================
# 数据加载和划分
# =================================================================
def load_data(species, sequences_dir='precomputed_sequences', labels_dir='processed_labels'):
    print(f"\n🔍 加载 {species} 数据...")

    config = SPECIES_CONFIG[species]

    sequences_file = os.path.join(sequences_dir, config['sequences_file'].split('/')[-1])
    if not os.path.exists(sequences_file):
        sequences_file = config['sequences_file']

    if not os.path.exists(sequences_file):
        print(f"❌ 序列文件不存在: {sequences_file}")
        return None, None, None

    try:
        seq_data = torch.load(sequences_file, map_location='cpu', weights_only=False)
        sequences = seq_data['sequences']
        gene_ids_seq = seq_data['target_genes']
        print(f"✅ 成功加载序列数据: {sequences.shape}")
    except Exception as e:
        print(f"❌ 加载序列数据失败: {e}")
        return None, None, None

    labels_file = os.path.join(labels_dir, config['labels_file'].split('/')[-1])
    if not os.path.exists(labels_file):
        labels_file = config['labels_file']

    if not os.path.exists(labels_file):
        print(f"❌ 标签文件不存在: {labels_file}")
        return None, None, None

    try:
        label_data = torch.load(labels_file, map_location='cpu', weights_only=False)
        labels = label_data['labels']
        gene_ids_label = label_data['gene_id']
        print(f"✅ 成功加载标签数据: {labels.shape}")
    except Exception as e:
        print(f"❌ 加载标签数据失败: {e}")
        return None, None, None

    gene_id_to_seq_idx = {gid: idx for idx, gid in enumerate(gene_ids_seq)}
    gene_id_to_label_idx = {gid: idx for idx, gid in enumerate(gene_ids_label)}

    common_genes = set(gene_ids_seq).intersection(set(gene_ids_label))
    print(f"📊 共同基因数量: {len(common_genes)}")

    if len(common_genes) == 0:
        print(f"❌ 序列和标签没有共同的基因ID")
        return None, None, None

    aligned_sequences = []
    aligned_labels = []
    aligned_gene_ids = []

    for gene_id in common_genes:
        seq_idx = gene_id_to_seq_idx[gene_id]
        label_idx = gene_id_to_label_idx[gene_id]
        aligned_sequences.append(sequences[seq_idx])
        aligned_labels.append(labels[label_idx])
        aligned_gene_ids.append(gene_id)

    aligned_sequences = torch.stack(aligned_sequences)
    aligned_labels = torch.stack(aligned_labels)

    print(f"✅ 数据对齐完成: {aligned_sequences.shape}, {aligned_labels.shape}")

    return aligned_sequences, aligned_labels, aligned_gene_ids


def split_data(sequences, labels, gene_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    total = len(sequences)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(total, generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_data = {
        'sequences': sequences[train_indices],
        'labels': labels[train_indices],
        'gene_ids': [gene_ids[i] for i in train_indices]
    }

    val_data = {
        'sequences': sequences[val_indices],
        'labels': labels[val_indices],
        'gene_ids': [gene_ids[i] for i in val_indices]
    }

    test_data = {
        'sequences': sequences[test_indices],
        'labels': labels[test_indices],
        'gene_ids': [gene_ids[i] for i in test_indices]
    }

    return train_data, val_data, test_data


# =================================================================
# 单次训练运行（固定种子）
# =================================================================
def run_single_seed(species, model_type, train_data, val_data, test_data, args, device, seed):
    """使用固定种子进行单次训练"""
    print(f"\n{'=' * 50}")
    print(f"🎲 种子: {seed}")
    print(f"{'=' * 50}")

    set_seed(seed)

    # 创建数据集
    transform = SequenceAugmentation(augment_prob=0.3) if args.augment else None

    train_dataset = GeneSequenceDataset(
        train_data['sequences'], train_data['labels'], train_data['gene_ids'],
        transform=transform, augment=args.augment
    )

    val_dataset = GeneSequenceDataset(
        val_data['sequences'], val_data['labels'], val_data['gene_ids'],
        augment=False
    )

    test_dataset = GeneSequenceDataset(
        test_data['sequences'], test_data['labels'], test_data['gene_ids'],
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # 创建模型
    if model_type == 'CNN':
        model = Conv1DRegression(input_channels=4, seq_length=SEQUENCE_LENGTH)
    else:  # Basenji2
        model = PlantCREBasenji2(
            input_channels=4, seq_length=SEQUENCE_LENGTH,
            channels_num=args.channels_num
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    trainer = ModelTrainer(
        model, model_type, device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience
    )

    print(f"\n🚀 开始训练...")
    training_history = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 测试集评估
    test_loss, test_preds, test_labels, test_gene_ids = trainer.validate(test_loader)
    evaluation_results = evaluate_regression(test_labels, test_preds, test_gene_ids)

    # 返回指标
    result = {
        'seed': seed,
        'best_epoch': int(training_history['best_epoch'] + 1),
        'best_val_loss': float(training_history['best_val_loss']),
        'test_loss': float(test_loss),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'test_r2': evaluation_results['r2'],
        'test_pearson': evaluation_results['pearson_corr'],
        'test_spearman': evaluation_results['spearman_corr'],
        'test_rmse': evaluation_results['rmse'],
        'test_mae': evaluation_results['mae'],
        'model_params': total_params
    }

    print(f"\n  📈 测试结果:")
    print(f"     R²: {result['test_r2']:.6f}")
    print(f"     Pearson: {result['test_pearson']:.6f}")
    print(f"     Spearman: {result['test_spearman']:.6f}")
    print(f"     RMSE: {result['test_rmse']:.6f}")

    return result


# =================================================================
# 多种子训练（只保存汇总指标）
# =================================================================
def train_model_multi_seed(species, model_type, base_train_data, base_val_data, base_test_data, args, device):
    """使用多个种子进行训练，只保存汇总指标"""
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    print(f"\n{'=' * 70}")
    print(f"🚀 多种子训练 - {species.upper()} - {model_type}")
    print(f"种子列表: {seeds}")
    print(f"{'=' * 70}")

    all_results = []
    for seed in seeds:
        # 重新划分数据（使用当前种子）
        train_data, val_data, test_data = split_data(
            base_train_data['sequences'], base_train_data['labels'], base_train_data['gene_ids'],
            args.train_ratio, args.val_ratio, args.test_ratio, seed=seed
        )

        result = run_single_seed(
            species, model_type, train_data, val_data, test_data, args, device, seed
        )
        if result:
            all_results.append(result)

    if not all_results:
        return None

    # 计算统计量
    metrics_keys = ['test_r2', 'test_pearson', 'test_spearman', 'test_rmse']

    print(f"\n{'=' * 70}")
    print(f"📊 {species.upper()} - {model_type} 多种子汇总结果")
    print(f"{'=' * 70}")

    print(f"\n{'指标':<12} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
    print(f"{'-' * 60}")

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
        print(f"{key:<12} {np.mean(values):<12.6f} {np.std(values, ddof=1):<12.6f} "
              f"{np.min(values):<12.6f} {np.max(values):<12.6f}")

    # 保存各种子指标到CSV
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(args.output_dir, f'{species}_{model_type}_seed_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 各种子指标已保存: {results_file}")

    # 保存汇总统计
    summary = {
        'species': species,
        'model_type': model_type,
        'seeds': seeds,
        'num_seeds': len(seeds),
        'metrics_summary': metrics_summary
    }

    with open(os.path.join(args.output_dir, f'{species}_{model_type}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 汇总统计已保存: {args.output_dir}/{species}_{model_type}_summary.json")

    # 打印均值±标准差格式
    print(f"\n📈 {species.upper()} - {model_type} 性能指标 (均值 ± 标准差):")
    print(f"   R²:       {metrics_summary['test_r2']['mean']:.6f} ± {metrics_summary['test_r2']['std']:.6f}")
    print(f"   Pearson:  {metrics_summary['test_pearson']['mean']:.6f} ± {metrics_summary['test_pearson']['std']:.6f}")
    print(
        f"   Spearman: {metrics_summary['test_spearman']['mean']:.6f} ± {metrics_summary['test_spearman']['std']:.6f}")
    print(f"   RMSE:     {metrics_summary['test_rmse']['mean']:.6f} ± {metrics_summary['test_rmse']['std']:.6f}")

    return results_df


# =================================================================
# 主训练类
# =================================================================
class HumanMouseModelTrainer:
    def __init__(self, output_dir='training_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

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
        elif isinstance(obj, bool):
            return bool(obj)
        else:
            return obj

    def train_species(self, species, args, device):
        print(f"\n{'=' * 70}")
        print(f"🌿 处理物种: {species.upper()}")
        print(f"{'=' * 70}")

        # 加载数据（只需加载一次）
        sequences, labels, gene_ids = load_data(
            species,
            sequences_dir=args.sequences_dir,
            labels_dir=args.labels_dir
        )

        if sequences is None:
            print(f"❌ 数据加载失败，跳过 {species}")
            return

        # 创建基础数据集（用于后续划分）
        base_train_data = {
            'sequences': sequences,
            'labels': labels,
            'gene_ids': gene_ids
        }

        # 训练CNN模型
        if 'cnn' in args.models or 'all' in args.models:
            train_model_multi_seed(
                species, 'CNN', base_train_data, None, None, args, device
            )

        # 训练Basenji2模型
        if 'basenji' in args.models or 'all' in args.models:
            train_model_multi_seed(
                species, 'Basenji2', base_train_data, None, None, args, device
            )

    def train_all(self, args):
        print(f"\n{'=' * 80}")
        print(f"🌱 开始训练人类和小鼠基因表达预测模型")
        print(f"随机种子列表: {args.seeds}")
        print(f"训练轮数: {args.epochs}")
        print(f"批次大小: {args.batch_size}")
        print(f"学习率: {args.learning_rate}")
        print(f"权重衰减: {args.weight_decay}")
        print(f"早停耐心: {args.patience}")
        print(f"数据增强: {args.augment}")
        print(f"训练模型: {args.models}")
        print(f"数据集划分: 训练 {args.train_ratio:.0%}, 验证 {args.val_ratio:.0%}, 测试 {args.test_ratio:.0%}")
        print(f"{'=' * 80}")

        # 保存实验配置
        config = vars(args)
        config['timestamp'] = datetime.now().isoformat()
        config['cuda_available'] = torch.cuda.is_available()

        config_file = os.path.join(self.output_dir, 'logs', 'experiment_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.convert_for_json(config), f, indent=2)
        print(f"⚙️ 实验配置已保存: {config_file}")

        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        print(f"\n🔧 使用设备: {device}")

        # 训练每个物种
        for species in args.species_list:
            self.train_species(species, args, device)

        print(f"\n{'=' * 80}")
        print("✅ 全部训练完成!")
        print(f"   结果保存在: {args.output_dir}")
        print(f"{'=' * 80}")


# =================================================================
# 主函数
# =================================================================
def main():
    parser = argparse.ArgumentParser(description='训练人类和小鼠基因表达预测模型 - 多种子实验')

    # 物种选择
    parser.add_argument('--species', type=str, default='all',
                        choices=['human', 'mouse', 'all'],
                        help='要训练的物种')

    # 模型选择
    parser.add_argument('--models', type=str, default='all',
                        choices=['cnn', 'basenji', 'all'],
                        help='要训练的模型')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')

    # 数据划分
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='测试集比例')

    # 随机种子
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'随机种子列表 (默认: {DEFAULT_SEEDS})')

    # Basenji2特定参数
    parser.add_argument('--channels_num', type=int, default=720,
                        help='Basenji2模型通道数')

    # 其他选项
    parser.add_argument('--no_augment', action='store_true',
                        help='禁用数据增强')
    parser.add_argument('--sequences_dir', type=str, default='precomputed_sequences',
                        help='预处理序列目录')
    parser.add_argument('--labels_dir', type=str, default='processed_labels',
                        help='标签目录')
    parser.add_argument('--output_dir', type=str, default='training_results',
                        help='输出目录')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')

    args = parser.parse_args()

    # 设置物种列表
    if args.species == 'all':
        args.species_list = ['human', 'mouse']
    else:
        args.species_list = [args.species]

    args.augment = not args.no_augment

    # 验证划分比例
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print(f"❌ 划分比例之和必须为1，当前和为 {args.train_ratio + args.val_ratio + args.test_ratio}")
        return

    # 打印配置
    print("=" * 80)
    print("🔬 CNN / Basenji2 基因表达预测 - 多种子实验")
    print("=" * 80)
    print(f"\n🔧 训练配置:")
    print(f"  物种: {args.species_list}")
    print(f"  模型: {args.models}")
    print(f"  随机种子: {args.seeds}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  早停耐心: {args.patience}")
    print(f"  数据增强: {args.augment}")
    print(f"  数据集划分: 训练 {args.train_ratio:.0%}, 验证 {args.val_ratio:.0%}, 测试 {args.test_ratio:.0%}")
    print(f"  Basenji2通道数: {args.channels_num}")
    print(f"  输出目录: {args.output_dir}")

    # 创建训练器并开始训练
    trainer = HumanMouseModelTrainer(output_dir=args.output_dir)
    trainer.train_all(args)


if __name__ == "__main__":
    main()