"""
DeepCBA模型训练 - 适配人类和小鼠物种
直接读取precomputed_sequences_ppi目录下的预处理文件
去掉留一染色体验证，使用标准的训练/验证/测试集划分
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
from sklearn.model_selection import train_test_split
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
# 物种配置 - 适配人类和小鼠
# =================================================================
SPECIES_CONFIG = {
    'human': {
        'name': 'human',
        'full_name': 'homo_sapiens',
        'expression_leaf': 'processed_labels/human_labels.pt',
        'max_nuclear_chrom': 22,
    },
    'mouse': {
        'name': 'mouse',
        'full_name': 'mus_musculus',
        'expression_leaf': 'processed_labels/mouse_labels.pt',
        'max_nuclear_chrom': 19,
    }
}


# =================================================================
# DeepCBA模型 - 精确移植自原始ppi_model.py
# =================================================================
class DeepCBAExactReproduction(nn.Module):
    """
    DeepCBA模型 v5.3 生产环境版
    策略：
    1. 恢复独立分支 (Independent Branches): 分别定义 branch_a/b，保证特征提取容量。
    2. 锁定 BN 稳定性 (affine=True): 解决量程漂移导致的 R² 为负问题。
    3. 维持池化对齐 (ceil_mode=True): 完美复现 6 时间步。
    """

    def __init__(self):
        super(DeepCBAExactReproduction, self).__init__()

        self.branch_a = self._make_branch()
        self.branch_b = self._make_branch()

        self.lstm_hidden_size = 128
        self.lstm_a = nn.LSTM(64, 128, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm_b = nn.LSTM(64, 128, num_layers=1, bidirectional=True, batch_first=True)

        self.flatten = nn.Flatten()
        self.dropout_fc = nn.Dropout(0.3)
        self.dense1 = nn.Linear(1536, 64)
        self.dense2 = nn.Linear(64, 1)

        self._initialize_weights()

    def _make_branch(self):
        """构建独立的分支块"""
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(4, 8), padding=0),
            nn.BatchNorm2d(64, momentum=0.01, eps=0.001, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 8), padding=(0, 3)),
            nn.BatchNorm2d(64, momentum=0.01, eps=0.001, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0, ceil_mode=True),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=(1, 8), padding=(0, 3)),
            nn.BatchNorm2d(128, momentum=0.01, eps=0.001, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 8), padding=(0, 3)),
            nn.BatchNorm2d(128, momentum=0.01, eps=0.001, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0, ceil_mode=True),
            nn.Dropout(0.3),

            nn.Conv2d(128, 64, kernel_size=(1, 8), padding=(0, 3)),
            nn.BatchNorm2d(64, momentum=0.01, eps=0.001, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 8), padding=(0, 3)),
            nn.BatchNorm2d(64, momentum=0.01, eps=0.001, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0, ceil_mode=True),
            nn.Dropout(0.3),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, 1, 4, 6000]
        dna_a, dna_b = torch.split(x, 3000, dim=3)

        conv_a = self.branch_a(dna_a)
        conv_b = self.branch_b(dna_b)

        batch_size = conv_a.size(0)
        feat_a, _ = self.lstm_a(conv_a.view(batch_size, 6, 64))
        feat_b, _ = self.lstm_b(conv_b.view(batch_size, 6, 64))

        scores = torch.matmul(feat_a, feat_b.transpose(-2, -1)) / 16.0
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, feat_b)

        out = self.flatten(context)
        out = self.dropout_fc(out)
        out = F.relu(self.dense1(out))
        return self.dense2(out)


# =================================================================
# PPI配对数据集
# =================================================================
class PPIPairDataset(Dataset):
    def __init__(self, sequences_tensor, gene_a_ids, expression_values,
                 transform=None, augment=False, seed=42):
        self.sequences = sequences_tensor
        self.gene_ids = gene_a_ids
        self.expressions = torch.FloatTensor(expression_values)
        self.transform = transform
        self.augment = augment
        self.seed = seed

        if len(self.expressions.shape) == 1:
            self.expressions = self.expressions.unsqueeze(1)

    def __len__(self):
        return len(self.sequences)

    def reverse_complement(self, sequence):
        reversed_seq = torch.flip(sequence, dims=[-1])
        complemented = torch.zeros_like(reversed_seq)
        complemented[0] = reversed_seq[3]
        complemented[1] = reversed_seq[2]
        complemented[2] = reversed_seq[1]
        complemented[3] = reversed_seq[0]
        return complemented

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        expression = self.expressions[idx]
        gene_id = self.gene_ids[idx]

        if self.augment and self.transform:
            sequence = self.transform(sequence, self.seed + idx)
        elif self.augment:
            local_rng = np.random.RandomState(self.seed + idx)
            if local_rng.rand() < 0.5:
                sequence = self.reverse_complement(sequence)

        return {
            'sequence': sequence,
            'expression': expression.squeeze(),
            'gene_id': gene_id
        }


class SequenceAugmentation:
    def __init__(self, augment_prob=0.3, seed=42):
        self.augment_prob = augment_prob
        self.seed = seed

    def __call__(self, sequence, sample_seed=None):
        if sample_seed is not None:
            local_rng = np.random.RandomState(sample_seed)
            augment_decision = local_rng.rand() < self.augment_prob
        else:
            local_rng = np.random.RandomState(self.seed)
            augment_decision = local_rng.rand() < self.augment_prob

        if not augment_decision:
            return sequence

        augmented = sequence.clone()

        if sample_seed is not None:
            if local_rng.rand() < 0.2:
                shuffle_mask = torch.rand(augmented.shape[-1]) < 0.01
                if shuffle_mask.any():
                    shuffle_indices = torch.randperm(shuffle_mask.sum())
                    augmented[:, shuffle_mask] = augmented[:, shuffle_mask][:, shuffle_indices]

            if local_rng.rand() < 0.2:
                zero_mask = torch.rand(augmented.shape[-1]) < 0.005
                augmented[:, zero_mask] = 0
        else:
            if local_rng.rand() < 0.2:
                shuffle_mask = torch.rand(augmented.shape[-1]) < 0.01
                if shuffle_mask.any():
                    shuffle_indices = torch.randperm(shuffle_mask.sum())
                    augmented[:, shuffle_mask] = augmented[:, shuffle_mask][:, shuffle_indices]

            if local_rng.rand() < 0.2:
                zero_mask = torch.rand(augmented.shape[-1]) < 0.005
                augmented[:, zero_mask] = 0

        return augmented


# =================================================================
# 训练器类
# =================================================================
class DeepCBATrainer:
    def __init__(self, model, device='cpu',
                 learning_rate=1e-4,
                 patience=15,
                 min_lr=1e-6,
                 seed=42):

        self.model = model.to(device)
        self.device = device
        self.seed = seed

        set_seed(seed)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-3,
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
            expressions = batch['expression'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(sequences).squeeze(-1)

            loss = self.criterion(outputs, expressions)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in pbar:
                sequences = batch['sequence'].to(self.device)
                expressions = batch['expression'].to(self.device)

                outputs = self.model(sequences).squeeze(-1)
                loss = self.criterion(outputs, expressions)

                total_loss += loss.item()
                num_batches += 1

                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(expressions.cpu().numpy())

        avg_loss = total_loss / num_batches
        return avg_loss, np.array(all_preds), np.array(all_targets)

    def train(self, train_loader, val_loader, epochs=100):
        train_losses = []
        val_losses = []
        learning_rates = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            train_losses.append(train_loss)

            val_loss, val_preds, val_targets = self.validate(val_loader)
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
                    print(f"  🚨 Early stopping at epoch {epoch + 1}")
                    print(f"     Best val loss: {self.best_loss:.6f} at epoch {self.best_epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  📊 Epoch {epoch + 1}/{epochs}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.6f}, "
                      f"Patience: {self.counter}/{self.patience}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"  ✅ Restored best model from epoch {self.best_epoch + 1}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
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
        results['spearman_corr'] = 0.0

    results['num_samples'] = int(len(y_true))

    return results


def print_evaluation_summary(results, title="DeepCBA模型性能"):
    print("\n" + "=" * 80)
    print(f"📈 {title}评估摘要")
    print("=" * 80)

    print(f"📊 样本数量: {results['num_samples']}")
    print(f"📊 回归指标:")
    print(f"  • R²: {results['r2']:.6f}")
    print(f"  • RMSE: {results['rmse']:.6f}")
    print(f"  • MAE: {results['mae']:.6f}")

    print(f"\n📊 相关性指标:")
    print(f"  • Pearson: {results['pearson_corr']:.6f}")
    print(f"  • Spearman: {results['spearman_corr']:.6f}")

    print("=" * 80)


# =================================================================
# 数据加载函数
# =================================================================
def load_ppi_data(species, sequences_dir='precomputed_sequences_ppi', labels_dir='processed_labels'):
    print(f"\n🔍 加载 {species} PPI数据...")

    sequences_file = os.path.join(sequences_dir, f'{species}_ppi_sequences.pt')

    if not os.path.exists(sequences_file):
        print(f"❌ PPI序列文件不存在: {sequences_file}")
        return None, None, None

    try:
        ppi_data = torch.load(sequences_file, map_location='cpu', weights_only=False)
        sequences = ppi_data['sequences']
        target_genes = ppi_data['target_genes']
        neighbor_genes = ppi_data['neighbor_genes']

        print(f"✅ 成功加载PPI序列数据:")
        print(f"   - 序列形状: {sequences.shape}")
        print(f"   - 基因A数量: {len(target_genes)}")

        if sequences.shape[-1] != 6000:
            print(f"⚠️ 警告: 序列长度不是6000bp: {sequences.shape[-1]}bp")

    except Exception as e:
        print(f"❌ 加载PPI序列文件失败: {e}")
        return None, None, None

    labels_file = os.path.join(labels_dir, f'{species}_labels.pt')

    if not os.path.exists(labels_file):
        print(f"❌ 标签文件不存在: {labels_file}")
        return None, None, None

    try:
        label_data = torch.load(labels_file, map_location='cpu', weights_only=False)
        labels = label_data['labels']
        gene_ids_label = label_data['gene_id']

        print(f"✅ 成功加载标签数据:")
        print(f"   - 标签形状: {labels.shape}")
        print(f"   - 基因数量: {len(gene_ids_label)}")

    except Exception as e:
        print(f"❌ 加载标签文件失败: {e}")
        return None, None, None

    # 对齐数据
    gene_to_label = {gid: idx for idx, gid in enumerate(gene_ids_label)}

    valid_indices = []
    valid_gene_a_ids = []
    valid_expressions = []

    for idx, gene_a in enumerate(target_genes):
        if gene_a in gene_to_label:
            label_idx = gene_to_label[gene_a]
            valid_indices.append(idx)
            valid_gene_a_ids.append(gene_a)
            valid_expressions.append(labels[label_idx].item())

    if not valid_indices:
        print(f"❌ 没有找到匹配的基因A表达量数据")
        return None, None, None

    valid_sequences = sequences[valid_indices]

    print(f"\n📊 数据对齐结果:")
    print(f"   - 总样本数: {len(target_genes)}")
    print(f"   - 有效样本数: {len(valid_indices)}")
    print(f"   - 对齐率: {len(valid_indices) / len(target_genes) * 100:.1f}%")

    self_copy_count = sum(1 for i in valid_indices if neighbor_genes[i] == target_genes[i])
    print(f"   - 其中自我复制基因: {self_copy_count} ({self_copy_count / len(valid_indices) * 100:.1f}%)")

    return valid_sequences, valid_gene_a_ids, valid_expressions


# =================================================================
# 单次训练运行（固定种子）
# =================================================================
def run_single_seed(species, sequences, gene_ids, expressions, args, device, seed):
    """使用固定种子进行单次训练"""
    print(f"\n{'=' * 50}")
    print(f"🎲 种子: {seed}")
    print(f"{'=' * 50}")

    set_seed(seed)

    # 划分数据集
    total = len(sequences)
    indices = list(range(total))

    train_idx, temp_idx = train_test_split(
        indices, train_size=0.7, random_state=seed, shuffle=True
    )
    val_ratio_adjusted = 0.15 / (0.15 + 0.15)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_ratio_adjusted, random_state=seed, shuffle=True
    )

    train_sequences = sequences[train_idx]
    train_gene_ids = [gene_ids[i] for i in train_idx]
    train_expressions = [expressions[i] for i in train_idx]

    val_sequences = sequences[val_idx]
    val_gene_ids = [gene_ids[i] for i in val_idx]
    val_expressions = [expressions[i] for i in val_idx]

    test_sequences = sequences[test_idx]
    test_gene_ids = [gene_ids[i] for i in test_idx]
    test_expressions = [expressions[i] for i in test_idx]

    print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}, 测试集: {len(test_idx)}")

    # 创建数据集
    transform = SequenceAugmentation(augment_prob=0.3, seed=seed) if args.augment else None

    train_dataset = PPIPairDataset(
        train_sequences, train_gene_ids, train_expressions,
        transform=transform, augment=args.augment, seed=seed
    )

    val_dataset = PPIPairDataset(
        val_sequences, val_gene_ids, val_expressions,
        augment=False
    )

    test_dataset = PPIPairDataset(
        test_sequences, test_gene_ids, test_expressions,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed)
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
    model = DeepCBAExactReproduction()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    # 训练
    trainer = DeepCBATrainer(
        model, device=device,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=seed
    )

    print(f"\n🚀 开始训练...")
    history = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 测试集评估
    test_loss, test_preds, test_targets = trainer.validate(test_loader)
    metrics = evaluate_regression(test_targets, test_preds)

    print(f"\n  📈 测试结果:")
    print(f"     R²: {metrics['r2']:.6f}")
    print(f"     Pearson: {metrics['pearson_corr']:.6f}")
    print(f"     Spearman: {metrics['spearman_corr']:.6f}")
    print(f"     RMSE: {metrics['rmse']:.6f}")

    result = {
        'seed': seed,
        'best_epoch': history['best_epoch'] + 1,
        'best_val_loss': float(history['best_val_loss']),
        'test_loss': float(test_loss),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'test_r2': metrics['r2'],
        'test_pearson': metrics['pearson_corr'],
        'test_spearman': metrics['spearman_corr'],
        'test_rmse': metrics['rmse'],
        'test_mae': metrics['mae'],
        'model_params': total_params
    }

    return result


# =================================================================
# 多种子训练
# =================================================================
def train_multi_seed(species, sequences, gene_ids, expressions, args, device):
    """使用多个种子进行训练"""
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    print(f"\n{'=' * 70}")
    print(f"🚀 多种子训练 - {species.upper()} - DeepCBA")
    print(f"种子列表: {seeds}")
    print(f"{'=' * 70}")

    all_results = []
    for seed in seeds:
        result = run_single_seed(species, sequences, gene_ids, expressions, args, device, seed)
        if result:
            all_results.append(result)

    if not all_results:
        return None

    # 计算统计量
    metrics_keys = ['test_r2', 'test_pearson', 'test_spearman', 'test_rmse']

    print(f"\n{'=' * 70}")
    print(f"📊 {species.upper()} - DeepCBA 多种子汇总结果")
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
    results_file = os.path.join(args.output_dir, f'{species}_deepcba_seed_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 各种子指标已保存: {results_file}")

    # 保存汇总统计
    summary = {
        'species': species,
        'model': 'DeepCBA',
        'seeds': seeds,
        'num_seeds': len(seeds),
        'metrics_summary': metrics_summary
    }

    with open(os.path.join(args.output_dir, f'{species}_deepcba_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 汇总统计已保存: {args.output_dir}/{species}_deepcba_summary.json")

    # 打印均值±标准差格式
    print(f"\n📈 {species.upper()} - DeepCBA 性能指标 (均值 ± 标准差):")
    print(f"   R²:       {metrics_summary['test_r2']['mean']:.6f} ± {metrics_summary['test_r2']['std']:.6f}")
    print(f"   Pearson:  {metrics_summary['test_pearson']['mean']:.6f} ± {metrics_summary['test_pearson']['std']:.6f}")
    print(f"   Spearman: {metrics_summary['test_spearman']['mean']:.6f} ± {metrics_summary['test_spearman']['std']:.6f}")
    print(f"   RMSE:     {metrics_summary['test_rmse']['mean']:.6f} ± {metrics_summary['test_rmse']['std']:.6f}")

    return results_df


# =================================================================
# 主函数
# =================================================================
def main():
    parser = argparse.ArgumentParser(description='DeepCBA模型训练 - 人类/小鼠版本 - 多种子实验')

    # 物种选择
    parser.add_argument('--species', type=str, default='all',
                        choices=['human', 'mouse', 'all'],
                        help='要训练的物种')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='初始学习率')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')
    parser.add_argument('--no_augment', action='store_true',
                        help='禁用数据增强')

    # 随机种子
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'随机种子列表 (默认: {DEFAULT_SEEDS})')

    # 路径参数
    parser.add_argument('--sequences_dir', type=str, default='precomputed_sequences_ppi',
                        help='PPI序列目录')
    parser.add_argument('--labels_dir', type=str, default='processed_labels',
                        help='标签目录')
    parser.add_argument('--output_dir', type=str, default='training_results_deepcba',
                        help='输出目录')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    args.augment = not args.no_augment

    # 解析物种列表
    if args.species.lower() == 'all':
        species_list = ['human', 'mouse']
    else:
        species_list = [args.species]

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 DeepCBA模型训练 - 人类/小鼠版本 - 多种子实验")
    print("=" * 80)
    print(f"\n🔧 训练配置:")
    print(f"  物种: {species_list}")
    print(f"  随机种子: {args.seeds}")
    print(f"  数据增强: {'启用' if args.augment else '禁用'}")
    print(f"  训练策略: 70/15/15 数据集划分")
    print(f"  模型: DeepCBA (精确移植)")
    print(f"  序列目录: {args.sequences_dir}")
    print(f"  标签目录: {args.labels_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  设备: {device}")
    print("=" * 80)

    # 保存配置
    config = {
        'species_list': species_list,
        'seeds': args.seeds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'augment': args.augment,
        'sequences_dir': args.sequences_dir,
        'labels_dir': args.labels_dir,
        'output_dir': args.output_dir,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }

    config_file = os.path.join(args.output_dir, 'experiment_config_deepcba.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n⚙️ 实验配置已保存: {config_file}")

    # 训练每个物种
    for species in species_list:
        print(f"\n{'=' * 60}")
        print(f"🌿 处理物种: {species.upper()}")
        print(f"{'=' * 60}")

        # 加载数据
        sequences, gene_ids, expressions = load_ppi_data(
            species,
            sequences_dir=args.sequences_dir,
            labels_dir=args.labels_dir
        )

        if sequences is None:
            print(f"❌ 数据加载失败，跳过 {species}")
            continue

        # 多种子训练
        train_multi_seed(species, sequences, gene_ids, expressions, args, device)

    print(f"\n{'=' * 80}")
    print("✅ 全部训练完成!")
    print(f"   结果保存在: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()