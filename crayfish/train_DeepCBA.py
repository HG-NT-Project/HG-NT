"""
DeepCBA模型训练 - 适配小龙虾
修改为单序列输入（3000bp），保持与DeepCRE/PlantCRE一致的输入格式
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
# 物种配置 - 适配小龙虾
# =================================================================
SPECIES_CONFIG = {
    'crayfish': {
        'name': 'crayfish',
        'full_name': 'procambarus_clarkii',
        'sequences_file': 'processed_sequence/crayfish_sequences.pt',
        'labels_file': 'crayfish_labels.csv',
    }
}

SEQUENCE_LENGTH = 3000


# =================================================================
# DeepCBA模型 - 修改为单序列输入版本
# =================================================================
class DeepCBA_SingleSequence(nn.Module):
    """
    DeepCBA模型 - 单序列版本（适配小龙虾）
    输入: [batch, 4, 3000] 单基因序列
    输出: 预测表达值
    """

    def __init__(self):
        super(DeepCBA_SingleSequence, self).__init__()

        self.conv_branch = self._make_branch()

        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(64, 128, num_layers=1, bidirectional=True, batch_first=True)

        self.flatten = nn.Flatten()
        self.dropout_fc = nn.Dropout(0.3)
        self.dense1 = nn.Linear(1536, 64)
        self.dense2 = nn.Linear(64, 1)

        self._initialize_weights()

    def _make_branch(self):
        """构建卷积分支"""
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
            x = x.unsqueeze(1)

        conv_out = self.conv_branch(x)
        batch_size = conv_out.size(0)

        lstm_input = conv_out.permute(0, 3, 1, 2).reshape(batch_size, 6, 64)
        lstm_out, _ = self.lstm(lstm_input)

        attention_scores = torch.matmul(lstm_out, lstm_out.transpose(-2, -1)) / 16.0
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, lstm_out)

        out = self.flatten(context)
        out = self.dropout_fc(out)
        out = F.relu(self.dense1(out))
        return self.dense2(out)


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
class DeepCBATrainer:
    def __init__(self, model, device='cpu',
                 learning_rate=1e-4,
                 weight_decay=1e-4,
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
            outputs = self.model(sequences).squeeze(-1)

            loss = self.criterion(outputs, labels.squeeze())
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
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in pbar:
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(sequences).squeeze(-1)
                loss = self.criterion(outputs, labels.squeeze())

                total_loss += loss.item()
                num_batches += 1

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.squeeze().cpu().numpy())

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss, np.array(all_preds), np.array(all_labels)

    def train(self, train_loader, val_loader, epochs=100):
        train_losses = []
        val_losses = []
        learning_rates = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            train_losses.append(train_loss)

            val_loss, val_preds, val_labels = self.validate(val_loader)
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

        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        results['spearman_corr'] = float(spearman_corr)
        results['spearman_p'] = float(spearman_p)

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
# 数据加载函数
# =================================================================
def load_crayfish_data(sequences_dir='processed_sequence', labels_file='crayfish_labels.csv'):
    print(f"\n🔍 加载小龙虾数据...")

    config = SPECIES_CONFIG['crayfish']

    sequences_file = os.path.join(sequences_dir, os.path.basename(config['sequences_file']))
    if not os.path.exists(sequences_file):
        print(f"❌ 序列文件不存在: {sequences_file}")
        return None, None, None

    try:
        seq_data = torch.load(sequences_file, map_location='cpu', weights_only=False)
        sequences = seq_data['sequences']
        gene_ids_seq = seq_data['target_genes']
        print(f"✅ 成功加载序列数据: {sequences.shape}")
        print(f"   基因数量: {len(gene_ids_seq)}")
    except Exception as e:
        print(f"❌ 加载序列数据失败: {e}")
        return None, None, None

    if not os.path.exists(labels_file):
        print(f"❌ 标签文件不存在: {labels_file}")
        return None, None, None

    try:
        labels_df = pd.read_csv(labels_file)
        print(f"✅ 成功加载标签数据: {labels_df.shape}")
        print(f"   列名: {list(labels_df.columns)}")

        if 'gene_id' not in labels_df.columns:
            possible_id_cols = ['GeneID', 'gene', 'Gene', 'id', 'ID']
            found = False
            for col in possible_id_cols:
                if col in labels_df.columns:
                    labels_df = labels_df.rename(columns={col: 'gene_id'})
                    found = True
                    print(f"🔄 已将 '{col}' 列重命名为 'gene_id'")
                    break
            if not found:
                print(f"❌ 未找到基因ID列，可用的列: {list(labels_df.columns)}")
                return None, None, None

        if 'label' not in labels_df.columns:
            possible_label_cols = ['mean_expression', 'expression', 'log2_expression', 'tpm']
            found = False
            for col in possible_label_cols:
                if col in labels_df.columns:
                    labels_df = labels_df.rename(columns={col: 'label'})
                    found = True
                    print(f"🔄 已将 '{col}' 列重命名为 'label'")
                    break
            if not found:
                print(f"❌ 未找到标签列，可用的列: {list(labels_df.columns)}")
                return None, None, None

        initial_count = len(labels_df)
        labels_df = labels_df[labels_df['label'].notna()].copy()
        if len(labels_df) < initial_count:
            print(f"🧹 过滤label为NaN: {initial_count} → {len(labels_df)}")

        gene_ids_label = labels_df['gene_id'].tolist()
        labels = torch.tensor(labels_df['label'].values, dtype=torch.float32)

        print(f"   标签数据: {len(gene_ids_label)} 个基因")

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

    for gene_id in gene_ids_seq:
        if gene_id in gene_id_to_label_idx:
            seq_idx = gene_id_to_seq_idx[gene_id]
            label_idx = gene_id_to_label_idx[gene_id]

            aligned_sequences.append(sequences[seq_idx])
            aligned_labels.append(labels[label_idx])
            aligned_gene_ids.append(gene_id)

    aligned_sequences = torch.stack(aligned_sequences)
    aligned_labels = torch.stack(aligned_labels)

    print(f"✅ 数据对齐完成: {aligned_sequences.shape}, {aligned_labels.shape}")

    return aligned_sequences, aligned_labels, aligned_gene_ids


# =================================================================
# 单次训练运行（固定种子）
# =================================================================
def run_single_seed(sequences, labels, gene_ids, args, device, seed):
    """使用固定种子进行单次训练"""
    print(f"\n{'=' * 50}")
    print(f"🎲 种子: {seed}")
    print(f"{'=' * 50}")

    set_seed(seed)

    # 划分数据集
    total = len(sequences)
    indices = list(range(total))

    train_idx, temp_idx = train_test_split(
        indices, train_size=args.train_ratio, random_state=seed, shuffle=True
    )
    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_ratio_adjusted, random_state=seed, shuffle=True
    )

    train_sequences = sequences[train_idx]
    train_labels = labels[train_idx]
    train_gene_ids = [gene_ids[i] for i in train_idx]

    val_sequences = sequences[val_idx]
    val_labels = labels[val_idx]
    val_gene_ids = [gene_ids[i] for i in val_idx]

    test_sequences = sequences[test_idx]
    test_labels = labels[test_idx]
    test_gene_ids = [gene_ids[i] for i in test_idx]

    print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}, 测试集: {len(test_idx)}")

    # 创建数据集
    transform = SequenceAugmentation(augment_prob=0.3) if args.augment else None

    train_dataset = GeneSequenceDataset(
        train_sequences, train_labels, train_gene_ids,
        transform=transform, augment=args.augment
    )

    val_dataset = GeneSequenceDataset(
        val_sequences, val_labels, val_gene_ids,
        augment=False
    )

    test_dataset = GeneSequenceDataset(
        test_sequences, test_labels, test_gene_ids,
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
    model = DeepCBA_SingleSequence()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    # 训练
    trainer = DeepCBATrainer(
        model, device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=seed
    )

    print(f"\n🚀 开始训练...")
    history = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 测试集评估
    test_loss, test_preds, test_labels_arr = trainer.validate(test_loader)
    metrics = evaluate_regression(test_labels_arr, test_preds)

    print(f"\n  📈 测试结果:")
    print(f"     R²: {metrics['r2']:.6f}")
    print(f"     Pearson: {metrics['pearson_corr']:.6f}")
    print(f"     Spearman: {metrics['spearman_corr']:.6f}")
    print(f"     RMSE: {metrics['rmse']:.6f}")

    result = {
        'seed': seed,
        'model_type': 'DeepCBA',
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
def train_multi_seed(sequences, labels, gene_ids, args, device):
    """使用多个种子进行训练"""
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    print(f"\n{'=' * 70}")
    print(f"🚀 多种子训练 - 小龙虾 DeepCBA")
    print(f"种子列表: {seeds}")
    print(f"{'=' * 70}")

    all_results = []
    for seed in seeds:
        result = run_single_seed(sequences, labels, gene_ids, args, device, seed)
        if result:
            all_results.append(result)

    if not all_results:
        return None

    # 保存各种子指标到CSV
    results_df = pd.DataFrame(all_results)
    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'crayfish_deepcba_seed_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 各种子指标已保存: {results_file}")

    # 计算统计量
    metrics_keys = ['test_r2', 'test_pearson', 'test_spearman', 'test_rmse']

    print(f"\n{'=' * 70}")
    print(f"📊 小龙虾 DeepCBA 多种子汇总结果")
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

    # 保存汇总统计
    summary = {
        'species': 'crayfish',
        'model': 'DeepCBA',
        'seeds': seeds,
        'num_seeds': len(seeds),
        'metrics_summary': metrics_summary
    }

    with open(os.path.join(results_dir, 'crayfish_deepcba_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 汇总统计已保存: {results_dir}/crayfish_deepcba_summary.json")

    # 打印均值±标准差格式
    print(f"\n📈 小龙虾 DeepCBA 性能指标 (均值 ± 标准差):")
    print(f"   R²:       {metrics_summary['test_r2']['mean']:.6f} ± {metrics_summary['test_r2']['std']:.6f}")
    print(f"   Pearson:  {metrics_summary['test_pearson']['mean']:.6f} ± {metrics_summary['test_pearson']['std']:.6f}")
    print(f"   Spearman: {metrics_summary['test_spearman']['mean']:.6f} ± {metrics_summary['test_spearman']['std']:.6f}")
    print(f"   RMSE:     {metrics_summary['test_rmse']['mean']:.6f} ± {metrics_summary['test_rmse']['std']:.6f}")

    return results_df


# =================================================================
# 主函数
# =================================================================
def main():
    parser = argparse.ArgumentParser(description='小龙虾DeepCBA模型训练 - 多种子实验')

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

    # 其他选项
    parser.add_argument('--no_augment', action='store_true',
                        help='禁用数据增强')
    parser.add_argument('--sequences_dir', type=str, default='processed_sequence',
                        help='预处理序列目录')
    parser.add_argument('--labels_file', type=str, default='crayfish_labels.csv',
                        help='标签文件路径')
    parser.add_argument('--output_dir', type=str, default='crayfish_deepcba_results',
                        help='输出目录')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    args.augment = not args.no_augment

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 小龙虾DeepCBA模型训练 - 多种子实验")
    print("=" * 80)
    print(f"\n🔧 训练配置:")
    print(f"  物种: 小龙虾 (Procambarus clarkii)")
    print(f"  模型: DeepCBA (单序列版本)")
    print(f"  随机种子: {args.seeds}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  早停耐心: {args.patience}")
    print(f"  数据增强: {args.augment}")
    print(f"  数据集划分: 训练 {args.train_ratio:.0%}, 验证 {args.val_ratio:.0%}, 测试 {args.test_ratio:.0%}")
    print(f"  序列长度: {SEQUENCE_LENGTH}bp")
    print(f"  序列目录: {args.sequences_dir}")
    print(f"  标签文件: {args.labels_file}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  设备: {device}")
    print("=" * 80)

    # 验证划分比例
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print(f"❌ 划分比例之和必须为1，当前和为 {args.train_ratio + args.val_ratio + args.test_ratio}")
        return

    # 保存配置
    config = {
        'seeds': args.seeds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'augment': args.augment,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'sequences_dir': args.sequences_dir,
        'labels_file': args.labels_file,
        'output_dir': args.output_dir,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }

    config_file = os.path.join(args.output_dir, 'crayfish_deepcba_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n⚙️ 实验配置已保存: {config_file}")

    # 加载数据
    sequences, labels, gene_ids = load_crayfish_data(
        sequences_dir=args.sequences_dir,
        labels_file=args.labels_file
    )

    if sequences is None:
        print(f"❌ 数据加载失败")
        return

    # 多种子训练
    train_multi_seed(sequences, labels, gene_ids, args, device)

    print(f"\n{'=' * 80}")
    print("✅ 全部训练完成!")
    print(f"   结果保存在: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()