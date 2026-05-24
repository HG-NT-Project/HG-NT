"""
Enformer 特征训练脚本 - MLP版本
- 自适应MLP: 根据物种自动调整网络结构
- 5次随机种子划分，完整统计指标（均值 ± 标准差）
- 支持同时训练人类和小鼠两个物种
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from datetime import datetime
import random
import warnings

warnings.filterwarnings('ignore')

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
FEATURES_DIR = "enformer_features_cache"


# =================================================================
# 固定随机种子
# =================================================================
def set_seed(seed=42):
    """固定所有随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =================================================================
# 安全数据加载 (PyTorch 2.6+ 兼容)
# =================================================================
def load_features_safe(features_file):
    """
    安全加载特征文件，兼容PyTorch 2.6+
    必须显式设置 weights_only=False 以避免 UnpicklingError
    """
    print(f"📂 加载特征文件: {features_file}")
    data = torch.load(features_file, map_location='cpu', weights_only=False)

    # 提取特征和标签
    features = data['features'].numpy() if torch.is_tensor(data['features']) else data['features']
    labels = data['labels_log'].numpy() if torch.is_tensor(data['labels_log']) else data['labels_log']

    print(f"   特征形状: {features.shape}")
    print(f"   标签形状: {labels.shape}")
    print(f"   特征维度: {features.shape[1]}")

    return features, labels


# =================================================================
# 数据集类
# =================================================================
class EnformerFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# =================================================================
# 自适应MLP（根据输入维度自动调整）
# =================================================================
class AdaptiveMLP(nn.Module):
    """
    自适应MLP回归头
    根据输入维度自动决定网络结构

    人类: input_dim -> 256 -> Dropout -> 128 -> Dropout -> 64 -> Dropout -> 1
    小鼠: input_dim -> 32 -> Dropout -> 16 -> Dropout -> 1
    """

    def __init__(self, input_dim, dropout=0.1):
        super().__init__()

        # 根据输入维度自动决定网络结构（动态适应人类/小鼠）
        if input_dim >= 100:  # 人类: 485维
            hidden_dims = [256, 128, 64]
        else:  # 小鼠: 43维
            hidden_dims = [32, 16]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # 打印网络结构
        print(f"   MLP结构: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 1")
        print(f"   总参数量: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


# =================================================================
# MLP训练器
# =================================================================
class MLPTrainer:
    def __init__(self, model, device, lr=1e-3, weight_decay=1e-4, patience=15):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                total_loss += loss.item()
                preds.extend(out.cpu().numpy())
                targets.extend(y.cpu().numpy())
        return total_loss / len(loader), np.array(preds), np.array(targets)

    def train(self, train_loader, val_loader, epochs=100):
        print(f"   开始训练 (早停耐心={self.patience})...")
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, _, _ = self.validate(val_loader)
            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.counter = 0
            else:
                self.counter += 1

            if (epoch + 1) % 20 == 0:
                print(f"     Epoch {epoch + 1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

            if self.counter >= self.patience:
                print(f"     早停于 Epoch {epoch + 1}")
                break

        self.model.load_state_dict(self.best_state)
        return self.best_loss


# =================================================================
# 评估函数
# =================================================================
def calculate_metrics(y_true, y_pred):
    """计算完整的评估指标"""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    if len(y_true) < 2:
        return None

    with np.errstate(invalid='ignore'):
        pearson = pearsonr(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred)[0]
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    pearson = pearson if not np.isnan(pearson) else 0.0
    spearman = spearman if not np.isnan(spearman) else 0.0

    return {
        'pearson': pearson,
        'spearman': spearman,
        'r2': r2,
        'rmse': rmse,
    }


# =================================================================
# 单次训练（固定种子）
# =================================================================
def run_single_seed(features, labels, seed, args):
    """使用固定种子进行单次训练"""
    set_seed(seed)

    input_dim = features.shape[1]

    print(f"\n  🎲 种子: {seed}")

    # 划分数据: 70% / 15% / 15%
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, train_size=0.7, random_state=seed, shuffle=True
    )
    val_ratio = 0.15 / 0.3
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=seed, shuffle=True
    )

    print(f"    训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 创建数据加载器
    train_loader = DataLoader(
        EnformerFeatureDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        EnformerFeatureDataset(X_val, y_val),
        batch_size=args.batch_size * 2
    )
    test_loader = DataLoader(
        EnformerFeatureDataset(X_test, y_test),
        batch_size=args.batch_size * 2
    )

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdaptiveMLP(input_dim=input_dim, dropout=args.dropout)
    print(f"    设备: {device}")

    # 训练
    trainer = MLPTrainer(
        model, device, lr=args.learning_rate,
        weight_decay=args.weight_decay, patience=args.patience
    )

    best_val_loss = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 测试集评估
    _, test_preds, test_targets = trainer.validate(test_loader)
    metrics = calculate_metrics(test_targets, test_preds)

    if metrics:
        print(f"    测试: R²={metrics['r2']:.6f}, Pearson={metrics['pearson']:.6f}, "
              f"Spearman={metrics['spearman']:.6f}, RMSE={metrics['rmse']:.6f}")

    return {
        'seed': int(seed),
        'best_val_loss': float(best_val_loss),
        'train_size': int(len(X_train)),
        'val_size': int(len(X_val)),
        'test_size': int(len(X_test)),
        'test_r2': float(metrics['r2']) if metrics else None,
        'test_pearson': float(metrics['pearson']) if metrics else None,
        'test_spearman': float(metrics['spearman']) if metrics else None,
        'test_rmse': float(metrics['rmse']) if metrics else None,
    } if metrics else None


# =================================================================
# 多种子训练
# =================================================================
def run_multi_seed(features_file, species, args):
    """使用多个种子进行训练"""

    # 安全加载数据
    features, labels = load_features_safe(features_file)
    input_dim = features.shape[1]

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    print(f"\n{'=' * 70}")
    print(f"🔬 多种子训练 - {species.upper()}")
    print(f"   特征维度: {input_dim}")
    print(f"   种子列表: {seeds}")
    print(f"{'=' * 70}")

    all_results = []
    for seed in seeds:
        result = run_single_seed(features, labels, seed, args)
        if result:
            all_results.append(result)

    if not all_results:
        return None, None

    # 统计汇总
    print(f"\n{'=' * 70}")
    print(f"📊 {species.upper()} 多种子汇总结果")
    print(f"{'=' * 70}")

    metrics_keys = ['test_r2', 'test_pearson', 'test_spearman', 'test_rmse']
    metric_names = {'test_r2': 'R²', 'test_pearson': 'Pearson',
                    'test_spearman': 'Spearman', 'test_rmse': 'RMSE'}

    print(f"\n{'指标':<12} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
    print(f"{'-' * 60}")

    metrics_summary = {}
    for key in metrics_keys:
        values = [r[key] for r in all_results if r[key] is not None]
        if values:
            metrics_summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': [float(v) for v in values]
            }
            print(f"{metric_names[key]:<12} {np.mean(values):<12.6f} {np.std(values, ddof=1):<12.6f} "
                  f"{np.min(values):<12.6f} {np.max(values):<12.6f}")

    # 保存结果到CSV
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(args.output_dir, f'{species}_seed_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 各种子指标已保存: {results_file}")

    # 保存汇总统计
    def convert_to_native(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj

    summary = {
        'species': species,
        'feature_dim': int(input_dim),
        'seeds': [int(s) for s in seeds],
        'num_seeds': len(seeds),
        'model_structure': {
            'human': '485 -> 256 -> 128 -> 64 -> 1',
            'mouse': '43 -> 32 -> 16 -> 1'
        },
        'metrics_summary': convert_to_native(metrics_summary),
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(args.output_dir, f'{species}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 汇总统计已保存: {args.output_dir}/{species}_summary.json")

    # 打印均值±标准差格式
    print(f"\n📈 {species.upper()} 性能指标 (均值 ± 标准差):")
    for key in metrics_keys:
        if key in metrics_summary:
            print(f"   {metric_names[key]}: {metrics_summary[key]['mean']:.6f} ± {metrics_summary[key]['std']:.6f}")

    return results_df, metrics_summary


# =================================================================
# 主函数
# =================================================================
def main():
    parser = argparse.ArgumentParser(description='Enformer特征训练 - 自适应MLP')
    parser.add_argument('--species', type=str, required=True,
                        choices=['human', 'mouse', 'all'],
                        help='物种 (human, mouse, 或 all 同时训练两个物种)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout比例')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'随机种子列表 (默认: {DEFAULT_SEEDS})')
    parser.add_argument('--features_dir', type=str, default=FEATURES_DIR,
                        help='特征文件目录')
    parser.add_argument('--output_dir', type=str, default='Results_Enformer',
                        help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 确定要训练的物种列表
    if args.species == 'all':
        species_list = ['human', 'mouse']
        print("=" * 80)
        print("🔬 Enformer 特征训练 - 自适应MLP")
        print("=" * 80)
        print(f"\n🌟 将依次训练两个物种: 人类 (Human) 和 小鼠 (Mouse)")
    else:
        species_list = [args.species]
        print("=" * 80)
        print("🔬 Enformer 特征训练 - 自适应MLP")
        print("=" * 80)

    print(f"\n配置:")
    print(f"  随机种子: {args.seeds}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  Dropout: {args.dropout}")
    print(f"  输出目录: {args.output_dir}")
    print("=" * 80)

    # 记录所有物种的结果
    all_results = {}

    # 依次训练每个物种
    for species in species_list:
        print(f"\n{'#' * 80}")
        print(f"# 开始训练: {species.upper()}")
        print(f"{'#' * 80}")

        features_file = os.path.join(args.features_dir, f'{species}_enformer_features.pt')

        if not os.path.exists(features_file):
            print(f"❌ 特征文件不存在: {features_file}")
            print("   请先运行 extract_enformer_features.py 提取特征")
            continue

        results_df, summary = run_multi_seed(features_file, species, args)
        all_results[species] = {'df': results_df, 'summary': summary}
        print(f"\n✅ {species.upper()} 训练完成!")

    # 打印最终汇总对比
    if len(species_list) > 1 and len(all_results) == 2:
        print(f"\n{'=' * 80}")
        print("🏆 最终汇总: 人类 vs 小鼠 性能对比")
        print(f"{'=' * 80}")
        print(f"\n{'指标':<12} {'人类 (Human)':<25} {'小鼠 (Mouse)':<25}")
        print(f"{'-' * 62}")

        metrics_keys = ['test_pearson', 'test_spearman', 'test_r2', 'test_rmse']
        metric_names = {'test_pearson': 'Pearson', 'test_spearman': 'Spearman',
                        'test_r2': 'R²', 'test_rmse': 'RMSE'}

        for metric_key in metrics_keys:
            human_val = "N/A"
            mouse_val = "N/A"

            if 'human' in all_results and all_results['human']['summary']:
                if metric_key in all_results['human']['summary']:
                    human_val = f"{all_results['human']['summary'][metric_key]['mean']:.6f} ± {all_results['human']['summary'][metric_key]['std']:.6f}"

            if 'mouse' in all_results and all_results['mouse']['summary']:
                if metric_key in all_results['mouse']['summary']:
                    mouse_val = f"{all_results['mouse']['summary'][metric_key]['mean']:.6f} ± {all_results['mouse']['summary'][metric_key]['std']:.6f}"

            print(f" {metric_names[metric_key]:<11} {human_val:<25} {mouse_val:<25}")

    print(f"\n{'=' * 80}")
    print("✅ 全部训练完成!")
    print(f"   结果保存在: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()