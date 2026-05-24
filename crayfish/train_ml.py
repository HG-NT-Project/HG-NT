"""
小龙虾基因表达回归预测
适配crayfish_basic_features.csv和crayfish_labels.csv
取消染色体过滤，处理Scaffold数据
支持多种子重复实验，输出性能指标的均值和标准差
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import warnings
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time
import random
import json
from datetime import datetime

# 添加LightGBM导入
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM已安装")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM未安装，跳过LightGBM模型")

# 添加XGBoost导入
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
    print("✅ XGBoost已安装")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost未安装，跳过XGBoost模型")

warnings.filterwarnings('ignore')
pd.options.display.width = 0

# 默认5个随机种子
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# =============================================================================================
#  文件路径配置
# =============================================================================================

FEATURE_FILE = 'generated_features/crayfish_basic_features.csv'
LABEL_FILE = 'crayfish_labels.csv'

# 使用的特征列
FEATURE_COLS = ['GC_promoter', 'CpG_promoter', 'GC_terminator', 'CpG_terminator', 'gene_length']

# 目标列名（初始值）
TARGET_COL = 'label'

# 是否对gene_length进行Log10变换
LOG_TRANSFORM_GENE_LENGTH = True


# =============================================================================================
#  固定随机种子函数
# =============================================================================================
def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch_seed_available = False
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch_seed_available = True
    except ImportError:
        pass

    print(f"✅ 随机种子已设置为: {seed}")


# =============================================================================================
#  数据加载和预处理
# =============================================================================================

def load_data():
    """
    加载特征和标签数据，进行内连接和清洗
    过滤掉目标值为NaN的样本
    """
    global TARGET_COL

    print(f"\n📂 加载特征文件: {FEATURE_FILE}")

    if not os.path.exists(FEATURE_FILE):
        print(f"❌ 特征文件不存在: {FEATURE_FILE}")
        return None, None

    features_df = pd.read_csv(FEATURE_FILE)
    print(f"✅ 特征数据加载成功: {features_df.shape}")
    print(f"   列名: {list(features_df.columns)}")

    missing_cols = [col for col in FEATURE_COLS if col not in features_df.columns]
    if missing_cols:
        print(f"❌ 特征文件中缺少列: {missing_cols}")
        return None, None

    print(f"\n📂 加载标签文件: {LABEL_FILE}")

    if not os.path.exists(LABEL_FILE):
        print(f"❌ 标签文件不存在: {LABEL_FILE}")
        return None, None

    labels_df = pd.read_csv(LABEL_FILE)
    print(f"✅ 标签数据加载成功: {labels_df.shape}")
    print(f"   列名: {list(labels_df.columns)}")

    if 'GeneID' in labels_df.columns:
        labels_df = labels_df.rename(columns={'GeneID': 'gene_id'})
        print(f"🔄 已将标签文件的 'GeneID' 列重命名为 'gene_id'")

    if 'gene_id' not in labels_df.columns:
        print(f"❌ 标签文件中缺少 'gene_id' 列")
        return None, None

    target_col_local = TARGET_COL
    if target_col_local not in labels_df.columns:
        print(f"⚠️ 标签文件中缺少目标列: {target_col_local}")
        possible_targets = ['mean_expression', 'expression', 'log2_expression', 'tpm', 'label']
        found = False
        for col in possible_targets:
            if col in labels_df.columns:
                print(f"   使用 '{col}' 作为目标列")
                target_col_local = col
                TARGET_COL = col
                found = True
                break
        if not found:
            print(f"❌ 未找到合适的目标列，可用的列有: {list(labels_df.columns)}")
            return None, None

    # 过滤目标值为NaN的样本
    print(f"\n🧹 过滤目标值为NaN的样本...")
    initial_label_count = len(labels_df)
    nan_count = labels_df[target_col_local].isna().sum()

    if nan_count > 0:
        print(f"   发现 {nan_count} 个样本的目标值为NaN")
        labels_df = labels_df[labels_df[target_col_local].notna()].copy()
        print(f"   过滤后标签数据: {initial_label_count} → {len(labels_df)} (移除 {initial_label_count - len(labels_df)} 条)")

    if len(labels_df) == 0:
        print(f"❌ 所有标签值均为NaN，无法进行训练")
        return None, None

    print(f"\n🔗 按 gene_id 进行内连接...")
    merged_df = features_df.merge(labels_df, on='gene_id', how='inner')
    print(f"✅ 合并后数据: {merged_df.shape}")
    print(f"   匹配率: {len(merged_df)}/{len(features_df)} ({len(merged_df) / len(features_df) * 100:.1f}%)")

    if merged_df.empty:
        print("❌ 合并后数据为空")
        return None, None

    # 数据清洗
    print(f"\n🧹 数据清洗...")
    initial_count = len(merged_df)

    if 'gene_length' not in merged_df.columns:
        print("❌ 数据中没有 'gene_length' 列")
        return None, None

    merged_df = merged_df[merged_df['gene_length'] > 0].copy()
    after_len_filter = len(merged_df)
    print(f"   剔除 gene_length <= 0: {initial_count} → {after_len_filter} (移除 {initial_count - after_len_filter} 条)")

    if LOG_TRANSFORM_GENE_LENGTH:
        print(f"\n📊 对 gene_length 进行 Log10 变换...")
        print(f"   变换前 - 范围: [{merged_df['gene_length'].min():.2f}, {merged_df['gene_length'].max():.2f}]")
        print(f"   变换前 - 均值: {merged_df['gene_length'].mean():.2f}")
        print(f"   变换前 - 中位数: {merged_df['gene_length'].median():.2f}")

        merged_df['gene_length'] = np.log10(merged_df['gene_length'])

        print(f"   变换后 - 范围: [{merged_df['gene_length'].min():.4f}, {merged_df['gene_length'].max():.4f}]")
        print(f"   变换后 - 均值: {merged_df['gene_length'].mean():.4f}")
        print(f"   变换后 - 中位数: {merged_df['gene_length'].median():.4f}")
        print(f"   ✅ Log10变换完成")
    else:
        print(f"   ⚠️ 跳过 gene_length 的 Log10 变换")

    for col in FEATURE_COLS:
        if col in merged_df.columns:
            null_count = merged_df[col].isnull().sum()
            if null_count > 0:
                print(f"   {col}: {null_count} 个缺失值，将剔除")
                merged_df = merged_df.dropna(subset=[col])

    target_nan_count = merged_df[target_col_local].isna().sum()
    if target_nan_count > 0:
        print(f"   目标值仍有 {target_nan_count} 个NaN，将剔除")
        merged_df = merged_df[merged_df[target_col_local].notna()].copy()

    final_count = len(merged_df)
    print(f"   最终数据量: {final_count}")

    if final_count == 0:
        print("❌ 清洗后数据为空")
        return None, None

    print(f"\n📈 目标值统计 ({target_col_local}):")
    print(f"   有效样本数: {final_count}")
    print(f"   范围: [{merged_df[target_col_local].min():.4f}, {merged_df[target_col_local].max():.4f}]")
    print(f"   均值: {merged_df[target_col_local].mean():.4f}")
    print(f"   中位数: {merged_df[target_col_local].median():.4f}")
    print(f"   标准差: {merged_df[target_col_local].std():.4f}")

    if merged_df[target_col_local].min() < 0:
        print(f"   ⚠️ 目标值存在负数，可能需要进行log变换")

    return merged_df, merged_df['gene_id'].values, target_col_local


# =============================================================================================
#  模型训练函数
# =============================================================================================

def train_regression_model(x_train, y_train, x_test, y_test, model_type='random_forest', use_linear_svm=True):
    """训练单个回归模型"""
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_test_std = scaler.transform(x_test)

    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_features='sqrt'
        )
        model.fit(x_train_std, y_train)

    elif model_type == 'svm':
        if use_linear_svm:
            model = LinearSVR(C=1.0, random_state=42, max_iter=5000, dual='auto')
            model.fit(x_train_std, y_train)
        else:
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', max_iter=10000)
            model.fit(x_train_std, y_train)

    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(x_train_std, y_train)

    elif model_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM未安装")
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(x_train_std, y_train)

    elif model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost未安装")
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(x_train_std, y_train)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    y_pred = model.predict(x_test_std)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    spearman_corr, spearman_p = spearmanr(y_test, y_pred)
    pearson_corr, pearson_p = pearsonr(y_test, y_pred)

    feature_importance = None
    if model_type in ['random_forest', 'lightgbm', 'xgboost']:
        feature_importance = model.feature_importances_
    elif model_type == 'ridge':
        feature_importance = np.abs(model.coef_)
    elif model_type == 'svm' and hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_) if hasattr(model, 'coef_') else None

    return {
        'model_type': model_type,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': np.sqrt(mse),
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'feature_importance': feature_importance,
        'model': model,
        'scaler': scaler
    }


# =============================================================================================
#  单次训练运行（固定种子）
# =============================================================================================

def run_single_seed(data_df, target_col, args, seed):
    """使用固定种子进行单次训练"""
    print(f"\n{'=' * 50}")
    print(f"🎲 种子: {seed}")
    print(f"{'=' * 50}")

    set_seed(seed)

    # 准备特征和目标
    X = data_df[FEATURE_COLS].values
    y = data_df[target_col].values

    # 划分数据集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=seed, shuffle=True
    )
    val_ratio_adjusted = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=seed, shuffle=True
    )

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 模型类型
    model_types = ['random_forest', 'ridge']
    if LIGHTGBM_AVAILABLE:
        model_types.append('lightgbm')
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')
    if args.use_svm:
        model_types.append('svm')

    results = []
    for model_type in model_types:
        try:
            result = train_regression_model(
                X_train, y_train, X_test, y_test,
                model_type,
                use_linear_svm=args.use_linear_svm
            )

            result_record = {
                'seed': seed,
                'model_type': model_type,
                'train_size': len(y_train),
                'val_size': len(y_val),
                'test_size': len(y_test),
                'total_size': len(X),
                'test_r2': result['r2'],
                'test_pearson': result['pearson_corr'],
                'test_spearman': result['spearman_corr'],
                'test_rmse': result['rmse'],
                'test_mae': result['mae']
            }

            if model_type == 'svm':
                result_record['svm_kernel'] = 'linear' if args.use_linear_svm else 'rbf'

            results.append(result_record)

            print(f"   {model_type.upper()}: R²={result['r2']:.6f}, Pearson={result['pearson_corr']:.6f}")

        except Exception as e:
            print(f"   ❌ {model_type}失败: {str(e)}")
            continue

    return results


# =============================================================================================
#  多种子训练
# =============================================================================================

def run_multi_seed(data_df, target_col, args):
    """使用多个种子进行训练"""
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    print(f"\n{'=' * 70}")
    print(f"🚀 多种子训练 - 小龙虾")
    print(f"种子列表: {seeds}")
    print(f"{'=' * 70}")

    all_results = []
    for seed in seeds:
        results = run_single_seed(data_df, target_col, args, seed)
        all_results.extend(results)

    if not all_results:
        return None

    # 按模型类型分组
    results_df = pd.DataFrame(all_results)

    # 创建输出目录
    results_dir = 'ml_results'
    os.makedirs(results_dir, exist_ok=True)

    # 计算统计量
    model_types = results_df['model_type'].unique()

    print(f"\n{'=' * 70}")
    print(f"📊 小龙虾多种子汇总结果")
    print(f"{'=' * 70}")

    metrics_keys = ['test_r2', 'test_pearson', 'test_spearman', 'test_rmse']

    for model_type in model_types:
        model_df = results_df[results_df['model_type'] == model_type]

        if len(model_df) == 0:
            continue

        print(f"\n{model_type.upper()} (基于 {len(model_df)} 个种子):")
        print(f"{'指标':<12} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
        print(f"{'-' * 60}")

        for key in metrics_keys:
            values = model_df[key].values
            print(f"{key:<12} {np.mean(values):<12.6f} {np.std(values, ddof=1):<12.6f} "
                  f"{np.min(values):<12.6f} {np.max(values):<12.6f}")

    # 保存各种子指标到CSV
    results_file = os.path.join(results_dir, 'crayfish_ml_seed_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 各种子指标已保存: {results_file}")

    # 保存汇总统计
    summary = {
        'species': 'crayfish',
        'seeds': seeds,
        'num_seeds': len(seeds),
        'model_results': {}
    }

    for model_type in model_types:
        model_df = results_df[results_df['model_type'] == model_type]
        if len(model_df) > 0:
            summary['model_results'][model_type] = {}
            for key in metrics_keys:
                values = model_df[key].values
                summary['model_results'][model_type][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values.tolist()
                }

            # 添加SVM核信息
            if model_type == 'svm' and 'svm_kernel' in model_df.columns:
                summary['model_results'][model_type]['svm_kernel'] = model_df['svm_kernel'].iloc[0]

    with open(os.path.join(results_dir, 'crayfish_ml_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 汇总统计已保存: {results_dir}/crayfish_ml_summary.json")

    # 打印均值±标准差格式
    print(f"\n📈 小龙虾性能指标 (均值 ± 标准差):")
    for model_type in model_types:
        model_df = results_df[results_df['model_type'] == model_type]
        if len(model_df) > 0:
            print(f"\n  {model_type.upper()}:")
            print(f"    R²:       {np.mean(model_df['test_r2']):.6f} ± {np.std(model_df['test_r2'], ddof=1):.6f}")
            print(f"    Pearson:  {np.mean(model_df['test_pearson']):.6f} ± {np.std(model_df['test_pearson'], ddof=1):.6f}")
            print(f"    Spearman: {np.mean(model_df['test_spearman']):.6f} ± {np.std(model_df['test_spearman'], ddof=1):.6f}")
            print(f"    RMSE:     {np.mean(model_df['test_rmse']):.6f} ± {np.std(model_df['test_rmse'], ddof=1):.6f}")

    return results_df


# =============================================================================================
#  主函数
# =============================================================================================

def main():
    parser = argparse.ArgumentParser(description='小龙虾基因表达回归预测 - 多种子实验')
    parser.add_argument('--use_linear_svm', action='store_true',
                        help='SVM使用LinearSVR线性核（速度较快）')
    parser.add_argument('--use_rbf_svm', action='store_true',
                        help='SVM使用RBF核（精度可能更高，但训练较慢）')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='测试集比例 (默认0.15)')
    parser.add_argument('--val_size', type=float, default=0.15,
                        help='验证集比例 (默认0.15)')
    parser.add_argument('--no_svm', action='store_true',
                        help='禁用SVM模型')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'随机种子列表 (默认: {DEFAULT_SEEDS})')

    args = parser.parse_args()

    # 确定SVM配置
    use_linear_svm = True
    if args.use_rbf_svm:
        use_linear_svm = False
        print("⚠️ RBF SVM训练较慢，请耐心等待...")
    elif args.use_linear_svm:
        use_linear_svm = True

    args.use_linear_svm = use_linear_svm
    args.use_svm = not args.no_svm

    print("=" * 60)
    print("小龙虾基因表达回归预测 - 多种子实验")
    print("=" * 60)

    models_list = ["随机森林", "岭回归"]
    if LIGHTGBM_AVAILABLE:
        models_list.append("LightGBM")
    if XGBOOST_AVAILABLE:
        models_list.append("XGBoost")
    if args.use_svm:
        if use_linear_svm:
            models_list.append("LinearSVR")
        else:
            models_list.append("RBF-SVR")

    print(f"模型: {', '.join(models_list)}")
    print(f"随机种子: {args.seeds}")
    print(f"特征: {', '.join(FEATURE_COLS)}")
    print(f"数据划分: 训练 {1 - args.test_size - args.val_size:.0%} / 验证 {args.val_size:.0%} / 测试 {args.test_size:.0%}")
    print("=" * 60)

    # 加载数据（只需加载一次）
    data_df, gene_ids, target_col = load_data()

    if data_df is None:
        print("❌ 数据加载失败")
        return

    # 创建输出目录
    results_dir = 'ml_results'
    os.makedirs(results_dir, exist_ok=True)

    # 保存配置
    config = {
        'seeds': args.seeds,
        'test_size': args.test_size,
        'val_size': args.val_size,
        'use_svm': args.use_svm,
        'use_linear_svm': use_linear_svm,
        'feature_cols': FEATURE_COLS,
        'log_transform_gene_length': LOG_TRANSFORM_GENE_LENGTH,
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(results_dir, 'crayfish_experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 多种子训练
    results_df = run_multi_seed(data_df, target_col, args)

    if results_df is not None:
        print(f"\n{'=' * 60}")
        print("✅ 分析完成!")
        print(f"   结果保存在: ml_results/")
        print("=" * 60)
    else:
        print(f"\n❌ 分析失败")


if __name__ == "__main__":
    main()