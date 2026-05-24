"""
三种机器学习回归模型完整实现
支持多种子重复实验，输出性能指标的均值和标准差
"""

import pandas as pd
import numpy as np
import torch
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import warnings
import os
import glob
from sklearn.model_selection import train_test_split

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

# 5个随机种子
SEEDS = [42, 123, 456, 789, 1024]

# =============================================================================================
#  标签文件配置
# =============================================================================================

LABEL_CONFIG = {
    'human': {
        'label_file': 'processed_labels/human_labels.pt',
        'species_name': 'human'
    },
    'mouse': {
        'label_file': 'processed_labels/mouse_labels.pt',
        'species_name': 'mouse'
    }
}


# =============================================================================================
#  核染色体判断函数
# =============================================================================================

def is_nuclear_chromosome(chromosome, species):
    chrom_str = str(chromosome).strip()
    if chrom_str.lower().startswith('chr'):
        chrom_str = chrom_str[3:]

    if species == 'human':
        try:
            chrom_num = int(chrom_str)
            return 1 <= chrom_num <= 22
        except:
            return chrom_str.upper() in ['X', 'Y']
    elif species == 'mouse':
        try:
            chrom_num = int(chrom_str)
            return 1 <= chrom_num <= 19
        except:
            return chrom_str.upper() in ['X', 'Y']
    else:
        try:
            chrom_num = int(chrom_str)
            return 1 <= chrom_num <= 22
        except:
            return chrom_str.upper() in ['X', 'Y']


# =============================================================================================
#  加载标签数据
# =============================================================================================

def load_label_data(species):
    config = LABEL_CONFIG[species]
    label_file = config['label_file']

    print(f"\n📂 加载标签文件: {label_file}")

    if not os.path.exists(label_file):
        print(f"❌ 标签文件不存在: {label_file}")
        return None, None

    data = torch.load(label_file, map_location='cpu')
    gene_ids = data['gene_id']
    labels = data['labels']

    print(f"✅ 加载成功: {len(gene_ids)} 个基因")

    label_df = pd.DataFrame({
        'gene_id': gene_ids,
        'log2_mean_expression': labels.numpy().flatten()
    })

    return label_df, gene_ids


# =============================================================================================
#  特征文件查找
# =============================================================================================

def find_feature_files():
    feature_dir = 'generated_features'
    if not os.path.exists(feature_dir):
        print(f"❌ 特征目录不存在: {feature_dir}")
        return {}

    feature_files = glob.glob(os.path.join(feature_dir, "*_features.csv"))
    files_by_species = {}

    for file_path in feature_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) >= 2:
            species = parts[0]
            if species not in files_by_species:
                files_by_species[species] = []
            files_by_species[species].append(file_path)

    print(f"\n📁 找到的特征文件:")
    for species, files in files_by_species.items():
        print(f"  {species}: {len(files)} 个文件")

    return files_by_species


# =============================================================================================
#  模型训练函数
# =============================================================================================

def train_regression_model(x_train, y_train, x_test, y_test, model_type='random_forest', use_linear_svm=False):
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_test_std = scaler.transform(x_test)

    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_features='sqrt')
    elif model_type == 'svm':
        if use_linear_svm:
            model = SVR(kernel='linear', C=1.0, epsilon=0.1)
        else:
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model.fit(x_train_std, y_train)
    y_pred = model.predict(x_test_std)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    spearman_corr, spearman_p = spearmanr(y_test, y_pred)
    pearson_corr, pearson_p = pearsonr(y_test, y_pred)

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
    }


# =============================================================================================
#  运行回归分析
# =============================================================================================

def run_regression_analysis(use_linear_svm=True, target_species=None, test_size=0.2, val_size=0.2, seeds=SEEDS):
    feature_files_by_species = find_feature_files()

    if not feature_files_by_species:
        print("❌ 未找到任何特征文件")
        return None

    model_types = ['random_forest', 'svm', 'ridge']
    if LIGHTGBM_AVAILABLE:
        model_types.append('lightgbm')
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')

    results_dir = 'ml_results'
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n📁 结果将保存在: {results_dir}/")
    print(f"📊 数据划分: 训练 {1 - test_size - val_size:.0%} / 验证 {val_size:.0%} / 测试 {test_size:.0%}")
    print(f"🎲 随机种子: {seeds}")

    species_list = [target_species] if target_species else list(feature_files_by_species.keys())
    all_results = []

    for species in species_list:
        if species not in feature_files_by_species or species not in LABEL_CONFIG:
            print(f"\n⚠️ 跳过 {species}")
            continue

        print(f"\n{'=' * 60}")
        print(f"处理物种: {species.upper()}")
        print(f"{'=' * 60}")

        label_df, _ = load_label_data(species)
        if label_df is None:
            continue

        for feature_file in feature_files_by_species[species]:
            print(f"\n📊 特征文件: {os.path.basename(feature_file)}")

            try:
                predictors = pd.read_csv(feature_file)

                required_cols = ['gene_id', 'Chromosome', 'Strand']
                if not all(col in predictors.columns for col in required_cols):
                    continue

                data = predictors.merge(label_df, on='gene_id', how='inner')
                if data.empty:
                    continue

                # 过滤核染色体
                data = data[data.apply(lambda row: is_nuclear_chromosome(row['Chromosome'], species), axis=1)].copy()

                # 删除不需要的列
                drop_cols = ['gene_id', 'Chromosome', 'log2_mean_expression', 'Strand']
                extra_drop = ['Start', 'End', 'gene_id_base']
                for col in extra_drop:
                    if col in data.columns:
                        drop_cols.append(col)

                feature_cols = [col for col in data.columns if col not in drop_cols]
                X = data[feature_cols].values
                y = data['log2_mean_expression'].values

                print(f"  特征数: {len(feature_cols)}, 样本数: {len(X)}")

                # 对每个种子进行训练
                for seed in seeds:
                    print(f"\n  🎲 种子: {seed}")

                    np.random.seed(seed)

                    # 划分数据集
                    X_train_val, X_test, y_train_val, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=seed
                    )
                    val_ratio_adjusted = val_size / (1 - test_size)
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=seed
                    )

                    print(f"    训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")

                    for model_type in model_types:
                        try:
                            if model_type == 'svm':
                                result = train_regression_model(
                                    X_train, y_train, X_test, y_test, model_type, use_linear_svm
                                )
                            else:
                                result = train_regression_model(
                                    X_train, y_train, X_test, y_test, model_type
                                )

                            all_results.append({
                                'species': species,
                                'feature_file': os.path.basename(feature_file),
                                'seed': seed,
                                'model_type': model_type,
                                'r2': result['r2'],
                                'rmse': result['rmse'],
                                'mae': result['mae'],
                                'pearson_corr': result['pearson_corr'],
                                'pearson_p': result['pearson_p'],
                                'spearman_corr': result['spearman_corr'],
                                'spearman_p': result['spearman_p'],
                                'train_size': len(X_train),
                                'val_size': len(X_val),
                                'test_size': len(X_test)
                            })

                            print(f"      {model_type}: R²={result['r2']:.4f}, Pearson={result['pearson_corr']:.4f}")

                        except Exception as e:
                            print(f"      ❌ {model_type} 失败: {e}")

            except Exception as e:
                print(f"❌ 处理失败: {e}")

    # 保存结果
    if all_results:
        results_df = pd.DataFrame(all_results)

        for species in results_df['species'].unique():
            species_df = results_df[results_df['species'] == species]
            output_file = os.path.join(results_dir, f"{species}_ml_results_seeds.csv")
            species_df.to_csv(output_file, index=False)
            print(f"\n✅ 保存 {species} 结果到: {output_file}")

        # 打印统计摘要
        print_summary_with_stats(results_df)

        return results_df
    else:
        print("❌ 未生成任何结果")
        return None


# =============================================================================================
#  打印统计摘要
# =============================================================================================

def print_summary_with_stats(results_df):
    print(f"\n{'=' * 80}")
    print("机器学习回归模型性能摘要（多种子重复实验）")
    print(f"{'=' * 80}")

    metrics = ['r2', 'pearson_corr', 'spearman_corr', 'rmse']

    for species in results_df['species'].unique():
        species_df = results_df[results_df['species'] == species]
        print(f"\n物种: {species.upper()}")
        print("-" * 50)

        for model_type in species_df['model_type'].unique():
            model_df = species_df[species_df['model_type'] == model_type]

            if len(model_df) > 0:
                print(f"\n  {model_type.upper()} (基于 {len(model_df)} 个种子):")
                for metric in metrics:
                    if metric in model_df.columns:
                        values = model_df[metric].values
                        mean_val = np.mean(values)
                        std_val = np.std(values, ddof=1)
                        print(f"    {metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")


# =============================================================================================
#  主函数
# =============================================================================================

def main():
    parser = argparse.ArgumentParser(description='机器学习回归分析 - 多种子重复实验')
    parser.add_argument('--use_linear_svm', action='store_true', help='SVM使用linear核')
    parser.add_argument('--use_rbf_svm', action='store_true', help='SVM使用rbf核')
    parser.add_argument('--species', type=str, default=None, choices=['human', 'mouse'], help='指定物种')
    parser.add_argument('--test_size', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS, help='随机种子列表')

    args = parser.parse_args()

    use_linear_svm = not args.use_rbf_svm

    print("=" * 60)
    print("机器学习回归分析 - 多种子重复实验")
    print("=" * 60)
    print(f"随机种子: {args.seeds}")
    print(f"数据划分: 训练 {1 - args.test_size - args.val_size:.0%} / 验证 {args.val_size:.0%} / 测试 {args.test_size:.0%}")
    print("=" * 60)

    results_df = run_regression_analysis(
        use_linear_svm, args.species, args.test_size, args.val_size, args.seeds
    )

    if results_df is not None:
        print(f"\n✅ 分析完成! 结果保存在 ml_results/")


if __name__ == "__main__":
    main()