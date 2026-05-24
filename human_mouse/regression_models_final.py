"""
三种机器学习回归模型完整实现（保留SVM）
添加LightGBM回归模型和XGBoost回归模型
只处理核染色体，优化训练效率
适配人类和小鼠数据（从processed_labels直接读取）
改为单次训练/验证/测试集划分
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
from tqdm import tqdm
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
pd.options.display.width = 0

# =============================================================================================
#  标签文件配置（直接使用processed_labels目录）
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
    """
    判断是否为核染色体（人类和小鼠）
    """
    chrom_str = str(chromosome).strip()

    # 去除可能的"chr"前缀
    if chrom_str.lower().startswith('chr'):
        chrom_str = chrom_str[3:]

    # 人类染色体：1-22, X, Y
    if species == 'human':
        # 数字染色体1-22
        try:
            chrom_num = int(chrom_str)
            return 1 <= chrom_num <= 22
        except:
            # 性染色体
            return chrom_str.upper() in ['X', 'Y']

    # 小鼠染色体：1-19, X, Y
    elif species == 'mouse':
        try:
            chrom_num = int(chrom_str)
            return 1 <= chrom_num <= 19
        except:
            return chrom_str.upper() in ['X', 'Y']

    # 默认情况
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
    """
    加载处理好的标签文件（从processed_labels目录）
    """
    config = LABEL_CONFIG[species]
    label_file = config['label_file']

    print(f"\n📂 加载标签文件: {label_file}")

    if not os.path.exists(label_file):
        print(f"❌ 标签文件不存在: {label_file}")
        return None, None

    # 加载PyTorch文件
    data = torch.load(label_file, map_location='cpu')

    # 从保存的字典中提取数据
    gene_ids = data['gene_id']
    labels = data['labels']
    columns = data.get('columns', ['mean_expression_log2'])

    print(f"✅ 加载成功:")
    print(f"  基因数: {len(gene_ids)}")
    print(f"  标签形状: {labels.shape}")
    print(f"  列名: {columns}")

    # 创建DataFrame（标签是 [N, 1] 形状）
    label_df = pd.DataFrame({
        'gene_id': gene_ids,
        'log2_mean_expression': labels.numpy().flatten()
    })

    return label_df, gene_ids


# =============================================================================================
#  特征文件查找
# =============================================================================================

def find_feature_files():
    """
    在generated_features目录下查找特征文件
    """
    feature_dir = 'generated_features'
    if not os.path.exists(feature_dir):
        print(f"❌ 特征目录不存在: {feature_dir}")
        return {}

    # 查找所有特征文件
    feature_files = glob.glob(os.path.join(feature_dir, "*_features.csv"))

    # 按物种分类
    files_by_species = {}

    for file_path in feature_files:
        filename = os.path.basename(file_path)

        # 解析文件名格式: {species}_{expr_name}_features.csv
        parts = filename.split('_')
        if len(parts) >= 2:
            species = parts[0]  # human 或 mouse

            if species not in files_by_species:
                files_by_species[species] = []

            files_by_species[species].append(file_path)

    print(f"\n📁 找到的特征文件:")
    for species, files in files_by_species.items():
        print(f"  {species}:")
        for f in files:
            print(f"    - {os.path.basename(f)}")

    return files_by_species


# =============================================================================================
#  模型训练函数
# =============================================================================================

def train_regression_model(x_train, y_train, x_test, y_test, model_type='random_forest', use_linear_svm=False):
    """
    训练单个回归模型
    """
    # 标准化特征
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_test_std = scaler.transform(x_test)

    # 选择模型
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_features='sqrt'
        )
    elif model_type == 'svm':
        if use_linear_svm:
            model = SVR(kernel='linear', C=1.0, epsilon=0.1)
        else:
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
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
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 训练模型
    model.fit(x_train_std, y_train)

    # 预测
    y_pred = model.predict(x_test_std)

    # 计算性能指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    spearman_corr, spearman_p = spearmanr(y_test, y_pred)
    pearson_corr, pearson_p = pearsonr(y_test, y_pred)

    # 特征重要性
    feature_importance = None
    if model_type in ['random_forest', 'lightgbm', 'xgboost']:
        feature_importance = model.feature_importances_

    return {
        'model_type': model_type,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'spearman_significant': spearman_p < 0.05,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'pearson_significant': pearson_p < 0.05,
        'feature_importance': feature_importance,
        'model': model,
        'scaler': scaler
    }


# =============================================================================================
#  运行回归分析（改为单次划分）
# =============================================================================================

def run_regression_analysis(use_linear_svm=True, target_species=None, test_size=0.2, val_size=0.2):
    """
    运行回归分析，使用单次训练/验证/测试集划分

    Parameters:
    -----------
    test_size: 测试集比例 (默认0.2)
    val_size: 验证集比例 (默认0.2，从训练集中划分)
    """
    # 查找特征文件
    feature_files_by_species = find_feature_files()

    if not feature_files_by_species:
        print("❌ 未找到任何特征文件")
        return None, None

    # 模型类型
    model_types = ['random_forest', 'svm', 'ridge']
    if LIGHTGBM_AVAILABLE:
        model_types.append('lightgbm')
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')

    # 存储所有结果
    all_results = []
    feature_importance_data = []

    # 创建结果目录
    results_dir = 'ml_results'
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n📁 结果将保存在: {results_dir}/")
    print(f"\n📊 数据划分: 训练 {1 - test_size - val_size:.0%} / 验证 {val_size:.0%} / 测试 {test_size:.0%}")

    # 确定要处理的物种
    if target_species:
        species_list = [target_species]
    else:
        species_list = list(feature_files_by_species.keys())

    # 遍历每个物种
    for species in species_list:
        if species not in feature_files_by_species:
            print(f"\n⚠️ 物种 {species} 没有特征文件，跳过")
            continue

        if species not in LABEL_CONFIG:
            print(f"\n⚠️ 物种 {species} 没有对应的标签文件配置，跳过")
            continue

        print(f"\n{'=' * 60}")
        print(f"处理物种: {species.upper()}")
        print(f"{'=' * 60}")

        # 加载标签数据
        label_df, gene_ids = load_label_data(species)
        if label_df is None:
            continue

        feature_files = feature_files_by_species[species]

        # 处理每个特征文件
        for feature_file in feature_files:
            print(f"\n📊 使用特征文件: {os.path.basename(feature_file)}")

            try:
                # 加载特征数据
                predictors = pd.read_csv(feature_file)
                print(f"  特征数据形状: {predictors.shape}")

                # 检查必要的列
                required_cols = ['gene_id', 'Chromosome', 'Strand']
                missing_cols = [col for col in required_cols if col not in predictors.columns]
                if missing_cols:
                    print(f"❌ 缺少必要列: {missing_cols}")
                    continue

                # 合并特征和标签
                data = predictors.merge(label_df, on='gene_id', how='inner')

                if data.empty:
                    print(f"⚠️ 数据合并后为空，跳过")
                    continue

                print(f"  合并后数据大小: {data.shape}")
                print(f"  匹配率: {len(data)}/{len(predictors)} ({len(data) / len(predictors) * 100:.1f}%)")

                # 过滤核染色体
                original_size = len(data)
                data = data[data.apply(lambda row: is_nuclear_chromosome(row['Chromosome'], species), axis=1)].copy()
                filtered_size = len(data)

                print(f"  染色体过滤: {original_size} → {filtered_size}")

                if data.empty:
                    print(f"⚠️ 过滤后数据为空，跳过")
                    continue

                print(f"\n📈 目标值统计:")
                print(f"  范围: [{data['log2_mean_expression'].min():.4f}, {data['log2_mean_expression'].max():.4f}]")
                print(f"  均值: {data['log2_mean_expression'].mean():.4f}")
                print(f"  中位数: {data['log2_mean_expression'].median():.4f}")

                # ===== 改为单次划分 =====
                # 删除不需要的列
                drop_cols = ['gene_id', 'Chromosome', 'log2_mean_expression']
                if 'Strand' in data.columns:
                    drop_cols.append('Strand')

                # 额外的可能列
                extra_drop = ['Start', 'End', 'gene_id_base']
                for col in extra_drop:
                    if col in data.columns:
                        drop_cols.append(col)

                # 分离特征和目标
                feature_cols = [col for col in data.columns if col not in drop_cols]
                X = data[feature_cols].values
                y = data['log2_mean_expression'].values

                print(f"\n🔀 划分数据集:")
                print(f"  特征数量: {len(feature_cols)}")

                # 先划分训练+验证 和 测试
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                # 再从训练+验证中划分训练和验证
                val_ratio_adjusted = val_size / (1 - test_size)  # 调整验证集比例
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=42
                )

                print(f"  训练集: {len(X_train)} 个样本 ({len(X_train) / len(X) * 100:.1f}%)")
                print(f"  验证集: {len(X_val)} 个样本 ({len(X_val) / len(X) * 100:.1f}%)")
                print(f"  测试集: {len(X_test)} 个样本 ({len(X_test) / len(X) * 100:.1f}%)")

                # 对每种模型进行训练和评估
                for model_type in model_types:
                    print(f"\n    训练 {model_type.upper()}...")

                    try:
                        # 在训练集上训练
                        if model_type == 'svm':
                            result = train_regression_model(
                                X_train, y_train, X_test, y_test,
                                model_type, use_linear_svm=use_linear_svm
                            )
                        else:
                            result = train_regression_model(
                                X_train, y_train, X_test, y_test, model_type
                            )

                        # 保存结果
                        result_record = {
                            'species': species,
                            'feature_file': os.path.basename(feature_file),
                            'model_type': model_type,
                            'mse': result['mse'],
                            'mae': result['mae'],
                            'r2': result['r2'],
                            'spearman_corr': result['spearman_corr'],
                            'spearman_p': result['spearman_p'],
                            'spearman_significant': result['spearman_significant'],
                            'pearson_corr': result['pearson_corr'],
                            'pearson_p': result['pearson_p'],
                            'pearson_significant': result['pearson_significant'],
                            'train_size': len(y_train),
                            'val_size': len(y_val),
                            'test_size': len(y_test),
                            'total_size': len(X)
                        }
                        all_results.append(result_record)

                        # 保存特征重要性
                        if result['feature_importance'] is not None:
                            for feat_name, importance in zip(feature_cols, result['feature_importance']):
                                feature_importance_data.append({
                                    'species': species,
                                    'feature_file': os.path.basename(feature_file),
                                    'feature': feat_name,
                                    'importance': float(importance),
                                    'model': model_type
                                })

                        # 打印当前结果
                        print(f"      R²: {result['r2']:.4f}")
                        print(f"      MSE: {result['mse']:.4f}")
                        print(f"      MAE: {result['mae']:.4f}")
                        print(f"      Spearman: {result['spearman_corr']:.4f} (p={result['spearman_p']:.4e})")
                        print(f"      Pearson: {result['pearson_corr']:.4f} (p={result['pearson_p']:.4e})")

                    except Exception as e:
                        print(f"      ❌ 训练{model_type}失败: {str(e)}")
                        continue

            except Exception as e:
                print(f"❌ 处理文件 {feature_file} 时出错: {str(e)}")
                continue

    # 保存所有结果
    if all_results:
        results_df = pd.DataFrame(all_results)

        # 按物种分别保存
        for species in results_df['species'].unique():
            species_df = results_df[results_df['species'] == species]
            output_file = os.path.join(results_dir, f"{species}_ml_results_single.csv")
            species_df.to_csv(output_file, index=False)
            print(f"\n✅ 保存 {species} 结果到: {output_file}")

        # 保存汇总结果
        summary_file = os.path.join(results_dir, "all_ml_results_single.csv")
        results_df.to_csv(summary_file, index=False)
        print(f"✅ 保存所有结果到: {summary_file}")

        # 保存特征重要性
        if feature_importance_data:
            importance_df = pd.DataFrame(feature_importance_data)
            imp_file = os.path.join(results_dir, "all_feature_importance_single.csv")
            importance_df.to_csv(imp_file, index=False)
            print(f"✅ 保存特征重要性到: {imp_file}")

        # 打印摘要
        print_summary(results_df)

        return results_df, importance_df if feature_importance_data else None

    else:
        print("❌ 未生成任何结果")
        return None, None


# =============================================================================================
#  打印摘要
# =============================================================================================

def print_summary(results_df):
    """
    打印性能摘要
    """
    print(f"\n{'=' * 80}")
    print("机器学习回归模型性能摘要（单次划分）")
    print(f"{'=' * 80}")

    for species in results_df['species'].unique():
        species_df = results_df[results_df['species'] == species]
        print(f"\n物种: {species.upper()}")
        print("-" * 40)

        for model_type in species_df['model_type'].unique():
            model_df = species_df[species_df['model_type'] == model_type]

            if len(model_df) > 0:
                print(f"\n  {model_type.upper()}:")
                print(f"    测试集大小: {model_df['test_size'].iloc[0]}")
                print(f"    R²: {model_df['r2'].iloc[0]:.4f}")
                print(f"    MSE: {model_df['mse'].iloc[0]:.4f}")
                print(f"    MAE: {model_df['mae'].iloc[0]:.4f}")
                print(f"    Spearman: {model_df['spearman_corr'].iloc[0]:.4f} (p={model_df['spearman_p'].iloc[0]:.4e})")
                print(f"    Pearson: {model_df['pearson_corr'].iloc[0]:.4f} (p={model_df['pearson_p'].iloc[0]:.4e})")

    print(f"\n{'=' * 80}")


# =============================================================================================
#  主函数
# =============================================================================================

def main():
    parser = argparse.ArgumentParser(description='使用processed_labels目录下的标签进行回归分析（单次划分）')
    parser.add_argument('--use_linear_svm', action='store_true',
                        help='SVM使用linear核（速度更快）')
    parser.add_argument('--use_rbf_svm', action='store_true',
                        help='SVM使用rbf核（精度可能更高）')
    parser.add_argument('--species', type=str, default=None,
                        choices=['human', 'mouse'],
                        help='指定处理的物种（不指定则处理所有）')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例 (默认0.2)')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='验证集比例 (默认0.2)')

    args = parser.parse_args()

    # 确定SVM核函数
    use_linear_svm = True
    if args.use_rbf_svm:
        use_linear_svm = False

    print("=" * 60)
    print("机器学习回归分析（使用processed_labels目录）- 单次划分")
    print("=" * 60)
    models_list = ["随机森林", "SVM", "岭回归"]
    if LIGHTGBM_AVAILABLE:
        models_list.append("LightGBM")
    if XGBOOST_AVAILABLE:
        models_list.append("XGBoost")
    print(f"模型: {', '.join(models_list)}")
    print(f"SVM核函数: {'linear' if use_linear_svm else 'rbf'}")
    print(f"目标物种: {args.species if args.species else '人类 + 小鼠'}")
    print(
        f"数据划分: 训练 {1 - args.test_size - args.val_size:.0%} / 验证 {args.val_size:.0%} / 测试 {args.test_size:.0%}")
    print("标签来源: processed_labels/ 目录")
    print("特征来源: generated_features/ 目录")
    print("=" * 60)

    results_df, _ = run_regression_analysis(
        use_linear_svm,
        args.species,
        test_size=args.test_size,
        val_size=args.val_size
    )

    if results_df is not None:
        print(f"\n{'=' * 60}")
        print("✅ 分析完成!")
        print("=" * 60)
    else:
        print(f"\n❌ 分析失败")


if __name__ == "__main__":
    main()