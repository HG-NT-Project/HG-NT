import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from scipy.stats import skew

# --- 配置区 ---
CONFIG = {
    'expression_file': 'gene.tpm.matrix.annot.xls',
    'index_file': 'processed_tf/gene_id_index.txt',
    'output_dir': 'processed_gcn',
    'num_train_samples': 300,
    'num_chunks': 30,
    'k': 10,
    'min_corr': 0.7
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clean_id(s):
    """极致 ID 清洗"""
    return str(s).replace('gene-', '').replace('rna-', '').split('.')[0].strip()


def build_gcn_with_nan_support():
    print(f"🔧 运行设备: {device}")

    # 1. 建立 46476 个锚点的基准 DataFrame
    with open(CONFIG['index_file'], 'r') as f:
        original_indices = [line.strip() for line in f]

    # 创建基准表，增加一个 clean_id 列用于匹配
    master_df = pd.DataFrame({
        'original_id': original_indices,
        'clean_id': [clean_id(idx) for idx in original_indices]
    })
    print(f"📍 锚点基因总数: {len(master_df)}")

    # 2. 解析表达矩阵
    print("📖 正在解析表达矩阵...")
    header_df = pd.read_csv(CONFIG['expression_file'], sep='\t', nrows=0)
    all_cols = header_df.columns.tolist()
    target_cols = [all_cols[0]] + all_cols[4:415]

    expr_df = pd.read_csv(
        CONFIG['expression_file'], sep='\t', usecols=target_cols,
        index_col=0, low_memory=False
    )

    # 3. 清洗矩阵 ID 并合并重复项
    expr_df.index = expr_df.index.map(clean_id)
    expr_df = expr_df.apply(pd.to_numeric, errors='coerce')
    expr_df = expr_df.groupby(expr_df.index).mean()
    expr_df.index.name = 'clean_id'

    # 4. 【核心改进】使用 Merge 进行全量对齐，确保长度永远等于 46476
    # left join 保证了以 master_df (锚点) 为准
    full_df = pd.merge(master_df, expr_df, on='clean_id', how='left')

    # 将原始 ID 重新设为索引，并删掉辅助列
    full_df.index = full_df['original_id']
    full_df = full_df.drop(columns=['original_id', 'clean_id'])

    # 统计缺失情况
    missing_count = full_df.isna().all(axis=1).sum()
    print(f"✅ 锚点对齐完成。缺失数据基因数: {missing_count} / {len(master_df)}")

    # 5. 样本抽样 (300 构图, 111 标签)
    all_samples = expr_df.columns.tolist()
    np.random.seed(42)
    num_train = min(len(all_samples), CONFIG['num_train_samples'])
    train_samples = np.random.choice(all_samples, num_train, replace=False)
    test_samples = [s for s in all_samples if s not in train_samples]

    # 6. 生成节点特征块
    chunk_size = max(1, len(train_samples) // CONFIG['num_chunks'])
    chunked_list = []
    for i in range(CONFIG['num_chunks']):
        group = train_samples[i * chunk_size: (i + 1) * chunk_size]
        if len(group) == 0:
            continue
        chunked_list.append(full_df[group].mean(axis=1, skipna=True))
    if not chunked_list:
        raise ValueError("没有可用于构图的训练样本，请检查表达矩阵列数。")
    node_features_df = pd.concat(chunked_list, axis=1)

    # 7. GPU 相关性构图
    edge_index = build_edges_gpu(node_features_df.fillna(0.0))

    # 8. 保存 GCN 网络文件
    torch.save({
        'edge_index': edge_index,
        'gene_list': original_indices,
        'node_features': torch.from_numpy(node_features_df.fillna(0.0).values)
    }, os.path.join(CONFIG['output_dir'], 'crayfish_gcn_network.pt'))

    # 9. 生成标签文件
    print("🧪 正在生成标签文件...")
    raw_means = full_df[test_samples].mean(axis=1, skipna=True)
    log_means = np.log1p(raw_means)

    # 此时 log_means 的索引就是 original_id，且顺序完全一致
    label_output = pd.DataFrame({
        'GeneID': original_indices,
        'label': log_means.loc[original_indices].values
    })
    label_output.to_csv("crayfish_labels.csv", index=False, na_rep='NaN')

    print("\n" + "=" * 60)
    print("📊 小龙虾 GCN 全量对齐报告 (V4 稳健版)")
    print("=" * 60)
    num_edges = edge_index.shape[1] if edge_index.dim() > 1 else 0
    print(f"1. 节点对齐: {len(original_indices)} (全量保留)")
    print(f"2. 真正缺失基因: {missing_count}")
    print(f"3. 共表达边数: {num_edges:,}")
    print(f"4. 顺序校验: {'一致' if list(label_output['GeneID']) == original_indices else '异常'}")
    print("=" * 60)


def build_edges_gpu(data_matrix):
    X = torch.from_numpy(data_matrix.values).to(device)
    X = X - torch.mean(X, dim=1, keepdim=True)
    norm = torch.sqrt(torch.sum(X ** 2, dim=1))
    edge_list = []
    for i in tqdm(range(X.shape[0]), desc="构建 GCN 边"):
        if norm[i] < 1e-8: continue
        X_i = X[i:i + 1]
        corrs = torch.abs(torch.mm(X_i, X.t())[0] / (norm[i] * norm + 1e-8))
        corrs[i] = 0
        valid_idx = torch.where(corrs >= CONFIG['min_corr'])[0]
        if len(valid_idx) > 0:
            k_act = min(CONFIG['k'], len(valid_idx))
            _, top_idx = torch.topk(corrs[valid_idx], k_act)
            for n_idx in valid_idx[top_idx]:
                edge_list.append([i, int(n_idx)])
        if i % 10000 == 0: torch.cuda.empty_cache()
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


if __name__ == "__main__":
    build_gcn_with_nan_support()