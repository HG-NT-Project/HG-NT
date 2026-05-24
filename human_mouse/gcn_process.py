import pandas as pd
import numpy as np
import torch
import os
import gzip
import re
from tqdm import tqdm
import gc

# --- 路径配置（人类和小鼠）---
EXPRESSION_FILES = {
    'human': 'GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.gct.gz',
    'mouse': 'E-GEOD-70484-query-results.tpmss.tsv'
}

# 基因ID模式匹配（用于过滤有效基因）
PATTERNS = {
    'human': r'^ENSG[0-9]+',  # 人类Ensembl ID
    'mouse': r'^ENSMUSG[0-9]+'  # 小鼠Ensembl ID
}

# 人类样本分块参数
HUMAN_NUM_CHUNKS = 20  # 将人类样本分成20份

# 输出目录
OUTPUT_DIR = 'processed_gcn'

# 输出文件
OUTPUT_FILES = {
    'human': os.path.join(OUTPUT_DIR, 'human_gcn_network.pt'),
    'mouse': os.path.join(OUTPUT_DIR, 'mouse_gcn_network.pt')
}

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 使用设备: {device}")
print(f"📊 人类样本分块参数: 分成 {HUMAN_NUM_CHUNKS} 份，每份取均值")


def get_clean_gene_id(id_str, species):
    """清理和标准化基因ID，去除版本号"""
    if not isinstance(id_str, str):
        return ""
    s = id_str.strip()
    s = s.split('.')[0]
    return s


def get_gct_dimensions(file_path):
    """获取GCT文件的维度信息"""
    with gzip.open(file_path, 'rt') as f:
        version_line = f.readline().strip()
        dim_line = f.readline().strip()
        header_line = f.readline().strip()

    dim_parts = dim_line.split('\t')
    n_genes = int(dim_parts[0])
    n_samples = int(dim_parts[1])
    headers = header_line.split('\t')

    return n_genes, n_samples, headers


def split_samples_into_chunks(sample_names, num_chunks=HUMAN_NUM_CHUNKS):
    """将样本分成指定数量的块"""
    chunk_size = len(sample_names) // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        if i == num_chunks - 1:
            end = len(sample_names)
        else:
            end = (i + 1) * chunk_size
        chunks.append(sample_names[start:end])
    return chunks


def load_human_data_chunked(file_path, pattern, num_chunks=HUMAN_NUM_CHUNKS):
    """
    分块加载人类数据，每块内取均值
    返回：基因×chunks矩阵 [n_genes, num_chunks]
    """
    print(f"\n📂 分块加载人类数据...")

    # 获取维度信息
    n_genes, n_samples, headers = get_gct_dimensions(file_path)
    print(f"  GCT维度: {n_genes} 基因 × {n_samples} 样本")

    # 样本列名（跳过Name和Description）
    sample_names = headers[2:]
    print(f"  样本总数: {len(sample_names)}")

    # 将样本分成num_chunks份
    sample_chunks = split_samples_into_chunks(sample_names, num_chunks)
    for i, chunk in enumerate(sample_chunks):
        print(f"  第 {i + 1} 份: {len(chunk)} 个样本")

    # 获取所有基因ID
    gene_ids = []
    gene_to_chunk_means = {}

    # 逐行读取文件
    print(f"\n🔄 逐行处理基因，计算每份样本的均值...")

    with gzip.open(file_path, 'rt') as f:
        # 跳过前三行
        for _ in range(3):
            f.readline()

        # 处理每一行
        for line in tqdm(f, desc="处理基因"):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            gene_id = parts[0]

            # 检查是否符合模式
            if not re.match(pattern, gene_id, re.IGNORECASE):
                continue

            # 清理基因ID
            clean_id = get_clean_gene_id(gene_id, 'human')
            gene_ids.append(clean_id)

            # 获取所有样本的数值
            all_vals = np.array(parts[2:], dtype=np.float32)

            # 分成num_chunks份计算均值
            chunk_means = []
            start_idx = 0
            for chunk in sample_chunks:
                chunk_size = len(chunk)
                chunk_vals = all_vals[start_idx:start_idx + chunk_size]
                chunk_mean = np.mean(chunk_vals)
                chunk_means.append(chunk_mean)
                start_idx += chunk_size

            gene_to_chunk_means[clean_id] = chunk_means

    # 创建数据矩阵
    print(f"\n📊 创建数据矩阵...")
    data_matrix = pd.DataFrame(
        [gene_to_chunk_means[gid] for gid in gene_ids],
        index=gene_ids,
        columns=[f'chunk_{i + 1}_mean' for i in range(num_chunks)]
    )

    print(f"  数据矩阵形状: {data_matrix.shape}")
    print(f"  基因数: {len(gene_ids)}")
    print(f"  特征数 (样本块): {num_chunks}")

    # 显示均值统计
    print(f"\n📈 样本块均值统计:")
    for i in range(min(3, num_chunks)):
        chunk_vals = data_matrix.iloc[:, i].values
        print(f"  第 {i + 1} 份: 均值={np.mean(chunk_vals):.4f}, "
              f"标准差={np.std(chunk_vals):.4f}")

    return data_matrix


def load_mouse_data(file_path, pattern):
    """加载小鼠数据（样本数少，直接加载）"""
    print(f"\n📂 加载小鼠数据...")

    # 跳过注释行
    skip_rows = 0
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        while first_line.startswith('#'):
            skip_rows += 1
            first_line = f.readline().strip()

    print(f"  跳过注释行数: {skip_rows}")
    df = pd.read_csv(file_path, sep='\t', skiprows=skip_rows)

    # 获取基因ID列
    if 'Gene ID' in df.columns:
        id_col = 'Gene ID'
    elif 'GeneID' in df.columns:
        id_col = 'GeneID'
    else:
        id_col = df.columns[0]

    print(f"  使用ID列: {id_col}")

    # 过滤有效基因
    df_filtered = df[df[id_col].str.contains(pattern, na=False, regex=True)].copy()
    print(f"\n🎯 基因ID过滤:")
    print(f"  原始基因数: {len(df)}")
    print(f"  过滤后基因数: {len(df_filtered)}")

    if len(df_filtered) == 0:
        print("   ❌ 没有符合模式的基因")
        return None

    # 清理基因ID
    df_filtered['Clean_ID'] = df_filtered[id_col].apply(
        lambda x: get_clean_gene_id(x, 'mouse')
    )

    # 去除重复
    df_filtered = df_filtered.drop_duplicates(subset=['Clean_ID'])
    df_filtered = df_filtered.set_index('Clean_ID')

    # 提取数值列（跳过Gene Name列）
    if 'Gene Name' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['Gene Name'])

    # 保留所有数值列
    numeric_cols = [col for col in df_filtered.columns if col != id_col]
    data_matrix = df_filtered[numeric_cols]
    data_matrix = data_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)

    print(f"\n📊 小鼠数据统计:")
    print(f"  基因数: {len(data_matrix)}")
    print(f"  样本数: {len(numeric_cols)}")

    return data_matrix


def build_edges_without_full_matrix_gpu(data_matrix, k=10, min_corr=0.7, chunk_size=2000, device='cuda'):
    """
    使用GPU直接构建边列表，不保存完整相关系数矩阵

    Args:
        data_matrix: 基因×样本矩阵
        k: 每个基因的最大邻居数
        min_corr: 最小相关系数阈值
        chunk_size: 计算时的块大小
        device: 'cuda' 或 'cpu'

    Returns:
        edge_list: 边列表
        degree_list: 每个基因的度数
        corr_stats: 相关系数统计
    """
    print(f"\n🔄 使用GPU直接构建边列表（不保存完整矩阵）...")

    # 转换为numpy然后转到GPU
    X_np = data_matrix.values.astype(np.float32)
    X = torch.from_numpy(X_np).to(device)
    num_genes = X.shape[0]

    # 中心化
    X = X - torch.mean(X, dim=1, keepdim=True)

    # 预计算每个向量的L2范数
    norm = torch.sqrt(torch.sum(X ** 2, dim=1))

    edge_list = []
    degree_list = []
    all_correlations = []  # 用于统计

    for i in tqdm(range(num_genes), desc="处理基因"):
        X_i = X[i:i + 1]  # [1, n_samples]
        norm_i = norm[i]

        # 分批计算与所有其他基因的相关性
        corr_row = []

        for j in range(0, num_genes, chunk_size):
            j_end = min(j + chunk_size, num_genes)
            X_j = X[j:j_end]
            norm_j = norm[j:j_end]

            # 计算点积
            d = torch.mm(X_i, X_j.t())[0]  # [chunk_size]

            # 计算相关系数
            corr_chunk = torch.abs(d / (norm_i * norm_j + 1e-8))
            corr_row.append(corr_chunk.cpu().numpy())

        corr_row = np.concatenate(corr_row)
        corr_row[i] = 0  # 自己不算

        # 收集用于统计（采样）
        if i < 1000:  # 只统计前1000个基因
            all_correlations.extend(corr_row[corr_row > 0])

        # 找到满足阈值的邻居
        valid_indices = np.where(corr_row >= min_corr)[0]

        if len(valid_indices) > 0:
            actual_k = min(k, len(valid_indices))
            # 获取top-k的索引
            top_indices = valid_indices[np.argsort(corr_row[valid_indices])[-actual_k:]]

            for neighbor_idx in top_indices:
                edge_list.append([i, int(neighbor_idx)])

            degree_list.append(actual_k)
        else:
            degree_list.append(0)

        # 定期清理GPU缓存
        if i % (chunk_size * 5) == 0 and device == 'cuda':
            torch.cuda.empty_cache()

    # 计算统计信息
    if all_correlations:
        corr_stats = {
            'mean': float(np.mean(all_correlations)),
            'median': float(np.median(all_correlations)),
            'std': float(np.std(all_correlations)),
            'max': float(np.max(all_correlations))
        }
    else:
        corr_stats = {}

    return edge_list, degree_list, corr_stats


def build_gcn_network(species, file_path, k=5, min_corr=0.5, chunk_size=2000, use_gpu=True):
    """
    为特定物种构建GCN网络 - GPU内存优化版
    """
    print(f"\n" + "=" * 70)
    print(f"🔥 为 {species.upper()} 构建GCN网络")
    print(f"   使用GPU: {use_gpu}")
    print(f"=" * 70)

    if not os.path.exists(file_path):
        print(f"   ❌ 找不到文件: {file_path}")
        return None

    # 根据物种加载数据
    if species == 'human':
        data_matrix = load_human_data_chunked(file_path, PATTERNS['human'], HUMAN_NUM_CHUNKS)
        method_note = f"chunked_{HUMAN_NUM_CHUNKS}_means"
    else:
        data_matrix = load_mouse_data(file_path, PATTERNS['mouse'])
        method_note = "raw_samples"

    if data_matrix is None:
        return None

    gene_list = data_matrix.index.tolist()
    num_genes = len(gene_list)

    # 根据基因数动态调整chunk_size
    if num_genes > 50000:
        process_chunk_size = 1000
    elif num_genes > 30000:
        process_chunk_size = 2000
    else:
        process_chunk_size = 3000

    print(f"\n📊 计算参数:")
    print(f"  基因数: {num_genes}")
    print(f"  处理块大小: {process_chunk_size}")

    # 选择设备
    compute_device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'

    # 直接构建边列表，不保存相关系数矩阵
    if compute_device == 'cuda':
        edge_list, degree_list, corr_stats = build_edges_without_full_matrix_gpu(
            data_matrix,
            k=k,
            min_corr=min_corr,
            chunk_size=process_chunk_size,
            device=compute_device
        )
    else:
        # 回退到CPU版本
        edge_list, degree_list, corr_stats = build_edges_without_full_matrix_cpu(
            data_matrix,
            k=k,
            min_corr=min_corr,
            chunk_size=process_chunk_size
        )

    # 统计相关系数分布
    if corr_stats:
        print(f"\n📈 Pearson相关系数统计（采样）:")
        print(f"  均值: {corr_stats['mean']:.4f}")
        print(f"  中位数: {corr_stats['median']:.4f}")
        print(f"  标准差: {corr_stats['std']:.4f}")
        print(f"  最大: {corr_stats['max']:.4f}")

    # 转换为边索引
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor([], dtype=torch.long)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存结果
    save_path = OUTPUT_FILES[species]
    torch.save({
        'edge_index': edge_index,
        'gene_list': gene_list,
        'num_genes': num_genes,
        'num_edges': edge_index.shape[1] if edge_index.numel() > 0 else 0,
        'k': k,
        'min_corr': min_corr,
        'species': species,
        'method': method_note,
        'num_chunks': HUMAN_NUM_CHUNKS if species == 'human' else None,
        'degree_distribution': degree_list,
        'correlation_stats': corr_stats,
        'used_gpu': compute_device == 'cuda',
        'timestamp': pd.Timestamp.now().isoformat()
    }, save_path)

    # 输出统计信息
    print(f"\n🚀 成功保存: {save_path}")
    print(f"   - 最终节点数: {num_genes}")
    print(f"   - 最终边数: {edge_index.shape[1]}")
    print(f"   - 平均度: {edge_index.shape[1] / num_genes:.2f}")
    print(f"   - 孤立节点数: {degree_list.count(0)} ({degree_list.count(0) / num_genes * 100:.2f}%)")

    # 清理内存
    if compute_device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return edge_index, gene_list


def build_edges_without_full_matrix_cpu(data_matrix, k=10, min_corr=0.7, chunk_size=1000):
    """
    CPU版本：直接构建边列表，不保存完整相关系数矩阵
    """
    print(f"\n🔄 使用CPU直接构建边列表（不保存完整矩阵）...")

    X = data_matrix.values.astype(np.float32)
    num_genes = X.shape[0]

    # 中心化
    X = X - np.mean(X, axis=1, keepdims=True)

    # 预计算每个向量的L2范数
    norm = np.sqrt(np.sum(X ** 2, axis=1))

    edge_list = []
    degree_list = []
    all_correlations = []  # 用于统计

    for i in tqdm(range(num_genes), desc="处理基因"):
        X_i = X[i:i + 1]  # [1, n_samples]
        norm_i = norm[i]

        # 分批计算与所有其他基因的相关性
        corr_row = []

        for j in range(0, num_genes, chunk_size):
            j_end = min(j + chunk_size, num_genes)
            X_j = X[j:j_end]
            norm_j = norm[j:j_end]

            # 计算点积
            d = np.dot(X_i, X_j.T)[0]  # [chunk_size]

            # 计算相关系数
            corr_chunk = np.abs(d / (norm_i * norm_j + 1e-8))
            corr_row.extend(corr_chunk)

        corr_row = np.array(corr_row)
        corr_row[i] = 0  # 自己不算

        # 收集用于统计（采样）
        if i < 1000:  # 只统计前1000个基因
            all_correlations.extend(corr_row[corr_row > 0])

        # 找到满足阈值的邻居
        valid_indices = np.where(corr_row >= min_corr)[0]

        if len(valid_indices) > 0:
            actual_k = min(k, len(valid_indices))
            # 获取top-k的索引
            top_indices = valid_indices[np.argsort(corr_row[valid_indices])[-actual_k:]]

            for neighbor_idx in top_indices:
                edge_list.append([i, int(neighbor_idx)])

            degree_list.append(actual_k)
        else:
            degree_list.append(0)

        # 定期清理
        if i % (chunk_size * 2) == 0:
            gc.collect()

    # 计算统计信息
    if all_correlations:
        corr_stats = {
            'mean': float(np.mean(all_correlations)),
            'median': float(np.median(all_correlations)),
            'std': float(np.std(all_correlations)),
            'max': float(np.max(all_correlations))
        }
    else:
        corr_stats = {}

    return edge_list, degree_list, corr_stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description='为人类和小鼠构建GCN网络')
    parser.add_argument('--k', type=int, default=10,
                        help='每个基因的最大邻居数 (默认: 10)')
    parser.add_argument('--min_corr', type=float, default=0.7,
                        help='最小相关系数阈值 (默认: 0.7)')
    parser.add_argument('--species', type=str, choices=['human', 'mouse', 'all'],
                        default='all', help='要处理的物种')
    parser.add_argument('--human_chunks', type=int, default=20,
                        help='人类样本分成的份数 (默认: 20)')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='计算相关系数时的块大小 (默认: 1000)')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU (默认自动选择GPU)')

    args = parser.parse_args()

    # 更新全局变量
    global HUMAN_NUM_CHUNKS
    HUMAN_NUM_CHUNKS = args.human_chunks

    print("=" * 70)
    print("🧬 GCN网络构建工具 (人类/小鼠) - GPU加速版")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📁 输出目录: {OUTPUT_DIR}")

    use_gpu = not args.cpu and torch.cuda.is_available()
    print(f"\n📐 网络构建参数:")
    print(f"  K (最大邻居数): {args.k}")
    print(f"  Min Correlation: {args.min_corr}")
    print(f"  人类样本分块: {args.human_chunks} 份")
    print(f"  处理块大小: {args.chunk_size}")
    print(f"  使用GPU: {use_gpu}")

    # 处理物种
    species_list = ['human', 'mouse'] if args.species == 'all' else [args.species]

    for species in species_list:
        build_gcn_network(
            species,
            EXPRESSION_FILES[species],
            k=args.k,
            min_corr=args.min_corr,
            chunk_size=args.chunk_size,
            use_gpu=use_gpu
        )

        # 强制垃圾回收
        if use_gpu:
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n" + "=" * 70)
    print("✅ GCN网络构建完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()