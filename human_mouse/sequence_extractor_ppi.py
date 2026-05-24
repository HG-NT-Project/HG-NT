# -*- coding: utf-8 -*-
"""
序列提取工具 - 提取表达基因及其最高PPI系数的邻居基因序列
生成6kb序列：基因序列(3kb) + 邻居基因序列(3kb)
找不到邻居基因时复制自身序列
用法：python sequence_extractor_ppi.py --species human
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime
import json
import warnings
import sys
import gzip
import re
from collections import defaultdict

# 尝试导入pyfaidx
try:
    from pyfaidx import Fasta

    PYFAIDX_AVAILABLE = True
except ImportError:
    PYFAIDX_AVAILABLE = False
    print("警告: pyfaidx未安装，无法读取FASTA文件")
    print("请安装: pip install pyfaidx")

warnings.filterwarnings('ignore')

# =============================================================================================
#  物种和数据集配置
# =============================================================================================

SPECIES_CONFIG = {
    'human': {
        'name': 'human',
        'gene_model': 'gencode.v49.primary_assembly.basic.annotation.gtf',  # 已解压的GTF
        'genome': 'GRCh38.primary_assembly.genome.fa',
        'expression': 'GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.gct.gz',  # 仍为.gz
        'ppi_edge_file': 'processed_ppi/human_ppi_edge_index.pt',
        'has_version_in_expression': True,
        'expression_format': 'gct',
    },
    'mouse': {
        'name': 'mouse',
        'gene_model': 'gencode.vM38.primary_assembly.basic.annotation.gtf',  # 已解压的GTF
        'genome': 'GRCm39.primary_assembly.genome.fa',
        'expression': 'E-GEOD-70484-query-results.tpmss.tsv',  # 小鼠的TSV文件（假设未压缩）
        'ppi_edge_file': 'processed_ppi/mouse_ppi_edge_index.pt',
        'has_version_in_expression': False,
        'expression_format': 'tsv',
    }
}

# 序列长度配置（基因序列3kb + 邻居基因序列3kb = 6kb）
GENE_UPSTREAM_LENGTH = 1000  # 基因起始点上游长度
GENE_DOWNSTREAM_LENGTH = 500  # 基因起始点下游/终止点上游长度
GENE_TOTAL_LENGTH = GENE_UPSTREAM_LENGTH + GENE_DOWNSTREAM_LENGTH + GENE_UPSTREAM_LENGTH + GENE_DOWNSTREAM_LENGTH  # 3000bp

TOTAL_LENGTH = GENE_TOTAL_LENGTH * 2  # 6000bp

OUTPUT_DIR = "precomputed_sequences_ppi"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_clean_id(id_str):
    """清理基因ID，去除版本号等"""
    if pd.isna(id_str) or str(id_str).lower() == 'nan': return ""
    s = str(id_str).strip().upper()
    if re.match(r'^\d+\.', s): s = s.split('.', 1)[1]
    return s.split('.')[0]


def open_file_maybe_gzip(filepath, mode='rt'):
    """智能打开文件，自动处理gzip压缩"""
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode)
    else:
        return open(filepath, mode)


# =============================================================================================
#  PPI网络处理函数
# =============================================================================================

def load_ppi_network(species_config):
    """
    加载PPI网络，为每个基因找到最高权重的邻居
    返回: dict {gene_id: best_neighbor_id}
    """
    ppi_file = species_config['ppi_edge_file']
    print(f"📊 加载PPI网络: {ppi_file}")

    # 1. 加载字典数据
    ppi_data = torch.load(ppi_file, map_location='cpu')

    # 2. 从字典中提取真正的边索引张量
    # 根据你的 read.py 结果，键名是 'edge_index'
    if isinstance(ppi_data, dict) and 'edge_index' in ppi_data:
        edge_index = ppi_data['edge_index']
        print(f"✅ 成功从字典中提取 edge_index")
    else:
        # 兜底逻辑：如果不是字典或者是其他结构
        edge_index = ppi_data

    # 3. 验证是否成功获取到张量
    if not hasattr(edge_index, 'shape'):
        print(f"❌ 错误：提取的内容不是张量。类型: {type(edge_index)}")
        sys.exit(1)

    print(f"PPI网络包含 {edge_index.shape[1]} 条边")

    # 4. 构建邻居映射表
    gene_neighbors = defaultdict(list)
    for i in range(edge_index.shape[1]):
        # 注意：这里要求 edge_index 存储的是整数索引
        g1, g2 = edge_index[0, i].item(), edge_index[1, i].item()
        gene_neighbors[g1].append(g2)
        gene_neighbors[g2].append(g1)

    # 为每个基因选择第一个邻居作为最高权重邻居
    best_neighbor_idx = {}
    for gene_idx, neighbors in gene_neighbors.items():
        if neighbors:
            best_neighbor_idx[gene_idx] = neighbors[0]

    print(f"为 {len(best_neighbor_idx)} 个基因找到了邻居")

    return best_neighbor_idx, edge_index


# =============================================================================================
#  文件处理辅助函数（支持gzip）
# =============================================================================================

def read_gct_ids_only(gct_file):
    """只读取GCT文件的ID列（支持gzip）"""
    print(f"内存优化模式：只读取GCT文件的ID列: {gct_file}")

    # 首先读取维度信息
    with open_file_maybe_gzip(gct_file, 'rt') as f:
        # 跳过版本行
        version_line = f.readline().strip()
        # 读取维度行
        dim_line = f.readline().strip()
        dim_parts = dim_line.split('\t')
        n_rows, n_cols = int(dim_parts[0]), int(dim_parts[1])
        print(f"GCT维度: {n_rows}行, {n_cols}列")
        # 读取表头（但不使用）
        header = f.readline().strip().split('\t')

    # 只读取Name列（第一列）
    # 使用usecols参数指定只加载第一列
    df = pd.read_csv(
        gct_file,
        sep='\t',
        skiprows=2,
        usecols=[0],  # 只加载第一列（Name列）
        names=['gene_id'],  # 重命名为gene_id
        compression='gzip' if gct_file.endswith('.gz') else None
    )

    print(f"成功读取 {len(df)} 个基因ID")
    return df


def read_tsv_ids_only(tsv_file):
    """只读取TSV文件的基因ID列（支持gzip）"""
    print(f"内存优化模式：只读取TSV文件的ID列: {tsv_file}")

    # 跳过注释行，找到表头
    skip_rows = 0
    with open_file_maybe_gzip(tsv_file, 'rt') as f:
        first_line = f.readline().strip()
        while first_line.startswith('#'):
            skip_rows += 1
            first_line = f.readline().strip()

    # 读取表头行
    header_line = first_line
    headers = header_line.split('\t')

    # 找到基因ID列的位置
    possible_id_cols = ['Gene ID', 'GeneID', 'gene_id', 'Gene', 'gene']
    id_col_idx = None

    for i, col in enumerate(headers):
        if col in possible_id_cols:
            id_col_idx = i
            break

    if id_col_idx is None:
        # 默认使用第一列
        id_col_idx = 0
        print(f"未找到标准ID列，使用第一列: {headers[0]}")

    # 只读取ID列
    df = pd.read_csv(
        tsv_file,
        sep='\t',
        skiprows=skip_rows,
        usecols=[id_col_idx],
        names=['gene_id'],
        header=0,
        compression='gzip' if tsv_file.endswith('.gz') else None
    )

    print(f"成功读取 {len(df)} 个基因ID")
    return df


def load_target_genes_from_expression(expression_file, species_config):
    """加载表达量文件中的基因ID"""
    print(f"加载表达量文件ID列: {expression_file}")

    expr_format = species_config.get('expression_format', 'tsv')
    has_version = species_config.get('has_version_in_expression', True)

    if expr_format == 'gct':
        df = read_gct_ids_only(expression_file)
    else:
        df = read_tsv_ids_only(expression_file)

    target_genes = df['gene_id'].tolist()

    if not has_version:
        df['gene_id_base'] = df['gene_id'].apply(
            lambda x: str(x).split('.')[0] if '.' in str(x) else str(x)
        )
        target_genes_base = df['gene_id_base'].tolist()
        print(f"从表达量文件中获取到 {len(target_genes)} 个目标基因 (无版本号匹配)")
        return target_genes, target_genes_base, df
    else:
        print(f"从表达量文件中获取到 {len(target_genes)} 个目标基因 (保留版本号)")
        return target_genes, target_genes, df


# =============================================================================================
#  GTF解析函数（使用解压后的文件）
# =============================================================================================

def parse_gtf_for_genes(gtf_file, target_genes, target_genes_base, has_version_in_expression):
    """解析GTF文件中指定的目标基因（使用解压后的文件）"""
    print(f"解析GTF文件 (只提取目标基因): {gtf_file}")

    if has_version_in_expression:
        target_set = set(target_genes)
        match_key = 'gene_id'
    else:
        target_set = set(target_genes_base)
        match_key = 'gene_id_base'

    print(f"目标基因数量: {len(target_set)}")

    gene_data = []
    found_genes = set()

    # 直接使用open，因为GTF已解压
    with open(gtf_file, 'rt') as f:
        for line in tqdm(f, desc="解析GTF"):
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            feature = parts[2]
            if feature != 'gene':
                continue

            # 提取gene_id
            attributes = parts[8]
            gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
            if not gene_id_match:
                continue

            gene_id = gene_id_match.group(1)
            gene_id_base = gene_id.split('.')[0] if '.' in gene_id else gene_id

            # 根据匹配键判断是否为目标基因
            if match_key == 'gene_id' and gene_id in target_set:
                is_target = True
            elif match_key == 'gene_id_base' and gene_id_base in target_set:
                is_target = True
            else:
                continue

            if is_target:
                chrom = parts[0]
                start = int(parts[3])
                end = int(parts[4])
                strand = parts[6]

                gene_data.append({
                    'chromosome': chrom,
                    'start': start,
                    'end': end,
                    'strand': strand,
                    'gene_id': gene_id,
                    'gene_id_base': gene_id_base
                })

                if match_key == 'gene_id':
                    found_genes.add(gene_id)
                else:
                    found_genes.add(gene_id_base)

    gene_df = pd.DataFrame(gene_data)
    print(f"从GTF中找到 {len(gene_df)}/{len(target_set)} 个目标基因的位置信息")

    # 检查未找到的基因
    not_found = target_set - found_genes
    if not_found:
        sample_size = min(10, len(not_found))
        print(f"⚠️ 未在GTF中找到 {len(not_found)} 个基因: {list(not_found)[:sample_size]}...")

    return gene_df


# =============================================================================================
#  序列提取类
# =============================================================================================

class PPISequencePreprocessor:
    """提取表达基因及其PPI邻居基因的序列"""

    def __init__(self, output_dir=OUTPUT_DIR):
        self.cache_dir = output_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"序列输出目录: {self.cache_dir}")

    def get_cache_filename(self, species):
        """生成缓存文件名"""
        filename = f"{species}_ppi_sequences.pt"
        return os.path.join(self.cache_dir, filename)

    def extract_gene_sequence(self, genome, gene_info):
        """
        提取单个基因的3kb序列（原始逻辑）
        """
        chrom = gene_info['chromosome']
        start = gene_info['start']
        end = gene_info['end']
        strand = gene_info['strand']

        try:
            if strand == '+':
                # 正链基因
                # 启动子: start前1000 + start后500
                promoter_seq = str(genome[chrom][start - GENE_UPSTREAM_LENGTH: start + GENE_DOWNSTREAM_LENGTH])
                # 终止子: end前500 + end后1000
                terminator_seq = str(genome[chrom][end - GENE_DOWNSTREAM_LENGTH: end + GENE_UPSTREAM_LENGTH])
                sequence = promoter_seq + terminator_seq
            else:
                # 负链基因
                # 启动子区域 (围绕 TSS/end)
                promoter_raw = str(genome[chrom][end - GENE_DOWNSTREAM_LENGTH: end + GENE_UPSTREAM_LENGTH])
                promoter_seq = self.reverse_complement(promoter_raw)
                # 终止子区域 (围绕 TTS/start)
                terminator_raw = str(genome[chrom][start - GENE_UPSTREAM_LENGTH: start + GENE_DOWNSTREAM_LENGTH])
                terminator_seq = self.reverse_complement(terminator_raw)
                # 按生物学顺序拼接: [启动子] + [终止子]
                sequence = promoter_seq + terminator_seq

            # 检查序列长度并填充/截断
            expected_length = GENE_TOTAL_LENGTH
            if len(sequence) < expected_length:
                sequence = sequence.ljust(expected_length, 'N')
            elif len(sequence) > expected_length:
                sequence = sequence[:expected_length]

            return sequence.upper()

        except Exception as e:
            print(f"提取序列失败: {gene_info.get('gene_id', 'unknown')}, 染色体: {chrom}, 链: {strand}, 错误: {str(e)}")
            return 'N' * GENE_TOTAL_LENGTH

    def reverse_complement(self, seq):
        """反向互补序列"""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N',
                      'a': 't', 't': 'a', 'g': 'c', 'c': 'g'}
        return ''.join(complement.get(base, 'N') for base in reversed(seq))

    def sequence_to_tensor(self, sequence):
        """序列转one-hot编码张量"""
        sequence = sequence.upper()
        seq_length = len(sequence)

        tensor = torch.zeros(4, seq_length, dtype=torch.float32)
        base_mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

        for i, base in enumerate(sequence):
            if base in base_mapping:
                tensor[base_mapping[base], i] = 1.0

        return tensor

    def find_best_neighbor_gene(self, gene_id, gene_idx_map, idx_to_gene, best_neighbor_idx):
        """
        找到基因的最佳邻居基因ID
        返回: (neighbor_gene_id, neighbor_found)
        """
        if gene_id not in gene_idx_map:
            return None, False

        gene_idx = gene_idx_map[gene_id]

        if gene_idx in best_neighbor_idx:
            neighbor_idx = best_neighbor_idx[gene_idx]
            if neighbor_idx in idx_to_gene:
                return idx_to_gene[neighbor_idx], True

        return None, False

    def preprocess_ppi_sequences(self, species, genome_file, gene_model_file,
                                 expression_file, ppi_edge_file, overwrite=False):
        """
        预处理表达基因及其PPI邻居基因的序列
        """
        cache_file = self.get_cache_filename(species)

        # 检查缓存
        if os.path.exists(cache_file) and not overwrite:
            print(f"📁 检查已预处理的序列: {cache_file}")
            try:
                data = torch.load(cache_file, map_location='cpu')
                if all(key in data for key in ['sequences', 'gene_info', 'target_genes', 'neighbor_genes']):
                    print(f"✅ 成功加载预处理序列，包含 {len(data['target_genes'])} 个基因对")
                    print(f"序列形状: {data['sequences'].shape}")
                    return data['sequences'], data['gene_info'], data['target_genes'], data['neighbor_genes']
                else:
                    print(f"⚠️ 缓存文件格式不完整，重新预处理")
            except Exception as e:
                print(f"❌ 加载预处理文件失败: {e}，重新预处理")

        print(f"🔄 预处理物种 {species} 的PPI基因对序列...")
        print(f"序列长度配置: 基因序列 {GENE_TOTAL_LENGTH}bp + 邻居序列 {GENE_TOTAL_LENGTH}bp = {TOTAL_LENGTH}bp")

        # 1. 从表达量文件中加载目标基因
        species_config = SPECIES_CONFIG[species]
        target_genes, target_genes_base, expr_df = load_target_genes_from_expression(
            expression_file, species_config
        )

        # 2. 从GTF中提取目标基因的位置信息
        has_version = species_config.get('has_version_in_expression', True)
        gene_df = parse_gtf_for_genes(gene_model_file, target_genes, target_genes_base, has_version)

        if len(gene_df) == 0:
            print(f"❌ 错误: 在GTF中未找到任何目标基因")
            return None, None, None, None

        # 3. 创建基因ID到索引的映射
        gene_to_idx = {}
        idx_to_gene = {}
        for i, row in gene_df.iterrows():
            gene_id = row['gene_id'] if has_version else row['gene_id_base']
            gene_to_idx[gene_id] = i
            idx_to_gene[i] = gene_id

        # 4. 加载PPI网络
        best_neighbor_idx, edge_index = load_ppi_network(species_config)

        # 5. 为每个表达基因找到最佳邻居
        gene_neighbor_map = {}
        neighbor_found_count = 0

        for gene_id in (target_genes if has_version else target_genes_base):
            neighbor_id, found = self.find_best_neighbor_gene(
                gene_id, gene_to_idx, idx_to_gene, best_neighbor_idx
            )
            if found:
                gene_neighbor_map[gene_id] = neighbor_id
                neighbor_found_count += 1
            else:
                gene_neighbor_map[gene_id] = gene_id  # 找不到邻居就用自身

        print(
            f"📊 邻居基因找到率: {neighbor_found_count}/{len(target_genes)} ({neighbor_found_count / len(target_genes):.2%})")

        # 6. 加载基因组
        print(f"加载基因组: {genome_file}")
        try:
            genome = Fasta(genome_file, as_raw=True, sequence_always_upper=True, read_ahead=10000)
        except Exception as e:
            print(f"⚠️ 加载基因组失败: {e}")
            sys.exit(1)

        # 7. 创建基因信息快速查找表
        gene_info_dict = {}
        for _, row in gene_df.iterrows():
            gene_id = row['gene_id'] if has_version else row['gene_id_base']
            gene_info_dict[gene_id] = row

        # 8. 提取基因对序列
        sequences = []
        gene_info_list = []
        target_genes_list = []
        neighbor_genes_list = []

        success_count = 0
        failed_count = 0
        self_neighbor_count = 0

        for gene_id in tqdm(target_genes if has_version else target_genes_base,
                            desc=f"提取 {species} PPI序列"):
            neighbor_id = gene_neighbor_map[gene_id]

            # 获取基因和邻居的信息
            gene_info = gene_info_dict.get(gene_id)
            neighbor_info = gene_info_dict.get(neighbor_id)

            if gene_info is None:
                failed_count += 1
                continue

            # 提取基因序列
            gene_seq = self.extract_gene_sequence(genome, gene_info)

            # 提取邻居序列（如果找不到，用基因序列）
            if neighbor_info is not None:
                neighbor_seq = self.extract_gene_sequence(genome, neighbor_info)
                if neighbor_id == gene_id:
                    self_neighbor_count += 1
            else:
                neighbor_seq = gene_seq
                self_neighbor_count += 1

            # 拼接序列
            combined_seq = gene_seq + neighbor_seq

            # 转换为tensor
            tensor = self.sequence_to_tensor(combined_seq)
            sequences.append(tensor)

            # 保存信息
            gene_info_list.append({
                'gene_id': gene_info['gene_id'],
                'gene_id_base': gene_info['gene_id_base'],
                'chromosome': gene_info['chromosome'],
                'start': gene_info['start'],
                'end': gene_info['end'],
                'strand': gene_info['strand'],
            })
            target_genes_list.append(gene_info['gene_id'])
            neighbor_genes_list.append(neighbor_id)

            success_count += 1

        print(f"\n✅ 成功提取 {success_count} 个基因对序列")
        print(f"❌ 失败 {failed_count} 个")
        print(f"🔄 使用自身作为邻居: {self_neighbor_count} 个")

        if sequences:
            all_sequences = torch.stack(sequences)
            print(f"\n✅ 最终序列张量形状: {all_sequences.shape}")

            # 保存统计信息
            stats = {
                'species': species,
                'total_target_genes': len(target_genes),
                'success_count': success_count,
                'failed_count': failed_count,
                'neighbor_found_count': neighbor_found_count,
                'self_neighbor_count': self_neighbor_count,
                'gene_sequence_length': GENE_TOTAL_LENGTH,
                'total_sequence_length': TOTAL_LENGTH,
                'timestamp': datetime.now().isoformat()
            }

            stats_file = cache_file.replace('.pt', '_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 统计信息已保存至: {stats_file}")

            # 保存数据
            data_to_save = {
                'sequences': all_sequences,
                'gene_info': gene_info_list,
                'target_genes': target_genes_list,
                'neighbor_genes': neighbor_genes_list,
                'species': species,
                'gene_sequence_length': GENE_TOTAL_LENGTH,
                'total_length': TOTAL_LENGTH,
                'timestamp': datetime.now().isoformat()
            }

            torch.save(data_to_save, cache_file)
            print(f"💾 预处理序列已保存至: {cache_file}")

            return all_sequences, gene_info_list, target_genes_list, neighbor_genes_list

        return None, None, None, None


# =============================================================================================
#  主函数
# =============================================================================================

def main():
    parser = argparse.ArgumentParser(description='提取表达基因及其PPI邻居基因序列')
    parser.add_argument('--species', type=str, required=True,
                        choices=['human', 'mouse'],
                        help='要处理的物种 (human 或 mouse)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='输出目录')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的缓存文件')
    parser.add_argument('--expression_file', type=str, default=None,
                        help='指定表达量文件（覆盖默认）')
    parser.add_argument('--list_species', action='store_true',
                        help='列出支持的物种')

    args = parser.parse_args()

    if args.list_species:
        print("支持的物种:")
        for species, config in SPECIES_CONFIG.items():
            print(f"  {species}: {config['name']}")
            print(f"    基因组: {config['genome']}")
            print(f"    基因模型: {config['gene_model']}")
            print(f"    表达量: {config['expression']}")
            print(f"    PPI文件: {config['ppi_edge_file']}")
            print()
        return

    if not PYFAIDX_AVAILABLE:
        print("错误: pyfaidx未安装，无法读取FASTA文件")
        print("请运行: pip install pyfaidx")
        sys.exit(1)

    print(f"🚀 开始提取PPI序列 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置: {vars(args)}")
    print(f"📏 序列提取配置:")
    print(f"  - 基因上游长度: {GENE_UPSTREAM_LENGTH}bp")
    print(f"  - 基因下游长度: {GENE_DOWNSTREAM_LENGTH}bp")
    print(f"  - 基因序列长度: {GENE_TOTAL_LENGTH}bp")
    print(f"  - 总序列长度: {TOTAL_LENGTH}bp (基因+邻居)")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    config_file = os.path.join(args.output_dir, f'extraction_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"⚙️  配置已保存至: {config_file}")

    # 获取物种配置
    species = args.species
    species_config = SPECIES_CONFIG[species]

    # 确定文件路径
    expression_file = args.expression_file if args.expression_file else species_config['expression']
    ppi_edge_file = species_config['ppi_edge_file']

    # 检查文件是否存在
    for file_path, desc in [
        (expression_file, "表达量文件"),
        (species_config['genome'], "基因组文件"),
        (species_config['gene_model'], "基因模型文件"),
        (ppi_edge_file, "PPI边文件")
    ]:
        if not os.path.exists(file_path):
            print(f"❌ {desc}不存在: {file_path}")
            sys.exit(1)

    # 创建预处理器
    preprocessor = PPISequencePreprocessor(args.output_dir)

    # 预处理PPI序列
    sequences, gene_info, target_genes, neighbor_genes = preprocessor.preprocess_ppi_sequences(
        species,
        species_config['genome'],
        species_config['gene_model'],
        expression_file,
        ppi_edge_file,
        args.overwrite
    )

    # 生成总结
    print(f"\n{'=' * 80}")
    print(f"📋 PPI序列提取总结")
    print(f"{'=' * 80}")

    if sequences is not None:
        print(f"\n物种: {species}")
        print(f"  成功提取基因对数: {len(target_genes)}")
        print(f"  序列张量形状: {sequences.shape}")
        print(f"  缓存文件: {preprocessor.get_cache_filename(species)}")

        # 统计邻居来源
        neighbor_stats = {
            '找到的邻居': sum(1 for i, n in enumerate(neighbor_genes)
                              if n != gene_info[i]['gene_id_base']),
            '自身作为邻居': sum(1 for i, n in enumerate(neighbor_genes)
                                if n == gene_info[i]['gene_id_base'])
        }
        print(f"\n邻居基因统计:")
        print(f"  - 找到真实邻居: {neighbor_stats['找到的邻居']}")
        print(f"  - 使用自身替代: {neighbor_stats['自身作为邻居']}")

        print(f"\n🎉 PPI序列提取完成!")
    else:
        print("❌ 未成功提取任何序列")

    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()