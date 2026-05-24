# -*- coding: utf-8 -*-
"""
序列提取工具 - 从表达量文件中读取目标基因，提取对应的基因组序列
修改为按照原始逻辑提取，但不加中间的20bp间隔区
用法：python sequence_extractor.py --species human
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
#  物种和数据集配置（更新为.fa后缀）
# =============================================================================================

SPECIES_CONFIG = {
    'human': {
        'name': 'human',
        'gene_model': 'gencode.v49.primary_assembly.basic.annotation.gtf',
        'genome': 'GRCh38.primary_assembly.genome.fa',
        'expression': 'GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.gct.gz',
        'has_version_in_expression': True,
        'expression_format': 'gct',
    },
    'mouse': {
        'name': 'mouse',
        'gene_model': 'gencode.vM38.primary_assembly.basic.annotation.gtf',
        'genome': 'GRCm39.primary_assembly.genome.fa',
        'expression': 'E-GEOD-70484-query-results.tpmss.tsv',
        'has_version_in_expression': False,
        'expression_format': 'tsv',
    }
}

# 序列长度配置（按照原始逻辑但不加20bp间隔）
UPSTREAM_LENGTH = 1000  # 基因起始点上游长度
DOWNSTREAM_LENGTH = 500  # 基因起始点下游/终止点上游长度
TOTAL_LENGTH = UPSTREAM_LENGTH + DOWNSTREAM_LENGTH + UPSTREAM_LENGTH + DOWNSTREAM_LENGTH  # 3000bp


# =============================================================================================
#  文件处理辅助函数
# =============================================================================================

def open_file_maybe_gzip(filepath, mode='rt'):
    """智能打开文件，自动处理gzip压缩"""
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode)
    else:
        return open(filepath, mode)


def read_gct_ids_only(gct_file):
    """
    内存优化：只读取GCT文件的ID列（Name），不加载整个表达量矩阵

    GCT格式：
    第1行: 版本号
    第2行: 维度信息 (行数\t列数)
    第3行: 表头 (Name\tDescription\t样本1\t样本2...)
    """
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

        # 读取表头
        header = f.readline().strip().split('\t')

    # 只读取Name列（第一列）
    df = pd.read_csv(
        gct_file,
        sep='\t',
        skiprows=2,
        usecols=[0],
        names=['gene_id'],
        compression='gzip' if gct_file.endswith('.gz') else None
    )

    print(f"成功读取 {len(df)} 个基因ID")
    return df


def read_tsv_ids_only(tsv_file):
    """
    内存优化：只读取TSV文件的基因ID列
    """
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
    """
    内存优化：只加载基因ID，不加载整个表达量矩阵
    """
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
#  GTF解析函数
# =============================================================================================

def parse_gtf_for_genes(gtf_file, target_genes, target_genes_base, has_version_in_expression):
    """
    只解析GTF文件中指定的目标基因，处理版本号差异
    """
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

    with open_file_maybe_gzip(gtf_file, 'rt') as f:
        for line in tqdm(f, desc="解析GTF"):
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            feature = parts[2]
            if feature != 'gene':
                continue

            attributes = parts[8]
            gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
            if not gene_id_match:
                continue

            gene_id = gene_id_match.group(1)
            gene_id_base = gene_id.split('.')[0] if '.' in gene_id else gene_id

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

    not_found = target_set - found_genes
    if not_found:
        sample_size = min(10, len(not_found))
        print(f"⚠️ 未在GTF中找到 {len(not_found)} 个基因: {list(not_found)[:sample_size]}...")

    return gene_df


# =============================================================================================
#  序列预处理管理器
# =============================================================================================

class SequencePreprocessor:
    """按需预处理目标基因序列，只处理有表达数据的基因"""

    def __init__(self, output_dir='precomputed_sequences'):
        self.cache_dir = output_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"序列输出目录: {self.cache_dir}")

    def get_cache_filename(self, species):
        """生成缓存文件名"""
        filename = f"{species}_sequences.pt"
        return os.path.join(self.cache_dir, filename)

    def extract_gene_sequence_original_logic(self, genome, gene_info):
        """
        原版提取方法 - 完全保留原始逻辑
        """
        chrom = gene_info['chromosome']
        start = gene_info['start']
        end = gene_info['end']
        strand = gene_info['strand']

        try:
            if strand == '+':
                # ==================== 正链基因 ====================
                # 启动子: start前1000 + start后500
                promoter_seq = str(genome[chrom][start - UPSTREAM_LENGTH: start + DOWNSTREAM_LENGTH])
                # 终止子: end前500 + end后1000
                terminator_seq = str(genome[chrom][end - DOWNSTREAM_LENGTH: end + UPSTREAM_LENGTH])
                sequence = promoter_seq + terminator_seq

            else:
                # ==================== 负链基因 ====================
                # 1. 启动子区域 (围绕 TSS/end)
                promoter_raw = str(genome[chrom][end - DOWNSTREAM_LENGTH: end + UPSTREAM_LENGTH])
                promoter_seq = self.reverse_complement(promoter_raw)

                # 2. 终止子区域 (围绕 TTS/start)
                terminator_raw = str(genome[chrom][start - UPSTREAM_LENGTH: start + DOWNSTREAM_LENGTH])
                terminator_seq = self.reverse_complement(terminator_raw)

                # 3. 按生物学顺序拼接: [启动子] + [终止子]
                sequence = promoter_seq + terminator_seq

            # 检查序列长度并填充/截断
            expected_length = TOTAL_LENGTH
            if len(sequence) < expected_length:
                sequence = sequence.ljust(expected_length, 'N')
            elif len(sequence) > expected_length:
                sequence = sequence[:expected_length]

            return sequence.upper()

        except Exception as e:
            print(f"提取序列失败: {gene_info['gene_id']}, 染色体: {chrom}, 链: {strand}, 错误: {str(e)}")
            return 'N' * TOTAL_LENGTH

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

    def preprocess_target_genes(self, species, genome_file, gene_model_file, expression_file, overwrite=False):
        """
        预处理有表达数据的目标基因序列
        """
        cache_file = self.get_cache_filename(species)

        # 检查是否已经预处理过
        if os.path.exists(cache_file) and not overwrite:
            print(f"📁 检查已预处理的序列: {cache_file}")
            try:
                data = torch.load(cache_file, map_location='cpu')
                if all(key in data for key in ['sequences', 'gene_info', 'target_genes']):
                    print(f"✅ 成功加载预处理序列，包含 {len(data['target_genes'])} 个基因")
                    print(f"序列形状: {data['sequences'].shape}")
                    return data['sequences'], data['gene_info'], data['target_genes']
                else:
                    print(f"⚠️ 缓存文件格式不完整，重新预处理")
            except Exception as e:
                print(f"❌ 加载预处理文件失败: {e}，重新预处理")

        print(f"🔄 预处理物种 {species} 的目标基因序列...")
        print(f"序列提取逻辑: 启动子({UPSTREAM_LENGTH}+{DOWNSTREAM_LENGTH}bp) + 终止子({DOWNSTREAM_LENGTH}+{UPSTREAM_LENGTH}bp)")
        print(f"总长度: {TOTAL_LENGTH}bp (不加20bp间隔区)")

        # 1. 从表达量文件中加载目标基因
        species_config = SPECIES_CONFIG[species]
        target_genes, target_genes_base, expr_df = load_target_genes_from_expression(
            expression_file, species_config
        )

        # 2. 从GTF中只提取目标基因的位置信息
        has_version = species_config.get('has_version_in_expression', True)
        gene_df = parse_gtf_for_genes(gene_model_file, target_genes, target_genes_base, has_version)

        if len(gene_df) == 0:
            print(f"❌ 错误: 在GTF中未找到任何目标基因")
            return None, None, None

        # 3. 加载基因组
        print(f"加载基因组: {genome_file}")

        try:
            genome = Fasta(genome_file, as_raw=True, sequence_always_upper=True, read_ahead=10000)
        except Exception as e:
            print(f"⚠️ 加载基因组失败: {e}")
            print(f"请确保基因组文件存在且格式正确: {genome_file}")
            sys.exit(1)

        # 4. 提取目标基因序列
        sequences = []
        gene_info_list = []
        success_count = 0
        failed_count = 0
        failed_genes = []

        # 创建基因ID到行的映射
        if has_version:
            gene_id_to_row = {row['gene_id']: row for _, row in gene_df.iterrows()}
        else:
            gene_id_to_row = {row['gene_id_base']: row for _, row in gene_df.iterrows()}

        for gene_id, gene_id_base in tqdm(zip(target_genes, target_genes_base),
                                          total=len(target_genes),
                                          desc=f"提取 {species} 序列"):

            match_id = gene_id if has_version else gene_id_base

            if match_id in gene_id_to_row:
                row = gene_id_to_row[match_id]
                try:
                    sequence = self.extract_gene_sequence_original_logic(genome, row)

                    if sequence and len(sequence) == TOTAL_LENGTH:
                        tensor = self.sequence_to_tensor(sequence)
                        sequences.append(tensor)
                        gene_info_list.append({
                            'gene_id': row['gene_id'],
                            'gene_id_base': row['gene_id_base'],
                            'chromosome': row['chromosome'],
                            'start': row['start'],
                            'end': row['end'],
                            'strand': row['strand'],
                        })
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_genes.append(gene_id)
                        if len(failed_genes) <= 10:
                            print(f"序列长度错误: {gene_id}, 期望 {TOTAL_LENGTH}bp, 实际 {len(sequence) if sequence else 0}bp")
                except Exception as e:
                    failed_count += 1
                    failed_genes.append(gene_id)
                    if len(failed_genes) <= 10:
                        print(f"提取序列异常: {gene_id}, 错误: {str(e)}")
            else:
                failed_count += 1
                failed_genes.append(gene_id)

        print(f"✅ 成功提取 {success_count} 个基因序列")
        print(f"❌ 失败 {failed_count} 个基因序列")

        if sequences:
            all_sequences = torch.stack(sequences)
            print(f"✅ 成功预处理 {len(sequences)} 个基因序列")
            print(f"序列张量形状: {all_sequences.shape}")

            # 保存统计信息
            stats = {
                'species': species,
                'total_target_genes': len(target_genes),
                'success_count': success_count,
                'failed_count': failed_count,
                'failed_genes': failed_genes[:50] if failed_genes else [],
                'upstream_length': UPSTREAM_LENGTH,
                'downstream_length': DOWNSTREAM_LENGTH,
                'total_length': TOTAL_LENGTH,
                'extraction_logic': 'original_without_20bp_gap',
                'timestamp': datetime.now().isoformat()
            }

            stats_file = cache_file.replace('.pt', '_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 统计信息已保存至: {stats_file}")

            # 保存到文件
            data_to_save = {
                'sequences': all_sequences,
                'gene_info': gene_info_list,
                'target_genes': [info['gene_id'] for info in gene_info_list],
                'species': species,
                'upstream_length': UPSTREAM_LENGTH,
                'downstream_length': DOWNSTREAM_LENGTH,
                'total_length': TOTAL_LENGTH,
                'extraction_logic': 'original_without_20bp_gap',
                'success_count': success_count,
                'failed_count': failed_count,
                'timestamp': datetime.now().isoformat()
            }

            torch.save(data_to_save, cache_file)
            print(f"💾 预处理序列已保存至: {cache_file}")

            return all_sequences, gene_info_list, data_to_save['target_genes']

        return None, None, None


# =============================================================================================
#  主函数
# =============================================================================================

def main():
    parser = argparse.ArgumentParser(description='提取目标基因序列（原始逻辑，不加20bp间隔）')
    parser.add_argument('--species', type=str, required=True,
                        choices=['human', 'mouse'],
                        help='要处理的物种 (human 或 mouse)')
    parser.add_argument('--output_dir', type=str, default='precomputed_sequences',
                        help='输出目录')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的缓存文件')
    parser.add_argument('--expression_file', type=str, default=None,
                        help='指定表达量文件（覆盖默认）')
    parser.add_argument('--list_species', action='store_true',
                        help='列出支持的物种')

    args = parser.parse_args()

    # 列出支持的物种
    if args.list_species:
        print("支持的物种:")
        for species, config in SPECIES_CONFIG.items():
            print(f"  {species}: {config['name']}")
            print(f"    基因组: {config['genome']}")
            print(f"    基因模型: {config['gene_model']}")
            print(f"    表达量: {config['expression']}")
            print(f"    ID版本号: {'有' if config.get('has_version_in_expression', True) else '无'}")
            print()
        return

    # 检查pyfaidx是否可用
    if not PYFAIDX_AVAILABLE:
        print("错误: pyfaidx未安装，无法读取FASTA文件")
        print("请运行: pip install pyfaidx")
        sys.exit(1)

    print(f"🚀 开始提取序列 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置: {vars(args)}")
    print(f"📏 序列提取配置:")
    print(f"  - 上游长度: {UPSTREAM_LENGTH}bp")
    print(f"  - 下游长度: {DOWNSTREAM_LENGTH}bp")
    print(f"  - 总长度: {TOTAL_LENGTH}bp")

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

    # 确定表达量文件
    expression_file = args.expression_file if args.expression_file else species_config['expression']

    # 检查文件是否存在
    for file_path, desc in [
        (expression_file, "表达量文件"),
        (species_config['genome'], "基因组文件"),
        (species_config['gene_model'], "基因模型文件")
    ]:
        if not os.path.exists(file_path):
            print(f"❌ {desc}不存在: {file_path}")
            sys.exit(1)

    # 创建预处理器
    preprocessor = SequencePreprocessor(args.output_dir)

    # 预处理目标基因序列
    sequences, gene_info, target_genes = preprocessor.preprocess_target_genes(
        species,
        species_config['genome'],
        species_config['gene_model'],
        expression_file,
        args.overwrite
    )

    # 生成总结
    print(f"\n{'=' * 80}")
    print(f"📋 序列提取总结")
    print(f"{'=' * 80}")

    if sequences is not None:
        print(f"\n物种: {species}")
        print(f"  成功提取基因数: {len(target_genes)}")
        print(f"  序列张量形状: {sequences.shape}")
        print(f"  缓存文件: {preprocessor.get_cache_filename(species)}")
        print(f"\n🎉 序列提取完成!")
    else:
        print("❌ 未成功提取任何序列")

    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()