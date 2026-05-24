# -*- coding: utf-8 -*-
"""
序列提取工具 - 从处理好的标签文件中读取目标基因，提取对应的基因组序列
修改为NT优化版：提取6kb序列以匹配Nucleotide-Transformer的1000 token窗口
直接保存原始序列字符串，不再进行one-hot转换
用法：python sequence_extractor_nt.py --species human
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
        'gene_model': 'gencode.v49.primary_assembly.basic.annotation.gtf',
        'gene_model_gz': 'gencode.v49.primary_assembly.basic.annotation.gtf.gz',
        'genome': 'GRCh38.primary_assembly.genome.fa',
        'labels_file': 'processed_labels/human_labels.pt',
        'has_version_in_labels': True,
        'chrom_prefix': 'chr',
    },
    'mouse': {
        'name': 'mouse',
        'gene_model': 'gencode.vM38.primary_assembly.basic.annotation.gtf',
        'gene_model_gz': 'gencode.vM38.primary_assembly.basic.annotation.gtf.gz',
        'genome': 'GRCm39.primary_assembly.genome.fa',
        'labels_file': 'processed_labels/mouse_labels.pt',
        'has_version_in_labels': True,
        'chrom_prefix': 'chr',
    }
}

# ===================== NT优化版长度配置 (6kb) =====================
UPSTREAM_TSS = 2000
DOWNSTREAM_TSS = 1000
UPSTREAM_TTS = 1000
DOWNSTREAM_TTS = 2000
TOTAL_LENGTH = UPSTREAM_TSS + DOWNSTREAM_TSS + UPSTREAM_TTS + DOWNSTREAM_TTS  # 6000bp


# =============================================================================================
#  文件处理辅助函数
# =============================================================================================

def find_available_file(file_paths, descriptions):
    """检查多个文件路径，返回第一个存在的文件"""
    for file_path, desc in zip(file_paths, descriptions):
        if os.path.exists(file_path):
            print(f"✅ 找到{desc}: {file_path}")
            return file_path
    return None


def open_file_maybe_gzip(filepath, mode='rt'):
    """智能打开文件，自动处理gzip压缩"""
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode)
    else:
        return open(filepath, mode)


def load_target_genes_from_labels(labels_file):
    """从处理好的标签文件中加载目标基因ID"""
    print(f"📂 从标签文件加载目标基因: {labels_file}")

    if not os.path.exists(labels_file):
        print(f"❌ 错误: 标签文件不存在: {labels_file}")
        print(f"请先运行 tpm_process.py 生成标签文件")
        sys.exit(1)

    data = torch.load(labels_file, map_location='cpu')

    if 'gene_id' in data:
        target_genes = data['gene_id']
    elif 'gene_ids' in data:
        target_genes = data['gene_ids']
    else:
        possible_keys = ['genes', 'target_genes', 'index']
        for key in possible_keys:
            if key in data:
                target_genes = data[key]
                break
        else:
            print(f"❌ 无法从标签文件中找到基因ID列表")
            print(f"文件中的键: {list(data.keys())}")
            sys.exit(1)

    print(f"✅ 成功加载 {len(target_genes)} 个目标基因")
    print(f"  基因ID示例: {target_genes[:5]}")

    # 生成无版本号的ID列表
    target_genes_base = []
    for gid in target_genes:
        if '.' in str(gid):
            target_genes_base.append(str(gid).split('.')[0])
        else:
            target_genes_base.append(str(gid))

    return target_genes, target_genes_base


# =============================================================================================
#  GTF解析函数
# =============================================================================================

def parse_gtf_for_genes(gtf_file, target_genes, target_genes_base, has_version_in_labels):
    """只解析GTF文件中指定的目标基因，处理版本号差异"""
    print(f"解析GTF文件 (只提取目标基因): {gtf_file}")

    if has_version_in_labels:
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
#  序列预处理管理器 (NT优化版 - 只保存字符串)
# =============================================================================================

class SequencePreprocessor:
    def __init__(self, output_dir='precomputed_sequences_NT'):
        self.cache_dir = output_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"NT优化版序列输出目录: {self.cache_dir}")

    def get_cache_filename(self, species):
        """生成缓存文件名"""
        filename = f"{species}_sequences_NT_6kb.pt"
        return os.path.join(self.cache_dir, filename)

    def extract_gene_sequence_for_nt(self, genome, gene_info, chrom_prefix=''):
        """
        NT优化版提取方法 - 修复染色体前缀问题
        """
        chrom = gene_info['chromosome']

        # 检查GTF中的染色体是否已经有前缀
        if chrom_prefix:
            if chrom.startswith(chrom_prefix):
                chrom_with_prefix = chrom
            else:
                chrom_with_prefix = f"{chrom_prefix}{chrom}"
        else:
            chrom_with_prefix = chrom

        start = gene_info['start']
        end = gene_info['end']
        strand = gene_info['strand']

        try:
            # 检查染色体是否存在
            if chrom_with_prefix not in genome:
                return 'N' * TOTAL_LENGTH

            if strand == '+':
                # 正链基因
                tss_start = max(1, start - UPSTREAM_TSS)
                tss_end = start + DOWNSTREAM_TSS
                tss_seq = str(genome[chrom_with_prefix][tss_start:tss_end])

                tts_start = max(1, end - UPSTREAM_TTS)
                tts_end = end + DOWNSTREAM_TTS
                tts_seq = str(genome[chrom_with_prefix][tts_start:tts_end])

                sequence = tss_seq + tts_seq

            else:
                # 负链基因
                tss_start = max(1, end - DOWNSTREAM_TSS)
                tss_end = end + UPSTREAM_TSS
                tss_raw = str(genome[chrom_with_prefix][tss_start:tss_end])
                tss_seq = self.reverse_complement(tss_raw)

                tts_start = max(1, start - DOWNSTREAM_TTS)
                tts_end = start + UPSTREAM_TTS
                tts_raw = str(genome[chrom_with_prefix][tts_start:tts_end])
                tts_seq = self.reverse_complement(tts_raw)

                sequence = tss_seq + tts_seq

            # 检查序列长度
            if len(sequence) < TOTAL_LENGTH:
                sequence = sequence.ljust(TOTAL_LENGTH, 'N')
            elif len(sequence) > TOTAL_LENGTH:
                sequence = sequence[:TOTAL_LENGTH]

            return sequence.upper()

        except Exception as e:
            return 'N' * TOTAL_LENGTH

    def reverse_complement(self, seq):
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N',
                      'a': 't', 't': 'a', 'g': 'c', 'c': 'g'}
        return ''.join(complement.get(base, 'N') for base in reversed(seq))

    def preprocess_target_genes(self, species, genome_file, gene_model_file, labels_file, overwrite=False):
        cache_file = self.get_cache_filename(species)

        if os.path.exists(cache_file) and not overwrite:
            print(f"📁 检查已预处理的序列: {cache_file}")
            try:
                data = torch.load(cache_file, map_location='cpu')
                if all(key in data for key in ['sequences', 'gene_info', 'target_genes']):
                    print(f"✅ 成功加载预处理序列，包含 {len(data['target_genes'])} 个基因")
                    return data['sequences'], data['gene_info'], data['target_genes']
            except Exception as e:
                print(f"❌ 加载预处理文件失败: {e}，重新预处理")

        print(f"🔄 预处理物种 {species} 的目标基因序列 (NT优化版 6kb)...")
        print(
            f"📏 序列配置: TSS上游{UPSTREAM_TSS}+下游{DOWNSTREAM_TSS}bp + TTS上游{UPSTREAM_TTS}+下游{DOWNSTREAM_TTS}bp = {TOTAL_LENGTH}bp")

        # 加载目标基因
        species_config = SPECIES_CONFIG[species]
        target_genes, target_genes_base = load_target_genes_from_labels(labels_file)

        # 解析GTF
        has_version = species_config.get('has_version_in_labels', True)
        gene_df = parse_gtf_for_genes(gene_model_file, target_genes, target_genes_base, has_version)

        if len(gene_df) == 0:
            print(f"❌ 错误: 在GTF中未找到任何目标基因")
            return None, None, None

        # 加载基因组
        print(f"加载基因组: {genome_file}")
        try:
            genome = Fasta(genome_file, as_raw=True, sequence_always_upper=True)
            print(f"✅ 基因组加载成功，包含 {len(genome.keys())} 条染色体")
        except Exception as e:
            print(f"❌ 加载基因组失败: {e}")
            sys.exit(1)

        # 提取序列
        sequences = []  # 直接存储字符串
        gene_info_list = []
        success_count = 0
        failed_count = 0
        failed_genes = []

        if has_version:
            gene_id_to_row = {row['gene_id']: row for _, row in gene_df.iterrows()}
        else:
            gene_id_to_row = {row['gene_id_base']: row for _, row in gene_df.iterrows()}

        # 获取染色体前缀配置
        chrom_prefix = species_config.get('chrom_prefix', '')

        # 创建进度条
        pbar = tqdm(zip(target_genes, target_genes_base),
                    total=len(target_genes),
                    desc=f"提取 {species} 6kb序列")

        for gene_id, gene_id_base in pbar:
            match_id = gene_id if has_version else gene_id_base

            if match_id in gene_id_to_row:
                row = gene_id_to_row[match_id]
                sequence = self.extract_gene_sequence_for_nt(genome, row, chrom_prefix)

                if sequence and len(sequence) == TOTAL_LENGTH:
                    sequences.append(sequence)
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
            else:
                failed_count += 1
                failed_genes.append(gene_id)

            # 动态更新进度条描述
            pbar.set_postfix({
                'success': success_count,
                'failed': failed_count
            })

        print(f"\n✅ 成功提取 {success_count} 个基因序列")
        print(f"❌ 失败 {failed_count} 个基因序列")

        if sequences:
            # 保存数据
            data_to_save = {
                'sequences': sequences,
                'gene_info': gene_info_list,
                'target_genes': [info['gene_id'] for info in gene_info_list],
                'total_length': TOTAL_LENGTH,
                'nt_tokens': TOTAL_LENGTH // 6,
                'timestamp': datetime.now().isoformat()
            }

            torch.save(data_to_save, cache_file)
            print(f"💾 预处理序列已保存至: {cache_file}")

            # 保存统计信息
            stats = {
                'species': species,
                'total_target_genes': len(target_genes),
                'success_count': success_count,
                'failed_count': failed_count,
                'failed_genes': failed_genes[:50],
                'total_length': TOTAL_LENGTH,
                'nt_tokens': TOTAL_LENGTH // 6,
                'timestamp': datetime.now().isoformat()
            }

            stats_file = cache_file.replace('.pt', '_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 统计信息已保存至: {stats_file}")

            return sequences, gene_info_list, [info['gene_id'] for info in gene_info_list]

        return None, None, None


# =============================================================================================
#  主函数
# =============================================================================================

def main():
    parser = argparse.ArgumentParser(description='提取目标基因序列 - NT优化版(6kb)')
    parser.add_argument('--species', type=str, required=True,
                        choices=['human', 'mouse'],
                        help='要处理的物种')
    parser.add_argument('--output_dir', type=str, default='precomputed_sequences_NT',
                        help='输出目录')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的缓存文件')
    parser.add_argument('--labels_file', type=str, default=None,
                        help='指定标签文件')

    args = parser.parse_args()

    if not PYFAIDX_AVAILABLE:
        print("错误: pyfaidx未安装")
        sys.exit(1)

    print(f"🚀 开始提取序列 (NT优化版 6kb) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置: {vars(args)}")

    os.makedirs(args.output_dir, exist_ok=True)

    species = args.species
    species_config = SPECIES_CONFIG[species]

    # 查找GTF文件
    gene_model_file = find_available_file(
        [species_config['gene_model'], species_config['gene_model_gz']],
        ["解压版GTF文件", "压缩版GTF文件"]
    )

    if gene_model_file is None:
        print(f"❌ 错误: 未找到GTF文件")
        sys.exit(1)

    # 确定标签文件
    labels_file = args.labels_file if args.labels_file else species_config['labels_file']

    # 检查文件
    for file_path, desc in [(labels_file, "标签文件"), (species_config['genome'], "基因组文件")]:
        if not os.path.exists(file_path):
            print(f"❌ {desc}不存在: {file_path}")
            sys.exit(1)

    # 创建预处理器
    preprocessor = SequencePreprocessor(args.output_dir)

    # 预处理
    sequences, gene_info, target_genes = preprocessor.preprocess_target_genes(
        species,
        species_config['genome'],
        gene_model_file,
        labels_file,
        args.overwrite
    )

    if sequences is not None:
        print(f"\n{'=' * 80}")
        print(f"✅ 处理完成！")
        print(f"物种: {species}")
        print(f"成功提取基因数: {len(target_genes)}")
        print(f"序列长度: {len(sequences[0])}bp")
        print(f"NT Token数: {len(sequences[0]) // 6}")
        print(f"缓存文件: {preprocessor.get_cache_filename(species)}")
    else:
        print("❌ 处理失败")


if __name__ == "__main__":
    main()