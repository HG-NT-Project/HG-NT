# -*- coding: utf-8 -*-
"""
序列提取工具 - 从小龙虾的anno.summary.xls文件中读取基因信息，提取对应的基因组序列
适配小龙虾数据：使用start和end代替TSS/TTS
提取启动子区（start附近）和终止子区（end附近），合并为3000bp序列
用法：python sequence_extractor.py
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
#  小龙虾文件配置
# =============================================================================================

# 输入文件
GENE_MODEL_FILE = 'anno.summary.xls'  # 基因注释文件
GENOME_FILE = 'ref.fa'  # 基因组文件
LABEL_FILE = 'crayfish_labels.csv'  # 表达量标签文件

# 输出目录
OUTPUT_DIR = 'processed_sequence'  # 修改为processed_sequence
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 序列长度配置（参考原始逻辑：启动子区1500bp + 终止子区1500bp = 3000bp）
PROMOTER_UPSTREAM = 1000  # 启动子区上游长度
PROMOTER_DOWNSTREAM = 500  # 启动子区下游长度
TERMINATOR_UPSTREAM = 500  # 终止子区上游长度
TERMINATOR_DOWNSTREAM = 1000  # 终止子区下游长度
TOTAL_LENGTH = PROMOTER_UPSTREAM + PROMOTER_DOWNSTREAM + TERMINATOR_UPSTREAM + TERMINATOR_DOWNSTREAM  # 3000bp


# =============================================================================================
#  数据加载函数
# =============================================================================================

def load_crayfish_labels(labels_file):
    """
    读取小龙虾的表达量标签文件（CSV格式）
    返回基因ID列表
    """
    print(f"📂 读取小龙虾标签文件: {labels_file}")

    if not os.path.exists(labels_file):
        print(f"❌ 标签文件不存在: {labels_file}")
        return None, None

    df = pd.read_csv(labels_file)
    print(f"✅ 标签数据加载成功: {df.shape}")
    print(f"   列名: {list(df.columns)}")

    # 检查gene_id列
    if 'gene_id' not in df.columns:
        # 尝试其他可能的列名
        possible_id_cols = ['GeneID', 'gene', 'Gene', 'id', 'ID']
        found = False
        for col in possible_id_cols:
            if col in df.columns:
                df = df.rename(columns={col: 'gene_id'})
                found = True
                print(f"🔄 已将 '{col}' 列重命名为 'gene_id'")
                break
        if not found:
            print(f"❌ 未找到基因ID列，可用的列: {list(df.columns)}")
            return None, None

    # 过滤掉label为NaN的样本
    if 'label' in df.columns:
        initial_count = len(df)
        df = df[df['label'].notna()].copy()
        print(f"🧹 过滤label为NaN: {initial_count} → {len(df)}")
    else:
        print(f"⚠️ 未找到'label'列，使用所有样本")

    gene_ids = df['gene_id'].tolist()
    print(f"📊 共 {len(gene_ids)} 个有效基因")

    return gene_ids, df


def parse_crayfish_annotation(anno_file, target_genes):
    print(f"📂 解析小龙虾注释文件: {anno_file}")

    df = pd.read_csv(anno_file, sep='\t')
    target_set = set(target_genes)

    gene_data = []
    # --- 核心修改：增加去重集合 ---
    seen_genes = set()

    strand_col = next((c for c in df.columns if c.lower() == 'strand'), None)

    for _, row in df.iterrows():
        gene_id_raw = str(row['GeneID'])
        parts = gene_id_raw.split(':')
        if len(parts) < 5: continue

        gene_id = parts[0]

        # --- 核心修改：如果该基因已经提取过，直接跳过 ---
        if gene_id not in target_set or gene_id in seen_genes:
            continue

        chrom = parts[2]
        raw_pos = parts[3]
        start = int(raw_pos.split('..')[0]) if '..' in raw_pos else int(raw_pos)
        end = int(raw_pos.split('..')[1]) if '..' in raw_pos else int(parts[4])

        strand = row[strand_col] if strand_col else (parts[4] if len(parts) > 4 else '+')

        gene_data.append({
            'gene_id': gene_id,
            'chromosome': chrom,
            'start': start,
            'end': end,
            'strand': strand
        })

        # 记录已处理的基因
        seen_genes.add(gene_id)

    gene_df = pd.DataFrame(gene_data)
    # 强制按照 target_genes 的原始顺序进行排序，确保 100% 对齐
    gene_df['gene_id'] = pd.Categorical(gene_df['gene_id'], categories=target_genes, ordered=True)
    gene_df = gene_df.sort_values('gene_id').reset_index(drop=True)

    print(f"✅ 去重完成！最终获得 {len(gene_df)} 个唯一基因的位置信息")
    return gene_df


# =============================================================================================
#  序列提取类
# =============================================================================================

class CrayfishSequenceExtractor:
    """小龙虾基因序列提取器"""

    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 序列输出目录: {self.output_dir}")

    def reverse_complement(self, seq):
        """反向互补序列"""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N',
                      'a': 't', 't': 'a', 'g': 'c', 'c': 'g'}
        return ''.join(complement.get(base, 'N') for base in reversed(seq))

    def extract_gene_sequence(self, genome, gene_info):
        """
        提取基因序列（启动子区 + 终止子区）
        使用start和end代替TSS和TTS
        """
        chrom = gene_info['chromosome']
        start = gene_info['start']
        end = gene_info['end']
        strand = gene_info['strand']

        try:
            if strand == '+' or strand == '1' or strand == '.':
                # ==================== 正链基因 ====================
                # 启动子区: start上游1000bp + start下游500bp
                promoter_start = max(0, start - PROMOTER_UPSTREAM)
                promoter_end = start + PROMOTER_DOWNSTREAM
                promoter_seq = str(genome[chrom][promoter_start:promoter_end])

                # 终止子区: end上游500bp + end下游1000bp
                terminator_start = max(0, end - TERMINATOR_UPSTREAM)
                terminator_end = end + TERMINATOR_DOWNSTREAM
                terminator_seq = str(genome[chrom][terminator_start:terminator_end])

                # 合并序列
                sequence = promoter_seq + terminator_seq

            else:
                # ==================== 负链基因 ====================
                # 负链需要反向互补

                # 启动子区: 围绕end（相当于正链的start）
                promoter_start = max(0, end - PROMOTER_DOWNSTREAM)
                promoter_end = end + PROMOTER_UPSTREAM
                promoter_raw = str(genome[chrom][promoter_start:promoter_end])
                promoter_seq = self.reverse_complement(promoter_raw)

                # 终止子区: 围绕start（相当于正链的end）
                terminator_start = max(0, start - TERMINATOR_DOWNSTREAM)
                terminator_end = start + TERMINATOR_UPSTREAM
                terminator_raw = str(genome[chrom][terminator_start:terminator_end])
                terminator_seq = self.reverse_complement(terminator_raw)

                # 按生物学顺序拼接: [启动子] + [终止子]
                sequence = promoter_seq + terminator_seq

            # 检查序列长度并填充/截断
            expected_length = TOTAL_LENGTH
            if len(sequence) < expected_length:
                sequence = sequence.ljust(expected_length, 'N')
            elif len(sequence) > expected_length:
                sequence = sequence[:expected_length]

            return sequence.upper()

        except Exception as e:
            print(f"⚠️ 提取序列失败: {gene_info['gene_id']}, 染色体: {chrom}, 链: {strand}, 错误: {str(e)}")
            return 'N' * TOTAL_LENGTH

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

    def extract_all_sequences(self, genome_file, gene_df, overwrite=False):
        """
        提取所有基因序列
        """
        cache_file = os.path.join(self.output_dir, 'crayfish_sequences.pt')

        # 检查缓存
        if os.path.exists(cache_file) and not overwrite:
            print(f"📁 检查已预处理的序列: {cache_file}")
            try:
                data = torch.load(cache_file, map_location='cpu')
                print(f"✅ 成功加载预处理序列，包含 {len(data['target_genes'])} 个基因")
                print(f"   序列形状: {data['sequences'].shape}")
                return data['sequences'], data['gene_info'], data['target_genes']
            except Exception as e:
                print(f"⚠️ 加载缓存文件失败: {e}，重新预处理")

        print(f"🔄 开始提取小龙虾基因序列...")
        print(f"   启动子区: {PROMOTER_UPSTREAM}+{PROMOTER_DOWNSTREAM}bp")
        print(f"   终止子区: {TERMINATOR_UPSTREAM}+{TERMINATOR_DOWNSTREAM}bp")
        print(f"   总长度: {TOTAL_LENGTH}bp")

        # 加载基因组
        print(f"📂 加载基因组: {genome_file}")
        if not os.path.exists(genome_file):
            print(f"❌ 基因组文件不存在: {genome_file}")
            return None, None, None

        try:
            genome = Fasta(genome_file, as_raw=True, sequence_always_upper=True, read_ahead=10000)
            print(f"✅ 基因组加载成功")
        except Exception as e:
            print(f"❌ 加载基因组失败: {e}")
            return None, None, None

        # 提取序列
        sequences = []
        gene_info_list = []
        success_count = 0
        failed_count = 0
        failed_genes = []

        # 创建基因ID到行的映射
        gene_id_to_row = {row['gene_id']: row for _, row in gene_df.iterrows()}

        for gene_id in tqdm(gene_df['gene_id'].tolist(), desc="提取序列"):
            if gene_id in gene_id_to_row:
                row = gene_id_to_row[gene_id]
                sequence = self.extract_gene_sequence(genome, row)

                if sequence and len(sequence) == TOTAL_LENGTH and 'N' * TOTAL_LENGTH != sequence:
                    tensor = self.sequence_to_tensor(sequence)
                    sequences.append(tensor)
                    gene_info_list.append({
                        'gene_id': row['gene_id'],
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
                        print(f"   ⚠️ 序列异常: {gene_id}")
            else:
                failed_count += 1
                failed_genes.append(gene_id)

        print(f"\n✅ 成功提取 {success_count} 个基因序列")
        print(f"❌ 失败 {failed_count} 个基因序列")

        if sequences:
            all_sequences = torch.stack(sequences)
            print(f"✅ 序列张量形状: {all_sequences.shape}")

            # 保存数据
            data_to_save = {
                'sequences': all_sequences,
                'gene_info': gene_info_list,
                'target_genes': [info['gene_id'] for info in gene_info_list],
                'species': 'crayfish',
                'promoter_upstream': PROMOTER_UPSTREAM,
                'promoter_downstream': PROMOTER_DOWNSTREAM,
                'terminator_upstream': TERMINATOR_UPSTREAM,
                'terminator_downstream': TERMINATOR_DOWNSTREAM,
                'total_length': TOTAL_LENGTH,
                'success_count': success_count,
                'failed_count': failed_count,
                'timestamp': datetime.now().isoformat()
            }

            torch.save(data_to_save, cache_file)
            print(f"💾 序列已保存至: {cache_file}")

            # 同时保存为FASTA格式方便查看
            fasta_file = os.path.join(self.output_dir, 'crayfish_sequences.fasta')
            with open(fasta_file, 'w') as f:
                for info, seq_tensor in zip(gene_info_list, sequences):
                    # 将tensor转回序列字符串
                    seq_str = ''
                    base_map = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
                    for i in range(seq_tensor.shape[1]):
                        for base_idx in range(4):
                            if seq_tensor[base_idx, i] == 1:
                                seq_str += base_map[base_idx]
                                break
                        else:
                            seq_str += 'N'
                    f.write(f">{info['gene_id']}\n{seq_str}\n")
            print(f"💾 FASTA格式已保存至: {fasta_file}")

            # 保存统计信息
            stats_file = os.path.join(self.output_dir, 'extraction_stats.json')
            with open(stats_file, 'w') as f:
                json.dump({
                    'total_genes': len(gene_df),
                    'success_count': success_count,
                    'failed_count': failed_count,
                    'failed_genes': failed_genes[:50],
                    'config': {
                        'promoter_upstream': PROMOTER_UPSTREAM,
                        'promoter_downstream': PROMOTER_DOWNSTREAM,
                        'terminator_upstream': TERMINATOR_UPSTREAM,
                        'terminator_downstream': TERMINATOR_DOWNSTREAM,
                        'total_length': TOTAL_LENGTH
                    },
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            print(f"📊 统计信息已保存至: {stats_file}")

            return all_sequences, gene_info_list, data_to_save['target_genes']

        return None, None, None


# =============================================================================================
#  主函数
# =============================================================================================

def main():
    parser = argparse.ArgumentParser(description='小龙虾基因序列提取工具')
    parser.add_argument('--anno_file', type=str, default=GENE_MODEL_FILE,
                        help='基因注释文件 (anno.summary.xls)')
    parser.add_argument('--genome_file', type=str, default=GENOME_FILE,
                        help='基因组文件 (ref.fa)')
    parser.add_argument('--label_file', type=str, default=LABEL_FILE,
                        help='表达量标签文件 (crayfish_labels.csv)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='输出目录 (默认: processed_sequence)')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的缓存文件')

    args = parser.parse_args()

    print("=" * 60)
    print("小龙虾基因序列提取工具")
    print("=" * 60)
    print(f"序列配置:")
    print(
        f"  启动子区: 上游{PROMOTER_UPSTREAM}bp + 下游{PROMOTER_DOWNSTREAM}bp = {PROMOTER_UPSTREAM + PROMOTER_DOWNSTREAM}bp")
    print(
        f"  终止子区: 上游{TERMINATOR_UPSTREAM}bp + 下游{TERMINATOR_DOWNSTREAM}bp = {TERMINATOR_UPSTREAM + TERMINATOR_DOWNSTREAM}bp")
    print(f"  总长度: {TOTAL_LENGTH}bp")
    print(f"  提取逻辑: 使用start和end代替TSS/TTS")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    # 更新输出目录
    if args.output_dir != OUTPUT_DIR:
        os.makedirs(args.output_dir, exist_ok=True)

    # 检查文件是否存在
    for file_path, desc in [
        (args.anno_file, "注释文件"),
        (args.genome_file, "基因组文件"),
        (args.label_file, "标签文件")
    ]:
        if not os.path.exists(file_path):
            print(f"❌ {desc}不存在: {file_path}")
            sys.exit(1)

    # 加载标签数据
    target_genes, labels_df = load_crayfish_labels(args.label_file)
    if target_genes is None:
        print("❌ 加载标签数据失败")
        sys.exit(1)

    # 解析注释文件
    gene_df = parse_crayfish_annotation(args.anno_file, target_genes)
    if gene_df is None or len(gene_df) == 0:
        print("❌ 解析注释文件失败")
        sys.exit(1)

    # 提取序列
    extractor = CrayfishSequenceExtractor(args.output_dir)
    sequences, gene_info, target_genes_out = extractor.extract_all_sequences(
        args.genome_file, gene_df, args.overwrite
    )

    # 打印总结
    print(f"\n{'=' * 60}")
    print("📋 序列提取总结")
    print(f"{'=' * 60}")

    if sequences is not None:
        print(f"\n✅ 成功提取基因数: {len(target_genes_out)}")
        print(f"   序列张量形状: {sequences.shape}")
        print(f"   输出目录: {args.output_dir}")
        print(f"\n🎉 序列提取完成!")
    else:
        print("❌ 未成功提取任何序列")

    print("=" * 60)


if __name__ == "__main__":
    main()