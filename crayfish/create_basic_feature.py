import pandas as pd
import numpy as np
from pyfaidx import Fasta
import os
from tqdm import tqdm

# ================= 配置区 =================
CONFIG = {
    'genome_fa': 'ref.fa',
    'anno_file': 'anno.summary.xls',
    'index_file': 'gene_id_index.txt',
    'output_file': 'generated_features/crayfish_basic_features.csv',
    'upstream': 2000,
    'downstream': 500
}

os.makedirs(os.path.dirname(CONFIG['output_file']), exist_ok=True)


def get_gc_content(seq):
    if not seq: return 0.0
    s = str(seq).upper()
    return (s.count('G') + s.count('C')) / len(s) if len(s) > 0 else 0.0


def get_cpg_density(seq):
    if not seq: return 0.0
    s = str(seq).upper()
    return (s.count('CG') / len(s)) * 100 if len(s) > 0 else 0.0


def extract_features_v12_with_audit():
    print(f"🚀 启动 V12 特征提取与深度审计系统...")

    # 1. 加载数据
    with open(CONFIG['index_file'], 'r') as f:
        indexed_genes = [line.strip() for line in f]
    fasta = Fasta(CONFIG['genome_fa'], sequence_always_upper=True)
    anno_df = pd.read_csv(CONFIG['anno_file'], sep='\t', usecols=[0])

    # 2. 坐标解析与链判定审计
    coords_lookup = {}
    strand_stats = {'+': 0, '-': 0}

    for raw_val in tqdm(anno_df.iloc[:, 0], desc="🔍 解析坐标"):
        parts = str(raw_val).split(':')
        if len(parts) >= 5:
            clean_id = parts[0].replace('gene-', '').split('.')[0]
            try:
                raw_span = parts[3]
                if '..' in raw_span:
                    s_coord, e_coord = int(raw_span.split('..')[0]), int(raw_span.split('..')[1])
                else:
                    s_coord, e_coord = int(parts[3]), int(parts[4])
                # 核心逻辑：Start > End 判定为负链
                if s_coord > e_coord:
                    actual_start, actual_end, strand = e_coord, s_coord, '-'
                else:
                    actual_start, actual_end, strand = s_coord, e_coord, '+'

                strand_stats[strand] += 1
                coords_lookup[clean_id] = {
                    'chrom': parts[2], 'start': actual_start,
                    'end': actual_end, 'strand': strand
                }
            except ValueError:
                continue

    # 3. 特征提取与异常检测
    results = []
    found_count = 0
    missing_count = 0
    error_count = 0

    for gid in tqdm(indexed_genes, desc="🧬 提取特征"):
        res = {
            'gene_id': gid,
            'strand': 'N/A',
            'GC_promoter': 0.0, 'CpG_promoter': 0.0,
            'GC_terminator': 0.0, 'CpG_terminator': 0.0,
            'gene_length': 0
        }

        match_id = gid.replace('gene-', '').split('.')[0]

        if match_id in coords_lookup:
            found_count += 1
            info = coords_lookup[match_id]
            chrom, start, end, strand = info['chrom'], info['start'], info['end'], info['strand']
            res['gene_length'] = end - start
            res['strand'] = strand

            try:
                if chrom in fasta:
                    c_len = len(fasta[chrom])
                    # 根据链方向确定 TSS/TES 对应的上游/下游
                    if strand == '+':
                        p_range = (max(0, start - CONFIG['upstream']), start)
                        t_range = (end, min(c_len, end + CONFIG['downstream']))
                    else:
                        p_range = (end, min(c_len, end + CONFIG['upstream']))
                        t_range = (max(0, start - CONFIG['downstream']), start)

                    p_seq = fasta[chrom][p_range[0]:p_range[1]].seq
                    t_seq = fasta[chrom][t_range[0]:t_range[1]].seq

                    res['GC_promoter'] = get_gc_content(p_seq)
                    res['CpG_promoter'] = get_cpg_density(p_seq)
                    res['GC_terminator'] = get_gc_content(t_seq)
                    res['CpG_terminator'] = get_cpg_density(t_seq)
                else:
                    error_count += 1
            except Exception:
                error_count += 1
        else:
            missing_count += 1

        results.append(res)

    # 4. 生成报告
    final_df = pd.DataFrame(results)
    final_df.to_csv(CONFIG['output_file'], index=False)

    print("\n" + "=" * 60)
    print("📊 小龙虾基因特征深度审计报告")
    print("=" * 60)
    print(f"1. 总体概况:")
    print(f"   - 目标基因总数 (Index): {len(indexed_genes)}")
    print(f"   - 成功匹配坐标基因: {found_count} ({(found_count / len(indexed_genes)) * 100:.2f}%)")
    print(f"   - 缺失坐标基因 (如MSTRG): {missing_count} ({(missing_count / len(indexed_genes)) * 100:.2f}%)")

    print(f"\n2. 链方向分布 (基于Anno表原始解析):")
    print(f"   - 正链 (+): {strand_stats['+']}")
    print(f"   - 负链 (-): {strand_stats['-']}")

    print(f"\n3. 特征质量评估:")
    print(f"   - 启动子 GC 均值: {final_df[final_df['gene_length'] > 0]['GC_promoter'].mean():.4f}")
    print(f"   - 终止子 GC 均值: {final_df[final_df['gene_length'] > 0]['GC_terminator'].mean():.4f}")
    print(f"   - 平均基因长度: {final_df[final_df['gene_length'] > 0]['gene_length'].mean():.2f} bp")

    if error_count > 0:
        print(f"\n⚠️ 警告：检测到 {error_count} 条记录存在基因组越界或序列读取异常。")
    print(f"\n💾 最终特征矩阵已保存至: {CONFIG['output_file']}")
    print("=" * 60)


if __name__ == "__main__":
    extract_features_v12_with_audit()