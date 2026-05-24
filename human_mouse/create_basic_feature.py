import pandas as pd
import pyranges as pr
from pyfaidx import Fasta
import warnings
import os
import argparse
from tqdm import tqdm
from collections import Counter

# 抑制警告
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.display.width = 0

# =============================================================================================
#  物种配置（去掉.gz后缀）
# =============================================================================================

SPECIES_CONFIG = {
    'human': {
        'name': 'Homo_sapiens',
        'gene_model': 'gencode.v49.primary_assembly.basic.annotation.gtf',  # 已解压
        'genome': 'GRCh38.primary_assembly.genome.fa',
        'expression': 'GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.gct.gz',
        'has_version_in_expression': True,
        'expression_format': 'gct',
    },
    'mouse': {
        'name': 'Mus_musculus',
        'gene_model': 'gencode.vM38.primary_assembly.basic.annotation.gtf',  # 已解压
        'genome': 'GRCm39.primary_assembly.genome.fa',
        'expression': 'E-GEOD-70484-query-results.tpmss.tsv',
        'has_version_in_expression': False,
        'expression_format': 'tsv',
    }
}

# 序列长度配置
UPSTREAM_LENGTH = 1000
DOWNSTREAM_LENGTH = 500


# =============================================================================================
#  文件处理辅助函数（简化）
# =============================================================================================

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if not os.path.exists(filepath):
        print(f"❌ {description}不存在: {filepath}")
        return False
    print(f"✅ {description}: {filepath}")
    return True


def read_gct_ids_only(gct_file):
    """只读取GCT文件的ID列"""
    print(f"读取GCT文件ID列: {gct_file}")

    # 处理可能带.gz的GCT文件
    compression = 'gzip' if gct_file.endswith('.gz') else None
    skiprows = 2  # GCT格式跳过前两行

    df = pd.read_csv(
        gct_file,
        sep='\t',
        skiprows=skiprows,
        usecols=[0],
        names=['gene_id'],
        compression=compression
    )

    df = df.dropna()
    print(f"成功读取 {len(df)} 个基因ID")
    return df


def read_tsv_ids_only(tsv_file):
    """只读取TSV文件的基因ID列"""
    print(f"读取TSV文件ID列: {tsv_file}")

    # 探测注释行数
    skip_rows = 0
    compression = 'gzip' if tsv_file.endswith('.gz') else None

    if compression:
        import gzip
        f = gzip.open(tsv_file, 'rt')
    else:
        f = open(tsv_file, 'r')

    with f:
        first_line = f.readline().strip()
        while first_line.startswith('#'):
            skip_rows += 1
            first_line = f.readline().strip()

    # 找ID列
    headers = first_line.split('\t')
    id_col_idx = 0
    possible_id_cols = ['Gene ID', 'GeneID', 'gene_id', 'Gene', 'gene']

    for i, col in enumerate(headers):
        if col in possible_id_cols:
            id_col_idx = i
            break

    df = pd.read_csv(
        tsv_file,
        sep='\t',
        skiprows=skip_rows,
        usecols=[id_col_idx],
        names=['gene_id'],
        header=0,
        compression=compression,
        dtype=str
    )

    df = df.dropna().drop_duplicates()
    print(f"成功读取 {len(df)} 个基因ID")
    return df


def load_target_genes(expression_file, species_config):
    """从表达量文件中加载目标基因"""
    print(f"\n加载表达量文件: {expression_file}")

    expr_format = species_config.get('expression_format', 'tsv')
    has_version = species_config.get('has_version_in_expression', True)

    if expr_format == 'gct':
        df = read_gct_ids_only(expression_file)
    else:
        df = read_tsv_ids_only(expression_file)

    if not has_version:
        df['gene_id_base'] = df['gene_id'].apply(
            lambda x: str(x).split('.')[0] if '.' in str(x) else str(x)
        )
        target_genes = df['gene_id_base'].tolist()
        print(f"获取到 {len(target_genes)} 个目标基因 (无版本号)")
    else:
        target_genes = df['gene_id'].tolist()
        print(f"获取到 {len(target_genes)} 个目标基因 (保留版本号)")

    print(f"  示例ID: {target_genes[:5]}")
    return target_genes, df


# =============================================================================================
#  特征计算函数
# =============================================================================================

def gc(seq):
    """计算GC含量"""
    if not seq or len(seq) == 0:
        return 0.0
    seq = str(seq).upper()
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq)


def cpg_perc(seq):
    """计算CpG百分比"""
    if not seq or len(seq) == 0:
        return 0.0
    seq = str(seq).upper()
    cpg_counts = seq.count('CG')
    return (cpg_counts / len(seq)) * 100


# =============================================================================================
#  特征提取函数
# =============================================================================================

def extract_gene_features(genome_file, gtf_file, target_genes, species_config):
    """提取目标基因特征"""
    if not target_genes:
        return pd.DataFrame()

    has_version = species_config.get('has_version_in_expression', True)

    # 直接读取已解压的GTF
    print(f"读取GTF文件: {gtf_file}")
    df = pr.read_gtf(gtf_file, as_df=True)
    print(f"共 {len(df)} 条记录")

    # 建立ID映射
    if not has_version:
        df['match_id'] = df['gene_id'].apply(lambda x: str(x).split('.')[0])
    else:
        df['match_id'] = df['gene_id']

    target_set = set(target_genes)

    # 筛选记录
    mask = df['match_id'].isin(target_set)
    gene_df = df[mask & (df['Feature'] == 'gene')].copy()
    utr_raw = df[mask & (df['Feature'] == 'UTR')].copy()
    cds_raw = df[mask & (df['Feature'] == 'CDS')].copy()

    print(f"找到 {len(gene_df)} 个基因, {len(utr_raw)} 个UTR, {len(cds_raw)} 个CDS")

    # 预计算CDS边界
    cds_limits = {}
    if not cds_raw.empty:
        for gid, group in cds_raw.groupby('gene_id'):
            cds_limits[gid] = {
                'Start': int(group['Start'].min()),
                'End': int(group['End'].max())
            }

    # 加载基因组
    print("加载基因组...")
    fasta = Fasta(genome_file, as_raw=False, sequence_always_upper=True)

    results = []
    failed = 0

    for _, grow in tqdm(gene_df.iterrows(), total=len(gene_df), desc="提取特征"):
        gid = grow['gene_id']
        chrom = grow['Chromosome']
        strand = grow['Strand']
        start = int(grow['Start'])
        end = int(grow['End'])

        # 初始化结果
        res = {
            'gene_id': gid,
            'Chromosome': chrom,
            'Strand': strand,
            'GC_promoter': 0.0,
            'CpG_promoter': 0.0,
            'GC_terminator': 0.0,
            'CpG_terminator': 0.0,
            'UTR5_length': 0,
            'UTR5_GC': 0.0,
            'UTR3_length': 0,
            'UTR3_GC': 0.0
        }

        # 提取启动子和终止子序列
        try:
            chrom_len = len(fasta[chrom])

            if strand == '+':
                p_start = max(1, start - UPSTREAM_LENGTH)
                p_end = min(chrom_len, start + DOWNSTREAM_LENGTH)
                t_start = max(1, end - DOWNSTREAM_LENGTH)
                t_end = min(chrom_len, end + UPSTREAM_LENGTH)
            else:
                p_start = max(1, end - DOWNSTREAM_LENGTH)
                p_end = min(chrom_len, end + UPSTREAM_LENGTH)
                t_start = max(1, start - UPSTREAM_LENGTH)
                t_end = min(chrom_len, start + DOWNSTREAM_LENGTH)

            # 启动子
            p_seq = fasta[chrom][p_start - 1:p_end]
            p_seq_str = p_seq.reverse.complement.seq if strand == '-' else p_seq.seq
            res['GC_promoter'] = gc(p_seq_str)
            res['CpG_promoter'] = cpg_perc(p_seq_str)

            # 终止子
            t_seq = fasta[chrom][t_start - 1:t_end]
            t_seq_str = t_seq.reverse.complement.seq if strand == '-' else t_seq.seq
            res['GC_terminator'] = gc(t_seq_str)
            res['CpG_terminator'] = cpg_perc(t_seq_str)

        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"⚠️ {gid} 序列提取失败: {e}")

        # 处理UTR
        limits = cds_limits.get(gid)
        curr_utrs = utr_raw[utr_raw['gene_id'] == gid]

        if limits and not curr_utrs.empty:
            c_min = limits['Start']
            c_max = limits['End']

            for _, urow in curr_utrs.iterrows():
                try:
                    us = int(urow['Start'])
                    ue = int(urow['End'])
                    u_len = ue - us
                    u_mid = (us + ue) / 2

                    # 判断5'或3'UTR
                    if strand == '+':
                        is_5p = u_mid < c_min
                    else:
                        is_5p = u_mid > c_max

                    # 提取序列
                    u_seq = fasta[chrom][us - 1:ue]
                    u_seq_str = u_seq.reverse.complement.seq if strand == '-' else u_seq.seq

                    if is_5p and u_len > res['UTR5_length']:
                        res['UTR5_length'] = u_len
                        res['UTR5_GC'] = gc(u_seq_str)
                    elif not is_5p and u_len > res['UTR3_length']:
                        res['UTR3_length'] = u_len
                        res['UTR3_GC'] = gc(u_seq_str)

                except:
                    continue

        results.append(res)

    return pd.DataFrame(results)


# =============================================================================================
#  主函数
# =============================================================================================

def main():
    parser = argparse.ArgumentParser(description='提取目标基因的基础特征（人类/小鼠）')
    parser.add_argument('--species', type=str, required=True,
                        choices=['human', 'mouse', 'all'],
                        help='要处理的物种')
    parser.add_argument('--expression_file', type=str, default=None,
                        help='指定表达量文件（覆盖默认）')
    parser.add_argument('--output_dir', type=str, default='generated_features',
                        help='输出目录')
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
            print()
        return

    print(f"\n{'=' * 60}")
    print(f"开始提取基础特征...")
    print(f"物种: {args.species}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'=' * 60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    species_list = ['human', 'mouse'] if args.species == 'all' else [args.species]

    for species in species_list:
        print(f"\n{'=' * 50}")
        print(f"处理物种: {species}")
        print(f"{'=' * 50}")

        config = SPECIES_CONFIG[species]
        expression_file = args.expression_file if args.expression_file else config['expression']

        # 检查文件
        files_ok = all([
            check_file_exists(config['gene_model'], "基因模型"),
            check_file_exists(config['genome'], "基因组"),
            check_file_exists(expression_file, "表达量")
        ])

        if not files_ok:
            print(f"❌ 文件检查失败，跳过")
            continue

        # 加载目标基因
        target_genes, _ = load_target_genes(expression_file, config)
        if not target_genes:
            print(f"❌ 未找到目标基因，跳过")
            continue

        # 提取特征
        features_df = extract_gene_features(
            config['genome'],
            config['gene_model'],
            target_genes,
            config
        )

        if not features_df.empty:
            expr_name = os.path.basename(expression_file).replace('.gz', '').replace('.gct', '').replace('.tsv', '')
            output_file = os.path.join(args.output_dir, f"{species}_{expr_name}_features.csv")
            features_df.to_csv(output_file, index=False)

            print(f"\n✅ 成功提取 {len(features_df)} 个基因的特征")
            print(f"💾 保存至: {output_file}")

            # 简单统计
            print(f"\n📊 统计:")
            print(f"  有5'UTR: {(features_df['UTR5_length'] > 0).sum()}")
            print(f"  有3'UTR: {(features_df['UTR3_length'] > 0).sum()}")
        else:
            print(f"❌ 未提取到特征")

    print(f"\n🎉 完成!")


if __name__ == "__main__":
    main()