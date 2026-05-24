import pandas as pd
import torch
import os
import gzip
import re
from tqdm import tqdm
from collections import defaultdict

# ================= 配置区 =================
CONFIG = {
    'human': {
        'mart_file': 'mart_export_human.txt',
        'seq_file': 'precomputed_sequences_NT/human_sequences_NT_6kb.pt',
        'gtf_file': 'gencode.v49.primary_assembly.basic.annotation.gtf',
        'string_links': '9606.protein.links.v12.0.min700.onlyAB.txt.gz',
    }
}


def get_clean_id(id_str):
    if pd.isna(id_str) or str(id_str).lower() == 'nan': return ""
    s = str(id_str).strip().upper()
    if re.match(r'^\d+\.', s): s = s.split('.', 1)[1]
    return s.split('.')[0]


def build_ppi_statistics(sp, cfg):
    print(f"\n{'=' * 95}\n📊 开始统计 {sp.upper()} PPI 有效对齐率 (基于模糊匹配逻辑)\n{'=' * 95}")

    # 1. 扫描 GTF 建立官方物理位置库 (Clean ID -> Full ID)
    gtf_clean_pool = set()
    print(f"📖 正在读取 GTF 文件...")
    with open(cfg['gtf_file'], 'r') as f:
        for line in f:
            if 'gene_id "' in line:
                raw_id = line.split('gene_id "')[1].split('"')[0]
                gtf_clean_pool.add(get_clean_id(raw_id))
    print(f"✅ GTF 中共有 {len(gtf_clean_pool):,} 个唯一基因 (Clean ID)")

    # 2. 加载序列文件中的表达基因
    print(f"📂 加载序列文件...")
    seq_data = torch.load(cfg['seq_file'], map_location='cpu')
    seq_genes_raw = seq_data['target_genes']
    seq_clean_pool = {get_clean_id(g) for g in seq_genes_raw if get_clean_id(g)}
    total_seq_genes = len(seq_clean_pool)
    print(f"✅ 序列库中共有 {total_seq_genes:,} 个有效基因 (Clean ID)")

    # 3. 建立 Protein -> Clean Gene 映射
    print(f"🧪 解析 BioMart 映射表...")
    prot_to_clean_gene = {}
    mart_df = pd.read_csv(cfg['mart_file'], sep='\t')
    for _, row in mart_df.iterrows():
        gid = get_clean_id(row['Gene stable ID'])
        sid = get_clean_id(row['STRING ID'])
        if sid and gid:
            prot_to_clean_gene[sid] = gid

    # 4. 扫描 STRING 连边并应用过滤逻辑
    # 逻辑：
    # A. 基因 A 必须在我们的 seq_file 中
    # B. 基因 B 必须在 GTF 文件中（保证能提取序列）

    expressed_gene_with_neighbor = set()
    total_valid_edges = 0

    print(f"🔗 正在扫描 STRING PPI 网络...")
    with gzip.open(cfg['string_links'], 'rt') as f:
        next(f)  # 跳过表头
        for line in tqdm(f, desc="处理连边", unit="条"):
            p1_raw, p2_raw, _ = line.strip().split()
            g1_c = prot_to_clean_gene.get(get_clean_id(p1_raw))
            g2_c = prot_to_clean_gene.get(get_clean_id(p2_raw))

            if not g1_c or not g2_c:
                continue

            # 审计核心逻辑：
            # 如果 g1 在我们的序列库里，且它的邻居 g2 在 GTF 里有物理位置
            if g1_c in seq_clean_pool and g2_c in gtf_clean_pool:
                expressed_gene_with_neighbor.add(g1_c)
                total_valid_edges += 1

            # 对称性检查：如果 g2 在序列库，g1 在 GTF
            if g2_c in seq_clean_pool and g1_c in gtf_clean_pool:
                expressed_gene_with_neighbor.add(g2_c)
                total_valid_edges += 1

    # 5. 计算最终利用率
    final_count = len(expressed_gene_with_neighbor)
    coverage_rate = (final_count / total_seq_genes) * 100

    print(f"\n" + "=" * 60)
    print(f"📈 {sp.upper()} 最终统计结论")
    print("-" * 60)
    print(f"1. 序列库 (seq_file) 总基因数:      {total_seq_genes:,}")
    print(f"2. 成功找到 PPI 邻居的基因数:       {final_count:,}")
    print(f"   (且邻居基因在 GTF 中有物理位置)")
    print(f"3. 最终有效覆盖率 (可生成对数据):    {coverage_rate:.2f}%")
    print(f"4. 累计可用 PPI 连边总数:           {total_valid_edges:,}")
    print("=" * 60)
    print(f"💡 结论：该逻辑下，你可以为 {coverage_rate:.2f}% 的表达基因构建“基因+邻居”的 6kb 序列对。")


if __name__ == "__main__":
    build_ppi_statistics('human', CONFIG['human'])