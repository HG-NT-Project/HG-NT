import pandas as pd
import torch
import os
import gzip
import re
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime  # 新增用于记录时间

# ================= 配置区 =================
CONFIG = {
    'human': {
        'tax_id': '9606',
        'mart_file': 'mart_export_human.txt',
        # 修改为 NT 版序列文件路径
        'seq_file': 'precomputed_sequences_NT/human_sequences_NT_6kb.pt',
        'gtf_file': 'gencode.v49.primary_assembly.basic.annotation.gtf',
        'string_links': '9606.protein.links.v12.0.min700.onlyAB.txt.gz',
    },
    'mouse': {
        'tax_id': '10090',
        'mart_file': 'mart_export_mouse.txt',
        # 修改为 NT 版序列文件路径
        'seq_file': 'precomputed_sequences_NT/mouse_sequences_NT_6kb.pt',
        'gtf_file': 'gencode.vM38.primary_assembly.basic.annotation.gtf',
        'string_links': '10090.protein.links.v12.0.min700.onlyAB.txt.gz',
    }
}

OUTPUT_DIR = "processed_ppi"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_clean_id(id_str):
    if pd.isna(id_str) or str(id_str).lower() == 'nan': return ""
    s = str(id_str).strip().upper()
    if re.match(r'^\d+\.', s): s = s.split('.', 1)[1]
    return s.split('.')[0]


def build_ppi_final(sp, cfg):
    print(f"\n{'=' * 95}\n🚀 启动 {sp.upper()} PPI 最终版对齐与深度利用率审计\n{'=' * 95}")

    # 1. 扫描 GTF 建立官方基因池
    gtf_genes = set()
    print(f"📖 扫描 GTF 文件...")
    with open(cfg['gtf_file'], 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            if 'gene_id "' in line:
                gtf_genes.add(get_clean_id(line.split('gene_id "')[1].split('"')[0]))

    # 2. 加载序列文件中的表达基因 (Index 基准)
    seq_data = torch.load(cfg['seq_file'], map_location='cpu')
    seq_genes_raw = seq_data['target_genes']
    gene_to_idx = {get_clean_id(g): i for i, g in enumerate(seq_genes_raw) if get_clean_id(g)}
    seq_genes_set = set(gene_to_idx.keys())

    # 3. 建立双路 Protein -> Gene 映射
    prot_to_gene = {}
    mart_df = pd.read_csv(cfg['mart_file'], sep='\t')
    for _, row in mart_df.iterrows():
        gid = get_clean_id(row['Gene stable ID'])
        if not gid: continue
        sid, pid = get_clean_id(row['STRING ID']), get_clean_id(row['Protein stable ID'])
        if sid: prot_to_gene[sid] = gid
        if pid: prot_to_gene[pid] = gid

    # 4. 扫描 STRING 连边并统计
    edge_list_final = []
    raw_links_count = 0
    mapped_links_count = 0

    # 用于统计：每个表达基因拥有多少个在 GTF 中的邻居
    # key: 表达基因ID, value: set(邻居基因ID)
    expressed_gene_neighbors = defaultdict(set)

    print(f"🔗 正在检索 STRING 连边并审计邻居状态...")
    with gzip.open(cfg['string_links'], 'rt') as f:
        next(f)
        for line in f:
            raw_links_count += 1
            p1_raw, p2_raw, _ = line.strip().split()
            g1, g2 = prot_to_gene.get(get_clean_id(p1_raw)), prot_to_gene.get(get_clean_id(p2_raw))

            if not g1 or not g2: continue
            mapped_links_count += 1

            # 统计邻居关系（只要 g1 在表达集中，且 g2 在 GTF 中，就算该表达基因有合法邻居）
            if g1 in seq_genes_set and g2 in gtf_genes:
                expressed_gene_neighbors[g1].add(g2)
            if g2 in seq_genes_set and g1 in gtf_genes:
                expressed_gene_neighbors[g2].add(g1)

            # 最终连边：只有当两个基因都在序列文件中时才保留
            if g1 in gene_to_idx and g2 in gene_to_idx:
                edge_list_final.append([gene_to_idx[g1], gene_to_idx[g2]])

    # 5. 计算统计指标
    genes_in_ppi = len(expressed_gene_neighbors)  # 在 PPI 网络中出现的表达基因
    genes_with_valid_neighbor = sum(1 for g in expressed_gene_neighbors if len(expressed_gene_neighbors[g]) > 0)

    # 6. 打印最终审计报告
    print(f"\n📊 {sp.upper()} 最终审计报告:")
    print("-" * 85)
    print(f"1. [基因池基数] 序列文件(表达基因): {len(seq_genes_set):,} | GTF官方库: {len(gtf_genes):,}")
    print(
        f"2. [表达基因入网] 序列文件中在 PPI 网络中出现的基因数: {genes_in_ppi:,} ({genes_in_ppi / len(seq_genes_set):.2%})")
    print(f"3. [邻居生存审计] 至少拥有一个 GTF 合法邻居的表达基因数: {genes_with_valid_neighbor:,}")
    print("-" * 85)
    print(f"📈 [PPI 网络利用率统计]:")
    print(f"   - 原始文件总连边: {raw_links_count:,}")
    print(f"   - 映射成功连边:   {mapped_links_count:,} (占原始数据 {mapped_links_count / raw_links_count:.2%})")
    print(f"   - 最终利用连边:   {len(edge_list_final):,} (占原始数据 {len(edge_list_final) / raw_links_count:.2%})")
    print("-" * 85)

    if edge_list_final:
        edge_index = torch.unique(torch.tensor(edge_list_final, dtype=torch.long).t().contiguous(), dim=1)
        save_path = f"{OUTPUT_DIR}/{sp}_ppi_edge_index.pt"

        # 修改保存逻辑：以字典形式保存 edge_index 和完整的 ID 列表
        save_dict = {
            'edge_index': edge_index,
            'gene_ids': seq_genes_raw,  # 保存原始 target_genes 以便后续重索引
            'species': sp,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(save_dict, save_path)
        print(f"✅ 最终 PPI 字典已保存至: {save_path}")
    else:
        print(f"⚠️ {sp.upper()} 未能构建有效连边。")


if __name__ == "__main__":
    for sp in ['human', 'mouse']:
        build_ppi_final(sp, CONFIG[sp])