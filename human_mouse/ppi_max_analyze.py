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
        'tax_id': '9606',
        'mart_file': 'mart_export_human.txt',
        'seq_file': 'precomputed_sequences/human_sequences.pt',
        'string_links': '9606.protein.links.v12.0.min700.onlyAB.txt.gz',
    },
    'mouse': {
        'tax_id': '10090',
        'mart_file': 'mart_export_mouse.txt',
        'seq_file': 'precomputed_sequences/mouse_sequences.pt',
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


def build_ppi_with_top_neighbor_audit(sp, cfg):
    print(f"\n{'=' * 95}\n🚀 启动 {sp.upper()} PPI 构建与“最强邻居”失踪审计\n{'=' * 95}")

    # 1. 加载序列文件基因池 (基准)
    seq_data = torch.load(cfg['seq_file'], map_location='cpu')
    seq_genes_raw = seq_data['target_genes']
    seq_genes_clean = {get_clean_id(g) for g in seq_genes_raw}
    gene_to_idx = {get_clean_id(g): i for i, g in enumerate(seq_genes_raw)}

    # 2. 构建 ID 映射桥梁 (Protein_ID -> Gene_ID)
    # 我们需要通过 BioMart 知道 STRING 里的蛋白对应哪个基因 ID
    prot_to_gene = {}
    print(f"📖 正在建立 Protein <-> Gene 映射桥梁...")
    mart_df = pd.read_csv(cfg['mart_file'], sep='\t')
    for _, row in mart_df.iterrows():
        gid, pid = get_clean_id(row['Gene stable ID']), get_clean_id(row['Protein stable ID'])
        if pid: prot_to_gene[pid] = gid

    # 3. 扫描 STRING Links 寻找每个基因的最强邻居
    # key: 序列文件中的基因ID, value: (最高得分, 该邻居的基因ID)
    top_neighbor_map = {}

    print(f"🔗 正在扫描 STRING 连边并锁定最强邻居...")
    with gzip.open(cfg['string_links'], 'rt') as f:
        next(f)  # 跳过表头
        for line in f:
            p1_raw, p2_raw, score = line.strip().split()
            cp1, cp2 = get_clean_id(p1_raw), get_clean_id(p2_raw)
            score = int(score)

            # 获取对应的基因 ID
            g1 = prot_to_gene.get(cp1)
            g2 = prot_to_gene.get(cp2)

            # 审计逻辑：如果 g1 在我们的序列文件中，记录它的最强邻居 g2
            if g1 in seq_genes_clean and g2:
                if g1 not in top_neighbor_map or score > top_neighbor_map[g1][0]:
                    top_neighbor_map[g1] = (score, g2)

            # 反向同理
            if g2 in seq_genes_clean and g1:
                if g2 not in top_neighbor_map or score > top_neighbor_map[g2][0]:
                    top_neighbor_map[g2] = (score, g1)

    # 4. 执行审计统计
    total_genes_with_ppi = len(top_neighbor_map)
    top_neighbor_in_seq = 0
    top_neighbor_missing = 0
    missing_details = []  # 记录缺失的例子

    for g_self, (score, g_neighbor) in top_neighbor_map.items():
        if g_neighbor in seq_genes_clean:
            top_neighbor_in_seq += 1
        else:
            top_neighbor_missing += 1
            if len(missing_details) < 10:
                missing_details.append((g_self, g_neighbor, score))

    # 5. 输出报告
    print(f"\n📊 {sp.upper()} “最强邻居”匹配审计报告:")
    print("-" * 75)
    print(f"1. 序列文件中成功匹配到 PPI 数据的基因数: {total_genes_with_ppi}")
    print(
        f"2. 其“最高系数邻居”也在序列文件中的数量:   {top_neighbor_in_seq} ({top_neighbor_in_seq / total_genes_with_ppi:.2%})")
    print(
        f"3. 其“最高系数邻居”在序列文件中失踪的数量: {top_neighbor_missing} ({top_neighbor_missing / total_genes_with_ppi:.2%})")
    print("-" * 75)

    if missing_details:
        print(f"⚠️ 典型缺失案例 (当前基因 -> 丢失的最强邻居 | 置信度):")
        for g_self, g_miss, score in missing_details:
            print(f"   - {g_self} -> {g_miss} (Score: {score})")
    print("-" * 75)


if __name__ == "__main__":
    for sp in ['human', 'mouse']:
        build_ppi_with_top_neighbor_audit(sp, CONFIG[sp])