import pandas as pd
import torch
import os
import re
from collections import defaultdict
from datetime import datetime  # 新增：用于记录保存时间

# ================= 配置区 =================
CONFIG = {
    'human': {
        'mart_file': 'mart_export_human.txt',
        # 修改为 NT 版序列文件路径，确保索引对齐
        'seq_file': 'precomputed_sequences_NT/human_sequences_NT_6kb.pt',
        'gtf_file': 'gencode.v49.primary_assembly.basic.annotation.gtf',
        'source_files': ['tp_data/human.source', 'tp_data/ENCODE'],
        'node_map': 'tp_data/human.node'
    },
    'mouse': {
        'mart_file': 'mart_export_mouse.txt',
        # 修改为 NT 版序列文件路径，确保索引对齐
        'seq_file': 'precomputed_sequences_NT/mouse_sequences_NT_6kb.pt',
        'gtf_file': 'gencode.vM38.primary_assembly.basic.annotation.gtf',
        'source_files': ['tp_data/mouse.source'],
        'node_map': 'tp_data/mouse.node'
    }
}

OUTPUT_DIR = "processed_tf"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_clean_id(id_str):
    if pd.isna(id_str) or str(id_str).lower() == 'nan': return ""
    s = str(id_str).strip().upper()
    if re.match(r'^\d+\.', s): s = s.split('.', 1)[1]
    return s.split('.')[0]


def is_gene_symbol(symbol):
    """过滤非编码干扰项，确保节点纯净"""
    s = str(symbol).upper()
    forbidden = ['MIR', 'LET', 'SNOR', 'SNOA', 'RNA', 'MIRNA']
    return not any(f in s for f in forbidden)


def run_tf_comprehensive_audit():
    for sp, cfg in CONFIG.items():
        print(f"\n{'=' * 95}\n🚀 启动 {sp.upper()} TF 调控网络全量审计与构建\n{'=' * 95}")

        # 1. 加载官方 GTF 基因池
        gtf_genes = set()
        print(f"📖 扫描 GTF 官方基因库...")
        with open(cfg['gtf_file'], 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                if 'gene_id "' in line:
                    gtf_genes.add(get_clean_id(line.split('gene_id "')[1].split('"')[0]))

        # 2. 加载表达值序列基因池
        seq_genes_raw = []  # 新增：保存原始 ID 列表用于字典存储
        seq_genes_set = set()
        gene_to_idx = {}
        if os.path.exists(cfg['seq_file']):
            seq_data = torch.load(cfg['seq_file'], map_location='cpu')
            seq_genes_raw = seq_data['target_genes']  # 获取标准顺序
            for i, g in enumerate(seq_genes_raw):
                gid = get_clean_id(g)
                seq_genes_set.add(gid)
                gene_to_idx[gid] = i
        print(f"📍 序列文件加载完成: {len(seq_genes_set):,} 个表达基因")

        # 3. 建立 ID 转换枢纽 (Entrez -> Ensembl)
        entrez_to_ensembl = {}
        mart_df = pd.read_csv(cfg['mart_file'], sep='\t')
        for _, row in mart_df.iterrows():
            ens = get_clean_id(row['Gene stable ID'])
            ent = str(row['NCBI gene (formerly Entrezgene) ID']).split('.')[0]
            if ens and ent != 'NAN': entrez_to_ensembl[ent] = ens

        # 4. 建立非基因过滤器 (处理异常行)
        bad_entrez = set()
        if os.path.exists(cfg['node_map']):
            try:
                node_df = pd.read_csv(cfg['node_map'], sep=r'\s+', engine='python', header=None, on_bad_lines='skip')
                for _, row in node_df.iterrows():
                    if len(row) >= 2 and not is_gene_symbol(row[1]):
                        bad_entrez.add(str(row[0]))
            except Exception:
                pass

        # 5. 扫描调控边并执行多维审计
        raw_total_lines = 0
        gtf_valid_edges = set()  # 双端均在 GTF 中
        training_valid_edges = []  # 双端均在序列中

        # 审计：表达基因在调控网中的邻居情况
        expressed_in_tf = set()
        expressed_with_gtf_neighbor = set()

        for src in cfg['source_files']:
            if not os.path.exists(src): continue
            print(f"🔗 正在解析源文件: {src}")
            with open(src, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    raw_total_lines += 1

                    e1, e2 = (parts[1], parts[3]) if len(parts) >= 4 else (parts[0], parts[1])
                    if e1 in bad_entrez or e2 in bad_entrez: continue

                    ens1, ens2 = entrez_to_ensembl.get(e1), entrez_to_ensembl.get(e2)
                    if not ens1 or not ens2: continue

                    # A. 官方有效审计 (双端均在 GTF)
                    if ens1 in gtf_genes and ens2 in gtf_genes:
                        gtf_valid_edges.add((ens1, ens2))

                        # 邻居生存统计
                        if ens1 in seq_genes_set:
                            expressed_in_tf.add(ens1)
                            expressed_with_gtf_neighbor.add(ens1)
                        if ens2 in seq_genes_set:
                            expressed_in_tf.add(ens2)
                            expressed_with_gtf_neighbor.add(ens2)

                        # B. 训练有效审计 (双端均在序列)
                        if ens1 in seq_genes_set and ens2 in seq_genes_set:
                            training_valid_edges.append([gene_to_idx[ens1], gene_to_idx[ens2]])

        # 6. 最终指标报告 (严格匹配用户格式)
        print(f"\n📊 {sp.upper()} TF 网络核心指标报告:")
        print("-" * 80)
        print(f"1. 原始文件总行数:               {raw_total_lines:,}")
        print(f"2. [官方有效] 双端均在 GTF 中的基因边: {len(gtf_valid_edges):,}")
        print(f"3. [训练有效] 双端均在序列中的边:      {len(training_valid_edges):,}")

        utilization = (len(training_valid_edges) / len(gtf_valid_edges)) * 100 if gtf_valid_edges else 0
        print(f"4. 网络数据利用率:               {utilization:.2f}%")
        print("-" * 80)

        # 额外的表达基因审计
        print(f"📈 表达基因邻居生存状态:")
        print(f"   - 在 TF 网络中出现的表达基因数: {len(expressed_in_tf):,}")
        print(f"   - 至少有一个邻居在 GTF 中的比例: {len(expressed_with_gtf_neighbor) / len(seq_genes_set):.2%}")
        print("-" * 80)

        if training_valid_edges:
            edge_index = torch.unique(torch.tensor(training_valid_edges, dtype=torch.long).t().contiguous(), dim=1)
            save_path = f"{OUTPUT_DIR}/{sp}_tf_edge_index.pt"

            # 修改保存逻辑：以字典形式保存 edge_index 和完整的 ID 列表
            save_dict = {
                'edge_index': edge_index,
                'gene_ids': seq_genes_raw,  # 保存原始 target_genes 以便后续重索引
                'species': sp,
                'timestamp': datetime.now().isoformat()
            }
            torch.save(save_dict, save_path)
            print(f"✅ 最终 TF 字典已保存至: {save_path}")


if __name__ == "__main__":
    run_tf_comprehensive_audit()