import pandas as pd
import re
import os


def high_precision_tf_audit_v4(meme_path, anno_path, index_path, output_dir="processed_tf"):
    print(f"🚀 启动高精度 TF-ID 审计 (V4) - 正在精准提取 Motif...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载基准索引
    with open(index_path, 'r') as f:
        indexed_genes = [line.strip() for line in f]
    gene_to_idx = {gid: i for i, gid in enumerate(indexed_genes)}

    # 2. 解析 MEME 基础信息
    motif_metadata = []
    BLACK_LIST = {'SV', 'D', 'BS', 'PAN', 'TX', 'DNA'}  # 增加常见噪音

    with open(meme_path, 'r') as f:
        for line in f:
            if line.startswith("MOTIF"):
                parts = line.split()
                if len(parts) >= 3:
                    symbol = parts[2].upper()
                    # 长度校验，避免过短的 Symbol 导致泛滥匹配
                    if symbol not in BLACK_LIST and len(symbol) > 1:
                        motif_metadata.append({'Motif_ID': parts[1], 'Symbol': symbol})

    m_df = pd.DataFrame(motif_metadata)

    # 3. 严格映射逻辑
    anno_df = pd.read_csv(anno_path, sep='\t')
    final_mapping = []

    for _, m_row in m_df.iterrows():
        symbol = m_row['Symbol']
        pattern = rf'\b{re.escape(symbol)}\b'

        # 必须同时包含 Symbol 和调控相关关键词
        matches = anno_df[
            (anno_df['NRANNO'].str.contains(pattern, case=False, na=False)) &
            (anno_df['NRANNO'].str.contains('factor|protein|zinc|binding|domain|transcription', case=False, na=False))
            ]

        for _, a_row in matches.iterrows():
            gid = str(a_row['GeneID']).split(':')[0]
            if gid in gene_to_idx:
                final_mapping.append({
                    'Node_Index': gene_to_idx[gid],
                    'LXC_GeneID': gid,
                    'Motif_ID': m_row['Motif_ID'],
                    'Motif_Symbol': symbol
                })

    bridge_df = pd.DataFrame(final_mapping).drop_duplicates()

    # 4. 提取唯一的 Motif ID 列表
    if bridge_df.empty or 'Motif_ID' not in bridge_df.columns:
        save_path = os.path.join(output_dir, "final_tf_motif_bridge_v3.csv")
        bridge_df.to_csv(save_path, index=False)
        print("\n⚠️ 未找到有效 TF-Motif 映射，已输出空桥接文件。")
        return []

    unique_motifs = bridge_df['Motif_ID'].unique().tolist()

    # 保存桥梁文件
    save_path = os.path.join(output_dir, "final_tf_motif_bridge_v3.csv")
    bridge_df.to_csv(save_path, index=False)

    # 5. 生成精准 FIMO 命令
    # 将 27 个 ID 拼接到 --motif 参数后面
    motif_args = " ".join([f"--motif {m}" for m in unique_motifs])
    fimo_cmd = (
        f"fimo --oc {output_dir}/fimo_out \\\n"
        f"     --thresh 1e-4 \\\n"
        f"     {motif_args} \\\n"
        f"     {meme_path} \\\n"
        f"     {output_dir}/all_targets_promoters.fasta"
    )

    print(f"\n" + "=" * 60)
    print(f"📊 审计报告:")
    print(f"   - 识别到有效转录因子基因: {bridge_df['LXC_GeneID'].nunique()} 个")
    print(f"   - 匹配到的 Motif 种类: {len(unique_motifs)} 种")
    print(f"\n🔥 请直接复制并运行以下精准扫描命令 (预计 10 分钟内完成):")
    print("-" * 60)
    print(fimo_cmd)
    print("=" * 60)

    return unique_motifs


if __name__ == "__main__":
    high_precision_tf_audit_v4(
        "jaspar_insects_core.meme.txt",
        "anno.summary.xls",
        "processed_tf/gene_id_index.txt"
    )