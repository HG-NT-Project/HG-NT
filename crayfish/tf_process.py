import pandas as pd
import torch
import os


def build_crayfish_edge_index(fimo_file, bridge_file, index_file, output_dir="processed_tf"):
    print("🚀 FIMO 扫描已完成，正在构建最终的图边张量...")

    # 1. 加载映射表 (TF -> Motif)
    bridge_df = pd.read_csv(bridge_file)
    motif_to_tfs = bridge_df.groupby('Motif_ID')['Node_Index'].apply(list).to_dict()

    # 2. 加载基准基因索引 (Node Index 锚点)
    with open(index_file, 'r') as f:
        indexed_genes = [line.strip() for line in f]
    gene_to_idx = {gid: i for i, gid in enumerate(indexed_genes)}

    # 3. 解析 FIMO 结果并转换 ID 为 Index
    edges = []
    # FIMO 默认输出文件是 fimo.tsv
    try:
        fimo_df = pd.read_csv(fimo_file, sep='\t', comment='#')
        # 只需要 motif_id 和 sequence_name (即靶基因 ID)
        fimo_df = fimo_df[['motif_id', 'sequence_name']]
    except Exception as e:
        print(f"❌ 读取 FIMO 文件失败: {e}")
        return

    for _, row in fimo_df.iterrows():
        m_id = str(row['motif_id'])
        target_id = str(row['sequence_name'])

        # 桥接：Motif -> TF_Index -> Target_Index
        if m_id in motif_to_tfs and target_id in gene_to_idx:
            target_idx = gene_to_idx[target_id]
            for source_idx in motif_to_tfs[m_id]:
                edges.append([source_idx, target_idx])

    if not edges:
        print("⚠️ 警告：未发现有效边。请检查 FIMO 输出或 ID 匹配表。")
        return

    # 4. 去重并保存为 PyTorch 张量
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.unique(edge_index, dim=1)  # 移除重复的物理接触点

    save_path = os.path.join(output_dir, "crayfish_tf_edge_index.pt")

    # 构建与 GCN 格式统一的字典
    save_data = {
        'edge_index': edge_index,
        'gene_list': indexed_genes
    }

    torch.save(save_data, save_path)

    print("\n" + "=" * 50)
    print(f"📊 小龙虾静态 TF 网络构建报告")
    print(f"   - 总边数: {edge_index.shape[1]:,}")
    print(f"   - 源节点 (TF) 数量: {len(torch.unique(edge_index[0]))}")
    print(f"   - 目标节点数量: {len(torch.unique(edge_index[1]))}")
    print(f"✅ 文件已保存: {save_path}")
    print("=" * 50)


if __name__ == "__main__":
    build_crayfish_edge_index(
        fimo_file="processed_tf/fimo_out/fimo.tsv",
        bridge_file="processed_tf/final_tf_motif_bridge_v3.csv",
        index_file="processed_tf/gene_id_index.txt"
    )