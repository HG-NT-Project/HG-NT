import torch
import os
import numpy as np
import pandas as pd
from datetime import datetime

# ================= 配置区 =================
# 输入：原有 GCN 路径
OLD_GCN_FILES = {
    'human': 'processed_gcn/human_gcn_network.pt',
    'mouse': 'processed_gcn/mouse_gcn_network.pt'
}

# 基准：NT 序列文件 (Embedding 的物理顺序依据)
NT_SEQ_FILES = {
    'human': 'precomputed_sequences_NT/human_sequences_NT_6kb.pt',
    'mouse': 'precomputed_sequences_NT/mouse_sequences_NT_6kb.pt'
}

# 输出路径
OUTPUT_DIR = 'processed_gcn'


# ==========================================

def get_clean_id(id_str):
    """标准化基因ID"""
    if not isinstance(id_str, str): return ""
    return id_str.strip().split('.')[0].upper()


def analyze_and_reindex(species):
    print(f"\n" + "=" * 80)
    print(f"📊 启动 {species.upper()} GCN 深度对齐分析与重索引")
    print("=" * 80)

    old_path = OLD_GCN_FILES[species]
    nt_path = NT_SEQ_FILES[species]

    if not os.path.exists(old_path) or not os.path.exists(nt_path):
        print(f"❌ 错误: 找不到输入文件，请确认路径。")
        return

    # 1. 加载数据
    old_data = torch.load(old_path, map_location='cpu')
    old_gene_list = old_data.get('gene_list', old_data.get('gene_ids', []))  #
    old_edge_index = old_data['edge_index']  #

    nt_data = torch.load(nt_path, map_location='cpu')
    nt_gene_ids = nt_data['target_genes']  #

    print(f"📍 原始状态:")
    print(f"   - 原始 GCN 节点数: {len(old_gene_list):,}")
    print(f"   - 原始 GCN 边总数: {old_edge_index.shape[1]:,}")
    print(f"   - 目标 NT 基因数:  {len(nt_gene_ids):,}")

    # 2. 建立映射关系与对应情况统计
    old_clean_ids = [get_clean_id(g) for g in old_gene_list]
    target_clean_ids = [get_clean_id(g) for g in nt_gene_ids]

    old_set = set(old_clean_ids)
    target_set = set(target_clean_ids)

    intersection = old_set.intersection(target_set)
    only_in_old = old_set - target_set
    only_in_target = target_set - old_set

    print(f"\n🔍 基因 ID 对应情况分析:")
    print(f"   - 两边完全匹配的基因数: {len(intersection):,} ({len(intersection) / len(target_set):.2%})")
    print(f"   - 仅在原始 GCN 中存在的基因: {len(only_in_old):,}")
    print(f"   - 在 NT 中存在但 GCN 缺失的基因: {len(only_in_target):,}")

    # 3. 索引重映射逻辑 (Index Re-mapping)
    # 目标：旧数字索引 -> 基因名 -> 新数字索引 (Embedding行号)
    idx_to_name = {i: name for i, name in enumerate(old_clean_ids)}
    name_to_new_idx = {name: i for i, name in enumerate(target_clean_ids)}

    # 4. 执行边转换与位移分析
    new_edges = []
    skipped_info = {'out_of_target': 0}
    index_shifts = []  # 记录位置变动

    old_edges = old_edge_index.t().tolist()
    for src_old, dst_old in old_edges:
        src_name = idx_to_name.get(src_old)
        dst_name = idx_to_name.get(dst_old)

        if src_name in name_to_new_idx and dst_name in name_to_new_idx:
            new_src = name_to_new_idx[src_name]
            new_dst = name_to_new_idx[dst_name]
            new_edges.append([new_src, new_dst])
            # 记录第一个节点的位移值作为采样
            index_shifts.append(abs(new_src - src_old))
        else:
            skipped_info['out_of_target'] += 1

    # 5. 生成新的 gene_ids 逻辑说明
    # 这里我们直接将 target_genes 作为新文件的 gene_ids，确保训练脚本读取时：
    # 索引 i 对应的就是 embedding 矩阵的第 i 行

    if new_edges:
        new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t().contiguous()

        # 封装为标准字典格式
        save_dict = {
            'edge_index': new_edge_tensor,
            'gene_ids': nt_gene_ids,  # 关键：此处为对齐后的完整有序列表
            'species': species,
            'num_genes': len(nt_gene_ids),
            'num_edges': new_edge_tensor.shape[1],
            'alignment_stats': {
                'overlap_count': len(intersection),
                'index_shift_avg': np.mean(index_shifts) if index_shifts else 0
            },
            'timestamp': datetime.now().isoformat()
        }

        output_file = os.path.join(OUTPUT_DIR, f"{species}_gcn_network_aligned.pt")
        torch.save(save_dict, output_file)

        print(f"\n✅ 处理完成与新文件生成:")
        print(f"   - 最终保留边数: {len(new_edges):,} (过滤了 {skipped_info['out_of_target']:,} 条无效边)")
        print(f"   - 索引平均位移: {save_dict['alignment_stats']['index_shift_avg']:.2f} 行")
        print(f"   - 新的 gene_ids: 已同步为 NT Embedding 顺序 (共 {len(nt_gene_ids)} 个)")
        print(f"   - 💾 保存路径: {output_file}")
    else:
        print(f"❌ 严重警告: 转换后边数为 0，请检查基因 ID 格式是否一致！")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sp in ['human', 'mouse']:
        analyze_and_reindex(sp)