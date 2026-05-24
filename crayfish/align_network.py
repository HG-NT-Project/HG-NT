import torch
import os
import pandas as pd
from tqdm import tqdm

# ===================== 配置区 =====================
# 基准坐标轴：定义了 0 到 46475 的基因顺序
INDEX_FILE = "gene_id_index.txt"

# 原始输入文件路径
RAW_TF_PATH = "processed_tf/crayfish_tf_edge_index.pt"
RAW_GCN_PATH = "processed_gcn/crayfish_gcn_network.pt"

# 输出目录：存储对齐后的纯 Tensor
OUT_DIR = "processed_aligned_networks"
os.makedirs(OUT_DIR, exist_ok=True)


def clean_id(s):
    """极致清洗逻辑：去除前缀和版本号，确保匹配率"""
    return str(s).replace('gene-', '').replace('rna-', '').split('.')[0].strip()


def build_absolute_networks():
    print(f"📖 步骤1: 加载绝对坐标基准 {INDEX_FILE}")
    if not os.path.exists(INDEX_FILE):
        print(f"❌ 错误: 找不到基准索引文件 {INDEX_FILE}")
        return

    with open(INDEX_FILE, 'r') as f:
        # 建立 ID -> 绝对索引的 O(1) 映射表
        master_ids = [clean_id(line.strip()) for line in f if line.strip()]

    id_to_idx = {gid: i for i, gid in enumerate(master_ids)}
    num_nodes = len(master_ids)
    print(f"✅ 基准映射建立完成，节点总数: {num_nodes}")

    # 需要处理的网络列表
    networks_to_process = [
        ('TF', RAW_TF_PATH, "tf_absolute.pt"),
        ('GCN', RAW_GCN_PATH, "gcn_absolute.pt")
    ]

    for net_name, raw_path, out_name in networks_to_process:
        if not os.path.exists(raw_path):
            print(f"⚠️ 跳过 {net_name}: 文件不存在")
            continue

        print(f"\n🔄 步骤2: 正在对齐 {net_name} 网络并转换为绝对坐标...")
        raw_data = torch.load(raw_path, map_location='cpu')

        # 1. 鲁棒性提取 edge_index
        if isinstance(raw_data, dict):
            edge_index = raw_data.get('edge_index')
            if edge_index is None:
                edge_index = raw_data.get('edges')
        else:
            edge_index = raw_data

        if edge_index is None:
            print(f"❌ 无法从 {net_name} 文件中提取边数据")
            continue

        # 2. 确保维度为 2 x E
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t()

        # 3. 核心转换与过滤
        # 注意：如果你的原始文件存的是数字索引且范围已在 0-46475 内，这里仅做校验
        # 如果存的是 ID 字符串，此处需要加入映射逻辑（目前默认你的原始文件已是初步索引）
        max_val = edge_index.max().item()
        min_val = edge_index.min().item()

        if max_val >= num_nodes or min_val < 0:
            print(f"   ⚠️ 发现越界索引 [{min_val}, {max_val}]，正在执行强力过滤...")
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) & \
                         (edge_index[0] >= 0) & (edge_index[1] >= 0)
            aligned_edges = edge_index[:, valid_mask]
        else:
            print(f"   ✅ 索引范围验证通过: [{min_val}, {max_val}]")
            aligned_edges = edge_index

        # 4. 强制转换为 Long 类型并保存纯 Tensor
        out_path = os.path.join(OUT_DIR, out_name)
        torch.save(aligned_edges.long().contiguous(), out_path)

        print(f"✅ {net_name} 转换完成:")
        print(f"   保留边数: {aligned_edges.shape[1]:,}")
        print(f"   文件已保存: {out_path}")

    print(f"\n✨ 所有网络已对齐至绝对坐标系！")


if __name__ == "__main__":
    build_absolute_networks()