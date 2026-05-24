"""
完整的序列重要性分析脚本 - 精确适配小龙虾数据格式
基于真实注释文件格式：GeneID:RNA-ID:染色体:起始:终止
只使用TF网络和GCN网络（无PPI）
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pyfaidx import Fasta
import os
import json
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# ===================== 配置参数 =====================
class Config:
    # 路径配置
    NT_MODEL_PATH = "../pretrain_model/Nucleotide-Transformer"
    M3_MODEL_PATH = "Results_Crayfish_Ablation_V2/models/crayfish_m3_seed42_best.pth"
    GENOME_FA = "ref.fa"
    ANNO_FILE = "anno.summary.xls"
    EMBEDDINGS_FILE = "crayfish_embeddings/crayfish_embeddings.pt"
    PREDICTIONS_CSV = "Results_Crayfish_Ablation_V2/predictions/crayfish_m3_seed42_predictions.csv"

    # 输出目录
    OUTPUT_DIR = "interect"
    MOTIF_DIR = "motif_candidate"

    # 分析参数
    N_HIGH_GENES = 100
    N_LOW_GENES = 100
    HIGH_EXPR_THRESHOLD = 5.0
    LOW_EXPR_THRESHOLD = 1.0

    # Motif识别参数
    MOTIF_MIN_LENGTH = 5
    MOTIF_MAX_LENGTH = 20
    IMPORTANCE_PERCENTILE = 95
    MIN_GAP_BETWEEN_MOTIFS = 10

    # 邻居分析参数
    TOP_NEIGHBORS_TO_REPORT = 20  # 每个基因保存的重要邻居数量

    # 可视化参数
    HEATMAP_DPI = 300
    SMOOTH_WINDOW = 50

    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== 注释解析（精确适配小龙虾格式）=====================
def parse_crayfish_annotations(anno_file):
    """
    解析小龙虾注释文件
    格式: gene-LOC123746682:rna-XM_069316765.1:chr85:19627149:19715183
    基因ID:RNA-ID:染色体:起始:终止
    """
    print(f"📖 解析注释文件: {anno_file}")
    df = pd.read_csv(anno_file, sep='\t')
    print(f"   总行数: {len(df)}")

    gene_info = {}
    failed_count = 0

    for _, row in df.iterrows():
        gene_id_raw = str(row['GeneID'])
        parts = gene_id_raw.split(':')

        if len(parts) >= 5:
            gene_id = parts[0]  # gene-LOC123746682
            chrom = parts[2]  # chr85

            # 起始和终止位置
            try:
                start = int(parts[3])
                end = int(parts[4])
            except ValueError:
                failed_count += 1
                continue

            # 默认链为+（注释中没有链信息）
            strand = '+'

            gene_info[gene_id] = {
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': strand
            }
        else:
            failed_count += 1

    print(f"   ✅ 成功解析: {len(gene_info)} 个基因")
    print(f"   ⚠️ 解析失败: {failed_count} 个")

    return gene_info


# ===================== 序列提取（保持与原训练一致）=====================
def extract_gene_sequence(genome, gene_info):
    """提取6000bp序列（TSS±2000 + TTS±1000等）"""
    chrom = gene_info['chrom']
    start = gene_info['start']
    end = gene_info['end']
    strand = gene_info['strand']

    # 检查染色体是否存在
    if chrom not in genome:
        # 尝试匹配染色体名
        for avail_chrom in genome.keys():
            if chrom in avail_chrom or avail_chrom in chrom:
                chrom = avail_chrom
                break
        else:
            return 'N' * 6000

    try:
        if strand == '+':
            # 正链：TSS区域 + TTS区域
            tss_seq = str(genome[chrom][max(0, start - 2000):start + 1000])
            tts_seq = str(genome[chrom][max(0, end - 1000):end + 2000])
            seq = tss_seq + tts_seq
        else:
            # 负链：反向互补
            def rc(s):
                return s[::-1].translate(str.maketrans('ATGCatgc', 'TACGtacg'))

            tss_seq = rc(str(genome[chrom][max(0, end - 1000):end + 2000]))
            tts_seq = rc(str(genome[chrom][max(0, start - 2000):start + 1000]))
            seq = tss_seq + tts_seq

        # 填充到6000bp
        seq = seq.ljust(6000, 'N')[:6000].upper()
        return seq

    except Exception as e:
        return 'N' * 6000


# ===================== 模型加载 =====================
def load_models():
    """加载NT和M3模型"""
    print("🤖 加载模型...")

    # NT模型
    tokenizer = AutoTokenizer.from_pretrained(Config.NT_MODEL_PATH)
    nt_model = AutoModelForMaskedLM.from_pretrained(
        Config.NT_MODEL_PATH,
        torch_dtype=torch.float16
    ).to(Config.DEVICE)
    nt_model.eval()

    for param in nt_model.parameters():
        param.requires_grad = False

    # M3模型
    from train_xr_xiaolongxia import ModelM3_MultiGraphConcat
    m3_model = ModelM3_MultiGraphConcat(input_dim=2560).to(Config.DEVICE)

    checkpoint = torch.load(Config.M3_MODEL_PATH, map_location=Config.DEVICE, weights_only=False)
    m3_model.load_state_dict(checkpoint['model_state_dict'])
    m3_model.eval()

    print(f"   ✅ 模型加载完成，设备: {Config.DEVICE}")
    return tokenizer, nt_model, m3_model


# ===================== 获取邻居信息 =====================
def load_network_neighbors():
    """加载网络结构，预计算每个基因的邻居（只使用TF和GCN网络）"""
    print("🔗 加载网络结构...")

    # 加载embeddings获取基因ID映射
    embed_data = torch.load(Config.EMBEDDINGS_FILE, map_location='cpu', weights_only=False)
    gene_ids = embed_data['gene_ids']
    gene_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
    idx_to_gene = {i: gid for i, gid in enumerate(gene_ids)}

    # 加载边
    tf_edge_index = torch.load("processed_tf/crayfish_tf_edge_index.pt")['edge_index']
    gcn_edge_index = torch.load("processed_gcn/crayfish_gcn_network.pt")['edge_index']

    # 预计算邻居
    tf_neighbors = defaultdict(list)
    gcn_neighbors = defaultdict(list)

    for i in range(tf_edge_index.shape[1]):
        src = tf_edge_index[0, i].item()
        dst = tf_edge_index[1, i].item()
        tf_neighbors[dst].append(src)

    for i in range(gcn_edge_index.shape[1]):
        src = gcn_edge_index[0, i].item()
        dst = gcn_edge_index[1, i].item()
        gcn_neighbors[dst].append(src)

    print(f"   ✅ 网络加载完成")
    print(f"      - TF网络边数: {tf_edge_index.shape[1]}")
    print(f"      - GCN网络边数: {gcn_edge_index.shape[1]}")
    return gene_to_idx, idx_to_gene, tf_neighbors, gcn_neighbors, embed_data['x']


# ===================== 单基因分析（已修复 In-place 错误）=====================
def analyze_single_gene(gene_id, group, true_expr, pred_expr,
                        gene_info, genome, tokenizer, nt_model, m3_model,
                        gene_to_idx, idx_to_gene, tf_neighbors, gcn_neighbors, all_embeddings):
    """分析单个基因的序列重要性，同时计算邻居节点的重要性"""
    # 1. 提取序列与基础检查
    if gene_id not in gene_info:
        return None
    sequence = extract_gene_sequence(genome, gene_info[gene_id])
    if sequence.count('N') > 3000:
        return None

    # 2. Tokenize
    inputs = tokenizer(sequence, return_tensors="pt", max_length=1000,
                       padding='max_length', truncation=True).to(Config.DEVICE)

    try:
        # 核心修复点：确保计算图允许梯度
        with torch.set_grad_enabled(True):
            # 获取序列嵌入 (由 NT 模型生成)
            input_embeds = nt_model.get_input_embeddings()(inputs['input_ids'])
            input_embeds = input_embeds.clone().detach().requires_grad_(True)

            # NT 前向传播
            nt_outputs = nt_model(
                inputs_embeds=input_embeds,
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True
            )

            last_hidden = nt_outputs.hidden_states[-1]
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            gene_embedding = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            neighbor_info = []  # 存储邻居梯度信息

            if gene_id in gene_to_idx:
                gene_idx = gene_to_idx[gene_id]

                # 获取TF和GCN邻居索引
                tf_neigh = tf_neighbors.get(gene_idx, [])[:32]
                gcn_neigh = gcn_neighbors.get(gene_idx, [])[:32]

                # 构建局部子图节点集
                all_nodes = [gene_idx] + list(set(tf_neigh + gcn_neigh))
                all_nodes = torch.tensor(all_nodes, dtype=torch.long)

                # --- 关键修复：使用非原地操作构建输入张量 x ---
                x_list = []
                for node_idx in all_nodes:
                    node_idx_item = node_idx.item()
                    if node_idx_item == gene_idx:
                        # 对于中心基因，使用刚刚从序列生成的带梯度的 embedding
                        x_list.append(gene_embedding)
                    else:
                        # 对于邻居基因，从预计算的 embeddings 中提取
                        # 这里使用 unsqueeze(0) 保持维度一致 [1, 2560]
                        neighbor_emb = all_embeddings[node_idx_item].unsqueeze(0).to(Config.DEVICE)
                        x_list.append(neighbor_emb)

                # 合并为 [Node_count, Hidden_dim] 并开启梯度追踪
                x = torch.cat(x_list, dim=0).requires_grad_(True)

                # M3 前向传播
                model_out = m3_model(x)
                outputs = model_out[0] if isinstance(model_out, tuple) else model_out

                # 提取中心基因（索引为0）的预测值
                pred = outputs[0] if outputs.dim() > 0 else outputs
                if pred.dim() > 0: pred = pred.squeeze()

                # 执行反向传播
                m3_model.zero_grad()
                nt_model.zero_grad()
                pred.backward()

                # A. 获取邻居节点的梯度重要性
                if x.grad is not None:
                    node_gradients = x.grad.abs().sum(dim=-1).cpu().numpy()

                    for i, node_idx_tensor in enumerate(all_nodes):
                        node_idx_val = node_idx_tensor.item()
                        node_name = idx_to_gene.get(node_idx_val, "Unknown")
                        importance_score = node_gradients[i]

                        # 识别邻居类型
                        if node_idx_val == gene_idx:
                            n_type = "Center"
                        elif node_idx_val in tf_neighbors[gene_idx]:
                            n_type = "TF_Neighbor"
                        elif node_idx_val in gcn_neighbors[gene_idx]:
                            n_type = "GCN_Neighbor"
                        else:
                            n_type = "Other"

                        neighbor_info.append({
                            'neighbor_id': node_name,
                            'type': n_type,
                            'gradient_importance': float(importance_score),
                            'is_center': (node_name == gene_id)
                        })

                # B. 获取序列嵌入的梯度（用于 Motif 分析）
                if input_embeds.grad is not None:
                    token_gradients = input_embeds.grad.abs().sum(dim=-1).squeeze(0).float().cpu().numpy()
                else:
                    return None

            else:
                # 备选：如果基因不在网络中，回退到基础预测逻辑
                pred = m3_model(gene_embedding)
                if pred.dim() > 0: pred = pred.squeeze()
                m3_model.zero_grad()
                nt_model.zero_grad()
                pred.backward()
                token_gradients = input_embeds.grad.abs().sum(dim=-1).squeeze(0).float().cpu().numpy()

    except Exception as e:
        print(f"    ⚠️ {gene_id} 计算异常: {e}")
        return None

    # 扩展与平滑（将 token 梯度映射回 6000bp 序列）
    base_gradients = np.repeat(token_gradients, 6)[:6000]
    smoothed = gaussian_filter1d(base_gradients, sigma=Config.SMOOTH_WINDOW / 3)

    return {
        'gene_id': gene_id,
        'group': group,
        'sequence': sequence,
        'prediction': float(pred),
        'true_expression': true_expr,
        'base_gradients': base_gradients,
        'smoothed_gradients': smoothed,
        'neighbor_info': neighbor_info
    }


# ===================== 单基因红色热力图 =====================
def plot_individual_heatmap(result, save_dir):
    """
    为单个基因生成红色系热力图并保存
    使用红色系列 (Reds)，标注TSS和TTS区域
    """
    importance = result['smoothed_gradients']
    gene_id = result['gene_id']
    group = result['group']

    plt.figure(figsize=(16, 4))

    # 将 1x6000 的向量转为热力图格式
    data_2d = importance.reshape(1, -1)

    # 使用红色系 cmap="Reds"
    ax = sns.heatmap(data_2d, cmap="Reds", cbar_kws={'label': 'Importance Score', 'shrink': 0.8},
                     xticklabels=False, yticklabels=False)

    # 绘制 TSS (2000bp) 和 TTS (4000bp) 的分界线
    ax.axvline(x=2000, color='#1f77b4', linestyle='--', linewidth=2, alpha=0.8, label='TSS (±2000)')
    ax.axvline(x=4000, color='#2ca02c', linestyle='--', linewidth=2, alpha=0.8, label='TTS (±2000)')

    # 标注区域
    ax.text(1000, 0.5, 'Promoter Region', ha='center', va='center',
            fontsize=10, color='black', alpha=0.7, transform=ax.transData)
    ax.text(3000, 0.5, 'Gene Body', ha='center', va='center',
            fontsize=10, color='black', alpha=0.7, transform=ax.transData)
    ax.text(5000, 0.5, 'Terminator Region', ha='center', va='center',
            fontsize=10, color='black', alpha=0.7, transform=ax.transData)

    # 添加统计信息
    stats_text = f"Mean Imp: {importance.mean():.4f} | Max Imp: {importance.max():.4f} | Group: {group}"
    plt.title(f"Sequence Importance Heatmap: {gene_id}\n{stats_text}", fontsize=12, fontweight='bold')
    plt.xlabel("Genomic Position (bp)", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'importance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 同时生成一个详细的曲线图作为补充
    plt.figure(figsize=(16, 6))
    plt.plot(importance, 'r-', linewidth=1, alpha=0.7)
    plt.fill_between(range(len(importance)), importance, alpha=0.3, color='red')
    plt.axvline(x=2000, color='blue', linestyle='--', linewidth=1.5, label='TSS')
    plt.axvline(x=4000, color='green', linestyle='--', linewidth=1.5, label='TTS')
    plt.xlabel('Position (bp)', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.title(f'Sequence Importance Profile: {gene_id} ({group} Expression)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'importance_profile.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ===================== 邻居重要性可视化 =====================
def plot_neighbor_importance(neighbor_df, gene_id, save_dir):
    """
    生成邻居重要性可视化图（只显示TF和GCN邻居）
    """
    if neighbor_df.empty:
        return

    # 排除中心基因本身，只显示邻居
    neighbor_only = neighbor_df[~neighbor_df['is_center']].head(20)

    if neighbor_only.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 条形图：Top 20 重要邻居
    ax1 = axes[0]
    top_neighbors = neighbor_only.nlargest(20, 'gradient_importance')

    colors = ['#ff6b6b' if t == 'TF_Neighbor' else '#4ecdc4' if t == 'GCN_Neighbor' else '#95a5a6'
              for t in top_neighbors['type']]

    bars = ax1.barh(range(len(top_neighbors)), top_neighbors['gradient_importance'].values, color=colors)
    ax1.set_yticks(range(len(top_neighbors)))
    ax1.set_yticklabels(top_neighbors['neighbor_id'].values, fontsize=8)
    ax1.set_xlabel('Gradient Importance', fontsize=12)
    ax1.set_title(f'Top 20 Important Neighbors for {gene_id}', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff6b6b', label='TF Neighbor'),
                       Patch(facecolor='#4ecdc4', label='GCN Neighbor')]
    ax1.legend(handles=legend_elements, loc='lower right')

    # 2. 饼图：邻居类型分布
    ax2 = axes[1]
    type_counts = neighbor_only['type'].value_counts()
    colors_pie = ['#ff6b6b', '#4ecdc4']
    wedges, texts, autotexts = ax2.pie(type_counts.values, labels=type_counts.index,
                                       autopct='%1.1f%%', colors=colors_pie[:len(type_counts)])
    ax2.set_title(f'Neighbor Type Distribution\nTotal: {len(neighbor_only)} neighbors',
                  fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'neighbor_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 额外：邻居重要性热力图（分别显示TF和GCN）
    if len(neighbor_only) > 5:
        plt.figure(figsize=(12, 8))
        # 按类型分组着色
        tf_data = neighbor_only[neighbor_only['type'] == 'TF_Neighbor']
        gcn_data = neighbor_only[neighbor_only['type'] == 'GCN_Neighbor']

        # 创建重要性矩阵
        importance_matrix = neighbor_only.set_index('neighbor_id')[['gradient_importance']].T
        sns.heatmap(importance_matrix, cmap='YlOrRd', annot=True, fmt='.4f',
                    cbar_kws={'label': 'Importance'}, xticklabels=True)
        plt.title(f'Neighbor Importance Heatmap: {gene_id}\n(TF: red tones, GCN: blue tones)',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Neighbor Genes')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'neighbor_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ===================== Motif识别 =====================
def find_motifs(result, threshold_percentile=95):
    """识别重要motif区域"""
    gradients = result['smoothed_gradients']
    sequence = result['sequence']

    threshold = np.percentile(gradients, threshold_percentile)

    motifs = []
    i = 0
    while i < len(gradients):
        if gradients[i] > threshold:
            start = i
            while i < len(gradients) and gradients[i] > threshold:
                i += 1
            end = i

            length = end - start
            if Config.MOTIF_MIN_LENGTH <= length <= Config.MOTIF_MAX_LENGTH:
                motif_seq = sequence[start:end]

                # 确定区域
                if start < 2000:
                    region = "TSS_upstream"
                elif start < 3000:
                    region = "TSS_downstream"
                elif start < 4000:
                    region = "TTS_upstream"
                else:
                    region = "TTS_downstream"

                motifs.append({
                    'gene_id': result['gene_id'],
                    'group': result['group'],
                    'start': start,
                    'end': end,
                    'length': length,
                    'sequence': motif_seq,
                    'mean_importance': gradients[start:end].mean(),
                    'max_importance': gradients[start:end].max(),
                    'region': region
                })
        else:
            i += 1

    return motifs


# ===================== 高贡献区域统计 =====================
def identify_high_impact_regions(results, group_name, percentile=90, min_length=20):
    """
    统计群体中贡献值较高的区域
    """
    if not results:
        return pd.DataFrame()

    # 聚合所有基因的平滑梯度
    all_grads = np.array([r['smoothed_gradients'] for r in results])
    mean_grads = np.mean(all_grads, axis=0)
    std_grads = np.std(all_grads, axis=0)

    # 设定高贡献阈值
    threshold = np.percentile(mean_grads, percentile)

    regions = []
    start = None

    for i in range(len(mean_grads)):
        if mean_grads[i] >= threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i
                length = end - start
                if length >= min_length:
                    # 确定区域类型
                    if start < 2000:
                        region_type = "Promoter_Upstream"
                    elif start < 3000:
                        region_type = "Promoter_Downstream"
                    elif start < 4000:
                        region_type = "Gene_Body"
                    else:
                        region_type = "Terminator"

                    regions.append({
                        'group': group_name,
                        'region_type': region_type,
                        'start': start,
                        'end': end,
                        'length': length,
                        'avg_importance': mean_grads[start:end].mean(),
                        'max_importance': mean_grads[start:end].max(),
                        'peak_position': start + np.argmax(mean_grads[start:end]),
                        'coverage_variance': std_grads[start:end].mean()
                    })
                start = None

    # 处理最后一个区域
    if start is not None:
        end = len(mean_grads)
        length = end - start
        if length >= min_length:
            if start < 2000:
                region_type = "Promoter_Upstream"
            elif start < 3000:
                region_type = "Promoter_Downstream"
            elif start < 4000:
                region_type = "Gene_Body"
            else:
                region_type = "Terminator"

            regions.append({
                'group': group_name,
                'region_type': region_type,
                'start': start,
                'end': end,
                'length': length,
                'avg_importance': mean_grads[start:end].mean(),
                'max_importance': mean_grads[start:end].max(),
                'peak_position': start + np.argmax(mean_grads[start:end]),
                'coverage_variance': std_grads[start:end].mean()
            })

    return pd.DataFrame(regions)


# ===================== 全局邻居重要性统计 =====================
def create_global_neighbor_summary(all_results, output_dir):
    """
    创建全局邻居重要性汇总统计（只统计TF和GCN）
    """
    all_neighbors = []

    for result in all_results:
        if 'neighbor_info' in result and result['neighbor_info']:
            for neighbor in result['neighbor_info']:
                if not neighbor['is_center']:  # 排除中心基因
                    all_neighbors.append({
                        'center_gene': result['gene_id'],
                        'center_group': result['group'],
                        'neighbor_id': neighbor['neighbor_id'],
                        'neighbor_type': neighbor['type'],
                        'importance': neighbor['gradient_importance']
                    })

    if not all_neighbors:
        return

    neighbor_df = pd.DataFrame(all_neighbors)

    # 1. 统计每个邻居作为重要调控因子的频率
    neighbor_freq = neighbor_df.groupby(['neighbor_id', 'neighbor_type']).agg({
        'importance': ['mean', 'std', 'count'],
        'center_gene': lambda x: list(x)
    }).round(4)
    neighbor_freq.columns = ['mean_importance', 'std_importance', 'frequency', 'regulated_genes']
    neighbor_freq = neighbor_freq.sort_values('frequency', ascending=False)
    neighbor_freq.to_csv(os.path.join(output_dir, 'global_neighbor_frequency.csv'))

    # 2. 按网络类型分别统计Top邻居
    tf_neighbors_only = neighbor_df[neighbor_df['neighbor_type'] == 'TF_Neighbor']
    gcn_neighbors_only = neighbor_df[neighbor_df['neighbor_type'] == 'GCN_Neighbor']

    tf_top = tf_neighbors_only.groupby('neighbor_id').size().sort_values(ascending=False).head(50)
    tf_top.to_csv(os.path.join(output_dir, 'top_tf_neighbors.csv'))

    gcn_top = gcn_neighbors_only.groupby('neighbor_id').size().sort_values(ascending=False).head(50)
    gcn_top.to_csv(os.path.join(output_dir, 'top_gcn_neighbors.csv'))

    # 3. 按表达组统计
    high_neighbors = neighbor_df[neighbor_df['center_group'] == 'High']
    low_neighbors = neighbor_df[neighbor_df['center_group'] == 'Low']

    high_summary = high_neighbors.groupby(['neighbor_id', 'neighbor_type']).size().reset_index(name='frequency')
    high_summary = high_summary.sort_values('frequency', ascending=False).head(50)
    high_summary.to_csv(os.path.join(output_dir, 'high_expr_top_neighbors.csv'), index=False)

    low_summary = low_neighbors.groupby(['neighbor_id', 'neighbor_type']).size().reset_index(name='frequency')
    low_summary = low_summary.sort_values('frequency', ascending=False).head(50)
    low_summary.to_csv(os.path.join(output_dir, 'low_expr_top_neighbors.csv'), index=False)

    # 4. 可视化Top调控因子
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top 20 最频繁出现的调控因子（总体）
    top20 = neighbor_freq.head(20)
    ax1 = axes[0, 0]
    colors = ['#ff6b6b' if 'TF' in t else '#4ecdc4' for t in top20.index.get_level_values('neighbor_type')]
    ax1.barh(range(len(top20)), top20['frequency'].values, color=colors)
    ax1.set_yticks(range(len(top20)))
    ax1.set_yticklabels(top20.index.get_level_values('neighbor_id'), fontsize=8)
    ax1.set_xlabel('Frequency (Number of target genes)', fontsize=12)
    ax1.set_title('Top 20 Most Frequent Regulatory Neighbors (Overall)', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()

    # Top TF邻居
    tf_top20 = tf_top.head(20)
    ax2 = axes[0, 1]
    ax2.barh(range(len(tf_top20)), tf_top20.values, color='#ff6b6b')
    ax2.set_yticks(range(len(tf_top20)))
    ax2.set_yticklabels(tf_top20.index, fontsize=8)
    ax2.set_xlabel('Frequency', fontsize=12)
    ax2.set_title('Top 20 TF Network Neighbors', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()

    # Top GCN邻居
    gcn_top20 = gcn_top.head(20)
    ax3 = axes[1, 0]
    ax3.barh(range(len(gcn_top20)), gcn_top20.values, color='#4ecdc4')
    ax3.set_yticks(range(len(gcn_top20)))
    ax3.set_yticklabels(gcn_top20.index, fontsize=8)
    ax3.set_xlabel('Frequency', fontsize=12)
    ax3.set_title('Top 20 GCN Network Neighbors', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()

    # 高表达 vs 低表达组的调控模式差异
    ax4 = axes[1, 1]
    high_top = high_neighbors['neighbor_id'].value_counts().head(15)
    low_top = low_neighbors['neighbor_id'].value_counts().head(15)

    all_top_genes = set(high_top.index) | set(low_top.index)
    comparison_data = []
    for gene in all_top_genes:
        comparison_data.append({
            'neighbor': gene,
            'high_freq': high_neighbors[high_neighbors['neighbor_id'] == gene].shape[0],
            'low_freq': low_neighbors[low_neighbors['neighbor_id'] == gene].shape[0]
        })

    comp_df = pd.DataFrame(comparison_data)
    comp_df = comp_df.sort_values('high_freq', ascending=False).head(15)

    x = np.arange(len(comp_df))
    width = 0.35
    ax4.bar(x - width / 2, comp_df['high_freq'], width, label='High Expression Group', color='red', alpha=0.7)
    ax4.bar(x + width / 2, comp_df['low_freq'], width, label='Low Expression Group', color='blue', alpha=0.7)
    ax4.set_xlabel('Neighbor Genes', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Regulatory Pattern: High vs Low Expression', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(comp_df['neighbor'], rotation=45, ha='right', fontsize=8)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'global_neighbor_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ 全局邻居统计完成")
    print(f"      - 总调控关系: {len(neighbor_df)}")
    print(f"      - TF网络调控关系: {len(tf_neighbors_only)}")
    print(f"      - GCN网络调控关系: {len(gcn_neighbors_only)}")
    print(f"      - 唯一调控因子: {neighbor_df['neighbor_id'].nunique()}")


# ===================== 基础可视化 =====================
def create_visualizations(all_results, output_dir):
    """创建基础可视化"""
    print("\n🎨 创建可视化...")

    high_results = [r for r in all_results if r['group'] == 'High']
    low_results = [r for r in all_results if r['group'] == 'Low']

    # 1. 热力图
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    if high_results:
        high_data = np.array([r['smoothed_gradients'] for r in high_results])
        high_expr = [r['true_expression'] for r in high_results]
        sort_idx = np.argsort(high_expr)
        high_data = high_data[sort_idx]

        im1 = axes[0, 0].imshow(high_data, aspect='auto', cmap='hot', interpolation='bilinear')
        axes[0, 0].set_title(f'High Expression Genes (n={len(high_results)})', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Genes (sorted by expression)')
        axes[0, 0].axvline(x=2000, color='white', linestyle='--', linewidth=1, alpha=0.5)
        axes[0, 0].axvline(x=4000, color='white', linestyle='--', linewidth=1, alpha=0.5)
        plt.colorbar(im1, ax=axes[0, 0], label='Importance')

    if low_results:
        low_data = np.array([r['smoothed_gradients'] for r in low_results])
        low_expr = [r['true_expression'] for r in low_results]
        sort_idx = np.argsort(low_expr)[::-1]
        low_data = low_data[sort_idx]

        im2 = axes[0, 1].imshow(low_data, aspect='auto', cmap='viridis', interpolation='bilinear')
        axes[0, 1].set_title(f'Low Expression Genes (n={len(low_results)})', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Genes (sorted by expression)')
        axes[0, 1].axvline(x=2000, color='white', linestyle='--', linewidth=1, alpha=0.5)
        axes[0, 1].axvline(x=4000, color='white', linestyle='--', linewidth=1, alpha=0.5)
        plt.colorbar(im2, ax=axes[0, 1], label='Importance')

    # 2. 平均曲线对比
    if high_results and low_results:
        high_mean = np.mean([r['smoothed_gradients'] for r in high_results], axis=0)
        low_mean = np.mean([r['smoothed_gradients'] for r in low_results], axis=0)
        high_std = np.std([r['smoothed_gradients'] for r in high_results], axis=0)
        low_std = np.std([r['smoothed_gradients'] for r in low_results], axis=0)

        x = np.arange(6000)
        axes[1, 0].plot(x, high_mean, 'r-', linewidth=2, label='High Expression')
        axes[1, 0].fill_between(x, high_mean - high_std, high_mean + high_std, alpha=0.3, color='red')
        axes[1, 0].plot(x, low_mean, 'b-', linewidth=2, label='Low Expression')
        axes[1, 0].fill_between(x, low_mean - low_std, low_mean + low_std, alpha=0.3, color='blue')
        axes[1, 0].axvline(x=2000, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=4000, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Position (bp)', fontsize=12)
        axes[1, 0].set_ylabel('Mean Importance', fontsize=12)
        axes[1, 0].set_title('Average Importance Profile', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 差异分析
        diff = high_mean - low_mean
        axes[1, 1].plot(x, diff, 'g-', linewidth=2)
        axes[1, 1].fill_between(x, 0, diff, where=(np.abs(diff) > 0.01), alpha=0.3, color='green')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].axvline(x=2000, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=4000, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Position (bp)', fontsize=12)
        axes[1, 1].set_ylabel('Difference (High - Low)', fontsize=12)
        axes[1, 1].set_title('Differential Importance', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), dpi=Config.HEATMAP_DPI)
    plt.close()

    print(f"   ✅ 基础可视化保存完成")


# ===================== 全局重要性可视化 =====================
def create_global_importance_plot(all_results, output_dir):
    """
    创建全局重要性汇总图
    """
    high_results = [r for r in all_results if r['group'] == 'High']
    low_results = [r for r in all_results if r['group'] == 'Low']

    if not high_results or not low_results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 平均曲线对比
    high_mean = np.mean([r['smoothed_gradients'] for r in high_results], axis=0)
    low_mean = np.mean([r['smoothed_gradients'] for r in low_results], axis=0)
    high_std = np.std([r['smoothed_gradients'] for r in high_results], axis=0)
    low_std = np.std([r['smoothed_gradients'] for r in low_results], axis=0)

    x = np.arange(6000)
    ax = axes[0, 0]
    ax.plot(x, high_mean, 'r-', linewidth=2, label='High Expression (n={})'.format(len(high_results)))
    ax.fill_between(x, high_mean - high_std, high_mean + high_std, alpha=0.3, color='red')
    ax.plot(x, low_mean, 'b-', linewidth=2, label='Low Expression (n={})'.format(len(low_results)))
    ax.fill_between(x, low_mean - low_std, low_mean + low_std, alpha=0.3, color='blue')
    ax.axvline(x=2000, color='gray', linestyle='--', alpha=0.5, label='TSS')
    ax.axvline(x=4000, color='gray', linestyle='--', alpha=0.5, label='TTS')
    ax.set_xlabel('Position (bp)', fontsize=12)
    ax.set_ylabel('Mean Importance', fontsize=12)
    ax.set_title('Average Importance Profile by Expression Group', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. 差异分析热力图
    ax = axes[0, 1]
    n_genes = min(len(high_results), len(low_results))
    diff_matrix = np.array([r['smoothed_gradients'] for r in high_results[:n_genes]]) - \
                  np.array([r['smoothed_gradients'] for r in low_results[:n_genes]])
    im = ax.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', interpolation='bilinear')
    ax.axvline(x=2000, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=4000, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Position (bp)', fontsize=12)
    ax.set_ylabel('Genes (High vs Low)', fontsize=12)
    ax.set_title('Differential Importance (High - Low)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Difference')

    # 3. 区域重要性条形图
    ax = axes[1, 0]
    regions = ['Promoter_Upstream\n(0-2000bp)', 'Promoter_Downstream\n(2000-3000bp)',
               'Gene_Body\n(3000-4000bp)', 'Terminator\n(4000-6000bp)']
    high_region_means = [
        np.mean([r['smoothed_gradients'][0:2000].mean() for r in high_results]),
        np.mean([r['smoothed_gradients'][2000:3000].mean() for r in high_results]),
        np.mean([r['smoothed_gradients'][3000:4000].mean() for r in high_results]),
        np.mean([r['smoothed_gradients'][4000:6000].mean() for r in high_results])
    ]
    low_region_means = [
        np.mean([r['smoothed_gradients'][0:2000].mean() for r in low_results]),
        np.mean([r['smoothed_gradients'][2000:3000].mean() for r in low_results]),
        np.mean([r['smoothed_gradients'][3000:4000].mean() for r in low_results]),
        np.mean([r['smoothed_gradients'][4000:6000].mean() for r in low_results])
    ]

    x_pos = np.arange(len(regions))
    width = 0.35
    ax.bar(x_pos - width / 2, high_region_means, width, label='High Expression', color='red', alpha=0.7)
    ax.bar(x_pos + width / 2, low_region_means, width, label='Low Expression', color='blue', alpha=0.7)
    ax.set_xlabel('Genomic Region', fontsize=12)
    ax.set_ylabel('Mean Importance', fontsize=12)
    ax.set_title('Region-wise Importance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 累计贡献曲线
    ax = axes[1, 1]
    high_sorted = np.sort([r['smoothed_gradients'].max() for r in high_results])[::-1]
    low_sorted = np.sort([r['smoothed_gradients'].max() for r in low_results])[::-1]
    ax.plot(range(1, len(high_sorted) + 1), np.cumsum(high_sorted) / np.sum(high_sorted),
            'r-', linewidth=2, label='High Expression')
    ax.plot(range(1, len(low_sorted) + 1), np.cumsum(low_sorted) / np.sum(low_sorted),
            'b-', linewidth=2, label='Low Expression')
    ax.set_xlabel('Number of Genes (sorted by max importance)', fontsize=12)
    ax.set_ylabel('Cumulative Contribution', fontsize=12)
    ax.set_title('Cumulative Importance Contribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'global_importance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ===================== 主程序 =====================
def main():
    print("=" * 80)
    print("🧬 小龙虾序列重要性分析（TF + GCN 网络）")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.MOTIF_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.OUTPUT_DIR, 'gene_data'), exist_ok=True)

    # 1. 解析注释
    print("\n1️⃣ 解析注释文件...")
    gene_info = parse_crayfish_annotations(Config.ANNO_FILE)

    # 2. 加载基因组
    print("\n2️⃣ 加载基因组...")
    genome = Fasta(Config.GENOME_FA, as_raw=True, sequence_always_upper=True)
    print(f"   ✅ 基因组加载完成，包含染色体: {', '.join(list(genome.keys())[:5])}...")

    # 3. 加载预测结果
    print("\n3️⃣ 加载预测结果...")
    predictions = pd.read_csv(Config.PREDICTIONS_CSV)
    predictions['abs_error'] = (predictions['true_expression'] - predictions['predicted_expression']).abs()

    # 筛选基因
    high_genes = predictions[predictions['true_expression'] >= Config.HIGH_EXPR_THRESHOLD].nsmallest(
        Config.N_HIGH_GENES, 'abs_error')
    low_genes = predictions[predictions['true_expression'] <= Config.LOW_EXPR_THRESHOLD].nsmallest(Config.N_LOW_GENES,
                                                                                                   'abs_error')

    print(f"   高表达基因: {len(high_genes)} 个")
    print(f"   低表达基因: {len(low_genes)} 个")

    # 4. 加载模型
    print("\n4️⃣ 加载模型...")
    tokenizer, nt_model, m3_model = load_models()

    # 5. 加载网络（只使用TF和GCN）
    print("\n5️⃣ 加载网络结构...")
    gene_to_idx, idx_to_gene, tf_neighbors, gcn_neighbors, all_embeddings = load_network_neighbors()

    # 6. 分析基因
    print(f"\n6️⃣ 开始分析基因...")
    all_results = []
    all_motifs = []

    for group, genes_df in [('High', high_genes), ('Low', low_genes)]:
        print(f"\n   📊 处理 {group} 表达组 ({len(genes_df)} 个基因)...")

        for _, row in tqdm(genes_df.iterrows(), total=len(genes_df), desc=f"   {group}"):
            gene_id = row['gene_id']
            true_expr = row['true_expression']
            pred_expr = row['predicted_expression']

            result = analyze_single_gene(
                gene_id, group, true_expr, pred_expr,
                gene_info, genome, tokenizer, nt_model, m3_model,
                gene_to_idx, idx_to_gene, tf_neighbors, gcn_neighbors, all_embeddings
            )

            if result:
                all_results.append(result)

                # 保存单个基因数据
                gene_dir = os.path.join(Config.OUTPUT_DIR, 'gene_data', gene_id)
                os.makedirs(gene_dir, exist_ok=True)

                # 生成单基因红色热力图
                plot_individual_heatmap(result, gene_dir)

                # 保存邻居分析结果
                if result['neighbor_info']:
                    neighbor_df = pd.DataFrame(result['neighbor_info'])
                    neighbor_df = neighbor_df.sort_values('gradient_importance', ascending=False)
                    neighbor_df.to_csv(os.path.join(gene_dir, 'neighbors_analysis.csv'), index=False)

                    # 生成邻居重要性可视化
                    plot_neighbor_importance(neighbor_df, gene_id, gene_dir)

                np.save(os.path.join(gene_dir, 'base_importance.npy'), result['base_gradients'])
                np.save(os.path.join(gene_dir, 'smoothed_importance.npy'), result['smoothed_gradients'])

                with open(os.path.join(gene_dir, 'sequence.fa'), 'w') as f:
                    f.write(f">{gene_id}\n{result['sequence']}\n")

                # 识别motif
                motifs = find_motifs(result, Config.IMPORTANCE_PERCENTILE)
                all_motifs.extend(motifs)

    print(f"\n   ✅ 成功分析 {len(all_results)} 个基因")

    # 7. 创建基础可视化
    create_visualizations(all_results, Config.OUTPUT_DIR)

    # 8. 保存motif结果
    if all_motifs:
        print("\n7️⃣ 保存Motif结果...")
        motif_df = pd.DataFrame(all_motifs)
        motif_df.to_csv(os.path.join(Config.MOTIF_DIR, 'all_candidate_motifs.csv'), index=False)

        # Top 100
        top_motifs = motif_df.nlargest(100, 'mean_importance')
        top_motifs.to_csv(os.path.join(Config.MOTIF_DIR, 'top_100_motifs.csv'), index=False)

        # 分组保存
        motif_df[motif_df['group'] == 'High'].to_csv(os.path.join(Config.MOTIF_DIR, 'high_expression_motifs.csv'),
                                                     index=False)
        motif_df[motif_df['group'] == 'Low'].to_csv(os.path.join(Config.MOTIF_DIR, 'low_expression_motifs.csv'),
                                                    index=False)

        # FASTA格式
        with open(os.path.join(Config.MOTIF_DIR, 'candidate_motifs.fa'), 'w') as f:
            for _, row in motif_df.iterrows():
                f.write(
                    f">{row['gene_id']}_{row['start']}_{row['end']}_{row['region']}_imp_{row['mean_importance']:.4f}\n")
                f.write(f"{row['sequence']}\n")

        print(f"   ✅ 保存 {len(all_motifs)} 个motif")

    # 9. 保存汇总结果
    summary_df = pd.DataFrame([{
        'gene_id': r['gene_id'],
        'group': r['group'],
        'prediction': r['prediction'],
        'true_expression': r['true_expression'],
        'mean_importance': r['smoothed_gradients'].mean(),
        'max_importance': r['smoothed_gradients'].max(),
        'tss_upstream': r['smoothed_gradients'][0:2000].mean(),
        'tss_downstream': r['smoothed_gradients'][2000:3000].mean(),
        'tts_upstream': r['smoothed_gradients'][3000:4000].mean(),
        'tts_downstream': r['smoothed_gradients'][4000:6000].mean(),
        'n_neighbors': len(r.get('neighbor_info', []))
    } for r in all_results])

    summary_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'gene_importance_summary.csv'), index=False)

    # 10. 全局区域重要性统计
    print("\n8️⃣ 统计各组高贡献区域分布...")

    high_expr_results = [r for r in all_results if r['group'] == 'High']
    low_expr_results = [r for r in all_results if r['group'] == 'Low']

    # 生成全局汇总图
    create_global_importance_plot(all_results, Config.OUTPUT_DIR)

    # 识别高贡献区域
    high_regions = identify_high_impact_regions(high_expr_results, "High_Expression", percentile=90)
    low_regions = identify_high_impact_regions(low_expr_results, "Low_Expression", percentile=90)
    all_regions = identify_high_impact_regions(all_results, "All_Genes", percentile=95)

    # 合并并保存
    all_stats = pd.concat([high_regions, low_regions, all_regions], ignore_index=True)
    all_stats.to_csv(os.path.join(Config.OUTPUT_DIR, 'high_importance_regions_stats.csv'), index=False)

    # 按区域类型统计
    if len(all_stats) > 0:
        region_summary = all_stats.groupby(['group', 'region_type']).agg({
            'avg_importance': 'mean',
            'length': 'mean',
            'peak_position': 'count'
        }).round(4)
        region_summary = region_summary.rename(columns={'peak_position': 'count'})
        region_summary.to_csv(os.path.join(Config.OUTPUT_DIR, 'region_type_summary.csv'))

        print(f"   ✅ 高贡献区域统计完成")
        print(f"      - High组: {len(high_regions)} 个区域")
        print(f"      - Low组: {len(low_regions)} 个区域")
        print(f"      - 全局: {len(all_regions)} 个区域")
    else:
        print(f"   ⚠️ 未检测到显著高贡献区域")

    # 11. 全局邻居重要性统计
    print("\n9️⃣ 统计全局邻居重要性...")
    create_global_neighbor_summary(all_results, Config.OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("✨ 分析完成！")
    print(f"📁 结果目录: {Config.OUTPUT_DIR}")
    print(f"🧬 Motif目录: {Config.MOTIF_DIR}")
    print(f"📊 成功分析: {len(all_results)} 个基因")
    print(f"🔍 发现Motif: {len(all_motifs)} 个")
    print("=" * 80)


if __name__ == "__main__":
    main()