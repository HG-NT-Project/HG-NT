"""
完整的序列位置重要性分析脚本
- 分析预测最准的前100个高表达和100个低表达基因
- 生成6000bp热力图
- 识别重要的motif候选区域
- 输出碱基级别的重要性分数
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
from scipy import stats
from scipy.ndimage import gaussian_filter1d
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
    N_HIGH_GENES = 100  # 高表达基因数量
    N_LOW_GENES = 100  # 低表达基因数量
    HIGH_EXPR_THRESHOLD = 5.0  # 高表达阈值
    LOW_EXPR_THRESHOLD = 1.0  # 低表达阈值

    # Motif识别参数
    MOTIF_MIN_LENGTH = 5
    MOTIF_MAX_LENGTH = 20
    IMPORTANCE_PERCENTILE = 95  # 重要性阈值百分位
    MIN_GAP_BETWEEN_MOTIFS = 10  # motif间最小间隔

    # 可视化参数
    HEATMAP_DPI = 300
    SMOOTH_WINDOW = 50  # 平滑窗口大小

    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1  # 逐个处理以保证梯度正确


# ===================== 模型加载 =====================
def load_models():
    """加载NT模型和M3模型"""
    print("🤖 加载模型...")

    # NT模型
    tokenizer = AutoTokenizer.from_pretrained(Config.NT_MODEL_PATH)
    nt_model = AutoModelForMaskedLM.from_pretrained(
        Config.NT_MODEL_PATH,
        torch_dtype=torch.float16
    ).to(Config.DEVICE)
    nt_model.eval()

    # 冻结NT模型
    for param in nt_model.parameters():
        param.requires_grad = False

    # M3模型
    from train_xr_xiaolongxia import ModelM3_MultiGraphConcat
    m3_model = ModelM3_MultiGraphConcat(input_dim=2560).to(Config.DEVICE)

    checkpoint = torch.load(Config.M3_MODEL_PATH, map_location=Config.DEVICE)
    m3_model.load_state_dict(checkpoint['model_state_dict'])
    m3_model.eval()

    print(f"✅ 模型加载完成，设备: {Config.DEVICE}")
    return tokenizer, nt_model, m3_model


# ===================== 数据处理 =====================
class DataManager:
    def __init__(self):
        self.load_data()

    def load_data(self):
        """加载所有必要的数据"""
        print("📊 加载数据...")

        # 1. 基因组
        self.genome = Fasta(Config.GENOME_FA, as_raw=True, sequence_always_upper=True)

        # 2. 注释文件
        self.annotations = self._parse_annotations()

        # 3. 预计算的embeddings
        embed_data = torch.load(Config.EMBEDDINGS_FILE, map_location='cpu')
        self.all_embeddings = embed_data['x']
        self.gene_ids = embed_data['gene_ids']
        self.gene_to_idx = {gid: i for i, gid in enumerate(self.gene_ids)}

        # 4. 预测结果
        self.predictions = pd.read_csv(Config.PREDICTIONS_CSV)

        # 5. 网络结构
        self.tf_edge_index = torch.load("processed_tf/crayfish_tf_edge_index.pt")['edge_index']
        self.gcn_edge_index = torch.load("processed_gcn/crayfish_gcn_network.pt")['edge_index']

        print(f"✅ 数据加载完成: {len(self.gene_ids)} 个基因")

    def _parse_annotations(self):
        """解析注释文件"""
        df = pd.read_csv(Config.ANNO_FILE, sep='\t')
        anno_dict = {}
        for _, row in df.iterrows():
            parts = str(row['GeneID']).split(':')
            if len(parts) >= 5:
                gene_id = parts[0]
                raw_span = parts[3]
                try:
                    if '..' in raw_span:
                        start = int(raw_span.split('..')[0])
                        end = int(raw_span.split('..')[1])
                        strand = parts[4] if len(parts) > 4 else '+'
                    else:
                        start = int(parts[3])
                        end = int(parts[4])
                        strand = parts[5] if len(parts) > 5 else '+'
                except (ValueError, IndexError):
                    continue
                anno_dict[gene_id] = {
                    'chrom': parts[2],
                    'start': start,
                    'end': end,
                    'strand': strand
                }
        return anno_dict

    def get_target_genes(self):
        """筛选预测最准的高表达和低表达基因"""
        df = self.predictions.copy()

        # 计算预测误差
        df['abs_error'] = (df['true_expression'] - df['predicted_expression']).abs()
        df['relative_error'] = df['abs_error'] / (df['true_expression'] + 1e-8)

        # 筛选高表达和低表达基因
        high_df = df[df['true_expression'] >= Config.HIGH_EXPR_THRESHOLD]
        low_df = df[df['true_expression'] <= Config.LOW_EXPR_THRESHOLD]

        # 选择预测最准的（绝对误差最小）
        high_genes = high_df.nsmallest(Config.N_HIGH_GENES, 'abs_error')
        low_genes = low_df.nsmallest(Config.N_LOW_GENES, 'abs_error')

        print(f"\n📊 筛选结果:")
        print(f"  高表达基因 (>={Config.HIGH_EXPR_THRESHOLD}): {len(high_df)} -> 选择 {len(high_genes)} 个")
        print(f"  低表达基因 (<={Config.LOW_EXPR_THRESHOLD}): {len(low_df)} -> 选择 {len(low_genes)} 个")
        print(f"  高表达平均误差: {high_genes['abs_error'].mean():.4f}")
        print(f"  低表达平均误差: {low_genes['abs_error'].mean():.4f}")

        return high_genes, low_genes

    def extract_sequence(self, gene_id):
        """提取6000bp序列"""
        if gene_id not in self.annotations:
            return None

        info = self.annotations[gene_id]
        chrom, start, end, strand = info['chrom'], info['start'], info['end'], info['strand']

        try:
            if strand in ['+', '1', '.']:
                # 正链
                tss_seq = str(self.genome[chrom][max(0, start - 2000):start + 1000])
                tts_seq = str(self.genome[chrom][max(0, end - 1000):end + 2000])
                seq = tss_seq + tts_seq
            else:
                # 负链
                def rc(s):
                    return s[::-1].translate(str.maketrans('ATGCatgc', 'TACGtacg'))

                tss_seq = rc(str(self.genome[chrom][max(0, end - 1000):end + 2000]))
                tts_seq = rc(str(self.genome[chrom][max(0, start - 2000):start + 1000]))
                seq = tss_seq + tts_seq

            # 填充到6000bp
            seq = seq.ljust(6000, 'N')[:6000].upper()
            return seq
        except Exception as e:
            print(f"  ⚠️ 提取序列失败 {gene_id}: {e}")
            return None

    def get_neighbors(self, gene_id):
        """获取基因的图邻居"""
        if gene_id not in self.gene_to_idx:
            return None

        gene_idx = self.gene_to_idx[gene_id]
        neighbors = {'tf': [], 'gcn': []}

        # TF网络邻居
        tf_mask = self.tf_edge_index[1] == gene_idx
        neighbors['tf'] = self.tf_edge_index[0][tf_mask].tolist()[:32]

        # GCN网络邻居
        gcn_mask = self.gcn_edge_index[1] == gene_idx
        neighbors['gcn'] = self.gcn_edge_index[0][gcn_mask].tolist()[:32]

        return neighbors


# ===================== 重要性分析器 =====================
class ImportanceAnalyzer:
    def __init__(self, tokenizer, nt_model, m3_model, data_manager):
        self.tokenizer = tokenizer
        self.nt_model = nt_model
        self.m3_model = m3_model
        self.data_manager = data_manager
        self.device = Config.DEVICE

    def analyze_gene(self, gene_id, group):
        """分析单个基因的序列重要性"""

        # 1. 提取序列
        sequence = self.data_manager.extract_sequence(gene_id)
        if sequence is None:
            return None

        # 2. 获取邻居
        neighbors = self.data_manager.get_neighbors(gene_id)
        if neighbors is None:
            return None

        # 3. Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=1000,
            padding='max_length',
            truncation=True
        ).to(self.device)

        # 4. 计算梯度
        try:
            with torch.enable_grad():
                # 获取输入嵌入并启用梯度
                input_embeds = self.nt_model.get_input_embeddings()(inputs['input_ids']).clone().detach().requires_grad_(
                    True)

                # NT模型前向传播
                nt_outputs = self.nt_model(
                    inputs_embeds=input_embeds,
                    attention_mask=inputs['attention_mask'],
                    output_hidden_states=True
                )

                # 平均池化
                last_hidden = nt_outputs.hidden_states[-1]
                mask = inputs['attention_mask'].unsqueeze(-1).float()
                gene_embedding = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

                # 构建子图
                gene_idx = self.data_manager.gene_to_idx[gene_id]
                all_nodes = set([gene_idx])
                all_nodes.update(neighbors['tf'])
                all_nodes.update(neighbors['gcn'])
                all_nodes = torch.tensor(list(all_nodes), dtype=torch.long)

                # 节点映射
                node_mapping = {old.item(): new for new, old in enumerate(all_nodes)}
                center_new = node_mapping[gene_idx]

                # 准备图输入
                x = self.data_manager.all_embeddings[all_nodes].clone().to(self.device)
                x[center_new] = gene_embedding.squeeze(0)

                # 提取子图边
                tf_sub = self._extract_subgraph_edges(
                    self.data_manager.tf_edge_index, all_nodes, node_mapping
                )
                gcn_sub = self._extract_subgraph_edges(
                    self.data_manager.gcn_edge_index, all_nodes, node_mapping
                )

                # M3模型预测
                self.m3_model.set_subgraphs(tf_sub.to(self.device), gcn_sub.to(self.device))
                pred = self.m3_model(x)[center_new]

                # 反向传播
                self.m3_model.zero_grad()
                self.nt_model.zero_grad()
                pred.backward()

                # 获取梯度
                token_gradients = input_embeds.grad.abs().sum(dim=-1).squeeze(0).cpu().numpy()

        except Exception as e:
            print(f"  ❌ 梯度计算失败 {gene_id}: {e}")
            return None

        # 5. 扩展到碱基级别
        base_gradients = np.repeat(token_gradients, 6)[:6000]

        # 6. 平滑处理
        smoothed = gaussian_filter1d(base_gradients, sigma=Config.SMOOTH_WINDOW / 3)

        return {
            'gene_id': gene_id,
            'group': group,
            'sequence': sequence,
            'prediction': pred.item(),
            'base_gradients': base_gradients,
            'smoothed_gradients': smoothed,
            'true_expression': self.data_manager.predictions[
                self.data_manager.predictions['gene_id'] == gene_id
                ]['true_expression'].values[0]
        }

    def _extract_subgraph_edges(self, edge_index, nodes, node_mapping):
        """提取子图边"""
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        mask = torch.isin(edge_index[0], nodes) & torch.isin(edge_index[1], nodes)
        if mask.sum() == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        sub_edges = edge_index[:, mask].clone()
        sub_edges[0] = torch.tensor([node_mapping[idx.item()] for idx in sub_edges[0]])
        sub_edges[1] = torch.tensor([node_mapping[idx.item()] for idx in sub_edges[1]])

        return sub_edges


# ===================== Motif识别器 =====================
class MotifFinder:
    def __init__(self):
        self.all_motifs = []

    def find_motifs(self, result, threshold_percentile=95):
        """识别重要的motif区域"""
        gradients = result['smoothed_gradients']
        sequence = result['sequence']

        # 计算阈值
        threshold = np.percentile(gradients, threshold_percentile)

        # 找出高于阈值的连续区域
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
                    # 计算motif的特征
                    motif_seq = sequence[start:end]
                    mean_importance = gradients[start:end].mean()
                    max_importance = gradients[start:end].max()

                    # 确定区域类型
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
                        'mean_importance': mean_importance,
                        'max_importance': max_importance,
                        'region': region,
                        'threshold_used': threshold
                    })
            else:
                i += 1

        # 合并相近的motifs
        merged_motifs = self._merge_nearby_motifs(motifs)

        return merged_motifs

    def _merge_nearby_motifs(self, motifs):
        """合并相近的motif"""
        if len(motifs) <= 1:
            return motifs

        merged = []
        motifs = sorted(motifs, key=lambda x: x['start'])

        current = motifs[0].copy()
        for next_motif in motifs[1:]:
            if next_motif['start'] - current['end'] <= Config.MIN_GAP_BETWEEN_MOTIFS:
                # 合并
                current['end'] = next_motif['end']
                current['length'] = current['end'] - current['start']
                current['sequence'] = current['sequence'] + next_motif['sequence'][
                    current['end'] - next_motif['start']:]
                current['mean_importance'] = max(current['mean_importance'], next_motif['mean_importance'])
                current['max_importance'] = max(current['max_importance'], next_motif['max_importance'])
            else:
                merged.append(current)
                current = next_motif.copy()

        merged.append(current)
        return merged

    def analyze_all_motifs(self, all_results):
        """分析所有motif的统计特征"""
        all_motifs = []
        for result in all_results:
            motifs = self.find_motifs(result)
            all_motifs.extend(motifs)

        # 转换为DataFrame
        df = pd.DataFrame(all_motifs)

        if len(df) == 0:
            return df, {}

        # 统计信息
        stats = {
            'total_motifs': len(df),
            'avg_length': df['length'].mean(),
            'avg_importance': df['mean_importance'].mean(),
            'high_group_count': len(df[df['group'] == 'High']),
            'low_group_count': len(df[df['group'] == 'Low']),
            'region_distribution': df['region'].value_counts().to_dict()
        }

        # 找出高频motif序列
        sequence_counts = df['sequence'].value_counts()
        stats['top_sequences'] = sequence_counts.head(10).to_dict()

        return df, stats


# ===================== 可视化器 =====================
class Visualizer:
    def __init__(self, output_dir, motif_dir):
        self.output_dir = output_dir
        self.motif_dir = motif_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(motif_dir, exist_ok=True)

    def plot_heatmap(self, all_results):
        """生成6000bp热力图"""
        print("\n🎨 生成热力图...")

        # 按组分离数据
        high_results = [r for r in all_results if r['group'] == 'High']
        low_results = [r for r in all_results if r['group'] == 'Low']

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 1. 高表达组热图
        if high_results:
            high_data = np.array([r['smoothed_gradients'] for r in high_results])
            # 按表达量排序
            high_expr = [r['true_expression'] for r in high_results]
            sort_idx = np.argsort(high_expr)
            high_data = high_data[sort_idx]

            im1 = axes[0, 0].imshow(high_data, aspect='auto', cmap='hot', interpolation='bilinear')
            axes[0, 0].set_title(f'High Expression Genes (n={len(high_results)})', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('Genes (sorted by expression)')
            axes[0, 0].axvline(x=2000, color='white', linestyle='--', linewidth=1, alpha=0.5)
            axes[0, 0].axvline(x=4000, color='white', linestyle='--', linewidth=1, alpha=0.5)
            plt.colorbar(im1, ax=axes[0, 0], label='Importance')

            # 添加区域标注
            axes[0, 0].text(1000, -2, 'TSS Up', ha='center', fontsize=10, color='white')
            axes[0, 0].text(2500, -2, 'TSS Down', ha='center', fontsize=10, color='white')
            axes[0, 0].text(3500, -2, 'TTS Up', ha='center', fontsize=10, color='white')
            axes[0, 0].text(5000, -2, 'TTS Down', ha='center', fontsize=10, color='white')

        # 2. 低表达组热图
        if low_results:
            low_data = np.array([r['smoothed_gradients'] for r in low_results])
            # 按表达量排序
            low_expr = [r['true_expression'] for r in low_results]
            sort_idx = np.argsort(low_expr)[::-1]  # 降序
            low_data = low_data[sort_idx]

            im2 = axes[0, 1].imshow(low_data, aspect='auto', cmap='viridis', interpolation='bilinear')
            axes[0, 1].set_title(f'Low Expression Genes (n={len(low_results)})', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Genes (sorted by expression)')
            axes[0, 1].axvline(x=2000, color='white', linestyle='--', linewidth=1, alpha=0.5)
            axes[0, 1].axvline(x=4000, color='white', linestyle='--', linewidth=1, alpha=0.5)
            plt.colorbar(im2, ax=axes[0, 1], label='Importance')

            axes[0, 1].text(1000, -2, 'TSS Up', ha='center', fontsize=10, color='white')
            axes[0, 1].text(2500, -2, 'TSS Down', ha='center', fontsize=10, color='white')
            axes[0, 1].text(3500, -2, 'TTS Up', ha='center', fontsize=10, color='white')
            axes[0, 1].text(5000, -2, 'TTS Down', ha='center', fontsize=10, color='white')

        # 3. 平均重要性曲线对比
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

            axes[1, 0].axvline(x=2000, color='black', linestyle='--', linewidth=1, alpha=0.5)
            axes[1, 0].axvline(x=4000, color='black', linestyle='--', linewidth=1, alpha=0.5)

            axes[1, 0].set_xlabel('Position (bp)', fontsize=12)
            axes[1, 0].set_ylabel('Mean Importance', fontsize=12)
            axes[1, 0].set_title('Average Importance Profile Comparison', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 添加区域背景
            axes[1, 0].axvspan(0, 2000, alpha=0.1, color='blue')
            axes[1, 0].axvspan(2000, 3000, alpha=0.1, color='green')
            axes[1, 0].axvspan(3000, 4000, alpha=0.1, color='orange')
            axes[1, 0].axvspan(4000, 6000, alpha=0.1, color='red')

        # 4. 差异分析
        if high_results and low_results:
            high_mean = np.mean([r['smoothed_gradients'] for r in high_results], axis=0)
            low_mean = np.mean([r['smoothed_gradients'] for r in low_results], axis=0)
            difference = high_mean - low_mean

            # 统计显著性
            high_data = np.array([r['smoothed_gradients'] for r in high_results])
            low_data = np.array([r['smoothed_gradients'] for r in low_results])

            p_values = []
            for i in range(6000):
                _, p = stats.ttest_ind(high_data[:, i], low_data[:, i])
                p_values.append(p)
            p_values = np.array(p_values)

            axes[1, 1].plot(x, difference, 'g-', linewidth=2, label='Difference (High - Low)')
            axes[1, 1].fill_between(x, 0, difference, where=(p_values < 0.05),
                                    alpha=0.3, color='green', label='p < 0.05')

            axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1, 1].axvline(x=2000, color='black', linestyle='--', linewidth=1, alpha=0.5)
            axes[1, 1].axvline(x=4000, color='black', linestyle='--', linewidth=1, alpha=0.5)

            axes[1, 1].set_xlabel('Position (bp)', fontsize=12)
            axes[1, 1].set_ylabel('Importance Difference', fontsize=12)
            axes[1, 1].set_title('Differential Importance (High - Low)', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # 添加区域背景
            axes[1, 1].axvspan(0, 2000, alpha=0.1, color='blue')
            axes[1, 1].axvspan(2000, 3000, alpha=0.1, color='green')
            axes[1, 1].axvspan(3000, 4000, alpha=0.1, color='orange')
            axes[1, 1].axvspan(4000, 6000, alpha=0.1, color='red')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_heatmap_analysis.png'),
                    dpi=Config.HEATMAP_DPI, bbox_inches='tight')
        plt.close()

        print(f"✅ 热力图已保存到 {self.output_dir}")

    def plot_individual_genes(self, results, max_genes=20):
        """绘制单个基因的重要性曲线（示例）"""
        print(f"\n📈 绘制单个基因示例 (最多{max_genes}个)...")

        # 选择示例基因
        high_samples = [r for r in results if r['group'] == 'High'][:max_genes // 2]
        low_samples = [r for r in results if r['group'] == 'Low'][:max_genes // 2]
        samples = high_samples + low_samples

        n_cols = 4
        n_rows = (len(samples) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()

        for i, result in enumerate(samples):
            ax = axes[i]

            x = np.arange(6000)
            color = 'red' if result['group'] == 'High' else 'blue'

            ax.plot(x, result['smoothed_gradients'], color=color, linewidth=1)
            ax.fill_between(x, 0, result['smoothed_gradients'], alpha=0.3, color=color)

            ax.axvline(x=2000, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axvline(x=4000, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

            ax.set_title(f"{result['gene_id']} ({result['group']})\n"
                         f"Pred: {result['prediction']:.3f}, True: {result['true_expression']:.3f}",
                         fontsize=10)
            ax.set_xlabel('Position (bp)', fontsize=8)
            ax.set_ylabel('Importance', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(len(samples), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'individual_gene_examples.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 单个基因示例已保存")


# ===================== 主分析流程 =====================
def main():
    print("=" * 80)
    print("🧬 序列重要性分析与Motif识别")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.MOTIF_DIR, exist_ok=True)

    # 1. 加载模型和数据
    tokenizer, nt_model, m3_model = load_models()
    data_manager = DataManager()

    # 2. 获取目标基因
    high_genes, low_genes = data_manager.get_target_genes()

    # 3. 创建分析器
    analyzer = ImportanceAnalyzer(tokenizer, nt_model, m3_model, data_manager)
    motif_finder = MotifFinder()
    visualizer = Visualizer(Config.OUTPUT_DIR, Config.MOTIF_DIR)

    # 4. 分析所有基因
    all_results = []

    print(f"\n🔬 开始分析 {len(high_genes) + len(low_genes)} 个基因...")

    for group, genes_df in [('High', high_genes), ('Low', low_genes)]:
        print(f"\n📊 处理 {group} 表达组 ({len(genes_df)} 个基因)...")

        for _, row in tqdm(genes_df.iterrows(), total=len(genes_df), desc=f"{group}组"):
            gene_id = row['gene_id']

            result = analyzer.analyze_gene(gene_id, group)
            if result:
                all_results.append(result)

                # 保存碱基级别重要性
                gene_dir = os.path.join(Config.OUTPUT_DIR, 'gene_data', gene_id)
                os.makedirs(gene_dir, exist_ok=True)

                np.save(os.path.join(gene_dir, 'base_importance.npy'), result['base_gradients'])
                np.save(os.path.join(gene_dir, 'smoothed_importance.npy'), result['smoothed_gradients'])

                # 保存序列
                with open(os.path.join(gene_dir, 'sequence.fa'), 'w') as f:
                    f.write(f">{gene_id}\n{result['sequence']}\n")

                # 保存元数据
                with open(os.path.join(gene_dir, 'metadata.json'), 'w') as f:
                    json.dump({
                        'gene_id': result['gene_id'],
                        'group': result['group'],
                        'prediction': float(result['prediction']),
                        'true_expression': float(result['true_expression'])
                    }, f, indent=2)

    print(f"\n✅ 成功分析 {len(all_results)} 个基因")

    # 5. 生成可视化
    visualizer.plot_heatmap(all_results)
    visualizer.plot_individual_genes(all_results)

    # 6. Motif识别
    print("\n🔍 识别重要Motif区域...")
    motif_df, motif_stats = motif_finder.analyze_all_motifs(all_results)

    if len(motif_df) > 0:
        # 保存motif结果
        motif_df.to_csv(os.path.join(Config.MOTIF_DIR, 'all_candidate_motifs.csv'), index=False)

        # 按重要性排序，保存top motifs
        top_motifs = motif_df.nlargest(100, 'mean_importance')
        top_motifs.to_csv(os.path.join(Config.MOTIF_DIR, 'top_100_motifs.csv'), index=False)

        # 分别保存高低表达组的motifs
        high_motifs = motif_df[motif_df['group'] == 'High']
        low_motifs = motif_df[motif_df['group'] == 'Low']

        high_motifs.to_csv(os.path.join(Config.MOTIF_DIR, 'high_expression_motifs.csv'), index=False)
        low_motifs.to_csv(os.path.join(Config.MOTIF_DIR, 'low_expression_motifs.csv'), index=False)

        # 保存FASTA格式的motif序列
        with open(os.path.join(Config.MOTIF_DIR, 'candidate_motifs.fa'), 'w') as f:
            for _, row in motif_df.iterrows():
                f.write(
                    f">{row['gene_id']}_{row['start']}_{row['end']}_{row['region']}_importance_{row['mean_importance']:.4f}\n")
                f.write(f"{row['sequence']}\n")

        # 保存统计信息
        with open(os.path.join(Config.MOTIF_DIR, 'motif_statistics.json'), 'w') as f:
            json.dump(motif_stats, f, indent=2)

        print(f"\n📊 Motif统计:")
        print(f"  总计发现: {motif_stats['total_motifs']} 个motif")
        print(f"  平均长度: {motif_stats['avg_length']:.1f} bp")
        print(f"  平均重要性: {motif_stats['avg_importance']:.4f}")
        print(f"  高表达组: {motif_stats['high_group_count']} 个")
        print(f"  低表达组: {motif_stats['low_group_count']} 个")

        print(f"\n📈 区域分布:")
        for region, count in motif_stats['region_distribution'].items():
            print(f"  {region}: {count} 个")

        print(f"\n🔝 Top 10 高频序列:")
        for seq, count in list(motif_stats['top_sequences'].items())[:10]:
            print(f"  {seq}: {count} 次")

    # 7. 保存总体结果
    summary_df = pd.DataFrame([{
        'gene_id': r['gene_id'],
        'group': r['group'],
        'prediction': r['prediction'],
        'true_expression': r['true_expression'],
        'mean_importance': r['smoothed_gradients'].mean(),
        'max_importance': r['smoothed_gradients'].max(),
        'tss_upstream_importance': r['smoothed_gradients'][0:2000].mean(),
        'tss_downstream_importance': r['smoothed_gradients'][2000:3000].mean(),
        'tts_upstream_importance': r['smoothed_gradients'][3000:4000].mean(),
        'tts_downstream_importance': r['smoothed_gradients'][4000:6000].mean()
    } for r in all_results])

    summary_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'gene_importance_summary.csv'), index=False)

    # 8. 生成报告
    report = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'total_genes_analyzed': len(all_results),
        'high_expression_genes': len([r for r in all_results if r['group'] == 'High']),
        'low_expression_genes': len([r for r in all_results if r['group'] == 'Low']),
        'average_prediction_accuracy': {
            'high': summary_df[summary_df['group'] == 'High']['prediction'].corr(
                summary_df[summary_df['group'] == 'High']['true_expression']
            ),
            'low': summary_df[summary_df['group'] == 'Low']['prediction'].corr(
                summary_df[summary_df['group'] == 'Low']['true_expression']
            )
        },
        'motif_statistics': motif_stats,
        'config': {
            'n_high_genes': Config.N_HIGH_GENES,
            'n_low_genes': Config.N_LOW_GENES,
            'high_expr_threshold': Config.HIGH_EXPR_THRESHOLD,
            'low_expr_threshold': Config.LOW_EXPR_THRESHOLD,
            'motif_min_length': Config.MOTIF_MIN_LENGTH,
            'motif_max_length': Config.MOTIF_MAX_LENGTH,
            'importance_percentile': Config.IMPORTANCE_PERCENTILE
        }
    }

    with open(os.path.join(Config.OUTPUT_DIR, 'analysis_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("✨ 分析完成！")
    print(f"📁 结果保存在: {Config.OUTPUT_DIR}")
    print(f"🧬 Motif候选在: {Config.MOTIF_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()