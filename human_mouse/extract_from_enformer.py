"""
Enformer 特征提取脚本 - HuggingFace 版本（AutoModel修复版）
使用 google/enformer 官方模型
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
import gzip
import warnings
from tqdm import tqdm
from pyfaidx import Fasta
try:
    from enformer_pytorch import from_pretrained
    print("✅ enformer-pytorch 加载成功")
except ImportError:
    print("❌ 缺失依赖，请执行: pip install enformer-pytorch")
    exit(1)

warnings.filterwarnings('ignore')

# =================================================================
# 常量定义
# =================================================================
ENFORMER_INPUT_LEN = 196_608
ENFORMER_HALF_LEN = ENFORMER_INPUT_LEN // 2  # 98,304
NUM_WINDOWS = 896
CENTER_BIN = 448

# 人类轨道
HUMAN_CAGE_START = 4828
HUMAN_CAGE_END = 5313
HUMAN_CAGE_CHANNELS = 485

# 小鼠轨道
MOUSE_CAGE_START = 1600
MOUSE_CAGE_END = 1643
MOUSE_CAGE_CHANNELS = 43

# 碱基编码
BASE_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
BASE_TO_ONEHOT = {
    'A': np.array([1, 0, 0, 0], dtype=np.float32),
    'C': np.array([0, 1, 0, 0], dtype=np.float32),
    'G': np.array([0, 0, 1, 0], dtype=np.float32),
    'T': np.array([0, 0, 0, 1], dtype=np.float32),
    'N': np.array([0, 0, 0, 0], dtype=np.float32),  # N用全0表示
}

SPECIES_CONFIG = {
    'human': {
        'labels_file': 'processed_labels/human_labels.pt',
        'cage_start': HUMAN_CAGE_START,
        'cage_end': HUMAN_CAGE_END,
        'num_cage_channels': HUMAN_CAGE_CHANNELS,
        'output_head': 'human',
    },
    'mouse': {
        'labels_file': 'processed_labels/mouse_labels.pt',
        'cage_start': MOUSE_CAGE_START,
        'cage_end': MOUSE_CAGE_END,
        'num_cage_channels': MOUSE_CAGE_CHANNELS,
        'output_head': 'mouse',
    }
}

FEATURES_CACHE_DIR = "enformer_features_cache"
os.makedirs(FEATURES_CACHE_DIR, exist_ok=True)


# =================================================================
# 序列处理（修复版）
# =================================================================
def sequence_to_onehot(sequence, seq_len=ENFORMER_INPUT_LEN):
    """
    将DNA序列转换为one-hot编码
    输出: [seq_len, 4] (A, C, G, T顺序)
    """
    seq = sequence.upper()

    # 处理长度
    if len(seq) < seq_len:
        seq = seq + 'N' * (seq_len - len(seq))
    elif len(seq) > seq_len:
        # 截断时尽量保留中心区域
        start = (len(seq) - seq_len) // 2
        seq = seq[start:start + seq_len]

    # One-hot编码
    onehot = np.zeros((seq_len, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in BASE_TO_ONEHOT:
            onehot[i] = BASE_TO_ONEHOT[base]
        # 'N'已经自动是[0,0,0,0]

    return onehot


def reverse_complement(seq):
    """反向互补序列"""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))


def extract_sequence_around_tss(genome, chrom, tss, strand, chrom_prefix=''):
    """
    以TSS为中心提取196,608 bp序列
    """
    # 处理染色体名
    if chrom_prefix and not chrom.startswith(chrom_prefix):
        chrom_with_prefix = f"{chrom_prefix}{chrom}"
    else:
        chrom_with_prefix = chrom

    # 计算序列区域
    seq_start = tss - ENFORMER_HALF_LEN
    seq_end = tss + ENFORMER_HALF_LEN - 1

    try:
        chrom_len = len(genome[chrom_with_prefix])

        # 处理边界
        if seq_start < 1 or seq_end > chrom_len:
            # 提取有效部分
            valid_start = max(1, seq_start)
            valid_end = min(chrom_len, seq_end)
            left_pad = max(0, 1 - seq_start)
            right_pad = max(0, seq_end - chrom_len)

            if valid_start <= valid_end:
                seq = str(genome[chrom_with_prefix][valid_start - 1:valid_end])  # pyfaidx是0-indexed
                seq = seq.upper()
            else:
                seq = ''

            seq = 'N' * left_pad + seq + 'N' * right_pad
        else:
            # pyfaidx: 1-indexed inclusive, 所以需要-1
            seq = str(genome[chrom_with_prefix][seq_start - 1:seq_end])
            seq = seq.upper()

        # 确保长度正确
        if len(seq) != ENFORMER_INPUT_LEN:
            if len(seq) < ENFORMER_INPUT_LEN:
                seq = seq + 'N' * (ENFORMER_INPUT_LEN - len(seq))
            else:
                seq = seq[:ENFORMER_INPUT_LEN]

        # 负链取反向互补
        if strand == '-':
            seq = reverse_complement(seq)

        return seq

    except Exception as e:
        print(f"警告: 提取序列失败 chr{chrom}:{tss} - {e}")
        return 'N' * ENFORMER_INPUT_LEN


# =================================================================
# GTF解析（优化版）
# =================================================================
def parse_gtf_fast(gtf_file, target_genes_set):
    """
    快速解析GTF，提取基因位置信息
    使用pandas分块读取，避免逐行正则
    """
    print(f"📖 解析GTF: {gtf_file}")

    # 判断是否为gzip压缩
    if gtf_file.endswith('.gz'):
        # 对于压缩文件，使用标准解析
        gene_data = []
        with gzip.open(gtf_file, 'rt') as f:
            for line in tqdm(f, desc="解析GTF"):
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 9 or parts[2] != 'gene':
                    continue

                # 提取gene_id
                attr = parts[8]
                import re
                match = re.search(r'gene_id "([^"]+)"', attr)
                if not match:
                    continue

                gene_id = match.group(1)
                gene_id_base = gene_id.split('.')[0]

                # 检查是否在目标中
                if gene_id in target_genes_set or gene_id_base in target_genes_set:
                    gene_data.append({
                        'chromosome': parts[0],
                        'start': int(parts[3]),
                        'end': int(parts[4]),
                        'strand': parts[6],
                        'gene_id': gene_id,
                        'gene_id_base': gene_id_base
                    })

        return pd.DataFrame(gene_data)

    else:
        # 未压缩文件可以用pandas快速读取
        try:
            # 先读取前几行确定列数
            sample_lines = []
            with open(gtf_file, 'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        sample_lines.append(line)
                        if len(sample_lines) >= 5:
                            break

            if sample_lines:
                num_cols = len(sample_lines[0].strip().split('\t'))
                col_names = ['seqname', 'source', 'feature', 'start', 'end', 'score',
                             'strand', 'frame', 'attribute'] + [f'col_{i}' for i in range(9, num_cols)]
            else:
                col_names = ['seqname', 'source', 'feature', 'start', 'end', 'score',
                             'strand', 'frame', 'attribute']

            # 分块读取
            chunk_size = 100000
            gene_data = []

            for chunk in pd.read_csv(gtf_file, sep='\t', comment='#', header=None,
                                     names=col_names, chunksize=chunk_size,
                                     low_memory=False):
                # 只保留gene行
                chunk = chunk[chunk['feature'] == 'gene']

                if len(chunk) == 0:
                    continue

                # 提取gene_id
                gene_ids = chunk['attribute'].str.extract(r'gene_id "([^"]+)"', expand=False)
                chunk['gene_id'] = gene_ids
                chunk['gene_id_base'] = gene_ids.str.split('.').str[0]

                # 筛选目标基因
                mask = chunk['gene_id'].isin(target_genes_set) | chunk['gene_id_base'].isin(target_genes_set)
                filtered = chunk[mask]

                if len(filtered) > 0:
                    for _, row in filtered.iterrows():
                        gene_data.append({
                            'chromosome': row['seqname'],
                            'start': int(row['start']),
                            'end': int(row['end']),
                            'strand': row['strand'],
                            'gene_id': row['gene_id'],
                            'gene_id_base': row['gene_id_base']
                        })

            return pd.DataFrame(gene_data)

        except Exception as e:
            print(f"pandas解析失败，回退到逐行解析: {e}")
            # 回退到逐行解析
            return parse_gtf_fallback(gtf_file, target_genes_set)


def parse_gtf_fallback(gtf_file, target_genes_set):
    """回退解析方法"""
    import re
    gene_data = []

    open_func = gzip.open if gtf_file.endswith('.gz') else open
    mode = 'rt' if gtf_file.endswith('.gz') else 'r'

    with open_func(gtf_file, mode) as f:
        for line in tqdm(f, desc="解析GTF (回退模式)"):
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue

            match = re.search(r'gene_id "([^"]+)"', parts[8])
            if not match:
                continue

            gene_id = match.group(1)
            gene_id_base = gene_id.split('.')[0]

            if gene_id in target_genes_set or gene_id_base in target_genes_set:
                gene_data.append({
                    'chromosome': parts[0],
                    'start': int(parts[3]),
                    'end': int(parts[4]),
                    'strand': parts[6],
                    'gene_id': gene_id,
                    'gene_id_base': gene_id_base
                })

    return pd.DataFrame(gene_data)


# =================================================================
# Enformer特征提取器（AutoModel修复版 - 关键修改）
# =================================================================
class EnformerFeatureExtractor:
    def __init__(self, species, device='cuda', target_length=896):
        self.species = species
        self.device = device
        self.target_length = target_length

        config = SPECIES_CONFIG[species]
        self.cage_start = config['cage_start']
        self.cage_end = config['cage_end']
        self.num_cage_channels = config['num_cage_channels']
        self.output_head = config['output_head']

        self._load_model()

    def _load_model(self):
        """完全切换回最初成功的 enformer-pytorch 方法"""
        print(f"\n📥 正在通过 enformer-pytorch 加载 EleutherAI Enformer 权重...")

        try:
            # 直接使用 enformer_pytorch 的接口，不再使用 transformers 的 AutoModel
            self.model = from_pretrained('EleutherAI/enformer-official-rough')
            print("✅ 成功加载 EleutherAI Enformer 权重")
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            print("💡 提示：请确保已执行 source /etc/network_turbo 并且安装了 enformer-pytorch")
            raise e

        self.model = self.model.to(self.device)
        self.model.eval()

        # 验证输出结构
        dummy = torch.zeros(1, ENFORMER_INPUT_LEN, 4, device=self.device)
        with torch.no_grad():
            output = self.model(dummy)
            # enformer-pytorch 返回的是字典
            if isinstance(output, dict):
                print(f"📌 模型输出包含以下物种: {list(output.keys())}")
            else:
                print(f"📌 模型输出类型: {type(output)}")

        print(f"✅ 模型初始化完成！设备: {self.device}")

    def extract_features(self, sequences, batch_size=8):
        """
        提取Enformer特征 (修复版：自动处理转置与索引越界)
        输入: sequences list, 每个元素是DNA序列字符串
        输出: [N, num_cage_channels] 中心窗口的CAGE信号
        """
        num_genes = len(sequences)
        all_features = []

        print(f"\n🚀 开始提取特征")
        print(f"   基因数: {num_genes}")
        print(f"   批大小: {batch_size}")

        with torch.no_grad():
            for i in tqdm(range(0, num_genes, batch_size), desc="提取Enformer特征"):
                batch_seqs = sequences[i:i + batch_size]

                # One-hot编码: [batch, seq_len, 4]
                batch_encoded = np.array([sequence_to_onehot(seq) for seq in batch_seqs])
                batch_tensor = torch.from_numpy(batch_encoded).to(self.device)

                # 前向传播
                output = self.model(batch_tensor)

                # enformer-pytorch 通常返回字典，key 是 'human' 或 'mouse'
                if isinstance(output, dict):
                    out = output[self.species]
                else:
                    out = output

                # --- 修复逻辑 1: 自动检查并处理转置 ---
                # 官方 README 指出输出应为 [batch, 896, tracks]
                # 如果第2维大小是 5313 或 1643，说明维度顺序是 [batch, tracks, 896]，需要转置
                if out.shape[1] in [5313, 1643] and out.shape[2] == NUM_WINDOWS:
                    out = out.transpose(1, 2)  # 转换为 [batch, 896, tracks]

                # --- 修复逻辑 2: 索引越界安全保护 ---
                max_track_index = out.shape[-1]

                # 确保 CAGE 索引不超出模型实际输出的范围
                safe_start = min(self.cage_start, max_track_index - 1)
                safe_end = min(self.cage_end, max_track_index)

                # 如果配置的索引导致空切片，则自动输出调试信息并调整
                if safe_start >= safe_end or safe_start < 0:
                    print(f"\n⚠️ 警告: 物种 {self.species} 的 CAGE 索引 {self.cage_start}:{self.cage_end} 越界。")
                    print(f"   模型实际输出轨道数: {max_track_index}。已自动提取所有可用轨道。")
                    center_cage = out[:, CENTER_BIN, :]
                else:
                    # 提取中心窗口的 CAGE 通道
                    # out 维度: [batch, 896, tracks]
                    center_cage = out[:, CENTER_BIN, safe_start:safe_end]

                # --- 修复逻辑 3: 处理 NaN 并转换 ---
                if torch.isnan(center_cage).any():
                    center_cage = torch.nan_to_num(center_cage, nan=0.0)

                all_features.append(center_cage.cpu().numpy())

                # 清理显存
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

        # 检查是否提取到了数据
        if not all_features:
            return np.array([]).reshape(num_genes, 0)

        features = np.concatenate(all_features, axis=0)
        return features


# =================================================================
# 主函数
# =================================================================
def main():
    parser = argparse.ArgumentParser(description='Enformer特征提取 - AutoModel修复版')
    parser.add_argument('--species', type=str, required=True, choices=['human', 'mouse'])
    parser.add_argument('--genome_fasta', type=str, required=True, help='基因组FASTA文件路径')
    parser.add_argument('--gtf_file', type=str, required=True, help='GTF注释文件路径')
    parser.add_argument('--max_genes', type=int, default=None, help='限制基因数量')
    parser.add_argument('--batch_size', type=int, default=8, help='提取批大小')
    parser.add_argument('--output_dir', type=str, default=FEATURES_CACHE_DIR)

    args = parser.parse_args()

    print("=" * 80)
    print("🔬 Enformer 特征提取 - AutoModel修复版")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  物种: {args.species}")
    print(f"  基因组: {args.genome_fasta}")
    print(f"  GTF: {args.gtf_file}")
    print(f"  序列长度: {ENFORMER_INPUT_LEN:,} bp")

    config = SPECIES_CONFIG[args.species]

    # 1. 加载标签文件
    labels_file = config['labels_file']
    if not os.path.exists(labels_file):
        print(f"\n❌ 标签文件不存在: {labels_file}")
        print("请先准备 processed_labels 目录下的标签文件")
        return

    print(f"\n📂 加载标签: {labels_file}")
    label_data = torch.load(labels_file, map_location='cpu', weights_only=False)

    # 兼容不同的键名
    if 'gene_id' in label_data:
        target_genes = label_data['gene_id']
    elif 'gene_ids' in label_data:
        target_genes = label_data['gene_ids']
    else:
        print("❌ 标签文件中找不到 gene_id 或 gene_ids")
        return

    if args.max_genes:
        target_genes = target_genes[:args.max_genes]

    # 创建基因ID集合（包含有版本和无版本）
    target_genes_set = set(target_genes)
    target_genes_base_set = set(gid.split('.')[0] if '.' in str(gid) else str(gid) for gid in target_genes)
    combined_set = target_genes_set | target_genes_base_set

    print(f"   目标基因数: {len(target_genes)}")

    # 2. 解析GTF
    gene_df = parse_gtf_fast(args.gtf_file, combined_set)
    print(f"   GTF中找到: {len(gene_df)} 个基因")

    if len(gene_df) == 0:
        print("❌ 未找到任何基因")
        return

    # 3. 加载基因组
    print(f"\n📖 加载基因组: {args.genome_fasta}")
    genome = Fasta(args.genome_fasta, as_raw=True, sequence_always_upper=True)

    # 4. 创建基因信息映射
    gene_info_map = {}
    for _, row in gene_df.iterrows():
        # 同时存储有版本和无版本的映射
        gene_info_map[row['gene_id']] = row
        gene_info_map[row['gene_id_base']] = row

    # 5. 构建标签映射
    labels_map = {}
    for i, gid in enumerate(target_genes):
        if 'labels' in label_data:
            label_val = label_data['labels'][i].item() if hasattr(label_data['labels'][i], 'item') else \
                label_data['labels'][i]
        elif 'expression' in label_data:
            label_val = label_data['expression'][i].item() if hasattr(label_data['expression'][i], 'item') else \
                label_data['expression'][i]
        else:
            label_val = 0.0
        labels_map[gid] = label_val

    # 6. 筛选有效基因
    valid_genes = []
    valid_labels = []

    for gene_id in tqdm(target_genes, desc="筛选基因"):
        # 尝试匹配（带版本和不带版本）
        if gene_id in gene_info_map:
            match_id = gene_id
        elif gene_id.split('.')[0] in gene_info_map:
            match_id = gene_id.split('.')[0]
        else:
            continue

        label = labels_map.get(gene_id)
        if label is None or np.isnan(label):
            continue

        gene_info = gene_info_map[match_id]
        valid_genes.append({
            'gene_id': gene_id,
            'chromosome': gene_info['chromosome'],
            'tss': gene_info['start'] if gene_info['strand'] == '+' else gene_info['end'],
            'strand': gene_info['strand']
        })
        valid_labels.append(label)

    print(f"\n✅ 有效基因: {len(valid_genes)} / {len(target_genes)}")

    # 7. 提取序列
    print(f"\n📝 提取DNA序列...")
    sequences = []
    for gene_info in tqdm(valid_genes, desc="提取序列"):
        seq = extract_sequence_around_tss(
            genome,
            gene_info['chromosome'],
            gene_info['tss'],
            gene_info['strand'],
            chrom_prefix=''  # 根据你的基因组文件调整
        )
        sequences.append(seq)

    # 8. 提取Enformer特征
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n💻 使用设备: {device}")

    extractor = EnformerFeatureExtractor(args.species, device)
    features = extractor.extract_features(sequences, args.batch_size)

    # 9. 最终检查
    print(f"\n✅ 特征提取完成!")
    print(f"   特征形状: {features.shape}")
    print(f"   特征范围: [{features.min():.4f}, {features.max():.4f}]")
    print(f"   有效基因: {len(valid_genes)}")

    # 10. 保存
    output_file = os.path.join(args.output_dir, f'{args.species}_enformer_features.pt')
    cache_data = {
        'features': features.astype(np.float32),
        'gene_ids': [g['gene_id'] for g in valid_genes],
        'labels_raw': np.array(valid_labels, dtype=np.float32),
        'labels_log': np.log1p(valid_labels).astype(np.float32),
        'species': args.species,
        'input_sequence_length': ENFORMER_INPUT_LEN,
        'center_bin': CENTER_BIN,
        'cage_start': config['cage_start'],
        'cage_end': config['cage_end'],
        'num_cage_channels': config['num_cage_channels'],
    }

    torch.save(cache_data, output_file)
    print(f"\n💾 特征已保存: {output_file}")

    # 11. 输出统计
    print(f"\n{'=' * 70}")
    print(f"📊 特征统计")
    print(f"{'=' * 70}")
    print(f"均值: {features.mean():.6f}")
    print(f"标准差: {features.std():.6f}")
    print(f"中位数: {np.median(features):.6f}")
    print(f"零值比例: {(features == 0).mean() * 100:.2f}%")


if __name__ == "__main__":
    main()