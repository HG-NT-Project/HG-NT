import torch
import os
import gc
import re
import gzip
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from datetime import datetime
import json
from pyfaidx import Fasta

# ===================== 配置区 =====================
# 小龙虾配置（保持原有的6000bp提取逻辑）
CRAYFISH_CONFIG = {
    'name': 'crayfish',
    'gene_model': 'anno.summary.xls',  # 小龙虾注释文件
    'genome': 'ref.fa',  # 小龙虾基因组文件
    'expression': 'crayfish_labels.csv',  # 表达量标签文件
    'gene_index': 'gene_id_index.txt',  # 基因ID索引文件（可选）
}

# NT优化版序列配置 (6000bp，保持原逻辑)
UPSTREAM_TSS = 2000  # TSS上游长度
DOWNSTREAM_TSS = 1000  # TSS下游长度
UPSTREAM_TTS = 1000  # TTS上游长度
DOWNSTREAM_TTS = 2000  # TTS下游长度
SEQ_LENGTH = 6000  # 总长度：2000+1000+1000+2000 = 6000bp
MAX_TOKENS = 1000  # 6000bp / 6 = 1000 tokens

# 模型和输出配置
MODEL_PATH = "../pretrain_model/Nucleotide-Transformer"  # NT模型路径
OUTPUT_DIR = "crayfish_embeddings"
BATCH_SIZE = 8  # 可以根据显存调整
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =================================================
# 序列提取函数（小龙虾适配版，支持索引文件）
# =================================================

def load_gene_index(index_file):
    """
    加载基因ID索引文件
    如果存在，返回基因ID列表（按顺序）
    """
    if not os.path.exists(index_file):
        print(f"⚠️ 索引文件不存在: {index_file}，将使用表达量文件中的顺序")
        return None

    print(f"📖 加载基因索引文件: {index_file}")
    with open(index_file, 'r') as f:
        gene_ids = [line.strip() for line in f if line.strip()]

    print(f"✅ 加载索引文件成功: {len(gene_ids)} 个基因")
    return gene_ids


def load_target_genes(expression_file, index_file=None):
    """
    从小龙虾表达量标签文件加载目标基因
    如果提供索引文件，则按照索引文件的顺序
    """
    print(f"📖 加载表达量文件: {expression_file}")

    if not os.path.exists(expression_file):
        print(f"❌ 表达量文件不存在: {expression_file}")
        return None, None

    df = pd.read_csv(expression_file)
    print(f"✅ 标签数据加载成功: {df.shape}")
    print(f"   列名: {list(df.columns)}")

    # 检查gene_id列
    if 'gene_id' not in df.columns:
        # 尝试其他可能的列名
        possible_id_cols = ['GeneID', 'gene', 'Gene', 'id', 'ID']
        found = False
        for col in possible_id_cols:
            if col in df.columns:
                df = df.rename(columns={col: 'gene_id'})
                found = True
                print(f"🔄 已将 '{col}' 列重命名为 'gene_id'")
                break
        if not found:
            print(f"❌ 未找到基因ID列，可用的列: {list(df.columns)}")
            return None, None

    # 过滤掉label为NaN的样本
    if 'label' in df.columns:
        initial_count = len(df)
        df = df[df['label'].notna()].copy()
        if len(df) < initial_count:
            print(f"🧹 过滤label为NaN: {initial_count} → {len(df)}")

    # 获取所有基因ID
    all_gene_ids = set(df['gene_id'].tolist())

    # 如果提供了索引文件，按照索引文件的顺序
    if index_file and os.path.exists(index_file):
        index_genes = load_gene_index(index_file)
        if index_genes:
            # 只保留在索引文件中的基因
            target_genes = [gid for gid in index_genes if gid in all_gene_ids]
            print(f"✅ 按照索引文件顺序，找到 {len(target_genes)}/{len(index_genes)} 个目标基因")
            # 同时保留完整的df用于后续
            df_filtered = df[df['gene_id'].isin(target_genes)].copy()
            # 按照索引顺序重新排序
            df_filtered['gene_id'] = pd.Categorical(df_filtered['gene_id'], categories=target_genes, ordered=True)
            df_filtered = df_filtered.sort_values('gene_id').reset_index(drop=True)
            return target_genes, df_filtered

    # 如果没有索引文件，使用原始顺序
    target_genes = df['gene_id'].tolist()
    print(f"✅ 找到 {len(target_genes)} 个目标基因")
    return target_genes, df


def parse_crayfish_annotation(anno_file, target_genes):
    """
    解析小龙虾的anno.summary.xls文件，提取目标基因的位置信息
    格式: "g00001:ref:NW_020715931.1:2406..3662:+"
    使用start和end代替TSS和TTS
    """
    print(f"📖 解析小龙虾注释文件: {anno_file}")

    if not os.path.exists(anno_file):
        print(f"❌ 注释文件不存在: {anno_file}")
        return None

    target_set = set(target_genes)
    gene_data = []
    found_genes = set()

    # 读取注释文件（制表符分隔）
    df = pd.read_csv(anno_file, sep='\t')
    print(f"✅ 注释数据加载成功: {df.shape}")

    # 确定Strand列名
    strand_col = None
    for col in df.columns:
        if col.lower() == 'strand':
            strand_col = col
            break

    for _, row in df.iterrows():
        # 解析GeneID列
        gene_id_raw = str(row['GeneID'])
        parts = gene_id_raw.split(':')

        if len(parts) < 5:
            continue

        gene_id = parts[0]  # 提取基因ID，如 g00001

        # 检查是否为目标基因且未重复
        if gene_id not in target_set or gene_id in found_genes:
            continue

        # 提取染色体
        chrom = parts[2]

        # 提取位置信息（处理 2406..3662 格式）
        raw_pos = parts[3]
        if '..' in raw_pos:
            start = int(raw_pos.split('..')[0])
            end = int(raw_pos.split('..')[1])
        else:
            start = int(raw_pos)
            end = int(parts[4]) if len(parts) > 4 else start

        # 提取链信息
        strand = parts[4] if len(parts) > 4 else '+'
        # 如果有单独的Strand列，优先使用
        if strand_col and strand_col in row:
            strand = row[strand_col]

        gene_data.append({
            'gene_id': gene_id,
            'chromosome': chrom,
            'start': start,
            'end': end,
            'strand': strand
        })

        found_genes.add(gene_id)

    gene_df = pd.DataFrame(gene_data)

    # 按照target_genes的原始顺序排序
    gene_df['gene_id'] = pd.Categorical(gene_df['gene_id'], categories=target_genes, ordered=True)
    gene_df = gene_df.sort_values('gene_id').reset_index(drop=True)

    print(f"✅ 在注释中找到 {len(gene_df)}/{len(target_set)} 个唯一基因的位置信息")

    # 检查未找到的基因
    not_found = target_set - found_genes
    if not_found:
        sample_size = min(10, len(not_found))
        print(f"⚠️ 未在注释中找到 {len(not_found)} 个基因: {list(not_found)[:sample_size]}...")

    return gene_df


def reverse_complement(seq):
    """反向互补（优化版）"""
    comp_table = str.maketrans('ATGCatgcNn', 'TACGtacgNn')
    return seq[::-1].translate(comp_table)


def extract_sequence(genome, gene_info):
    """
    提取6000bp序列 - 使用start和end代替TSS/TTS
    保持原提取逻辑不变
    """
    chrom = gene_info['chromosome']
    start = gene_info['start']
    end = gene_info['end']
    strand = gene_info['strand']

    try:
        if strand == '+' or strand == '1' or strand == '.':
            # ==================== 正链基因 ====================
            # TSS区域: start上游2000bp + start下游1000bp
            tss_start = max(0, start - UPSTREAM_TSS)
            tss_seq = str(genome[chrom][tss_start:start + DOWNSTREAM_TSS])

            # TTS区域: end上游1000bp + end下游2000bp
            tts_start = max(0, end - UPSTREAM_TTS)
            tts_seq = str(genome[chrom][tts_start:end + DOWNSTREAM_TTS])

            # 合并序列
            seq = tss_seq + tts_seq

        else:
            # ==================== 负链基因 ====================
            # TSS区域: 围绕end（相当于正链的start）
            tss_start = max(0, end - DOWNSTREAM_TSS)
            tss_raw = str(genome[chrom][tss_start:end + UPSTREAM_TSS])
            tss_seq = reverse_complement(tss_raw)

            # TTS区域: 围绕start（相当于正链的end）
            tts_start = max(0, start - DOWNSTREAM_TTS)
            tts_raw = str(genome[chrom][tts_start:start + UPSTREAM_TTS])
            tts_seq = reverse_complement(tts_raw)

            # 合并序列
            seq = tss_seq + tts_seq

        # 填充或截断（保持6000bp）
        if len(seq) < SEQ_LENGTH:
            seq = seq.ljust(SEQ_LENGTH, 'N')
        elif len(seq) > SEQ_LENGTH:
            seq = seq[:SEQ_LENGTH]

        return seq.upper()

    except Exception as e:
        # 如果提取失败，返回全N序列
        print(f"⚠️ 提取失败 {gene_info['gene_id']}: {e}")
        return 'N' * SEQ_LENGTH


# =================================================
# 主函数：一步到位（带跳过检查）
# =================================================

def extract_and_embed(use_index=True):
    """
    一步完成：序列提取 + 特征提取（小龙虾版本）
    增加：如果输出文件已存在，自动跳过
    use_index: 是否使用基因索引文件
    """

    # 输出文件路径
    output_file = f"{OUTPUT_DIR}/crayfish_embeddings.pt"

    # 检查输出文件是否已存在
    if os.path.exists(output_file):
        print(f"⏩ 发现已存在文件: {output_file}，正在跳过...")
        return {
            'num_sequences': '已存在（跳过）',
            'embedding_dim': '已存在',
            'skipped': True
        }

    print(f"\n{'=' * 70}")
    print(f"🦞 处理物种: 小龙虾 (Procambarus clarkii)")
    print(f"{'=' * 70}")

    # 获取配置
    config = CRAYFISH_CONFIG
    expression_file = config['expression']
    genome_file = config['genome']
    anno_file = config['gene_model']
    index_file = config.get('gene_index', None)

    # 检查文件
    for f, name in [(expression_file, "表达量"), (genome_file, "基因组"), (anno_file, "注释文件")]:
        if not os.path.exists(f):
            print(f"❌ {name}文件不存在: {f}")
            return None

    # 检查索引文件（可选）
    if use_index and index_file and os.path.exists(index_file):
        print(f"✅ 将使用索引文件: {index_file}")
        use_index_file = True
    else:
        use_index_file = False
        if use_index and index_file:
            print(f"⚠️ 索引文件不存在: {index_file}，将不使用索引")

    # ========== 步骤1: 加载目标基因 ==========
    if use_index_file:
        target_genes, expr_df = load_target_genes(expression_file, index_file)
    else:
        target_genes, expr_df = load_target_genes(expression_file, None)

    if target_genes is None:
        print(f"❌ 加载目标基因失败")
        return None

    # ========== 步骤2: 解析注释文件获取位置 ==========
    gene_df = parse_crayfish_annotation(anno_file, target_genes)
    if gene_df is None or len(gene_df) == 0:
        print(f"❌ 未找到任何基因的位置信息")
        return None

    # ========== 步骤3: 加载基因组 ==========
    print(f"📖 加载基因组: {genome_file}")
    genome = Fasta(genome_file, as_raw=True, sequence_always_upper=True, read_ahead=10000)

    # ========== 步骤4: 提取序列 ==========
    print(f"🧬 提取序列 (长度: {SEQ_LENGTH}bp)...")
    print(f"   TSS区域: 上游{UPSTREAM_TSS}bp + 下游{DOWNSTREAM_TSS}bp = {UPSTREAM_TSS + DOWNSTREAM_TSS}bp")
    print(f"   TTS区域: 上游{UPSTREAM_TTS}bp + 下游{DOWNSTREAM_TTS}bp = {UPSTREAM_TTS + DOWNSTREAM_TTS}bp")
    print(f"   总长度: {SEQ_LENGTH}bp")

    sequences = []
    gene_ids = []
    gene_info_list = []

    # 创建基因ID到行的映射
    gene_id_to_row = {row['gene_id']: row for _, row in gene_df.iterrows()}

    # 统计失败情况
    failed_count = 0
    n_count_high = 0
    missing_in_anno = 0

    for gene_id in tqdm(target_genes, desc="提取序列"):
        if gene_id in gene_id_to_row:
            seq = extract_sequence(genome, gene_id_to_row[gene_id])
            sequences.append(seq)
            gene_ids.append(gene_id)
            gene_info_list.append(gene_id_to_row[gene_id])

            # 检查是否为全N序列或高N含量
            if seq.count('N') > SEQ_LENGTH * 0.5:
                n_count_high += 1
                failed_count += 1
        else:
            missing_in_anno += 1
            failed_count += 1
            # 对于找不到的基因，用全N序列填充以保持顺序
            sequences.append('N' * SEQ_LENGTH)
            gene_ids.append(gene_id)
            gene_info_list.append({
                'gene_id': gene_id,
                'chromosome': 'unknown',
                'start': 0,
                'end': 0,
                'strand': '+',
                'missing': True
            })

    print(f"✅ 成功提取 {len([s for s in sequences if s.count('N') < SEQ_LENGTH * 0.5])}/{len(target_genes)} 条有效序列")
    if missing_in_anno > 0:
        print(f"⚠️ 在注释中找不到的基因: {missing_in_anno} 个（已用N填充）")
    if n_count_high > 0:
        print(f"⚠️ 高N含量序列: {n_count_high} 个")
    if failed_count > 0:
        print(f"⚠️ 总计问题基因: {failed_count} 个")

    if len(sequences) == 0:
        print("❌ 没有成功提取任何序列")
        return None

    # ========== 步骤5: 加载NT模型 ==========
    print(f"🤖 加载NT模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
        model = AutoModelForMaskedLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=False
        ).to(DEVICE).eval()
    except Exception as e:
        print(f"❌ 加载NT模型失败: {e}")
        print("请确保模型路径正确: {MODEL_PATH}")
        return None

    # ========== 步骤6: 提取特征 ==========
    print(f"🎯 提取特征 (批次大小: {BATCH_SIZE})...")
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="提取特征"):
            batch_seqs = sequences[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=MAX_TOKENS
            ).to(DEVICE)

            outputs = model(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                encoder_attention_mask=inputs['attention_mask'],
                output_hidden_states=True
            )

            # Mean pooling
            last_hidden = outputs.hidden_states[-1]
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_embeds = (sum_embeddings / sum_mask).to(torch.float32).cpu()

            all_embeddings.append(mean_embeds)

            # 清理显存
            if i % (BATCH_SIZE * 100) == 0:
                torch.cuda.empty_cache()

    # ========== 步骤7: 保存结果 ==========
    final_embeddings = torch.cat(all_embeddings, dim=0)

    save_dict = {
        'x': final_embeddings,
        'gene_ids': gene_ids,
        'gene_info': gene_info_list,
        'species': 'crayfish',
        'model': 'Nucleotide-Transformer',
        'embedding_dim': final_embeddings.shape[1],
        'num_sequences': len(gene_ids),
        'seq_length': SEQ_LENGTH,
        'max_tokens': MAX_TOKENS,
        'upstream_tss': UPSTREAM_TSS,
        'downstream_tss': DOWNSTREAM_TSS,
        'upstream_tts': UPSTREAM_TTS,
        'downstream_tts': DOWNSTREAM_TTS,
        'failed_count': failed_count,
        'missing_in_anno': missing_in_anno,
        'n_count_high': n_count_high,
        'use_index': use_index_file,
        'index_file': index_file if use_index_file else None,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(save_dict, output_file)
    print(f"✅ 保存完成: {output_file}")
    print(f"📊 特征形状: {final_embeddings.shape}")
    print(f"📊 特征维度: {final_embeddings.shape[1]}")

    # 清理内存
    del sequences, all_embeddings, final_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    return save_dict


# =================================================
# 主程序
# =================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='小龙虾NT一步到位：序列提取+特征提取')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小（建议根据显存调整，默认8）')
    parser.add_argument('--force_reload', action='store_true',
                        help='强制重新提取（忽略已存在的文件）')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='NT模型路径')
    parser.add_argument('--no_index', action='store_true',
                        help='不使用基因索引文件，按表达量文件顺序')
    parser.add_argument('--index_file', type=str, default=None,
                        help='指定基因索引文件路径（默认：gene_id_index.txt）')

    args = parser.parse_args()

    # 更新配置 - 在main函数内部使用局部变量
    batch_size = args.batch_size
    model_path = args.model_path

    # 更新索引文件路径
    if args.index_file:
        CRAYFISH_CONFIG['gene_index'] = args.index_file

    print(f"🦞 小龙虾NT一步到位工具")
    print(f"🔧 设备: {DEVICE}")
    print(f"📦 批次大小: {batch_size}")
    print(f"📏 序列长度: {SEQ_LENGTH}bp → {MAX_TOKENS} tokens")
    print(f"📏 提取配置: TSS区域 {UPSTREAM_TSS}+{DOWNSTREAM_TSS}bp, TTS区域 {UPSTREAM_TTS}+{DOWNSTREAM_TTS}bp")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📄 使用索引: {'否' if args.no_index else '是'}")
    if not args.no_index:
        print(f"   索引文件: {CRAYFISH_CONFIG.get('gene_index', 'gene_id_index.txt')}")
    if args.force_reload:
        print(f"🔄 强制重新加载模式: 将覆盖已存在的文件")
    print()

    # 如果强制重新加载，删除已存在的文件
    if args.force_reload:
        output_file = f"{OUTPUT_DIR}/crayfish_embeddings.pt"
        if os.path.exists(output_file):
            print(f"🗑️ 删除已存在文件: {output_file}")
            os.remove(output_file)

    # 执行提取和嵌入
    result = extract_and_embed(use_index=not args.no_index)

    # 总结
    print(f"\n{'=' * 70}")
    print(f"📋 处理总结")
    print(f"{'=' * 70}")

    if result:
        if result.get('skipped', False):
            print(f"\n🦞 小龙虾:")
            print(f"   ⏩ 状态: {result['num_sequences']}")
            print(f"   📁 输出文件已存在: {OUTPUT_DIR}/crayfish_embeddings.pt")
        else:
            print(f"\n🦞 小龙虾:")
            print(f"   ✅ 成功: {result['num_sequences']} 个基因")
            if 'missing_in_anno' in result and result['missing_in_anno'] > 0:
                print(f"   ⚠️ 注释中缺失: {result['missing_in_anno']} 个（已用N填充）")
            if 'n_count_high' in result and result['n_count_high'] > 0:
                print(f"   ⚠️ 高N含量序列: {result['n_count_high']} 个")
            if 'failed_count' in result and result['failed_count'] > 0:
                print(f"   ⚠️ 问题基因总计: {result['failed_count']} 个")
            print(f"   📊 Embedding维度: {result['embedding_dim']}")
            print(f"   📁 输出: {OUTPUT_DIR}/crayfish_embeddings.pt")
            if result.get('use_index', False):
                print(f"   📄 使用索引文件: {result.get('index_file', 'N/A')}")

    print(f"\n✨ 完成！")


if __name__ == "__main__":
    main()