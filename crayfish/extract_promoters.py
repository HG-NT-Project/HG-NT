import os
import pandas as pd
import pysam
from tqdm import tqdm

# ================= 配置区 =================
UPSTREAM_LENGTH = 2000
DOWNSTREAM_LENGTH = 500
TOTAL_LENGTH = UPSTREAM_LENGTH + DOWNSTREAM_LENGTH

OUTPUT_DIR = "processed_tf"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def reverse_complement(seq):
    """参考你之前的逻辑进行反向互补"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
                  'a': 't', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
    return "".join(complement.get(base, 'N') for base in reversed(seq))


def extract_tf_promoters(anno_path, fasta_path):
    print(f"🚀 启动 TF 启动子提取 (范围: TSS -{UPSTREAM_LENGTH} 到 +{DOWNSTREAM_LENGTH})")

    # 1. 加载数据并生成唯一索引表
    df = pd.read_csv(anno_path, sep='\t')
    gene_ids = []
    seen = set()
    gene_coords = {}

    # 检查 Strand 列
    strand_col = next((c for c in df.columns if c.lower() == 'strand'), None)

    for _, row in df.iterrows():
        parts = str(row['GeneID']).split(':')
        if len(parts) < 5: continue

        gid = parts[0]
        if gid not in seen:
            # --- 修正点：处理 6095..6880 这种坐标格式 ---
            raw_pos = parts[3]
            if '..' in raw_pos:
                start_val = int(raw_pos.split('..')[0])
                end_val = int(raw_pos.split('..')[1])
            else:
                start_val = int(raw_pos)
                end_val = int(parts[4]) if len(parts) > 4 else start_val

            gene_ids.append(gid)
            seen.add(gid)
            gene_coords[gid] = {
                'chrom': parts[2],
                'start': start_val,
                'end': end_val,
                'strand': row[strand_col] if strand_col else '+'
            }

    # 保存索引表（确保图节点顺序一致）
    index_path = os.path.join(OUTPUT_DIR, "gene_id_index.txt")
    with open(index_path, "w") as f:
        for gid in gene_ids:
            f.write(f"{gid}\n")
    print(f"✅ 索引表已生成: {index_path} (共 {len(gene_ids)} 个节点)")

    # 2. 按照索引顺序提取序列
    fa = pysam.FastaFile(fasta_path)
    output_fasta = os.path.join(OUTPUT_DIR, "all_targets_promoters.fasta")

    success_count = 0
    with open(output_fasta, "w") as out:
        for gid in tqdm(gene_ids, desc="提取 TF 序列"):
            info = gene_coords[gid]
            chrom, start, end, strand = info['chrom'], info['start'], info['end'], info['strand']

            try:
                # 统一使用 start 或 end 代替 TSS
                if strand == '+' or strand == '1' or strand == '.':
                    s_pos = max(0, start - UPSTREAM_LENGTH)
                    e_pos = start + DOWNSTREAM_LENGTH
                    final_seq = fa.fetch(chrom, s_pos, e_pos)
                else:
                    # 负链逻辑：以终止端为起始进行反向提取
                    s_pos = max(0, end - DOWNSTREAM_LENGTH)
                    e_pos = end + UPSTREAM_LENGTH
                    raw_seq = fa.fetch(chrom, s_pos, e_pos)
                    final_seq = reverse_complement(raw_seq)

                final_seq = final_seq.upper()
                if len(final_seq) < TOTAL_LENGTH:
                    final_seq = final_seq.ljust(TOTAL_LENGTH, 'N')
                else:
                    final_seq = final_seq[:TOTAL_LENGTH]

                out.write(f">{gid}\n{final_seq}\n")
                success_count += 1
            except Exception:
                out.write(f">{gid}\n{'N' * TOTAL_LENGTH}\n")

    print(f"🎉 提取完成! 成功: {success_count}, 文件: {output_fasta}")


if __name__ == "__main__":
    extract_tf_promoters("anno.summary.xls", "ref.fa")