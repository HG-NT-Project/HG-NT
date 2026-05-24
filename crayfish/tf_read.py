import pandas as pd
import re


def audit_tf_resources(anno_path, tpm_path):
    print("🔍 正在扫描潜在的转录因子 (TF)...")

    # 1. 加载注释表
    anno_df = pd.read_csv(anno_path, sep='\t')

    # 2. 定义 TF 特征关键词 (涵盖主要的 DNA 结合域)
    tf_keywords = [
        'transcription factor', 'zinc finger', 'homeobox', 'bHLH',
        'Sox', 'MYB', 'MADS-box', 'bZIP', 'forkhead', 'WRKY', 'GATA'
    ]
    pattern = '|'.join(tf_keywords)

    # 3. 筛选候选 TF
    # 搜索 NRANNO 和 UniANNO 列
    is_tf = anno_df['NRANNO'].str.contains(pattern, case=False, na=False) | \
            anno_df['UniANNO'].str.contains(pattern, case=False, na=False)

    tf_df = anno_df[is_tf].copy()

    # 4. 提取 ID 映射关系
    # 格式解析: gene-LOCxxx:rna-xxx:chr:start:end
    tf_df['Clean_ID'] = tf_df['GeneID'].str.split(':').str[0]

    # 5. 交叉校验 TPM 矩阵
    tpm_df = pd.read_csv(tpm_path, sep='\t', usecols=[0], low_memory=False)
    tpm_ids = set(tpm_df.iloc[:, 0].astype(str).tolist())

    active_tfs = tf_df[tf_df['Clean_ID'].isin(tpm_ids)]

    print("\n" + "=" * 40)
    print("        TF 资源审计报告")
    print("=" * 40)
    print(f"1. 从注释中识别到的潜在 TF 总数: {len(tf_df)}")
    print(f"2. 在 TPM 矩阵中有表达值的有效 TF: {len(active_tfs)}")
    print("-" * 40)
    print("3. 前 5 个 TF 及其功能预览:")
    print(active_tfs[['Clean_ID', 'NRANNO']].head(5).to_string(index=False))
    print("=" * 40)

    return active_tfs


if __name__ == "__main__":
    # 替换为你的文件名
    active_tfs = audit_tf_resources("anno.summary.xls", "gene.tpm.matrix.annot.xls")