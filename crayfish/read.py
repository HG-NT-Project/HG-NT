import pandas as pd
import re
from collections import Counter


def check_overlap_status(matrix_path, gtf_path, anno_path):
    print("🔍 启动三方数据对齐状态审计...")

    # 1. 加载 TPM 矩阵 ID 分类
    tpm_df = pd.read_csv(matrix_path, sep='\t', usecols=[0], low_memory=False)
    tpm_ids = set(tpm_df.iloc[:, 0].astype(str).tolist())
    loc_ids = {i for i in tpm_ids if "LOC" in i}
    mstrg_ids = {i for i in tpm_ids if "MSTRG" in i}
    common_ids = tpm_ids - loc_ids - mstrg_ids

    print(f"\n📈 表达矩阵组成：总计 {len(tpm_ids)} 个基因")
    print(f"   - LOC 类: {len(loc_ids)}")
    print(f"   - MSTRG 类: {len(mstrg_ids)}")
    print(f"   - 其他类: {len(common_ids)}")

    # 2. 扫描 GTF 文件的 ID 库
    gtf_ids = set()
    print(f"\n📖 正在扫描 GTF: {gtf_path} ...")
    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith("#"): continue
            # 匹配 gene_id "XXX";
            match = re.search(r'gene_id "([^"]+)"', line)
            if match:
                gtf_ids.add(match.group(1))

    # 3. 扫描 Anno 表的 ID 库
    anno_ids = set()
    print(f"📖 正在扫描 Anno: {anno_path} ...")
    anno_df = pd.read_csv(anno_path, sep='\t', usecols=[0])
    # 拆解复合 ID: gene-LOCxxx:rna-xxx... 取第一部分
    anno_ids = set(anno_df.iloc[:, 0].str.split(':').str[0].tolist())

    # 4. 交叉验证匹配状态
    def check_mapping(source_set, target_set, name):
        direct = source_set.intersection(target_set)
        # 尝试去掉 'gene-' 前缀后再比对
        source_no_prefix = {i.replace('gene-', '') for i in source_set}
        target_no_prefix = {i.replace('gene-', '') for i in target_set}
        fixed = source_no_prefix.intersection(target_no_prefix)
        return len(direct), len(fixed)

    # 结果统计
    loc_in_gtf = check_mapping(loc_ids, gtf_ids, "LOC->GTF")
    mstrg_in_gtf = check_mapping(mstrg_ids, gtf_ids, "MSTRG->GTF")
    loc_in_anno = check_mapping(loc_ids, anno_ids, "LOC->Anno")
    mstrg_in_anno = check_mapping(mstrg_ids, anno_ids, "MSTRG->Anno")

    print("\n" + "=" * 50)
    print("           🧬 最终匹配状态报告")
    print("=" * 50)
    print(f"1. LOC 类基因 ({len(loc_ids)} 个):")
    print(f"   - 与 GTF 匹配: 直接 {loc_in_gtf[0]} | 修正前缀后 {loc_in_gtf[1]}")
    print(f"   - 与 Anno 匹配: 直接 {loc_in_anno[0]} | 修正前缀后 {loc_in_anno[1]}")

    print(f"\n2. MSTRG 类基因 ({len(mstrg_ids)} 个):")
    print(f"   - 与 GTF 匹配: {mstrg_in_gtf[0]} (若为0，说明GTF版本不对)")
    print(f"   - 与 Anno 匹配: {mstrg_in_anno[0]} (若为0，属正常，因它是新基因)")

    print(f"\n3. 其他类基因 ({len(common_ids)} 个):")
    other_match = check_mapping(common_ids, gtf_ids, "Other->GTF")
    print(f"   - 与 GTF 匹配: 直接 {other_match[0]} | 修正前缀后 {other_match[1]}")
    print("=" * 50)

    if mstrg_in_gtf[0] == 0:
        print("\n🚨 警告：MSTRG 匹配失败！")
        print("原因：你当前使用的 'ref.gtf' 是参考基因组自带的，不含你的新预测记录。")
        print("解决：请在目录下寻找是否有 'stringtie_merged.gtf' 或类似的合并文件。")


if __name__ == "__main__":
    check_overlap_status("gene.tpm.matrix.annot.xls", "ref.gtf", "anno.summary.xls")