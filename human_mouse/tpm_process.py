import pandas as pd
import numpy as np
import torch
import os
import subprocess
import argparse
import gzip
from tqdm import tqdm

# 配置信息
CONFIG = {
    'human': {
        'expression': 'GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.gct.gz',
        'feature_file': 'precomputed_sequences/human_sequences.pt',
        'output': 'human_labels.pt'
    },
    'mouse': {
        'expression': 'E-GEOD-70484-query-results.tpmss.tsv',
        'feature_file': 'precomputed_sequences/mouse_sequences.pt',
        'output': 'mouse_labels.pt'
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--species', choices=['human', 'mouse', 'all'], required=True)
    args = parser.parse_args()

    # 确定要处理的物种
    species_list = ['human', 'mouse'] if args.species == 'all' else [args.species]

    for species in species_list:
        conf = CONFIG[species]
        print(f"\n{'-' * 20} 启动 {species} 表达量深度对齐与均值化 {'-' * 20}")

        # 1. 加载基准基因 ID
        if not os.path.exists(conf['feature_file']):
            print(f"❌ 错误: 未找到序列文件 {conf['feature_file']}")
            continue

        data = torch.load(conf['feature_file'], map_location='cpu')
        target_ids = data['target_genes']
        target_set = set(target_ids)
        print(f"✅ 已加载基准 ID: {len(target_ids)} 个")

        # 显示前几个target_ids作为示例
        print(f"  基准ID示例: {target_ids[:5]}")

        if species == 'human':
            # --- 人类数据：流式计算均值 (保持不变) ---
            id_tmp = 'ids_filter_temp.txt'
            with open(id_tmp, 'w') as f:
                f.write('\n'.join(target_ids))

            cmd = ["zgrep", "-F", "-f", id_tmp, conf['expression']]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)

            gene_to_mean = {}
            try:
                with tqdm(total=len(target_ids), desc="人类进度", unit="gene") as pbar:
                    for line in process.stdout:
                        parts = line.strip().split('\t')
                        if not parts:
                            continue
                        gid = parts[0]
                        if gid in target_set:
                            try:
                                # GCT格式：第0列ID，第1列Description，第2列开始是数值
                                vals = np.array(parts[2:], dtype=np.float32)
                                gene_to_mean[gid] = np.log2(vals + 1).mean()
                                pbar.update(1)
                            except Exception as e:
                                continue
            finally:
                process.terminate()
                if os.path.exists(id_tmp):
                    os.remove(id_tmp)

            final_labels = [gene_to_mean.get(gid, 0.0) for gid in target_ids]
            result_df = pd.DataFrame({'label': final_labels}, index=target_ids)

        else:
            # --- 小鼠数据：修正为正确读取数值列 ---
            print("📖 读取并计算小鼠跨组织均值...")

            # 读取文件，跳过注释行
            df = pd.read_csv(conf['expression'], sep='\t', comment='#')
            print(f"  原始数据形状: {df.shape}")
            print(f"  列名: {df.columns.tolist()[:5]}...")

            # 检查数据前几行
            print("\n  数据预览（前3行）:")
            for i in range(min(3, len(df))):
                print(f"    {df.iloc[i, 0]}: {df.iloc[i, 2:5].tolist()}...")

            # ID对齐：target_ids 带版本号，需要去掉版本号才能匹配
            # 方法1：把target_ids去掉版本号
            target_ids_base = [gid.split('.')[0] for gid in target_ids]
            print(f"\n  基准ID去版本号示例: {target_ids_base[:5]}")

            # 方法2：把表达量中的ID加上版本号（更复杂，不建议）
            # 这里使用方法1

            # 从表达量文件提取基因ID（无版本号）
            gene_ids_raw = df.iloc[:, 0].str.split('.').str[0].tolist()
            print(f"  表达量文件ID示例: {gene_ids_raw[:5]}")

            # 正确的数值列：从第2列开始（跳过 Gene ID 和 Gene Name）
            numeric_cols = df.columns[2:].tolist()
            print(f"\n  提取数值列（从第2列开始）...")
            print(f"  找到 {len(numeric_cols)} 个数值列")
            print(f"  前3个数值列: {numeric_cols[:3]}")

            # 提取数值矩阵
            values = df[numeric_cols].values.astype(float)

            # 计算每个基因的log2均值
            log_means = np.log2(values + 1).mean(axis=1)

            # 创建Series用于对齐（使用无版本号的ID作为索引）
            value_series = pd.Series(log_means, index=gene_ids_raw)

            # 对齐target_ids（使用去版本号的target_ids）
            aligned_values = value_series.reindex(target_ids_base).fillna(0).values

            # 检查非零值
            non_zero = (aligned_values > 0).sum()
            print(f"\n  非零表达值基因数: {non_zero}/{len(target_ids)} ({non_zero / len(target_ids) * 100:.1f}%)")

            # 显示几个示例
            print("\n  表达值示例（前5个基因）:")
            for i in range(min(5, len(target_ids))):
                print(f"    {target_ids[i]}: {aligned_values[i]:.4f}")

            # 创建结果DataFrame（使用原始带版本号的target_ids作为index）
            result_df = pd.DataFrame({'label': aligned_values}, index=target_ids)

        # 2. 保存结果 (统一保存为 [N, 1] 形状)
        output_dir = 'processed_labels'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, conf['output'])

        label_tensor = torch.tensor(result_df.values, dtype=torch.float32)

        # 确保形状是 [N, 1]
        if len(label_tensor.shape) == 1:
            label_tensor = label_tensor.unsqueeze(1)

        torch.save({
            'gene_id': result_df.index.tolist(),
            'labels': label_tensor,
            'columns': ['mean_expression_log2']
        }, save_path)

        print(f"\n✅ 处理完成！")
        print(f"📊 最终张量形状: {label_tensor.shape}")
        print(f"💾 路径: {save_path}")

        # 显示统计信息
        print(f"\n📊 统计信息:")
        print(f"  最小值: {result_df['label'].min():.4f}")
        print(f"  最大值: {result_df['label'].max():.4f}")
        print(f"  均值: {result_df['label'].mean():.4f}")
        print(f"  中位数: {result_df['label'].median():.4f}")


if __name__ == "__main__":
    main()