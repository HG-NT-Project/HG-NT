#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键查看两个物种的GTF（自动解压）
"""

import gzip
import os

GTF_FILES = {
    'human': 'gencode.v49.primary_assembly.basic.annotation.gtf.gz',
    'mouse': 'gencode.vM38.primary_assembly.basic.annotation.gtf.gz'
}


def quick_peek():
    """快速查看两个GTF"""

    for species, gz_file in GTF_FILES.items():
        print(f"\n{'=' * 50}")
        print(f"🔍 {species.upper()}")
        print(f"{'=' * 50}")

        # 检查文件是否存在
        if not os.path.exists(gz_file):
            print(f"❌ 文件不存在: {gz_file}")
            continue

        try:
            # 直接解压并读取
            print(f"\n📦 正在解压读取: {gz_file}")

            with gzip.open(gz_file, 'rt') as f:
                # 显示前5行（包括注释）
                print("\n📋 前5行内容:")
                for i in range(5):
                    line = f.readline().strip()
                    if line.startswith('#'):
                        print(f"  📝 注释{i + 1}: {line[:100]}")
                    else:
                        print(f"  📊 数据{i + 1}: {line[:100]}")

                # 找第一个非注释行解析
                f.seek(0)  # 回到文件开头
                for line in f:
                    if not line.startswith('#'):
                        parts = line.strip().split('\t')
                        print(f"\n🔬 第一个数据行解析:")
                        print(f"  染色体: {parts[0]}")
                        print(f"  来源: {parts[1]}")
                        print(f"  特征类型: {parts[2]}")
                        print(f"  起始位置: {parts[3]}")
                        print(f"  结束位置: {parts[4]}")
                        print(f"  链方向: {parts[6]}")
                        print(f"  属性: {parts[8][:150]}...")
                        break

                # 统计前10000行的特征类型分布
                print(f"\n📊 特征类型统计（前10000行）:")
                f.seek(0)
                feature_count = {}
                line_count = 0

                for line in f:
                    if line_count >= 10000:
                        break
                    if not line.startswith('#'):
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            feature = parts[2]
                            feature_count[feature] = feature_count.get(feature, 0) + 1
                    line_count += 1

                # 按出现次数排序显示
                for feature, count in sorted(feature_count.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {feature:15} : {count:6d}")

                # 特别关注UTR相关的特征
                print(f"\n🎯 UTR相关特征:")
                utr_features = [f for f in feature_count.keys() if 'utr' in f.lower()]
                if utr_features:
                    for f in utr_features:
                        print(f"  ✅ {f}")
                else:
                    print(f"  ❌ 未找到UTR特征")

        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()


def decompress_and_show(gz_file, output_txt=None):
    """解压并显示完整内容（可选保存）"""
    if not os.path.exists(gz_file):
        print(f"❌ 文件不存在: {gz_file}")
        return

    # 解压后的文件名
    if output_txt is None:
        output_txt = gz_file[:-3]  # 去掉.gz

    print(f"\n📦 正在解压 {gz_file} -> {output_txt}")

    try:
        # 解压
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_txt, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"✅ 解压完成: {output_txt}")

        # 显示解压后文件的前10行
        print(f"\n📋 解压后文件前10行:")
        with open(output_txt, 'r') as f:
            for i in range(10):
                line = f.readline().strip()
                print(f"  {i + 1}: {line[:100]}")

    except Exception as e:
        print(f"❌ 解压失败: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='查看GTF文件')
    parser.add_argument('--decompress', action='store_true',
                        help='同时解压文件')

    args = parser.parse_args()

    # 快速查看
    quick_peek()

    # 如果需要解压
    if args.decompress:
        print(f"\n{'=' * 50}")
        print("📦 解压文件")
        print(f"{'=' * 50}")

        for species, gz_file in GTF_FILES.items():
            if os.path.exists(gz_file):
                decompress_and_show(gz_file)


if __name__ == "__main__":
    main()