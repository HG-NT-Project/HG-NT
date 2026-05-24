"""
快速测试脚本 - 只分析1个基因
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pyfaidx import Fasta
import os
import sys

# 添加路径
sys.path.append(os.getcwd())
from train_xr_xiaolongxia import ModelM3_MultiGraphConcat


def parse_gene_coords(gene_id_value):
    """兼容两种注释格式: gene:rna:chr:start:end 或 gene:rna:chr:start..end:strand"""
    parts = str(gene_id_value).split(':')
    if len(parts) < 5:
        return None

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
        return None

    return {'chrom': parts[2], 'start': start, 'end': end, 'strand': strand}

# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 使用设备: {DEVICE}")

# 1. 加载数据
print("\n1️⃣ 加载数据...")
predictions = pd.read_csv("Results_Crayfish_Ablation_V2/predictions/crayfish_m3_seed42_predictions.csv")
print(f"  预测结果: {predictions.shape}")

# 选择第一个基因测试
test_gene = predictions.iloc[0]['gene_id']
true_expr = predictions.iloc[0]['true_expression']
print(f"  测试基因: {test_gene}, 真实表达量: {true_expr}")

# 2. 提取序列
print("\n2️⃣ 提取序列...")
genome = Fasta("ref.fa", as_raw=True, sequence_always_upper=True)
anno_df = pd.read_csv("anno.summary.xls", sep='\t')

gene_info = None
for _, row in anno_df.iterrows():
    parts = str(row['GeneID']).split(':')
    if len(parts) >= 1 and parts[0] == test_gene:
        gene_info = parse_gene_coords(row['GeneID'])
        if gene_info is not None:
            break

if gene_info:
    chrom, start, end, strand = gene_info['chrom'], gene_info['start'], gene_info['end'], gene_info['strand']
    if strand in ['+', '1', '.']:
        tss_seq = str(genome[chrom][max(0, start - 2000):start + 1000])
        tts_seq = str(genome[chrom][max(0, end - 1000):end + 2000])
        seq = tss_seq + tts_seq
    else:
        def rc(s):
            return s[::-1].translate(str.maketrans('ATGCatgc', 'TACGtacg'))


        tss_seq = rc(str(genome[chrom][max(0, end - 1000):end + 2000]))
        tts_seq = rc(str(genome[chrom][max(0, start - 2000):start + 1000]))
        seq = tss_seq + tts_seq

    seq = seq.ljust(6000, 'N')[:6000].upper()
    print(f"  序列长度: {len(seq)}")
else:
    print("  ❌ 未找到基因注释")
    exit()

# 3. 加载模型
print("\n3️⃣ 加载模型...")
tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/Nucleotide-Transformer")
nt_model = AutoModelForMaskedLM.from_pretrained(
    "../pretrain_model/Nucleotide-Transformer",
    torch_dtype=torch.float16
).to(DEVICE)
nt_model.eval()
for param in nt_model.parameters():
    param.requires_grad = False

m3_model = ModelM3_MultiGraphConcat(input_dim=2560).to(DEVICE)
checkpoint = torch.load("Results_Crayfish_Ablation_V2/models/crayfish_m3_seed42_best.pth", map_location=DEVICE)
m3_model.load_state_dict(checkpoint['model_state_dict'])
m3_model.eval()

print("  ✅ 模型加载完成")

# 4. 计算梯度
print("\n4️⃣ 计算梯度...")
inputs = tokenizer(
    seq,
    return_tensors="pt",
    max_length=1000,
    padding='max_length',
    truncation=True
).to(DEVICE)

with torch.enable_grad():
    input_embeds = nt_model.get_input_embeddings()(inputs['input_ids']).clone().detach().requires_grad_(True)

    nt_outputs = nt_model(
        inputs_embeds=input_embeds,
        attention_mask=inputs['attention_mask'],
        output_hidden_states=True
    )

    last_hidden = nt_outputs.hidden_states[-1]
    mask = inputs['attention_mask'].unsqueeze(-1).float()
    gene_embedding = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    # 简化预测（无图结构）
    pred = m3_model.regressor(
        m3_model.fusion(
            torch.cat([gene_embedding, gene_embedding], dim=-1)
        )
    ).squeeze()

    m3_model.zero_grad()
    nt_model.zero_grad()
    pred.backward()

    token_gradients = input_embeds.grad.abs().sum(dim=-1).squeeze(0).cpu().numpy()

print(f"  预测值: {pred.item():.4f}")
print(f"  梯度形状: {token_gradients.shape}")

# 5. 扩展到碱基级别
print("\n5️⃣ 扩展到碱基级别...")
base_gradients = np.repeat(token_gradients, 6)[:6000]

# 6. 绘制
print("\n6️⃣ 绘制重要性曲线...")
plt.figure(figsize=(12, 4))
plt.plot(base_gradients, 'b-', linewidth=0.5, alpha=0.7)
plt.axvline(x=2000, color='red', linestyle='--', label='TSS')
plt.axvline(x=4000, color='green', linestyle='--', label='TTS')
plt.xlabel('Position (bp)')
plt.ylabel('Importance')
plt.title(f'Gene {test_gene} Importance Profile\nPred: {pred.item():.3f}, True: {true_expr:.3f}')
plt.legend()
plt.tight_layout()
plt.savefig('test_importance.png', dpi=150)
print("  ✅ 保存为 test_importance.png")

print("\n✅ 测试完成！")