import torch
import os
import gc
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

# ===================== 配置区 =====================
SPECIES_LIST = ['human', 'mouse']  # 支持的物种列表
MODEL_PATH = "../pretrain_model/Nucleotide-Transformer"
INPUT_DIR = "precomputed_sequences_NT"
OUTPUT_DIR = "processed_features"

BATCH_SIZE = 2  # 3090/4090 建议 2-4，若 OOM 请改 1
MAX_LENGTH = 1000  # 对应 6kb 序列
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =================================================

def run_extraction():
    print(f"🚀 启动多物种特征提取流程: {SPECIES_LIST}")

    # 1. 初始化模型和分词器 (放在循环外只加载一次)
    print(f"⏳ 正在加载 NT-2.5B 模型权重...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=False
    ).to(DEVICE).eval()

    for species in SPECIES_LIST:
        input_file = f"{INPUT_DIR}/{species}_sequences_NT_6kb.pt"
        output_file = f"{OUTPUT_DIR}/{species}_nt_embeddings.pt"

        if not os.path.exists(input_file):
            print(f"⚠️ 跳过 {species}: 找不到输入文件 {input_file}")
            continue

        print(f"\n🧬 正在处理物种: [{species.upper()}]")

        # 加载序列数据
        data = torch.load(input_file, map_location='cpu')
        sequences = data['sequences']
        gene_ids = data['target_genes']

        all_embeddings = []

        # 批量提取
        with torch.no_grad():
            pbar = tqdm(range(0, len(sequences), BATCH_SIZE), desc=f"{species} 提取进度")
            for i in pbar:
                batch_seqs = sequences[i: i + BATCH_SIZE]

                # 分词
                inputs = tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=MAX_LENGTH
                ).to(DEVICE)

                # 推理
                outputs = model(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    encoder_attention_mask=inputs['attention_mask'],
                    output_hidden_states=True
                )

                # 获取最后一层 Hidden States 并进行 Mean Pooling
                last_hidden = outputs.hidden_states[-1]
                mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)

                # 转为 float32 存入 CPU 以节省显存
                mean_embeds = (sum_embeddings / sum_mask).to(torch.float32).cpu()
                all_embeddings.append(mean_embeds)

                # 每 500 个 batch 手动清理碎片（可选，增加稳定性）
                if i % 500 == 0:
                    torch.cuda.empty_cache()

        # 汇总并保存当前物种
        final_x = torch.cat(all_embeddings, dim=0)
        save_dict = {
            'x': final_x,
            'gene_ids': gene_ids,
            'species': species,
            'model': 'Nucleotide-Transformer-2.5b'
        }
        torch.save(save_dict, output_file)
        print(f"✅ {species.upper()} 完成！形状: {final_x.shape} -> {output_file}")

        # 处理完一个物种后清理内存
        del all_embeddings, final_x, data
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✨ 所有物种提取任务已完成！")


if __name__ == "__main__":
    run_extraction()