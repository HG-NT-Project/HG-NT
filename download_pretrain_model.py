import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

repo_id = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
local_dir = "pretrain_model/Nucleotide-Transformer"

# 手动补下缺失的核心权重分片
file_to_download = "pytorch_model-00001-of-00002.bin"

print(f"🚀 正在补全缺失的核心权重: {file_to_download}...")
hf_hub_download(
    repo_id=repo_id,
    filename=file_to_download,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("✅ 权重补全成功！")