import torch

# 加载保存的特征
data = torch.load('enformer_features_cache/human_enformer_features.pt', weights_only=False)

# 查看包含哪些键（通常会有 features, gene_ids 等）
print(data.keys())

# 查看特征矩阵的形状
features = data['features']
print(f"特征形状: {features.shape}") # 预期应该是 (61417, 485)