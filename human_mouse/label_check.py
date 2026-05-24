import torch
import numpy as np

# 加载标签
data = torch.load('processed_labels/mouse_labels.pt', map_location='cpu')
labels = data['labels'].numpy()

# 检查范围
print(f"Min: {labels.min():.4f}")
print(f"Max: {labels.max():.4f}")
print(f"Mean: {labels.mean():.4f}")
print(f"Max/Min: {labels.max() / (labels.min() + 1e-8):.2f}")

# 如果是log处理过的，原始值应该是 expm1(label)
if labels.min() >= 0 and labels.max() < 20:  # log空间的典型范围
    print("⚠️ 标签看起来可能已经在log空间")
    original_scale = np.expm1(labels)
    print(f"原始尺度范围: {original_scale.min():.2f} - {original_scale.max():.2f}")