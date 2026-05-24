import torch
data = torch.load('processed_ppi/human_ppi_edge_index.pt')
print(f"数据类型: {type(data)}")
if isinstance(data, dict):
    print(f"包含的键: {data.keys()}")