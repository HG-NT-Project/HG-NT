"""
网络来源消融实验模型定义 - 完美解耦版 (Human/Mouse)
修复多卡并行 Bug，将子图完全通过 forward 传递，确保 DDP/DP 架构下的严谨性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DeepRegressor(nn.Module):
    """通用深度回归头 (in -> 512 -> 256 -> 128 -> 1)"""

    def __init__(self, in_dim, dropout=0.3):
        super(DeepRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# =================================================================
# 模型基类：统一接口与基础组件，各消融模型继承后自包含自身逻辑
# =================================================================

class AblationBaseModel(nn.Module):
    """网络消融模型通用基类，隔离基础序列投影与回归头"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(AblationBaseModel, self).__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.regressor = DeepRegressor(hidden_dim, dropout)


# =================================================================
# M3a: 三路全网络 (PPI + TF + GCN) - 包含3个可学习参数
# =================================================================

class ModelM3a_AllNetworks(AblationBaseModel):
    """M3a: 三路全网络 - PPI + TF + GCN 可学习加权和架构"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3a_AllNetworks, self).__init__(input_dim, hidden_dim, dropout)

        self.use_ppi, self.use_tf, self.use_gcn = True, True, True
        self.num_networks = 3

        # 三个卷积层
        self.ppi_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)

        # 可学习融合参数 (3个网络)
        self.fusion_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, x, ppi_sub=None, tf_sub=None, gcn_sub=None):
        """核心重构：子图显式在 forward 中传入，闭绝多卡分发状态泄露 Bug"""
        s_feat = self.seq_proj(x)

        # 三个图卷积通路 (无边时使用全零填充，防止原序列特征梯度串扰)
        p_info = F.elu(self.ppi_conv(s_feat, ppi_sub)) if (
                    ppi_sub is not None and ppi_sub.numel() > 0) else torch.zeros_like(s_feat)
        t_info = F.elu(self.tf_conv(s_feat, tf_sub)) if (
                    tf_sub is not None and tf_sub.numel() > 0) else torch.zeros_like(s_feat)
        g_info = F.elu(self.gcn_conv(s_feat, gcn_sub)) if (
                    gcn_sub is not None and gcn_sub.numel() > 0) else torch.zeros_like(s_feat)

        # Softmax 加权融合
        fusion_weights = F.softmax(self.fusion_logits, dim=0)
        graph_feat = fusion_weights[0] * p_info + fusion_weights[1] * t_info + fusion_weights[2] * g_info

        weights_dict = {
            "aggregation_type": "GCN_weighted_sum",
            "networks_used": "PPI+TF+GCN",
            "num_networks": 3,
            "fusion_type": "learnable_weighted_sum_softmax",
            "has_learnable_weights": True,
            "ppi_weight": fusion_weights[0].item(),
            "tf_weight": fusion_weights[1].item(),
            "gcn_weight": fusion_weights[2].item(),
            "fusion_gate": fusion_weights[0].item()
        }

        # 残差连接
        final_feat = s_feat + graph_feat
        return self.regressor(final_feat).squeeze(-1), weights_dict


# =================================================================
# 单网络消融分支 (M3b, M3c, M3d) - 彻底剔除可学习融合参数
# =================================================================

class ModelM3b_OnlyPPI(AblationBaseModel):
    """M3b: 单路图 - 仅使用 PPI 网络 (无可学习参数)"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3b_OnlyPPI, self).__init__(input_dim, hidden_dim, dropout)
        self.use_ppi, self.use_tf, self.use_gcn = True, False, False
        self.num_networks = 1
        self.ppi_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)

    def forward(self, x, ppi_sub=None, **kwargs):
        s_feat = self.seq_proj(x)
        graph_feat = F.elu(self.ppi_conv(s_feat, ppi_sub)) if (
                    ppi_sub is not None and ppi_sub.numel() > 0) else torch.zeros_like(s_feat)

        weights_dict = {
            "aggregation_type": "GCN_weighted_sum", "networks_used": "PPI", "num_networks": 1,
            "fusion_type": "single_network_direct_pass", "has_learnable_weights": False,
            "ppi_weight": 1.0, "fusion_gate": 1.0
        }
        return self.regressor(s_feat + graph_feat).squeeze(-1), weights_dict


class ModelM3c_OnlyTF(AblationBaseModel):
    """M3c: 单路图 - 仅使用 TF 网络 (无可学习参数)"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3c_OnlyTF, self).__init__(input_dim, hidden_dim, dropout)
        self.use_ppi, self.use_tf, self.use_gcn = False, True, False
        self.num_networks = 1
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)

    def forward(self, x, tf_sub=None, **kwargs):
        s_feat = self.seq_proj(x)
        graph_feat = F.elu(self.tf_conv(s_feat, tf_sub)) if (
                    tf_sub is not None and tf_sub.numel() > 0) else torch.zeros_like(s_feat)

        weights_dict = {
            "aggregation_type": "GCN_weighted_sum", "networks_used": "TF", "num_networks": 1,
            "fusion_type": "single_network_direct_pass", "has_learnable_weights": False,
            "tf_weight": 1.0, "fusion_gate": 1.0
        }
        return self.regressor(s_feat + graph_feat).squeeze(-1), weights_dict


class ModelM3d_OnlyGCN(AblationBaseModel):
    """M3d: 单路图 - 仅使用 GCN 网络 (无可学习参数)"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3d_OnlyGCN, self).__init__(input_dim, hidden_dim, dropout)
        self.use_ppi, self.use_tf, self.use_gcn = False, False, True
        self.num_networks = 1
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)

    def forward(self, x, gcn_sub=None, **kwargs):
        s_feat = self.seq_proj(x)
        graph_feat = F.elu(self.gcn_conv(s_feat, gcn_sub)) if (
                    gcn_sub is not None and gcn_sub.numel() > 0) else torch.zeros_like(s_feat)

        weights_dict = {
            "aggregation_type": "GCN_weighted_sum", "networks_used": "GCN", "num_networks": 1,
            "fusion_type": "single_network_direct_pass", "has_learnable_weights": False,
            "gcn_weight": 1.0, "fusion_gate": 1.0
        }
        return self.regressor(s_feat + graph_feat).squeeze(-1), weights_dict


# =================================================================
# 双网络消融分支 (M3e, M3f, M3g) - 包含2个可学习参数
# =================================================================

class ModelM3e_PPI_TF(AblationBaseModel):
    """M3e: 双路图 - PPI + TF 网络 (包含2个可学习参数)"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3e_PPI_TF, self).__init__(input_dim, hidden_dim, dropout)
        self.use_ppi, self.use_tf, self.use_gcn = True, True, False
        self.num_networks = 2
        self.ppi_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.fusion_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, x, ppi_sub=None, tf_sub=None, **kwargs):
        s_feat = self.seq_proj(x)
        p_info = F.elu(self.ppi_conv(s_feat, ppi_sub)) if (
                    ppi_sub is not None and ppi_sub.numel() > 0) else torch.zeros_like(s_feat)
        t_info = F.elu(self.tf_conv(s_feat, tf_sub)) if (
                    tf_sub is not None and tf_sub.numel() > 0) else torch.zeros_like(s_feat)

        fusion_weights = F.softmax(self.fusion_logits, dim=0)
        graph_feat = fusion_weights[0] * p_info + fusion_weights[1] * t_info

        weights_dict = {
            "aggregation_type": "GCN_weighted_sum", "networks_used": "PPI+TF", "num_networks": 2,
            "fusion_type": "learnable_weighted_sum_softmax", "has_learnable_weights": True,
            "ppi_weight": fusion_weights[0].item(), "tf_weight": fusion_weights[1].item(),
            "fusion_gate": fusion_weights[0].item()
        }
        return self.regressor(s_feat + graph_feat).squeeze(-1), weights_dict


class ModelM3f_PPI_GCN(AblationBaseModel):
    """M3f: 双路图 - PPI + GCN 网络 (包含2个可学习参数)"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3f_PPI_GCN, self).__init__(input_dim, hidden_dim, dropout)
        self.use_ppi, self.use_tf, self.use_gcn = True, False, True
        self.num_networks = 2
        self.ppi_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.fusion_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, x, ppi_sub=None, gcn_sub=None, **kwargs):
        s_feat = self.seq_proj(x)
        p_info = F.elu(self.ppi_conv(s_feat, ppi_sub)) if (
                    ppi_sub is not None and ppi_sub.numel() > 0) else torch.zeros_like(s_feat)
        g_info = F.elu(self.gcn_conv(s_feat, gcn_sub)) if (
                    gcn_sub is not None and gcn_sub.numel() > 0) else torch.zeros_like(s_feat)

        fusion_weights = F.softmax(self.fusion_logits, dim=0)
        graph_feat = fusion_weights[0] * p_info + fusion_weights[1] * g_info

        weights_dict = {
            "aggregation_type": "GCN_weighted_sum", "networks_used": "PPI+GCN", "num_networks": 2,
            "fusion_type": "learnable_weighted_sum_softmax", "has_learnable_weights": True,
            "ppi_weight": fusion_weights[0].item(), "gcn_weight": fusion_weights[1].item(),
            "fusion_gate": fusion_weights[0].item()
        }
        return self.regressor(s_feat + graph_feat).squeeze(-1), weights_dict


class ModelM3g_TF_GCN(AblationBaseModel):
    """M3g: 双路图 - TF + GCN 网络 (包含2个可学习参数)"""

    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.3):
        super(ModelM3g_TF_GCN, self).__init__(input_dim, hidden_dim, dropout)
        self.use_ppi, self.use_tf, self.use_gcn = False, True, True
        self.num_networks = 2
        self.tf_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.fusion_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, x, tf_sub=None, gcn_sub=None, **kwargs):
        s_feat = self.seq_proj(x)
        t_info = F.elu(self.tf_conv(s_feat, tf_sub)) if (
                    tf_sub is not None and tf_sub.numel() > 0) else torch.zeros_like(s_feat)
        g_info = F.elu(self.gcn_conv(s_feat, gcn_sub)) if (
                    gcn_sub is not None and gcn_sub.numel() > 0) else torch.zeros_like(s_feat)

        fusion_weights = F.softmax(self.fusion_logits, dim=0)
        graph_feat = fusion_weights[0] * t_info + fusion_weights[1] * g_info

        weights_dict = {
            "aggregation_type": "GCN_weighted_sum", "networks_used": "TF+GCN", "num_networks": 2,
            "fusion_type": "learnable_weighted_sum_softmax", "has_learnable_weights": True,
            "tf_weight": fusion_weights[0].item(), "gcn_weight": fusion_weights[1].item(),
            "fusion_gate": fusion_weights[0].item()
        }
        return self.regressor(s_feat + graph_feat).squeeze(-1), weights_dict


# =================================================================
# 模型工厂函数
# =================================================================

def build_model(model_name, input_dim=2560, dropout=0.3):
    models = {
        'm3a': ModelM3a_AllNetworks, 'm3b': ModelM3b_OnlyPPI, 'm3c': ModelM3c_OnlyTF,
        'm3d': ModelM3d_OnlyGCN, 'm3e': ModelM3e_PPI_TF, 'm3f': ModelM3f_PPI_GCN, 'm3g': ModelM3g_TF_GCN,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. 可选: {list(models.keys())}")
    return models[model_name](input_dim=input_dim, dropout=dropout)


MODEL_NAMES = ['m3a', 'm3b', 'm3c', 'm3d', 'm3e', 'm3f', 'm3g']

if __name__ == "__main__":
    print("=" * 60)
    print("测试完美重构版模型正向传播与工厂模式")
    print("=" * 60)

    input_dim, batch_size = 2560, 4
    mock_edge = torch.zeros((2, 0), dtype=torch.long)

    for model_name in MODEL_NAMES:
        model = build_model(model_name, input_dim=input_dim)
        total_params = sum(p.numel() for p in model.parameters())
        has_fusion = hasattr(model, 'fusion_logits') and model.fusion_logits is not None

        print(f"{model_name.upper()}: {total_params:,} 参数 | 可学习融合参数: {'有' if has_fusion else '无'}")

        # 模拟正向传播，完全解耦子图赋值
        x = torch.randn(batch_size, input_dim)
        output, weights = model(x, ppi_sub=mock_edge, tf_sub=mock_edge, gcn_sub=mock_edge)
        print(f"  --> 输出 Tensor 维度: {output.shape} | 统计记录类型: {weights['fusion_type']}")
    print("\n✅ 所有独立解耦消融模型通过测试，可以无缝挂载于新 Trainer 中！")