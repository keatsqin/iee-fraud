"""
模型模块
包含GNN模型、社群检测器和混合模型
"""
from .gnn_models import (
    HeteroGraphSAGE, 
    HeteroGAT, 
    GraphAutoEncoder,
    create_model
)
from .community_detector import CommunityDetector

# 混合模型（需要lightgbm）
try:
    from .hybrid_model import HybridFraudDetector, train_hybrid_model
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False

__all__ = [
    'HeteroGraphSAGE',
    'HeteroGAT',
    'GraphAutoEncoder',
    'create_model',
    'CommunityDetector',
    'HybridFraudDetector',
    'train_hybrid_model',
]
