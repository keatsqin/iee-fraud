"""
方案二：特征级融合（Feature-level Fusion）
将 CommunityDetector 计算的社群统计特征注入 LightGBM 输入特征中。
每条边继承其所属社群的统计特征（密度、卡商比、拓扑/行为异常分等）。
"""
import json
import os
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from loguru import logger

from .hybrid_model import HybridFraudDetector
from .community_detector import CommunityDetector


# 社群特征列名（固定顺序，共 14 维）
COMMUNITY_FEATURE_NAMES = [
    'comm_total_transactions',
    'comm_fraud_rate',
    'comm_num_cards',
    'comm_num_merchants',
    'comm_density',
    'comm_card_merchant_ratio',
    'comm_time_concentration',
    'comm_night_ratio',
    'comm_amt_mean',
    'comm_amt_std',
    'comm_round_amt_ratio',
    'comm_topology_anomaly',
    'comm_behavior_anomaly',
    'comm_risk_score',
]

_N_COMM_FEATS = len(COMMUNITY_FEATURE_NAMES)


class CommunityFeatureFusionDetector(HybridFraudDetector):
    """
    特征级融合检测器：在 HybridFraudDetector 基础上，
    将每条边所属社群的统计特征拼接到 LightGBM 输入中。

    使用流程：
        1. 训练 GNN，获取 embeddings
        2. 运行 CommunityDetector（detect + compute_community_stats）
        3. detector.set_community_detector(community_detector)
        4. detector.train(...)  /  detector.evaluate(...)
    """

    def __init__(self, lgb_params: dict = None):
        super().__init__(lgb_params)
        self._community_detector: Optional[CommunityDetector] = None
        self._node_to_comm: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 社群检测器绑定
    # ------------------------------------------------------------------

    def set_community_detector(self, detector: CommunityDetector):
        """
        绑定已完成社群检测和统计计算的 CommunityDetector。
        必须在 train / predict / evaluate 之前调用。
        """
        if not detector.community_stats:
            raise ValueError(
                "CommunityDetector.compute_community_stats() must be called before binding"
            )
        self._community_detector = detector

        # 构建 node_key -> community_id 映射
        self._node_to_comm = {}
        for comm_id, nodes in detector.communities.items():
            for node in nodes:
                self._node_to_comm[node] = comm_id

        logger.info(
            f"Community detector bound: {len(detector.communities)} communities, "
            f"{len(self._node_to_comm)} nodes mapped"
        )

    # ------------------------------------------------------------------
    # 社群特征构建
    # ------------------------------------------------------------------

    def _get_comm_feature_vector(self, comm_id: Optional[int]) -> np.ndarray:
        """返回指定社群的 14 维特征向量；社群不存在时返回全零向量。"""
        if comm_id is None or self._community_detector is None:
            return np.zeros(_N_COMM_FEATS, dtype=np.float32)

        stats = self._community_detector.community_stats.get(comm_id)
        if stats is None:
            return np.zeros(_N_COMM_FEATS, dtype=np.float32)

        num_merchants = stats['num_merchants']
        card_merchant_ratio = (
            stats['num_cards'] / num_merchants if num_merchants > 0 else 0.0
        )

        return np.array([
            stats['total_transactions'],
            stats['fraud_rate'],
            stats['num_cards'],
            stats['num_merchants'],
            stats['density'],
            card_merchant_ratio,
            stats.get('time_concentration', 0.0),
            stats.get('night_ratio', 0.0),
            stats.get('amt_mean', 0.0),
            stats.get('amt_std', 0.0),
            stats.get('round_amt_ratio', 0.0),
            self._community_detector.compute_topology_anomaly(comm_id),
            self._community_detector.compute_behavior_anomaly(comm_id),
            self._community_detector.compute_risk_score(comm_id),
        ], dtype=np.float32)

    def _build_community_features(
        self,
        card_indices: np.ndarray,
        merchant_indices: np.ndarray,
    ) -> np.ndarray:
        """
        为每条边构建社群特征矩阵 [num_edges, _N_COMM_FEATS]。
        边所属社群 = 卡节点和商户节点同属的社群；若不同则视为无社群（全零）。
        """
        n = len(card_indices)
        comm_feats = np.zeros((n, _N_COMM_FEATS), dtype=np.float32)

        for i in range(n):
            card_node = f"card_{card_indices[i]}"
            merchant_node = f"merchant_{merchant_indices[i]}"
            card_comm = self._node_to_comm.get(card_node)
            merch_comm = self._node_to_comm.get(merchant_node)
            # 只有两端节点在同一社群时才赋予社群特征
            if card_comm is not None and card_comm == merch_comm:
                comm_feats[i] = self._get_comm_feature_vector(card_comm)

        in_comm = int(np.any(comm_feats != 0, axis=1).sum())
        logger.debug(f"Community features: {in_comm}/{n} edges matched a community")
        return comm_feats

    # ------------------------------------------------------------------
    # 特征准备（覆盖父类）
    # ------------------------------------------------------------------

    def prepare_features(
        self,
        data,
        embeddings: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在父类特征（card_emb + merchant_emb + emb_diff + emb_prod + edge_feat）
        基础上，拼接 14 维社群统计特征。
        """
        if self._community_detector is None:
            logger.warning("No community detector set; community features will be all zeros")

        # 父类特征
        X_base, y = super().prepare_features(data, embeddings, mask)

        # 重新获取边的节点索引（与父类逻辑一致）
        edge_index = data['card', 'transacts', 'merchant'].edge_index
        mask_indices = mask.nonzero(as_tuple=True)[0]
        card_indices = edge_index[0, mask_indices].cpu().numpy()
        merchant_indices = edge_index[1, mask_indices].cpu().numpy()

        comm_feats = self._build_community_features(card_indices, merchant_indices)
        X = np.concatenate([X_base, comm_feats], axis=1)

        # 更新特征名（仅首次，父类已写入 base 部分）
        if self.feature_names is not None and len(self.feature_names) == X_base.shape[1]:
            self.feature_names = self.feature_names + COMMUNITY_FEATURE_NAMES

        logger.info(
            f"Feature-level fusion: {X_base.shape[1]} base + "
            f"{_N_COMM_FEATS} community = {X.shape[1]} total features"
        )
        return X, y
