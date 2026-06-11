"""
集成 Borderline Shifting 思想
"""
import json
import os
import numpy as np
import torch
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
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

    def train_with_borderline_shifting(
        self,
        data,
        embeddings: Dict[str, torch.Tensor],
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 100,
        borderline_low: float = 0.30,
        borderline_high: float = 0.70,
        strategy: str = 'both',
        borderline_weight: float = 2.0,
        borderline_copy_times: int = 1,
    ) -> Dict:
        """边界样本增强训练：先识别边界样本，再对同类训练样本加权/复制并重训。"""
        X_train, y_train = self.prepare_features(data, embeddings, train_mask)
        X_val, y_val = self.prepare_features(data, embeddings, val_mask)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        X_train, y_train = self._apply_smote(X_train, y_train)

        params = dict(self.lgb_params)
        params['metric'] = 'average_precision'

        base_train = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        base_val = lgb.Dataset(X_val, label=y_val, reference=base_train)

        base_model = lgb.train(
            params,
            base_train,
            num_boost_round=num_boost_round,
            valid_sets=[base_train, base_val],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )

        val_probs = base_model.predict(X_val)
        border_mask = (val_probs >= borderline_low) & (val_probs <= borderline_high)
        border_count = int(border_mask.sum())

        train_weights = np.ones(len(y_train), dtype=np.float32)
        matched_train_count = 0

        if border_count > 0:
            val_border_labels = y_val[border_mask]
            target_indices_by_label = {
                0: np.where(y_train == 0)[0],
                1: np.where(y_train == 1)[0],
            }

            matched_train_idx = []
            for lb in (0, 1):
                need = int((val_border_labels == lb).sum())
                if need <= 0:
                    continue
                pool = target_indices_by_label[lb]
                if len(pool) == 0:
                    continue
                choice = np.random.choice(pool, size=need, replace=(need > len(pool)))
                matched_train_idx.extend(choice.tolist())

            matched_train_idx = np.array(matched_train_idx, dtype=np.int64) if matched_train_idx else np.array([], dtype=np.int64)
            matched_train_count = len(matched_train_idx)

            if matched_train_count > 0 and strategy in ('weight', 'both'):
                train_weights[matched_train_idx] *= borderline_weight

            if matched_train_count > 0 and strategy in ('copy', 'both') and borderline_copy_times > 0:
                X_extra = np.repeat(X_train[matched_train_idx], borderline_copy_times, axis=0)
                y_extra = np.repeat(y_train[matched_train_idx], borderline_copy_times, axis=0)
                w_extra = np.repeat(train_weights[matched_train_idx], borderline_copy_times, axis=0)
                X_train = np.concatenate([X_train, X_extra], axis=0)
                y_train = np.concatenate([y_train, y_extra], axis=0)
                train_weights = np.concatenate([train_weights, w_extra], axis=0)

        logger.info(
            f"Borderline Shifting: val_border={border_count}, matched_train={matched_train_count}, "
            f"strategy={strategy}, weight={borderline_weight}, copy_times={borderline_copy_times}"
        )

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=train_weights,
            feature_name=self.feature_names,
        )
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )

        val_pred = self.model.predict(X_val)
        val_auc = roc_auc_score(y_val, val_pred)
        val_ap = average_precision_score(y_val, val_pred)
        self.best_threshold = self.find_optimal_threshold(val_pred, y_val)

        logger.info(
            f"Validation AP: {val_ap:.4f}  AUC: {val_auc:.4f}  "
            f"optimal threshold: {self.best_threshold:.2f}"
        )

        return {
            'val_auc': float(val_auc),
            'val_ap': float(val_ap),
            'best_iteration': int(self.model.best_iteration),
            'best_threshold': float(self.best_threshold),
            'borderline_samples': int(border_count),
            'matched_train_samples': int(matched_train_count),
        }
