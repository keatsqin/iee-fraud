"""
GNN + LightGBM 混合模型（含社群级特征注入）
===================================================
在 hybrid_model.py 基础上新增：
  1. prepare_features_with_community()  — 拼接社群级特征
  2. train_with_community() / predict_with_community() / evaluate_with_community()
  3. final_score = α * lgb_prob + (1-α) * community_risk_score  融合推理
  4. 输出边列表时附加 fused_probability 字段
  5. train_hybrid_model_with_community() — 复制原 train_hybrid_model 接口，新增消融对比
"""
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    logger.warning("imbalanced-learn not installed. SMOTE disabled. Install with: pip install imbalanced-learn")


# ──────────────────────────────────────────────────────────────────────────────
# 社群数据加载工具
# ──────────────────────────────────────────────────────────────────────────────

def load_community_data(community_json_path: str) -> Tuple[Dict[int, Dict], Dict[str, int]]:
    """
    从 suspicious_communities.json 读取社群统计和节点→社群映射。

    Returns
    -------
    community_stats : Dict[int, Dict]
        {community_id: {fraud_rate, density, ...}}
    node_to_community : Dict[str, int]
        {"card_0": 0, "card_32": 0, ...}
    """
    with open(community_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # stats key 可能是字符串，统一转为 int
    community_stats: Dict[int, Dict] = {}
    for k, v in raw.get("stats", {}).items():
        cid = int(k)
        stats = dict(v)
        # fraud_transactions 有时存为字符串
        if "fraud_transactions" in stats:
            stats["fraud_transactions"] = int(stats["fraud_transactions"])
        community_stats[cid] = stats

    # 构建 node → community 映射
    node_to_community: Dict[str, int] = {}
    for k, members in raw.get("communities", {}).items():
        cid = int(k)
        for node in members:
            node_to_community[node] = cid

    logger.info(
        f"Loaded {len(community_stats)} community stats, "
        f"{len(node_to_community)} node→community mappings"
    )
    return community_stats, node_to_community


# ──────────────────────────────────────────────────────────────────────────────
# 主类
# ──────────────────────────────────────────────────────────────────────────────

class HybridFraudDetectorWithCommunity:
    """
    GNN + LightGBM 混合欺诈检测器（含社群级特征）

    与 HybridFraudDetector 保持相同的 LightGBM 超参，仅在特征层新增
    6 维社群特征，以便消融实验中完全受控其他变量。
    """

    # 融合推理权重：final_score = α * lgb_prob + (1-α) * community_risk_score
    ALPHA: float = 0.7

    def __init__(self, lgb_params: dict = None, alpha: float = 0.7):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")

        self.ALPHA = alpha

        # ── 与 hybrid_model.py 完全相同的超参，保证消融对照 ──
        self.lgb_params = lgb_params or {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 384,
            "max_depth": 15,
            "learning_rate": 0.03,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "min_child_samples": 50,
            "reg_alpha": 0.05,
            "reg_lambda": 0.05,
            "verbose": -1,
            "n_jobs": -1,
            "is_unbalance": True,
            "max_bin": 255,
        }

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.best_threshold = 0.5  # 将在验证集上搜索更新

    # ── 特征构建 ──────────────────────────────────────────────────────────────

    @staticmethod
    def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
        """在验证集上搜索最大化 F1 的最优预测阈值"""
        thresholds = np.arange(0.1, 0.91, 0.02)
        best_f1, best_thr = 0.0, 0.5
        for thr in thresholds:
            preds = (probs >= thr).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        return best_thr

    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对训练特征应用 SMOTE 过采样"""
        if not HAS_SMOTE:
            return X, y
        fraud_count = int(y.sum())
        if fraud_count < 6:
            return X, y
        k = min(5, fraud_count - 1)
        try:
            smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=k)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(f"SMOTE: {len(y)} → {len(y_res)} 训练样本 (+{len(y_res)-len(y)} 合成欺诈样本)")
            return X_res, y_res
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}, skipping augmentation")
            return X, y

    def _extract_community_features(
        self,
        card_indices: np.ndarray,
        community_stats: Dict[int, Dict],
        node_to_community: Dict[str, int],
    ) -> np.ndarray:
        """
        为每条边（由 card_index 标识）提取 6 维社群特征。

        特征列（与用户需求规格完全一致）:
          0  fraud_rate           社群欺诈率
          1  density              社群密度
          2  time_concentration   时间集中度
          3  night_ratio          夜间交易比例
          4  card_merchant_ratio  num_cards / max(num_merchants, 1)
          5  log_txn_norm         log1p(total_transactions) / 10 截断到1
        """
        rows = []
        for idx in card_indices:
            card_node = f"card_{idx}"
            comm_id = node_to_community.get(card_node, -1)

            if comm_id != -1 and comm_id in community_stats:
                s = community_stats[comm_id]
                feat = [
                    float(s["fraud_rate"]),
                    float(s["density"]),
                    float(s.get("time_concentration", 0.0)),
                    float(s.get("night_ratio", 0.0)),
                    float(s["num_cards"]) / max(float(s["num_merchants"]), 1.0),
                    min(float(np.log1p(s["total_transactions"])) / 10.0, 1.0),
                ]
            else:
                feat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            rows.append(feat)

        return np.array(rows, dtype=np.float32)

    def _community_risk_score(
        self,
        card_indices: np.ndarray,
        community_stats: Dict[int, Dict],
        node_to_community: Dict[str, int],
    ) -> np.ndarray:
        """
        推理阶段用于融合的标量社群风险分（0~1）。
        直接取该卡所属社群的 fraud_rate，
        不在任何社群的节点默认给全局均值 0.05。
        """
        scores = []
        for idx in card_indices:
            card_node = f"card_{idx}"
            comm_id = node_to_community.get(card_node, -1)
            if comm_id != -1 and comm_id in community_stats:
                scores.append(float(community_stats[comm_id]["fraud_rate"]))
            else:
                scores.append(0.05)
        return np.array(scores, dtype=np.float32)

    def prepare_features_with_community(
        self,
        data,
        embeddings: Dict[str, torch.Tensor],
        community_stats: Dict[int, Dict],
        node_to_community: Dict[str, int],
        mask: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备包含社群特征的特征矩阵"""
        edge_index = data["card", "transacts", "merchant"].edge_index
        edge_attr = data["card", "transacts", "merchant"].edge_attr
        edge_label = data["card", "transacts", "merchant"].edge_label

        mask_indices = mask.nonzero(as_tuple=True)[0]

        # 节点嵌入
        card_emb = embeddings["card"].cpu().numpy()
        merchant_emb = embeddings["merchant"].cpu().numpy()

        card_indices = edge_index[0, mask_indices].cpu().numpy()
        merchant_indices = edge_index[1, mask_indices].cpu().numpy()

        card_features = card_emb[card_indices]
        merchant_features = merchant_emb[merchant_indices]

        edge_features = edge_attr[mask_indices].cpu().numpy()

        emb_diff = card_features - merchant_features
        emb_prod = card_features * merchant_features

        # ── 新增：社群级特征 ──────────────────────────────────────────────────
        community_features = self._extract_community_features(
            card_indices, community_stats, node_to_community
        )

        X = np.concatenate(
            [card_features, merchant_features, emb_diff, emb_prod,
             edge_features, community_features],
            axis=1,
        )

        y = edge_label[mask_indices].cpu().numpy()

        # 特征名（只在首次调用时初始化，避免 train/val/test 重复追加）
        if self.feature_names is None:
            n_emb = card_features.shape[1]
            n_edge = edge_features.shape[1]
            self.feature_names = (
                [f"card_emb_{i}" for i in range(n_emb)]
                + [f"merchant_emb_{i}" for i in range(n_emb)]
                + [f"emb_diff_{i}" for i in range(n_emb)]
                + [f"emb_prod_{i}" for i in range(n_emb)]
                + [f"edge_feat_{i}" for i in range(n_edge)]
                + [f"comm_{i}" for i in range(community_features.shape[1])]
            )

        logger.info(
            f"Prepared {X.shape[0]} samples with {X.shape[1]} features "
            f"(+{community_features.shape[1]} community features)"
        )
        return X, y

    # ── 训练 ──────────────────────────────────────────────────────────────────

    def train(
        self,
        data,
        embeddings: Dict[str, torch.Tensor],
        community_stats: Dict[int, Dict],
        node_to_community: Dict[str, int],
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 100,
    ) -> Dict:
        """训练含社群特征的 LightGBM 模型（含 SMOTE，以 AP 为主指标）"""
        logger.info("Training Hybrid GNN + LightGBM + Community Model")

        X_train, y_train = self.prepare_features_with_community(
            data, embeddings, community_stats, node_to_community, train_mask
        )
        X_val, y_val = self.prepare_features_with_community(
            data, embeddings, community_stats, node_to_community, val_mask
        )

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(
            f"Train: {len(y_train)} samples, Fraud: {int(y_train.sum())} "
            f"({y_train.mean()*100:.2f}%)"
        )
        logger.info(
            f"Val:   {len(y_val)} samples, Fraud: {int(y_val.sum())} "
            f"({y_val.mean()*100:.2f}%)"
        )

        # SMOTE 过采样
        X_train, y_train = self._apply_smote(X_train, y_train)

        # 以 average_precision 为主评估指标
        params = dict(self.lgb_params)
        params["metric"] = "average_precision"

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        val_pred = self.model.predict(X_val)
        val_auc = roc_auc_score(y_val, val_pred)
        val_ap = average_precision_score(y_val, val_pred)

        # 搜索最优阈值
        self.best_threshold = self.find_optimal_threshold(val_pred, y_val)

        logger.info(f"Validation AP: {val_ap:.4f}  AUC: {val_auc:.4f}  optimal threshold: {self.best_threshold:.2f}")

        return {
            "val_auc": val_auc,
            "val_ap": val_ap,
            "best_iteration": self.model.best_iteration,
            "best_threshold": self.best_threshold,
        }

    # ── 推理 ──────────────────────────────────────────────────────────────────

    def predict_with_community(
        self,
        data,
        embeddings: Dict[str, torch.Tensor],
        community_stats: Dict[int, Dict],
        node_to_community: Dict[str, int],
        mask: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        推理：返回 (lgb_prob, fused_prob)

        fused_prob = α * lgb_prob + (1-α) * community_risk_score
        """
        X, _ = self.prepare_features_with_community(
            data, embeddings, community_stats, node_to_community, mask
        )
        X = self.scaler.transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        lgb_prob = self.model.predict(X)

        # 社群风险分（标量）
        edge_index = data["card", "transacts", "merchant"].edge_index
        mask_indices = mask.nonzero(as_tuple=True)[0]
        card_indices = edge_index[0, mask_indices].cpu().numpy()
        comm_risk = self._community_risk_score(card_indices, community_stats, node_to_community)

        fused_prob = self.ALPHA * lgb_prob + (1 - self.ALPHA) * comm_risk
        return lgb_prob, fused_prob

    # ── 评估 ──────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        data,
        embeddings: Dict[str, torch.Tensor],
        community_stats: Dict[int, Dict],
        node_to_community: Dict[str, int],
        mask: torch.Tensor,
        threshold: float = None,
    ) -> Dict:
        """
        评估模型（默认使用验证集搜索到的最优阈值），同时报告：
          - lgb_only 指标（消融对照）
          - fused    指标（α * lgb + (1-α) * community_risk）
        """
        X, y = self.prepare_features_with_community(
            data, embeddings, community_stats, node_to_community, mask
        )
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        lgb_prob = self.model.predict(X_scaled)

        edge_index = data["card", "transacts", "merchant"].edge_index
        mask_indices = mask.nonzero(as_tuple=True)[0]
        card_indices = edge_index[0, mask_indices].cpu().numpy()
        comm_risk = self._community_risk_score(card_indices, community_stats, node_to_community)

        fused_prob = self.ALPHA * lgb_prob + (1 - self.ALPHA) * comm_risk

        thr = threshold if threshold is not None else self.best_threshold

        def _metrics(probs, label, thr):
            preds = (probs >= thr).astype(int)
            auc = roc_auc_score(label, probs)
            ap = average_precision_score(label, probs)
            prec, rec, f1, _ = precision_recall_fscore_support(
                label, preds, average="binary", zero_division=0
            )
            cm = confusion_matrix(label, preds)
            return dict(auc=auc, ap=ap, precision=prec, recall=rec, f1=f1,
                        confusion_matrix=cm, probabilities=probs, predictions=preds,
                        threshold=thr)

        lgb_metrics = _metrics(lgb_prob, y, thr)
        fused_metrics = _metrics(fused_prob, y, thr)

        edge_output = []
        for i, (lp, fp, label) in enumerate(zip(lgb_prob, fused_prob, y)):
            edge_output.append({
                "edge_idx": int(mask_indices[i].item()),
                "card_node": f"card_{card_indices[i]}",
                "lgb_probability": float(lp),
                "community_risk_score": float(comm_risk[i]),
                "fused_probability": float(fp),
                "label": int(label),
            })

        return {
            "lgb_only": lgb_metrics,
            "fused": fused_metrics,
            "labels": y,
            "alpha": self.ALPHA,
            "edge_output": edge_output,
        }

    # ── 特征重要性 ────────────────────────────────────────────────────────────

    def get_feature_importance(self, top_k: int = 30) -> pd.DataFrame:
        if self.model is None:
            return None
        importance = self.model.feature_importance(importance_type="gain")
        df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)
        return df.head(top_k)

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def save(self, path: str):
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "lgb_params": self.lgb_params,
                "alpha": self.ALPHA,
                "best_threshold": self.best_threshold,
            },
            path,
        )
        logger.info(f"HybridCommunity model saved to {path}")

    def load(self, path: str):
        import joblib
        d = joblib.load(path)
        self.model = d["model"]
        self.scaler = d["scaler"]
        self.feature_names = d["feature_names"]
        self.lgb_params = d["lgb_params"]
        self.ALPHA = d.get("alpha", 0.7)
        self.best_threshold = d.get("best_threshold", 0.5)
        logger.info(f"HybridCommunity model loaded from {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 顶层训练函数（含消融对比）
# ──────────────────────────────────────────────────────────────────────────────

def train_hybrid_model_with_community(
    data,
    gnn_trainer,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    community_json_path: str,
    baseline_results_path: Optional[str] = None,
    save_path: Optional[str] = None,
    alpha: float = 0.7,
) -> Tuple["HybridFraudDetectorWithCommunity", Dict]:
    """
    训练含社群特征的混合模型，并与基线（hybrid_model.py）结果做消融对比。

    Parameters
    ----------
    data              : PyG HeteroData
    gnn_trainer       : FraudDetectionTrainer（已训练完的 GNN）
    train_mask        : 训练边掩码
    val_mask          : 验证边掩码
    test_mask         : 测试边掩码
    community_json_path : suspicious_communities.json 路径
    baseline_results_path : hybrid_gnn_lgb_results.json 路径（可选，用于对比）
    save_path         : 模型保存路径（.pkl）
    alpha             : 融合权重 α
    """
    # 1. GNN 嵌入
    logger.info("Getting GNN embeddings...")
    embeddings = gnn_trainer.get_embeddings(data)

    # 2. 社群数据
    community_stats, node_to_community = load_community_data(community_json_path)

    # 3. 创建并训练模型
    hybrid = HybridFraudDetectorWithCommunity(alpha=alpha)
    train_results = hybrid.train(
        data, embeddings, community_stats, node_to_community,
        train_mask, val_mask,
        num_boost_round=5000, early_stopping_rounds=150,
    )

    # 4. 测试集评估
    logger.info("Evaluating on test set...")
    eval_results = hybrid.evaluate(
        data, embeddings, community_stats, node_to_community, test_mask
    )

    lgb_res = eval_results["lgb_only"]
    fused_res = eval_results["fused"]

    logger.info("=" * 60)
    logger.info("Hybrid + Community Model — LGB-only (ablation baseline)")
    logger.info(f"  Threshold: {lgb_res['threshold']:.2f}")
    logger.info(f"  AP:        {lgb_res['ap']:.4f}")
    logger.info(f"  AUC:       {lgb_res['auc']:.4f}")
    logger.info(f"  Precision: {lgb_res['precision']:.4f}")
    logger.info(f"  Recall:    {lgb_res['recall']:.4f}")
    logger.info(f"  F1:        {lgb_res['f1']:.4f}")
    logger.info(f"  Confusion Matrix:\n{lgb_res['confusion_matrix']}")

    logger.info("─" * 60)
    logger.info(f"Hybrid + Community Model — Fused (α={alpha})")
    logger.info(f"  Threshold: {fused_res['threshold']:.2f}")
    logger.info(f"  AP:        {fused_res['ap']:.4f}")
    logger.info(f"  AUC:       {fused_res['auc']:.4f}")
    logger.info(f"  Precision: {fused_res['precision']:.4f}")
    logger.info(f"  Recall:    {fused_res['recall']:.4f}")
    logger.info(f"  F1:        {fused_res['f1']:.4f}")
    logger.info(f"  Confusion Matrix:\n{fused_res['confusion_matrix']}")

    # 5. 特征重要性
    logger.info("\nTop 20 Important Features:")
    importance = hybrid.get_feature_importance(20)
    for _, row in importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")

    # 6. 消融对比（与基线 JSON 结果）
    ablation_delta = None
    if baseline_results_path and os.path.isfile(baseline_results_path):
        with open(baseline_results_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        base_test = baseline.get("test", {})

        def _delta(new_val, base_val):
            diff = new_val - base_val
            sign = "+" if diff >= 0 else ""
            return f"{sign}{diff:+.4f}"

        logger.info("=" * 60)
        logger.info("Ablation Study — Δ vs Baseline (hybrid_model.py)")
        logger.info(f"  {'Metric':<12} {'Baseline':>10} {'LGB+Comm':>10} "
                    f"{'Fused':>10} {'Δ(LGB+C)':>10} {'Δ(Fused)':>10}")
        for metric in ("auc", "ap", "precision", "recall", "f1"):
            b = base_test.get(metric, 0.0)
            lc = lgb_res[metric]
            fu = fused_res[metric]
            logger.info(
                f"  {metric:<12} {b:>10.4f} {lc:>10.4f} {fu:>10.4f} "
                f"{_delta(lc, b):>10} {_delta(fu, b):>10}"
            )

        ablation_delta = {
            metric: {
                "baseline": float(base_test.get(metric, 0.0)),
                "lgb_with_community": float(lgb_res[metric]),
                "fused": float(fused_res[metric]),
                "delta_lgb_comm": float(lgb_res[metric] - base_test.get(metric, 0.0)),
                "delta_fused": float(fused_res[metric] - base_test.get(metric, 0.0)),
            }
            for metric in ("auc", "ap", "precision", "recall", "f1")
        }

    # 7. 保存
    if save_path:
        hybrid.save(save_path)

        results_path = save_path.replace(".pkl", "_results.json")
        results_json = {
            "model": "hybrid_gnn_lgb_community",
            "alpha": alpha,
            "train": {
                "best_iteration": int(train_results.get("best_iteration", 0)),
                "val_auc": float(train_results.get("val_auc", 0)),
                "val_ap": float(train_results.get("val_ap", 0)),
            },
            "test_lgb_only": {
                "auc": float(lgb_res["auc"]),
                "ap": float(lgb_res["ap"]),
                "precision": float(lgb_res["precision"]),
                "recall": float(lgb_res["recall"]),
                "f1": float(lgb_res["f1"]),
                "confusion_matrix": lgb_res["confusion_matrix"].tolist(),
                "threshold": float(lgb_res["threshold"]),
            },
            "test_fused": {
                "auc": float(fused_res["auc"]),
                "ap": float(fused_res["ap"]),
                "precision": float(fused_res["precision"]),
                "recall": float(fused_res["recall"]),
                "f1": float(fused_res["f1"]),
                "confusion_matrix": fused_res["confusion_matrix"].tolist(),
                "threshold": float(fused_res["threshold"]),
            },
            "ablation_vs_baseline": ablation_delta,
            "feature_importance": importance.to_dict("records"),
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {results_path}")

        # 保存边级 fused_probability 输出
        edge_output_path = save_path.replace(".pkl", "_edge_output.json")
        with open(edge_output_path, "w", encoding="utf-8") as f:
            json.dump(eval_results["edge_output"], f, indent=2)
        logger.info(f"Edge-level fused output saved to {edge_output_path}")

    return hybrid, {
        "train": train_results,
        "lgb_only": lgb_res,
        "fused": fused_res,
        "ablation": ablation_delta,
        "feature_importance": importance,
        "edge_output": eval_results["edge_output"],
    }
