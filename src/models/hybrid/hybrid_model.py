"""
GNN + LightGBM 混合模型
"""
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
from loguru import logger
from typing import Dict, Tuple, Optional
import os

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


class HybridFraudDetector:
    """GNN + LightGBM 混合欺诈检测器"""
    
    def __init__(self, lgb_params: dict = None):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        self.lgb_params = lgb_params or {
            'objective': 'binary',#目标函数为二分类逻辑回归
            'metric': 'auc',#评估指标为ROC曲线下面积，适合不平衡数据
            'boosting_type': 'gbdt',#梯度提升决策树（最常用）
            'num_leaves': 384,
            'max_depth': 15,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 50,
            'reg_alpha': 0.05,#L1正则化，特征选择
            'reg_lambda': 0.05,#L2正则化，权重衰减
            'verbose': -1,#不输出训练日志
            'n_jobs': -1,#使用所有可用CPU核心并行训练
            'is_unbalance': True,#自动调整正负样本权重，处理类别不平衡
            'max_bin': 255,#特征离散化的最大箱数，节省内存
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.best_threshold = 0.5  # 将在验证集上搜索更新
    
    @staticmethod
    def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
        """在验证集上搜索最大化 F1 的最优预测阈值"""
        thresholds = np.arange(0.1, 0.91, 0.02)
        best_f1, best_thr = 0.0, 0.5
        for thr in thresholds:
            preds = (probs >= thr).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, average='binary', zero_division=0
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

    def prepare_features(
        self,
        data,
        embeddings: Dict[str, torch.Tensor],
        mask: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备混合特征"""
        edge_index = data['card', 'transacts', 'merchant'].edge_index
        edge_attr = data['card', 'transacts', 'merchant'].edge_attr
        edge_label = data['card', 'transacts', 'merchant'].edge_label
        
        # 获取mask对应的边
        mask_indices = mask.nonzero(as_tuple=True)[0]
        
        # 节点嵌入特征
        card_emb = embeddings['card'].cpu().numpy()
        merchant_emb = embeddings['merchant'].cpu().numpy()
        
        card_indices = edge_index[0, mask_indices].cpu().numpy()
        merchant_indices = edge_index[1, mask_indices].cpu().numpy()
        
        card_features = card_emb[card_indices]
        merchant_features = merchant_emb[merchant_indices]
        
        # 边特征
        edge_features = edge_attr[mask_indices].cpu().numpy()
        
        # 交互特征
        emb_diff = card_features - merchant_features
        emb_prod = card_features * merchant_features
        
        # 拼接
        X = np.concatenate([
            card_features,
            merchant_features,
            emb_diff,
            emb_prod,
            edge_features,
        ], axis=1)
        
        y = edge_label[mask_indices].cpu().numpy()
        
        # 生成特征名
        if self.feature_names is None:
            n_emb = card_features.shape[1]
            n_edge = edge_features.shape[1]
            self.feature_names = (
                [f'card_emb_{i}' for i in range(n_emb)] +
                [f'merchant_emb_{i}' for i in range(n_emb)] +
                [f'emb_diff_{i}' for i in range(n_emb)] +
                [f'emb_prod_{i}' for i in range(n_emb)] +
                [f'edge_feat_{i}' for i in range(n_edge)]
            )
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        return X, y
    
    def train(
        self,
        data,
        embeddings: Dict[str, torch.Tensor],
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 100
    ) -> Dict:
        """训练LightGBM模型（含 SMOTE 过采样，以 AP 为主指标）"""
        logger.info("Training Hybrid GNN + LightGBM Model")

        X_train, y_train = self.prepare_features(data, embeddings, train_mask)
        X_val, y_val = self.prepare_features(data, embeddings, val_mask)

        # 标准化
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"Train: {len(y_train)} samples, Fraud: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
        logger.info(f"Val: {len(y_val)} samples, Fraud: {y_val.sum()} ({y_val.mean()*100:.2f}%)")

        # SMOTE 过采样（在标准化之后）
        X_train, y_train = self._apply_smote(X_train, y_train)

        # 更新 LightGBM 评估指标为 average_precision
        params = dict(self.lgb_params)
        params['metric'] = 'average_precision'

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100)
        ]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )

        val_pred = self.model.predict(X_val)
        val_auc = roc_auc_score(y_val, val_pred)
        val_ap = average_precision_score(y_val, val_pred)

        # 搜索最优阈值
        self.best_threshold = self.find_optimal_threshold(val_pred, y_val)

        logger.info(f"Validation AP: {val_ap:.4f}  AUC: {val_auc:.4f}  optimal threshold: {self.best_threshold:.2f}")

        return {
            'val_auc': val_auc,
            'val_ap': val_ap,
            'best_iteration': self.model.best_iteration,
            'best_threshold': self.best_threshold,
        }
    
    def predict(
        self,
        data,
        embeddings: Dict[str, torch.Tensor],
        mask: torch.Tensor
    ) -> np.ndarray:
        """预测"""
        X, _ = self.prepare_features(data, embeddings, mask)
        X = self.scaler.transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X)
    
    def evaluate(
        self,
        data,
        embeddings: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        threshold: float = None
    ) -> Dict:
        """评估模型（默认使用验证集搜索到的最优阈值）"""
        X, y = self.prepare_features(data, embeddings, mask)
        X = self.scaler.transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        probs = self.model.predict(X)
        thr = threshold if threshold is not None else self.best_threshold
        preds = (probs >= thr).astype(int)

        auc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)
        precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average='binary', zero_division=0)
        cm = confusion_matrix(y, preds)

        return {
            'auc': auc,
            'ap': ap,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'probabilities': probs,
            'predictions': preds,
            'labels': y,
            'threshold': thr,
        }
    
    def get_feature_importance(self, top_k: int = 30) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model is None:
            return None
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_imp.head(top_k)
    
    def save(self, path: str):
        """保存模型"""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'lgb_params': self.lgb_params,
            'best_threshold': self.best_threshold,
        }, path)
        logger.info(f"Hybrid model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.lgb_params = data['lgb_params']
        self.best_threshold = data.get('best_threshold', 0.5)
        logger.info(f"Hybrid model loaded from {path}")



def train_hybrid_model(
    data,
    gnn_trainer,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    save_path: str = None
) -> Tuple[HybridFraudDetector, Dict]:
    """训练混合模型"""
    logger.info("Getting GNN embeddings...")
    embeddings = gnn_trainer.get_embeddings(data)
    
    # 创建混合模型
    hybrid = HybridFraudDetector()
    
    # 训练
    train_results = hybrid.train(data, embeddings, train_mask, val_mask, 
                                  num_boost_round=5000, early_stopping_rounds=150)
    
    # 测试
    logger.info("Hybrid Model Test Results:")

    test_results = hybrid.evaluate(data, embeddings, test_mask)

    logger.info(f"  Threshold: {test_results['threshold']:.2f}")
    logger.info(f"  AP:        {test_results['ap']:.4f}")
    logger.info(f"  AUC:       {test_results['auc']:.4f}")
    logger.info(f"  Precision: {test_results['precision']:.4f}")
    logger.info(f"  Recall: {test_results['recall']:.4f}")
    logger.info(f"  F1: {test_results['f1']:.4f}")
    logger.info(f"  Confusion Matrix:\n{test_results['confusion_matrix']}")
    
    # 特征重要性
    logger.info("\nTop 20 Important Features:")
    importance = hybrid.get_feature_importance(20)
    for i, row in importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    # 保存
    if save_path:
        hybrid.save(save_path)
        
        # 保存模型结果
        results_path = save_path.replace('.pkl', '_results.json')
        results_json = {
            'test': {
                'auc': float(test_results['auc']),
                'ap': float(test_results['ap']),
                'precision': float(test_results['precision']),
                'recall': float(test_results['recall']),
                'f1': float(test_results['f1']),
                'confusion_matrix': test_results['confusion_matrix'].tolist(),
                'threshold': float(test_results['threshold']),
            },
            'train': {
                'best_iteration': int(train_results.get('best_iteration', 0)),
                'val_auc': float(train_results.get('val_auc', 0)),
                'val_ap': float(train_results.get('val_ap', 0)),
                'best_threshold': float(train_results.get('best_threshold', 0.5)),
            },
            'feature_importance': importance.to_dict('records')
        }
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        logger.info(f"Model results saved to {results_path}")
    
    return hybrid, {
        'train': train_results,
        'test': test_results,
        'feature_importance': importance
    }
