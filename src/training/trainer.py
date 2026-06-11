"""
模型训练模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch_geometric.data import HeteroData
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, average_precision_score
)
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger
import os

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    logger.warning("imbalanced-learn not installed. SMOTE disabled. Install with: pip install imbalanced-learn")


class FraudDetectionTrainer:
    """欺诈检测模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        fraud_weight: float = 10.0,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0
    ):
        """初始化训练器"""
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 使用余弦退火学习率
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2)
        
        # 损失函数
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.fraud_weight = fraud_weight
        
        class_weights = torch.tensor([1.0, fraud_weight]).to(self.device)
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.history = {
            'train_loss': [], 'val_auc': [], 'val_ap': [], 'val_f1': [], 'lr': []
        }
        self.best_threshold = 0.5  # 将在验证集上搜索更新
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Using Focal Loss: {use_focal_loss}, gamma={focal_gamma}")
    
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

    def _apply_smote(self, edge_index, edge_attr, edge_label, train_mask):
        """对训练边特征应用 SMOTE 过采样，返回增强后的 (edge_index, edge_attr, labels)"""
        mask_indices = train_mask.nonzero(as_tuple=True)[0]
        orig_edge_index = edge_index[:, mask_indices]
        orig_labels = edge_label[mask_indices]

        if not HAS_SMOTE or edge_attr is None:
            orig_edge_attr = edge_attr[mask_indices] if edge_attr is not None else None
            return orig_edge_index, orig_edge_attr, orig_labels

        orig_edge_attr = edge_attr[mask_indices]
        y = orig_labels.cpu().numpy()
        fraud_count = int(y.sum())

        if fraud_count < 6:
            return orig_edge_index, orig_edge_attr, orig_labels

        X = orig_edge_attr.cpu().numpy()
        k = min(5, fraud_count - 1)
        try:
            smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=k)
            X_res, y_res = smote.fit_resample(X, y)
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}, skipping augmentation")
            return orig_edge_index, orig_edge_attr, orig_labels

        n_orig = len(y)
        X_synthetic = X_res[n_orig:]
        y_synthetic = y_res[n_orig:]

        # 为合成样本找最近的真实欺诈边，借用其节点索引
        fraud_bool = y == 1
        X_fraud = X[fraud_bool]
        fraud_positions = mask_indices[fraud_bool].cpu().numpy()

        X_fraud_n = X_fraud / (np.linalg.norm(X_fraud, axis=1, keepdims=True) + 1e-8)
        X_syn_n = X_synthetic / (np.linalg.norm(X_synthetic, axis=1, keepdims=True) + 1e-8)
        nearest = (X_syn_n @ X_fraud_n.T).argmax(axis=1)
        syn_positions = fraud_positions[nearest]

        syn_edge_index = edge_index[:, syn_positions]
        syn_edge_attr = torch.tensor(X_synthetic, dtype=torch.float32, device=self.device)
        syn_labels = torch.tensor(y_synthetic, dtype=torch.long, device=self.device)

        aug_edge_index = torch.cat([orig_edge_index, syn_edge_index], dim=1)
        aug_edge_attr = torch.cat([orig_edge_attr, syn_edge_attr], dim=0)
        aug_labels = torch.cat([orig_labels, syn_labels], dim=0)

        logger.debug(f"SMOTE: {n_orig} → {len(y_res)} 训练样本 (+{len(y_synthetic)} 合成欺诈边)")
        return aug_edge_index, aug_edge_attr, aug_labels

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 类别权重
        weights = torch.where(targets == 1, self.fraud_weight, 1.0)
        
        focal_loss = weights * ((1 - pt) ** self.focal_gamma) * ce_loss
        return focal_loss.mean()
    
    def _prepare_batch(self, data: HeteroData) -> Tuple:
        """准备批次数据"""
        x_dict = {
            'card': data['card'].x.to(self.device),
            'merchant': data['merchant'].x.to(self.device)
        }
        
        edge_index_dict = {
            ('card', 'transacts', 'merchant'): 
                data['card', 'transacts', 'merchant'].edge_index.to(self.device),
            ('merchant', 'rev_transacts', 'card'): 
                data['merchant', 'rev_transacts', 'card'].edge_index.to(self.device),
        }
        
        edge_index = data['card', 'transacts', 'merchant'].edge_index.to(self.device)
        edge_label = data['card', 'transacts', 'merchant'].edge_label.to(self.device)
        
        # 边特征
        edge_attr = None
        edge_attr_dict = None
        if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
            edge_attr = data['card', 'transacts', 'merchant'].edge_attr.to(self.device)
            edge_attr_dict = {
                ('card', 'transacts', 'merchant'): edge_attr,
                ('merchant', 'rev_transacts', 'card'): edge_attr,
            }
        
        return x_dict, edge_index_dict, edge_index, edge_label, edge_attr, edge_attr_dict
    
    def train_epoch(self, data: HeteroData, train_mask: torch.Tensor) -> float:
        """训练一个epoch（含 SMOTE 过采样）"""
        self.model.train()

        x_dict, edge_index_dict, edge_index, edge_label, edge_attr, edge_attr_dict = self._prepare_batch(data)
        train_mask = train_mask.to(self.device)

        self.optimizer.zero_grad()

        node_embeddings = self.model(x_dict, edge_index_dict, edge_attr_dict)

        # SMOTE 增强训练边
        aug_edge_index, aug_edge_attr, aug_labels = self._apply_smote(
            edge_index, edge_attr, edge_label, train_mask
        )

        predictions = self.model.predict_edge(node_embeddings, aug_edge_index, aug_edge_attr)

        if self.use_focal_loss:
            loss = self.focal_loss(predictions, aug_labels)
        else:
            loss = self.ce_criterion(predictions, aug_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, data: HeteroData, eval_mask: torch.Tensor) -> Dict:
        """评估模型"""
        self.model.eval()
        
        x_dict, edge_index_dict, edge_index, edge_label, edge_attr, edge_attr_dict = self._prepare_batch(data)
        eval_mask = eval_mask.to(self.device)
        
        node_embeddings = self.model(x_dict, edge_index_dict, edge_attr_dict)
        
        eval_edge_attr = edge_attr[eval_mask] if edge_attr is not None else None
        predictions = self.model.predict_edge(
            node_embeddings, 
            edge_index[:, eval_mask],
            eval_edge_attr
        )
        
        # 计算指标
        probs = F.softmax(predictions, dim=1)[:, 1].cpu().numpy()
        labels = edge_label[eval_mask].cpu().numpy()

        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        ap = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

        # 用最优阈值（若已搜索）做二值化，否则用 0.5
        threshold = getattr(self, 'best_threshold', 0.5)
        preds = (probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        cm = confusion_matrix(labels, preds)

        return {
            'auc': auc, 'ap': ap, 'precision': precision, 'recall': recall, 'f1': f1,
            'confusion_matrix': cm, 'predictions': preds, 'probabilities': probs, 'labels': labels
        }

    def train(
        self,
        data: HeteroData,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        num_epochs: int = 200,
        early_stopping_patience: int = 30,
        save_best: bool = True,
        save_path: str = None
    ) -> Dict:
        """完整训练流程（以 AP 为主指标做 early stopping，训练后搜索最优阈值）"""
        best_val_ap = 0.0
        patience_counter = 0
        best_model_state = None

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Train edges: {train_mask.sum().item()}, Val edges: {val_mask.sum().item()}")
        if HAS_SMOTE:
            logger.info("SMOTE oversampling enabled for training")

        if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
            edge_dim = data['card', 'transacts', 'merchant'].edge_attr.shape[1]
            logger.info(f"Using {edge_dim} edge features")
        else:
            logger.warning("No edge features found!")

        for epoch in tqdm(range(num_epochs), desc="Training"):
            train_loss = self.train_epoch(data, train_mask)
            val_metrics = self.evaluate(data, val_mask)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_ap'].append(val_metrics['ap'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['lr'].append(current_lr)

            # 以 AP 为主指标做 early stopping
            if val_metrics['ap'] > best_val_ap:
                best_val_ap = val_metrics['ap']
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                # 同步搜索最优阈值
                self.best_threshold = self.find_optimal_threshold(
                    val_metrics['probabilities'], val_metrics['labels']
                )
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch}: Loss={train_loss:.4f}, "
                    f"Val AP={val_metrics['ap']:.4f}, AUC={val_metrics['auc']:.4f}, "
                    f"F1={val_metrics['f1']:.4f}, Thr={self.best_threshold:.2f}, LR={current_lr:.6f}"
                )

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info(
                f"Restored best model: Val AP={best_val_ap:.4f}, "
                f"optimal threshold={self.best_threshold:.2f}"
            )

        if save_best and save_path:
            self.save_model(save_path)

        return {'history': self.history, 'best_val_ap': best_val_ap, 'best_threshold': self.best_threshold}
    
    def test(self, data: HeteroData, test_mask: torch.Tensor) -> Dict:
        """测试模型（使用验证集搜索到的最优阈值）"""
        test_metrics = self.evaluate(data, test_mask)

        logger.info("=" * 50)
        logger.info("Test Results:")
        logger.info(f"  Threshold: {self.best_threshold:.2f}")
        logger.info(f"  AP:        {test_metrics['ap']:.4f}")
        logger.info(f"  AUC:       {test_metrics['auc']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"  F1:        {test_metrics['f1']:.4f}")
        logger.info(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")
        logger.info("=" * 50)

        return test_metrics
    
    def get_embeddings(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """获取节点嵌入"""
        return self.model.get_embeddings(data, self.device)
    
    def save_model(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_threshold': self.best_threshold,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        if 'best_threshold' in checkpoint:
            self.best_threshold = checkpoint['best_threshold']
        logger.info(f"Model loaded from {path}")


class UnsupervisedTrainer:
    """无监督训练器 - 用于图自编码器"""
    
    def __init__(self, model: nn.Module, device: str = 'auto', lr: float = 0.001):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = model.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=lr)
        logger.info(f"Unsupervised trainer initialized on device: {self.device}")
    
    def negative_sampling(self, edge_index, num_nodes, num_neg_samples):
        neg_row = torch.randint(0, num_nodes[0], (num_neg_samples,))
        neg_col = torch.randint(0, num_nodes[1], (num_neg_samples,))
        return torch.stack([neg_row, neg_col], dim=0)
    
    def train_epoch(self, data: HeteroData):
        self.model.train()
        
        x_dict = {
            'card': data['card'].x.to(self.device),
            'merchant': data['merchant'].x.to(self.device)
        }
        edge_index_dict = {
            ('card', 'transacts', 'merchant'): 
                data['card', 'transacts', 'merchant'].edge_index.to(self.device),
            ('merchant', 'rev_transacts', 'card'): 
                data['merchant', 'rev_transacts', 'card'].edge_index.to(self.device),
        }
        
        edge_attr_dict = None
        if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
            edge_attr = data['card', 'transacts', 'merchant'].edge_attr.to(self.device)
            edge_attr_dict = {
                ('card', 'transacts', 'merchant'): edge_attr,
                ('merchant', 'rev_transacts', 'card'): edge_attr,
            }
        
        pos_edge_index = data['card', 'transacts', 'merchant'].edge_index.to(self.device)
        
        num_nodes = (data['card'].x.shape[0], data['merchant'].x.shape[0])
        neg_edge_index = self.negative_sampling(
            pos_edge_index, num_nodes, pos_edge_index.shape[1]
        ).to(self.device)
        
        self.optimizer.zero_grad()
        pos_pred, neg_pred, z_dict = self.model(
            x_dict, edge_index_dict, pos_edge_index, neg_edge_index, edge_attr_dict
        )
        loss = self.model.reconstruction_loss(pos_pred, neg_pred)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), z_dict
    
    def compute_anomaly_scores(self, data: HeteroData):
        self.model.eval()
        
        with torch.no_grad():
            x_dict = {
                'card': data['card'].x.to(self.device),
                'merchant': data['merchant'].x.to(self.device)
            }
            edge_index_dict = {
                ('card', 'transacts', 'merchant'): 
                    data['card', 'transacts', 'merchant'].edge_index.to(self.device),
                ('merchant', 'rev_transacts', 'card'): 
                    data['merchant', 'rev_transacts', 'card'].edge_index.to(self.device),
            }
            
            edge_attr_dict = None
            if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
                edge_attr = data['card', 'transacts', 'merchant'].edge_attr.to(self.device)
                edge_attr_dict = {
                    ('card', 'transacts', 'merchant'): edge_attr,
                    ('merchant', 'rev_transacts', 'card'): edge_attr,
                }
            
            edge_index = data['card', 'transacts', 'merchant'].edge_index.to(self.device)
            z_dict = self.model.encode(x_dict, edge_index_dict, edge_attr_dict)
            anomaly_scores = self.model.anomaly_score(z_dict, edge_index)
        
        return anomaly_scores.cpu()
