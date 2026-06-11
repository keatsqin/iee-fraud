"""
模型训练模块 - 包含监督训练器和无监督训练器
"""
import torch                              # PyTorch深度学习框架
import torch.nn as nn                     # PyTorch神经网络模块
import torch.nn.functional as F           # PyTorch函数
from torch.optim import Adam, AdamW       # 优化器
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts  # 学习率调度器
from torch_geometric.data import HeteroData  # PyG异构数据
from sklearn.metrics import (               # scikit-learn评估指标
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, average_precision_score
)
from tqdm import tqdm                     # 进度条
import numpy as np                         # NumPy数值计算
from typing import Dict, Tuple, Optional  # 类型提示
from loguru import logger                 # 日志记录
import os                                 # 操作系统接口


class FraudDetectionTrainer:
    """欺诈检测模型训练器 - 负责GNN模型的训练和评估"""

    def __init__(
        self,
        model: nn.Module,                  # 要训练的模型
        device: str = 'auto',              # 设备选择
        lr: float = 0.001,                 # 学习率
        weight_decay: float = 1e-4,        # 权重衰减（L2正则化）
        fraud_weight: float = 10.0,        # 欺诈样本权重（处理类别不平衡）
        use_focal_loss: bool = True,       # 是否使用Focal Loss
        focal_gamma: float = 2.0           # Focal Loss的gamma参数
    ):
        """初始化训练器"""
        # 设备选择：自动选择GPU或CPU
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 将模型移动到指定设备
        self.model = model.to(self.device)
        # 使用AdamW优化器（带权重衰减的Adam）
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # 使用余弦退火学习率调度器 - 能动态调整学习率
        # T_0=20表示每20个epoch重置一次，T_mult=2表示每次重置后周期倍增
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2)

        # 损失函数配置
        self.use_focal_loss = use_focal_loss  # 是否使用Focal Loss
        self.focal_gamma = focal_gamma        # Focal Loss参数
        self.fraud_weight = fraud_weight      # 欺诈样本权重

        # 创建类别权重：正常样本权重1.0，欺诈样本权重fraud_weight
        class_weights = torch.tensor([1.0, fraud_weight]).to(self.device)
        # 使用交叉熵损失，带类别权重
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 训练历史记录
        self.history = {
            'train_loss': [],    # 训练损失历史
            'val_auc': [],       # 验证集AUC历史
            'val_ap': [],        # 验证集AP历史
            'val_f1': [],        # 验证集F1历史
            'lr': []            # 学习率历史
        }

        # 记录初始化信息
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")  # 模型参数总数
        logger.info(f"Using Focal Loss: {use_focal_loss}, gamma={focal_gamma}")

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算Focal Loss - 用于处理类别不平衡问题

        Focal Loss是交叉熵损失的改进版本，通过减少易分类样本的损失，
        让模型更关注难分类的样本。

        Args:
            logits: 模型输出logits [batch_size, 2]
            targets: 真实标签 [batch_size]

        Returns:
            torch.Tensor: Focal Loss值
        """
        # 计算交叉熵损失（不取平均）
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # 计算pt = p(y_pred=true)
        pt = torch.exp(-ce_loss)

        # 根据标签设置权重：欺诈样本权重更高
        weights = torch.where(targets == 1, self.fraud_weight, 1.0)

        # Focal Loss公式：权重 * (1-pt)^gamma * ce_loss
        focal_loss = weights * ((1 - pt) ** self.focal_gamma) * ce_loss
        return focal_loss.mean()

    def _prepare_batch(self, data: HeteroData) -> Tuple:
        """准备批次数据 - 将HeteroData转换为模型可用的格式

        Args:
            data: HeteroData对象

        Returns:
            Tuple: (x_dict, edge_index_dict, edge_index, edge_label, edge_attr, edge_attr_dict)
        """
        # 准备节点特征字典
        x_dict = {
            'card': data['card'].x.to(self.device),
            'merchant': data['merchant'].x.to(self.device)
        }

        # 准备边索引字典（包括正向和反向边）
        edge_index_dict = {
            ('card', 'transacts', 'merchant'):
                data['card', 'transacts', 'merchant'].edge_index.to(self.device),
            ('merchant', 'rev_transacts', 'card'):
                data['merchant', 'rev_transacts', 'card'].edge_index.to(self.device),
        }

        # 准备边相关的张量
        edge_index = data['card', 'transacts', 'merchant'].edge_index.to(self.device)  # 用于预测
        edge_label = data['card', 'transacts', 'merchant'].edge_label.to(self.device)  # 用于计算损失

        # 准备边特征（如果有）
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
        """训练一个epoch

        Args:
            data: HeteroData对象
            train_mask: 训练集掩码 [num_edges]

        Returns:
            float: 本epoch的平均损失
        """
        self.model.train()  # 设置模型为训练模式

        # 准备批次数据
        x_dict, edge_index_dict, edge_index, edge_label, edge_attr, edge_attr_dict = self._prepare_batch(data)
        train_mask = train_mask.to(self.device)

        # 清零梯度
        self.optimizer.zero_grad()

        # 前向传播 - 获取节点嵌入
        node_embeddings = self.model(x_dict, edge_index_dict, edge_attr_dict)

        # 预测边的欺诈概率
        train_edge_attr = edge_attr[train_mask] if edge_attr is not None else None
        predictions = self.model.predict_edge(
            node_embeddings,
            edge_index[:, train_mask],  # 只预测训练集的边
            train_edge_attr
        )

        # 计算损失
        if self.use_focal_loss:
            loss = self.focal_loss(predictions, edge_label[train_mask])  # 使用Focal Loss
        else:
            loss = self.ce_criterion(predictions, edge_label[train_mask])  # 使用交叉熵损失

        # 反向传播
        loss.backward()
        # 梯度裁剪 - 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # 更新参数
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()  # 不计算梯度
    def evaluate(self, data: HeteroData, eval_mask: torch.Tensor) -> Dict:
        """评估模型性能

        Args:
            data: HeteroData对象
            eval_mask: 评估集掩码 [num_edges]

        Returns:
            Dict: 包含各种评估指标的字典
        """
        self.model.eval()  # 设置模型为评估模式

        # 准备批次数据
        x_dict, edge_index_dict, edge_index, edge_label, edge_attr, edge_attr_dict = self._prepare_batch(data)
        eval_mask = eval_mask.to(self.device)

        # 获取节点嵌入
        node_embeddings = self.model(x_dict, edge_index_dict, edge_attr_dict)

        # 预测评估集
        eval_edge_attr = edge_attr[eval_mask] if edge_attr is not None else None
        predictions = self.model.predict_edge(
            node_embeddings,
            edge_index[:, eval_mask],
            eval_edge_attr
        )

        # 计算各种评估指标
        # 转换为概率（第二个类别是欺诈）
        probs = F.softmax(predictions, dim=1)[:, 1].cpu().numpy()
        preds = predictions.argmax(dim=1).cpu().numpy()  # 预测类别
        labels = edge_label[eval_mask].cpu().numpy()    # 真实标签

        # 计算各项指标
        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0  # AUC
        ap = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0  # AP
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0  # 精确率、召回率、F1
        )
        cm = confusion_matrix(labels, preds)  # 混淆矩阵

        return {
            'auc': auc,                         # AUC
            'ap': ap,                           # Average Precision
            'precision': precision,             # 精确率
            'recall': recall,                   # 召回率
            'f1': f1,                           # F1分数
            'confusion_matrix': cm,             # 混淆矩阵
            'predictions': preds,               # 预测结果
            'probabilities': probs,             # 预测概率
            'labels': labels                   # 真实标签
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
        """完整训练流程

        Args:
            data: HeteroData对象
            train_mask: 训练集掩码
            val_mask: 验证集掩码
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_best: 是否保存最佳模型
            save_path: 模型保存路径

        Returns:
            Dict: 训练结果和历史
        """
        # 初始化变量
        best_val_auc = 0                  # 最佳验证集AUC
        patience_counter = 0              # 早停计数器
        best_model_state = None          # 最佳模型状态

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Train edges: {train_mask.sum().item()}, Val edges: {val_mask.sum().item()}")

        # 检查是否有边特征
        if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
            edge_dim = data['card', 'transacts', 'merchant'].edge_attr.shape[1]
            logger.info(f"Using {edge_dim} edge features")
        else:
            logger.warning("No edge features found!")

        # 训练循环
        for epoch in tqdm(range(num_epochs), desc="Training"):
            # 训练一个epoch
            train_loss = self.train_epoch(data, train_mask)
            # 评估验证集
            val_metrics = self.evaluate(data, val_mask)

            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_ap'].append(val_metrics['ap'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['lr'].append(current_lr)

            # Early stopping检查
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            # 每10个epoch或最后一个epoch打印日志
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch}: Loss={train_loss:.4f}, "
                    f"Val AUC={val_metrics['auc']:.4f}, AP={val_metrics['ap']:.4f}, "
                    f"F1={val_metrics['f1']:.4f}, LR={current_lr:.6f}"
                )

            # 早停检查
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # 恢复最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with Val AUC={best_val_auc:.4f}")

        # 保存模型
        if save_best and save_path:
            self.save_model(save_path)

        return {'history': self.history, 'best_val_auc': best_val_auc}

    def test(self, data: HeteroData, test_mask: torch.Tensor) -> Dict:
        """在测试集上评估模型

        Args:
            data: HeteroData对象
            test_mask: 测试集掩码

        Returns:
            Dict: 测试集评估结果
        """
        test_metrics = self.evaluate(data, test_mask)

        # 打印测试结果
        logger.info("=" * 50)
        logger.info("Test Results:")
        logger.info(f"  AUC: {test_metrics['auc']:.4f}")              # 测试集AUC
        logger.info(f"  AP: {test_metrics['ap']:.4f}")               # 测试集AP
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")   # 精确率
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")        # 召回率
        logger.info(f"  F1: {test_metrics['f1']:.4f}")              # F1分数
        logger.info(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")  # 混淆矩阵
        logger.info("=" * 50)

        return test_metrics

    def get_embeddings(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """获取节点嵌入表示

        Args:
            data: HeteroData对象

        Returns:
            Dict[str, torch.Tensor]: 包含节点嵌入的字典
        """
        # 使用模型的get_embeddings方法
        return self.model.get_embeddings(data, self.device)

    def save_model(self, path: str):
        """保存模型到文件

        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)  # 创建目录
        # 保存模型状态、优化器状态和历史记录
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """从文件加载模型

        Args:
            path: 模型文件路径
        """
        checkpoint = torch.load(path, map_location=self.device)  # 加载检查点
        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 加载历史记录（如果存在）
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        logger.info(f"Model loaded from {path}")


class UnsupervisedTrainer:
    """无监督训练器 - 用于图自编码器的训练"""

    def __init__(self, model: nn.Module, device: str = 'auto', lr: float = 0.001):
        """初始化无监督训练器"""
        # 设备选择
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = model.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=lr)  # 使用Adam优化器
        logger.info(f"Unsupervised trainer initialized on device: {self.device}")

    def negative_sampling(self, edge_index, num_nodes, num_neg_samples):
        """负采样 - 生成负样本边

        Args:
            edge_index: 正样本边索引 [2, num_edges]
            num_nodes: (num_cards, num_merchants) 节点数
            num_neg_samples: 负样本数量

        Returns:
            torch.Tensor: 负样本边索引 [2, num_neg_samples]
        """
        # 随机生成负样本
        neg_row = torch.randint(0, num_nodes[0], (num_neg_samples,))
        neg_col = torch.randint(0, num_nodes[1], (num_neg_samples,))
        return torch.stack([neg_row, neg_col], dim=0)

    def train_epoch(self, data: HeteroData):
        """训练一个epoch

        Args:
            data: HeteroData对象

        Returns:
            Tuple: (loss, z_dict) 损失和节点嵌入
        """
        self.model.train()

        # 准备数据
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

        # 准备边特征（如果有）
        edge_attr_dict = None
        if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
            edge_attr = data['card', 'transacts', 'merchant'].edge_attr.to(self.device)
            edge_attr_dict = {
                ('card', 'transacts', 'merchant'): edge_attr,
                ('merchant', 'rev_transacts', 'card'): edge_attr,
            }

        # 正样本边
        pos_edge_index = data['card', 'transacts', 'merchant'].edge_index.to(self.device)

        # 负采样
        num_nodes = (data['card'].x.shape[0], data['merchant'].x.shape[0])
        neg_edge_index = self.negative_sampling(
            pos_edge_index, num_nodes, pos_edge_index.shape[1]
        ).to(self.device)

        # 清零梯度
        self.optimizer.zero_grad()

        # 前向传播
        pos_pred, neg_pred, z_dict = self.model(
            x_dict, edge_index_dict, pos_edge_index, neg_edge_index, edge_attr_dict
        )
        # 计算损失
        loss = self.model.reconstruction_loss(pos_pred, neg_pred)

        # 反向传播
        loss.backward()
        self.optimizer.step()

        return loss.item(), z_dict

    def compute_anomaly_scores(self, data: HeteroData):
        """计算异常分数

        Args:
            data: HeteroData对象

        Returns:
            torch.Tensor: 异常分数 [num_edges]
        """
        self.model.eval()

        with torch.no_grad():  # 不计算梯度
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

            # 准备边特征（如果有）
            edge_attr_dict = None
            if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
                edge_attr = data['card', 'transacts', 'merchant'].edge_attr.to(self.device)
                edge_attr_dict = {
                    ('card', 'transacts', 'merchant'): edge_attr,
                    ('merchant', 'rev_transacts', 'card'): edge_attr,
                }

            # 编码得到节点嵌入
            z_dict = self.model.encode(x_dict, edge_index_dict, edge_attr_dict)
            # 计算异常分数
            anomaly_scores = self.model.anomaly_score(z_dict, data['card', 'transacts', 'merchant'].edge_index.to(self.device))

        return anomaly_scores.cpu()  # 返回到CPU