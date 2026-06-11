"""
GNN+LightGBM 直接对比脚本
无需修改原有代码，直接运行即可

保存为: compare_gnn_lgb.py

使用方法:
    python compare_gnn_lgb.py --data_path data/processed/graph_data.pt
"""

import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
from loguru import logger
import warnings
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# 将项目根目录（fraud-detection-gnn/）添加到路径，与运行目录无关
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# 导入您的原有模块
from src.models.gnn_models import create_model

warnings.filterwarnings('ignore')


class DirectGNNLightGBMComparator:
    """直接对比器 - 可直接运行"""

    def __init__(self, device='auto'):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.results = {}
        logger.info(f"初始化对比器，设备: {self.device}")

    def extract_embeddings(self, model, data):
        """提取GNN节点嵌入"""
        model.eval()
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

            # 边特征
            edge_attr_dict = None
            if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
                edge_attr = data['card', 'transacts', 'merchant'].edge_attr.to(self.device)
                edge_attr_dict = {
                    ('card', 'transacts', 'merchant'): edge_attr,
                    ('merchant', 'rev_transacts', 'card'): edge_attr,
                }

            node_embeddings = model(x_dict, edge_index_dict, edge_attr_dict)

            return node_embeddings['card'].cpu().numpy(), node_embeddings['merchant'].cpu().numpy()

    def build_edge_features(self, data, card_emb, merchant_emb):
        """构建边特征"""
        edge_index = data['card', 'transacts', 'merchant'].edge_index.numpy()

        # 节点嵌入特征
        card_edge_emb = card_emb[edge_index[0]]
        merchant_edge_emb = merchant_emb[edge_index[1]]
        features = np.concatenate([card_edge_emb, merchant_edge_emb], axis=1)

        # 原始边特征
        if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
            edge_attr = data['card', 'transacts', 'merchant'].edge_attr.numpy()
            features = np.concatenate([features, edge_attr], axis=1)
            logger.info(f"添加边特征，维度: {edge_attr.shape[1]}")

        labels = data['card', 'transacts', 'merchant'].edge_label.numpy()

        return features, labels

    def train_lightgbm(self, X_train, y_train, X_val, y_val, model_name):
        """训练LightGBM"""
        # 计算正样本权重
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
            'scale_pos_weight': pos_weight
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        logger.info(f"{model_name} - 训练LightGBM...")

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        # 预测
        y_pred_prob = model.predict(X_val)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # 计算指标
        auc = roc_auc_score(y_val, y_pred_prob)
        ap = average_precision_score(y_val, y_pred_prob)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='binary', zero_division=0
        )

        return model, {'auc': auc, 'ap': ap, 'f1': f1, 'precision': precision, 'recall': recall}

    def compare(self, data, configs):
        """
        直接对比

        Args:
            data: HeteroData数据
            configs: 模型配置列表
                例如: [
                    {'name': 'GraphSAGE', 'type': 'graphsage', 'hidden': 128, 'out': 64},
                    {'name': 'GAT', 'type': 'gat', 'hidden': 128, 'out': 64, 'heads': 4}
                ]
        """
        # 获取数据维度
        card_dim = data['card'].x.shape[1]
        merchant_dim = data['merchant'].x.shape[1]
        edge_dim = data['card', 'transacts', 'merchant'].edge_attr.shape[1] if \
            hasattr(data['card', 'transacts', 'merchant'], 'edge_attr') else 0

        logger.info(f"数据维度: Card={card_dim}, Merchant={merchant_dim}, Edge={edge_dim}")

        # 获取标签
        labels = data['card', 'transacts', 'merchant'].edge_label.numpy()

        # 划分数据集
        indices = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42)

        logger.info(f"数据集划分: 训练={len(train_idx)}, 验证={len(val_idx)}, 测试={len(test_idx)}")

        # 对每个模型进行对比
        for config in configs:
            model_name = config['name']
            logger.info("=" * 60)
            logger.info(f"开始测试: {model_name}")
            logger.info("=" * 60)

            try:
                # 创建GNN模型
                if config['type'] == 'graphsage':
                    model = create_model(
                        model_type='graphsage',
                        card_in_channels=card_dim,
                        merchant_in_channels=merchant_dim,
                        edge_in_channels=edge_dim,
                        hidden_channels=config.get('hidden', 128),
                        out_channels=config.get('out', 64),
                        num_layers=config.get('num_layers', 2),
                        dropout=config.get('dropout', 0.3)
                    )
                elif config['type'] == 'gat':
                    model = create_model(
                        model_type='gat',
                        card_in_channels=card_dim,
                        merchant_in_channels=merchant_dim,
                        edge_in_channels=edge_dim,
                        hidden_channels=config.get('hidden', 128),
                        out_channels=config.get('out', 64),
                        num_layers=config.get('num_layers', 2),
                        heads=config.get('heads', 4),
                        dropout=config.get('dropout', 0.3)
                    )
                else:
                    logger.error(f"未知模型类型: {config['type']}")
                    continue

                model = model.to(self.device)
                logger.info(f"{model_name} - 参数数量: {sum(p.numel() for p in model.parameters()):,}")

                # 提取嵌入
                card_emb, merchant_emb = self.extract_embeddings(model, data)

                # 构建特征
                X_all, y_all = self.build_edge_features(data, card_emb, merchant_emb)

                # 划分特征
                X_train, X_val, X_test = X_all[train_idx], X_all[val_idx], X_all[test_idx]
                y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]

                # 训练LightGBM
                lgb_model, val_metrics = self.train_lightgbm(X_train, y_train, X_val, y_val, model_name)

                # 测试集评估
                y_test_prob = lgb_model.predict(X_test)
                y_test_pred = (y_test_prob > 0.5).astype(int)

                # 计算测试指标
                test_auc = roc_auc_score(y_test, y_test_prob)
                test_ap = average_precision_score(y_test, y_test_prob)
                test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                    y_test, y_test_pred, average='binary', zero_division=0
                )
                test_cm = confusion_matrix(y_test, y_test_pred)

                # 存储结果
                self.results[model_name] = {
                    'val': val_metrics,
                    'test': {
                        'auc': test_auc,
                        'ap': test_ap,
                        'f1': test_f1,
                        'precision': test_precision,
                        'recall': test_recall,
                        'confusion_matrix': test_cm
                    },
                    'y_test': y_test,
                    'y_prob': y_test_prob,
                    'feature_importance': lgb_model.feature_importance(),
                    'model': lgb_model
                }

                logger.info(f"{model_name} - 验证集 AUC={val_metrics['auc']:.4f}, F1={val_metrics['f1']:.4f}")
                logger.info(f"{model_name} - 测试集 AUC={test_auc:.4f}, F1={test_f1:.4f}")

            except Exception as e:
                logger.error(f"{model_name} 失败: {e}")
                import traceback
                traceback.print_exc()

        return self.results

    def print_results(self):
        """打印结果"""
        if not self.results:
            logger.warning("无结果")
            return

        print("\n" + "=" * 100)
        print("GNN + LightGBM 对比结果")
        print("=" * 100)

        # 创建结果表
        df_data = []
        for name, res in self.results.items():
            df_data.append({
                '模型': name,
                '验证AUC': f"{res['val']['auc']:.4f}",
                '验证F1': f"{res['val']['f1']:.4f}",
                '测试AUC': f"{res['test']['auc']:.4f}",
                '测试AP': f"{res['test']['ap']:.4f}",
                '测试F1': f"{res['test']['f1']:.4f}",
                '测试Precision': f"{res['test']['precision']:.4f}",
                '测试Recall': f"{res['test']['recall']:.4f}"
            })

        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))

        # 混淆矩阵
        print("\n" + "-" * 100)
        print("混淆矩阵:")
        for name, res in self.results.items():
            cm = res['test']['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            print(f"\n{name}:")
            print(f"  TN={tn:6d}, FP={fp:6d}")
            print(f"  FN={fn:6d}, TP={tp:6d}")

    def plot_results(self, save_path=None):
        """绘制对比图"""
        if not self.results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # ROC曲线
        ax1 = axes[0, 0]
        for name, res in self.results.items():
            fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
            ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC={res['test']['auc']:.4f})")
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # PR曲线
        ax2 = axes[0, 1]
        for name, res in self.results.items():
            precision, recall, _ = precision_recall_curve(res['y_test'], res['y_prob'])
            ax2.plot(recall, precision, lw=2, label=f"{name} (AP={res['test']['ap']:.4f})")
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 指标对比柱状图
        ax3 = axes[1, 0]
        metrics = ['auc', 'ap', 'f1']
        names = list(self.results.keys())
        x = np.arange(len(metrics))
        width = 0.35

        for i, name in enumerate(names):
            values = [self.results[name]['test'][m] for m in metrics]
            ax3.bar(x + i * width, values, width, label=name)

        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Test Metrics Comparison')
        ax3.set_xticks(x + width / 2)
        ax3.set_xticklabels(['AUC', 'AP', 'F1'])
        ax3.legend()
        ax3.set_ylim([0, 1])

        # 特征重要性对比
        ax4 = axes[1, 1]
        for name, res in self.results.items():
            importance = res['feature_importance']
            top_k = min(20, len(importance))
            top_indices = np.argsort(importance)[-top_k:]
            ax4.barh(range(top_k), importance[top_indices], label=name, alpha=0.7)

        ax4.set_xlabel('Feature Importance')
        ax4.set_title('Feature Importance (Top 20)')
        ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")

        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GNN+LightGBM直接对比')
    parser.add_argument('--data_path', type=str,
                        default=str(_PROJECT_ROOT / 'outputs' / 'graph' / 'fraud_graph.pt'),
                        help='图数据文件路径（.pt）')
    parser.add_argument('--output_dir', type=str,
                        default=str(_PROJECT_ROOT / 'outputs' / 'gnn_lgb_comparison'),
                        help='输出目录')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    logger.info(f"加载数据: {args.data_path}")
    data = torch.load(args.data_path, weights_only=False)

    # 创建对比器
    comparator = DirectGNNLightGBMComparator(device=args.device)

    # 定义要对比的模型
    configs = [
        {
            'name': 'GraphSAGE',
            'type': 'graphsage',
            'hidden': 128,
            'out': 64,
            'num_layers': 2,
            'dropout': 0.3
        },
        {
            'name': 'GAT',
            'type': 'gat',
            'hidden': 128,
            'out': 64,
            'num_layers': 2,
            'heads': 4,
            'dropout': 0.3
        }
    ]

    # 运行对比
    results = comparator.compare(data, configs)

    # 打印结果
    comparator.print_results()

    # 绘制图表
    save_path = os.path.join(args.output_dir, f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    comparator.plot_results(save_path)

    # 保存结果
    results_df = pd.DataFrame([
        {
            'Model': name,
            'Val_AUC': res['val']['auc'],
            'Val_F1': res['val']['f1'],
            'Test_AUC': res['test']['auc'],
            'Test_AP': res['test']['ap'],
            'Test_F1': res['test']['f1'],
            'Test_Precision': res['test']['precision'],
            'Test_Recall': res['test']['recall']
        }
        for name, res in results.items()
    ])

    csv_path = os.path.join(args.output_dir, f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"结果已保存: {csv_path}")

    logger.info("对比完成！")


if __name__ == "__main__":
    main()

    """
    # 指定输出目录
python compare_gnn_lgb.py --data_path data/processed/graph_data.pt --output_dir ./my_results
    """