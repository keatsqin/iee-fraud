"""
PyTorch Geometric 图构建模块
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from loguru import logger
from typing import Tuple, Dict, Optional


class PyGGraphBuilder:
    """PyG图构建器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.card_mapping = {}  # card_id -> node_index
        self.merchant_mapping = {}  # merchant_id -> node_index
        self.card_scaler = StandardScaler()
        self.merchant_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        
    def load_from_parquet(self, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """从Parquet或CSV文件加载数据"""
        logger.info(f"Loading data from {data_dir}")
        
        # 加载Card特征（优先CSV，parquet可能是目录格式）
        card_csv = os.path.join(data_dir, 'card_features.csv')
        card_parquet = os.path.join(data_dir, 'card_features.parquet')
        if os.path.exists(card_csv) and os.path.isfile(card_csv):
            card_df = pd.read_csv(card_csv)
        elif os.path.exists(card_parquet):
            # parquet可能是文件或目录（Spark格式）
            card_df = pd.read_parquet(card_parquet)
        else:
            raise FileNotFoundError(f"Card features not found in {data_dir}")
        logger.info(f"Card features loaded: {len(card_df)} cards")
        
        # 加载Merchant特征
        merchant_csv = os.path.join(data_dir, 'merchant_features.csv')
        merchant_parquet = os.path.join(data_dir, 'merchant_features.parquet')
        if os.path.exists(merchant_csv) and os.path.isfile(merchant_csv):
            merchant_df = pd.read_csv(merchant_csv)
        elif os.path.exists(merchant_parquet):
            merchant_df = pd.read_parquet(merchant_parquet)
        else:
            raise FileNotFoundError(f"Merchant features not found in {data_dir}")
        logger.info(f"Merchant features loaded: {len(merchant_df)} merchants")
        
        # 加载边数据
        edges_csv = os.path.join(data_dir, 'edges.csv')
        edges_parquet = os.path.join(data_dir, 'edges.parquet')
        if os.path.exists(edges_csv) and os.path.isfile(edges_csv):
            edges_df = pd.read_csv(edges_csv)
        elif os.path.exists(edges_parquet):
            edges_df = pd.read_parquet(edges_parquet)
        else:
            raise FileNotFoundError(f"Edges not found in {data_dir}")
        logger.info(f"Edges loaded: {len(edges_df)} transactions")
        
        # 加载图统计（可选）
        stats_path = os.path.join(data_dir, 'graph_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.graph_stats = json.load(f)
            logger.info(f"Graph stats loaded: {self.graph_stats}")
        
        return card_df, merchant_df, edges_df
    
    def _build_node_mappings(self, card_df: pd.DataFrame, merchant_df: pd.DataFrame):
        """构建节点ID到索引的映射，将业务ID（字符串）转换为连续整数编号的过程，让图算法能高效处理"""
        def normalize_id(x):
            try:
                return str(int(float(x)))
            except:
                return str(x)#防止同一ID多种格式
        
        card_ids = card_df['card_id'].apply(normalize_id).unique()
        self.card_mapping = {card_id: idx for idx, card_id in enumerate(card_ids)}
        
        # Merchant映射 - 保持字符串格式
        merchant_ids = merchant_df['merchant_id'].astype(str).unique()
        self.merchant_mapping = {merchant_id: idx for idx, merchant_id in enumerate(merchant_ids)}#enumerate	自动生成连续整数索引（0,1,2...）
        
        logger.info(f"Node mappings built: {len(self.card_mapping)} cards, {len(self.merchant_mapping)} merchants")
    
    def _prepare_card_features(self, card_df: pd.DataFrame) -> torch.Tensor:
        #表示多维数组（张量），用于存储数值数据并进行 GPU 加速计算
        """准备Card节点特征张量"""
        feature_cols = [col for col in card_df.columns if col != 'card_id']#张量（Tensor） 是多维数组的数学概念，是标量、向量、矩阵的高维推广
        
        # 按映射顺序排列
        card_df = card_df.set_index('card_id')
        
        features = []
        for card_id in self.card_mapping.keys():
            if card_id in card_df.index:
                features.append(card_df.loc[card_id, feature_cols].values)
            else:
                features.append(np.zeros(len(feature_cols)))
        
        features = np.array(features, dtype=np.float32)
        
        # 处理NaN和Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准化
        features = self.card_scaler.fit_transform(features)
        
        logger.info(f"Card features shape: {features.shape}")
        return torch.tensor(features, dtype=torch.float32)
    
    def _prepare_merchant_features(self, merchant_df: pd.DataFrame) -> torch.Tensor:
        """准备Merchant节点特征张量"""
        feature_cols = [col for col in merchant_df.columns if col != 'merchant_id']
        
        merchant_df = merchant_df.set_index('merchant_id')
        
        features = []
        for merchant_id in self.merchant_mapping.keys():
            if merchant_id in merchant_df.index:
                features.append(merchant_df.loc[merchant_id, feature_cols].values)
            else:
                features.append(np.zeros(len(feature_cols)))
        
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = self.merchant_scaler.fit_transform(features)
        
        logger.info(f"Merchant features shape: {features.shape}")
        return torch.tensor(features, dtype=torch.float32)

    def _prepare_edges(self, edges_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备边索引、边特征和边标签"""
        exclude_cols = ['TransactionID', 'card_id', 'merchant_id', 'isFraud', 
                       'card_id_str', 'merchant_id_str', 'card_idx', 'merchant_idx']
        
        # 所有数值列作为边特征
        edge_feature_cols = [col for col in edges_df.columns 
                            if col not in exclude_cols and edges_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        logger.info(f"Using {len(edge_feature_cols)} edge features: {edge_feature_cols[:10]}...")
        
        # 定义ID标准化函数
        def normalize_card_id(x):
            try:
                return str(int(float(x)))
            except:
                return str(x)
        
        # 向量化处理ID,将边数据中的ID标准化为字符串格式
        edges_df = edges_df.copy()
        edges_df['card_id_str'] = edges_df['card_id'].apply(normalize_card_id)# 标准化卡片ID
        edges_df['merchant_id_str'] = edges_df['merchant_id'].astype(str)# 商户ID转字符串
        
        # 映射到索引
        edges_df['card_idx'] = edges_df['card_id_str'].map(self.card_mapping)
        edges_df['merchant_idx'] = edges_df['merchant_id_str'].map(self.merchant_mapping)
        
        # 过滤无法映射的边
        valid_mask = edges_df['card_idx'].notna() & edges_df['merchant_idx'].notna()
        unmapped_count = (~valid_mask).sum()#布尔掩码,取反，计算未映射（无效）边的数量
        
        if unmapped_count > 0:
            logger.warning(f"Skipped {unmapped_count} edges due to unmapped nodes")
        
        valid_edges = edges_df[valid_mask]
        
        # 转换为张量，将有效的边索引转换为 PyTorch 张量，构建图神经网络所需的 edge_index
        card_indices = valid_edges['card_idx'].astype(int).values
        merchant_indices = valid_edges['merchant_idx'].astype(int).values
        edge_index = torch.tensor([card_indices, merchant_indices], dtype=torch.long)#合并为 PyTorch 长整型张量
        
        # 边特征
        edge_features = valid_edges[edge_feature_cols].fillna(0).values.astype(np.float32)
        edge_features = np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准化
        edge_features = self.edge_scaler.fit_transform(edge_features)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        # 边标签
        edge_labels = valid_edges['isFraud'].fillna(0).astype(int).values
        edge_label = torch.tensor(edge_labels, dtype=torch.long)
        
        logger.info(f"Edges prepared: {edge_index.shape[1]} edges, {edge_attr.shape[1]} features")
        logger.info(f"Fraud ratio: {edge_label.sum().item() / len(edge_label):.4f}")
        
        return edge_index, edge_attr, edge_label
    
    def build_hetero_graph(
        self, 
        card_df: pd.DataFrame, 
        merchant_df: pd.DataFrame, 
        edges_df: pd.DataFrame
    ) -> HeteroData:
        """构建PyG异构图"""
        logger.info("Building heterogeneous graph...")
        
        # 1. 构建节点映射
        self._build_node_mappings(card_df, merchant_df)
        
        # 2. 准备节点特征
        card_features = self._prepare_card_features(card_df)
        merchant_features = self._prepare_merchant_features(merchant_df)
        
        # 3. 准备边数据
        edge_index, edge_attr, edge_label = self._prepare_edges(edges_df)
        
        # 4. 构建HeteroData
        data = HeteroData()
        
        # 节点特征
        data['card'].x = card_features
        data['card'].num_nodes = len(self.card_mapping)#该类型节点的总数量，卡片映射字典的长度（即唯一卡片数）
        
        data['merchant'].x = merchant_features
        data['merchant'].num_nodes = len(self.merchant_mapping)
        
        # 边 (card -> merchant): 交易关系
        data['card', 'transacts', 'merchant'].edge_index = edge_index# 边连接关系
        data['card', 'transacts', 'merchant'].edge_attr = edge_attr# 边特征
        data['card', 'transacts', 'merchant'].edge_label = edge_label  # 边标签（监督学习）
        
        # 反向边 (merchant -> card): 用于消息双向传递
        data['merchant', 'rev_transacts', 'card'].edge_index = edge_index.flip(0)
        data['merchant', 'rev_transacts', 'card'].edge_attr = edge_attr
        
        # 保存映射（用于后续分析）
        data.card_mapping = self.card_mapping
        data.merchant_mapping = self.merchant_mapping
        
        # 保存特征列名（用于解释）
        data.card_feature_names = [col for col in card_df.columns if col != 'card_id']
        data.merchant_feature_names = [col for col in merchant_df.columns if col != 'merchant_id']
        
        logger.info(f"HeteroData built: {data}")
        
        return data

    def split_edges_by_time(
        self, 
        data: HeteroData, 
        edges_df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15#定义验证集比例的参数，表示将 15% 的数据作为验证集。validation ratio
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按时间顺序划分边"""
        logger.info("Splitting edges by time...")
        
        # 按时间排序
        edges_df = edges_df.sort_values('TransactionDT').reset_index(drop=True)
        
        num_edges = len(edges_df)
        train_size = int(num_edges * train_ratio)
        val_size = int(num_edges * val_ratio)
        
        # 创建mask
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)
        
        train_mask[:train_size] = True
        val_mask[train_size:train_size + val_size] = True
        test_mask[train_size + val_size:] = True
        
        # 统计各集合的欺诈率
        edge_label = data['card', 'transacts', 'merchant'].edge_label
        
        train_fraud_rate = edge_label[train_mask].float().mean().item()#将单元素张量转换为 Python 标量
        val_fraud_rate = edge_label[val_mask].float().mean().item()
        test_fraud_rate = edge_label[test_mask].float().mean().item()
        
        logger.info(f"Train: {train_mask.sum().item()} edges, fraud rate: {train_fraud_rate:.4f}")
        logger.info(f"Val: {val_mask.sum().item()} edges, fraud rate: {val_fraud_rate:.4f}")
        logger.info(f"Test: {test_mask.sum().item()} edges, fraud rate: {test_fraud_rate:.4f}")
        
        return train_mask, val_mask, test_mask
    
    def save_graph(self, data: HeteroData, output_path: str):
        """保存图数据"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(data, output_path)
        logger.info(f"Graph saved to {output_path}")
    
    def load_graph(self, input_path: str) -> HeteroData:
        """加载图数据"""
        data = torch.load(input_path, weights_only=False)
        logger.info(f"Graph loaded from {input_path}")
        return data


def build_graph_from_parquet(
    data_dir: str,
    output_path: str = None,
    config: dict = None
) -> HeteroData:
    """从Parquet文件构建图"""
    builder = PyGGraphBuilder(config)
    
    # 加载数据
    card_df, merchant_df, edges_df = builder.load_from_parquet(data_dir)
    
    # 构建图
    data = builder.build_hetero_graph(card_df, merchant_df, edges_df)
    
    # 划分数据集
    train_mask, val_mask, test_mask = builder.split_edges_by_time(data, edges_df)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # 保存
    if output_path:
        builder.save_graph(data, output_path)
    
    return data


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python graph_builder.py <parquet_dir> [output_path]")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/fraud_graph.pt"
    
    data = build_graph_from_parquet(data_dir, output_path)
    print(f"\nGraph built successfully!")
    print(f"Card nodes: {data['card'].num_nodes}")
    print(f"Merchant nodes: {data['merchant'].num_nodes}")
    print(f"Edges: {data['card', 'transacts', 'merchant'].edge_index.shape[1]}")
