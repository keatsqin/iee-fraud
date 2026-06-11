"""
社群检测模块
"""
import numpy as np
import torch
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from loguru import logger

try:
    import sys as _sys
    import importlib as _importlib
    # 防止同目录的 community.py 遮蔽 python-louvain 包
    _saved = [p for p in _sys.path if 'src' in p or 'models' in p]
    for _p in _saved:
        _sys.path.remove(_p)
    import community.community_louvain as community_louvain
    for _p in _saved:
        _sys.path.insert(0, _p)
except ImportError:
    community_louvain = None
    logger.warning("python-louvain not installed, Louvain detection will be unavailable")


class CommunityDetector:
    """社群检测器"""
    
    def __init__(self):
        self.G = None  # NetworkX图
        self.communities = {}  # 社群字典
        self.community_stats = {}  # 社群统计
        self.suspicious_communities = []  # 可疑社群列表
        
    def build_networkx_graph(
        self, 
        edge_index: torch.Tensor, 
        card_mapping: Dict, 
        merchant_mapping: Dict,
        edge_attrs: Optional[Dict] = None
    ) -> nx.Graph:
        """
        从PyG数据构建NetworkX图
        
        Args:
            edge_index: 边索引 [2, num_edges]
            card_mapping: card_id -> node_index
            merchant_mapping: merchant_id -> node_index
            edge_attrs: 边属性字典（可选）
            
        Returns:
            NetworkX图
        """
        logger.info("Building NetworkX graph...")
        
        G = nx.Graph()
        
        # 反转映射
        idx_to_card = {v: k for k, v in card_mapping.items()}
        idx_to_merchant = {v: k for k, v in merchant_mapping.items()}
        
        # 添加节点
        for card_id, idx in card_mapping.items():
            G.add_node(f"card_{idx}", node_type='card', original_id=card_id)
        
        for merchant_id, idx in merchant_mapping.items():
            G.add_node(f"merchant_{idx}", node_type='merchant', original_id=merchant_id)
        
        # 添加边
        edge_index = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
        
        for i in range(edge_index.shape[1]):
            card_node = f"card_{edge_index[0, i]}"
            merchant_node = f"merchant_{edge_index[1, i]}"
            
            if G.has_edge(card_node, merchant_node):
                G[card_node][merchant_node]['weight'] += 1
            else:
                G.add_edge(card_node, merchant_node, weight=1)
        
        self.G = G
        logger.info(f"NetworkX graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G

    def detect_louvain(self, resolution: float = 1.0) -> Dict[int, List[str]]:
        """
        使用Louvain算法进行社群检测
        
        Args:
            resolution: 分辨率参数（越大社群越小）
            
        Returns:
            社群字典 {community_id: [node_ids]}
        """
        if community_louvain is None:
            raise ImportError("python-louvain not installed")
        
        if self.G is None:
            raise ValueError("Please build NetworkX graph first")
        
        logger.info(f"Running Louvain community detection (resolution={resolution})...")
        
        partition = community_louvain.best_partition(
            self.G, 
            weight='weight',
            resolution=resolution
        )
        
        
        # 整理社群
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        self.communities = dict(communities)
        logger.info(f"Louvain detected {len(self.communities)} communities")
        
        return self.communities
    
    def detect_embedding_cluster(
        self, 
        card_embeddings: torch.Tensor,
        merchant_embeddings: torch.Tensor,
        method: str = 'kmeans',
        n_clusters: int = 50,
        **kwargs
    ) -> Dict[int, List[str]]:
        """
        基于GNN嵌入的社群检测
        
        Args:
            card_embeddings: Card节点嵌入
            merchant_embeddings: Merchant节点嵌入
            method: 聚类方法 ('kmeans', 'dbscan')
            n_clusters: 聚类数（kmeans专用）
            
        Returns:
            社群字典
        """
        logger.info(f"Running embedding-based clustering ({method})...")
        
        # 转换为numpy
        if isinstance(card_embeddings, torch.Tensor):
            card_embeddings = card_embeddings.cpu().numpy()
        if isinstance(merchant_embeddings, torch.Tensor):
            merchant_embeddings = merchant_embeddings.cpu().numpy()
        
        # 合并嵌入
        all_embeddings = np.vstack([card_embeddings, merchant_embeddings])
        
        # 标准化
        scaler = StandardScaler()
        all_embeddings = scaler.fit_transform(all_embeddings)
        
        # 聚类
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(all_embeddings)
        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(all_embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # 整理社群
        n_cards = card_embeddings.shape[0]
        communities = defaultdict(list)
        
        for i, label in enumerate(labels):
            if label == -1:  # DBSCAN噪声点
                continue
            if i < n_cards:
                communities[label].append(f"card_{i}")
            else:
                communities[label].append(f"merchant_{i - n_cards}")
        
        self.communities = dict(communities)
        logger.info(f"Embedding clustering detected {len(self.communities)} communities")
        
        return self.communities

    def compute_community_stats(
        self, 
        edge_labels: torch.Tensor,
        edge_index: torch.Tensor,
        edge_times: Optional[torch.Tensor] = None,
        edge_amounts: Optional[torch.Tensor] = None
    ) -> Dict[int, Dict]:
        """
        计算每个社群的统计特征
        
        Args:
            edge_labels: 边的欺诈标签
            edge_index: 边索引
            edge_times: 交易时间（可选）
            edge_amounts: 交易金额（可选）
            
        Returns:
            社群统计字典
        """
        logger.info("Computing community statistics...")
        
        # 转换为numpy
        if isinstance(edge_labels, torch.Tensor):
            edge_labels = edge_labels.numpy()
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()
        if edge_times is not None and isinstance(edge_times, torch.Tensor):
            edge_times = edge_times.numpy()
        if edge_amounts is not None and isinstance(edge_amounts, torch.Tensor):
            edge_amounts = edge_amounts.numpy()
        
        # 构建节点到社群的映射
        node_to_community = {}
        for comm_id, nodes in self.communities.items():
            for node in nodes:
                node_to_community[node] = comm_id
        
        # 统计每个社群
        community_data = defaultdict(lambda: {
            'total': 0, 
            'fraud': 0, 
            'cards': set(), 
            'merchants': set(),
            'times': [],
            'amounts': []
        })
        
        for i in range(edge_index.shape[1]):
            card_node = f"card_{edge_index[0, i]}"
            merchant_node = f"merchant_{edge_index[1, i]}"
            
            # 找到边所属的社群
            card_comm = node_to_community.get(card_node)
            merchant_comm = node_to_community.get(merchant_node)
            
            # 如果卡和商户在同一社群
            if card_comm is not None and card_comm == merchant_comm:
                community_data[card_comm]['total'] += 1
                community_data[card_comm]['fraud'] += edge_labels[i]
                community_data[card_comm]['cards'].add(card_node)
                community_data[card_comm]['merchants'].add(merchant_node)
                
                if edge_times is not None:
                    community_data[card_comm]['times'].append(edge_times[i])
                if edge_amounts is not None:
                    community_data[card_comm]['amounts'].append(edge_amounts[i])
        
        # 计算统计指标
        self.community_stats = {}
        for comm_id, data in community_data.items():
            if data['total'] > 0:
                stats = {
                    'total_transactions': data['total'],
                    'fraud_transactions': data['fraud'],
                    'fraud_rate': data['fraud'] / data['total'],
                    'num_cards': len(data['cards']),
                    'num_merchants': len(data['merchants']),
                }
                
                # 计算密度（实际边数 / 可能边数）
                possible_edges = stats['num_cards'] * stats['num_merchants']
                stats['density'] = data['total'] / possible_edges if possible_edges > 0 else 0
                
                # 时间集中度（标准差越小，集中度越高）
                if data['times']:
                    time_std = np.std(data['times'])
                    time_range = np.max(data['times']) - np.min(data['times']) + 1
                    stats['time_concentration'] = 1 - (time_std / time_range) if time_range > 0 else 1
                    
                    # 夜间交易比例（假设时间是小时）
                    hours = np.array(data['times']) % 24
                    stats['night_ratio'] = np.mean((hours >= 0) & (hours < 6))
                else:
                    stats['time_concentration'] = 0
                    stats['night_ratio'] = 0
                
                # 金额统计
                if data['amounts']:
                    stats['amt_mean'] = np.mean(data['amounts'])
                    stats['amt_std'] = np.std(data['amounts'])
                    # 整数金额比例
                    stats['round_amt_ratio'] = np.mean(
                        np.abs(np.array(data['amounts']) - np.round(data['amounts'])) < 0.01
                    )
                else:
                    stats['amt_mean'] = 0
                    stats['amt_std'] = 0
                    stats['round_amt_ratio'] = 0
                
                self.community_stats[comm_id] = stats
        
        logger.info(f"Computed stats for {len(self.community_stats)} communities")
        return self.community_stats
    
    def compute_topology_anomaly(self, comm_id: int) -> float:
        """
        计算拓扑异常分数
        基于密度：密集子图更可疑
        
        Args:
            comm_id: 社群ID
            
        Returns:
            拓扑异常分数 [0, 1]
        """
        if comm_id not in self.community_stats:
            return 0.0
        
        stats = self.community_stats[comm_id]
        
        # 密度分数
        density_score = min(stats['density'], 1.0)
        
        # 多卡少商户模式（套现特征）
        if stats['num_merchants'] > 0:
            card_merchant_ratio = stats['num_cards'] / stats['num_merchants']
            ratio_score = min(card_merchant_ratio / 10, 1.0)  # 10:1以上得满分
        else:
            ratio_score = 0
        
        # 综合拓扑异常分数
        return 0.6 * density_score + 0.4 * ratio_score
    
    def compute_behavior_anomaly(self, comm_id: int) -> float:
        """
        计算行为异常分数
        基于时间、金额等行为模式
        
        Args:
            comm_id: 社群ID
            
        Returns:
            行为异常分数 [0, 1]
        """
        if comm_id not in self.community_stats:
            return 0.0
        
        stats = self.community_stats[comm_id]
        
        # 时间集中度分数
        time_score = stats.get('time_concentration', 0)
        
        # 夜间交易分数
        night_score = stats.get('night_ratio', 0) * 2  # 夜间交易权重加倍
        night_score = min(night_score, 1.0)
        
        # 整数金额分数
        round_amt_score = stats.get('round_amt_ratio', 0)
        
        # 综合行为异常分数
        return 0.4 * time_score + 0.3 * night_score + 0.3 * round_amt_score
    
    def compute_risk_score(self, comm_id: int) -> float:
        """
        计算综合风险评分
        融合拓扑异常和行为异常
        
        Args:
            comm_id: 社群ID
            
        Returns:
            风险评分 [0, 1]
        """
        if comm_id not in self.community_stats:
            return 0.0
        
        stats = self.community_stats[comm_id]
        
        # 欺诈率分数
        fraud_score = min(stats['fraud_rate'] * 5, 1.0)  # 20%欺诈率得满分
        
        # 拓扑异常分数
        topology_score = self.compute_topology_anomaly(comm_id)
        
        # 行为异常分数
        behavior_score = self.compute_behavior_anomaly(comm_id)
        
        # 交易量加权（交易量大的社群更重要）
        volume_weight = min(np.log1p(stats['total_transactions']) / 10, 1.0)
        
        # 综合风险评分
        base_score = (
            0.4 * fraud_score + 
            0.3 * topology_score + 
            0.3 * behavior_score
        )
        
        return base_score * (0.5 + 0.5 * volume_weight)

    def identify_suspicious_communities(
        self,
        fraud_rate_threshold: float = 0.1,
        min_transactions: int = 10,
        min_cards: int = 3,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        识别可疑欺诈社群
        
        Args:
            fraud_rate_threshold: 欺诈率阈值
            min_transactions: 最小交易数
            min_cards: 最小卡数
            top_k: 返回前k个（可选）
            
        Returns:
            可疑社群列表
        """
        logger.info("Identifying suspicious communities...")
        
        suspicious = []
        for comm_id, stats in self.community_stats.items():
            # 筛选条件
            if (stats['fraud_rate'] >= fraud_rate_threshold and
                stats['total_transactions'] >= min_transactions and
                stats['num_cards'] >= min_cards):
                
                risk_score = self.compute_risk_score(comm_id)
                
                suspicious.append({
                    'community_id': comm_id,
                    'nodes': self.communities[comm_id],
                    'risk_score': risk_score,
                    'topology_anomaly': self.compute_topology_anomaly(comm_id),
                    'behavior_anomaly': self.compute_behavior_anomaly(comm_id),
                    **stats
                })
        
        # 按风险分数排序
        suspicious.sort(key=lambda x: x['risk_score'], reverse=True)
        
        if top_k:
            suspicious = suspicious[:top_k]
        
        self.suspicious_communities = suspicious
        logger.info(f"Identified {len(suspicious)} suspicious communities")
        
        return suspicious
    
    def evaluate_community_detection(
        self, 
        edge_labels: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Dict:
        """
        评估社群级别的检测效果
        
        Args:
            edge_labels: 边的欺诈标签
            edge_index: 边索引
            
        Returns:
            评估指标
        """
        if isinstance(edge_labels, torch.Tensor):
            edge_labels = edge_labels.numpy()
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()
        
        # 总欺诈交易数
        total_fraud = edge_labels.sum()
        total_transactions = len(edge_labels)
        
        # 可疑社群覆盖的欺诈交易
        fraud_in_suspicious = sum(c['fraud_transactions'] for c in self.suspicious_communities)
        tx_in_suspicious = sum(c['total_transactions'] for c in self.suspicious_communities)
        
        # 社群级别召回率：可疑社群覆盖了多少欺诈交易
        community_recall = fraud_in_suspicious / total_fraud if total_fraud > 0 else 0
        
        # 社群级别精确率：可疑社群中有多少是真实欺诈
        community_precision = fraud_in_suspicious / tx_in_suspicious if tx_in_suspicious > 0 else 0
        
        # F1
        if community_precision + community_recall > 0:
            community_f1 = 2 * community_precision * community_recall / (community_precision + community_recall)
        else:
            community_f1 = 0
        
        metrics = {
            'total_fraud': int(total_fraud),
            'total_transactions': total_transactions,
            'suspicious_communities': len(self.suspicious_communities),
            'fraud_in_suspicious': int(fraud_in_suspicious),
            'tx_in_suspicious': tx_in_suspicious,
            'community_recall': community_recall,
            'community_precision': community_precision,
            'community_f1': community_f1
        }
        
        logger.info("=" * 50)
        logger.info("Community Detection Evaluation:")
        logger.info(f"  Total fraud transactions: {total_fraud}")
        logger.info(f"  Suspicious communities: {len(self.suspicious_communities)}")
        logger.info(f"  Fraud covered: {fraud_in_suspicious} ({community_recall*100:.1f}%)")
        logger.info(f"  Community Precision: {community_precision*100:.1f}%")
        logger.info(f"  Community Recall: {community_recall*100:.1f}%")
        logger.info(f"  Community F1: {community_f1*100:.1f}%")
        logger.info("=" * 50)
        
        return metrics
    
    def get_community_subgraph(self, community_id: int) -> nx.Graph:
        """
        获取指定社群的子图
        
        Args:
            community_id: 社群ID
            
        Returns:
            子图
        """
        if community_id not in self.communities:
            raise ValueError(f"Community {community_id} not found")
        
        nodes = self.communities[community_id]
        return self.G.subgraph(nodes).copy()
    
    def export_results(self, output_path: str):
        """
        导出社群检测结果（JSON格式，可导入Neo4j）
        
        Args:
            output_path: 输出路径
        """
        import json
        import os
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 准备导出数据
        export_data = {
            'communities': {str(k): v for k, v in self.communities.items()},
            'stats': {str(k): v for k, v in self.community_stats.items()},
            'suspicious': self.suspicious_communities
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Results exported to {output_path}")
    
    def print_top_suspicious(self, top_k: int = 10):
        """打印TOP可疑社群"""
        logger.info(f"\nTOP {top_k} Suspicious Communities:")
        logger.info("-" * 60)
        
        for i, comm in enumerate(self.suspicious_communities[:top_k], 1):
            logger.info(f"{i}. Community {comm['community_id']}")
            logger.info(f"   Risk Score: {comm['risk_score']:.3f}")
            logger.info(f"   Fraud Rate: {comm['fraud_rate']*100:.1f}%")
            logger.info(f"   Transactions: {comm['total_transactions']}")
            logger.info(f"   Cards: {comm['num_cards']}, Merchants: {comm['num_merchants']}")
            logger.info(f"   Topology Anomaly: {comm['topology_anomaly']:.3f}")
            logger.info(f"   Behavior Anomaly: {comm['behavior_anomaly']:.3f}")
            logger.info("")
