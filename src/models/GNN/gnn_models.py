"""
图神经网络模型模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional


class HeteroGraphSAGE(nn.Module):
    """异构图GraphSAGE模型"""
    
    def __init__(
        self,
        card_in_channels: int,
        merchant_in_channels: int,
        edge_in_channels: int = 0,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_in_channels = edge_in_channels
        
        # 输入投影层
        self.card_lin = Linear(card_in_channels, hidden_channels)
        self.merchant_lin = Linear(merchant_in_channels, hidden_channels)
        
        # 异构图卷积层（标准SAGEConv）
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            
            conv = HeteroConv({
                ('card', 'transacts', 'merchant'): SAGEConv(in_ch, out_ch),
                ('merchant', 'rev_transacts', 'card'): SAGEConv(in_ch, out_ch),
            }, aggr='mean')
            self.convs.append(conv)
        
        # 边特征编码器
        if edge_in_channels > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
            classifier_in = out_channels * 3  # card + merchant + edge
        else:
            self.edge_encoder = None
            classifier_in = out_channels * 2
        
        # 边分类头
        self.edge_classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)
        )
    
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        edge_index_dict: Dict[Tuple, torch.Tensor],
        edge_attr_dict: Dict[Tuple, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        x_dict = {
            'card': F.relu(self.card_lin(x_dict['card'])),
            'merchant': F.relu(self.merchant_lin(x_dict['merchant']))
        }
        
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < self.num_layers - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                         for key, x in x_dict.items()}
        
        return x_dict
    
    def predict_edge(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """预测边的欺诈概率"""
        card_emb = x_dict['card'][edge_index[0]]
        merchant_emb = x_dict['merchant'][edge_index[1]]
        
        if self.edge_encoder is not None and edge_attr is not None:
            edge_feat = self.edge_encoder(edge_attr)
            edge_emb = torch.cat([card_emb, merchant_emb, edge_feat], dim=1)
        else:
            edge_emb = torch.cat([card_emb, merchant_emb], dim=1)
        
        return self.edge_classifier(edge_emb)
    
    def get_embeddings(self, data: HeteroData, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """获取所有节点的嵌入表示"""
        self.eval()
        with torch.no_grad():
            x_dict = {
                'card': data['card'].x.to(device),
                'merchant': data['merchant'].x.to(device)
            }
            edge_index_dict = {
                ('card', 'transacts', 'merchant'): data['card', 'transacts', 'merchant'].edge_index.to(device),
                ('merchant', 'rev_transacts', 'card'): data['merchant', 'rev_transacts', 'card'].edge_index.to(device),
            }
            return self(x_dict, edge_index_dict)


class HeteroGAT(nn.Module):
    """异构图GAT模型"""
    
    def __init__(
        self,
        card_in_channels: int,
        merchant_in_channels: int,
        edge_in_channels: int = 0,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_in_channels = edge_in_channels
        
        # 输入投影层
        self.card_lin = Linear(card_in_channels, hidden_channels)
        self.merchant_lin = Linear(merchant_in_channels, hidden_channels)
        
        # GAT卷积层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels if i == 0 else hidden_channels * heads
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            h = heads if i < num_layers - 1 else 1
            concat = i < num_layers - 1
            
            conv = HeteroConv({
                ('card', 'transacts', 'merchant'): GATConv(
                    in_ch, out_ch, heads=h, concat=concat, dropout=dropout, add_self_loops=False
                ),
                ('merchant', 'rev_transacts', 'card'): GATConv(
                    in_ch, out_ch, heads=h, concat=concat, dropout=dropout, add_self_loops=False
                ),
            }, aggr='mean')
            self.convs.append(conv)
        
        # 边特征编码器
        if edge_in_channels > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
            classifier_in = out_channels * 3
        else:
            self.edge_encoder = None
            classifier_in = out_channels * 2
        
        # 边分类头
        self.edge_classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)
        )
    
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        edge_index_dict: Dict[Tuple, torch.Tensor],
        edge_attr_dict: Dict[Tuple, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        x_dict = {
            'card': F.relu(self.card_lin(x_dict['card'])),
            'merchant': F.relu(self.merchant_lin(x_dict['merchant']))
        }
        
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < self.num_layers - 1:
                x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        
        return x_dict
    
    def predict_edge(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """预测边的欺诈概率"""
        card_emb = x_dict['card'][edge_index[0]]
        merchant_emb = x_dict['merchant'][edge_index[1]]
        
        if self.edge_encoder is not None and edge_attr is not None:
            edge_feat = self.edge_encoder(edge_attr)
            edge_emb = torch.cat([card_emb, merchant_emb, edge_feat], dim=1)
        else:
            edge_emb = torch.cat([card_emb, merchant_emb], dim=1)
        
        return self.edge_classifier(edge_emb)
    
    def get_embeddings(self, data: HeteroData, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """获取所有节点的嵌入表示"""
        self.eval()
        with torch.no_grad():
            x_dict = {
                'card': data['card'].x.to(device),
                'merchant': data['merchant'].x.to(device)
            }
            edge_index_dict = {
                ('card', 'transacts', 'merchant'): data['card', 'transacts', 'merchant'].edge_index.to(device),
                ('merchant', 'rev_transacts', 'card'): data['merchant', 'rev_transacts', 'card'].edge_index.to(device),
            }
            return self(x_dict, edge_index_dict)


class GraphAutoEncoder(nn.Module):
    """图自编码器 - 用于无监督异常检测"""
    
    def __init__(self, encoder: nn.Module, hidden_channels: int = 64):
        super().__init__()
        self.encoder = encoder
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x_dict, edge_index_dict, edge_attr_dict=None):
        return self.encoder(x_dict, edge_index_dict, edge_attr_dict)
    
    def decode(self, z_dict, edge_index):
        card_emb = z_dict['card'][edge_index[0]]
        merchant_emb = z_dict['merchant'][edge_index[1]]
        edge_emb = torch.cat([card_emb, merchant_emb], dim=1)
        return self.decoder(edge_emb).squeeze()
    
    def forward(self, x_dict, edge_index_dict, pos_edge_index, neg_edge_index=None, edge_attr_dict=None):
        z_dict = self.encode(x_dict, edge_index_dict, edge_attr_dict)
        pos_pred = self.decode(z_dict, pos_edge_index)
        
        if neg_edge_index is not None:
            neg_pred = self.decode(z_dict, neg_edge_index)
            return pos_pred, neg_pred, z_dict
        return pos_pred, z_dict
    
    def reconstruction_loss(self, pos_pred, neg_pred):
        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        return pos_loss + neg_loss
    
    def anomaly_score(self, z_dict, edge_index):
        pred = self.decode(z_dict, edge_index)
        return 1 - pred


def create_model(
    model_type: str,
    card_in_channels: int,
    merchant_in_channels: int,
    edge_in_channels: int = 0,
    hidden_channels: int = 128,
    out_channels: int = 64,
    num_layers: int = 2,
    **kwargs
) -> nn.Module:
    """
    创建GNN模型的工厂函数
    
    Args:
        model_type: 模型类型 ('graphsage', 'gat', 'autoencoder')
        card_in_channels: Card特征维度
        merchant_in_channels: Merchant特征维度
        edge_in_channels: 边特征维度
        hidden_channels: 隐藏层维度
        out_channels: 输出维度
        num_layers: 层数
    """
    if model_type.lower() == 'graphsage':
        return HeteroGraphSAGE(
            card_in_channels=card_in_channels,
            merchant_in_channels=merchant_in_channels,
            edge_in_channels=edge_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_type.lower() == 'gat':
        return HeteroGAT(
            card_in_channels=card_in_channels,
            merchant_in_channels=merchant_in_channels,
            edge_in_channels=edge_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=kwargs.get('heads', 4),
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_type.lower() == 'autoencoder':
        encoder = HeteroGraphSAGE(
            card_in_channels=card_in_channels,
            merchant_in_channels=merchant_in_channels,
            edge_in_channels=edge_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=kwargs.get('dropout', 0.3)
        )
        return GraphAutoEncoder(encoder, out_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
