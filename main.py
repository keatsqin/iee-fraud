"""
欺诈检测GNN系统主程序
"""
import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from loguru import logger

# 添加src到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processing import PyGGraphBuilder
from src.models import create_model, CommunityDetector
from src.training import FraudDetectionTrainer

# 尝试导入混合模型
try:
    from src.models import HybridFraudDetector, train_hybrid_model
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False


def setup_logging(log_dir: str):
    """配置日志"""
    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        os.path.join(log_dir, "pipeline_{time}.log"),
        rotation="10 MB",
        level="INFO"
    )


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description='信用卡欺诈检测GNN系统')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='../ieee-fraud-detection', help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--model', type=str, default='graphsage', choices=['graphsage', 'gat'], help='GNN模型类型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--device', type=str, default='auto', help='设备 (auto/cpu/cuda)')
    parser.add_argument('--skip_spark', action='store_true', help='跳过Spark处理')
    parser.add_argument('--skip_training', action='store_true', help='跳过模型训练')
    parser.add_argument('--use_hybrid', action='store_true', help='使用GNN+LightGBM混合模型')
    return parser.parse_args()


def run_spark_processing(args, config) -> str:
    """Phase 1: Spark数据处理"""
    logger.info("=" * 60)
    logger.info("Phase 1: Spark数据处理")
    logger.info("=" * 60)
    
    processed_dir = os.path.join(args.output_dir, 'processed')
    
    # 检查是否已处理
    if args.skip_spark:
        # 检查data/processed目录
        alt_processed_dir = os.path.join(project_root, 'data', 'processed')
        if os.path.exists(os.path.join(alt_processed_dir, 'edges.csv')) or \
           os.path.exists(os.path.join(alt_processed_dir, 'edges.parquet')):
            logger.info(f"Using existing processed data from {alt_processed_dir}")
            return alt_processed_dir
        if os.path.exists(os.path.join(processed_dir, 'edges.csv')) or \
           os.path.exists(os.path.join(processed_dir, 'edges.parquet')):
            logger.info(f"Using existing processed data from {processed_dir}")
            return processed_dir
        logger.error("No processed data found! Run without --skip_spark first.")
        sys.exit(1)
    
    # 动态导入SparkDataProcessor
    try:
        from src.data_processing import SparkDataProcessor
    except ImportError:
        logger.error("PySpark not installed! Install it or use --skip_spark with pre-processed data.")
        sys.exit(1)
    
    # 数据路径
    transaction_path = os.path.join(args.data_dir, 'train_transaction.csv')
    identity_path = os.path.join(args.data_dir, 'train_identity.csv')
    
    if not os.path.exists(transaction_path):
        logger.error(f"Transaction file not found: {transaction_path}")
        sys.exit(1)
    
    # 运行Spark处理
    processor = SparkDataProcessor(config)
    try:
        processor.run_pipeline(transaction_path, identity_path, processed_dir)
    finally:
        processor.stop()
    
    return processed_dir


def run_graph_building(processed_dir: str, args) -> tuple:
    """Phase 2: 图构建"""
    logger.info("=" * 60)
    logger.info("Phase 2: PyG图构建")
    logger.info("=" * 60)
    
    graph_path = os.path.join(args.output_dir, 'graph', 'fraud_graph.pt')
    
    # 检查是否已构建
    if os.path.exists(graph_path):
        logger.info(f"Loading existing graph from {graph_path}")
        builder = PyGGraphBuilder()
        data = builder.load_graph(graph_path)
        
        # 加载edges_df用于时间划分（支持CSV和Parquet）
        import pandas as pd
        edges_csv = os.path.join(processed_dir, 'edges.csv')
        edges_parquet = os.path.join(processed_dir, 'edges.parquet')
        if os.path.exists(edges_csv) and os.path.isfile(edges_csv):
            edges_df = pd.read_csv(edges_csv)
        elif os.path.exists(edges_parquet):
            # parquet可能是目录（Spark输出）或文件
            edges_df = pd.read_parquet(edges_parquet)
        else:
            raise FileNotFoundError(f"Edges file not found in {processed_dir}")
        
        if not hasattr(data, 'train_mask'):
            train_mask, val_mask, test_mask = builder.split_edges_by_time(data, edges_df)
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
            builder.save_graph(data, graph_path)
        
        return data, edges_df
    
    # 构建新图
    builder = PyGGraphBuilder()
    card_df, merchant_df, edges_df = builder.load_from_parquet(processed_dir)
    data = builder.build_hetero_graph(card_df, merchant_df, edges_df)
    
    # 按时间划分
    train_mask, val_mask, test_mask = builder.split_edges_by_time(data, edges_df)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # 保存
    builder.save_graph(data, graph_path)
    
    return data, edges_df


def run_model_training(data, args, config) -> tuple:
    """Phase 3: GNN模型训练"""
    logger.info("=" * 60)
    logger.info("Phase 3: GNN模型训练")
    logger.info("=" * 60)
    
    model_path = os.path.join(args.output_dir, 'models', f'{args.model}_fraud_detector.pt')
    embeddings_path = os.path.join(args.output_dir, 'embeddings', 'node_embeddings.pt')
    
    # 获取特征维度
    card_in_dim = data['card'].x.shape[1]
    merchant_in_dim = data['merchant'].x.shape[1]
    
    # 获取边特征维度
    edge_in_dim = 0
    if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
        edge_in_dim = data['card', 'transacts', 'merchant'].edge_attr.shape[1]
        logger.info(f"Edge features: {edge_in_dim} dimensions")
    
    # 模型配置
    model_config = config.get('model', {})
    hidden_dim = model_config.get('hidden_dim', 128)
    embed_dim = model_config.get('embed_dim', 64)
    num_layers = model_config.get('num_layers', 2)
    
    # 创建模型
    model = create_model(
        model_type=args.model,
        card_in_channels=card_in_dim,
        merchant_in_channels=merchant_in_dim,
        edge_in_channels=edge_in_dim,
        hidden_channels=hidden_dim,
        out_channels=embed_dim,
        num_layers=num_layers,
        dropout=model_config.get('dropout', 0.3),
        heads=model_config.get('heads', 4)
    )
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Card features: {card_in_dim}, Merchant features: {merchant_in_dim}, Edge features: {edge_in_dim}")
    logger.info(f"Hidden dim: {hidden_dim}, Embed dim: {embed_dim}, Layers: {num_layers}")
    
    # 检查是否跳过训练
    if args.skip_training and os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        trainer = FraudDetectionTrainer(model, device=args.device)
        trainer.load_model(model_path)
    else:
        # 训练配置
        train_config = config.get('training', {})
        
        # 创建训练器
        trainer = FraudDetectionTrainer(
            model,
            device=args.device,
            lr=float(train_config.get('lr', 0.001)),
            weight_decay=float(train_config.get('weight_decay', 1e-4)),
            fraud_weight=float(train_config.get('fraud_weight', 15.0)),
            use_focal_loss=train_config.get('use_focal_loss', True),
            focal_gamma=float(train_config.get('focal_gamma', 2.0))
        )
        
        # 训练
        train_results = trainer.train(
            data,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            num_epochs=args.epochs,
            early_stopping_patience=train_config.get('early_stopping_patience', 10),
            save_best=True,
            save_path=model_path
        )
        
        # 测试
        test_results = trainer.test(data, data.test_mask)
        
        # 保存GNN结果
        import json
        gnn_results_path = model_path.replace('.pt', '_results.json')
        gnn_results_json = {
            'test': {
                'auc': float(test_results['auc']),
                'ap': float(test_results['ap']),
                'precision': float(test_results['precision']),
                'recall': float(test_results['recall']),
                'f1': float(test_results['f1']),
                'confusion_matrix': test_results['confusion_matrix'].tolist()
            },
            'train': {
                'best_val_auc': float(train_results.get('best_val_auc', 0)),
                'history': {
                    'train_loss': [float(x) for x in trainer.history.get('train_loss', [])],
                    'val_auc': [float(x) for x in trainer.history.get('val_auc', [])],
                    'val_ap': [float(x) for x in trainer.history.get('val_ap', [])],
                    'val_f1': [float(x) for x in trainer.history.get('val_f1', [])]
                }
            }
        }
        with open(gnn_results_path, 'w', encoding='utf-8') as f:
            json.dump(gnn_results_json, f, indent=2, ensure_ascii=False)
        logger.info(f"GNN results saved to {gnn_results_path}")
    
    # 获取嵌入
    embeddings = trainer.get_embeddings(data)
    
    # 保存嵌入
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    torch.save({
        'card': embeddings['card'].cpu(),
        'merchant': embeddings['merchant'].cpu()
    }, embeddings_path)
    logger.info(f"Embeddings saved to {embeddings_path}")
    
    return trainer, embeddings


def run_hybrid_training(data, trainer, args) -> tuple:
    """Phase 3.5: GNN + LightGBM 混合模型训练"""
    if not HAS_HYBRID:
        logger.warning("LightGBM not installed, skipping hybrid model")
        return None, None
    
    logger.info("=" * 60)
    logger.info("Phase 3.5: GNN + LightGBM 混合模型训练")
    logger.info("=" * 60)
    
    hybrid_path = os.path.join(args.output_dir, 'models', 'hybrid_gnn_lgb.pkl')
    
    hybrid_model, results = train_hybrid_model(
        data,
        trainer,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        save_path=hybrid_path
    )
    
    return hybrid_model, results


def run_community_detection(data, embeddings, edges_df, args, config) -> CommunityDetector:
    """Phase 4: 社群检测"""
    logger.info("=" * 60)
    logger.info("Phase 4: 社群检测与欺诈团伙识别")
    logger.info("=" * 60)
    
    detector = CommunityDetector()
    
    # 构建NetworkX图
    edge_index = data['card', 'transacts', 'merchant'].edge_index
    detector.build_networkx_graph(
        edge_index, 
        data.card_mapping, 
        data.merchant_mapping
    )
    
    # Louvain社群检测
    comm_config = config.get('community', {})
    try:
        detector.detect_louvain(resolution=comm_config.get('louvain_resolution', 1.0))
    except ImportError:
        logger.warning("Louvain not available, using embedding clustering only")
        detector.detect_embedding_cluster(
            embeddings['card'],
            embeddings['merchant'],
            method='kmeans',
            n_clusters=comm_config.get('kmeans_clusters', 50)
        )
    
    # 计算社群统计
    edge_labels = data['card', 'transacts', 'merchant'].edge_label
    
    # 获取时间和金额（如果有）
    edge_times = None
    edge_amounts = None
    if 'hour' in edges_df.columns:
        edge_times = torch.tensor(edges_df['hour'].values)
    if 'TransactionAmt' in edges_df.columns:
        edge_amounts = torch.tensor(edges_df['TransactionAmt'].values)
    
    detector.compute_community_stats(
        edge_labels, 
        edge_index,
        edge_times=edge_times,
        edge_amounts=edge_amounts
    )
    
    # 打印TOP可疑社群
    logger.info("All community fraud rates:")
    for comm_id, stats in sorted(detector.community_stats.items(), 
                                   key=lambda x: x[1]['fraud_rate'], reverse=True)[:15]:
        logger.info(f"  Community {comm_id}: fraud_rate={stats['fraud_rate']*100:.2f}%, "
                   f"tx={stats['total_transactions']}, cards={stats['num_cards']}")
    
    fraud_threshold = 0.02
    min_tx = 100
    min_cards_threshold = 10
    
    logger.info(f"Filtering: fraud_rate>={fraud_threshold}, tx>={min_tx}, cards>={min_cards_threshold}")
    
    suspicious_list = []
    for comm_id, stats in detector.community_stats.items():
        if (stats['fraud_rate'] >= fraud_threshold and 
            stats['total_transactions'] >= min_tx and 
            stats['num_cards'] >= min_cards_threshold):
            suspicious_list.append((comm_id, stats))
    
    logger.info(f"Found {len(suspicious_list)} suspicious communities")
    
    detector.identify_suspicious_communities(
        fraud_rate_threshold=fraud_threshold,
        min_transactions=min_tx,
        min_cards=min_cards_threshold
    )
    
    # 评估
    detector.evaluate_community_detection(edge_labels, edge_index)
    
    # 打印TOP可疑社群
    detector.print_top_suspicious(top_k=10)
    
    # 导出结果
    results_path = os.path.join(args.output_dir, 'communities', 'suspicious_communities.json')
    detector.export_results(results_path)
    
    return detector


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置日志
    setup_logging(os.path.join(args.output_dir, 'logs'))
    
    # 加载配置
    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)
    
    logger.info("=" * 60)
    logger.info("信用卡欺诈检测GNN系统")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Phase 1: Spark数据处理
        processed_dir = run_spark_processing(args, config)
        
        # Phase 2: 图构建
        data, edges_df = run_graph_building(processed_dir, args)
        
        # Phase 3: 模型训练
        trainer, embeddings = run_model_training(data, args, config)
        
        # Phase 3.5: 混合模型（可选）
        hybrid_model = None
        if args.use_hybrid:
            hybrid_model, hybrid_results = run_hybrid_training(data, trainer, args)
        
        # Phase 4: 社群检测
        detector = run_community_detection(data, embeddings, edges_df, args, config)
        
        # 完成
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info(f"输出文件:")
        logger.info(f"  - 处理数据: {processed_dir}/")
        logger.info(f"  - 图数据: {args.output_dir}/graph/fraud_graph.pt")
        logger.info(f"  - GNN模型: {args.output_dir}/models/{args.model}_fraud_detector.pt")
        if hybrid_model:
            logger.info(f"  - 混合模型: {args.output_dir}/models/hybrid_gnn_lgb.pkl")
        logger.info(f"  - 嵌入: {args.output_dir}/embeddings/node_embeddings.pt")
        logger.info(f"  - 社群: {args.output_dir}/communities/suspicious_communities.json")
        logger.info("")
        logger.info("下一步: 运行可视化平台")
        logger.info("  python src/visualization/app.py")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
