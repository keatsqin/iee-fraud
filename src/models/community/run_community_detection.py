"""
社群检测完整运行脚本
从已有的 fraud_graph.pt 加载数据并执行社群检测
运行方式（从项目根目录）:
    python -m src.models.run_community_detection
或直接运行:
    python src/models/run_community_detection.py
"""
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from loguru import logger

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.community_detector import CommunityDetector
from src.data_processing import PyGGraphBuilder

# ── 路径配置 ──────────────────────────────────────────────────────────────────
GRAPH_PATH  = PROJECT_ROOT / "outputs" / "graph" / "fraud_graph.pt"
OUTPUT_DIR  = PROJECT_ROOT / "outputs" / "communities"


def convert_to_serializable(obj):
    """
    递归转换 numpy/torch 类型为 Python 原生类型，用于 JSON 序列化
    
    Args:
        obj: 任意类型的对象
    
    Returns:
        可 JSON 序列化的 Python 对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        # 处理 PyTorch 张量
        if obj.numel() == 1:
            return float(obj) if obj.is_floating_point() else int(obj)
        else:
            return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj


def load_graph_data(graph_path: Path):
    """
    从 fraud_graph.pt 加载图数据，提取社群检测所需的张量。

    Returns:
        dict with keys: edge_index, edge_labels, timestamps, amounts,
                        card_mapping, merchant_mapping
    """
    logger.info(f"Loading graph from {graph_path}")
    builder = PyGGraphBuilder()
    data = builder.load_graph(str(graph_path))
    logger.info(f"Graph: {data}")

    edge_index = data['card', 'transacts', 'merchant'].edge_index  # [2, E]
    edge_label  = data['card', 'transacts', 'merchant'].edge_label  # [E]

    num_cards     = data['card'].x.shape[0]
    num_merchants = data['merchant'].x.shape[0]

    # 简单的恒等映射：card_id == node_index
    card_mapping     = {i: i for i in range(num_cards)}
    merchant_mapping = {i: i for i in range(num_merchants)}

    # 尝试从 edge_attr 提取时间戳和金额（列顺序依赖图构建时的特征顺序）
    # 如果无法确定，退回到安全默认值
    timestamps = None
    amounts    = None
    if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
        edge_attr = data['card', 'transacts', 'merchant'].edge_attr  # [E, F]
        logger.info(f"edge_attr shape: {edge_attr.shape}")
        # 第 0 列通常是 TransactionDT（时间），第 1 列是 TransactionAmt（金额）
        if edge_attr.shape[1] >= 2:
            timestamps = edge_attr[:, 0].float()
            amounts    = edge_attr[:, 1].float()
        elif edge_attr.shape[1] == 1:
            amounts = edge_attr[:, 0].float()

    if timestamps is None:
        timestamps = torch.arange(edge_index.shape[1], dtype=torch.float32)
        logger.warning("No timestamp column found, using edge index as fallback")
    if amounts is None:
        amounts = torch.ones(edge_index.shape[1], dtype=torch.float32)
        logger.warning("No amount column found, using 1.0 as fallback")

    # 标准化时间戳为小时
    if timestamps.max() > 1000:
        timestamps = timestamps / 3600

    # 金额对数变换
    amounts = torch.log1p(amounts.clamp(min=0))

    logger.info(f"Edges: {edge_index.shape[1]}, Fraud rate: {edge_label.float().mean().item():.4f}")

    return {
        'edge_index':       edge_index,
        'edge_labels':      edge_label.float(),
        'timestamps':       timestamps,
        'amounts':          amounts,
        'card_mapping':     card_mapping,
        'merchant_mapping': merchant_mapping,
    }


def run_community_detection(data: dict, output_dir: Path):
    """运行社群检测完整流程"""
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = CommunityDetector()

    # ── Step 1: 构建 NetworkX 图 ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Building NetworkX graph")
    logger.info("=" * 60)
    detector.build_networkx_graph(
        edge_index=data['edge_index'],
        card_mapping=data['card_mapping'],
        merchant_mapping=data['merchant_mapping'],
        edge_attrs={
            'times':  data['timestamps'],
            'amounts': data['amounts'],
            'labels':  data['edge_labels'],
        }
    )

    # ── Step 2: Louvain 社群检测 ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Community detection (Louvain)")
    logger.info("=" * 60)
    communities = detector.detect_louvain(resolution=1.0)
    logger.info(f"Detected {len(communities)} communities")

    # ── Step 3: 计算社群统计 ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Computing community statistics")
    logger.info("=" * 60)
    detector.compute_community_stats(
        edge_labels=data['edge_labels'],
        edge_index=data['edge_index'],
        edge_times=data['timestamps'],
        edge_amounts=data['amounts'],
    )

    # ── Step 4: 识别可疑社群（改为风险排序）──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Ranking communities by risk score")
    logger.info("=" * 60)
    
    # 获取所有社群的风险排序
    all_ranked = detector.identify_suspicious_communities(top_k=None)  # 返回所有
    
    # 输出前10高风险社群
    logger.info("Top 10 highest risk communities:")
    for i, comm in enumerate(all_ranked[:10]):
        logger.info(f"  #{i+1}: Community {comm['community_id']} | "
                    f"Risk: {comm['risk_score']:.4f} | "
                    f"Fraud Rate: {comm['fraud_rate']:.4f} | "
                    f"Transactions: {comm['total_transactions']}")

    # ── Step 5: 评估检测效果（可选，用于有标签数据）──────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Evaluating detection performance (supervised)")
    logger.info("=" * 60)
    metrics = detector.evaluate_community_detection(
        edge_labels=data['edge_labels'],
        edge_index=data['edge_index'],
    )

    # ── Step 6: 打印摘要 ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    detector.print_top_suspicious(top_k=10)

    # ── Step 7: 导出结果（使用类型转换函数）──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Exporting results")
    logger.info("=" * 60)

    # 完整 JSON（社群 + 统计 + 可疑列表）- 转换后保存
    detector.export_results(str(output_dir / "community_detection_results.json"))

    # 保存所有社群的排序结果（CSV）
    sorted_results = []
    for idx, comm in enumerate(all_ranked):
        sorted_results.append({
            'rank': idx + 1,
            'community_id': comm['community_id'],
            'risk_score': comm['risk_score'],
            'fraud_rate': comm['fraud_rate'],
            'total_transactions': comm['total_transactions'],
            'fraud_transactions': comm['fraud_transactions'],
            'num_cards': comm['num_cards'],
            'num_merchants': comm['num_merchants'],
            'density': comm.get('density', 0),
            'topology_anomaly': comm.get('topology_anomaly', 0),
            'behavior_anomaly': comm.get('behavior_anomaly', 0),
            'time_concentration': comm.get('time_concentration', 0),
            'night_ratio': comm.get('night_ratio', 0),
            'round_amt_ratio': comm.get('round_amt_ratio', 0),
            'avg_amount': comm.get('amt_mean', 0),
        })
    
    # 保存所有社群（按风险排序）
    pd.DataFrame(sorted_results).to_csv(
        output_dir / "all_communities_ranked.csv", 
        index=False
    )
    logger.info(f"Saved all_communities_ranked.csv ({len(sorted_results)} communities)")
    
    # 保存前10个高风险社群（用于快速查看）
    top_k = 10
    pd.DataFrame(sorted_results[:top_k]).to_csv(
        output_dir / f"top_{top_k}_suspicious_communities.csv", 
        index=False
    )
    logger.info(f"Saved top_{top_k}_suspicious_communities.csv")
    
    # 兼容旧格式：保存可疑社群CSV（所有按风险排序的社群）
    pd.DataFrame(sorted_results).to_csv(
        output_dir / "suspicious_communities.csv", 
        index=False
    )
    logger.info(f"Saved suspicious_communities.csv ({len(sorted_results)} rows - all communities ranked)")

    # 评估指标 JSON - 使用转换函数
    with open(output_dir / "evaluation_metrics.json", 'w') as f:
        serializable_metrics = convert_to_serializable(metrics)
        json.dump(serializable_metrics, f, indent=2)
    logger.info("Saved evaluation_metrics.json")

    # 汇总统计 JSON - 使用转换函数
    summary = {
        'total_communities':    len(detector.communities),
        'suspicious_communities': len(all_ranked),  # 所有社群都排序输出
        'top_10_risk_scores': [
            {'rank': i+1, 'community_id': comm['community_id'], 'risk_score': comm['risk_score']}
            for i, comm in enumerate(all_ranked[:10])
        ],
        'evaluation':           metrics,
    }
    with open(output_dir / "summary_stats.json", 'w') as f:
        serializable_summary = convert_to_serializable(summary)
        json.dump(serializable_summary, f, indent=2)
    logger.info("Saved summary_stats.json")

    logger.info("\n" + "=" * 60)
    logger.info("Community detection completed!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 60)

    return detector, metrics


def analyze_community_details(detector: CommunityDetector, community_id: int, output_dir: Path):
    """分析特定社群的详细信息"""
    logger.info(f"Analyzing community {community_id}...")

    subgraph = detector.get_community_subgraph(community_id)
    stats    = detector.community_stats.get(community_id, {})

    detail = {
        'community_id': community_id,
        'node_count':   len(detector.communities[community_id]),
        'edge_count':   subgraph.number_of_edges(),
        'stats':        stats,
        'nodes':        list(detector.communities[community_id]),
    }

    out_file = output_dir / f"community_{community_id}_detail.json"
    with open(out_file, 'w') as f:
        # 使用转换函数确保可序列化
        serializable_detail = convert_to_serializable(detail)
        json.dump(serializable_detail, f, indent=2, default=str)
    logger.info(f"Saved {out_file.name}")

    return detail


if __name__ == "__main__":
    logger.add("community_detection.log", rotation="500 MB")

    if not GRAPH_PATH.exists():
        logger.error(f"Graph file not found: {GRAPH_PATH}")
        logger.error("Run the full pipeline first to generate fraud_graph.pt")
        sys.exit(1)

    # 1. 加载数据
    data = load_graph_data(GRAPH_PATH)

    # 2. 运行社群检测
    detector, metrics = run_community_detection(data, OUTPUT_DIR)

    # 3. 分析前 3 个最可疑社群（按风险排序）
    if hasattr(detector, 'suspicious_communities') and detector.suspicious_communities:
        logger.info("Analyzing top suspicious communities in detail...")
        # 只分析实际存在的社群（最多3个）
        num_to_analyze = min(3, len(detector.suspicious_communities))
        for i in range(num_to_analyze):
            comm = detector.suspicious_communities[i]
            analyze_community_details(detector, comm['community_id'], OUTPUT_DIR)
    else:
        logger.info("No suspicious communities found to analyze in detail.")

    # 4. 最终统计
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total communities:    {len(detector.communities)}")
    logger.info(f"All communities ranked and saved")
    
    if hasattr(detector, 'suspicious_communities') and detector.suspicious_communities:
        # 输出前3名风险分数（如果存在）
        logger.info(f"Top 1 risk score:     {detector.suspicious_communities[0]['risk_score']:.4f} (Community {detector.suspicious_communities[0]['community_id']})")
        
        if len(detector.suspicious_communities) >= 2:
            logger.info(f"Top 2 risk score:     {detector.suspicious_communities[1]['risk_score']:.4f} (Community {detector.suspicious_communities[1]['community_id']})")
        
        if len(detector.suspicious_communities) >= 3:
            logger.info(f"Top 3 risk score:     {detector.suspicious_communities[2]['risk_score']:.4f} (Community {detector.suspicious_communities[2]['community_id']})")
        
        # 输出前5名的简要信息
        logger.info("\nTop 5 communities by risk score:")
        for i in range(min(5, len(detector.suspicious_communities))):
            comm = detector.suspicious_communities[i]
            logger.info(f"  #{i+1}: Community {comm['community_id']} | Risk: {comm['risk_score']:.4f} | Fraud Rate: {comm['fraud_rate']:.4f} | Transactions: {comm['total_transactions']}")
    else:
        logger.info("No communities found!")
    
    logger.info(f"\nCommunity F1:         {metrics['community_f1']:.4f}")
    logger.info(f"Community Recall:     {metrics['community_recall']:.4f}")
    logger.info(f"Community Precision:  {metrics['community_precision']:.4f}")
    logger.info("=" * 60)