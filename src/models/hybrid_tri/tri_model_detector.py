#!/usr/bin/env python3
"""
三模型协同欺诈检测器
GNN（有监督）+ 图自编码器（无监督）+ 社群检测（拓扑分析）

运行方式（从项目根目录）：
    python src/tri_model_detector.py
    python src/tri_model_detector.py --ae_epochs 30 --device cpu
    python src/tri_model_detector.py --skip_gnn_train  # 复用已有 GNN 模型

输出文件（outputs/tri_model/）：
    gnn_results.json          GNN 测试集指标
    ae_anomaly_scores.npy     AE 对所有边的异常分数
    community_analysis.json   Louvain/KMeans 社群分析结果
    fusion_scores.csv         测试集每条边的三模型分数 + 融合分数
    summary_report.json       三模型 + 融合 汇总指标
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
)
from loguru import logger
from torch_geometric.data import HeteroData

# ── 路径 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import create_model, CommunityDetector
from src.training import FraudDetectionTrainer, UnsupervisedTrainer

# ── 常量 ──────────────────────────────────────────────────────────────────────
GRAPH_PATH   = str(PROJECT_ROOT / "outputs" / "graph"   / "fraud_graph.pt")
GNN_MODEL    = str(PROJECT_ROOT / "outputs" / "models"  / "graphsage_fraud_detector.pt")
OUTPUT_DIR   = str(PROJECT_ROOT / "outputs" / "tri_model")
PROCESSED    = str(PROJECT_ROOT / "data"    / "processed")

# 融合权重（三模型信号线性叠加）
W_GNN  = 0.50
W_AE   = 0.30
W_COMM = 0.20


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════
def _device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _load_graph() -> "HeteroData":
    logger.info(f"加载图: {GRAPH_PATH}")
    data = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    logger.info(
        f"  card: {data['card'].x.shape}  "
        f"merchant: {data['merchant'].x.shape}  "
        f"edges: {data['card','transacts','merchant'].edge_index.shape[1]}"
    )
    return data


def _load_edges_df() -> pd.DataFrame:
    """加载 edges.parquet / edges.csv，用于社群统计"""
    for fmt in ("edges.csv", "edges.parquet"):
        p = os.path.join(PROCESSED, fmt)
        if os.path.exists(p):
            return pd.read_csv(p) if fmt.endswith(".csv") else pd.read_parquet(p)
    logger.warning("edges 文件未找到，社群时间/金额统计将被跳过")
    return pd.DataFrame()


def _metrics(labels: np.ndarray, probs: np.ndarray,
             threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    auc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    ap   = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)
    return dict(auc=float(auc), ap=float(ap),
                precision=float(p), recall=float(r), f1=float(f1),
                confusion_matrix=cm.tolist())


def _save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: GNN 有监督训练 / 加载
# ══════════════════════════════════════════════════════════════════════════════
def run_gnn(data, device: str, epochs: int, skip_train: bool) -> dict:
    """
    训练或加载 GNN，返回测试集各边预测概率。

    Returns:
        {
          'probs':   np.ndarray (N_test,)   欺诈概率
          'labels':  np.ndarray (N_test,)   真实标签
          'idx':     np.ndarray (N_test,)   边在全图中的索引
          'metrics': dict
        }
    """
    logger.info("=" * 60)
    logger.info("Step 1 / 3  GNN 有监督训练")
    logger.info("=" * 60)

    card_dim     = data["card"].x.shape[1]
    merchant_dim = data["merchant"].x.shape[1]
    edge_dim     = data["card", "transacts", "merchant"].edge_attr.shape[1] \
                   if hasattr(data["card", "transacts", "merchant"], "edge_attr") else 0

    model = create_model(
        model_type="graphsage",
        card_in_channels=card_dim,
        merchant_in_channels=merchant_dim,
        edge_in_channels=edge_dim,
        hidden_channels=256,
        out_channels=128,
        num_layers=3,
        dropout=0.4
    )

    trainer = FraudDetectionTrainer(
        model, device=device,
        lr=0.001, weight_decay=1e-4,
        fraud_weight=20.0, use_focal_loss=True, focal_gamma=2.0
    )

    if skip_train and os.path.exists(GNN_MODEL):
        logger.info(f"加载已有 GNN 模型: {GNN_MODEL}")
        trainer.load_model(GNN_MODEL)
    else:
        logger.info(f"训练 GNN  epochs={epochs}")
        os.makedirs(os.path.dirname(GNN_MODEL), exist_ok=True)
        trainer.train(
            data,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            num_epochs=epochs,
            early_stopping_patience=40,
            save_best=True,
            save_path=GNN_MODEL
        )

    # 测试集推理
    test_results = trainer.test(data, data.test_mask)
    test_idx     = data.test_mask.nonzero(as_tuple=True)[0].numpy()

    metrics = _metrics(test_results["labels"], test_results["probabilities"])
    logger.info(
        f"  GNN  AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}  "
        f"F1={metrics['f1']:.4f}  Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}"
    )

    # 保存 GNN 结果
    gnn_out = dict(metrics=metrics, epochs_trained=epochs,
                   model_path=GNN_MODEL)
    _save_json(gnn_out, os.path.join(OUTPUT_DIR, "gnn_results.json"))

    return dict(
        probs=test_results["probabilities"],
        labels=test_results["labels"],
        idx=test_idx,
        metrics=metrics,
        trainer=trainer           # 后续社群检测用 embeddings
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 图自编码器 无监督异常检测
# ══════════════════════════════════════════════════════════════════════════════
def run_ae(data, device: str, epochs: int) -> np.ndarray:
    """
    训练图自编码器，返回 **全图所有边** 的异常分数向量（形状 N_edges,）。
    anomaly_score = 1 − reconstruction_probability
    """
    logger.info("=" * 60)
    logger.info("Step 2 / 3  图自编码器 无监督异常检测")
    logger.info("=" * 60)

    card_dim     = data["card"].x.shape[1]
    merchant_dim = data["merchant"].x.shape[1]
    edge_dim     = data["card", "transacts", "merchant"].edge_attr.shape[1] \
                   if hasattr(data["card", "transacts", "merchant"], "edge_attr") else 0

    ae_model = create_model(
        model_type="autoencoder",
        card_in_channels=card_dim,
        merchant_in_channels=merchant_dim,
        edge_in_channels=edge_dim,
        hidden_channels=256,
        out_channels=128,
        num_layers=3,
        dropout=0.4
    )

    ae_trainer = UnsupervisedTrainer(ae_model, device=device, lr=0.001)

    logger.info(f"训练 AE  epochs={epochs}")
    for ep in range(epochs):
        loss, _ = ae_trainer.train_epoch(data)
        if ep % 10 == 0 or ep == epochs - 1:
            logger.info(f"  AE epoch {ep:3d}  loss={loss:.4f}")

    # 计算全图所有边的异常分数
    scores = ae_trainer.compute_anomaly_scores(data).numpy()   # (N_edges,)
    logger.info(
        f"  AE 异常分数  mean={scores.mean():.4f}  "
        f"std={scores.std():.4f}  "
        f"max={scores.max():.4f}"
    )

    # 保存
    out_path = os.path.join(OUTPUT_DIR, "ae_anomaly_scores.npy")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(out_path, scores)
    logger.info(f"  → {out_path}")

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: 社群检测
# ══════════════════════════════════════════════════════════════════════════════
def run_community(data, embeddings: dict, edges_df: pd.DataFrame) -> dict:
    """
    运行社群检测，返回：
      {
        'detector':    CommunityDetector 实例
        'node_risk':   dict  card_node_str -> risk_score  (0 表示未在可疑社群)
        'metrics':     dict  community-level precision/recall/f1
      }
    """
    logger.info("=" * 60)
    logger.info("Step 3 / 3  社群检测 & 欺诈团伙识别")
    logger.info("=" * 60)

    edge_index = data["card", "transacts", "merchant"].edge_index

    detector = CommunityDetector()
    detector.build_networkx_graph(
        edge_index,
        data.card_mapping,
        data.merchant_mapping
    )

    # Louvain 优先；库不可用则 K-Means
    try:
        detector.detect_louvain(resolution=1.5)
        method = "louvain"
    except ImportError:
        logger.warning("python-louvain 未安装，回退到 K-Means 嵌入聚类")
        card_emb     = embeddings["card"]
        merchant_emb = embeddings["merchant"]
        detector.detect_embedding_cluster(
            card_emb, merchant_emb,
            method="kmeans", n_clusters=100
        )
        method = "kmeans"

    logger.info(f"  检测方法={method}  社群数={len(detector.communities)}")

    # 时间 / 金额（可选）
    edge_labels  = data["card", "transacts", "merchant"].edge_label
    edge_times   = torch.tensor(edges_df["hour"].values.astype(float)) \
                   if "hour"           in edges_df.columns else None
    edge_amounts = torch.tensor(edges_df["TransactionAmt"].values.astype(float)) \
                   if "TransactionAmt" in edges_df.columns else None

    detector.compute_community_stats(
        edge_labels, edge_index,
        edge_times=edge_times, edge_amounts=edge_amounts
    )

    detector.identify_suspicious_communities(
        fraud_rate_threshold=0.02,
        min_transactions=100,
        min_cards=10
    )
    detector.print_top_suspicious(top_k=10)

    comm_metrics = detector.evaluate_community_detection(edge_labels, edge_index)

    # ── 构建 card_node -> risk_score 查找表 ──────────────────────────────────
    node_to_comm = {}
    for comm_id, nodes in detector.communities.items():
        for n in nodes:
            node_to_comm[n] = comm_id

    comm_risk_map = {}
    for susp in detector.suspicious_communities:
        comm_id = susp["community_id"]
        for n in detector.communities.get(comm_id, []):
            if n.startswith("card_"):
                comm_risk_map[n] = susp["risk_score"]

    # 社群统计导出
    export_data = {
        "method":                method,
        "total_communities":     len(detector.communities),
        "suspicious_communities": len(detector.suspicious_communities),
        "community_metrics":     comm_metrics,
        "top_suspicious": [
            {k: v for k, v in s.items() if k != "nodes"}   # 去掉大列表
            for s in detector.suspicious_communities[:20]
        ]
    }
    _save_json(export_data, os.path.join(OUTPUT_DIR, "community_analysis.json"))

    logger.info(
        f"  社群检测  Precision={comm_metrics['community_precision']:.4f}  "
        f"Recall={comm_metrics['community_recall']:.4f}  "
        f"F1={comm_metrics['community_f1']:.4f}"
    )

    return dict(
        detector=detector,
        node_risk=comm_risk_map,
        metrics=comm_metrics
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: 三模型信号融合
# ══════════════════════════════════════════════════════════════════════════════
def run_fusion(data, gnn_out: dict, ae_scores: np.ndarray,
               comm_out: dict) -> dict:
    """
    在测试集上对三模型分数做加权线性融合，评估融合效果。

    融合公式：
        fused = W_GNN * gnn_prob + W_AE * ae_score_norm + W_COMM * comm_risk
    """
    logger.info("=" * 60)
    logger.info("Step 4 / 4  三模型信号融合")
    logger.info("=" * 60)

    test_idx     = gnn_out["idx"]           # 测试边在全图的下标
    gnn_probs    = gnn_out["probs"]         # (N_test,)
    labels       = gnn_out["labels"]        # (N_test,)
    edge_index   = data["card", "transacts", "merchant"].edge_index.numpy()
    node_risk    = comm_out["node_risk"]

    # ── AE 分数：取测试子集，归一化到 [0,1] ──────────────────────────────────
    ae_test    = ae_scores[test_idx]
    ae_min, ae_max = ae_test.min(), ae_test.max()
    ae_norm    = (ae_test - ae_min) / (ae_max - ae_min + 1e-8)

    # ── 社群风险分数：每条测试边 → 查 src card 的风险值 ──────────────────────
    comm_risk = np.zeros(len(test_idx), dtype=np.float32)
    for i, eidx in enumerate(test_idx):
        card_node = f"card_{edge_index[0, eidx]}"
        comm_risk[i] = node_risk.get(card_node, 0.0)

    # ── 融合 ──────────────────────────────────────────────────────────────────
    fused = W_GNN * gnn_probs + W_AE * ae_norm + W_COMM * comm_risk
    fused = np.clip(fused, 0.0, 1.0)

    # ── 评估 ──────────────────────────────────────────────────────────────────
    fused_metrics = _metrics(labels, fused)
    logger.info(
        f"  融合  AUC={fused_metrics['auc']:.4f}  AP={fused_metrics['ap']:.4f}  "
        f"F1={fused_metrics['f1']:.4f}  "
        f"Precision={fused_metrics['precision']:.4f}  "
        f"Recall={fused_metrics['recall']:.4f}"
    )

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "fusion_scores.csv")
    df = pd.DataFrame({
        "edge_idx":    test_idx,
        "label":       labels.astype(int),
        "gnn_prob":    np.round(gnn_probs, 6),
        "ae_score":    np.round(ae_norm,   6),
        "comm_risk":   np.round(comm_risk, 6),
        "fused_score": np.round(fused,     6),
        "fused_pred":  (fused >= 0.5).astype(int),
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"  → {csv_path}")

    return dict(
        fused=fused, labels=labels,
        metrics=fused_metrics,
        df=df
    )


# ══════════════════════════════════════════════════════════════════════════════
# 汇总报告
# ══════════════════════════════════════════════════════════════════════════════
def save_summary(gnn_out: dict, ae_scores: np.ndarray,
                 comm_out: dict, fusion_out: dict,
                 elapsed: float):
    cm   = comm_out["metrics"]
    report = {
        "timestamp":      datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "fusion_weights": {"gnn": W_GNN, "ae": W_AE, "community": W_COMM},
        "models": {
            "GNN_GraphSAGE": {
                **gnn_out["metrics"],
                "note": "有监督边分类，Focal Loss γ=2, fraud_weight=20"
            },
            "AE_GraphAutoEncoder": {
                "ae_score_mean": float(ae_scores.mean()),
                "ae_score_std":  float(ae_scores.std()),
                "note": "无监督重建，anomaly_score=1-p_reconstruct，无标签评估指标"
            },
            "CommunityDetector": {
                **cm,
                "note": "Louvain/KMeans 社群拓扑分析，社群级精确率/召回率"
            },
            "Fusion": {
                **fusion_out["metrics"],
                "note": f"线性融合 {W_GNN}×GNN + {W_AE}×AE + {W_COMM}×Comm"
            }
        }
    }
    _save_json(report, os.path.join(OUTPUT_DIR, "summary_report.json"))

    # 控制台打印对比表
    SEP = "-" * 70
    print(f"\n{'三模型协同检测汇总报告':^70}")
    print(SEP)
    print(f"{'Model':<26}{'AUC':>8}{'AP':>8}{'F1':>8}{'Prec':>8}{'Recall':>8}")
    print(SEP)
    for name, m in report["models"].items():
        if "auc" in m:
            print(f"{name:<26}{m['auc']:>8.4f}{m.get('ap',0):>8.4f}"
                  f"{m['f1']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}")
        else:
            print(f"{name:<26}{'—':>8}{'—':>8}{'—':>8}{'—':>8}{'—':>8}")
    print(SEP)
    print(f"\n输出目录: {OUTPUT_DIR}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="三模型协同欺诈检测")
    p.add_argument("--gnn_epochs",  type=int, default=50,
                   help="GNN 训练轮数（默认 50）")
    p.add_argument("--ae_epochs",   type=int, default=50,
                   help="AE  训练轮数（默认 50）")
    p.add_argument("--device",      type=str, default="auto",
                   help="cpu / cuda / auto")
    p.add_argument("--skip_gnn_train", action="store_true",
                   help="若已有 GNN 模型则直接加载，跳过训练")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args   = parse_args()
    device = _device(args.device)
    t0     = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.add(
        os.path.join(OUTPUT_DIR, "logs", "tri_model_{time}.log"),
        rotation="10 MB", level="INFO"
    )

    print(f"\n{'='*60}")
    print("  三模型协同欺诈检测器")
    print(f"  GNN epochs : {args.gnn_epochs}")
    print(f"  AE  epochs : {args.ae_epochs}")
    print(f"  Device     : {device}")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # ── 加载公共数据 ──────────────────────────────────────────────────────────
    data     = _load_graph()
    edges_df = _load_edges_df()

    # ── Step 1: GNN ───────────────────────────────────────────────────────────
    gnn_out = run_gnn(
        data, device,
        epochs=args.gnn_epochs,
        skip_train=args.skip_gnn_train
    )

    # GNN 嵌入（供社群检测 K-Means 使用）
    embeddings = gnn_out["trainer"].get_embeddings(data)

    # ── Step 2: AE ────────────────────────────────────────────────────────────
    ae_scores = run_ae(data, device, epochs=args.ae_epochs)

    # ── Step 3: 社群检测 ──────────────────────────────────────────────────────
    comm_out = run_community(data, embeddings, edges_df)

    # ── Step 4: 融合 ──────────────────────────────────────────────────────────
    fusion_out = run_fusion(data, gnn_out, ae_scores, comm_out)

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    save_summary(gnn_out, ae_scores, comm_out, fusion_out, elapsed)


if __name__ == "__main__":
    main()
