"""
单独重跑改进后的 Hybrid 模型（跳过 Spark + GNN 训练）
使用已有的:
  - outputs/graph/fraud_graph.pt
  - outputs/models/graphsage_fraud_detector.pt
  - outputs/communities/suspicious_communities.json  (hybrid_com 用)
"""
import sys
import os
import torch
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.models import create_model, HybridFraudDetector, train_hybrid_model
from src.models.hybrid_com import HybridFraudDetectorWithCommunity, train_hybrid_model_with_community
from src.training import FraudDetectionTrainer
from src.data_processing import PyGGraphBuilder

# ── 路径配置 ──────────────────────────────────────────────────────────────────
OUTPUT_DIR       = project_root / "outputs"
GRAPH_PATH       = OUTPUT_DIR / "graph" / "fraud_graph.pt"
GNN_MODEL_PATH   = OUTPUT_DIR / "models" / "graphsage_fraud_detector.pt"
HYBRID_SAVE_PATH = str(OUTPUT_DIR / "models" / "hybrid_gnn_lgb_v2.pkl")
COMMUNITY_JSON   = OUTPUT_DIR / "communities" / "suspicious_communities.json"
HYBRID_COM_PATH  = str(OUTPUT_DIR / "models" / "hybrid_gnn_lgb_community_v2.pkl")

# ── 1. 加载图数据 ─────────────────────────────────────────────────────────────
logger.info("Loading graph...")
builder = PyGGraphBuilder()
data = builder.load_graph(str(GRAPH_PATH))
logger.info(f"Graph loaded: {data}")

# ── 2. 重建 GNN 模型并加载权重 ────────────────────────────────────────────────
card_in_dim     = data['card'].x.shape[1]
merchant_in_dim = data['merchant'].x.shape[1]
edge_in_dim     = data['card', 'transacts', 'merchant'].edge_attr.shape[1] \
                  if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr') else 0

model = create_model(
    model_type='graphsage',
    card_in_channels=card_in_dim,
    merchant_in_channels=merchant_in_dim,
    edge_in_channels=edge_in_dim,
    hidden_channels=256,
    out_channels=128,
    num_layers=3,
    dropout=0.3,
)

trainer = FraudDetectionTrainer(model, device='auto')
trainer.load_model(str(GNN_MODEL_PATH))
logger.info("GNN model loaded.")

# ── 3. 重跑 hybrid_model（含 SMOTE + 最优阈值 + AP 主指标）────────────────────
logger.info("=" * 60)
logger.info("Re-training Hybrid GNN+LightGBM with improvements...")
hybrid, results = train_hybrid_model(
    data,
    trainer,
    train_mask=data.train_mask,
    val_mask=data.val_mask,
    test_mask=data.test_mask,
    save_path=HYBRID_SAVE_PATH,
)

test = results['test']
logger.info("=== Hybrid v2 Test Results ===")
logger.info(f"  Threshold: {test.get('threshold', 0.5):.2f}")
logger.info(f"  AP:        {test['ap']:.4f}")
logger.info(f"  AUC:       {test['auc']:.4f}")
logger.info(f"  Precision: {test['precision']:.4f}")
logger.info(f"  Recall:    {test['recall']:.4f}")
logger.info(f"  F1:        {test['f1']:.4f}")

# ── 4. 重跑 hybrid_com（含社群特征 + SMOTE + 最优阈值）──────────────────────
if COMMUNITY_JSON.exists():
    logger.info("=" * 60)
    logger.info("Re-training Hybrid+Community with improvements...")
    hybrid_com, com_results = train_hybrid_model_with_community(
        data,
        trainer,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        community_json_path=str(COMMUNITY_JSON),
        baseline_results_path=str(OUTPUT_DIR / "models" / "hybrid_gnn_lgb_results.json"),
        save_path=HYBRID_COM_PATH,
        alpha=0.7,
    )
else:
    logger.warning(f"Community JSON not found at {COMMUNITY_JSON}, skipping hybrid_com.")

logger.info("Done. Results saved to outputs/models/")
import json

# 保存Hybrid结果
hybrid_results = {
    'test_ap': test['ap'],
    'test_auc': test['auc'],
    'test_precision': test['precision'],
    'test_recall': test['recall'],
    'test_f1': test['f1'],
    'threshold': test.get('threshold', 0.5)
}

with open(OUTPUT_DIR / 'models' / 'hybrid_v2_results.json', 'w') as f:
    json.dump(hybrid_results, f, indent=2)
    logger.info(f"Results saved to {OUTPUT_DIR / 'models' / 'hybrid_v2_results.json'}")