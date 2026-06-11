"""
集成 Borderline Shifting 思想
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gnn_models import HeteroGraphSAGE
from src.models.community_detector import CommunityDetector
from src.models.community_feature_fusion_v2 import CommunityFeatureFusionDetector
from src.models.hybrid_model import HybridFraudDetector
from src.training.trainer import FraudDetectionTrainer
from src.data_processing import PyGGraphBuilder

# ── 路径配置 ──────────────────────────────────────────────────────────────────
GRAPH_PATH  = PROJECT_ROOT / "outputs" / "graph" / "fraud_graph.pt"
OUTPUT_DIR  = PROJECT_ROOT / "outputs" / "community_fusion_Borderline"
GNN_CKPT    = OUTPUT_DIR / "gnn_model.pt"

# ── 超参数 ────────────────────────────────────────────────────────────────────
GNN_HIDDEN   = 128
GNN_OUT      = 64
GNN_LAYERS   = 2
GNN_EPOCHS   = 150
GNN_PATIENCE = 30
TRAIN_RATIO  = 0.7
VAL_RATIO    = 0.15   # test = 1 - TRAIN - VAL

# Borderline Shifting（方案A：最小改动）
BORDERLINE_LOW = 0.30
BORDERLINE_HIGH = 0.70
BORDERLINE_STRATEGY = "both"   # "weight" | "copy" | "both"
BORDERLINE_WEIGHT = 2.0
BORDERLINE_COPY_TIMES = 1


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    logger.info(f"Loading graph from {GRAPH_PATH}")
    builder = PyGGraphBuilder()
    data = builder.load_graph(str(GRAPH_PATH))
    logger.info(f"Graph loaded: {data}")

    edge_label = data['card', 'transacts', 'merchant'].edge_label
    n_edges = edge_label.shape[0]
    logger.info(f"Total edges: {n_edges}, Fraud rate: {edge_label.float().mean():.4f}")
    return data, n_edges, edge_label


def make_masks(n_edges: int, edge_label: torch.Tensor):
    """按比例划分 train/val/test mask，保持欺诈比例"""
    indices = np.arange(n_edges)
    labels  = edge_label.numpy()

    idx_train, idx_tmp = train_test_split(
        indices, test_size=1 - TRAIN_RATIO, stratify=labels, random_state=42
    )
    val_size = VAL_RATIO / (1 - TRAIN_RATIO)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=1 - val_size,
        stratify=labels[idx_tmp], random_state=42
    )

    def to_mask(idx):
        m = torch.zeros(n_edges, dtype=torch.bool)
        m[idx] = True
        return m

    train_mask = to_mask(idx_train)
    val_mask   = to_mask(idx_val)
    test_mask  = to_mask(idx_test)

    logger.info(
        f"Masks — train: {train_mask.sum()} "
        f"val: {val_mask.sum()} "
        f"test: {test_mask.sum()}"
    )
    return train_mask, val_mask, test_mask


# ─────────────────────────────────────────────────────────────────────────────
# GNN 训练 / 加载
# ─────────────────────────────────────────────────────────────────────────────

def get_gnn_trainer(data, train_mask, val_mask):
    card_dim     = data['card'].x.shape[1]
    merchant_dim = data['merchant'].x.shape[1]
    edge_dim     = (data['card', 'transacts', 'merchant'].edge_attr.shape[1]
                    if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr') else 0)

    model = HeteroGraphSAGE(
        card_in_channels=card_dim,
        merchant_in_channels=merchant_dim,
        edge_in_channels=edge_dim,
        hidden_channels=GNN_HIDDEN,
        out_channels=GNN_OUT,
        num_layers=GNN_LAYERS,
    )
    trainer = FraudDetectionTrainer(model, device='auto')

    if GNN_CKPT.exists():
        logger.info(f"Loading existing GNN checkpoint: {GNN_CKPT}")
        trainer.load_model(str(GNN_CKPT))
    else:
        logger.info("Training GNN from scratch...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        trainer.train(
            data, train_mask, val_mask,
            num_epochs=GNN_EPOCHS,
            early_stopping_patience=GNN_PATIENCE,
            save_best=True,
            save_path=str(GNN_CKPT),
        )

    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# 社群检测
# ─────────────────────────────────────────────────────────────────────────────

def run_community_detection(data, n_edges: int):
    edge_index = data['card', 'transacts', 'merchant'].edge_index
    edge_label = data['card', 'transacts', 'merchant'].edge_label.float()

    num_cards     = data['card'].x.shape[0]
    num_merchants = data['merchant'].x.shape[0]
    card_mapping     = {i: i for i in range(num_cards)}
    merchant_mapping = {i: i for i in range(num_merchants)}

    # 提取时间戳和金额
    timestamps = amounts = None
    if hasattr(data['card', 'transacts', 'merchant'], 'edge_attr'):
        ea = data['card', 'transacts', 'merchant'].edge_attr
        if ea.shape[1] >= 2:
            timestamps = ea[:, 0].float()
            amounts    = ea[:, 1].float()
        elif ea.shape[1] == 1:
            amounts = ea[:, 0].float()

    if timestamps is None:
        timestamps = torch.arange(n_edges, dtype=torch.float32)
    if amounts is None:
        amounts = torch.ones(n_edges, dtype=torch.float32)

    if timestamps.max() > 1000:
        timestamps = timestamps / 3600
    amounts = torch.log1p(amounts.clamp(min=0))

    detector = CommunityDetector()
    detector.build_networkx_graph(edge_index, card_mapping, merchant_mapping)

    logger.info("Running Louvain community detection...")
    detector.detect_louvain(resolution=1.0)
    logger.info(f"Detected {len(detector.communities)} communities")

    detector.compute_community_stats(
        edge_labels=edge_label,
        edge_index=edge_index,
        edge_times=timestamps,
        edge_amounts=amounts,
    )
    logger.info(f"Community stats computed for {len(detector.community_stats)} communities")

    return detector


# ─────────────────────────────────────────────────────────────────────────────
# 训练 & 评估
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(data, embeddings, community_detector,
                       train_mask, val_mask, test_mask):
    results = {}

    # ── Baseline：原始 HybridFraudDetector ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("Baseline: HybridFraudDetector (no community features)")
    logger.info("=" * 60)
    baseline = HybridFraudDetector()
    baseline.train(data, embeddings, train_mask, val_mask,
                   num_boost_round=2000, early_stopping_rounds=100)
    base_res = baseline.evaluate(data, embeddings, test_mask)
    results['baseline'] = {
        'auc': float(base_res['auc']),
        'ap':  float(base_res['ap']),
        'f1':  float(base_res['f1']),
        'precision': float(base_res['precision']),
        'recall':    float(base_res['recall']),
        'threshold': float(base_res['threshold']),
    }
    logger.info(
        f"Baseline  AUC={base_res['auc']:.4f}  AP={base_res['ap']:.4f}  "
        f"F1={base_res['f1']:.4f}  P={base_res['precision']:.4f}  R={base_res['recall']:.4f}"
    )

    # ── 融合模型：CommunityFeatureFusionDetector ──────────────────────────────
    logger.info("=" * 60)
    logger.info("Fusion: CommunityFeatureFusionDetector (+14 community features)")
    logger.info("=" * 60)
    fusion = CommunityFeatureFusionDetector()
    fusion.set_community_detector(community_detector)
    fusion_train_res = fusion.train_with_borderline_shifting(
        data, embeddings, train_mask, val_mask,
        num_boost_round=2000, early_stopping_rounds=100,
        borderline_low=BORDERLINE_LOW,
        borderline_high=BORDERLINE_HIGH,
        strategy=BORDERLINE_STRATEGY,
        borderline_weight=BORDERLINE_WEIGHT,
        borderline_copy_times=BORDERLINE_COPY_TIMES,
    )
    fuse_res = fusion.evaluate(data, embeddings, test_mask)
    results['fusion'] = {
        'auc': float(fuse_res['auc']),
        'ap':  float(fuse_res['ap']),
        'f1':  float(fuse_res['f1']),
        'precision': float(fuse_res['precision']),
        'recall':    float(fuse_res['recall']),
        'threshold': float(fuse_res['threshold']),
        'borderline_samples': int(fusion_train_res.get('borderline_samples', 0)),
        'matched_train_samples': int(fusion_train_res.get('matched_train_samples', 0)),
    }
    logger.info(
        f"Fusion    AUC={fuse_res['auc']:.4f}  AP={fuse_res['ap']:.4f}  "
        f"F1={fuse_res['f1']:.4f}  P={fuse_res['precision']:.4f}  R={fuse_res['recall']:.4f}"
    )

    # ── 对比摘要 ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Comparison Summary")
    logger.info("=" * 60)
    for metric in ('auc', 'ap', 'f1', 'precision', 'recall'):
        delta = results['fusion'][metric] - results['baseline'][metric]
        sign  = "+" if delta >= 0 else ""
        logger.info(
            f"  {metric.upper():10s}  baseline={results['baseline'][metric]:.4f}  "
            f"fusion={results['fusion'][metric]:.4f}  delta={sign}{delta:.4f}"
        )

    # 特征重要性（融合模型 top 20）
    imp = fusion.get_feature_importance(top_k=20)
    logger.info("\nTop 20 feature importance (fusion model):")
    for _, row in imp.iterrows():
        logger.info(f"  {row['feature']:35s} {row['importance']:.2f}")

    return results, baseline, fusion


# ─────────────────────────────────────────────────────────────────────────────
# 保存结果
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, baseline: HybridFraudDetector,
                 fusion: CommunityFeatureFusionDetector):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON 对比结果
    out_json = OUTPUT_DIR / "comparison_results_v2.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {out_json}")

    # 模型文件
    baseline.save(str(OUTPUT_DIR / "baseline_hybrid_v2.pkl"))
    fusion.save(str(OUTPUT_DIR / "fusion_hybrid_v2.pkl"))

    # 特征重要性 CSV
    imp = fusion.get_feature_importance(top_k=50)
    imp.to_csv(OUTPUT_DIR / "fusion_feature_importance_v2.csv", index=False)
    logger.info(f"Saved fusion_feature_importance.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.add(str(OUTPUT_DIR / "run.log"), rotation="200 MB", enqueue=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not GRAPH_PATH.exists():
        logger.error(f"Graph file not found: {GRAPH_PATH}")
        sys.exit(1)

    # 1. 加载数据 & 划分 mask
    data, n_edges, edge_label = load_data()
    train_mask, val_mask, test_mask = make_masks(n_edges, edge_label)

    # 2. 训练 / 加载 GNN，获取嵌入
    trainer    = get_gnn_trainer(data, train_mask, val_mask)
    embeddings = trainer.get_embeddings(data)
    logger.info(
        f"Embeddings — card: {embeddings['card'].shape}, "
        f"merchant: {embeddings['merchant'].shape}"
    )

    # 3. 社群检测
    community_detector = run_community_detection(data, n_edges)

    # 4. 训练 & 评估两个模型
    results, baseline, fusion = train_and_evaluate(
        data, embeddings, community_detector,
        train_mask, val_mask, test_mask
    )

    # 5. 保存
    save_results(results, baseline, fusion)

    logger.info("Done. Results saved to: " + str(OUTPUT_DIR))
