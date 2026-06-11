#!/usr/bin/env python3
"""
对比实验：GraphSAGE+LightGBM（基线）vs GraphSAGE+LightGBM+AE+社群检测（增强）

两个模型共用同一个 GNN backbone（保证公平对比），差别只在 LightGBM 的输入特征：
  Model A（基线）  : card_emb ‖ merchant_emb ‖ diff ‖ prod ‖ edge_feat
  Model B（增强）  : 同上 + ae_anomaly_score + community_risk_score

运行（从项目根目录）：
    python src/enhanced_hybrid_comparator.py
    python src/enhanced_hybrid_comparator.py --skip_gnn_train
    python src/enhanced_hybrid_comparator.py --gnn_epochs 50 --ae_epochs 30

输出目录：outputs/comparison_tri/
"""

import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
)
from loguru import logger

# ── 路径 ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import create_model, CommunityDetector
from src.training import FraudDetectionTrainer, UnsupervisedTrainer

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ── 常量 ─────────────────────────────────────────────────────────────────────
GRAPH_PATH  = str(PROJECT_ROOT / "outputs" / "graph"  / "fraud_graph.pt")
GNN_PT      = str(PROJECT_ROOT / "outputs" / "models" / "graphsage_fraud_detector.pt")
OUT_DIR     = str(PROJECT_ROOT / "outputs" / "comparison_tri")
PROCESSED   = str(PROJECT_ROOT / "data"    / "processed")

LGB_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "num_leaves":       384,
    "max_depth":        15,
    "learning_rate":    0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq":     5,
    "min_child_samples":50,
    "reg_alpha":        0.05,
    "reg_lambda":       0.05,
    "verbose":          -1,
    "n_jobs":           -1,
    "is_unbalance":     True,
    "max_bin":          255,
}

# ══════════════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════════════
def _device(arg):
    return "cuda" if (arg == "auto" and torch.cuda.is_available()) else (
        "cpu" if arg == "auto" else arg)

def _load_graph():
    logger.info(f"加载图: {GRAPH_PATH}")
    data = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    ei   = data["card", "transacts", "merchant"].edge_index
    logger.info(f"  card {data['card'].x.shape}  merchant {data['merchant'].x.shape}"
                f"  edges {ei.shape[1]}")
    return data

def _load_edges_df():
    for fn in ("edges.csv", "edges.parquet"):
        p = os.path.join(PROCESSED, fn)
        if os.path.exists(p):
            return pd.read_csv(p) if fn.endswith(".csv") else pd.read_parquet(p)
    return pd.DataFrame()

def _metrics(y, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    auc  = roc_auc_score(y, probs)  if len(np.unique(y)) > 1 else 0.0
    ap   = average_precision_score(y, probs) if len(np.unique(y)) > 1 else 0.0
    p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary",
                                                   zero_division=0)
    cm = confusion_matrix(y, preds)
    return dict(auc=float(auc), ap=float(ap),
                precision=float(p), recall=float(r), f1=float(f1),
                confusion_matrix=cm.tolist())

def _save(obj, rel):
    path = os.path.join(OUT_DIR, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  → {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – GNN（共用 backbone）
# ══════════════════════════════════════════════════════════════════════════════
def run_gnn(data, device, epochs, skip_train):
    logger.info("━"*60)
    logger.info("Step 1  GNN 有监督训练（两个模型共用此 backbone）")
    logger.info("━"*60)

    cd = data["card"].x.shape[1]
    md = data["merchant"].x.shape[1]
    ed = data["card","transacts","merchant"].edge_attr.shape[1] \
         if hasattr(data["card","transacts","merchant"], "edge_attr") else 0

    model   = create_model("graphsage", card_in_channels=cd,
                           merchant_in_channels=md, edge_in_channels=ed,
                           hidden_channels=256, out_channels=128,
                           num_layers=3, dropout=0.4)
    trainer = FraudDetectionTrainer(model, device=device, lr=0.001,
                                    weight_decay=1e-4, fraud_weight=20.0,
                                    use_focal_loss=True, focal_gamma=2.0)

    if skip_train and os.path.exists(GNN_PT):
        logger.info(f"加载已有 GNN: {GNN_PT}")
        trainer.load_model(GNN_PT)
    else:
        logger.info(f"训练 GNN  epochs={epochs}")
        os.makedirs(os.path.dirname(GNN_PT), exist_ok=True)
        trainer.train(data, train_mask=data.train_mask, val_mask=data.val_mask,
                      num_epochs=epochs, early_stopping_patience=40,
                      save_best=True, save_path=GNN_PT)

    embeddings = trainer.get_embeddings(data)
    logger.info(f"  嵌入  card {embeddings['card'].shape}"
                f"  merchant {embeddings['merchant'].shape}")
    return trainer, embeddings

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – AE 无监督异常分数
# ══════════════════════════════════════════════════════════════════════════════
def run_ae(data, device, epochs):
    logger.info("━"*60)
    logger.info("Step 2  图自编码器（无监督异常分数）")
    logger.info("━"*60)

    cd = data["card"].x.shape[1]
    md = data["merchant"].x.shape[1]
    ed = data["card","transacts","merchant"].edge_attr.shape[1] \
         if hasattr(data["card","transacts","merchant"], "edge_attr") else 0

    ae_model  = create_model("autoencoder", card_in_channels=cd,
                             merchant_in_channels=md, edge_in_channels=ed,
                             hidden_channels=256, out_channels=128,
                             num_layers=3, dropout=0.4)
    ae_trainer = UnsupervisedTrainer(ae_model, device=device, lr=0.001)

    for ep in range(epochs):
        loss, _ = ae_trainer.train_epoch(data)
        if ep % max(1, epochs//5) == 0 or ep == epochs-1:
            logger.info(f"  AE epoch {ep:3d}/{epochs}  loss={loss:.4f}")

    # 全图所有边的异常分数（N_edges,）
    scores = ae_trainer.compute_anomaly_scores(data).numpy()
    logger.info(f"  AE scores  mean={scores.mean():.4f}  "
                f"std={scores.std():.4f}  max={scores.max():.4f}")

    np.save(os.path.join(OUT_DIR, "ae_anomaly_scores.npy"), scores)
    logger.info(f"  → {OUT_DIR}/ae_anomaly_scores.npy")
    return scores          # shape (N_edges,)

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – 社群检测（每个 card 节点的风险分）
# ══════════════════════════════════════════════════════════════════════════════
def run_community(data, embeddings, edges_df):
    logger.info("━"*60)
    logger.info("Step 3  社群检测（每条边 → card 节点风险分）")
    logger.info("━"*60)

    edge_index = data["card","transacts","merchant"].edge_index
    edge_label = data["card","transacts","merchant"].edge_label

    detector = CommunityDetector()
    detector.build_networkx_graph(edge_index, data.card_mapping,
                                  data.merchant_mapping)

    # Louvain 优先，不可用则 KMeans
    try:
        detector.detect_louvain(resolution=1.5)
        method = "louvain"
    except ImportError:
        logger.warning("python-louvain 不可用，改用 KMeans")
        detector.detect_embedding_cluster(embeddings["card"],
                                          embeddings["merchant"],
                                          method="kmeans", n_clusters=100)
        method = "kmeans"
    logger.info(f"  检测方法={method}  社群数={len(detector.communities)}")

    edge_times   = (torch.tensor(edges_df["hour"].values.astype(float))
                    if "hour"           in edges_df.columns else None)
    edge_amounts = (torch.tensor(edges_df["TransactionAmt"].values.astype(float))
                    if "TransactionAmt" in edges_df.columns else None)

    detector.compute_community_stats(edge_label, edge_index,
                                     edge_times=edge_times,
                                     edge_amounts=edge_amounts)
    detector.identify_suspicious_communities(fraud_rate_threshold=0.02,
                                             min_transactions=100, min_cards=10)
    comm_metrics = detector.evaluate_community_detection(edge_label, edge_index)
    detector.print_top_suspicious(top_k=5)

    # card_node_str -> risk_score（对所有社群，不仅限于可疑社群）
    node_to_comm = {}
    for cid, nodes in detector.communities.items():
        for n in nodes:
            node_to_comm[n] = cid

    node_risk = {}
    for n, cid in node_to_comm.items():
        node_risk[n] = detector.compute_risk_score(cid)

    _save({"method": method,
           "total_communities": len(detector.communities),
           "suspicious_communities": len(detector.suspicious_communities),
           "community_metrics": comm_metrics,
           "top_suspicious": [{k: v for k, v in s.items() if k != "nodes"}
                               for s in detector.suspicious_communities[:20]]},
          "community_analysis.json")

    logger.info(f"  社群  Precision={comm_metrics['community_precision']:.4f}"
                f"  Recall={comm_metrics['community_recall']:.4f}"
                f"  F1={comm_metrics['community_f1']:.4f}")
    return node_risk, comm_metrics

# ══════════════════════════════════════════════════════════════════════════════
# 特征构造
# ══════════════════════════════════════════════════════════════════════════════
def _base_features(data, embeddings, mask):
    """GraphSAGE 嵌入特征（Model A 的全部特征）"""
    ei         = data["card","transacts","merchant"].edge_index
    ea         = data["card","transacts","merchant"].edge_attr
    edge_label = data["card","transacts","merchant"].edge_label

    idx  = mask.nonzero(as_tuple=True)[0]
    cemb = embeddings["card"].cpu().numpy()
    memb = embeddings["merchant"].cpu().numpy()

    cf = cemb[ei[0, idx].cpu().numpy()]
    mf = memb[ei[1, idx].cpu().numpy()]
    ef = ea[idx].cpu().numpy()

    X = np.concatenate([cf, mf, cf - mf, cf * mf, ef], axis=1)
    y = edge_label[idx].cpu().numpy()

    feat_names = (
        [f"card_emb_{i}"     for i in range(cf.shape[1])] +
        [f"merchant_emb_{i}" for i in range(mf.shape[1])] +
        [f"emb_diff_{i}"     for i in range(cf.shape[1])] +
        [f"emb_prod_{i}"     for i in range(cf.shape[1])] +
        [f"edge_feat_{i}"    for i in range(ef.shape[1])]
    )
    return X, y, idx.numpy(), feat_names

def _extra_features(idx_arr, ae_scores, node_risk, edge_index_np):
    """AE 异常分数 + 社群风险分（归一化）"""
    ae_vals = ae_scores[idx_arr]
    ae_min, ae_max = ae_scores.min(), ae_scores.max()
    ae_norm = (ae_vals - ae_min) / (ae_max - ae_min + 1e-8)

    comm_vals = np.array([
        node_risk.get(f"card_{edge_index_np[0, i]}", 0.0)
        for i in idx_arr
    ], dtype=np.float32)

    return np.stack([ae_norm, comm_vals], axis=1), ["ae_anomaly_score", "community_risk"]

# ══════════════════════════════════════════════════════════════════════════════
# LightGBM 训练 + 评估
# ══════════════════════════════════════════════════════════════════════════════
def _train_lgb(X_tr, y_tr, X_va, y_va, feat_names, label):
    if not HAS_LGB:
        raise ImportError("pip install lightgbm")
    logger.info(f"  [{label}] 训练 LightGBM  特征数={X_tr.shape[1]}")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0))
    X_va = scaler.transform(np.nan_to_num(X_va, nan=0, posinf=0, neginf=0))

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    return model, scaler

def _eval_lgb(model, scaler, X, y, feat_names, label):
    X  = scaler.transform(np.nan_to_num(X, nan=0, posinf=0, neginf=0))
    pr = model.predict(X)
    m  = _metrics(y, pr)
    logger.info(f"  [{label}] Test  AUC={m['auc']:.4f}  AP={m['ap']:.4f}"
                f"  F1={m['f1']:.4f}  Prec={m['precision']:.4f}"
                f"  Recall={m['recall']:.4f}")

    imp = pd.DataFrame({"feature": feat_names,
                        "importance": model.feature_importance("gain")
                        }).sort_values("importance", ascending=False)
    return m, pr, imp


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="基线 vs 增强混合模型对比")
    p.add_argument("--gnn_epochs",     type=int, default=50,
                   help="GNN 训练轮数（默认 50）")
    p.add_argument("--ae_epochs",      type=int, default=50,
                   help="AE  训练轮数（默认 50）")
    p.add_argument("--device",         type=str, default="auto",
                   help="cpu / cuda / auto")
    p.add_argument("--skip_gnn_train", action="store_true",
                   help="复用 outputs/models/graphsage_fraud_detector.pt，跳过 GNN 训练")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args   = parse_args()
    device = _device(args.device)
    t0     = time.time()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)
    logger.add(os.path.join(OUT_DIR, "logs", "run_{time}.log"),
               rotation="10 MB", level="INFO")

    print(f"\n{'='*62}")
    print("  对比实验：GraphSAGE+LightGBM  vs  +AE+社群检测")
    print(f"  GNN epochs : {args.gnn_epochs}")
    print(f"  AE  epochs : {args.ae_epochs}")
    print(f"  Device     : {device}")
    print(f"  Output     : {OUT_DIR}")
    print(f"{'='*62}\n")

    # ── 公共数据 ──────────────────────────────────────────────────────────────
    data     = _load_graph()
    edges_df = _load_edges_df()
    ei_np    = data["card", "transacts", "merchant"].edge_index.numpy()  # (2, N)

    # ── Step 1: GNN（两个模型共用） ───────────────────────────────────────────
    _, embeddings = run_gnn(data, device, args.gnn_epochs, args.skip_gnn_train)

    # ── Step 2: AE 异常分 ─────────────────────────────────────────────────────
    ae_scores = run_ae(data, device, args.ae_epochs)   # (N_edges,)

    # ── Step 3: 社群风险分 ────────────────────────────────────────────────────
    node_risk, comm_metrics = run_community(data, embeddings, edges_df)

    # ── 特征构造 ──────────────────────────────────────────────────────────────
    logger.info("━"*60)
    logger.info("Step 4  构造特征 & 训练两个 LightGBM")
    logger.info("━"*60)

    X_tr_base, y_tr, tr_idx, base_names = _base_features(data, embeddings, data.train_mask)
    X_va_base, y_va, va_idx, _          = _base_features(data, embeddings, data.val_mask)
    X_te_base, y_te, te_idx, _          = _base_features(data, embeddings, data.test_mask)

    ex_tr, extra_names = _extra_features(tr_idx, ae_scores, node_risk, ei_np)
    ex_va, _           = _extra_features(va_idx, ae_scores, node_risk, ei_np)
    ex_te, _           = _extra_features(te_idx, ae_scores, node_risk, ei_np)

    X_tr_enh = np.concatenate([X_tr_base, ex_tr], axis=1)
    X_va_enh = np.concatenate([X_va_base, ex_va], axis=1)
    X_te_enh = np.concatenate([X_te_base, ex_te], axis=1)

    all_names_base = base_names
    all_names_enh  = base_names + extra_names

    logger.info(f"  Model A 特征维度: {X_tr_base.shape[1]}")
    logger.info(f"  Model B 特征维度: {X_tr_enh.shape[1]}"
                f"  (+{len(extra_names)} 新增: {extra_names})")

    # ── Model A: 基线 GraphSAGE + LightGBM ───────────────────────────────────
    logger.info("── Model A（基线）──")
    lgb_a, scaler_a = _train_lgb(X_tr_base, y_tr, X_va_base, y_va,
                                  all_names_base, "Model-A")
    metrics_a, probs_a, imp_a = _eval_lgb(lgb_a, scaler_a,
                                           X_te_base, y_te, all_names_base, "Model-A")

    # ── Model B: 增强 GraphSAGE + LightGBM + AE + 社群 ───────────────────────
    logger.info("── Model B（增强）──")
    lgb_b, scaler_b = _train_lgb(X_tr_enh, y_tr, X_va_enh, y_va,
                                  all_names_enh, "Model-B")
    metrics_b, probs_b, imp_b = _eval_lgb(lgb_b, scaler_b,
                                           X_te_enh, y_te, all_names_enh, "Model-B")

    # ── 保存特征重要性 ─────────────────────────────────────────────────────────
    imp_a.to_csv(os.path.join(OUT_DIR, "feature_importance_A.csv"), index=False)
    imp_b.to_csv(os.path.join(OUT_DIR, "feature_importance_B.csv"), index=False)
    logger.info(f"  → feature_importance_A/B.csv")

    # 增强模型中 AE / 社群特征的排名
    for name in extra_names:
        row = imp_b[imp_b["feature"] == name]
        if not row.empty:
            rank = imp_b.index.get_loc(row.index[0]) + 1
            logger.info(f"  [{name}] 在 Model-B 中重要性排名: {rank}/{len(imp_b)}"
                        f"  gain={row['importance'].values[0]:.2f}")

    # ── 每条测试边的预测 CSV ──────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "edge_idx":   te_idx,
        "label":      y_te.astype(int),
        "prob_A":     np.round(probs_a, 6),
        "pred_A":     (probs_a >= 0.5).astype(int),
        "prob_B":     np.round(probs_b, 6),
        "pred_B":     (probs_b >= 0.5).astype(int),
        "ae_score":   np.round(ex_te[:, 0], 6),
        "comm_risk":  np.round(ex_te[:, 1], 6),
    })
    pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)
    logger.info(f"  → test_predictions.csv")

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)

    def _delta(key):
        return round(metrics_b[key] - metrics_a[key], 4)

    report = {
        "timestamp":      datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "test_samples":   int(len(y_te)),
        "test_fraud_ratio": float(y_te.mean()),
        "model_A_baseline": {
            "description": "GraphSAGE + LightGBM",
            "feature_dim": int(X_tr_base.shape[1]),
            **metrics_a
        },
        "model_B_enhanced": {
            "description": "GraphSAGE + LightGBM + AE anomaly + Community risk",
            "feature_dim": int(X_tr_enh.shape[1]),
            "extra_features": extra_names,
            **metrics_b
        },
        "delta_B_minus_A": {k: _delta(k) for k in ("auc","ap","f1","precision","recall")},
        "community_module": comm_metrics,
    }
    _save(report, "comparison_report.json")

    # ── 控制台对比表 ──────────────────────────────────────────────────────────
    SEP = "─" * 68
    print(f"\n{'对比结果（测试集）':^68}")
    print(SEP)
    print(f"{'Model':<38}{'AUC':>7}{'AP':>7}{'F1':>7}{'Prec':>7}{'Recall':>7}")
    print(SEP)
    for tag, m, desc in [
        ("A 基线", metrics_a, "GraphSAGE + LightGBM"),
        ("B 增强", metrics_b, "GraphSAGE + LightGBM + AE + 社群"),
    ]:
        print(f"  {tag}  {desc:<32}"
              f"{m['auc']:>7.4f}{m['ap']:>7.4f}{m['f1']:>7.4f}"
              f"{m['precision']:>7.4f}{m['recall']:>7.4f}")
    print(SEP)
    deltas = report["delta_B_minus_A"]
    print(f"  {'Delta (B−A)':<36}"
          f"{deltas['auc']:>+7.4f}{deltas['ap']:>+7.4f}{deltas['f1']:>+7.4f}"
          f"{deltas['precision']:>+7.4f}{deltas['recall']:>+7.4f}")
    print(SEP)
    print(f"\n  输出目录: {OUT_DIR}\n")


if __name__ == "__main__":
    main()
