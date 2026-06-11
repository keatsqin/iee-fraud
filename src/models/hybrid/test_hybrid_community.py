"""
测试社群特征融合模块 (hybrid_com.py)
覆盖：
  - 模型初始化
  - 社群特征提取 / 社群风险分
  - prepare_features_with_community
  - train / predict_with_community / evaluate
  - get_feature_importance
  - save / load
  - load_community_data（真实文件存在时）
  - 顶层函数 train_hybrid_model_with_community（mock 集成）
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.models.hybrid_com import (
    HybridFraudDetectorWithCommunity,
    load_community_data,
    train_hybrid_model_with_community,
)

# ──────────────────────────────────────────────────────────────────────────────
# 公共 Mock 数据
# ──────────────────────────────────────────────────────────────────────────────

# 社群统计（2 个社群）
MOCK_COMMUNITY_STATS = {
    0: {
        "fraud_rate": 0.6,
        "density": 0.4,
        "time_concentration": 0.8,
        "night_ratio": 0.3,
        "num_cards": 10,
        "num_merchants": 5,
        "total_transactions": 50,
        "fraud_transactions": 30,
    },
    1: {
        "fraud_rate": 0.1,
        "density": 0.2,
        "time_concentration": 0.3,
        "night_ratio": 0.1,
        "num_cards": 20,
        "num_merchants": 10,
        "total_transactions": 100,
        "fraud_transactions": 10,
    },
}

# card_0, card_1 在社群 0；card_3 在社群 1；card_2 不在任何社群
MOCK_NODE_TO_COMMUNITY = {
    "card_0": 0,
    "card_1": 0,
    "card_3": 1,
}


def _make_mock_data(n_edges: int = 20, n_card: int = 5, n_merchant: int = 5,
                    emb_dim: int = 8, n_edge_feat: int = 4):
    """
    构造一个模拟 PyG HeteroData 对象，包含 card/merchant 节点及 transacts 边。
    返回 (data, card_emb_tensor, merchant_emb_tensor, train_mask, val_mask, test_mask)
    """
    data = MagicMock()

    # 边索引：card -> merchant，随机映射
    rng = np.random.default_rng(42)
    src = torch.tensor(rng.integers(0, n_card, n_edges), dtype=torch.long)
    dst = torch.tensor(rng.integers(0, n_merchant, n_edges), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)  # shape (2, n_edges)

    edge_attr = torch.tensor(
        rng.random((n_edges, n_edge_feat)).astype(np.float32)
    )
    # 二分类标签
    labels = torch.tensor(
        rng.integers(0, 2, n_edges).astype(np.float32)
    )

    rel = data.__getitem__.return_value
    rel.edge_index = edge_index
    rel.edge_attr = edge_attr
    rel.edge_label = labels

    # 让 data["card", "transacts", "merchant"] 返回 rel
    data.__getitem__ = lambda _self, _key: rel

    # 节点嵌入
    card_emb = torch.tensor(
        rng.random((n_card, emb_dim)).astype(np.float32)
    )
    merchant_emb = torch.tensor(
        rng.random((n_merchant, emb_dim)).astype(np.float32)
    )
    embeddings = {"card": card_emb, "merchant": merchant_emb}

    # 掩码：前 12 训练，12-16 验证，16-20 测试
    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask[:12] = True
    val_mask = torch.zeros(n_edges, dtype=torch.bool)
    val_mask[12:16] = True
    test_mask = torch.zeros(n_edges, dtype=torch.bool)
    test_mask[16:] = True

    return data, embeddings, train_mask, val_mask, test_mask


# ──────────────────────────────────────────────────────────────────────────────
# 测试函数
# ──────────────────────────────────────────────────────────────────────────────

def test_model_initialization():
    """测试模型初始化及参数正确性"""
    detector = HybridFraudDetectorWithCommunity(alpha=0.7)
    assert detector.ALPHA == 0.7
    assert detector.model is None
    assert detector.feature_names is None
    assert "objective" in detector.lgb_params
    assert detector.lgb_params["objective"] == "binary"

    # 自定义 alpha
    det2 = HybridFraudDetectorWithCommunity(alpha=0.5)
    assert det2.ALPHA == 0.5

    logger.info("✅ test_model_initialization passed")
    return True


def test_extract_community_features():
    """测试 _extract_community_features 的维度与边界条件"""
    detector = HybridFraudDetectorWithCommunity(alpha=0.7)

    # card_0, card_1: 在社群0；card_2: 不在任何社群；card_3: 在社群1
    card_indices = np.array([0, 1, 2, 3])
    features = detector._extract_community_features(
        card_indices, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY
    )

    assert features.shape == (4, 6), f"期望 (4, 6)，实际 {features.shape}"
    assert features.dtype == np.float32

    # card_0 / card_1 应与社群 0 的 fraud_rate 一致
    assert math.isclose(features[0, 0], 0.6, abs_tol=1e-5)
    assert math.isclose(features[1, 0], 0.6, abs_tol=1e-5)

    # card_2 不在社群，特征全为 0
    np.testing.assert_array_equal(features[2], np.zeros(6, dtype=np.float32))

    # card_3 属于社群 1
    assert math.isclose(features[3, 0], 0.1, abs_tol=1e-5)

    # log_txn_norm 截断到 1
    assert features[0, 5] <= 1.0
    assert features[3, 5] <= 1.0

    logger.info(f"✅ test_extract_community_features passed. features:\n{features}")
    return True


def test_community_risk_score():
    """测试 _community_risk_score 的返回值"""
    detector = HybridFraudDetectorWithCommunity(alpha=0.7)

    card_indices = np.array([0, 2, 3])  # card_0(comm0), card_2(无社群), card_3(comm1)
    scores = detector._community_risk_score(
        card_indices, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY
    )

    assert scores.shape == (3,)
    assert math.isclose(scores[0], 0.6, abs_tol=1e-5)   # 社群 0 fraud_rate
    assert math.isclose(scores[1], 0.05, abs_tol=1e-5)  # 默认值
    assert math.isclose(scores[2], 0.1, abs_tol=1e-5)   # 社群 1 fraud_rate

    logger.info(f"✅ test_community_risk_score passed. scores: {scores}")
    return True


def test_prepare_features_with_community():
    """测试特征矩阵构建（含社群特征拼接）"""
    data, embeddings, train_mask, _, _ = _make_mock_data(
        n_edges=20, n_card=5, n_merchant=5, emb_dim=8, n_edge_feat=4
    )
    detector = HybridFraudDetectorWithCommunity(alpha=0.7)

    X, y = detector.prepare_features_with_community(
        data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY, train_mask
    )

    # emb_dim=8：card + merchant + diff + prod = 32; edge=4; community=6 → 42
    expected_feat_dim = 8 * 4 + 4 + 6
    assert X.shape[1] == expected_feat_dim, (
        f"期望 {expected_feat_dim} 维特征，实际 {X.shape[1]}"
    )
    assert X.shape[0] == train_mask.sum().item()
    assert y.shape[0] == train_mask.sum().item()

    # feature_names 应在首次调用后被设置
    assert detector.feature_names is not None
    assert len(detector.feature_names) == expected_feat_dim
    assert "comm_0" in detector.feature_names

    logger.info(f"✅ test_prepare_features_with_community passed. X.shape={X.shape}")
    return True


def test_train_and_predict():
    """测试完整训练 + predict_with_community 流程"""
    data, embeddings, train_mask, val_mask, test_mask = _make_mock_data(
        n_edges=40, n_card=5, n_merchant=5, emb_dim=8, n_edge_feat=4
    )
    # 调整掩码：更多训练样本
    train_mask = torch.zeros(40, dtype=torch.bool)
    train_mask[:24] = True
    val_mask = torch.zeros(40, dtype=torch.bool)
    val_mask[24:32] = True
    test_mask = torch.zeros(40, dtype=torch.bool)
    test_mask[32:] = True

    detector = HybridFraudDetectorWithCommunity(alpha=0.7)
    train_results = detector.train(
        data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY,
        train_mask, val_mask,
        num_boost_round=50,      # 快速测试
        early_stopping_rounds=10,
    )

    assert "val_auc" in train_results
    assert "val_ap" in train_results
    assert "best_iteration" in train_results
    assert 0.0 <= train_results["val_auc"] <= 1.0

    # predict_with_community
    lgb_prob, fused_prob = detector.predict_with_community(
        data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY, test_mask
    )
    n_test = test_mask.sum().item()
    assert lgb_prob.shape == (n_test,)
    assert fused_prob.shape == (n_test,)
    assert np.all(lgb_prob >= 0) and np.all(lgb_prob <= 1)
    assert np.all(fused_prob >= 0) and np.all(fused_prob <= 1)

    # fused = α * lgb + (1-α) * comm_risk，验证融合公式
    edge_index = data["card", "transacts", "merchant"].edge_index
    mask_indices = test_mask.nonzero(as_tuple=True)[0]
    card_indices = edge_index[0, mask_indices].cpu().numpy()
    comm_risk = detector._community_risk_score(
        card_indices, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY
    )
    expected_fused = 0.7 * lgb_prob + 0.3 * comm_risk
    np.testing.assert_allclose(fused_prob, expected_fused, rtol=1e-5)

    logger.info(f"✅ test_train_and_predict passed. val_auc={train_results['val_auc']:.4f}")
    return True


def test_evaluate():
    """测试 evaluate 的结构和字段完整性"""
    data, embeddings, train_mask, val_mask, test_mask = _make_mock_data(
        n_edges=40, n_card=5, n_merchant=5, emb_dim=8, n_edge_feat=4
    )
    train_mask = torch.zeros(40, dtype=torch.bool)
    train_mask[:24] = True
    val_mask = torch.zeros(40, dtype=torch.bool)
    val_mask[24:32] = True
    test_mask = torch.zeros(40, dtype=torch.bool)
    test_mask[32:] = True

    detector = HybridFraudDetectorWithCommunity(alpha=0.7)
    detector.train(
        data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY,
        train_mask, val_mask,
        num_boost_round=50, early_stopping_rounds=10,
    )

    results = detector.evaluate(
        data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY,
        test_mask, threshold=0.5
    )

    # 顶层键
    for key in ("lgb_only", "fused", "labels", "alpha", "edge_output"):
        assert key in results, f"缺少键: {key}"

    # 每个 metrics dict 应包含完整字段
    for section in ("lgb_only", "fused"):
        m = results[section]
        for field in ("auc", "ap", "precision", "recall", "f1",
                      "confusion_matrix", "probabilities", "predictions"):
            assert field in m, f"{section} 缺少字段: {field}"
        assert 0.0 <= m["auc"] <= 1.0
        assert m["confusion_matrix"].shape == (2, 2)

    assert math.isclose(results["alpha"], 0.7, abs_tol=1e-6)

    # edge_output 每条记录应含 fused_probability
    n_test = test_mask.sum().item()
    assert len(results["edge_output"]) == n_test
    for rec in results["edge_output"]:
        for field in ("edge_idx", "card_node", "lgb_probability",
                      "community_risk_score", "fused_probability", "label"):
            assert field in rec, f"edge_output 记录缺少字段: {field}"
        assert 0.0 <= rec["fused_probability"] <= 1.0

    logger.info("✅ test_evaluate passed")
    return True


def test_get_feature_importance():
    """测试特征重要性提取"""
    data, embeddings, train_mask, val_mask, _ = _make_mock_data(
        n_edges=40, n_card=5, n_merchant=5, emb_dim=8, n_edge_feat=4
    )
    train_mask = torch.zeros(40, dtype=torch.bool)
    train_mask[:24] = True
    val_mask = torch.zeros(40, dtype=torch.bool)
    val_mask[24:32] = True

    detector = HybridFraudDetectorWithCommunity(alpha=0.7)

    # 未训练时返回 None
    assert detector.get_feature_importance() is None

    detector.train(
        data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY,
        train_mask, val_mask,
        num_boost_round=50, early_stopping_rounds=10,
    )

    df = detector.get_feature_importance(top_k=10)
    assert df is not None
    assert len(df) <= 10
    assert "feature" in df.columns and "importance" in df.columns
    # 按 gain 降序
    assert df["importance"].is_monotonic_decreasing or df.shape[0] == 1

    # comm_* 特征应出现在列表中
    all_features = detector.feature_names
    comm_feats = [f for f in all_features if f.startswith("comm_")]
    assert len(comm_feats) == 6

    logger.info(f"✅ test_get_feature_importance passed. top feature: {df.iloc[0]['feature']}")
    return True


def test_save_and_load():
    """测试模型持久化与恢复"""
    data, embeddings, train_mask, val_mask, test_mask = _make_mock_data(
        n_edges=40, n_card=5, n_merchant=5, emb_dim=8, n_edge_feat=4
    )
    train_mask = torch.zeros(40, dtype=torch.bool)
    train_mask[:24] = True
    val_mask = torch.zeros(40, dtype=torch.bool)
    val_mask[24:32] = True
    test_mask = torch.zeros(40, dtype=torch.bool)
    test_mask[32:] = True

    detector = HybridFraudDetectorWithCommunity(alpha=0.65)
    detector.train(
        data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY,
        train_mask, val_mask,
        num_boost_round=50, early_stopping_rounds=10,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "subdir", "test_model.pkl")
        detector.save(model_path)
        assert os.path.isfile(model_path)

        # 恢复
        loaded = HybridFraudDetectorWithCommunity()
        loaded.load(model_path)
        assert math.isclose(loaded.ALPHA, 0.65, abs_tol=1e-6)
        assert loaded.model is not None
        assert loaded.feature_names == detector.feature_names

        # 恢复后预测结果应一致
        lgb1, fused1 = detector.predict_with_community(
            data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY, test_mask
        )
        lgb2, fused2 = loaded.predict_with_community(
            data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY, test_mask
        )
        np.testing.assert_allclose(lgb1, lgb2, rtol=1e-5)
        np.testing.assert_allclose(fused1, fused2, rtol=1e-5)

    logger.info("✅ test_save_and_load passed")
    return True


def test_load_community_data_from_file():
    """测试从真实 JSON 文件加载社群数据"""
    community_json = Path("outputs/communities/suspicious_communities.json")

    if not community_json.exists():
        logger.warning(
            f"社群文件不存在，跳过此测试: {community_json}\n"
            "请先运行: python main.py --data_dir ../ieee-fraud-detection --output_dir outputs"
        )
        return None  # 跳过

    stats, mapping = load_community_data(str(community_json))

    assert isinstance(stats, dict)
    assert isinstance(mapping, dict)
    assert len(stats) > 0
    assert len(mapping) > 0

    # 每个 stats 条目应包含必需字段
    for cid, s in stats.items():
        assert isinstance(cid, int)
        for field in ("fraud_rate", "density", "num_cards", "num_merchants",
                      "total_transactions"):
            assert field in s, f"社群 {cid} 缺少字段: {field}"

    # 节点名格式检查
    sample_nodes = list(mapping.keys())[:5]
    for node in sample_nodes:
        assert isinstance(node, str)

    logger.info(
        f"✅ test_load_community_data_from_file passed. "
        f"{len(stats)} 社群, {len(mapping)} 节点映射"
    )
    for cid in list(stats.keys())[:3]:
        logger.info(f"  社群 {cid}: {stats[cid]}")
    return True


def test_load_community_data_from_mock_json():
    """测试 load_community_data 解析逻辑（写临时 JSON）"""
    raw = {
        "stats": {
            "0": {
                "fraud_rate": 0.5,
                "density": 0.3,
                "time_concentration": 0.7,
                "night_ratio": 0.2,
                "num_cards": 8,
                "num_merchants": 4,
                "total_transactions": 40,
                "fraud_transactions": "20",  # 字符串格式
            },
            "1": {
                "fraud_rate": 0.1,
                "density": 0.1,
                "num_cards": 5,
                "num_merchants": 5,
                "total_transactions": 20,
                "fraud_transactions": 2,
            },
        },
        "communities": {
            "0": ["card_0", "card_1", "merchant_0"],
            "1": ["card_2"],
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(raw, f)
        tmp_path = f.name

    try:
        stats, mapping = load_community_data(tmp_path)

        assert 0 in stats and 1 in stats
        # fraud_transactions 字符串应被转换为 int
        assert isinstance(stats[0]["fraud_transactions"], int)
        assert stats[0]["fraud_transactions"] == 20

        assert mapping["card_0"] == 0
        assert mapping["card_1"] == 0
        assert mapping["card_2"] == 1
        assert mapping["merchant_0"] == 0
    finally:
        os.unlink(tmp_path)

    logger.info("✅ test_load_community_data_from_mock_json passed")
    return True


def test_alpha_fusion_boundary():
    """测试 α=0 和 α=1 时融合分退化为纯社群/纯 LGB"""
    data, embeddings, train_mask, val_mask, test_mask = _make_mock_data(
        n_edges=40, n_card=5, n_merchant=5, emb_dim=8, n_edge_feat=4
    )
    train_mask = torch.zeros(40, dtype=torch.bool)
    train_mask[:24] = True
    val_mask = torch.zeros(40, dtype=torch.bool)
    val_mask[24:32] = True
    test_mask = torch.zeros(40, dtype=torch.bool)
    test_mask[32:] = True

    for alpha in (0.0, 1.0):
        det = HybridFraudDetectorWithCommunity(alpha=alpha)
        det.train(
            data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY,
            train_mask, val_mask,
            num_boost_round=30, early_stopping_rounds=10,
        )
        lgb_prob, fused_prob = det.predict_with_community(
            data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY, test_mask
        )

        edge_index = data["card", "transacts", "merchant"].edge_index
        mask_indices = test_mask.nonzero(as_tuple=True)[0]
        card_indices = edge_index[0, mask_indices].cpu().numpy()
        comm_risk = det._community_risk_score(
            card_indices, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY
        )

        if alpha == 1.0:
            np.testing.assert_allclose(fused_prob, lgb_prob, rtol=1e-5,
                                       err_msg="α=1 时 fused 应等于 lgb_prob")
        else:  # alpha == 0.0
            np.testing.assert_allclose(fused_prob, comm_risk, rtol=1e-5,
                                       err_msg="α=0 时 fused 应等于 community_risk")

    logger.info("✅ test_alpha_fusion_boundary passed")
    return True


def test_feature_names_not_duplicated():
    """多次调用 prepare_features_with_community 不应追加重复的 feature_names"""
    data, embeddings, train_mask, val_mask, _ = _make_mock_data()
    detector = HybridFraudDetectorWithCommunity(alpha=0.7)

    detector.prepare_features_with_community(
        data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY, train_mask
    )
    first_names = list(detector.feature_names)

    detector.prepare_features_with_community(
        data, embeddings, MOCK_COMMUNITY_STATS, MOCK_NODE_TO_COMMUNITY, val_mask
    )
    second_names = list(detector.feature_names)

    assert first_names == second_names, "多次调用后 feature_names 不应改变"
    logger.info("✅ test_feature_names_not_duplicated passed")
    return True


def test_train_hybrid_model_with_community_integration():
    """
    集成测试：train_hybrid_model_with_community 顶层函数。
    使用 mock GNN trainer + 临时社群 JSON 文件。
    """
    data, embeddings, train_mask, val_mask, test_mask = _make_mock_data(
        n_edges=60, n_card=5, n_merchant=5, emb_dim=8, n_edge_feat=4
    )
    train_mask = torch.zeros(60, dtype=torch.bool)
    train_mask[:36] = True
    val_mask = torch.zeros(60, dtype=torch.bool)
    val_mask[36:48] = True
    test_mask = torch.zeros(60, dtype=torch.bool)
    test_mask[48:] = True

    # 构造 mock GNN trainer
    gnn_trainer = MagicMock()
    gnn_trainer.get_embeddings.return_value = embeddings

    # 写临时社群 JSON
    raw = {
        "stats": {str(k): v for k, v in MOCK_COMMUNITY_STATS.items()},
        "communities": {
            "0": ["card_0", "card_1"],
            "1": ["card_3"],
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        community_json = os.path.join(tmpdir, "suspicious_communities.json")
        with open(community_json, "w", encoding="utf-8") as f:
            json.dump(raw, f)

        save_path = os.path.join(tmpdir, "hybrid_comm_model.pkl")

        hybrid, results = train_hybrid_model_with_community(
            data=data,
            gnn_trainer=gnn_trainer,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            community_json_path=community_json,
            baseline_results_path=None,
            save_path=save_path,
            alpha=0.7,
        )

        # 返回值检查
        assert isinstance(hybrid, HybridFraudDetectorWithCommunity)
        for key in ("train", "lgb_only", "fused", "ablation", "feature_importance",
                    "edge_output"):
            assert key in results, f"顶层结果缺少键: {key}"

        assert os.path.isfile(save_path), "模型文件未保存"
        results_json_path = save_path.replace(".pkl", "_results.json")
        assert os.path.isfile(results_json_path), "结果 JSON 未保存"

        edge_output_path = save_path.replace(".pkl", "_edge_output.json")
        assert os.path.isfile(edge_output_path), "边级输出 JSON 未保存"

        # 验证结果 JSON 结构
        with open(results_json_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["model"] == "hybrid_gnn_lgb_community"
        assert math.isclose(saved["alpha"], 0.7, abs_tol=1e-6)
        for section in ("test_lgb_only", "test_fused"):
            for metric in ("auc", "ap", "precision", "recall", "f1"):
                assert metric in saved[section]

    logger.info("✅ test_train_hybrid_model_with_community_integration passed")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 完整训练并保存到项目目录（使用真实数据）
# ──────────────────────────────────────────────────────────────────────────────

def run_full_training_and_save(
    graph_path: str = "outputs/graph/fraud_graph.pt",
    embeddings_path: str = "outputs/embeddings/node_embeddings.pt",
    community_json_path: str = "outputs/communities/suspicious_communities.json",
    gnn_model_path: str = "outputs/models/graphsage_fraud_detector.pt",
    baseline_results_path: str = "outputs/models/hybrid_gnn_lgb_results.json",
    save_path: str = "outputs/models/hybrid_gnn_lgb_community.pkl",
    alpha: float = 0.7,
) -> bool:
    """
    使用真实数据完整训练 HybridFraudDetectorWithCommunity 并保存产物到项目目录。

    产物：
      outputs/models/hybrid_gnn_lgb_community.pkl         — 模型权重
      outputs/models/hybrid_gnn_lgb_community_results.json — 评估指标 + 消融对比
      outputs/models/hybrid_gnn_lgb_community_edge_output.json — 边级 fused_probability

    Prerequisites（以下文件需提前存在）：
      outputs/graph/fraud_graph.pt
      outputs/embeddings/node_embeddings.pt
      outputs/communities/suspicious_communities.json
      outputs/models/graphsage_fraud_detector.pt
    """
    import torch

    # ── 1. 检查前置文件 ────────────────────────────────────────────────────────
    for p in (graph_path, embeddings_path, community_json_path, gnn_model_path):
        if not Path(p).exists():
            logger.error(f"前置文件不存在，跳过完整训练: {p}")
            logger.info("请先运行: python main.py --data_dir ../ieee-fraud-detection --output_dir outputs")
            return None  # 跳过

    logger.info("=" * 60)
    logger.info("完整训练 HybridFraudDetectorWithCommunity（真实数据）")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"设备: {device}")

    # ── 2. 加载图数据（HeteroData，含 train_mask / val_mask / test_mask）────────
    logger.info(f"加载图数据: {graph_path}")
    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    # 从 data 上读取已划分好的掩码
    train_mask = data.train_mask
    val_mask   = data.val_mask
    test_mask  = data.test_mask
    logger.info(
        f"边划分 — 训练: {train_mask.sum().item()}, "
        f"验证: {val_mask.sum().item()}, "
        f"测试: {test_mask.sum().item()}"
    )

    # ── 3. 加载预训练 GNN 嵌入（直接用保存的嵌入，无需重新推理）─────────────────
    logger.info(f"加载节点嵌入: {embeddings_path}")
    embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    logger.info(
        f"嵌入维度 — card: {embeddings['card'].shape}, "
        f"merchant: {embeddings['merchant'].shape}"
    )

    # ── 4. 构造 gnn_trainer wrapper（仅需 get_embeddings 接口）─────────────────
    # 嵌入已预计算，直接用一个轻量 wrapper 返回，无需加载完整 GNN 模型
    class _EmbeddingWrapper:
        def __init__(self, emb):
            self._emb = emb
        def get_embeddings(self, *_):
            return self._emb

    gnn_trainer = _EmbeddingWrapper(embeddings)

    # ── 5. 调用顶层训练函数 ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    hybrid, results = train_hybrid_model_with_community(
        data=data,
        gnn_trainer=gnn_trainer,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        community_json_path=community_json_path,
        baseline_results_path=baseline_results_path if Path(baseline_results_path).exists() else None,
        save_path=save_path,
        alpha=alpha,
    )

    # ── 6. 汇报结果 ────────────────────────────────────────────────────────────
    lgb_res   = results["lgb_only"]
    fused_res = results["fused"]

    logger.info("=" * 60)
    logger.info("训练完成 — 测试集指标汇总")
    logger.info(f"  {'指标':<10} {'LGB-only':>10} {'Fused':>10}")
    for metric in ("auc", "ap", "precision", "recall", "f1"):
        logger.info(
            f"  {metric:<10} {lgb_res[metric]:>10.4f} {fused_res[metric]:>10.4f}"
        )
    logger.info(f"  模型已保存: {save_path}")

    # ── 7. 简单断言保证产物完整 ────────────────────────────────────────────────
    assert os.path.isfile(save_path), "模型文件未生成"
    assert os.path.isfile(save_path.replace(".pkl", "_results.json")), "结果 JSON 未生成"
    assert os.path.isfile(save_path.replace(".pkl", "_edge_output.json")), "边级输出未生成"
    assert 0.0 <= lgb_res["auc"] <= 1.0
    assert 0.0 <= fused_res["auc"] <= 1.0

    logger.info("✅ run_full_training_and_save 完成")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("测试社群特征融合模块 (hybrid_com.py)")
    logger.info("=" * 60)

    tests = [
        ("模型初始化",                   test_model_initialization),
        ("社群特征提取",                  test_extract_community_features),
        ("社群风险分",                    test_community_risk_score),
        ("特征矩阵构建",                  test_prepare_features_with_community),
        ("训练与预测",                    test_train_and_predict),
        ("评估结果结构",                  test_evaluate),
        ("特征重要性",                    test_get_feature_importance),
        ("模型保存与加载",                test_save_and_load),
        ("JSON 文件加载（mock）",         test_load_community_data_from_mock_json),
        ("α 边界值融合",                  test_alpha_fusion_boundary),
        ("feature_names 不重复追加",      test_feature_names_not_duplicated),
        ("顶层函数集成测试",              test_train_hybrid_model_with_community_integration),
        ("社群数据加载（真实文件）",       test_load_community_data_from_file),
        ("完整训练并保存（真实数据）",     run_full_training_and_save),
    ]

    passed, failed, skipped = 0, 0, 0
    for name, fn in tests:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"运行: {name}")
        try:
            result = fn()
            if result is None:
                skipped += 1
                logger.info(f"⏭  跳过: {name}")
            elif result:
                passed += 1
            else:
                failed += 1
                logger.error(f"❌ 失败: {name}")
        except Exception as e:
            failed += 1
            logger.exception(f"❌ 异常: {name} — {e}")

    logger.info("\n" + "=" * 60)
    logger.info(f"测试完成  ✅ {passed} 通过  ❌ {failed} 失败  ⏭ {skipped} 跳过")
    logger.info("=" * 60)
    if failed > 0:
        sys.exit(1)
