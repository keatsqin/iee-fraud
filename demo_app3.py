"""
欺诈检测交互式演示应用
GraphSAGE + LightGBM 混合模型实时推理
运行: streamlit run demo_app.py
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import joblib
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── 页面配置 ───────────────────────────────────────────────
st.set_page_config(
    page_title="欺诈检测系统 Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 常量 ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs/community_fusion/fusion_hybrid.pkl")
GNN_PATH   = os.path.join(BASE_DIR, "outputs/community_fusion/gnn_model.pt")
DATA_DIR   = os.path.join(BASE_DIR, "data/processed")
RESULTS_PATH = os.path.join(BASE_DIR, "outputs/community_fusion/comparison_results.json")

FRAUD_THRESHOLD = 0.26  # 进一步降低阈值，提高召回与敏感性

# ─── 加载资源（缓存）────────────────────────────────────────
@st.cache_resource(show_spinner="加载模型中...")
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle  # keys: model, scaler, feature_names, lgb_params

@st.cache_resource(show_spinner="加载图数据中...")
def load_graph_data():
    card_df     = pd.read_parquet(os.path.join(DATA_DIR, "card_features.parquet"))
    merchant_df = pd.read_parquet(os.path.join(DATA_DIR, "merchant_features.parquet"))
    edges_df    = pd.read_parquet(os.path.join(DATA_DIR, "edges.parquet"))
    return card_df, merchant_df, edges_df

@st.cache_resource(show_spinner="加载GNN嵌入中...")
def load_gnn_embeddings(card_df, merchant_df, edges_df):
    from src.models.GNN.gnn_models import HeteroGraphSAGE
    from src.data_processing.graph_builder import PyGGraphBuilder

    builder = PyGGraphBuilder()
    data = builder.build_hetero_graph(card_df, merchant_df, edges_df)

    card_in   = data['card'].x.shape[1]
    merch_in  = data['merchant'].x.shape[1]

    gnn_ckpt_path = GNN_PATH
    if not os.path.exists(gnn_ckpt_path):
        legacy_path = os.path.join(BASE_DIR, "outputs/models/graphsage_fraud_detector.pt")
        gnn_ckpt_path = legacy_path if os.path.exists(legacy_path) else GNN_PATH

    ckpt = torch.load(gnn_ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    edge_in   = sd['edge_encoder.0.weight'].shape[1] if 'edge_encoder.0.weight' in sd else 0
    hidden    = sd['edge_encoder.0.weight'].shape[0] if 'edge_encoder.0.weight' in sd else 256
    out_ch    = sd['edge_encoder.4.weight'].shape[0] if 'edge_encoder.4.weight' in sd else 128
    num_layers = sum(1 for k in sd
                     if k.startswith('convs.') and k.endswith('.lin_l.bias')
                     and 'card___transacts___merchant' in k)

    model = HeteroGraphSAGE(
        card_in_channels=card_in,
        merchant_in_channels=merch_in,
        edge_in_channels=edge_in,
        hidden_channels=hidden,
        out_channels=out_ch,
        num_layers=num_layers,
        dropout=0.4
    )
    model.load_state_dict(sd)
    model.eval()

    embeddings = model.get_embeddings(data, device='cpu')
    return embeddings, builder, data

@st.cache_data(show_spinner=False)
def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)

# ─── 概率校准函数 ────────────────────────────────────────────────
def calibrate_probability(prob, method='stretch'):
    """
    对原始概率进行校准，使其更敏感

    Args:
        prob: 原始概率 (0-1)
        method: 校准方法
            - 'stretch': 线性拉伸，将[0.2,0.8]映射到[0,1]
            - 'power': 幂变换
            - 'sigmoid': sigmoid缩放
    """
    if method == 'stretch':
        if prob < 0.2:
            calibrated = 0.0
        elif prob > 0.8:
            calibrated = 1.0
        else:
            calibrated = (prob - 0.2) / 0.6
    elif method == 'power':
        calibrated = np.power(prob, 0.5)
    elif method == 'sigmoid':
        scaled = (prob - 0.5) * 3
        calibrated = 1 / (1 + np.exp(-scaled))
    else:
        calibrated = prob

    calibrated = np.clip(calibrated, 0.0, 1.0)
    return calibrated

# ─── 推理核心 ──────────────────────────────────────────────────
def predict_transaction(
        bundle, embeddings, builder, data,
        card_id: str, merchant_id: str,
        tx_amt: float, hour: int, day_of_week: int,
        card_tx_1h: int, card_tx_24h: int,
        time_since_last_tx: int, card_tx_seq: int,
        is_new_card: int, is_burst: int,
        extra_edge_feats: dict = None
):
    """构造单笔交易特征向量并预测"""
    if extra_edge_feats is None:
        extra_edge_feats = {}

    model = bundle['model']
    scaler = bundle['scaler']
    feat_names = bundle['feature_names']

    # 1. GNN 节点嵌入
    card_emb_all = embeddings['card'].cpu().numpy()
    merchant_emb_all = embeddings['merchant'].cpu().numpy()

    def norm_card(x):
        try:
            return str(int(float(x)))
        except:
            return str(x)

    card_key = norm_card(card_id)
    merch_key = str(merchant_id)

    card_idx = builder.card_mapping.get(card_key)
    merch_idx = builder.merchant_mapping.get(merch_key)

    if card_idx is None:
        card_emb = card_emb_all.mean(axis=0)
    else:
        card_emb = card_emb_all[card_idx]

    if merch_idx is None:
        merchant_emb = merchant_emb_all.mean(axis=0)
    else:
        merchant_emb = merchant_emb_all[merch_idx]

    emb_diff = card_emb - merchant_emb
    emb_prod = card_emb * merchant_emb

    # 2. 边特征（仅 edge_feat_* 进入图构建器的 edge_scaler）
    edge_feat_cols = [c for c in feat_names if c.startswith('edge_feat_')]
    expected_edge_dim = int(getattr(builder.edge_scaler, 'n_features_in_', len(edge_feat_cols)))

    if len(edge_feat_cols) == 0:
        edge_feat_cols = [f'edge_feat_{i}' for i in range(expected_edge_dim)]

    edge_feat_cols = edge_feat_cols[:expected_edge_dim]

    # 在当前模型中无法直接从 UI 还原训练时原始 125 维边字段，默认用 0（经 edge_scaler 标准化后）
    edge_row = pd.Series(index=edge_feat_cols, data=0.0, dtype=np.float32)
    edge_raw = edge_row.values.astype(np.float32).reshape(1, -1)
    edge_scaled = builder.edge_scaler.transform(edge_raw)

    # 3. 按模型 feature_names 组装，确保与训练维度严格一致
    feat_map = {}

    for i in range(card_emb.shape[0]):
        feat_map[f'card_emb_{i}'] = float(card_emb[i])
    for i in range(merchant_emb.shape[0]):
        feat_map[f'merchant_emb_{i}'] = float(merchant_emb[i])
    for i in range(emb_diff.shape[0]):
        feat_map[f'emb_diff_{i}'] = float(emb_diff[i])
    for i in range(emb_prod.shape[0]):
        feat_map[f'emb_prod_{i}'] = float(emb_prod[i])

    for i, col in enumerate(edge_feat_cols):
        feat_map[col] = float(edge_scaled[0, i])

    # 社群特征（comm_*）若存在，当前单笔推理默认置 0
    for c in feat_names:
        if c.startswith('comm_') and c not in feat_map:
            feat_map[c] = 0.0

    X = np.array([feat_map.get(c, 0.0) for c in feat_names], dtype=np.float32).reshape(1, -1)
    X = scaler.transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if hasattr(model, 'predict_proba'):
        raw_prob = float(model.predict_proba(X)[0][1])
    else:
        raw_prob = float(model.predict(X)[0])

    calibrated_prob = calibrate_probability(raw_prob, method='sigmoid')
    prob = 0.7 * calibrated_prob + 0.3 * raw_prob

    # 风险增强
    risk_signals = 0
    risk_signals += int(tx_amt >= 1500)
    risk_signals += int(0 <= hour < 6)
    risk_signals += int(card_tx_1h >= 3)
    risk_signals += int(card_tx_24h >= 15)
    risk_signals += int(time_since_last_tx <= 120)
    risk_signals += int(is_new_card == 1)
    risk_signals += int(is_burst == 1)

    if risk_signals >= 2:
        boost = min(0.16, 0.02 * risk_signals)
        prob += boost

    prob = float(np.clip(prob, 0.0, 1.0))

    return prob, edge_feat_cols, edge_row

# ─── UI 工具 ─────────────────────────────────────────────────
def fraud_gauge(prob):
    color = "#e74c3c" if prob >= FRAUD_THRESHOLD else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%', 'font': {'size': 36}},
        title={'text': "欺诈概率", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30],  'color': '#d5f5e3'},
                {'range': [30, 60], 'color': '#fdebd0'},
                {'range': [60, 100],'color': '#fadbd8'},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': FRAUD_THRESHOLD * 100
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig

# ─── 主界面 ──────────────────────────────────────────────────
def main():
    st.title("🔍 欺诈交易检测系统")
    st.caption("GraphSAGE + LightGBM 混合模型 · 实时推理演示 ")

    # 加载资源
    try:
        bundle = load_model()
        card_df, merchant_df, edges_df = load_graph_data()
        embeddings, builder, data = load_gnn_embeddings(card_df, merchant_df, edges_df)
        results = load_results()
    except Exception as e:
        st.error(f"资源加载失败: {e}")
        st.stop()

    # ── 侧边栏：模型指标 ──────────────────────────────────────
    with st.sidebar:
        st.header("📊 模型性能")
        if results is not None:
            if 'test' in results:
                t = results['test']
            elif 'baseline' in results:
                t = results['baseline']
            elif 'fusion' in results:
                t = results['fusion']
            else:
                t = None

            if t is not None:
                if 'auc' in t:
                    st.metric("AUC", f"{t['auc']:.4f}")
                if 'ap' in t:
                    st.metric("AP", f"{t['ap']:.4f}")
                if 'precision' in t:
                    st.metric("Precision", f"{t['precision']:.4f}")
                if 'recall' in t:
                    st.metric("Recall", f"{t['recall']:.4f}")
                if 'f1' in t:
                    st.metric("F1", f"{t['f1']:.4f}")

                st.divider()
                if 'train' in results and isinstance(results['train'], dict):
                    if 'best_iteration' in results['train']:
                        st.caption(f"最优迭代轮次: {results['train']['best_iteration']}")
                    if 'val_auc' in results['train']:
                        st.caption(f"验证集 AUC: {results['train']['val_auc']:.4f}")
                else:
                    source = 'fusion' if 'fusion' in results else ('baseline' if 'baseline' in results else 'unknown')
                    st.caption(f"指标来源: {source}")
                st.divider()
            else:
                st.warning("模型性能数据格式不支持")
        else:
            st.warning("模型性能数据未加载")

        st.subheader("数据集概览")
        st.caption(f"卡片数: {len(card_df):,}")
        st.caption(f"商户数: {len(merchant_df):,}")
        st.caption(f"交易数: {len(edges_df):,}")
        fraud_cnt = int(edges_df['isFraud'].sum())
        st.caption(f"欺诈交易: {fraud_cnt:,} ({fraud_cnt/len(edges_df)*100:.1f}%)")

        st.divider()
        st.info("🔧 改进说明:\n• 使用predict_proba获取概率\n• 增加特征交互项\n• 使用sigmoid校准并融合原始概率\n• 多风险信号触发动态增强\n• 降低检测阈值")

    # ── Tab 布局 ──────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🧪 模拟交易检测", "🔎 历史交易查询"])

    # ════════════════════════════════════════════════════════
    # Tab 1: 模拟交易
    # ════════════════════════════════════════════════════════
    with tab1:
        st.subheader("模拟一笔交易")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("**选择用户 & 商户**")

            card_ids = sorted(card_df['card_id'].astype(str).tolist())
            merchant_ids = sorted(merchant_df['merchant_id'].astype(str).tolist())

            selected_card = st.selectbox("选择卡片 (card_id)", card_ids, index=0)
            selected_merchant = st.selectbox("选择商户 (merchant_id)", merchant_ids, index=0)

            st.markdown("**交易基本信息**")

            tx_amt = st.number_input(
                "交易金额 ($)",
                min_value=0.01, max_value=None,
                value=100.0,
                step=10.0, format="%.2f"
            )
            hour = st.slider(
                "交易小时 (0-23)", 0, 23,
                value=datetime.now().hour
            )
            day_of_week = st.selectbox(
                "星期",
                options=[0,1,2,3,4,5,6],
                format_func=lambda x: ["周一","周二","周三","周四","周五","周六","周日"][x],
                index=datetime.now().weekday()
            )

        with col_right:
            st.markdown("**行为特征**")

            card_tx_1h = st.number_input(
                "过去1小时交易次数",
                min_value=0,
                max_value=None,
                value=0,
                step=1
            )
            card_tx_24h = st.number_input(
                "过去24小时交易次数",
                min_value=0,
                max_value=None,
                value=5,
                step=1
            )
            time_since_last_tx = st.number_input(
                "距上次交易时间(秒)",
                min_value=0,
                max_value=None,
                value=3600,
                step=60
            )
            card_tx_seq = st.number_input(
                "该卡历史总交易序号",
                min_value=1,
                max_value=None,
                value=10,
                step=1
            )
            is_new_card = st.checkbox(
                "是否新卡 (首次交易)",
                value=False
            )
            is_burst = st.checkbox(
                "是否突发交易 (1小时内≥3笔)",
                value=False
            )

        st.divider()

        run_btn = st.button("🚀 立即检测", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("推理中..."):
                try:
                    prob, _, _ = predict_transaction(
                        bundle, embeddings, builder, data,
                        card_id=selected_card,
                        merchant_id=selected_merchant,
                        tx_amt=tx_amt,
                        hour=hour,
                        day_of_week=day_of_week,
                        card_tx_1h=card_tx_1h,
                        card_tx_24h=card_tx_24h,
                        time_since_last_tx=time_since_last_tx,
                        card_tx_seq=card_tx_seq,
                        is_new_card=int(is_new_card),
                        is_burst=int(is_burst),
                        extra_edge_feats={}
                    )

                    is_fraud = prob >= FRAUD_THRESHOLD
                    res_col1, res_col2 = st.columns([1, 1])

                    with res_col1:
                        st.plotly_chart(fraud_gauge(prob), use_container_width=True)

                    with res_col2:
                        st.markdown("### 检测结果")
                        if is_fraud:
                            st.error(f"⚠️ **高风险交易 — 疑似欺诈**")
                        else:
                            st.success(f"✅ **正常交易**")

                        st.markdown(f"""
| 字段 | 值 |
|------|-----|
| 卡片 ID | `{selected_card}` |
| 商户 ID | `{selected_merchant}` |
| 金额 | **${tx_amt:.2f}** |
| 时间 | {hour:02d}:00 · {["周一","周二","周三","周四","周五","周六","周日"][day_of_week]} |
| 欺诈概率 | **{prob*100:.2f}%** |
| 判定阈值 | {FRAUD_THRESHOLD*100:.0f}% |
| 结论 | {"🔴 欺诈" if is_fraud else "🟢 正常"} |
""")
                        risks = []
                        if tx_amt > 1000:
                            risks.append(f"大额交易 (${tx_amt:.0f})")
                        if 0 <= hour < 6:
                            risks.append("深夜时段")
                        if card_tx_1h >= 3:
                            risks.append(f"1小时内{card_tx_1h}笔交易")
                        if is_new_card:
                            risks.append("新卡首次使用")
                        if is_burst:
                            risks.append("突发高频交易")
                        if time_since_last_tx < 60:
                            risks.append("交易间隔过短")
                        if risks:
                            st.warning("风险因素: " + " · ".join(risks))

                except Exception as e:
                    st.error(f"推理失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # ════════════════════════════════════════════════════════
    # Tab 2: 历史交易查询
    # ════════════════════════════════════════════════════════
    with tab2:
        st.subheader("历史交易查询")

        q_col1, q_col2 = st.columns(2)
        with q_col1:
            query_card = st.selectbox("选择卡片", sorted(card_df['card_id'].astype(str).tolist()), key="q_card")
        with q_col2:
            show_n = st.slider("显示最近N笔", 5, 50, 20)

        card_txs = edges_df[edges_df['card_id'].astype(str) == str(query_card)].copy()
        card_txs = card_txs.sort_values('TransactionDT', ascending=False).head(show_n)

        if len(card_txs) == 0:
            st.info("该卡片无历史交易记录")
        else:
            fraud_count = int(card_txs['isFraud'].sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("交易笔数", len(card_txs))
            m2.metric("欺诈笔数", fraud_count)
            m3.metric("平均金额", f"${card_txs['TransactionAmt'].mean():.2f}")
            m4.metric("最大金额", f"${card_txs['TransactionAmt'].max():.2f}")

            fig_ts = px.scatter(card_txs, x='TransactionDT', y='TransactionAmt',
                                color=card_txs['isFraud'].map({0:'正常', 1:'欺诈'}),
                                color_discrete_map={'正常':'#2ecc71','欺诈':'#e74c3c'},
                                size='TransactionAmt', title="交易金额时序",
                                labels={'TransactionDT':'时间戳','TransactionAmt':'金额($)'})
            fig_ts.update_layout(height=300, margin=dict(t=40,b=20))
            st.plotly_chart(fig_ts, use_container_width=True)

            display_cols = ['TransactionID','TransactionAmt','hour','day_of_week',
                            'is_night','is_weekend','card_tx_1h','card_tx_24h','isFraud']
            display_cols = [c for c in display_cols if c in card_txs.columns]
            styled = card_txs[display_cols].style.apply(
                lambda row: ['background-color: #fadbd8' if row['isFraud'] == 1
                             else 'background-color: #d5f5e3' for _ in row], axis=1
            )
            st.dataframe(styled, use_container_width=True)

# ─── 运行入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    main()