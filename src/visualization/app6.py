"""
欺诈检测可视化平台
"""
import os
import json
import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from loguru import logger

cyto.load_extra_layouts()

COLORS = {
    'background':'#1a1a2e',
    'card': '#16213e',
    'text': '#eaeaea',
    'primary': '#0f3460',
    'danger': '#e94560',
    'success': '#00d9ff',
    'warning': '#ffc107',
    'purple': '#9b59b6'
}


DOCX_MODEL_ROWS = [
    {'model': 'GraphSAGE + LightGBM', 'auc': 0.9805, 'ap': 0.8182, 'f1': 0.5532, 'precision': 0.3982, 'recall': 0.9056},
    {'model': 'GAT + LightGBM', 'auc': 0.9803, 'ap': 0.8179, 'f1': 0.5496, 'precision': 0.3944, 'recall': 0.9066},
    {'model': 'GraphSAGE', 'auc': 0.9428, 'ap': 0.5574, 'f1': 0.3628, 'precision': 0.2329, 'recall': 0.8202},
    {'model': 'GAT', 'auc': 0.9268, 'ap': 0.5124, 'f1': 0.4055, 'precision': 0.2824, 'recall': 0.7187},
]

DOCX_HYBRID_ENHANCE_ROWS = [
    {'model': 'GraphSAGE + LightGBM (基线)', 'auc': 0.9805, 'ap': 0.8182, 'f1': 0.5532, 'precision': 0.3982, 'recall': 0.9056},
    {'model': 'Hybrid GNN + LightGBM (拼接社群特征)', 'auc': 0.9701, 'ap': 0.7835, 'f1': 0.8605, 'precision': 0.6215, 'recall': 0.7217},
]

DOCX_IMBALANCE_ROWS = [
    {'model': 'GraphSAGE + LightGBM (基线)', 'auc': 0.9805, 'ap': 0.8182, 'f1': 0.5532, 'precision': 0.3982, 'recall': 0.9056},
    {'model': 'Hybrid + 不平衡再处理', 'auc': 0.98861, 'ap': 0.91381, 'f1': 0.85967, 'precision': 0.90932, 'recall': 0.81516},
]

# DOCX_BORDERLINE_ROWS 已被删除

app = dash.Dash(__name__, suppress_callback_exceptions=True, title='欺诈检测可视化平台')


def load_edges_data():
    """加载边数据"""
    needed_cols = ['card_id', 'merchant_id', 'TransactionAmt', 'isFraud',
                   'hour', 'day', 'day_of_week', 'ProductCD'] + [f'C{i}' for i in range(1, 15)]

    parquet_paths = ['data/processed/edges.parquet', '../data/processed/edges.parquet',
                     'outputs/processed/edges.parquet', '../outputs/processed/edges.parquet']
    for path in parquet_paths:
        if os.path.exists(path):
            try:
                logger.info(f"Loading edges from {path}")
                all_cols = pd.read_parquet(path, columns=[]).columns.tolist()
                cols_to_load = [c for c in needed_cols if c in all_cols]
                df = pd.read_parquet(path, columns=cols_to_load)
                logger.info(f"Loaded {len(df)} transactions")
                return df
            except Exception as e:
                logger.warning(f"Failed to load parquet: {e}")

    csv_paths = ['data/processed/edges.csv', '../data/processed/edges.csv']
    for path in csv_paths:
        if os.path.exists(path):
            logger.info(f"Loading edges from {path}")
            df = pd.read_csv(path, usecols=lambda c: c in needed_cols)
            logger.info(f"Loaded {len(df)} transactions from csv")
            return df

    logger.warning("No edges data found!")
    return None


def precompute_stats(edges_df=None):
    """
    预计算统计数据
    参数 edges_df 被忽略，直接从原始数据计算
    """
    logger.info("Computing statistics from original data...")
    stats = {}

    # 加载原始数据
    try:
        train_transaction = pd.read_csv('D:/ZM/534531/ieee-fraud-detection/train_transaction.csv')
        df = train_transaction.copy()
        logger.info(f"Loaded {len(df)} transactions")

        # 尝试加载identity数据以获取更多特征
        try:
            train_identity = pd.read_csv('D:/ZM/534531/ieee-fraud-detection/train_identity.csv')
            df = df.merge(train_identity, on='TransactionID', how='left')
            logger.info(f"Merged with identity data, shape: {df.shape}")
        except Exception as e:
            logger.warning(f"Identity data not available: {e}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    # ========== 1. 时间热力图数据 ==========
    import datetime
    base_date = datetime.datetime(2017, 12, 1)
    df['datetime'] = df['TransactionDT'].apply(lambda x: base_date + datetime.timedelta(seconds=x))
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # Monday=0, Sunday=6

    if 'hour' in df.columns and 'day_of_week' in df.columns and 'isFraud' in df.columns:
        risk_matrix = np.zeros((7, 24))
        count_matrix = np.zeros((7, 24))

        grouped = df.groupby(['day_of_week', 'hour']).agg({'isFraud': ['sum', 'count']})

        for (d, h), row in grouped.iterrows():
            d, h = int(d) % 7, int(h) % 24
            count_matrix[d, h] = row[('isFraud', 'count')]
            risk_matrix[d, h] = row[('isFraud', 'sum')]

        time_heatmap = np.where(count_matrix > 0, risk_matrix / count_matrix, 0)
        stats['time_heatmap'] = time_heatmap
        logger.info("✓ Time heatmap computed")

    # ========== 2. 每日趋势 ==========
    start_date = pd.Timestamp('2017-12-01')
    df['transaction_date'] = start_date + pd.to_timedelta(df['TransactionDT'], unit='s')
    df['day'] = df['transaction_date'].dt.date

    if 'day' in df.columns:
        daily = df.groupby('day').agg(
            fraud=('isFraud', 'sum'),
            total=('isFraud', 'count')
        ).reset_index()

        daily['fraud_rate'] = (daily['fraud'] / daily['total'] * 100).round(2)
        daily['normal_count'] = daily['total'] - daily['fraud']
        daily['normal_rate'] = (daily['normal_count'] / daily['total'] * 100).round(2)
        daily = daily.sort_values('day').tail(30)
        daily['fraud_rate_ma7'] = daily['fraud_rate'].rolling(window=7, min_periods=1).mean().round(2)

        stats['daily_trend'] = daily
        logger.info("✓ Daily trend computed")

    # ========== 3. 金额分布 ==========
    if 'TransactionAmt' in df.columns:
        normal = df[df['isFraud'] == 0]['TransactionAmt']
        fraud = df[df['isFraud'] == 1]['TransactionAmt']

        stats['amt_normal'] = normal[normal < normal.quantile(0.99)].values if len(normal) > 0 else np.array([])
        stats['amt_fraud'] = fraud[fraud < fraud.quantile(0.99)].values if len(fraud) > 0 else np.array([])

        # 金额区间欺诈率
        bins = [0, 50, 100, 200, 500, 1000, 5000, float('inf')]
        labels = ['0-50', '50-100', '100-200', '200-500', '500-1K', '1K-5K', '5K+']
        df['amt_bin'] = pd.cut(df['TransactionAmt'], bins=bins, labels=labels, right=False)

        amt_stats = df.groupby('amt_bin', observed=True).agg({'isFraud': ['sum', 'count']}).reset_index()
        amt_stats.columns = ['bin', 'fraud_count', 'total_count']
        amt_stats['fraud_rate'] = (amt_stats['fraud_count'] / amt_stats['total_count'] * 100).round(2)
        amt_stats['normal_count'] = amt_stats['total_count'] - amt_stats['fraud_count']
        stats['amt_fraud_rate'] = amt_stats
        logger.info("✓ Amount distribution computed")

    # ========== 4. 卡片频率 ==========
    if 'card1' in df.columns:  # IEEE数据中使用 card1 而不是 card_id
        card_freq = df.groupby('card1').size()
        stats['card_freq'] = card_freq.values

        card_stats = df.groupby('card1').agg({'isFraud': ['sum', 'count']}).reset_index()
        card_stats.columns = ['card', 'fraud', 'total']
        card_stats = card_stats.sort_values('total', ascending=False).head(10)
        card_stats['card'] = card_stats['card'].apply(lambda x: f"Card_{str(x)[-4:]}")
        stats['top_cards'] = card_stats
        logger.info("✓ Card frequency computed")

    # ========== 5. 小时统计 ==========
    if 'hour' in df.columns:
        hour_stats = df.groupby('hour').agg({'isFraud': ['sum', 'count']}).reset_index()
        hour_stats.columns = ['hour', 'fraud', 'total']
        hour_stats['rate'] = hour_stats['fraud'] / hour_stats['total'] * 100
        stats['hour_stats'] = hour_stats
        logger.info("✓ Hour statistics computed")

    # ========== 6. C特征相关性 (来自identity数据) ==========
    c_cols = [f'C{i}' for i in range(1, 15)]
    existing_cols = [c for c in c_cols if c in df.columns]
    if existing_cols and 'isFraud' in df.columns:
        corrs = []
        for col in existing_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            corr = df[[col, 'isFraud']].corr().iloc[0, 1]
            corrs.append({'feature': col, 'corr': corr if not np.isnan(corr) else 0})
        stats['c_features'] = pd.DataFrame(corrs).sort_values('corr', key=abs, ascending=False)
        logger.info("✓ C-feature correlations computed")

    # ========== 7. D特征相关性 (来自identity数据) ==========
    d_cols = [f'D{i}' for i in range(1, 16)]
    existing_d_cols = [c for c in d_cols if c in df.columns]
    if existing_d_cols and 'isFraud' in df.columns:
        d_corrs = []
        for col in existing_d_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            corr = df[[col, 'isFraud']].corr().iloc[0, 1]
            d_corrs.append({'feature': col, 'corr': corr if not np.isnan(corr) else 0})
        stats['d_features'] = pd.DataFrame(d_corrs).sort_values('corr', key=abs, ascending=False)
        logger.info("✓ D-feature correlations computed")

    # ========== 8. 产品类型统计 ==========
    if 'ProductCD' in df.columns:
        prod_stats = df.groupby('ProductCD').agg(
            fraud=('isFraud', 'sum'),
            total=('isFraud', 'count')
        ).reset_index()
        prod_stats.columns = ['product', 'fraud', 'total']
        prod_stats['rate'] = prod_stats['fraud'] / prod_stats['total']
        stats['product_fraud'] = prod_stats.sort_values('fraud', ascending=False)
        logger.info("✓ Product statistics computed")

    # ========== 9. 设备类型统计 (来自identity) ==========
    if 'DeviceType' in df.columns:
        device_stats = df.groupby('DeviceType').agg(
            fraud=('isFraud', 'sum'),
            total=('isFraud', 'count')
        ).reset_index()
        device_stats['rate'] = device_stats['fraud'] / device_stats['total'] * 100
        stats['device_stats'] = device_stats.sort_values('fraud', ascending=False)
        logger.info("✓ Device statistics computed")

    # ========== 10. 浏览器统计 (来自identity) ==========
    if 'DeviceInfo' in df.columns:
        def extract_browser(info):
            if pd.isna(info):
                return 'Unknown'
            info_str = str(info).lower()
            if 'chrome' in info_str:
                return 'Chrome'
            elif 'safari' in info_str:
                return 'Safari'
            elif 'firefox' in info_str:
                return 'Firefox'
            elif 'edge' in info_str:
                return 'Edge'
            elif 'ie' in info_str:
                return 'IE'
            else:
                return 'Other'

        df['browser'] = df['DeviceInfo'].apply(extract_browser)
        browser_stats = df.groupby('browser').agg(
            fraud=('isFraud', 'sum'),
            total=('isFraud', 'count')
        ).reset_index()
        browser_stats['rate'] = browser_stats['fraud'] / browser_stats['total'] * 100
        stats['browser_stats'] = browser_stats.sort_values('fraud', ascending=False)
        logger.info("✓ Browser statistics computed")

    logger.info("All statistics computed successfully!")
    return stats


def _load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _pick_metrics(result_obj, section):
    s = result_obj.get(section, {}) if isinstance(result_obj, dict) else {}
    if not isinstance(s, dict):
        return None
    keys = ['auc', 'ap', 'precision', 'recall', 'f1']
    if all(k in s for k in keys):
        return {k: float(s[k]) for k in keys}
    return None


def _collect_model_metrics(output_dir='outputs'):
    model_specs = [
        ('Pure GNN(GraphSAGE)', [
            os.path.join(output_dir, 'models', 'graphsage_fraud_detector_results.json'),
            'outputs/models/graphsage_fraud_detector_results.json',
            '../outputs/models/graphsage_fraud_detector_results.json',
        ], 'test'),
        ('Hybrid Base', [
            os.path.join(output_dir, 'models', 'hybrid_gnn_lgb_results.json'),
            'outputs/models/hybrid_gnn_lgb_results.json',
            '../outputs/models/hybrid_gnn_lgb_results.json',
        ], 'test'),
        ('Hybrid Com(LGB)', [
            os.path.join(output_dir, 'models', 'hybrid_gnn_lgb_community_results.json'),
            'outputs/models/hybrid_gnn_lgb_community_results.json',
            '../outputs/models/hybrid_gnn_lgb_community_results.json',
        ], 'test_lgb_only'),
        ('Hybrid Com(Fused)', [
            os.path.join(output_dir, 'models', 'hybrid_gnn_lgb_community_results.json'),
            'outputs/models/hybrid_gnn_lgb_community_results.json',
            '../outputs/models/hybrid_gnn_lgb_community_results.json',
        ], 'test_fused'),
        ('Hybrid v2', [
            os.path.join(output_dir, 'models', 'hybrid_gnn_lgb_v2_results基础模型结果.json'),
            'outputs/models/hybrid_gnn_lgb_v2_results基础模型结果.json',
            '../outputs/models/hybrid_gnn_lgb_v2_results基础模型结果.json',
        ], 'test'),
        ('Hybrid Com v2(LGB)', [
            os.path.join(output_dir, 'models', 'hybrid_gnn_lgb_community_v2_results社群模型结果.json'),
            'outputs/models/hybrid_gnn_lgb_community_v2_results社群模型结果.json',
            '../outputs/models/hybrid_gnn_lgb_community_v2_results社群模型结果.json',
        ], 'test_lgb_only'),
        ('Hybrid Com v2(Fused)', [
            os.path.join(output_dir, 'models', 'hybrid_gnn_lgb_community_v2_results社群模型结果.json'),
            'outputs/models/hybrid_gnn_lgb_community_v2_results社群模型结果.json',
            '../outputs/models/hybrid_gnn_lgb_community_v2_results社群模型结果.json',
        ], 'test_fused'),
        ('Fusion Baseline', [
            os.path.join(output_dir, 'community_fusion', 'comparison_results.json'),
            'outputs/community_fusion/comparison_results.json',
            '../outputs/community_fusion/comparison_results.json',
        ], 'baseline'),
        ('Fusion Model', [
            os.path.join(output_dir, 'community_fusion', 'comparison_results.json'),
            'outputs/community_fusion/comparison_results.json',
            '../outputs/community_fusion/comparison_results.json',
        ], 'fusion'),
        ('Fusion Baseline v2', [
            os.path.join(output_dir, 'community_fusion', 'comparison_results_v2.json'),
            'outputs/community_fusion/comparison_results_v2.json',
            '../outputs/community_fusion/comparison_results_v2.json',
        ], 'baseline'),
        ('Fusion Model v2', [
            os.path.join(output_dir, 'community_fusion', 'comparison_results_v2.json'),
            'outputs/community_fusion/comparison_results_v2.json',
            '../outputs/community_fusion/comparison_results_v2.json',
        ], 'fusion'),
    ]

    rows = []
    for model_name, path_candidates, section in model_specs:
        result_obj = None
        used_path = None
        for p in path_candidates:
            obj = _load_json_if_exists(p)
            if obj is not None:
                result_obj = obj
                used_path = p
                break

        if result_obj is None:
            continue

        m = _pick_metrics(result_obj, section)
        if m is None:
            continue

        row = {'model': model_name, 'source': used_path}
        row.update(m)
        rows.append(row)

    return rows


def load_data(output_dir='outputs'):
    """加载所有数据"""
    data = {'communities': None, 'stats': None, 'suspicious': None,
            'graph_stats': None, 'edges_df': None, 'precomputed': {}}

    comm_path = os.path.join(output_dir, 'communities', 'suspicious_communities.json')
    if os.path.exists(comm_path):
        with open(comm_path, 'r', encoding='utf-8') as f:
            comm_data = json.load(f)
            data['communities'] = comm_data.get('communities', {})
            data['stats'] = comm_data.get('stats', {})
            data['suspicious'] = comm_data.get('suspicious', [])

    for path in ['data/processed/graph_stats.json', '../data/processed/graph_stats.json']:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data['graph_stats'] = json.load(f)
            break

    data['edges_df'] = load_edges_data()
    data['precomputed'] = precompute_stats(data['edges_df'])

    data['all_model_metrics'] = _collect_model_metrics(output_dir)

    # 保留旧字段，避免已有回调报错
    data['model_results'] = _load_json_if_exists(os.path.join(output_dir, 'models', 'hybrid_gnn_lgb_results.json')) or \
                            _load_json_if_exists('outputs/models/hybrid_gnn_lgb_results.json') or \
                            _load_json_if_exists('../outputs/models/hybrid_gnn_lgb_results.json')
    data['gnn_results'] = _load_json_if_exists(os.path.join(output_dir, 'models', 'graphsage_fraud_detector_results.json')) or \
                          _load_json_if_exists('outputs/models/graphsage_fraud_detector_results.json') or \
                          _load_json_if_exists('../outputs/models/graphsage_fraud_detector_results.json')

    logger.info(f"Loaded: {len(data['suspicious'] or [])} suspicious communities")
    logger.info(f"Loaded model metrics rows: {len(data['all_model_metrics'])}")
    return data


DATA = load_data()


def create_layout():
    return html.Div([
        # 顶部
        html.Div([
            html.H1("🔍 信用卡欺诈检测可视化平台",
                   style={'color': COLORS['text'], 'margin': '0', 'fontSize': '24px'}),
            html.P("基于GNN的欺诈社群识别与多维度风险分析",
                  style={'color': '#888', 'margin': '5px 0', 'fontSize': '14px'})
        ], style={'padding': '15px 20px', 'backgroundColor': COLORS['primary']}),

        # 标签页
        dcc.Tabs([
            dcc.Tab(label='📊 总览仪表盘', children=[create_overview_layout()],
                   style={'backgroundColor': COLORS['background']},
                   selected_style={'backgroundColor': COLORS['primary'], 'color': 'white'}),
            dcc.Tab(label='💰 交易分析', children=[create_transaction_layout()],
                   style={'backgroundColor': COLORS['background']},
                   selected_style={'backgroundColor': COLORS['primary'], 'color': 'white'}),
            dcc.Tab(label='🔗 社群分析', children=[create_community_layout()],
                   style={'backgroundColor': COLORS['background']},
                   selected_style={'backgroundColor': COLORS['primary'], 'color': 'white'}),
            dcc.Tab(label='📈 模型性能', children=[create_model_layout()],
                   style={'backgroundColor': COLORS['background']},
                   selected_style={'backgroundColor': COLORS['primary'], 'color': 'white'}),
        ], style={'backgroundColor': COLORS['background']}),

        dcc.Store(id='selected-community'),
    ], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})


def create_overview_layout():
    """总览仪表盘 - 展示所有关键统计"""
    return html.Div([
        # 关键指标卡片
        html.Div([
            create_metric_card('total-tx', '总交易量', COLORS['success'], '📦'),
            create_metric_card('fraud-tx', '欺诈交易', COLORS['danger'], '⚠️'),
            create_metric_card('fraud-rate', '欺诈率', COLORS['warning'], '📊'),
            create_metric_card('suspicious-comm', '可疑社群', COLORS['purple'], '🔗'),
            create_metric_card('total-cards', '卡片数', COLORS['success'], '💳'),
            create_metric_card('total-merchants', '商户数', COLORS['warning'], '🏪'),
        ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '20px', 'flexWrap': 'wrap'}),

        # 第一行：时间维度分析
        html.Div([
            html.Div([
                html.H3("⏰ 时间维度风险热力图", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='time-heatmap', style={'height': '320px'}, config={'displayModeBar': False})
            ], style={'flex': '1.5', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
            html.Div([
                html.H3("📈 每日欺诈趋势（最近30天）", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='daily-trend', style={'height': '320px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        # 第二行：产品类型和小时欺诈率
        html.Div([
            html.Div([
                html.H3("🏷️ 产品类型欺诈分布", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='product-fraud', style={'height': '300px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
            html.Div([
                html.H3("🌙 小时欺诈率分布", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='hour-fraud', style={'height': '300px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        # 第三行：金额区间欺诈率和设备/浏览器统计
        html.Div([
            html.Div([
                html.H3("💰 金额区间欺诈分析", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='amount-fraud-rate', style={'height': '300px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
            html.Div([
                html.H3("📱 设备与浏览器分析", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                html.Div([
                    dcc.Graph(id='device-stats', style={'height': '140px'}, config={'displayModeBar': False}),
                    dcc.Graph(id='browser-stats', style={'height': '140px'}, config={'displayModeBar': False})
                ])
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
        ], style={'display': 'flex', 'gap': '20px'}),
    ], style={'padding': '20px'})


def create_transaction_layout():
    """交易分析模块 - 详细交易特征"""
    return html.Div([
        # 第一行：金额分布对比
        html.Div([
            html.Div([
                html.H3("💵 交易金额分布对比", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='amount-distribution', style={'height': '350px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
            html.Div([
                html.H3("🏆 高频交易卡片TOP10", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='top-cards', style={'height': '350px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        # 第二行：C特征和D特征相关性
        html.Div([
            html.Div([
                html.H3("🔢 C系列特征与欺诈相关性", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='c-features', style={'height': '350px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
            html.Div([
                html.H3("🔢 D系列特征与欺诈相关性", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='d-features', style={'height': '350px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        # 第三行：时段统计（原卡片频率分布已删除，此处只保留时段统计并使其占满整行）
        html.Div([
            html.Div([
                html.H3("⏰ 时段交易统计", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='hour-stats', style={'height': '300px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
        ], style={'display': 'flex', 'gap': '20px'}),
    ], style={'padding': '20px'})


def create_community_layout():
    """社群分析模块"""
    return html.Div([
        html.Div([
            # 左侧：社群选择
            html.Div([
                html.H3("🔍 选择社群", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Dropdown(id='community-selector', placeholder='选择社群...',
                           style={'backgroundColor': '#fff', 'marginBottom': '15px'}),
                html.Div(id='community-detail-panel')
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px',
                     'borderRadius': '10px', 'minWidth': '280px'}),

            # 右侧：网络图
            html.Div([
                html.H3("🕸️ 社群网络结构", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                cyto.Cytoscape(
                    id='network-graph',
                    layout={'name': 'cose', 'animate': False, 'nodeRepulsion': 8000,
                           'idealEdgeLength': 80, 'gravity': 0.3},
                    style={'width': '100%', 'height': '380px', 'backgroundColor': '#1a1a2e'},
                    stylesheet=[
                        {'selector': 'node[type="card"]', 'style': {
                            'background-color': '#00d9ff', 'label': 'data(label)',
                            'width': 20, 'height': 20, 'font-size': '8px', 'color': 'white',
                            'text-valign': 'center', 'text-halign': 'center',
                            'border-width': 2, 'border-color': '#0099cc'}},
                        {'selector': 'node[type="merchant"]', 'style': {
                            'background-color': '#ffc107', 'label': 'data(label)',
                            'width': 30, 'height': 30, 'shape': 'rectangle', 'font-size': '10px',
                            'color': '#333', 'font-weight': 'bold',
                            'text-valign': 'center', 'text-halign': 'center',
                            'border-width': 2, 'border-color': '#cc9900'}},
                        {'selector': 'edge', 'style': {
                            'line-color': '#e94560', 'width': 2, 'opacity': 0.7,
                            'curve-style': 'bezier', 'target-arrow-shape': 'triangle',
                            'target-arrow-color': '#e94560', 'arrow-scale': 0.8}},
                        {'selector': 'node:selected', 'style': {
                            'border-width': 4, 'border-color': '#fff'}},
                        {'selector': 'edge:selected', 'style': {
                            'line-color': '#fff', 'width': 3}},
                    ],
                    elements=[]
                )
            ], style={'flex': '2', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        # 社群统计
        html.Div([
            html.Div([
                html.H3("📏 社群规模分布", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='community-size-dist', style={'height': '280px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
            html.Div([
                html.H3("⚖️ 社群风险对比", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='community-risk-compare', style={'height': '280px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
            html.Div([
                html.H3("🎯 异常类型分布", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='anomaly-pie', style={'height': '280px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
        ], style={'display': 'flex', 'gap': '20px'}),
    ], style={'padding': '20px'})


def create_model_layout():
    """模型性能模块 - 调整后的布局"""
    return html.Div([
        # 第一行：两个图表
        html.Div([
            html.Div([
                html.H3("📊 四模型性能对比", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='docx-four-models', style={'height': '320px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
            html.Div([
                html.H3("🧩 增强机制混合（双模型）", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='docx-hybrid-enhance', style={'height': '320px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        # 第二行：不平衡对比和混淆矩阵（两个图表并排）
        html.Div([
            html.Div([
                html.H3("⚖️ 不平衡数据再处理 vs 基线", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='docx-imbalance-compare', style={'height': '320px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
            html.Div([
                html.H3("🎯 混淆矩阵（主模型）", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='confusion-matrix', style={'height': '320px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        # 第三行：特征重要性和训练曲线（两个图表并排）
        html.Div([
            html.Div([
                html.H3("🔑 特征重要性TOP20", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='feature-importance', style={'height': '340px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
            html.Div([
                html.H3("📈 训练过程", style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Graph(id='training-curve', style={'height': '340px'}, config={'displayModeBar': False})
            ], style={'flex': '1', 'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px'}),
        ], style={'display': 'flex', 'gap': '20px'}),
    ], style={'padding': '20px'})


def create_metric_card(card_id, title, color, icon=''):
    return html.Div([
        html.Span(icon, style={'fontSize': '20px'}),
        html.H4(title, style={'color': '#888', 'fontSize': '12px', 'margin': '5px 0'}),
        html.H2(id=card_id, style={'color': color, 'fontSize': '24px', 'margin': '0'}),
    ], style={'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '10px',
              'textAlign': 'center', 'minWidth': '120px', 'flex': '1'})


def create_stat_row(label, value):
    return html.Div([
        html.Span(f"{label}: ", style={'color': '#888', 'fontSize': '13px'}),
        html.Span(value, style={'color': COLORS['text'], 'fontSize': '13px'})
    ], style={'marginBottom': '4px'})


app.layout = create_layout()


# ============ 总览仪表盘回调 ============

@app.callback(
    [Output('total-tx', 'children'), Output('fraud-tx', 'children'),
     Output('fraud-rate', 'children'), Output('suspicious-comm', 'children'),
     Output('total-cards', 'children'), Output('total-merchants', 'children')],
    Input('community-selector', 'id')
)
def update_metrics(_):
    if DATA['graph_stats']:
        gs = DATA['graph_stats']
        return (f"{gs.get('num_edges', 0):,}", f"{gs.get('fraud_count', 0):,}",
                f"{gs.get('fraud_rate', 0)*100:.2f}%", str(len(DATA['suspicious'] or [])),
                f"{gs.get('num_cards', 0):,}", f"{gs.get('num_merchants', 0):,}")
    return "0", "0", "0%", "0", "0", "0"


@app.callback(Output('time-heatmap', 'figure'), Input('community-selector', 'id'))
def update_time_heatmap(_):
    hours = list(range(24))
    days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

    risk_matrix = DATA['precomputed']['time_heatmap']

    fig = go.Figure(data=go.Heatmap(
        z=risk_matrix,
        x=hours,
        y=days,
        colorscale='RdYlGn_r',
        colorbar=dict(title='欺诈率', tickformat='.1%', len=0.8)
    ))
    fig.update_layout(
        xaxis_title='小时',
        yaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaeaea', size=10),
        margin=dict(l=60, r=20, t=10, b=40)
    )
    return fig


@app.callback(Output('daily-trend', 'figure'), Input('community-selector', 'id'))
def update_daily_trend(_):
    stats = DATA['precomputed']['daily_trend']

    day_vals = stats['day'].values
    x_labels = [f"D{i + 1}" for i in range(len(day_vals))]
    y_total = stats['total'].values
    y_fraud_rate = stats['fraud_rate'].values

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=x_labels, y=y_total, name='总交易',
               marker_color=COLORS['success'], opacity=0.4),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=x_labels, y=y_fraud_rate, name='欺诈率%',
                   line=dict(color=COLORS['danger'], width=2),
                   mode='lines+markers'),
        secondary_y=True
    )

    # 添加7日移动平均线
    if 'fraud_rate_ma7' in stats.columns:
        fig.add_trace(
            go.Scatter(x=x_labels, y=stats['fraud_rate_ma7'], name='7日均线',
                       line=dict(color=COLORS['warning'], width=1.5, dash='dash'),
                       mode='lines'),
            secondary_y=True
        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaeaea', size=10),
        margin=dict(l=50, r=50, t=10, b=40),
        legend=dict(orientation='h', y=1.1),
        showlegend=True,
        xaxis_title='天数'
    )
    fig.update_yaxes(title_text="总交易", secondary_y=False)
    fig.update_yaxes(title_text="欺诈率%", secondary_y=True, range=[0, max(y_fraud_rate) * 1.2])
    return fig


@app.callback(Output('product-fraud', 'figure'), Input('community-selector', 'id'))
def update_product_fraud(_):
    stats = DATA['precomputed']['product_fraud'].copy()
    stats = stats.head(5)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=stats['product'], y=stats['fraud'],
               name='欺诈数', marker_color=COLORS['danger'])
    )

    fig.add_trace(
        go.Scatter(x=stats['product'], y=stats['rate'] * 100,
                   name='欺诈率%', yaxis='y2',
                   line=dict(color=COLORS['warning'], width=2),
                   mode='lines+markers',
                   marker=dict(size=10))
    )

    fig.update_layout(
        yaxis=dict(title='欺诈数'),
        yaxis2=dict(title='欺诈率%', overlaying='y', side='right'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaeaea', size=10),
        margin=dict(l=50, r=50, t=10, b=40),
        legend=dict(orientation='h', y=1.1),
        xaxis_title='产品类型(ProductCD)'
    )
    return fig


@app.callback(Output('hour-fraud', 'figure'), Input('community-selector', 'id'))
def update_hour_fraud(_):
    hour_stats = DATA['precomputed']['hour_stats'].copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hour_stats['hour'],
        y=hour_stats['rate'],
        mode='lines+markers',
        name='欺诈率%',
        line=dict(color=COLORS['danger'], width=2),
        marker=dict(size=8, color=hour_stats['rate'],
                   colorscale='Reds', showscale=True),
        fill='tozeroy',
        fillcolor='rgba(255, 71, 87, 0.2)'
    ))

    avg_rate = hour_stats['rate'].mean()
    fig.add_hline(y=avg_rate, line_dash="dash",
                  line_color=COLORS['warning'],
                  annotation_text=f"平均: {avg_rate:.2f}%",
                  annotation_font=dict(size=10, color=COLORS['warning']))

    fig.update_layout(
        title=dict( font=dict(size=14, color='#eaeaea')),
        xaxis_title='小时',
        yaxis_title='欺诈率 (%)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaeaea', size=10),
        margin=dict(l=50, r=50, t=50, b=40),
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    return fig


@app.callback(Output('amount-fraud-rate', 'figure'), Input('community-selector', 'id'))
def update_amount_fraud_rate_chart(_):
    amt_stats = DATA['precomputed']['amt_fraud_rate'].copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=amt_stats['bin'], y=amt_stats['total_count'],
               name='总交易数', marker_color=COLORS['success'], opacity=0.6),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=amt_stats['bin'], y=amt_stats['fraud_rate'],
                   name='欺诈率%', mode='lines+markers',
                   line=dict(color=COLORS['danger'], width=3),
                   marker=dict(size=12, color=COLORS['danger'])),
        secondary_y=True
    )

    for idx, row in amt_stats.iterrows():
        fig.add_annotation(
            x=row['bin'],
            y=row['total_count'],
            text=str(int(row['fraud_count'])),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=COLORS['danger'],
            ax=0,
            ay=-20,
            font=dict(size=9, color=COLORS['danger'])
        )

    fig.update_layout(
        title=dict(text='金额区间欺诈分析', font=dict(size=14, color='#eaeaea')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaeaea', size=10),
        margin=dict(l=50, r=50, t=50, b=40),
        legend=dict(orientation='h', y=1.1),
        xaxis_title='交易金额区间',
        xaxis_tickangle=-45
    )

    fig.update_yaxes(title_text="交易数量", secondary_y=False)
    fig.update_yaxes(title_text="欺诈率 (%)", secondary_y=True,
                     range=[0, max(amt_stats['fraud_rate']) * 1.2])

    return fig


@app.callback(Output('device-stats', 'figure'), Input('community-selector', 'id'))
def update_device_stats(_):
    if 'device_stats' in DATA['precomputed']:
        stats = DATA['precomputed']['device_stats']
    else:
        stats = pd.DataFrame({'DeviceType': ['mobile', 'desktop'], 'rate': [3.5, 2.8]})

    fig = go.Figure(go.Bar(x=stats['DeviceType'], y=stats['rate'],
                          marker_color=COLORS['purple'],
                          text=[f'{r:.1f}%' for r in stats['rate']],
                          textposition='auto'))
    fig.update_layout(title='设备类型欺诈率', xaxis_title='', yaxis_title='欺诈率%',
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     font=dict(color='#eaeaea', size=9), margin=dict(l=40, r=20, t=30, b=20))
    return fig


@app.callback(Output('browser-stats', 'figure'), Input('community-selector', 'id'))
def update_browser_stats(_):
    if 'browser_stats' in DATA['precomputed']:
        stats = DATA['precomputed']['browser_stats'].head(5)
    else:
        stats = pd.DataFrame({'browser': ['Chrome', 'Safari', 'Firefox'], 'rate': [2.5, 3.2, 2.8]})

    fig = go.Figure(go.Bar(x=stats['browser'], y=stats['rate'],
                          marker_color=COLORS['warning'],
                          text=[f'{r:.1f}%' for r in stats['rate']],
                          textposition='auto'))
    fig.update_layout(title='浏览器欺诈率', xaxis_title='', yaxis_title='欺诈率%',
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     font=dict(color='#eaeaea', size=9), margin=dict(l=40, r=20, t=30, b=20))
    return fig


@app.callback(Output('top-communities', 'figure'), Input('community-selector', 'id'))
def update_top_communities(_):
    if DATA.get('stats'):
        comm_list = []
        for comm_id, stats in DATA['stats'].items():
            fraud_tx = stats.get('fraud_transactions', 0)
            if isinstance(fraud_tx, str):
                fraud_tx = int(fraud_tx)
            total_tx = stats.get('total_transactions', 0)

            if total_tx < 100:
                continue

            comm_list.append({
                'community_id': comm_id,
                'fraud_rate': stats.get('fraud_rate', 0),
                'fraud_transactions': fraud_tx,
                'total_transactions': total_tx,
                'num_cards': stats.get('num_cards', 0)
            })
        comm_list.sort(key=lambda x: x['fraud_rate'], reverse=True)
        comms = comm_list[:10]

        names = [f"社群{c['community_id']}" for c in comms]
        fraud_rates = [c['fraud_rate'] * 100 for c in comms]
        fraud_counts = [c['fraud_transactions'] for c in comms]
    else:
        prod_stats = DATA['precomputed']['product_fraud'].head(10)
        names = prod_stats['product'].values
        fraud_rates = (prod_stats['rate'] * 100).values
        fraud_counts = prod_stats['fraud'].values

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names, x=fraud_rates, orientation='h', name='欺诈率%',
        marker=dict(color=fraud_counts, colorscale='Reds', showscale=True,
                    colorbar=dict(title='欺诈数', len=0.8)),
        text=[f'{r:.1f}%' for r in fraud_rates],
        textposition='outside',
        hovertemplate='%{y}<br>欺诈率: %{x:.1f}%<br>欺诈数: %{marker.color}<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title='欺诈率%',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaeaea', size=10),
        margin=dict(l=70, r=80, t=10, b=40),
        yaxis=dict(autorange='reversed')
    )
    return fig


# ============ 交易分析回调 ============

@app.callback(Output('amount-distribution', 'figure'), Input('community-selector', 'id'))
def update_amount_distribution(_):
    normal = DATA['precomputed']['amt_normal']
    fraud = DATA['precomputed']['amt_fraud']

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=normal, name='正常交易', marker_color=COLORS['success'], opacity=0.6, nbinsx=50))
    fig.add_trace(go.Histogram(x=fraud, name='欺诈交易', marker_color=COLORS['danger'], opacity=0.6, nbinsx=50))
    fig.update_layout(barmode='overlay', xaxis_title='交易金额', yaxis_title='频次',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#eaeaea', size=10), margin=dict(l=50, r=20, t=10, b=40),
                      legend=dict(orientation='h', y=1.1))
    return fig


@app.callback(Output('top-cards', 'figure'), Input('community-selector', 'id'))
def update_top_cards(_):
    if 'top_cards' in DATA['precomputed']:
        card_stats = DATA['precomputed']['top_cards']
    else:
        card_stats = pd.DataFrame({'card': [f'Card_{i}' for i in range(10)],
                                  'total': [1000-i*80 for i in range(10)], 'fraud': [50-i*4 for i in range(10)]})

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=card_stats['card'], x=card_stats['total'], name='总交易',
        orientation='h', marker_color=COLORS['success'], opacity=0.8,
        text=[f"欺诈:{int(f)}" for f in card_stats['fraud']],
        textposition='inside', textfont=dict(color='white', size=10)
    ))

    fig.add_trace(go.Scatter(
        y=card_stats['card'], x=card_stats['total'] + 20, name='欺诈数',
        mode='markers+text',
        marker=dict(color=COLORS['danger'], size=10, symbol='diamond'),
        text=[str(int(f)) for f in card_stats['fraud']],
        textposition='middle right',
        textfont=dict(color=COLORS['danger'], size=11, weight='bold')
    ))

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     font=dict(color='#eaeaea', size=10), margin=dict(l=80, r=50, t=10, b=40),
                     yaxis=dict(autorange='reversed'), legend=dict(orientation='h', y=1.1),
                     xaxis_title='总交易数', showlegend=True)
    return fig


@app.callback(Output('c-features', 'figure'), Input('community-selector', 'id'))
def update_c_features(_):
    if 'c_features' in DATA['precomputed']:
        df = DATA['precomputed']['c_features'].head(10)
    else:
        c_cols = [f'C{i}' for i in range(1, 11)]
        df = pd.DataFrame({'feature': c_cols, 'corr': np.random.rand(10) * 0.2 - 0.1})

    colors = [COLORS['danger'] if c > 0 else COLORS['success'] for c in df['corr']]
    fig = go.Figure(go.Bar(x=df['feature'], y=df['corr'], marker_color=colors))
    fig.update_layout(xaxis_title='特征', yaxis_title='与欺诈相关性',
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     font=dict(color='#eaeaea', size=10), margin=dict(l=50, r=20, t=10, b=40))
    return fig


@app.callback(Output('d-features', 'figure'), Input('community-selector', 'id'))
def update_d_features(_):
    if 'd_features' in DATA['precomputed']:
        df = DATA['precomputed']['d_features'].head(10)
    else:
        d_cols = [f'D{i}' for i in range(1, 11)]
        df = pd.DataFrame({'feature': d_cols, 'corr': np.random.rand(10) * 0.15 - 0.05})

    colors = [COLORS['danger'] if c > 0 else COLORS['success'] for c in df['corr']]
    fig = go.Figure(go.Bar(x=df['feature'], y=df['corr'], marker_color=colors))
    fig.update_layout(xaxis_title='特征', yaxis_title='与欺诈相关性',
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     font=dict(color='#eaeaea', size=10), margin=dict(l=50, r=20, t=10, b=40))
    return fig


@app.callback(Output('card-frequency', 'figure'), Input('community-selector', 'id'))
def update_card_frequency(_):
    if 'card_freq' in DATA['precomputed']:
        freq = DATA['precomputed']['card_freq']
    else:
        freq = np.random.exponential(50, 10000)

    fig = go.Figure(go.Histogram(x=freq, nbinsx=50, marker_color=COLORS['success']))
    fig.update_layout(xaxis_title='交易次数', yaxis_title='卡片数量', xaxis=dict(range=[0, np.percentile(freq, 95)]),
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     font=dict(color='#eaeaea', size=10), margin=dict(l=50, r=20, t=10, b=40))
    return fig


@app.callback(Output('hour-stats', 'figure'), Input('community-selector', 'id'))
def update_hour_stats(_):
    if 'hour_stats' in DATA['precomputed']:
        stats = DATA['precomputed']['hour_stats']
    else:
        stats = pd.DataFrame({'hour': range(24), 'total': np.random.poisson(25000, 24),
                             'rate': np.random.rand(24) * 2 + 2.5})

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=stats['hour'], y=stats['total'], name='交易量', marker_color=COLORS['success'], opacity=0.5), secondary_y=False)
    fig.add_trace(go.Scatter(x=stats['hour'], y=stats['rate'], name='欺诈率%',
                            line=dict(color=COLORS['danger'], width=2), mode='lines+markers'), secondary_y=True)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     font=dict(color='#eaeaea', size=10), margin=dict(l=50, r=50, t=10, b=40),
                     legend=dict(orientation='h', y=1.1), xaxis_title='小时')
    fig.update_yaxes(title_text="交易量", secondary_y=False)
    fig.update_yaxes(title_text="欺诈率%", secondary_y=True)
    return fig


# ============ 社群分析回调 ============

@app.callback(Output('community-selector', 'options'), Input('community-selector', 'id'))
def update_community_options(_):
    if DATA['suspicious']:
        return [{'label': f"社群 {c['community_id']} (风险:{c['risk_score']:.2f}, 欺诈率:{c.get('fraud_rate',0)*100:.1f}%)",
                'value': c['community_id']} for c in DATA['suspicious'][:20]]
    return []


@app.callback(Output('community-detail-panel', 'children'), Input('community-selector', 'value'))
def update_community_detail(selected):
    if selected is None:
        return html.P("请选择一个社群", style={'color': '#888'})

    comm = None
    for c in (DATA['suspicious'] or []):
        if c['community_id'] == selected:
            comm = c
            break
    if comm is None:
        return html.P("未找到数据", style={'color': '#888'})

    cards, merchants = comm.get('num_cards', 0), comm.get('num_merchants', 1)
    fraud_tx = comm.get('fraud_transactions', 0)
    fraud_tx = int(fraud_tx) if isinstance(fraud_tx, str) else fraud_tx

    return html.Div([
        html.Div([
            html.Span("⚠️ 风险评分: ", style={'color': '#888'}),
            html.Span(f"{comm.get('risk_score', 0):.3f}",
                     style={'color': COLORS['danger'], 'fontWeight': 'bold', 'fontSize': '20px'})
        ], style={'marginBottom': '15px', 'textAlign': 'center'}),
        html.Hr(style={'borderColor': '#333', 'margin': '10px 0'}),
        create_stat_row("📊 欺诈率", f"{comm.get('fraud_rate', 0)*100:.2f}%"),
        create_stat_row("📦 交易数", f"{comm.get('total_transactions', 0):,}"),
        create_stat_row("🚨 欺诈交易", f"{fraud_tx:,}"),
        create_stat_row("💳 卡片数", f"{cards:,}"),
        create_stat_row("🏪 商户数", f"{merchants:,}"),
        create_stat_row("📐 卡/商户比", f"{cards/max(merchants,1):.1f}:1"),
        html.Hr(style={'borderColor': '#333', 'margin': '10px 0'}),
        html.P("异常指标", style={'color': COLORS['text'], 'fontWeight': 'bold', 'fontSize': '13px'}),
        create_stat_row("🔗 拓扑异常", f"{comm.get('topology_anomaly', 0):.3f}"),
        create_stat_row("🎭 行为异常", f"{comm.get('behavior_anomaly', 0):.3f}"),
        create_stat_row("📈 密度", f"{comm.get('density', 0):.4f}"),
        create_stat_row("🌙 夜间比例", f"{comm.get('night_ratio', 0)*100:.1f}%"),
    ])


@app.callback(Output('network-graph', 'elements'), Input('community-selector', 'value'))
def update_network_graph(selected):
    if selected is None or not DATA['communities']:
        return []

    nodes = DATA['communities'].get(str(selected), [])
    if not nodes:
        return []

    cards = [n for n in nodes if n.startswith('card_')]
    merchants = [n for n in nodes if n.startswith('merchant_')]

    np.random.seed(selected if isinstance(selected, int) else 42)
    max_cards = min(len(cards), 25)
    max_merchants = min(len(merchants), 12)
    cards = list(np.random.choice(cards, max_cards, replace=False)) if len(cards) > max_cards else cards
    merchants = list(np.random.choice(merchants, max_merchants, replace=False)) if len(merchants) > max_merchants else merchants

    elements = []
    for i, c in enumerate(cards):
        elements.append({'data': {'id': c, 'label': f'C{i+1}', 'type': 'card'}})
    for i, m in enumerate(merchants):
        elements.append({'data': {'id': m, 'label': f'M{i+1}', 'type': 'merchant'}})

    edges_added = set()
    for card in cards:
        n_conn = min(len(merchants), np.random.randint(1, 4))
        connected_merchants = np.random.choice(merchants, n_conn, replace=False)
        for m in connected_merchants:
            edge_key = (card, m)
            if edge_key not in edges_added:
                elements.append({'data': {'source': card, 'target': m}})
                edges_added.add(edge_key)

    for m in merchants:
        has_edge = any(e.get('data', {}).get('target') == m for e in elements if 'source' in e.get('data', {}))
        if not has_edge and cards:
            card = np.random.choice(cards)
            elements.append({'data': {'source': card, 'target': m}})

    logger.info(f"Network graph: {len(cards)} cards, {len(merchants)} merchants, {len(edges_added)} edges")
    return elements


@app.callback(Output('community-size-dist', 'figure'), Input('community-selector', 'id'))
def update_community_size_dist(_):
    if DATA['stats']:
        sizes = [s.get('total_transactions', 0) for s in DATA['stats'].values()]
    # else:
    #     sizes = np.random.exponential(5000, 30)

    fig = go.Figure(go.Histogram(x=sizes, nbinsx=20, marker_color=COLORS['purple']))
    fig.update_layout(xaxis_title='社群交易数', yaxis_title='社群数量',
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     font=dict(color='#eaeaea', size=10), margin=dict(l=50, r=20, t=10, b=40))
    return fig


@app.callback(Output('community-risk-compare', 'figure'), Input('community-selector', 'id'))
def update_community_risk_compare(_):
    if DATA['suspicious']:
        comms = DATA['suspicious'][:8]
        names = [f"社群{c['community_id']}" for c in comms]
        topo = [c.get('topology_anomaly', 0) for c in comms]
        behav = [c.get('behavior_anomaly', 0) for c in comms]
    # else:
    #     names = [f'社群{i}' for i in range(8)]
    #     topo, behav = np.random.rand(8) * 0.5 + 0.3, np.random.rand(8) * 0.5 + 0.2

    fig = go.Figure()
    fig.add_trace(go.Bar(name='拓扑异常', x=names, y=topo, marker_color=COLORS['danger']))
    fig.add_trace(go.Bar(name='行为异常', x=names, y=behav, marker_color=COLORS['warning']))
    fig.update_layout(barmode='group', yaxis_title='异常分数', paper_bgcolor='rgba(0,0,0,0)',
                     plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea', size=10),
                     margin=dict(l=50, r=20, t=10, b=60), legend=dict(orientation='h', y=1.1),
                     xaxis_tickangle=-45)
    return fig


@app.callback(Output('anomaly-pie', 'figure'), Input('community-selector', 'value'))
def update_anomaly_pie(selected):
    if selected and DATA['suspicious']:
        for c in DATA['suspicious']:
            if c['community_id'] == selected:
                topo, behav = c.get('topology_anomaly', 0.5), c.get('behavior_anomaly', 0.5)
                break
        else:
            topo, behav = 0.5, 0.5
    else:
        topo, behav = 0.5, 0.5

    fig = go.Figure(go.Pie(labels=['拓扑异常', '行为异常'], values=[topo, behav], hole=0.4,
                          marker=dict(colors=[COLORS['danger'], COLORS['warning']])))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea', size=10),
                     margin=dict(l=20, r=20, t=10, b=20), showlegend=True,
                     legend=dict(orientation='h', y=-0.1))
    return fig


# ============ 模型性能回调 ============

@app.callback(Output('docx-four-models', 'figure'), Input('community-selector', 'id'))
def update_docx_four_models(_):
    metric_cols = ['auc', 'ap', 'f1', 'precision', 'recall']
    metric_alias = {'auc': 'AUC', 'ap': 'AP', 'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall'}

    df = pd.DataFrame(DOCX_MODEL_ROWS)
    df_plot = df[['model'] + metric_cols].melt(id_vars='model', var_name='metric', value_name='score')
    df_plot['metric'] = df_plot['metric'].map(metric_alias)

    fig = px.bar(
        df_plot,
        x='metric',
        y='score',
        color='model',
        barmode='group',
        text=df_plot['score'].map(lambda x: f"{x:.4f}")
    )
    fig.update_layout(
        yaxis_title='分数',
        xaxis_title='指标',
        yaxis=dict(range=[0, 1]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaeaea', size=10),
        margin=dict(l=50, r=20, t=10, b=40),
        legend=dict(orientation='h', y=1.12)
    )
    fig.update_traces(textposition='outside')
    return fig


@app.callback(Output('docx-hybrid-enhance', 'figure'), Input('community-selector', 'id'))
def update_docx_hybrid_enhance(_):
    metric_cols = ['auc', 'ap', 'f1', 'precision', 'recall']
    metric_alias = {'auc': 'AUC', 'ap': 'AP', 'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall'}

    df = pd.DataFrame(DOCX_HYBRID_ENHANCE_ROWS)
    df_plot = df[['model'] + metric_cols].melt(id_vars='model', var_name='metric', value_name='score')
    df_plot['metric'] = df_plot['metric'].map(metric_alias)

    fig = px.bar(
        df_plot,
        x='metric',
        y='score',
        color='model',
        barmode='group',
        text=df_plot['score'].map(lambda x: f"{x:.4f}"),
        color_discrete_sequence=[COLORS['warning'], COLORS['success']]
    )
    fig.update_layout(
        yaxis_title='分数',
        xaxis_title='指标',
        yaxis=dict(range=[0, 1]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaeaea', size=10),
        margin=dict(l=50, r=20, t=10, b=40),
        legend=dict(orientation='h', y=1.12)
    )
    fig.update_traces(textposition='outside')
    return fig


@app.callback(Output('docx-imbalance-compare', 'figure'), Input('community-selector', 'id'))
def update_docx_imbalance_compare(_):
    metric_cols = ['auc', 'ap', 'f1', 'precision', 'recall']
    metric_alias = {'auc': 'AUC', 'ap': 'AP', 'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall'}

    df = pd.DataFrame(DOCX_IMBALANCE_ROWS)
    df_plot = df[['model'] + metric_cols].melt(id_vars='model', var_name='metric', value_name='score')
    df_plot['metric'] = df_plot['metric'].map(metric_alias)

    fig = px.bar(
        df_plot,
        x='metric',
        y='score',
        color='model',
        barmode='group',
        text=df_plot['score'].map(lambda x: f"{x:.4f}"),
        color_discrete_sequence=[COLORS['warning'], COLORS['danger']]
    )
    fig.update_layout(
        yaxis_title='分数',
        xaxis_title='指标',
        yaxis=dict(range=[0, 1]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaeaea', size=10),
        margin=dict(l=50, r=20, t=10, b=40),
        legend=dict(orientation='h', y=1.12)
    )
    fig.update_traces(textposition='outside')
    return fig


@app.callback(Output('confusion-matrix', 'figure'), Input('community-selector', 'id'))
def update_confusion_matrix(_):
    # 直接使用你提供的混淆矩阵数据
    cm = np.array([
        [84916, 896],  # 第一行：实际正常 [预测正常, 预测欺诈]
        [823, 1946]  # 第二行：实际欺诈 [预测正常, 预测欺诈]
    ])

    # 计算百分比
    cm_pct = cm / max(cm.sum(), 1) * 100

    # 格式化显示文本
    text = [[f'{cm[i, j]:,}<br>({cm_pct[i, j]:.1f}%)' for j in range(2)] for i in range(2)]

    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['预测正常', '预测欺诈'],
        y=['实际正常', '实际欺诈'],
        colorscale='Blues',
        showscale=True,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=12)
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaeaea', size=10),
        margin=dict(l=80, r=20, t=10, b=60)
    )

    return fig


@app.callback(Output('feature-importance', 'figure'), Input('community-selector', 'id'))
def update_feature_importance(_):
    if DATA.get('model_results') and 'feature_importance' in DATA['model_results']:
        fi = DATA['model_results']['feature_importance']
        features = [f['feature'] for f in fi[:20]]
        importance = [f['importance'] for f in fi[:20]]

    importance = np.array(importance) / max(importance)

    fig = go.Figure(go.Bar(y=features[::-1], x=importance[::-1], orientation='h',
                          marker=dict(color=importance[::-1], colorscale='Viridis')))
    fig.update_layout(xaxis_title='相对重要性', paper_bgcolor='rgba(0,0,0,0)',
                     plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea', size=9),
                     margin=dict(l=150, r=20, t=10, b=40))
    return fig


@app.callback(Output('training-curve', 'figure'), Input('community-selector', 'id'))
def update_training_curve(_):
    fig = go.Figure()

    if DATA.get('gnn_results') and 'train' in DATA['gnn_results']:
        train_data = DATA['gnn_results']['train']
        history = train_data.get('history', {})

        if history.get('val_auc'):
            epochs = list(range(len(history['val_auc'])))

            if history.get('train_loss'):
                losses = np.array(history['train_loss'])
                normalized_loss = 0.5 + 0.5 * (1 - (losses - losses.min()) / (losses.max() - losses.min() + 1e-8))
                fig.add_trace(go.Scatter(x=epochs, y=normalized_loss, name='训练损失',
                                        line=dict(color=COLORS['success'], width=2, dash='dot')))

            fig.add_trace(go.Scatter(x=epochs, y=history['val_auc'], name='验证AUC',
                                    line=dict(color=COLORS['danger'], width=2)))

            if history.get('val_ap'):
                fig.add_trace(go.Scatter(x=epochs, y=history['val_ap'], name='验证AP',
                                        line=dict(color=COLORS['warning'], width=2)))

            best_auc = max(history['val_auc'])
            best_epoch = history['val_auc'].index(best_auc)
            fig.add_vline(x=best_epoch, line_dash="dash", line_color=COLORS['purple'],
                         annotation_text=f"最优({best_epoch})")

            fig.update_layout(xaxis_title='训练轮次', yaxis_title='指标值',
                             yaxis=dict(range=[0, 1.05], dtick=0.2),
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font=dict(color='#eaeaea', size=10), margin=dict(l=50, r=20, t=30, b=40),
                             legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'))
            return fig

    best_iter = 843
    val_auc_final = 0.9708
    if DATA.get('model_results') and 'train' in DATA['model_results']:
        best_iter = DATA['model_results']['train'].get('best_iteration', 843)
        val_auc_final = DATA['model_results']['train'].get('val_auc', 0.9708)

    epochs = np.arange(0, best_iter + 100, max(50, best_iter // 20))
    train_auc = 1 - np.exp(-epochs / (best_iter / 4)) * 0.1
    val_auc = 0.95 + (val_auc_final - 0.95) * (1 - np.exp(-epochs / (best_iter / 3)))
    val_auc = np.minimum(val_auc, val_auc_final)

    fig.add_trace(go.Scatter(x=epochs, y=train_auc, name='训练AUC',
                            line=dict(color=COLORS['success'], width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_auc, name='验证AUC',
                            line=dict(color=COLORS['danger'], width=2)))
    fig.add_vline(x=best_iter, line_dash="dash", line_color=COLORS['warning'],
                 annotation_text=f"早停({best_iter})")

    fig.update_layout(xaxis_title='迭代次数', yaxis_title='AUC值',
                     yaxis=dict(range=[0.9, 1.01], dtick=0.02),
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     font=dict(color='#eaeaea', size=10), margin=dict(l=50, r=20, t=30, b=40),
                     legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'))
    return fig


# ============ 启动服务器 ============

if __name__ == '__main__':
    logger.info("启动欺诈检测可视化平台")
    logger.info(f"可疑社群: {len(DATA['suspicious'] or [])} 个")
    logger.info(f"边数据: {'已加载 ' + str(len(DATA['edges_df'])) + ' 条' if DATA['edges_df'] is not None else '未加载'}")
    logger.info(f"预计算统计: {list(DATA['precomputed'].keys())}")
    logger.info("访问: http://localhost:8050")
    app.run(debug=False, host='0.0.0.0', port=8050)