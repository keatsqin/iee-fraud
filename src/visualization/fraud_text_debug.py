# 诊断脚本：查找真实的欺诈交易数据
import pandas as pd
import numpy as np

# 加载数据
DATA_DIR = "data/processed"
edges_df = pd.read_parquet(f"{DATA_DIR}/edges.parquet")
card_df = pd.read_parquet(f"{DATA_DIR}/card_features.parquet")
merchant_df = pd.read_parquet(f"{DATA_DIR}/merchant_features.parquet")

# 1. 查看欺诈交易的真实特征分布
fraud_txs = edges_df[edges_df['isFraud'] == 1]
normal_txs = edges_df[edges_df['isFraud'] == 0]

print("=" * 60)
print("欺诈交易 vs 正常交易 特征对比")
print("=" * 60)

# 关键特征对比
features = ['TransactionAmt', 'hour', 'card_tx_1h', 'card_tx_24h',
            'time_since_last_tx', 'is_new_card', 'is_burst']

for feat in features:
    if feat in edges_df.columns:
        fraud_mean = fraud_txs[feat].mean()
        normal_mean = normal_txs[feat].mean()
        print(f"{feat:20s}: 欺诈={fraud_mean:.2f} | 正常={normal_mean:.2f}")

print("\n" + "=" * 60)
print("真实欺诈交易样本（前5条）")
print("=" * 60)

# 显示真实的欺诈交易
fraud_samples = fraud_txs[['card_id', 'merchant_id', 'TransactionAmt', 'hour',
                           'card_tx_1h', 'card_tx_24h', 'time_since_last_tx',
                           'card_tx_seq', 'is_new_card', 'is_burst']].head(5)

print(fraud_samples.to_string())

print("\n" + "=" * 60)
print("可用于测试的欺诈卡片ID")
print("=" * 60)

# 找出有欺诈记录的卡片
fraud_cards = fraud_txs['card_id'].unique()[:10]
print("欺诈卡片示例:", list(fraud_cards))

print("\n" + "=" * 60)
print("可用于测试的欺诈商户ID")
print("=" * 60)

# 找出有欺诈记录的商户
fraud_merchants = fraud_txs['merchant_id'].unique()[:10]
print("欺诈商户示例:", list(fraud_merchants))