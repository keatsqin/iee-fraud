"""
Spark数据处理模块
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, IntegerType, StringType
import os
from loguru import logger


class SparkDataProcessor:
    """Spark数据处理器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.spark = self._create_spark_session()
        self.df = None
        self.card_features = None
        self.merchant_features = None
        
    def _create_spark_session(self) -> SparkSession:
        """创建Spark会话"""
        spark_config = self.config.get('spark', {})
        
        spark = SparkSession.builder \
            .appName(spark_config.get('app_name', 'FraudDetectionGNN')) \
            .master(spark_config.get('master', 'local[*]')) \
            .config("spark.driver.memory", spark_config.get('driver_memory', '12g')) \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Spark session created: {spark.version}")
        return spark
    
    def load_data(self, transaction_path: str, identity_path: str = None) -> 'SparkDataProcessor':
        """加载交易数据和身份数据"""
        logger.info(f"Loading transaction data from {transaction_path}")
        
        self.df = self.spark.read.csv(transaction_path, header=True, inferSchema=True)
        logger.info(f"Transaction data loaded: {self.df.count()} rows")
        
        if identity_path and os.path.exists(identity_path):
            logger.info(f"Loading identity data from {identity_path}")
            identity_df = self.spark.read.csv(identity_path, header=True, inferSchema=True)
            self.df = self.df.join(identity_df, on='TransactionID', how='left')
            logger.info(f"Identity data merged")
        
        return self
    
    def clean_data(self) -> 'SparkDataProcessor':
        """数据清洗和基础特征"""
        logger.info("Cleaning data...")
        
        # 创建Card ID和Merchant ID
        self.df = self.df.withColumn('card_id', F.col('card1').cast('string'))
        self.df = self.df.withColumn(
            'merchant_id',
            F.concat_ws('_', 
                F.col('ProductCD').cast('string'),
                F.coalesce(F.col('addr1').cast('string'), F.lit('-1'))
            )
        )
        
        # UID特征
        self.df = self.df.withColumn(
            'uid',
            F.concat_ws('_',
                F.col('card1').cast('string'),
                F.coalesce(F.col('addr1').cast('string'), F.lit('-1')),
                F.coalesce(F.col('P_emaildomain'), F.lit('unknown'))
            )
        )
        
        self.df = self.df.withColumn(
            'uid2',
            F.concat_ws('_',
                F.col('card1').cast('string'),
                F.coalesce(F.col('addr1').cast('string'), F.lit('-1')),
                F.coalesce(F.col('D1').cast('string'), F.lit('-1'))
            )
        )
        
        # 时间特征
        self.df = self.df.withColumn('hour', (F.col('TransactionDT') / 3600 % 24).cast(IntegerType()))
        self.df = self.df.withColumn('day_of_week', (F.col('TransactionDT') / 86400 % 7).cast(IntegerType()))
        self.df = self.df.withColumn('day', (F.col('TransactionDT') / 86400).cast(IntegerType()))
        self.df = self.df.withColumn('is_night', F.when((F.col('hour') >= 0) & (F.col('hour') < 6), 1).otherwise(0))
        self.df = self.df.withColumn('is_weekend', F.when(F.col('day_of_week').isin([5, 6]), 1).otherwise(0))
        
        # 金额特征
        self.df = self.df.withColumn('amt_decimal', F.col('TransactionAmt') - F.floor(F.col('TransactionAmt')))
        self.df = self.df.withColumn('is_round_amt', F.when(F.col('amt_decimal') < 0.01, 1).otherwise(0))
        self.df = self.df.withColumn('log_amt', F.log1p(F.col('TransactionAmt')))
        
        # 填充缺失值
        numeric_cols = ['TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 
                       'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df = self.df.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))
        
        logger.info("Data cleaning completed")
        return self
    
    def compute_time_window_features(self) -> 'SparkDataProcessor':
        """计算时间窗口特征"""
        logger.info("Computing time window features...")
        
        # 定义时间窗口（秒）
        HOUR_1 = 3600
        HOUR_6 = 6 * 3600
        HOUR_24 = 24 * 3600
        DAY_7 = 7 * 24 * 3600
        
        # 按card_id和时间排序的窗口
        card_time_window = Window.partitionBy('card_id').orderBy('TransactionDT')
        
        # 1小时内的历史交易统计
        card_1h_window = Window.partitionBy('card_id').orderBy('TransactionDT').rangeBetween(-HOUR_1, -1)
        # 6小时内
        card_6h_window = Window.partitionBy('card_id').orderBy('TransactionDT').rangeBetween(-HOUR_6, -1)
        # 24小时内
        card_24h_window = Window.partitionBy('card_id').orderBy('TransactionDT').rangeBetween(-HOUR_24, -1)
        
        # Card时间窗口特征
        self.df = self.df.withColumn('card_tx_1h', F.count('*').over(card_1h_window))
        self.df = self.df.withColumn('card_tx_6h', F.count('*').over(card_6h_window))
        self.df = self.df.withColumn('card_tx_24h', F.count('*').over(card_24h_window))
        
        self.df = self.df.withColumn('card_amt_1h', F.coalesce(F.sum('TransactionAmt').over(card_1h_window), F.lit(0.0)))
        self.df = self.df.withColumn('card_amt_6h', F.coalesce(F.sum('TransactionAmt').over(card_6h_window), F.lit(0.0)))
        self.df = self.df.withColumn('card_amt_24h', F.coalesce(F.sum('TransactionAmt').over(card_24h_window), F.lit(0.0)))
        
        # 平均金额
        self.df = self.df.withColumn('card_avg_amt_1h', 
            F.when(F.col('card_tx_1h') > 0, F.col('card_amt_1h') / F.col('card_tx_1h')).otherwise(0))
        self.df = self.df.withColumn('card_avg_amt_24h', 
            F.when(F.col('card_tx_24h') > 0, F.col('card_amt_24h') / F.col('card_tx_24h')).otherwise(0))
        
        # 当前金额与历史平均的比值
        self.df = self.df.withColumn('amt_ratio_1h',
            F.when(F.col('card_avg_amt_1h') > 0, F.col('TransactionAmt') / F.col('card_avg_amt_1h')).otherwise(1))
        self.df = self.df.withColumn('amt_ratio_24h',
            F.when(F.col('card_avg_amt_24h') > 0, F.col('TransactionAmt') / F.col('card_avg_amt_24h')).otherwise(1))
        
        # Merchant时间窗口特征
        merchant_1h_window = Window.partitionBy('merchant_id').orderBy('TransactionDT').rangeBetween(-HOUR_1, -1)
        merchant_24h_window = Window.partitionBy('merchant_id').orderBy('TransactionDT').rangeBetween(-HOUR_24, -1)
        
        self.df = self.df.withColumn('merchant_tx_1h', F.count('*').over(merchant_1h_window))
        self.df = self.df.withColumn('merchant_tx_24h', F.count('*').over(merchant_24h_window))
        self.df = self.df.withColumn('merchant_unique_cards_1h', F.approx_count_distinct('card_id').over(merchant_1h_window))
        
        # 交易序号（第几笔交易）
        self.df = self.df.withColumn('card_tx_seq', F.row_number().over(card_time_window))
        
        # 距离上一笔交易的时间间隔
        self.df = self.df.withColumn('prev_tx_dt', F.lag('TransactionDT').over(card_time_window))
        self.df = self.df.withColumn('time_since_last_tx', 
            F.coalesce(F.col('TransactionDT') - F.col('prev_tx_dt'), F.lit(999999)))
        
        # 是否是新卡（第一笔交易）
        self.df = self.df.withColumn('is_new_card', F.when(F.col('card_tx_seq') == 1, 1).otherwise(0))
        
        # 短时间内多笔交易（burst activity）
        self.df = self.df.withColumn('is_burst', F.when(F.col('card_tx_1h') >= 3, 1).otherwise(0))
        
        # UID时间窗口特征
        uid_time_window = Window.partitionBy('uid').orderBy('TransactionDT')
        uid_1h_window = Window.partitionBy('uid').orderBy('TransactionDT').rangeBetween(-HOUR_1, -1)
        uid_24h_window = Window.partitionBy('uid').orderBy('TransactionDT').rangeBetween(-HOUR_24, -1)
        
        self.df = self.df.withColumn('uid_tx_1h', F.count('*').over(uid_1h_window))
        self.df = self.df.withColumn('uid_tx_24h', F.count('*').over(uid_24h_window))
        self.df = self.df.withColumn('uid_amt_24h', F.coalesce(F.sum('TransactionAmt').over(uid_24h_window), F.lit(0.0)))
        self.df = self.df.withColumn('uid_tx_seq', F.row_number().over(uid_time_window))
        
        logger.info("Time window features computed!")
        return self
    
    def compute_frequency_encoding(self) -> 'SparkDataProcessor':
        """频率编码"""
        logger.info("Computing frequency encoding...")
        
        # 只对最重要的3个类别做频率编码
        cat_cols = ['card1', 'addr1', 'P_emaildomain']
        
        for col in cat_cols:
            if col in self.df.columns:
                # 计算频率
                freq_df = self.df.groupBy(col).count().withColumnRenamed('count', f'{col}_freq')
                self.df = self.df.join(F.broadcast(freq_df), on=col, how='left')
                self.df = self.df.withColumn(f'{col}_freq', F.coalesce(F.col(f'{col}_freq'), F.lit(0)))
                # 缓存以减少重复计算
                self.df = self.df.cache()
        
        logger.info("Frequency encoding completed!")
        return self
    
    def compute_target_encoding(self) -> 'SparkDataProcessor':
        """Target Encoding"""
        logger.info("Computing target encoding...")
        
        # 按时间排序，取前70%作为计算基础
        total_count = self.df.count()
        train_cutoff = int(total_count * 0.7)
        
        # 获取时间阈值
        time_threshold = self.df.orderBy('TransactionDT').limit(train_cutoff) \
            .agg(F.max('TransactionDT')).collect()[0][0]
        
        # 只用历史数据计算欺诈率
        train_df = self.df.filter(F.col('TransactionDT') <= time_threshold).cache()
        
        # 全局欺诈率
        global_fraud_rate = train_df.agg(F.mean('isFraud')).collect()[0][0]
        k = 10  # 平滑参数
        
        # 只对最重要的2个类别计算target encoding
        target_cols = ['card1', 'uid']
        
        for col in target_cols:
            if col in self.df.columns:
                # 计算该类别的历史欺诈率
                fraud_rate_df = train_df.groupBy(col).agg(
                    F.mean('isFraud').alias(f'{col}_fraud_rate'),
                    F.count('*').alias(f'{col}_count_te')
                )
                
                # 贝叶斯平滑
                fraud_rate_df = fraud_rate_df.withColumn(
                    f'{col}_te',
                    (F.col(f'{col}_fraud_rate') * F.col(f'{col}_count_te') + global_fraud_rate * k) / 
                    (F.col(f'{col}_count_te') + k)
                ).select(col, f'{col}_te')
                
                # 使用broadcast join减少内存
                self.df = self.df.join(F.broadcast(fraud_rate_df), on=col, how='left')
                self.df = self.df.withColumn(f'{col}_te', F.coalesce(F.col(f'{col}_te'), F.lit(global_fraud_rate)))
        
        train_df.unpersist()
        logger.info("Target encoding completed!")
        return self

    def compute_card_features(self) -> DataFrame:
        """计算Card节点特征"""
        logger.info("Computing card features...")
        
        card_features = self.df.groupBy('card_id').agg(
            # 基础统计
            F.count('*').alias('tx_count'),
            F.mean('TransactionAmt').alias('amt_mean'),
            F.stddev('TransactionAmt').alias('amt_std'),
            F.max('TransactionAmt').alias('amt_max'),
            F.min('TransactionAmt').alias('amt_min'),
            F.mean('is_round_amt').alias('amt_decimal_ratio'),
            F.mean('log_amt').alias('log_amt_mean'),
            
            # 交易多样性
            F.countDistinct('merchant_id').alias('unique_merchants'),
            F.countDistinct('ProductCD').alias('unique_products'),
            F.countDistinct('addr1').alias('unique_addr'),
            
            # 时间特征
            F.mean('is_night').alias('night_ratio'),
            F.mean('is_weekend').alias('weekend_ratio'),
            (F.max('TransactionDT') - F.min('TransactionDT')).alias('time_span'),
            F.stddev('TransactionDT').alias('dt_std'),
            F.stddev('hour').alias('hour_std'),
            
            # 时间窗口特征聚合
            F.mean('card_tx_1h').alias('avg_tx_1h'),
            F.mean('card_tx_24h').alias('avg_tx_24h'),
            F.max('card_tx_1h').alias('max_tx_1h'),
            F.max('card_tx_24h').alias('max_tx_24h'),
            F.mean('amt_ratio_24h').alias('avg_amt_ratio'),
            F.max('amt_ratio_24h').alias('max_amt_ratio'),
            F.mean('is_burst').alias('burst_ratio'),
            F.mean('time_since_last_tx').alias('avg_time_between_tx'),
            F.min('time_since_last_tx').alias('min_time_between_tx'),
            
            # Target Encoding特征
            F.first('card1_te').alias('card1_te'),
            
            # 欺诈率（注意：这个在实际应用中不能用，但训练时可以作为参考）
            F.mean('isFraud').alias('fraud_rate'),
        )
        
        card_features = card_features.fillna(0)
        self.card_features = card_features
        logger.info(f"Card features computed: {card_features.count()} cards, {len(card_features.columns)} features")
        return card_features
    
    def compute_merchant_features(self) -> DataFrame:
        """计算Merchant节点特征"""
        logger.info("Computing merchant features...")
        
        merchant_features = self.df.groupBy('merchant_id').agg(
            F.count('*').alias('tx_count'),
            F.countDistinct('card_id').alias('unique_cards'),
            F.countDistinct('uid').alias('unique_uids'),
            F.mean('TransactionAmt').alias('amt_mean'),
            F.stddev('TransactionAmt').alias('amt_std'),
            F.max('TransactionAmt').alias('amt_max'),
            F.mean('isFraud').alias('fraud_rate'),
            (F.sum('TransactionAmt') / F.countDistinct('card_id')).alias('avg_amt_per_card'),
            F.stddev('TransactionDT').alias('dt_std'),
            F.mean('is_night').alias('night_ratio'),
            F.mean('is_round_amt').alias('round_amt_ratio'),
            F.mean('is_weekend').alias('weekend_ratio'),
            F.stddev('hour').alias('hour_std'),
            
            # 时间窗口特征
            F.mean('merchant_tx_1h').alias('avg_tx_1h'),
            F.mean('merchant_tx_24h').alias('avg_tx_24h'),
            F.max('merchant_tx_1h').alias('max_tx_1h'),
            F.max('merchant_tx_24h').alias('max_tx_24h'),
            F.mean('merchant_unique_cards_1h').alias('avg_unique_cards_1h'),
        )
        
        # 卡复用率
        card_tx_counts = self.df.groupBy('merchant_id', 'card_id').count()
        reuse_stats = card_tx_counts.groupBy('merchant_id').agg(
            (F.sum(F.when(F.col('count') > 1, 1).otherwise(0)) / F.count('*')).alias('card_reuse_rate')
        )
        merchant_features = merchant_features.join(reuse_stats, on='merchant_id', how='left')
        
        # 时间集中度
        max_dt_std = merchant_features.agg(F.max('dt_std')).collect()[0][0] or 1
        merchant_features = merchant_features.withColumn('time_concentration', 1 - F.col('dt_std') / max_dt_std)
        
        merchant_features = merchant_features.fillna(0)
        self.merchant_features = merchant_features
        logger.info(f"Merchant features computed: {merchant_features.count()} merchants, {len(merchant_features.columns)} features")
        return merchant_features
    
    def prepare_edges(self) -> DataFrame:
        """准备边数据"""
        logger.info("Preparing edge data...")
        
        # 基础边特征 - 添加ProductCD用于可视化
        edge_cols = [
            'TransactionID', 'card_id', 'merchant_id', 'TransactionAmt', 'amt_decimal',
            'TransactionDT', 'hour', 'day', 'day_of_week', 'is_night', 'is_weekend', 'isFraud',
            'log_amt', 'ProductCD',  # 添加ProductCD
            # 时间窗口特征（关键！）
            'card_tx_1h', 'card_tx_6h', 'card_tx_24h',
            'card_amt_1h', 'card_amt_24h',
            'amt_ratio_1h', 'amt_ratio_24h',
            'merchant_tx_1h', 'merchant_tx_24h',
            'card_tx_seq', 'time_since_last_tx',
            'is_new_card', 'is_burst',
            # UID特征
            'uid_tx_1h', 'uid_tx_24h', 'uid_amt_24h', 'uid_tx_seq',
            # Target Encoding（简化版）
            'card1_te', 'uid_te',
            # 频率编码（简化版）
            'card1_freq', 'addr1_freq', 'P_emaildomain_freq',
        ]
        
        # C系列特征
        c_cols = [f'C{i}' for i in range(1, 15)]
        # D系列特征
        d_cols = [f'D{i}' for i in range(1, 16)]
        # V系列特征（选择重要的）
        v_cols = [f'V{i}' for i in range(12, 35)] + \
                 [f'V{i}' for i in range(53, 75)] + \
                 [f'V{i}' for i in range(75, 95)]
        
        all_cols = edge_cols + c_cols + d_cols + v_cols
        existing_cols = [c for c in all_cols if c in self.df.columns]
        edges = self.df.select(existing_cols)
        edges = edges.fillna(0)
        
        logger.info(f"Edge data prepared: {edges.count()} edges, {len(existing_cols)} features")
        return edges

    def compute_graph_stats(self) -> dict:
        """计算图统计"""
        logger.info("Computing graph statistics...")
        
        stats = {}
        stats['num_cards'] = self.df.select('card_id').distinct().count()
        stats['num_merchants'] = self.df.select('merchant_id').distinct().count()
        stats['num_edges'] = self.df.count()
        
        fraud_stats = self.df.agg(
            F.sum('isFraud').alias('fraud_count'),
            F.mean('isFraud').alias('fraud_rate')
        ).collect()[0]
        stats['fraud_count'] = int(fraud_stats['fraud_count'])
        stats['fraud_rate'] = float(fraud_stats['fraud_rate'])
        
        card_degree = self.df.groupBy('card_id').count()
        degree_stats = card_degree.agg(
            F.mean('count').alias('avg_degree'),
            F.max('count').alias('max_degree')
        ).collect()[0]
        stats['card_avg_degree'] = float(degree_stats['avg_degree'])
        stats['card_max_degree'] = int(degree_stats['max_degree'])
        
        logger.info(f"Graph stats: {stats['num_cards']} cards, {stats['num_merchants']} merchants, "
                   f"{stats['num_edges']} edges, fraud rate: {stats['fraud_rate']:.4f}")
        return stats
    
    def export_to_parquet(self, output_dir: str):
        """导出数据"""
        import json
        import platform
        
        os.makedirs(output_dir, exist_ok=True)
        use_csv = platform.system() == 'Windows'
        
        # Card特征
        if self.card_features is not None:
            if use_csv:
                card_path = os.path.join(output_dir, 'card_features.csv')
                self.card_features.toPandas().to_csv(card_path, index=False)
            else:
                card_path = os.path.join(output_dir, 'card_features.parquet')
                self.card_features.write.mode('overwrite').parquet(card_path)
            logger.info(f"Card features exported to {card_path}")
        
        # Merchant特征
        if self.merchant_features is not None:
            if use_csv:
                merchant_path = os.path.join(output_dir, 'merchant_features.csv')
                self.merchant_features.toPandas().to_csv(merchant_path, index=False)
            else:
                merchant_path = os.path.join(output_dir, 'merchant_features.parquet')
                self.merchant_features.write.mode('overwrite').parquet(merchant_path)
            logger.info(f"Merchant features exported to {merchant_path}")
        
        # 边数据
        edges = self.prepare_edges()
        if use_csv:
            edges_path = os.path.join(output_dir, 'edges.csv')
            logger.info("Converting edges to CSV...")
            edges.toPandas().to_csv(edges_path, index=False)
        else:
            edges_path = os.path.join(output_dir, 'edges.parquet')
            edges.write.mode('overwrite').parquet(edges_path)
        logger.info(f"Edges exported to {edges_path}")
        
        # 图统计
        stats = self.compute_graph_stats()
        stats_path = os.path.join(output_dir, 'graph_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Graph stats exported to {stats_path}")
        
        return output_dir
    
    def run_pipeline(self, transaction_path: str, identity_path: str, output_dir: str) -> str:
        """运行完整流水线"""
        logger.info("Starting Spark Pipeline")
        
        self.load_data(transaction_path, identity_path)
        self.clean_data()
        self.compute_time_window_features()
        # 频率编码
        self.compute_frequency_encoding()
        self.compute_target_encoding()
        self.compute_card_features()
        self.compute_merchant_features()
        self.export_to_parquet(output_dir)
        
        logger.info("Spark Pipeline Completed")
        return output_dir
    
    def stop(self):
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def process_ieee_data(transaction_path, identity_path=None, output_dir="data/processed", spark_config=None):
    config = {'spark': spark_config or {'app_name': 'FraudDetectionGNN', 'master': 'local[*]', 'driver_memory': '12g'}}
    processor = SparkDataProcessor(config)
    try:
        return processor.run_pipeline(transaction_path, identity_path, output_dir)
    finally:
        processor.stop()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python spark_processor.py <transaction_csv> [identity_csv] [output_dir]")
        sys.exit(1)
    
    transaction_path = sys.argv[1]
    identity_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/processed"
    process_ieee_data(transaction_path, identity_path, output_dir)
