"""
数据处理模块
包含Spark数据处理、特征工程和PyG图构建
"""

# PyG图构建（总是可用）
from .graph_builder import PyGGraphBuilder, build_graph_from_parquet#相对导入，只能在包内导入


# Spark处理（可选，服务器上可能没装pyspark）
try:
    from .spark_processor import SparkDataProcessor, process_ieee_data
    __all__ = [
        'SparkDataProcessor', 
        'process_ieee_data',
        'PyGGraphBuilder',
        'build_graph_from_parquet'
    ]
except ImportError:
    __all__ = [
        'PyGGraphBuilder',
        'build_graph_from_parquet'
    ]
