"""
训练模块
包含监督和无监督训练器
"""
from .trainer import FraudDetectionTrainer, UnsupervisedTrainer

__all__ = ['FraudDetectionTrainer', 'UnsupervisedTrainer']
