"""
对比实验模块
提供批量实验运行、指标收集和结果导出功能
"""

from .config import DEFAULT_PARAMS, VARIABLE_RANGES, NUM_RUNS, METRICS
from .batch_runner import BatchExperimentRunner
from .metrics import MetricsCollector

__all__ = [
    'DEFAULT_PARAMS',
    'VARIABLE_RANGES',
    'NUM_RUNS',
    'METRICS',
    'BatchExperimentRunner',
    'MetricsCollector',
]
