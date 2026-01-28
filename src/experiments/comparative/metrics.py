"""
指标收集模块
提供从算法结果中提取指标、计算存活率和导出CSV的功能
支持多算法对比，按算法分目录保存结果
"""

import os
import csv
from typing import Dict, List, Any
import numpy as np

from .config import METRICS, VARIABLE_DIR_MAP, SUMMARY_FILE, ALGORITHMS


class MetricsCollector:
    """指标收集器"""

    def __init__(self, results_dir: str = None):
        """
        Args:
            results_dir: 结果保存目录
        """
        if results_dir is None:
            results_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'results'
            )
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def collect_metrics(self, results: Dict, problem: Any = None) -> Dict[str, float]:
        """
        从算法结果中提取5个评估指标

        Args:
            results: 算法返回的结果字典
            problem: RTMONFProblem实例（用于计算存活率）

        Returns:
            包含5个指标的字典
        """
        metrics = {
            'objective_value': results.get('utility', 0.0),
            'task_cost': results.get('total_cost', 0.0),
            'expected_completion_rate': results.get('completion_ratio', 0.0),
            'runtime_seconds': results.get('execution_time', 0.0),
            'avg_agent_survival_rate': self._compute_survival_rate(results, problem),
        }
        return metrics

    def _compute_survival_rate(self, results: Dict, problem: Any = None) -> float:
        """
        计算智能体平均存活率

        Args:
            results: 算法结果
            problem: 问题实例

        Returns:
            存活率 (0-1)
        """
        # 优先从failure_statistics获取
        if 'failure_statistics' in results:
            fs = results['failure_statistics']
            total = fs.get('total_agents', 0)
            failed = fs.get('num_failed', 0)
            isolated = fs.get('num_isolated', 0)
            if total > 0:
                functional = total - failed - isolated
                return functional / total

        # 从problem实例计算
        if problem is not None:
            total = len(problem.agents)
            if total > 0:
                functional = sum(1 for a in problem.agents.values() if a.is_functional)
                return functional / total

        return 1.0  # 默认全部存活

    def aggregate_results(self, results_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        聚合多次实验结果，计算平均值和95%置信区间

        Args:
            results_list: 多次实验的指标列表

        Returns:
            平均指标字典（包含均值、标准差和95%置信区间）
        """
        if not results_list:
            return {metric: 0.0 for metric in METRICS}

        n = len(results_list)
        aggregated = {}
        for metric in METRICS:
            values = [r.get(metric, 0.0) for r in results_list]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if n > 1 else 0.0
            # 95%置信区间: mean ± 1.96 * std / sqrt(n)
            ci_95 = 1.96 * std_val / np.sqrt(n) if n > 1 else 0.0
            aggregated[metric] = mean_val
            aggregated[f'{metric}_std'] = std_val
            aggregated[f'{metric}_ci95'] = ci_95

        return aggregated

    def export_variable_results(self,
                                 variable_name: str,
                                 results: Dict[Any, Dict[str, Dict[str, float]]]) -> List[str]:
        """
        导出单变量实验结果到CSV

        目录结构: results/{变量目录}/{算法名}_results.csv

        Args:
            variable_name: 变量名称
            results: {变量值: {算法名: 平均指标字典}}

        Returns:
            CSV文件路径列表
        """
        # 获取变量目录名
        var_dir = VARIABLE_DIR_MAP.get(variable_name, variable_name)
        var_dir_path = os.path.join(self.results_dir, var_dir)
        os.makedirs(var_dir_path, exist_ok=True)
        filepaths = []

        # 从结果中提取实际运行的算法列表
        actual_algorithms = set()
        for var_value in results.values():
            actual_algorithms.update(var_value.keys())

        # 只为实际运行的算法保存结果文件
        for algo in actual_algorithms:
            filepath = os.path.join(var_dir_path, f'{algo}_results.csv')

            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # 写入表头（包含均值和95%置信区间）
                header = ['variable_value']
                for m in METRICS:
                    header.extend([m, f'{m}_ci95'])
                writer.writerow(header)

                # 写入数据行
                for var_value in sorted(results.keys()):
                    algo_results = results[var_value]
                    metrics = algo_results.get(algo, {})
                    row = [var_value]
                    for m in METRICS:
                        row.append(metrics.get(m, 0.0))
                        row.append(metrics.get(f'{m}_ci95', 0.0))
                    writer.writerow(row)

            print(f"结果已保存到: {filepath}")
            filepaths.append(filepath)

        return filepaths

    def export_summary(self, all_results: Dict[str, Dict[Any, Dict[str, Dict[str, float]]]]) -> str:
        """
        导出所有实验的汇总结果

        Args:
            all_results: {变量名: {变量值: {算法名: 平均指标字典}}}

        Returns:
            CSV文件路径
        """
        filepath = os.path.join(self.results_dir, SUMMARY_FILE)

        # 从结果中提取实际运行的算法列表
        actual_algorithms = set()
        for var_results in all_results.values():
            for algo_results in var_results.values():
                actual_algorithms.update(algo_results.keys())

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入表头（包含均值和95%置信区间）
            header = ['experiment_type', 'variable_name', 'variable_value', 'algorithm']
            for m in METRICS:
                header.extend([m, f'{m}_ci95'])
            writer.writerow(header)

            # 写入数据行 - 只写入实际运行的算法
            for var_name, var_results in all_results.items():
                exp_type = VARIABLE_DIR_MAP.get(var_name, var_name)
                for var_value in sorted(var_results.keys()):
                    algo_results = var_results[var_value]
                    for algo in actual_algorithms:
                        if algo in algo_results:
                            metrics = algo_results[algo]
                            row = [exp_type, var_name, var_value, algo]
                            for m in METRICS:
                                row.append(metrics.get(m, 0.0))
                                row.append(metrics.get(f'{m}_ci95', 0.0))
                            writer.writerow(row)

        print(f"汇总结果已保存到: {filepath}")
        return filepath

    def print_results(self, variable_name: str, results: Dict[Any, Dict[str, Dict[str, float]]]):
        """打印实验结果（多算法对比）"""
        print(f"\n{'='*90}")
        print(f"变量实验结果: {variable_name}")
        print('='*90)

        # 从结果中提取实际运行的算法列表
        actual_algorithms = set()
        for var_value in results.values():
            actual_algorithms.update(var_value.keys())
        actual_algorithms = sorted(actual_algorithms)

        for var_value in sorted(results.keys()):
            print(f"\n--- {variable_name} = {var_value} ---")
            algo_results = results[var_value]

            # 表头
            print(f"{'算法':<12}", end='')
            for metric in METRICS:
                short_name = metric.replace('_', ' ')[:12]
                print(f"{short_name:<15}", end='')
            print()
            print('-'*90)

            # 数据行 - 只打印实际运行的算法
            for algo in actual_algorithms:
                metrics = algo_results.get(algo, {})
                print(f"{algo:<12}", end='')
                for metric in METRICS:
                    val = metrics.get(metric, 0.0)
                    if metric == 'runtime_seconds':
                        print(f"{val:<15.4f}", end='')
                    elif metric in ['expected_completion_rate', 'avg_agent_survival_rate']:
                        print(f"{val*100:<15.2f}", end='')
                    else:
                        print(f"{val:<15.4f}", end='')
                print()

        print('='*90)
