"""
评估器模块
提供算法结果的评估和报告生成功能
"""
from typing import Dict, List, Any
import numpy as np


class Evaluator:
    """评估器类"""

    def __init__(self):
        """初始化评估器"""
        self.metrics = [
            'feasible',
            'total_cost',
            'execution_cost',
            'migration_cost',
            'replenishment_cost',
            'completion_ratio',
            'utility',
            'num_assigned_tasks',
            'num_interrupted_tasks',
            'num_replenished_nodes'
        ]

    def evaluate(self, results: Dict, algorithm_name: str) -> Dict:
        """
        评估单个算法结果

        Args:
            results: 算法返回的结果字典
            algorithm_name: 算法名称

        Returns:
            评估报告字典
        """
        report = {
            'algorithm': algorithm_name,
            'feasible': results.get('feasible', False),
            'violations': results.get('violations', []),
        }

        # 提取主要指标
        for metric in self.metrics:
            if metric in results:
                report[metric] = results[metric]

        # 计算额外指标
        if 'failure_statistics' in results:
            fs = results['failure_statistics']
            report['num_failed_nodes'] = fs.get('num_failed', 0)
            report['num_isolated_nodes'] = fs.get('num_isolated', 0)

        if 'replenishment_statistics' in results:
            rs = results['replenishment_statistics']
            report['replenishment_success_rate'] = self._compute_replenishment_rate(rs)

        if 'risk_field_statistics' in results:
            rfs = results['risk_field_statistics']
            report['avg_risk_potential'] = rfs.get('avg_potential', 0.0)

        return report

    def _compute_replenishment_rate(self, rep_stats: Dict) -> float:
        """计算补位成功率"""
        replenished = rep_stats.get('num_replenished', 0)
        unrecoverable = rep_stats.get('num_unrecoverable', 0)
        total = replenished + unrecoverable
        if total == 0:
            return 1.0
        return replenished / total

    def compare_algorithms(self, results_dict: Dict[str, Dict]) -> Dict:
        """
        对比多个算法的结果

        Args:
            results_dict: {algorithm_name: results}

        Returns:
            对比报告
        """
        comparison = {
            'algorithms': list(results_dict.keys()),
            'metrics': {}
        }

        for metric in self.metrics:
            comparison['metrics'][metric] = {}
            for algo_name, results in results_dict.items():
                if metric in results:
                    comparison['metrics'][metric][algo_name] = results[metric]

        # 找出最优算法
        comparison['best'] = self._find_best_algorithm(results_dict)

        return comparison

    def _find_best_algorithm(self, results_dict: Dict[str, Dict]) -> Dict:
        """找出各指标最优的算法"""
        best = {}

        # 效用值最大
        utility_scores = {
            name: results.get('utility', -float('inf'))
            for name, results in results_dict.items()
            if results.get('feasible', False)
        }
        if utility_scores:
            best['utility'] = max(utility_scores, key=utility_scores.get)

        # 完成率最高
        completion_scores = {
            name: results.get('completion_ratio', 0.0)
            for name, results in results_dict.items()
        }
        if completion_scores:
            best['completion_ratio'] = max(completion_scores, key=completion_scores.get)

        # 总代价最低
        cost_scores = {
            name: results.get('total_cost', float('inf'))
            for name, results in results_dict.items()
            if results.get('feasible', False)
        }
        if cost_scores:
            best['total_cost'] = min(cost_scores, key=cost_scores.get)

        return best

    def print_report(self, report: Dict):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print(f"算法评估报告: {report.get('algorithm', 'Unknown')}")
        print("=" * 60)

        print(f"\n可行性: {'是' if report.get('feasible', False) else '否'}")

        if report.get('violations'):
            print(f"约束违反: {len(report['violations'])}项")
            for v in report['violations'][:5]:  # 只显示前5项
                print(f"  - {v}")

        print(f"\n--- 代价指标 ---")
        print(f"总代价: {report.get('total_cost', 0.0):.4f}")
        print(f"  履行代价: {report.get('execution_cost', 0.0):.4f}")
        print(f"  迁移代价: {report.get('migration_cost', 0.0):.4f}")
        print(f"  补位代价: {report.get('replenishment_cost', 0.0):.4f}")

        print(f"\n--- 性能指标 ---")
        print(f"任务完成率: {report.get('completion_ratio', 0.0) * 100:.2f}%")
        print(f"效用值: {report.get('utility', 0.0):.4f}")

        print(f"\n--- 统计信息 ---")
        print(f"已分配任务: {report.get('num_assigned_tasks', 0)}")
        print(f"中断任务: {report.get('num_interrupted_tasks', 0)}")
        print(f"失效节点: {report.get('num_failed_nodes', 0)}")
        print(f"孤岛节点: {report.get('num_isolated_nodes', 0)}")
        print(f"补位节点: {report.get('num_replenished_nodes', 0)}")

        if 'replenishment_success_rate' in report:
            print(f"补位成功率: {report['replenishment_success_rate'] * 100:.2f}%")

        print("=" * 60)

    def print_comparison(self, comparison: Dict):
        """打印对比报告"""
        print("\n" + "=" * 70)
        print("算法对比报告")
        print("=" * 70)

        algorithms = comparison['algorithms']
        print(f"\n对比算法: {', '.join(algorithms)}")

        print(f"\n{'指标':<20}", end='')
        for algo in algorithms:
            print(f"{algo:<15}", end='')
        print()
        print("-" * 70)

        for metric, values in comparison['metrics'].items():
            print(f"{metric:<20}", end='')
            for algo in algorithms:
                val = values.get(algo, 'N/A')
                if isinstance(val, float):
                    print(f"{val:<15.4f}", end='')
                else:
                    print(f"{str(val):<15}", end='')
            print()

        print(f"\n--- 最优算法 ---")
        for metric, algo in comparison.get('best', {}).items():
            print(f"{metric}: {algo}")

        print("=" * 70)

    def generate_latex_table(self, results_list: List[Dict]) -> str:
        """
        生成LaTeX表格

        Args:
            results_list: 结果列表

        Returns:
            LaTeX表格字符串
        """
        if not results_list:
            return ""

        # 表头
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{算法性能对比}\n"
        latex += "\\begin{tabular}{l" + "c" * len(results_list) + "}\n"
        latex += "\\hline\n"

        # 算法名称行
        latex += "指标"
        for r in results_list:
            latex += f" & {r.get('algorithm', 'Unknown')}"
        latex += " \\\\\n\\hline\n"

        # 数据行
        metrics_to_show = [
            ('total_cost', '总代价'),
            ('completion_ratio', '完成率'),
            ('utility', '效用值'),
            ('num_assigned_tasks', '分配任务数'),
            ('num_replenished_nodes', '补位节点数')
        ]

        for metric_key, metric_name in metrics_to_show:
            latex += metric_name
            for r in results_list:
                val = r.get(metric_key, 'N/A')
                if isinstance(val, float):
                    if metric_key == 'completion_ratio':
                        latex += f" & {val * 100:.2f}\\%"
                    else:
                        latex += f" & {val:.4f}"
                else:
                    latex += f" & {val}"
            latex += " \\\\\n"

        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        return latex
