"""
任务数量变量实验脚本
固定其他参数，变化任务数量 [10, 20, 30, 40, 50]
对比算法: CPLEX, RTM-RPF
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.comparative.config import DEFAULT_PARAMS, VARIABLE_RANGES, NUM_RUNS, ALGORITHMS
from experiments.comparative.batch_runner import BatchExperimentRunner


def run_task_count_experiment(num_runs: int = NUM_RUNS, verbose: bool = True):
    """
    运行任务数量变量实验（多算法对比）

    Args:
        num_runs: 每个实验重复次数
        verbose: 是否打印详细信息
    """
    print("="*60)
    print("任务数量变量实验（多算法对比）")
    print("="*60)
    print(f"变量范围: {VARIABLE_RANGES['num_tasks']}")
    print(f"对比算法: {ALGORITHMS}")
    print(f"重复次数: {num_runs}")
    print(f"默认参数: {DEFAULT_PARAMS}")
    print("="*60)

    runner = BatchExperimentRunner(num_runs=num_runs, verbose=verbose)

    results = runner.run_variable_experiment(
        variable_name='num_tasks',
        variable_values=VARIABLE_RANGES['num_tasks'],
        base_params=DEFAULT_PARAMS
    )

    # 打印结果
    runner.metrics_collector.print_results('num_tasks', results)

    # 导出CSV
    filepath = runner.metrics_collector.export_variable_results('num_tasks', results)

    print(f"\n实验完成! 结果已保存到: {filepath}")
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='任务数量变量实验（多算法对比）')
    parser.add_argument('--num_runs', type=int, default=NUM_RUNS,
                        help=f'每个实验重复次数 (默认: {NUM_RUNS})')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式，减少输出')

    args = parser.parse_args()

    run_task_count_experiment(
        num_runs=args.num_runs,
        verbose=not args.quiet
    )
