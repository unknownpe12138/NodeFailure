"""
权重配置实验脚本
测试不同 λ1 和 λ2 配置对算法性能的影响

实验配置：
- 侧重于完成率: λ1=0.25, λ2=0.75
- 均衡配置: λ1=0.5, λ2=0.5
- 侧重于成本控制: λ1=0.75, λ2=0.25

固定参数：
- 智能体数量: 25
- 任务数量变化范围: [10, 20, 30, 40, 50]
"""

import sys
import os
import time
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.scenario_generator import ScenarioGenerator
from experiments.runner import ExperimentRunner
from experiments.comparative.config import ALGORITHMS, NUM_RUNS
from experiments.comparative.metrics import MetricsCollector


# 权重配置
WEIGHT_CONFIGS = {
    'completion_focused': {'lambda1': 0.25, 'lambda2': 0.75, 'name': '侧重完成率'},
    'balanced': {'lambda1': 0.5, 'lambda2': 0.5, 'name': '均衡配置'},
    'cost_focused': {'lambda1': 0.75, 'lambda2': 0.25, 'name': '侧重成本'},
}

# 固定参数
FIXED_PARAMS = {
    'num_agents': 25,
    'num_layers': 5,
    'failure_rate': 0.1,
    'connection_prob': 0.4,
    'num_capabilities': 10,
    'capability_coverage': 0.35,
    'num_roles_per_agent': 3,
}

# 任务数量范围
TASK_COUNTS = [10, 20, 30, 40, 50]


def run_weight_config_experiment(num_runs: int = NUM_RUNS,
                                  algorithms: list = None,
                                  verbose: bool = True):
    """
    运行权重配置实验

    Args:
        num_runs: 每个配置重复次数
        algorithms: 要测试的算法列表
        verbose: 是否打印详细信息
    """
    if algorithms is None:
        algorithms = ALGORITHMS

    print("="*70)
    print("权重配置实验 - 测试不同 λ1 和 λ2 配置")
    print("="*70)
    print(f"算法: {algorithms}")
    print(f"重复次数: {num_runs}")
    print(f"任务数量范围: {TASK_COUNTS}")
    print(f"固定智能体数量: {FIXED_PARAMS['num_agents']}")
    print("="*70)

    runner = ExperimentRunner()
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'weight_config'
    )
    os.makedirs(results_dir, exist_ok=True)

    metrics_collector = MetricsCollector(results_dir)

    # 存储所有结果
    all_results = {}

    start_time = time.time()

    # 对每个权重配置进行实验
    for config_key, config in WEIGHT_CONFIGS.items():
        print(f"\n{'#'*70}")
        print(f"# 权重配置: {config['name']} (λ1={config['lambda1']}, λ2={config['lambda2']})")
        print(f"{'#'*70}")

        config_results = {}

        # 对每个任务数量进行实验
        for num_tasks in TASK_COUNTS:
            print(f"\n{'='*60}")
            print(f"任务数量: {num_tasks}")
            print(f"{'='*60}")

            # 准备实验参数
            params = FIXED_PARAMS.copy()
            params['num_tasks'] = num_tasks
            params['lambda1'] = config['lambda1']
            params['lambda2'] = config['lambda2']

            # 每个算法的多次运行结果
            algo_run_results = {algo: [] for algo in algorithms}

            # 运行多次实验
            for run_idx in range(num_runs):
                seed = 42 + run_idx

                if verbose and (run_idx + 1) % 10 == 0:
                    print(f"  进度: {run_idx + 1}/{num_runs}")

                try:
                    # 生成场景
                    generator = ScenarioGenerator(seed=seed)
                    problem = generator.generate_scenario(
                        num_agents=params['num_agents'],
                        num_layers=params['num_layers'],
                        num_tasks=params['num_tasks'],
                        num_roles_per_agent=params['num_roles_per_agent'],
                        connection_prob=params['connection_prob'],
                        failure_rate=params['failure_rate'],
                        num_capabilities=params['num_capabilities'],
                        capability_coverage=params['capability_coverage'],
                        lambda1=params['lambda1'],
                        lambda2=params['lambda2'],
                    )

                    # 对每个算法运行
                    for algorithm in algorithms:
                        # 重置问题状态
                      problem.reset()

                        # 运行算法
                        algo_results = runner.run_algorithm(
                            problem=problem,
                            algorithm_name=algorithm,
                            execute_failure=True,
                            random_seed=seed
                        )

                        # 收集指标
                        metrics = metrics_collector.collect_metrics(algo_results, problem)
                        algo_run_results[algorithm].append(metrics)

                except Exception as e:
                    print(f"  警告: 第{run_idx + 1}次实验失败 - {e}")
                    continue

            # 聚合每个算法的结果
            algo_aggregated = {}
            for algorithm in algorithms:
                run_results = algo_run_results[algorithm]
                if run_results:
                    aggregated = metrics_collector.aggregate_results(run_results)
                    algo_aggregated[algorithm] = aggregated

                    if verbose:
                        print(f"  {algorithm}: 有效实验 {len(run_results)}/{num_runs}")

            config_results[num_tasks] = algo_aggregated

        all_results[config_key] = config_results

        # 导出当前配置的结果
        export_config_results(config_key, config, config_results, results_dir)

    end_time = time.time()
    total_time = end_time - start_time

    # 导出汇总结果
    export_summary(all_results, results_dir)

    print("\n" + "="*70)
    print("权重配置实验完成!")
    print("="*70)
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"结果保存在: {results_dir}")
    print("="*70)

    return all_results


def export_config_results(config_key: str, config: dict, results: dict, results_dir: str):
    """导出单个配置的结果"""

    for algorithm in ALGORITHMS:
        rows = []

        for num_tasks, algo_results in results.items():
            if algorithm in algo_results:
                metrics = algo_results[algorithm]
                row = {
                    'num_tasks': num_tasks,
                    'lambda1': config['lambda1'],
                    'lambda2': config['lambda2'],
                    'config_name': config['name'],
                    'objective_value': metrics.get('objective_value', 0),
                    'objective_value_ci95': metrics.get('objective_value_ci95', 0),
                    'task_cost': metrics.get('task_cost', 0),
                    'task_cost_ci95': metrics.get('task_cost_ci95', 0),
                    'expected_completion_rate': metrics.get('expected_completion_rate', 0),
                    'expected_completion_rate_ci95': metrics.get('expected_completion_rate_ci95', 0),
                    'runtime_seconds': metrics.get('runtime_seconds', 0),
                    'runtime_seconds_ci95': metrics.get('runtime_seconds_ci95', 0),
                }
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            filename = f"{algorithm}_{config_key}_results.csv"
            filepath = os.path.join(results_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"  导出: {filename}")


def export_summary(all_results: dict, results_dir: str):
    """导出汇总结果"""

    rows = []

    for config_key, config_results in all_results.items():
        config = WEIGHT_CONFIGS[config_key]

        for num_tasks, algo_results in config_results.items():
            for algorithm, metrics in algo_results.items():
                row = {
                    'config_key': config_key,
                    'config_name': config['name'],
                    'lambda1': config['lambda1'],
                    'lambda2': config['lambda2'],
                    'num_tasks': num_tasks,
                    'algorithm': algorithm,
                    'objective_value': metrics.get('objective_value', 0),
                    'task_cost': metrics.get('task_cost', 0),
                    'expected_completion_rate': metrics.get('expected_completion_rate', 0),
                    'runtime_seconds': metrics.get('runtime_seconds', 0),
                }
                rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        filepath = os.path.join(results_dir, 'weight_config_summary.csv')
        df.to_csv(filepath, index=False)
        print(f"\n汇总结果已保存: weight_config_summary.csv")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='权重配置实验')
    parser.add_argument('--num_runs', type=int, default=100,
                        help='每个配置重复次数 (默认: 100)')
    parser.add_argument('--algorithms', type=str, default=None,
                        help='指定算法列表，用逗号分隔 (例如: CPLEX,RTM-RPF)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='详细模式')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式')

    args = parser.parse_args()

    # 解析算法列表
    algorithms = None
    if args.algorithms:
        algorithms = [a.strip() for a in args.algorithms.split(',')]

    verbose = args.verbose and not args.quiet

    run_weight_config_experiment(
        num_runs=args.num_runs,
        algorithms=algorithms,
        verbose=verbose
    )
