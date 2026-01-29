"""
运行所有对比实验脚本
依次运行4个变量实验，并生成汇总结果
对比算法: CPLEX, RTM-RPF
"""

import sys
import os
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.comparative.config import DEFAULT_PARAMS, VARIABLE_RANGES, NUM_RUNS, ALGORITHMS
from experiments.comparative.batch_runner import BatchExperimentRunner


def run_all_experiments(num_runs: int = NUM_RUNS, verbose: bool = True, algorithms: list = None,
                       use_batch_mode: bool = False, batch_ratio: float = 0.2, batch_strategy: str = 'random'):
    """
    运行所有变量实验（多算法对比）

    Args:
        num_runs: 每个实验重复次数
        verbose: 是否打印详细信息
        algorithms: 指定算法列表
        use_batch_mode: 是否使用分批分配模式
        batch_ratio: 分批比例
        batch_strategy: 分批策略
    """
    algo_list = algorithms if algorithms else ALGORITHMS
    mode_str = f"分批模式 (比例={batch_ratio}, 策略={batch_strategy})" if use_batch_mode else "原有模式"
    print("="*70)
    print(f"对比实验 - 运行所有变量实验（多算法对比）- {mode_str}")
    print("="*70)
    print(f"实验组数: {len(VARIABLE_RANGES)}")
    print(f"对比算法: {algo_list}")
    print(f"每组重复次数: {num_runs}")
    print(f"默认参数: {DEFAULT_PARAMS}")
    if use_batch_mode:
        print(f"分批配置: 比例={batch_ratio}, 策略={batch_strategy}")
    print("="*70)

    start_time = time.time()

    runner = BatchExperimentRunner(
        num_runs=num_runs,
        algorithms=algo_list,
        verbose=verbose,
        use_batch_mode=use_batch_mode,
        batch_ratio=batch_ratio,
        batch_strategy=batch_strategy
    )

    # 运行所有实验并导出CSV
    all_results = runner.run_all_experiments(export_csv=True)

    end_time = time.time()
    total_time = end_time - start_time

    # 打印汇总信息
    print("\n" + "="*70)
    print("所有实验完成!")
    print("="*70)
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"实验组数: {len(all_results)}")
    print(f"对比算法: {ALGORITHMS}")

    total_var_values = sum(len(v) for v in all_results.values())
    total_experiments = total_var_values * len(ALGORITHMS) * num_runs
    print(f"总实验数: {total_var_values} 个变量值 × {len(ALGORITHMS)} 个算法 × {num_runs} 次重复 = {total_experiments} 次")

    print("\n生成的结果文件:")
    results_dir = runner.metrics_collector.results_dir
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(results_dir, filename)
            print(f"  - {filepath}")

    print("="*70)

    return all_results


def run_single_variable(variable_name: str, num_runs: int = NUM_RUNS, verbose: bool = True, algorithms: list = None,
                       use_batch_mode: bool = False, batch_ratio: float = 0.2, batch_strategy: str = 'random'):
    """
    运行单个变量实验（多算法对比）

    Args:
        variable_name: 变量名称 (num_tasks, num_agents, failure_rate, num_layers)
        num_runs: 每个实验重复次数
        verbose: 是否打印详细信息
        algorithms: 指定算法列表
        use_batch_mode: 是否使用分批分配模式
        batch_ratio: 分批比例
        batch_strategy: 分批策略
    """
    if variable_name not in VARIABLE_RANGES:
        print(f"错误: 未知变量 '{variable_name}'")
        print(f"可用变量: {list(VARIABLE_RANGES.keys())}")
        return None

    algo_list = algorithms if algorithms else ALGORITHMS
    mode_str = f"分批模式 (比例={batch_ratio})" if use_batch_mode else "原有模式"
    print("="*60)
    print(f"运行单变量实验: {variable_name}（多算法对比）- {mode_str}")
    print(f"对比算法: {algo_list}")
    print("="*60)

    runner = BatchExperimentRunner(
        num_runs=num_runs,
        algorithms=algo_list,
        verbose=verbose,
        use_batch_mode=use_batch_mode,
        batch_ratio=batch_ratio,
        batch_strategy=batch_strategy
    )

    results = runner.run_variable_experiment(
        variable_name=variable_name,
        variable_values=VARIABLE_RANGES[variable_name],
        base_params=DEFAULT_PARAMS
    )

    runner.metrics_collector.print_results(variable_name, results)
    runner.metrics_collector.export_variable_results(variable_name, results)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='对比实验运行器（多算法对比）')
    parser.add_argument('--num_runs', type=int, default=NUM_RUNS,
                        help=f'每个实验重复次数 (默认: {NUM_RUNS})')
    parser.add_argument('--variable', type=str, default=None,
                        choices=['num_tasks', 'num_agents', 'failure_rate', 'num_layers', 'weight_config'],
                        help='只运行指定变量的实验 (默认: 运行所有)')
    parser.add_argument('--quiet', action='store_true', default=True,
                        help='安静模式，减少输出（默认开启）')
    parser.add_argument('--verbose', action='store_true',
                        help='详细模式，打印详细信息')
    parser.add_argument('--algorithms', type=str, default=None,
                        help='指定算法列表，用逗号分隔 (例如: SPTM,LBTM)')
    parser.add_argument('--batch_mode', action='store_true',
                        help='使用分批分配模式')
    parser.add_argument('--batch_ratio', type=float, default=0.2,
                        help='分批比例 (默认: 0.2，即每批20%%的任务)')
    parser.add_argument('--batch_strategy', type=str, default='random',
                        choices=['random', 'priority', 'urgency'],
                        help='分批策略 (默认: random)')

    args = parser.parse_args()

    # 解析算法列表
    algorithms = None
    if args.algorithms:
        algorithms = [a.strip() for a in args.algorithms.split(',')]

    if args.variable:
        run_single_variable(
            variable_name=args.variable,
            num_runs=args.num_runs,
            verbose=args.verbose,
            algorithms=algorithms,
            use_batch_mode=args.batch_mode,
            batch_ratio=args.batch_ratio,
            batch_strategy=args.batch_strategy
        )
    else:
        run_all_experiments(
            num_runs=args.num_runs,
            verbose=args.verbose,
            algorithms=algorithms,
            use_batch_mode=args.batch_mode,
            batch_ratio=args.batch_ratio,
            batch_strategy=args.batch_strategy
        )
