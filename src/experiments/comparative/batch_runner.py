"""
批量实验运行器
提供批量运行实验、聚合结果的功能
支持多算法对比
"""

import sys
import os
from typing import Dict, List, Any, Optional
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.scenario_generator import ScenarioGenerator
from experiments.runner import ExperimentRunner
from experiments.comparative.config import DEFAULT_PARAMS, VARIABLE_RANGES, NUM_RUNS, ALGORITHMS
from experiments.comparative.metrics import MetricsCollector


class BatchExperimentRunner:
    """批量实验运行器"""

    def __init__(self,
                 num_runs: int = NUM_RUNS,
                 algorithms: List[str] = None,
                 results_dir: str = None,
                 verbose: bool = True,
                 use_batch_mode: bool = False,
                 batch_ratio: float = 0.2,
                 batch_strategy: str = 'random'):
        """
        Args:
            num_runs: 每个实验重复次数
            algorithms: 要对比的算法列表
            results_dir: 结果保存目录
            verbose: 是否打印详细信息
            use_batch_mode: 是否使用分批分配模式
            batch_ratio: 分批比例（仅在use_batch_mode=True时有效）
            batch_strategy: 分批策略（仅在use_batch_mode=True时有效）
        """
        self.num_runs = num_runs
        self.algorithms = algorithms if algorithms else ALGORITHMS
        self.verbose = verbose
        self.use_batch_mode = use_batch_mode
        self.batch_ratio = batch_ratio
        self.batch_strategy = batch_strategy
        self.metrics_collector = MetricsCollector(results_dir)
        self.runner = ExperimentRunner()

    def run_single_experiment(self,
                               params: Dict,
                               algorithm: str,
                               seed: int = 42) -> Dict[str, float]:
        """
        运行单次实验（单个算法）

        Args:
            params: 实验参数
            algorithm: 算法名称
            seed: 随机种子

        Returns:
            5个评估指标
        """
        # 创建场景生成器
        generator = ScenarioGenerator(seed=seed)

        # 生成问题实例
        problem = generator.generate_scenario(
            num_agents=params.get('num_agents', DEFAULT_PARAMS['num_agents']),
            num_layers=params.get('num_layers', DEFAULT_PARAMS['num_layers']),
            num_tasks=params.get('num_tasks', DEFAULT_PARAMS['num_tasks']),
            num_roles_per_agent=params.get('num_roles_per_agent', DEFAULT_PARAMS['num_roles_per_agent']),
            connection_prob=params.get('connection_prob', DEFAULT_PARAMS['connection_prob']),
            failure_rate=params.get('failure_rate', DEFAULT_PARAMS['failure_rate']),
            num_capabilities=params.get('num_capabilities', DEFAULT_PARAMS['num_capabilities']),
            capability_coverage=params.get('capability_coverage', DEFAULT_PARAMS['capability_coverage']),
            lambda1=params.get('lambda1', DEFAULT_PARAMS['lambda1']),
            lambda2=params.get('lambda2', DEFAULT_PARAMS['lambda2']),
        )

        # 根据模式运行算法
        if self.use_batch_mode:
            # 分批分配模式
            results = self.runner.run_batch_allocation(
                problem=problem,
                algorithm_name=algorithm,
                batch_ratio=self.batch_ratio,
                batch_strategy=self.batch_strategy,
                random_seed=seed
            )
        else:
            # 原有模式（一次性分配）
            results = self.runner.run_algorithm(
                problem=problem,
                algorithm_name=algorithm,
                execute_failure=True,
                random_seed=seed
            )

        # 收集指标
        metrics = self.metrics_collector.collect_metrics(results, problem)

        return metrics

    def run_algorithms_comparison(self,
                                   params: Dict,
                                   seed: int = 42) -> Dict[str, Dict[str, float]]:
        """
        运行所有算法的单次对比实验

        Args:
            params: 实验参数
            seed: 随机种子

        Returns:
            {算法名: 指标字典}
        """
        results = {}

        for algorithm in self.algorithms:
            metrics = self.run_single_experiment(params, algorithm, seed)
            results[algorithm] = metrics

        return results

    def run_algorithms_on_same_scenario(self, params: Dict, seed: int) -> Dict[str, Dict[str, float]]:
        """
        在相同的失效场景上运行所有算法（修复：确保公平对比）

        Args:
            params: 实验参数
            seed: 随机种子

        Returns:
            {算法名: 指标字典}
        """
        from experiments.scenario_generator import ScenarioGenerator
        from core.failure import FailureModel
        import copy

        # 1. 创建共享的问题实例
        generator = ScenarioGenerator(seed=seed)
        problem = generator.generate_scenario(
            num_agents=params.get('num_agents', DEFAULT_PARAMS['num_agents']),
            num_layers=params.get('num_layers', DEFAULT_PARAMS['num_layers']),
            num_tasks=params.get('num_tasks', DEFAULT_PARAMS['num_tasks']),
            num_roles_per_agent=params.get('num_roles_per_agent', DEFAULT_PARAMS['num_roles_per_agent']),
            connection_prob=params.get('connection_prob', DEFAULT_PARAMS['connection_prob']),
            failure_rate=params.get('failure_rate', DEFAULT_PARAMS['failure_rate']),
            num_capabilities=params.get('num_capabilities', DEFAULT_PARAMS['num_capabilities']),
            capability_coverage=params.get('capability_coverage', DEFAULT_PARAMS['capability_coverage']),
            lambda1=params.get('lambda1', DEFAULT_PARAMS['lambda1']),
            lambda2=params.get('lambda2', DEFAULT_PARAMS['lambda2']),
        )

        # 2. 执行一次失效判定（所有算法共享）
        failure_model = FailureModel()
        failure_model.update_all_failure_probabilities(problem.agents, problem.network)
        failure_model.execute_monte_carlo_death(problem.agents, seed)
        failure_model.identify_cascade_failures(problem.agents, problem.network)
        failure_model.identify_interrupted_tasks(problem.agents, problem.tasks)

        # 保存失效状态和初始状态
        initial_state = {
            'physical_state': {aid: agent.physical_state for aid, agent in problem.agents.items()},
            'functional_state': {aid: agent.functional_state for aid, agent in problem.agents.items()},
            'load': {aid: agent.load for aid, agent in problem.agents.items()},
            'current_role': {aid: agent.current_role for aid, agent in problem.agents.items()},
        }

        results = {}

        # 3. 对每个算法运行（使用相同的失效场景）
        for algorithm in self.algorithms:
            # 恢复初始状态
            for aid, agent in problem.agents.items():
                agent.physical_state = initial_state['physical_state'][aid]
                agent.functional_state = initial_state['functional_state'][aid]
                agent.load = initial_state['load'][aid]
                agent.current_role = initial_state['current_role'][aid]

            # 运行算法（不再执行失效判定）
            algo_results = self.runner.run_algorithm(
                problem=problem,
                algorithm_name=algorithm,
                execute_failure=False,  # 关键：不再执行失效判定
                random_seed=seed
            )

            # 添加失效统计
            algo_results['failure_statistics'] = failure_model.get_statistics()

            # 收集指标
            metrics = self.metrics_collector.collect_metrics(algo_results, problem)
            results[algorithm] = metrics

        return results

    def run_variable_experiment(self,
                                 variable_name: str,
                                 variable_values: List = None,
                                 base_params: Dict = None) -> Dict[Any, Dict[str, Dict[str, float]]]:
        """
        运行单变量实验（所有算法对比）

        Args:
            variable_name: 变量名称
            variable_values: 变量取值列表
            base_params: 基础参数（其他参数固定为此值）

        Returns:
            {变量值: {算法名: 平均指标字典}}
        """
        if variable_values is None:
            variable_values = VARIABLE_RANGES.get(variable_name, [])

        if base_params is None:
            base_params = DEFAULT_PARAMS.copy()

        # 特殊处理权重配置实验
        if variable_name == 'weight_config':
            return self.run_weight_config_experiment(variable_values, base_params)

        results = {}

        for var_value in variable_values:
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"运行实验: {variable_name} = {var_value}")
                print(f"算法: {self.algorithms}")
                print(f"重复次数: {self.num_runs}")
                print('='*50)

            # 设置当前变量值
            params = base_params.copy()
            params[variable_name] = var_value

            # 每个算法的多次运行结果
            algo_run_results = {algo: [] for algo in self.algorithms}

            # 运行多次实验
            for run_idx in range(self.num_runs):
                seed = 42 + run_idx

                if self.verbose and (run_idx + 1) % 10 == 0:
                    print(f"  进度: {run_idx + 1}/{self.num_runs}")

                try:
                    # 使用新方法：所有算法在相同场景上运行
                    run_results = self.run_algorithms_on_same_scenario(params, seed)

                    # 分配到各算法的结果列表
                    for algorithm in self.algorithms:
                        if algorithm in run_results:
                            algo_run_results[algorithm].append(run_results[algorithm])
                except Exception as e:
                    # 始终打印错误信息，便于调试
                    print(f"  警告: 第{run_idx + 1}次实验失败 - {e}")
                    continue

            # 聚合每个算法的结果
            algo_aggregated = {}
            for algorithm in self.algorithms:
                run_results = algo_run_results[algorithm]
                if run_results:
                    aggregated = self.metrics_collector.aggregate_results(run_results)
                    algo_aggregated[algorithm] = aggregated

                    if self.verbose:
                        print(f"  {algorithm}: 有效实验 {len(run_results)}/{self.num_runs}")

            results[var_value] = algo_aggregated

        return results

    def run_weight_config_experiment(self,
                                     config_keys: List[str],
                                     base_params: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        运行权重配置实验（特殊处理）

        Args:
            config_keys: 权重配置键列表 ['completion_focused', 'balanced', 'cost_focused']
            base_params: 基础参数

        Returns:
            {配置键: {算法名: 平均指标字典}}
        """
        from experiments.comparative.config import WEIGHT_CONFIGS, WEIGHT_CONFIG_FIXED_PARAMS

        results = {}

        # 任务数量范围
        task_counts = WEIGHT_CONFIG_FIXED_PARAMS['num_tasks']

        for config_key in config_keys:
            if config_key not in WEIGHT_CONFIGS:
                print(f"警告: 未知的权重配置 {config_key}")
                continue

            config = WEIGHT_CONFIGS[config_key]

            if self.verbose:
                print(f"\n{'='*50}")
                print(f"权重配置: {config_key}")
                print(f"  λ1={config['lambda1']}, λ2={config['lambda2']}")
                print(f"  任务数量范围: {task_counts}")
                print(f"  固定智能体数量: {WEIGHT_CONFIG_FIXED_PARAMS['num_agents']}")
                print('='*50)

            # 对每个任务数量运行实验
            config_results = {}

            for num_tasks in task_counts:
                if self.verbose:
                    print(f"\n  任务数量: {num_tasks}")

                # 设置参数
                params = base_params.copy()
                params['num_agents'] = WEIGHT_CONFIG_FIXED_PARAMS['num_agents']
                params['num_tasks'] = num_tasks
                params['lambda1'] = config['lambda1']
                params['lambda2'] = config['lambda2']

                # 每个算法的多次运行结果
                algo_run_results = {algo: [] for algo in self.algorithms}

                # 运行多次实验
                for run_idx in range(self.num_runs):
                    seed = 42 + run_idx

                    if self.verbose and (run_idx + 1) % 10 == 0:
                        print(f"    进度: {run_idx + 1}/{self.num_runs}")

                    try:
                        run_results = self.run_algorithms_on_same_scenario(params, seed)

                        for algorithm in self.algorithms:
                            if algorithm in run_results:
                                algo_run_results[algorithm].append(run_results[algorithm])
                    except Exception as e:
                        print(f"    警告: 第{run_idx + 1}次实验失败 - {e}")
                        continue

                # 聚合每个算法的结果
                for algorithm in self.algorithms:
                    run_results = algo_run_results[algorithm]
                    if run_results:
                        aggregated = self.metrics_collector.aggregate_results(run_results)

                        # 添加配置信息
                        aggregated['lambda1'] = config['lambda1']
                        aggregated['lambda2'] = config['lambda2']
                        aggregated['num_tasks'] = num_tasks
                        aggregated['config_key'] = config_key

                        # 存储结果
                        key = f"{config_key}_{num_tasks}"
                        if key not in config_results:
                            config_results[key] = {}
                        config_results[key][algorithm] = aggregated

                        if self.verbose:
                            print(f"    {algorithm}: 有效实验 {len(run_results)}/{self.num_runs}")

            results[config_key] = config_results

        return results

    def run_all_experiments(self,
                            export_csv: bool = True) -> Dict[str, Dict[Any, Dict[str, Dict[str, float]]]]:
        """
        运行所有变量实验

        Args:
            export_csv: 是否导出CSV文件

        Returns:
            {变量名: {变量值: {算法名: 平均指标字典}}}
        """
        all_results = {}

        for variable_name, variable_values in VARIABLE_RANGES.items():
            if self.verbose:
                print(f"\n{'#'*60}")
                print(f"# 开始实验组: {variable_name}")
                print(f"# 取值范围: {variable_values}")
                print(f"# 对比算法: {self.algorithms}")
                print('#'*60)

            results = self.run_variable_experiment(variable_name, variable_values)
            all_results[variable_name] = results

            # 打印结果
            if self.verbose:
                self.metrics_collector.print_results(variable_name, results)

            # 导出单变量结果
            if export_csv:
                self.metrics_collector.export_variable_results(variable_name, results)

        # 导出汇总结果
        if export_csv:
            self.metrics_collector.export_summary(all_results)

        return all_results


def main():
    """主函数 - 用于测试"""
    print("批量实验运行器测试（多算法对比）")
    print("="*50)

    # 使用较少的重复次数进行测试
    runner = BatchExperimentRunner(num_runs=3, verbose=True)

    # 测试单变量实验
    results = runner.run_variable_experiment(
        variable_name='num_tasks',
        variable_values=[10, 20]
    )

    runner.metrics_collector.print_results('num_tasks', results)


if __name__ == '__main__':
    main()
