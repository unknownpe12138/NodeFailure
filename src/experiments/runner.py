"""
实验运行器模块
提供算法执行和实验管理功能
"""
import time
from typing import Dict, List, Optional, Any
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problem import RTMONFProblem
from algorithms.rtm_rpf import RTM_RPF
from experiments.evaluator import Evaluator


class ExperimentRunner:
    """实验运行器"""

    def __init__(self):
        """初始化运行器"""
        self.evaluator = Evaluator()
        self.results_history: List[Dict] = []

    def run_algorithm(self,
                      problem: RTMONFProblem,
                      algorithm_name: str = "RTM-RPF",
                      algorithm_params: Optional[Dict] = None,
                      execute_failure: bool = True,
                      random_seed: Optional[int] = None) -> Dict:
        """
        运行指定算法

        Args:
            problem: RTM-ONF问题实例
            algorithm_name: 算法名称
            algorithm_params: 算法参数
            execute_failure: 是否执行失效判定
            random_seed: 随机种子

        Returns:
            算法结果字典
        """
        if algorithm_params is None:
            algorithm_params = {}

        start_time = time.time()

        if algorithm_name == "RTM-RPF":
            results = self._run_rtm_rpf(
                problem, algorithm_params, execute_failure, random_seed
            )
        elif algorithm_name == "CPLEX":
            results = self._run_cplex(
                problem, algorithm_params, execute_failure, random_seed
            )
        elif algorithm_name == "GREEDY":
            results = self._run_greedy(problem, execute_failure, random_seed)
        elif algorithm_name == "RANDOM":
            results = self._run_random(problem, execute_failure, random_seed)
        else:
            raise ValueError(f"未知算法: {algorithm_name}")

        end_time = time.time()
        results['execution_time'] = end_time - start_time
        results['algorithm'] = algorithm_name

        self.results_history.append(results)
        return results

    def _run_rtm_rpf(self,
                     problem: RTMONFProblem,
                     params: Dict,
                     execute_failure: bool,
                     random_seed: Optional[int]) -> Dict:
        """运行RTM-RPF算法"""
        algorithm = RTM_RPF(
            problem=problem,
            alpha_risk=params.get('alpha_risk', 0.8),
            eta=params.get('eta', 1.5),
            gamma=params.get('gamma', 2.0),
            kappa_task=params.get('kappa_task', 0.5),
            eta_rer=params.get('eta_rer', 0.1),
            kappa_link=params.get('kappa_link', 0.1),
            L_max=params.get('L_max', 10.0),
            alpha1=params.get('alpha1', 0.3),
            alpha2=params.get('alpha2', 0.3),
            alpha3=params.get('alpha3', 0.2),
            alpha4=params.get('alpha4', 0.2),
            random_seed=random_seed
        )

        results = algorithm.solve(execute_failure=execute_failure)
        results['summary'] = algorithm.get_solution_summary()

        return results

    def _run_cplex(self,
                   problem: RTMONFProblem,
                   params: Dict,
                   execute_failure: bool,
                   random_seed: Optional[int]) -> Dict:
        """
        运行CPLEX精确求解算法
        """
        from algorithms.cplex_solver import CPLEX_RTMONF_Solver, check_cplex_available
        from core.failure import FailureModel

        if not check_cplex_available():
            raise ImportError("CPLEX未安装，请先执行: pip install docplex")

        agents = problem.agents
        network = problem.network
        tasks = problem.tasks

        # 初始化失效模型
        failure_model = FailureModel()
        failure_model.update_all_failure_probabilities(agents, network)

        # 执行失效判定
        if execute_failure:
            failure_model.execute_monte_carlo_death(agents, random_seed)
            failure_model.identify_cascade_failures(agents, network)
            failure_model.identify_interrupted_tasks(agents, tasks)

        # 创建CPLEX求解器
        solver = CPLEX_RTMONF_Solver(
            problem=problem,
            failure_model=failure_model,
            time_limit=params.get('time_limit', 300),
            mip_gap=params.get('mip_gap', 0.01)
        )

        # 求解
        results = solver.solve()
        results['failure_statistics'] = failure_model.get_statistics()

        return results

    def _run_greedy(self,
                    problem: RTMONFProblem,
                    execute_failure: bool,
                    random_seed: Optional[int]) -> Dict:
        """
        运行贪心基线算法
        简单地将任务分配给适配度最高的功能有效节点
        """
        from core.failure import FailureModel

        agents = problem.agents
        network = problem.network
        tasks = problem.tasks

        # 初始化失效模型
        failure_model = FailureModel()
        failure_model.update_all_failure_probabilities(agents, network)

        # 执行失效判定
        if execute_failure:
            failure_model.execute_monte_carlo_death(agents, random_seed)
            failure_model.identify_cascade_failures(agents, network)
            failure_model.identify_interrupted_tasks(agents, tasks)

        # 贪心任务分配
        task_assignment = {}
        migration_flows = {}
        role_assignment = {aid: agent.current_role for aid, agent in agents.items()}

        functional_agents = {aid: agent for aid, agent in agents.items() if agent.is_functional}
        agent_loads = {aid: agent.load for aid, agent in functional_agents.items()}

        for task in tasks.sort_by_priority():
            best_agent = None
            best_fitness = -1.0

            for agent_id, agent in functional_agents.items():
                if agent_loads[agent_id] + task.workload > problem.L_crit:
                    continue

                fitness = agent.compute_role_task_fitness(task.requirements)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_agent = agent_id

            if best_agent is not None:
                task_assignment[task.task_id] = best_agent
                agent_loads[best_agent] += task.workload

                source = task.current_agent if task.current_agent else best_agent
                migration_flows[(source, best_agent, task.task_id)] = 1

        # 评估结果
        results = problem.evaluate_solution(
            task_assignment, role_assignment, migration_flows, {}
        )
        results['failure_statistics'] = failure_model.get_statistics()

        return results

    def _run_random(self,
                    problem: RTMONFProblem,
                    execute_failure: bool,
                    random_seed: Optional[int]) -> Dict:
        """
        运行随机基线算法
        随机将任务分配给功能有效节点
        """
        from core.failure import FailureModel

        if random_seed is not None:
            np.random.seed(random_seed)

        agents = problem.agents
        network = problem.network
        tasks = problem.tasks

        # 初始化失效模型
        failure_model = FailureModel()
        failure_model.update_all_failure_probabilities(agents, network)

        # 执行失效判定
        if execute_failure:
            failure_model.execute_monte_carlo_death(agents, random_seed)
            failure_model.identify_cascade_failures(agents, network)
            failure_model.identify_interrupted_tasks(agents, tasks)

        # 随机任务分配
        task_assignment = {}
        migration_flows = {}
        role_assignment = {aid: agent.current_role for aid, agent in agents.items()}

        functional_agents = [aid for aid, agent in agents.items() if agent.is_functional]
        agent_loads = {aid: agents[aid].load for aid in functional_agents}

        for task in tasks.get_all_tasks():
            # 过滤可用节点
            available = [
                aid for aid in functional_agents
                if agent_loads[aid] + task.workload <= problem.L_crit
            ]

            if available:
                assigned = np.random.choice(available)
                task_assignment[task.task_id] = assigned
                agent_loads[assigned] += task.workload

                source = task.current_agent if task.current_agent else assigned
                migration_flows[(source, assigned, task.task_id)] = 1

        # 评估结果
        results = problem.evaluate_solution(
            task_assignment, role_assignment, migration_flows, {}
        )
        results['failure_statistics'] = failure_model.get_statistics()

        return results

    def run_comparison(self,
                       problem: RTMONFProblem,
                       algorithms: List[str] = None,
                       num_runs: int = 1,
                       execute_failure: bool = True) -> Dict[str, Dict]:
        """
        运行多算法对比实验

        Args:
            problem: 问题实例
            algorithms: 算法列表
            num_runs: 每个算法运行次数
            execute_failure: 是否执行失效判定

        Returns:
            {algorithm_name: averaged_results}
        """
        if algorithms is None:
            algorithms = ["RTM-RPF", "GREEDY", "RANDOM"]

        all_results = {algo: [] for algo in algorithms}

        for run_idx in range(num_runs):
            seed = 42 + run_idx

            for algo in algorithms:
                # 重置问题状态
                problem.reset()

                results = self.run_algorithm(
                    problem=problem,
                    algorithm_name=algo,
                    execute_failure=execute_failure,
                    random_seed=seed
                )
                all_results[algo].append(results)

        # 计算平均结果
        averaged_results = {}
        for algo, results_list in all_results.items():
            averaged_results[algo] = self._average_results(results_list)

        return averaged_results

    def _average_results(self, results_list: List[Dict]) -> Dict:
        """计算多次运行的平均结果"""
        if not results_list:
            return {}

        averaged = {'algorithm': results_list[0].get('algorithm', 'Unknown')}

        # 数值指标取平均
        numeric_keys = [
            'total_cost', 'execution_cost', 'migration_cost', 'replenishment_cost',
            'completion_ratio', 'utility', 'execution_time',
            'num_assigned_tasks', 'num_interrupted_tasks', 'num_replenished_nodes'
        ]

        for key in numeric_keys:
            values = [r.get(key, 0.0) for r in results_list if key in r]
            if values:
                averaged[key] = np.mean(values)
                averaged[f'{key}_std'] = np.std(values)

        # 可行性取众数
        feasible_counts = sum(1 for r in results_list if r.get('feasible', False))
        averaged['feasible'] = feasible_counts > len(results_list) / 2
        averaged['feasible_rate'] = feasible_counts / len(results_list)

        return averaged

    def print_results(self, results: Dict):
        """打印结果"""
        report = self.evaluator.evaluate(results, results.get('algorithm', 'Unknown'))
        self.evaluator.print_report(report)

    def print_comparison(self, comparison_results: Dict[str, Dict]):
        """打印对比结果"""
        comparison = self.evaluator.compare_algorithms(comparison_results)
        self.evaluator.print_comparison(comparison)

    def get_history(self) -> List[Dict]:
        """获取历史结果"""
        return self.results_history

    def clear_history(self):
        """清除历史结果"""
        self.results_history.clear()
