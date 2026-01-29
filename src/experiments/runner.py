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
from algorithms.sptm import SPTM
from algorithms.lbtm import LBTM
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
        elif algorithm_name == "SPTM":
            results = self._run_sptm(problem, execute_failure, random_seed)
        elif algorithm_name == "LBTM":
            results = self._run_lbtm(problem, execute_failure, random_seed)
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

    def _run_sptm(self,
                  problem: RTMONFProblem,
                  execute_failure: bool,
                  random_seed: Optional[int]) -> Dict:
        """运行SPTM算法（基于最短路径的任务迁移）"""
        algorithm = SPTM(problem=problem, random_seed=random_seed)
        results = algorithm.solve(execute_failure=execute_failure)
        results['summary'] = algorithm.get_solution_summary()
        return results

    def _run_lbtm(self,
                  problem: RTMONFProblem,
                  execute_failure: bool,
                  random_seed: Optional[int]) -> Dict:
        """运行LBTM算法（基于负载均衡的任务迁移）"""
        algorithm = LBTM(problem=problem, random_seed=random_seed)
        results = algorithm.solve(execute_failure=execute_failure)
        results['summary'] = algorithm.get_solution_summary()
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
            algorithms = ["RTM-RPF", "SPTM", "LBTM", "GREEDY", "RANDOM"]

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

    def run_batch_allocation(self,
                            problem: RTMONFProblem,
                            algorithm_name: str = "RTM-RPF",
                            batch_ratio: float = 0.2,
                            batch_strategy: str = 'random',
                            algorithm_params: Optional[Dict] = None,
                            random_seed: int = 42) -> Dict:
        """
        分批分配主循环（简化版，只记录最终结果）

        Args:
            problem: RTMONFProblem实例
            algorithm_name: 算法名称
            batch_ratio: 每批任务比例
            batch_strategy: 分批策略 ('random', 'priority', 'urgency')
            algorithm_params: 算法参数
            random_seed: 随机种子

        Returns:
            最终结果（与原有格式一致）
        """
        if algorithm_params is None:
            algorithm_params = {}

        start_time = time.time()

        # 1. 初始化
        problem.reset()
        np.random.seed(random_seed)

        # 2. 任务分批
        num_batches = int(1.0 / batch_ratio)
        task_batches = problem.tasks.split_into_batches(num_batches, batch_strategy)

        print(f"\n=== 分批分配模式 ===")
        print(f"  批次数: {num_batches}, 每批比例: {batch_ratio}")
        print(f"  分批策略: {batch_strategy}")
        print(f"  算法: {algorithm_name}")

        # 3. 初始化失效模型
        from core.failure import FailureModel
        failure_model = FailureModel()
        failure_model.update_all_failure_probabilities(problem.agents, problem.network)

        # 4. 累积分配记录
        cumulative_assignment = {}  # {task_id: agent_id}

        # 5. 分批循环
        for batch_idx, task_batch in enumerate(task_batches):
            print(f"\n--- 批次 {batch_idx + 1}/{num_batches} ---")
            print(f"  新任务数: {len(task_batch)}")

            # 5.1 收集待分配任务（新任务 + 中断任务）
            interrupted_tasks = self._get_interrupted_tasks(cumulative_assignment, problem.agents)
            tasks_to_assign = task_batch + interrupted_tasks

            print(f"  中断任务数: {len(interrupted_tasks)}")
            print(f"  总待分配: {len(tasks_to_assign)}")

            # 5.2 调用算法分配任务
            batch_assignment = self._run_algorithm_for_batch(
                problem, algorithm_name, tasks_to_assign,
                algorithm_params, random_seed + batch_idx
            )

            # 5.3 更新累积分配
            for task_id in interrupted_tasks:
                if task_id in cumulative_assignment:
                    del cumulative_assignment[task_id]
            cumulative_assignment.update(batch_assignment)

            print(f"  本批分配成功: {len(batch_assignment)}")

            # 5.4 更新失效概率
            failure_model.update_all_failure_probabilities(problem.agents, problem.network)

            # 5.5 执行失效判定
            batch_seed = random_seed + batch_idx * 1000
            newly_failed = failure_model.execute_monte_carlo_death(problem.agents, batch_seed)

            # 5.6 识别级联失效
            if newly_failed:
                failure_model.identify_cascade_failures(problem.agents, problem.network)
                print(f"  新失效节点: {len(newly_failed)}")

        # 6. 计算最终结果
        end_time = time.time()
        final_result = self._compute_final_metrics(problem, cumulative_assignment, failure_model)
        final_result['execution_time'] = end_time - start_time
        final_result['algorithm'] = algorithm_name
        final_result['mode'] = 'batch'
        final_result['batch_ratio'] = batch_ratio
        final_result['num_batches'] = num_batches

        print(f"\n=== 分批分配完成 ===")
        print(f"  总分配任务数: {len(cumulative_assignment)}")
        print(f"  总失效节点数: {len(failure_model.failed_agents)}")
        print(f"  执行时间: {final_result['execution_time']:.2f}s")

        self.results_history.append(final_result)
        return final_result

    def _get_interrupted_tasks(self, cumulative_assignment: Dict[int, int],
                              agents: Dict) -> List[int]:
        """获取需要重新分配的中断任务"""
        interrupted = []
        for task_id, agent_id in list(cumulative_assignment.items()):
            agent = agents[agent_id]
            if agent.physical_state == 0 or agent.functional_state == 0:
                interrupted.append(task_id)
        return interrupted

    def _run_algorithm_for_batch(self, problem: RTMONFProblem,
                                 algorithm_name: str,
                                 task_ids: List[int],
                                 algorithm_params: Dict,
                                 seed: int) -> Dict[int, int]:
        """为一批任务运行算法"""
        if algorithm_name == 'RTM-RPF':
            algo = RTM_RPF(
                problem=problem,
                alpha_risk=algorithm_params.get('alpha_risk', 0.8),
                eta=algorithm_params.get('eta', 1.5),
                gamma=algorithm_params.get('gamma', 2.0),
                kappa_task=algorithm_params.get('kappa_task', 0.5),
                eta_rer=algorithm_params.get('eta_rer', 0.1),
                kappa_link=algorithm_params.get('kappa_link', 0.1),
                L_max=algorithm_params.get('L_max', 10.0),
                alpha1=algorithm_params.get('alpha1', 0.3),
                alpha2=algorithm_params.get('alpha2', 0.3),
                alpha3=algorithm_params.get('alpha3', 0.2),
                alpha4=algorithm_params.get('alpha4', 0.2),
                random_seed=seed
            )
            result = algo.solve(execute_failure=False, task_subset=task_ids)
        elif algorithm_name == 'CPLEX':
            from algorithms.cplex_solver import CPLEX_RTMONF_Solver
            algo = CPLEX_RTMONF_Solver(problem)
            result = algo.solve(execute_failure=False, task_subset=task_ids)
        elif algorithm_name == 'SPTM':
            algo = SPTM(problem, random_seed=seed)
            result = algo.solve(execute_failure=False, task_subset=task_ids)
        elif algorithm_name == 'LBTM':
            algo = LBTM(problem, random_seed=seed)
            result = algo.solve(execute_failure=False, task_subset=task_ids)
        else:
            raise ValueError(f"未知算法: {algorithm_name}")

        return result.get('task_assignment', {})

    def _compute_final_metrics(self, problem: RTMONFProblem,
                              cumulative_assignment: Dict[int, int],
                              failure_model) -> Dict:
        """计算最终性能指标（与原有格式一致）"""
        # 应用累积分配到任务集
        for task_id, agent_id in cumulative_assignment.items():
            task = problem.tasks.get_task(task_id)
            if task:
                task.assigned_agent = agent_id

        # 计算指标
        total_tasks = len(problem.tasks.get_all_tasks())
        assigned_tasks = len(cumulative_assignment)
        completion_ratio = assigned_tasks / total_tasks if total_tasks > 0 else 0.0

        # 计算总成本
        total_cost = 0.0
        for task_id, agent_id in cumulative_assignment.items():
            task = problem.tasks.get_task(task_id)
            if task:
                total_cost += task.workload

        # 计算效用（简化版）
        utility = completion_ratio * 100.0 - total_cost * 0.1

        return {
            'feasible': completion_ratio > 0,
            'utility': utility,
            'total_cost': total_cost,
            'completion_ratio': completion_ratio,
            'task_assignment': cumulative_assignment,
            'num_assigned_tasks': assigned_tasks,
            'total_failed_agents': len(failure_model.failed_agents),
            'failure_statistics': failure_model.get_statistics()
        }


    def clear_history(self):
        """清除历史结果"""
        self.results_history.clear()
