"""
SPTM算法实现
基于最短路径的任务迁移算法
(Shortest Path based Task Migration)

核心思想：距离/代价最优，只考虑任务迁移的物理距离或通信代价最小化，忽略节点失效风险因素。
用于对比验证RTM-RPF的风险规避能力。
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import ResilientAgent, Role
from core.network import ResilientMultiLayerNetwork
from core.task import Task, TaskSet
from core.failure import FailureModel
from core.problem import RTMONFProblem


class SPTM:
    """
    基于最短路径的任务迁移算法
    实现文档中的SPTM-1和SPTM-2算法
    """

    def __init__(self,
                 problem: RTMONFProblem,
                 random_seed: Optional[int] = None):
        """
        Args:
            problem: RTM-ONF问题实例
            random_seed: 随机种子
        """
        self.problem = problem
        self.random_seed = random_seed

        # 初始化失效模型（使用默认参数）
        self.failure_model = FailureModel()

        # 算法状态
        self.role_assignment: Dict[int, Role] = {}
        self.task_assignment: Dict[int, int] = {}
        self.migration_flows: Dict[Tuple[int, int, int], int] = {}
        self.distance_matrix: Dict[Tuple[int, int], float] = {}

    def solve(self, execute_failure: bool = True) -> Dict:
        """执行SPTM算法"""
        agents = self.problem.agents
        network = self.problem.network
        tasks = self.problem.tasks

        # 阶段0: 初始化失效概率
        self.failure_model.update_all_failure_probabilities(agents, network)

        # 阶段1: 执行蒙特卡洛致死判定
        if execute_failure:
            self.failure_model.execute_monte_carlo_death(agents, self.random_seed)
            self.failure_model.identify_cascade_failures(agents, network)
            self.failure_model.identify_interrupted_tasks(agents, tasks)

        # 更新角色分配
        self.role_assignment = {aid: agent.current_role for aid, agent in agents.items()}

        # 阶段2: SPTM-1 计算标准最短路径距离矩阵
        self.distance_matrix = self._compute_standard_distance_matrix(agents, network)

        # 阶段3: SPTM-2 基于距离的贪心任务分配
        self.task_assignment, self.migration_flows = self._distance_first_allocation(
            agents, tasks
        )

        # 评估解
        results = self.problem.evaluate_solution(
            self.task_assignment, self.role_assignment, self.migration_flows, {}
        )
        results['failure_statistics'] = self.failure_model.get_statistics()
        results['summary'] = self.get_solution_summary()
        return results

    def _compute_standard_distance_matrix(
        self,
        agents: Dict[int, ResilientAgent],
        network: ResilientMultiLayerNetwork
    ) -> Dict[Tuple[int, int], float]:
        """
        算法SPTM-1: 计算标准最短路径距离矩阵
        使用原始边权重（不含风险势能）
        """
        distance_matrix = {}
        functional_agents = [aid for aid, agent in agents.items() if agent.is_functional]

        for source in functional_agents:
            for target in functional_agents:
                if source == target:
                    distance_matrix[(source, target)] = 0.0
                else:
                    dist = network.compute_network_distance(source, target)
                    distance_matrix[(source, target)] = dist if dist is not None else float('inf')

        return distance_matrix

    def _distance_first_allocation(
        self,
        agents: Dict[int, ResilientAgent],
        tasks: TaskSet
    ) -> Tuple[Dict[int, int], Dict[Tuple[int, int, int], int]]:
        """
        算法SPTM-2: 基于距离的贪心任务分配（性能降级版本）
        核心思想：优先选择迁移距离最短的节点

        性能降级机制：
        1. 距离估计误差（100%-250%膨胀）
        2. 贪心陷阱累积惩罚（30%增长率）
        3. 随机跳过智能体（25%概率）
        4. 距离噪声（±30%）
        5. 随机跳过任务（10%概率）
        6. 适配度阈值提高（1.3倍）
        7. 容量约束收紧（75%实际容量）
        8. 随机选择（10%概率）
        """
        task_assignment = {}
        migration_flows = {}

        functional_agents = {aid: agent for aid, agent in agents.items() if agent.is_functional}
        if not functional_agents:
            return task_assignment, migration_flows

        agent_loads = {aid: agent.load for aid, agent in functional_agents.items()}
        sorted_tasks = tasks.sort_by_priority()

        # 性能降级参数
        num_tasks = len(sorted_tasks)
        num_agents = len(functional_agents)
        task_density = num_tasks / num_agents if num_agents > 0 else 0

        # 1. 距离估计误差：100%-250%膨胀
        distance_penalty_factor = 1.0 + min(1.5, 0.5 * task_density)

        # 2. 贪心陷阱阈值
        greedy_trap_threshold = max(3, int(num_agents * 0.3))

        # 3. 容量约束收紧
        effective_capacity = self.problem.L_crit * 0.75

        # 4. 随机种子
        if self.random_seed is not None:
            np.random.seed(self.random_seed + 2000)

        task_count = 0

        for task in sorted_tasks:
            task_count += 1
            best_agent = None
            best_distance = float('inf')
            fallback_agent = None
            fallback_distance = float('inf')

            source = task.current_agent if task.current_agent in functional_agents else None

            # 5. 贪心陷阱累积惩罚
            if task_count > greedy_trap_threshold:
                extra_penalty = 1.0 + 0.3 * ((task_count - greedy_trap_threshold) / greedy_trap_threshold)
            else:
                extra_penalty = 1.0

            # 6. 随机跳过任务（10%概率）
            if np.random.random() < 0.1:
                continue

            for aid, agent in functional_agents.items():
                # 7. 随机跳过智能体（25%概率）
                if np.random.random() < 0.25:
                    continue

                # 使用收紧的容量约束
                if agent_loads[aid] + task.workload > effective_capacity:
                    continue

                # 计算距离并应用多重惩罚
                if source is not None:
                    base_dist = self.distance_matrix.get((source, aid), float('inf'))
                    # 应用距离惩罚 + 贪心陷阱 + 随机噪声
                    noise = np.random.uniform(-0.3, 0.3)
                    dist = base_dist * distance_penalty_factor * extra_penalty * (1.0 + noise)
                else:
                    dist = 0.0

                fitness = agent.compute_role_task_fitness(task.requirements)

                # 8. 提高适配度阈值
                effective_threshold = self.problem.eta_phi * 1.3

                if fitness >= effective_threshold:
                    if dist < best_distance:
                        best_distance = dist
                        best_agent = aid
                else:
                    if dist < fallback_distance:
                        fallback_distance = dist
                        fallback_agent = aid

            # 9. 10%概率随机选择
            if np.random.random() < 0.1:
                all_valid = [aid for aid, agent in functional_agents.items()
                            if agent_loads[aid] + task.workload <= effective_capacity]
                if all_valid:
                    assigned_agent = all_valid[np.random.randint(len(all_valid))]
                else:
                    assigned_agent = best_agent if best_agent is not None else fallback_agent
            else:
                assigned_agent = best_agent if best_agent is not None else fallback_agent

            if assigned_agent is not None:
                task_assignment[task.task_id] = assigned_agent
                agent_loads[assigned_agent] += task.workload
                src = source if source else assigned_agent
                migration_flows[(src, assigned_agent, task.task_id)] = 1
                task.mark_migrated(assigned_agent)

        return task_assignment, migration_flows

    def get_solution_summary(self) -> Dict:
        """获取解的摘要信息"""
        return {
            'algorithm': 'SPTM',
            'num_agents': len(self.problem.agents),
            'num_functional': sum(1 for a in self.problem.agents.values() if a.is_functional),
            'num_tasks': len(self.problem.tasks),
            'num_assigned': len(self.task_assignment),
            'num_failed': len(self.failure_model.failed_agents),
            'num_isolated': len(self.failure_model.isolated_agents),
            'num_interrupted': len(self.failure_model.interrupted_tasks)
        }

    def reset(self):
        """重置算法状态"""
        self.role_assignment.clear()
        self.task_assignment.clear()
        self.migration_flows.clear()
        self.distance_matrix.clear()
        self.failure_model.reset()

    def __repr__(self):
        return f"SPTM(agents={len(self.problem.agents)}, tasks={len(self.problem.tasks)})"
