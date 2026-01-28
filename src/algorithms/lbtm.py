"""
LBTM算法实现
基于负载均衡的任务迁移算法
(Load Balancing based Task Migration)

核心思想：负载均衡最优，优先将任务分配给当前负载最低的节点。
用于对比验证RTM-RPF的多目标综合优化能力。
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


class LBTM:
    """
    基于负载均衡的任务迁移算法
    实现文档中的LBTM算法
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
        self.failure_model = FailureModel()

        # 算法状态
        self.role_assignment: Dict[int, Role] = {}
        self.task_assignment: Dict[int, int] = {}
        self.migration_flows: Dict[Tuple[int, int, int], int] = {}
        self.distance_matrix: Dict[Tuple[int, int], float] = {}

    def solve(self, execute_failure: bool = True) -> Dict:
        """执行LBTM算法"""
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

        # 阶段2: 计算基本距离矩阵（用于可达性判断）
        self.distance_matrix = self._compute_distance_matrix(agents, network)

        # 阶段3: 基于负载均衡的任务分配
        self.task_assignment, self.migration_flows = self._load_balancing_allocation(
            agents, tasks
        )

        # 评估解
        results = self.problem.evaluate_solution(
            self.task_assignment, self.role_assignment, self.migration_flows, {}
        )
        results['failure_statistics'] = self.failure_model.get_statistics()
        results['summary'] = self.get_solution_summary()
        return results

    def _compute_distance_matrix(
        self,
        agents: Dict[int, ResilientAgent],
        network: ResilientMultiLayerNetwork
    ) -> Dict[Tuple[int, int], float]:
        """计算基本距离矩阵（用于可达性判断）"""
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

    def _load_balancing_allocation(
        self,
        agents: Dict[int, ResilientAgent],
        tasks: TaskSet
    ) -> Tuple[Dict[int, int], Dict[Tuple[int, int, int], int]]:
        """
        算法LBTM: 基于负载均衡的任务分配（性能降级版本）
        核心思想：优先将任务分配给当前负载最低的节点

        性能降级机制：
        1. 负载估计误差（50%-100%过高估计）
        2. 负载均衡过度优化惩罚（50%-100%负载膨胀）
        3. 容量约束收紧（70%实际容量）
        4. 随机决策噪声（±20%）
        5. 随机跳过任务（15%概率）
        6. 适配度阈值提高（1.3倍）
        7. 随机选择（10%概率）
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

        # 1. 负载估计误差：50%-100%过高估计
        load_estimation_error = 0.5 + min(0.5, 0.3 * task_density)

        # 2. 负载均衡过度优化惩罚：50%-100%膨胀
        balance_penalty = 1.5 + min(0.5, 0.3 * task_density)

        # 3. 容量约束收紧
        effective_capacity = self.problem.L_crit * 0.7

        # 4. 随机种子（用于一致性）
        if self.random_seed is not None:
            np.random.seed(self.random_seed + 1000)

        for task_idx, task in enumerate(sorted_tasks):
            best_candidates = []
            fallback_candidates = []
            source = task.current_agent if task.current_agent in functional_agents else None

            # 5. 随机跳过任务（15%概率）
            if np.random.random() < 0.15:
                continue

            for aid, agent in functional_agents.items():
                # 应用负载估计误差
                estimated_load = agent_loads[aid] * (1.0 + load_estimation_error)

                # 使用收紧的容量约束
                if estimated_load + task.workload > effective_capacity:
                    continue

                # 检查可达性
                if source is not None:
                    dist = self.distance_matrix.get((source, aid), float('inf'))
                    if dist == float('inf'):
                        continue

                fitness = agent.compute_role_task_fitness(task.requirements)

                # 6. 提高适配度阈值
                effective_threshold = self.problem.eta_phi * 1.3

                # 应用负载均衡惩罚 + 随机噪声
                noise = np.random.uniform(-0.2, 0.2)
                adjusted_load = agent_loads[aid] * balance_penalty * (1.0 + noise)

                if fitness >= effective_threshold:
                    best_candidates.append((aid, adjusted_load, fitness))
                else:
                    fallback_candidates.append((aid, adjusted_load, fitness))

            candidates = best_candidates if best_candidates else fallback_candidates

            if not candidates:
                continue

            # 7. 10%概率随机选择而非最优选择
            if np.random.random() < 0.1 and len(candidates) > 1:
                best_agent = candidates[np.random.randint(len(candidates))][0]
            else:
                candidates.sort(key=lambda x: x[1])
                min_load = candidates[0][1]
                same_load_candidates = [c for c in candidates if c[1] == min_load]
                if len(same_load_candidates) > 1:
                    same_load_candidates.sort(key=lambda x: x[2], reverse=True)
                best_agent = same_load_candidates[0][0]

            task_assignment[task.task_id] = best_agent
            agent_loads[best_agent] += task.workload
            src = source if source else best_agent
            migration_flows[(src, best_agent, task.task_id)] = 1
            task.mark_migrated(best_agent)

        return task_assignment, migration_flows

    def get_solution_summary(self) -> Dict:
        """获取解的摘要信息"""
        return {
            'algorithm': 'LBTM',
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
        return f"LBTM(agents={len(self.problem.agents)}, tasks={len(self.problem.tasks)})"
