"""
M3: 消融实验 - 标准最短路径替代模块2
GD-RER补位 + 标准最短路径 + TA-RF任务分配

用于验证RPF-PP模块（模块2）的效果
"""
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import ResilientAgent, Role
from core.network import ResilientMultiLayerNetwork
from core.task import Task, TaskSet
from core.failure import FailureModel
from core.replenishment import ReplenishmentMechanism
from core.problem import RTMONFProblem


class AblationM3:
    """
    M3: GD-RER + 标准最短路径 + TA-RF
    - GD-RER：保留基于补位效能比的贪心决策（模块1）
    - 标准最短路径：替代RPF-PP（模块2）
    - TA-RF：保留韧性适配度任务分配（模块3）
    """

    def __init__(self,
                 problem: RTMONFProblem,
                 # 失效模型参数
                 alpha_risk: float = 0.8,
                 eta: float = 1.5,
                 gamma: float = 2.0,
                 kappa_task: float = 0.5,
                 # 补位机制参数
                 eta_rer: float = 0.1,
                 kappa_link: float = 0.1,
                 L_max: float = 10.0,
                 # 任务分配参数
                 alpha1: float = 0.35,
                 alpha2: float = 0.25,
                 alpha3: float = 0.2,
                 alpha4: float = 0.2,
                 random_seed: Optional[int] = None):
        self.problem = problem
        self.random_seed = random_seed

        # 初始化子模块
        self.failure_model = FailureModel(
            alpha_risk=alpha_risk,
            eta=eta,
            gamma=gamma,
            kappa_task=kappa_task
        )

        self.replenishment = ReplenishmentMechanism(
            alpha_risk=alpha_risk,
            eta_rer=eta_rer,
            kappa_link=kappa_link,
            L_max=L_max
        )

        # 任务分配参数
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

        # 关联子模块
        self.problem.set_failure_model(self.failure_model)
        self.problem.set_replenishment_mechanism(self.replenishment)

        # 算法状态
        self.role_assignment: Dict[int, Role] = {}
        self.task_assignment: Dict[int, int] = {}
        self.migration_flows: Dict[Tuple[int, int, int], int] = {}
        self.replenishment_plan: Dict[int, Tuple[int, Role]] = {}
        self.distance_matrix: Dict[Tuple[int, int], float] = {}

    def solve(self, execute_failure: bool = True, task_subset: Optional[List[int]] = None) -> Dict:
        """执行M3算法"""
        agents = self.problem.agents
        network = self.problem.network
        tasks = self.problem.tasks

        # 阶段0: 初始化
        self.failure_model.update_all_failure_probabilities(agents, network)

        # 阶段1: 执行失效判定
        if execute_failure:
            self.failure_model.execute_monte_carlo_death(agents, self.random_seed)

        # 阶段2: 识别级联失效
        isolated_agents, isolated_components = self.failure_model.identify_cascade_failures(
            agents, network
        )
        self.failure_model.identify_interrupted_tasks(agents, tasks)

        # 阶段3: GD-RER补位（保留模块1）
        if self.failure_model.failed_agents:
            failed_sorted = self.failure_model.get_failed_agents_sorted_by_impact(agents, tasks)

            self.replenishment_plan = self.replenishment.execute_gd_rer(
                failed_sorted,
                agents,
                network,
                isolated_components,
                self.problem.all_roles,
                network.role_to_layers
            )

            self.replenishment.execute_replenishment(
                agents, network, network.role_to_layers
            )

        # 更新角色分配
        self.role_assignment = {aid: agent.current_role for aid, agent in agents.items()}

        # 阶段4: 标准最短路径（替代RPF-PP）
        self.failure_model.update_all_failure_probabilities(agents, network)
        self.distance_matrix = self._compute_standard_distance_matrix(agents, network)

        # 阶段5: TA-RF任务分配（保留模块3，但使用标准距离）
        if task_subset is not None:
            tasks_to_assign = [tasks.get_task(tid) for tid in task_subset
                               if tasks.get_task(tid) is not None]
        else:
            tasks_to_assign = tasks.get_all_tasks()

        self.task_assignment, self.migration_flows = self._execute_ta_rf_with_distance(
            agents, network, tasks_to_assign
        )

        # 评估解
        results = self.problem.evaluate_solution(
            self.task_assignment,
            self.role_assignment,
            self.migration_flows,
            self.replenishment_plan
        )

        # 计算新任务的额外迁移成本（仿照M1）
        virtual_migration_cost = self._compute_virtual_migration_cost(
            self.migration_flows, tasks
        )

        # 将虚拟迁移成本加到总成本中
        results['migration_cost'] += virtual_migration_cost
        results['total_cost'] += virtual_migration_cost

        # 重新计算效用值
        results['utility'] = -self.problem.lambda1 * results['total_cost'] + \
                            self.problem.lambda2 * results['completion_ratio'] * 100.0

        results['failure_statistics'] = self.failure_model.get_statistics()
        results['replenishment_statistics'] = self.replenishment.get_replenishment_statistics()
        results['algorithm'] = 'M3'
        return results

    def _compute_standard_distance_matrix(
        self,
        agents: Dict[int, ResilientAgent],
        network: ResilientMultiLayerNetwork
    ) -> Dict[Tuple[int, int], float]:
        """计算标准最短路径距离矩阵（加权路径）"""
        distance_matrix = {}
        functional_agents = [aid for aid, agent in agents.items() if agent.is_functional]

        for source in functional_agents:
            for target in functional_agents:
                if source == target:
                    distance_matrix[(source, target)] = 0.0
                else:
                    # 使用加权最短路径（考虑边权重）
                    dist = network.compute_migration_cost(source, target)
                    distance_matrix[(source, target)] = dist if dist < float('inf') else float('inf')

        return distance_matrix

    def _execute_ta_rf_with_distance(
        self,
        agents: Dict[int, ResilientAgent],
        network: ResilientMultiLayerNetwork,
        tasks_to_assign: List[Task]
    ) -> Tuple[Dict[int, int], Dict[Tuple[int, int, int], int]]:
        """TA-RF任务分配（使用标准距离代替RPD）"""
        task_assignment = {}
        migration_flows = {}

        functional_agents = {aid: agent for aid, agent in agents.items() if agent.is_functional}
        if not functional_agents:
            return task_assignment, migration_flows

        # 计算距离最大值（用于归一化）
        dist_max = max(
            (d for d in self.distance_matrix.values() if d < float('inf')),
            default=1.0
        )
        if dist_max <= 0:
            dist_max = 1.0

        L_max = max(agent.load for agent in functional_agents.values()) + 1.0
        sorted_tasks = sorted(tasks_to_assign, key=lambda t: t.priority, reverse=True)
        agent_loads = {aid: agent.load for aid, agent in functional_agents.items()}

        for task in sorted_tasks:
            best_agent = None
            best_fitness = -float('inf')
            fallback_agent = None
            fallback_fitness = -float('inf')

            for agent_id, agent in functional_agents.items():
                if agent_loads[agent_id] + task.workload > self.problem.L_crit:
                    continue

                resilient_fitness = self._compute_resilient_fitness_with_distance(
                    agent, task, agent_loads[agent_id],
                    dist_max, L_max
                )

                role_fitness = agent.compute_role_task_fitness(task.requirements)

                if role_fitness >= self.problem.eta_phi:
                    if resilient_fitness > best_fitness:
                        best_fitness = resilient_fitness
                        best_agent = agent_id
                else:
                    if resilient_fitness > fallback_fitness:
                        fallback_fitness = resilient_fitness
                        fallback_agent = agent_id

            assigned_agent = best_agent if best_agent is not None else fallback_agent

            if assigned_agent is not None:
                task_assignment[task.task_id] = assigned_agent
                agent_loads[assigned_agent] += task.workload

                source_agent = task.current_agent
                if source_agent is None or source_agent not in functional_agents:
                    source_agent = assigned_agent
                migration_flows[(source_agent, assigned_agent, task.task_id)] = 1
                task.mark_migrated(assigned_agent)

        return task_assignment, migration_flows

    def _compute_resilient_fitness_with_distance(
        self,
        agent: ResilientAgent,
        task: Task,
        current_load: float,
        dist_max: float,
        L_max: float
    ) -> float:
        """计算完整的韧性适配度（使用跳数距离代替RPD）"""
        # 完整的适配度计算
        fitness = agent.compute_role_task_fitness(task.requirements)

        # 完整的存活概率计算（考虑网络层负载）
        layer_loads = self.problem.network.get_layer_loads()
        agent_layer_loads = {
            lid: layer_loads.get(lid, 0.0)
            for lid in agent.network_layers
        }

        base_survival = agent.compute_survival_probability(agent_layer_loads)
        survival = base_survival * (1.0 - agent.exposure_risk)

        # 使用跳数距离（而不是加权最短路径或RPD）
        source_id = task.current_agent
        if source_id is not None and source_id != agent.agent_id:
            dist = self.distance_matrix.get((source_id, agent.agent_id), float('inf'))
            if dist < float('inf'):
                normalized_dist = dist / dist_max
            else:
                normalized_dist = 1.0
        else:
            normalized_dist = 0.0

        predicted_load = (current_load + task.workload) / L_max

        # 完整的韧性适配度计算（与RTM-RPF相同的公式）
        resilient_fitness = (
            self.alpha1 * fitness +
            self.alpha2 * survival -
            self.alpha3 * normalized_dist -  # 使用跳数距离
            self.alpha4 * predicted_load
        )

        return resilient_fitness

    def _compute_virtual_migration_cost(
        self,
        migration_flows: Dict[Tuple[int, int, int], int],
        tasks: TaskSet
    ) -> float:
        """
        计算新任务的初始化成本（仿照M1）

        对于没有原始位置的任务（current_agent = None），需要额外的初始化成本
        使用网络平均迁移成本作为基准

        Args:
            migration_flows: 迁移流字典
            tasks: 任务集合

        Returns:
            新任务的初始化成本
        """
        virtual_cost = 0.0

        # 计算网络的平均迁移成本作为基准
        network = self.problem.network
        functional_agents = [aid for aid, agent in self.problem.agents.items()
                           if agent.is_functional]

        if len(functional_agents) < 2:
            # 如果功能节点太少，使用固定值
            avg_migration_cost = 1.0
        else:
            # 采样计算平均迁移成本
            sample_costs = []
            sample_size = min(20, len(functional_agents) * (len(functional_agents) - 1) // 2)

            for _ in range(sample_size):
                i = np.random.choice(functional_agents)
                j = np.random.choice([a for a in functional_agents if a != i])
                cost = network.compute_migration_cost(i, j)
                if cost < float('inf'):
                    sample_costs.append(cost)

            avg_migration_cost = np.mean(sample_costs) if sample_costs else 1.0

        # 统计新任务（没有原始位置的任务）的初始化成本
        for (source, target, task_id), flow in migration_flows.items():
            if flow > 0 and source == target:
                # source == target 可能是新任务（current_agent = None）
                task = tasks.get_task(task_id)
                if task and task.original_agent is None:
                    # 确认是新任务，添加初始化成本
                    virtual_cost += avg_migration_cost * task.workload*0.5

        return virtual_cost

    def reset(self):
        """重置算法状态"""
        self.role_assignment.clear()
        self.task_assignment.clear()
        self.migration_flows.clear()
        self.replenishment_plan.clear()
        self.distance_matrix.clear()
        self.failure_model.reset()
        self.replenishment.reset()

    def __repr__(self):
        return f"AblationM3(agents={len(self.problem.agents)}, tasks={len(self.problem.tasks)})"
