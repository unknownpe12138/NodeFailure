"""
M4: 消融实验 - 首次适配贪心替代模块3
GD-RER补位 + RPF-PP路径规划 + 首次适配贪心

用于验证TA-RF模块（模块3）的效果
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
from core.risk_field import RiskPotentialField
from core.problem import RTMONFProblem


class AblationM4:
    """
    M4: GD-RER + RPF-PP + 首次适配贪心
    - GD-RER：保留基于补位效能比的贪心决策（模块1）
    - RPF-PP：保留风险势能场路径规划（模块2）
    - 首次适配贪心：替代TA-RF（模块3）
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

        self.risk_field = RiskPotentialField(gamma=gamma)

        # 关联子模块
        self.problem.set_failure_model(self.failure_model)
        self.problem.set_replenishment_mechanism(self.replenishment)
        self.problem.set_risk_field(self.risk_field)

        # 算法状态
        self.role_assignment: Dict[int, Role] = {}
        self.task_assignment: Dict[int, int] = {}
        self.migration_flows: Dict[Tuple[int, int, int], int] = {}
        self.replenishment_plan: Dict[int, Tuple[int, Role]] = {}
        self.path_reliability: Dict[int, float] = {}

    def solve(self, execute_failure: bool = True, task_subset: Optional[List[int]] = None) -> Dict:
        """执行M4算法"""
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

        # 阶段4: RPF-PP路径规划（保留模块2）
        self.failure_model.update_all_failure_probabilities(agents, network)

        migration_task_ids = self.failure_model.get_migration_task_set(tasks)
        source_agents = set()
        for task_id in migration_task_ids:
            task = tasks.get_task(task_id)
            if task and task.current_agent is not None:
                source_agents.add(task.current_agent)

        for aid, agent in agents.items():
            if agent.is_functional:
                source_agents.add(aid)

        rpd_matrix = self.risk_field.execute_rpf_pp(agents, network, source_agents)

        # 阶段5: 贪心任务分配（替代TA-RF）
        if task_subset is not None:
            tasks_to_assign = [tasks.get_task(tid) for tid in task_subset
                               if tasks.get_task(tid) is not None]
        else:
            tasks_to_assign = tasks.get_all_tasks()

        self.task_assignment, self.migration_flows = self._greedy_allocation(
            agents, tasks_to_assign, rpd_matrix
        )

        # 计算路径可靠性
        task_locations = {
            task.task_id: task.current_agent
            for task in tasks.get_all_tasks()
            if task.current_agent is not None
        }
        self.path_reliability = self.risk_field.get_path_reliabilities(
            task_locations, self.task_assignment, agents
        )

        # 评估解
        results = self.problem.evaluate_solution(
            self.task_assignment,
            self.role_assignment,
            self.migration_flows,
            self.replenishment_plan,
            self.path_reliability
        )
        results['failure_statistics'] = self.failure_model.get_statistics()
        results['replenishment_statistics'] = self.replenishment.get_replenishment_statistics()
        results['risk_field_statistics'] = self.risk_field.get_statistics()
        results['algorithm'] = 'M4'
        return results

    def _greedy_allocation(
        self,
        agents: Dict[int, ResilientAgent],
        tasks_to_assign: List[Task],
        rpd_matrix: Dict[Tuple[int, int], float]
    ) -> Tuple[Dict[int, int], Dict[Tuple[int, int, int], int]]:
        """
        简单贪心分配（Simple Greedy）
        替代TA-RF，用于验证韧性适配度任务分配的效果

        选择适配度最高的智能体，而不是首次适配
        """
        task_assignment = {}
        migration_flows = {}

        functional_agents = {aid: agent for aid, agent in agents.items() if agent.is_functional}
        if not functional_agents:
            return task_assignment, migration_flows

        # 按优先级排序任务
        sorted_tasks = sorted(tasks_to_assign, key=lambda t: t.priority, reverse=True)

        # 初始化智能体负载
        agent_loads = {aid: agent.load for aid, agent in functional_agents.items()}

        # 简单贪心：综合考虑适配度、负载和距离
        for task in sorted_tasks:
            best_agent = None
            best_score = -float('inf')
            fallback_agent = None
            fallback_score = -float('inf')
            source = task.current_agent if task.current_agent in functional_agents else None

            # 遍历所有智能体，选择综合得分最高的
            for agent_id, agent in functional_agents.items():
                # 检查负载约束
                if agent_loads[agent_id] + task.workload > self.problem.L_crit:
                    continue

                # 检查可达性（使用RPD矩阵）
                rpd = float('inf')
                if source is not None and source != agent_id:
                    rpd = rpd_matrix.get((source, agent_id), float('inf'))
                    if rpd == float('inf'):
                        continue
                elif source == agent_id:
                    rpd = 0.0  # 本地执行

                # 计算适配度
                fitness = agent.compute_role_task_fitness(task.requirements)

                # 计算综合得分：适配度 + 负载均衡 + 距离因素
                # 负载因素：负载越低越好
                load_factor = 1.0 - (agent_loads[agent_id] / self.problem.L_crit)

                # 距离因素：距离越短越好（使用RPD）
                if source is not None:
                    # 获取RPD矩阵中的最大值作为归一化基准
                    rpd_values = [v for v in rpd_matrix.values() if v < float('inf')]
                    rpd_max = max(rpd_values) if rpd_values else 1.0
                    distance_factor = 1.0 - (rpd / rpd_max if rpd_max > 0 else 0.0)
                else:
                    distance_factor = 1.0  # 新任务，距离因素不重要

                # 综合得分：适配度权重0.5，负载权重0.3，距离权重0.2
                score = 0.5 * fitness + 0.3 * load_factor + 0.2 * distance_factor

                if fitness >= self.problem.eta_phi:
                    # 满足适配度阈值，选择得分最高的
                    if score > best_score:
                        best_score = score
                        best_agent = agent_id
                else:
                    # 不满足适配度阈值，作为fallback
                    if score > fallback_score:
                        fallback_score = score
                        fallback_agent = agent_id

            # 优先使用满足适配度的，否则使用fallback
            assigned_agent = best_agent if best_agent is not None else fallback_agent

            if assigned_agent is not None:
                task_assignment[task.task_id] = assigned_agent
                agent_loads[assigned_agent] += task.workload

                # 记录迁移流
                source_agent = source if source else assigned_agent
                migration_flows[(source_agent, assigned_agent, task.task_id)] = 1

                # 更新任务状态
                task.mark_migrated(assigned_agent)

        return task_assignment, migration_flows

    def reset(self):
        """重置算法状态"""
        self.role_assignment.clear()
        self.task_assignment.clear()
        self.migration_flows.clear()
        self.replenishment_plan.clear()
        self.path_reliability.clear()
        self.failure_model.reset()
        self.replenishment.reset()
        self.risk_field.reset()

    def __repr__(self):
        return f"AblationM4(agents={len(self.problem.agents)}, tasks={len(self.problem.tasks)})"
