"""
M2: 消融实验 - 随机补位替代模块1
随机补位 + RPF-PP路径规划 + TA-RF任务分配

用于验证GD-RER模块（模块1）的效果
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
from core.risk_field import RiskPotentialField
from core.problem import RTMONFProblem


class AblationM2:
    """
    M2: 随机补位 + 保留模块2和模块3
    - 随机补位：替代GD-RER（模块1）
    - RPF-PP：保留风险势能场路径规划（模块2）
    - TA-RF：保留韧性适配度任务分配（模块3）
    """

    def __init__(self,
                 problem: RTMONFProblem,
                 # 失效模型参数
                 alpha_risk: float = 0.8,
                 eta: float = 1.5,
                 gamma: float = 2.0,
                 kappa_task: float = 0.5,
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

        self.risk_field = RiskPotentialField(gamma=gamma)

        # 任务分配参数
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

        # 关联子模块
        self.problem.set_failure_model(self.failure_model)
        self.problem.set_risk_field(self.risk_field)

        # 算法状态
        self.role_assignment: Dict[int, Role] = {}
        self.task_assignment: Dict[int, int] = {}
        self.migration_flows: Dict[Tuple[int, int, int], int] = {}
        self.replenishment_plan: Dict[int, Tuple[int, Role]] = {}
        self.path_reliability: Dict[int, float] = {}

    def solve(self, execute_failure: bool = True, task_subset: Optional[List[int]] = None) -> Dict:
        """执行M2算法"""
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

        # 阶段3: 随机补位（替代GD-RER）
        if self.failure_model.failed_agents:
            self.replenishment_plan = self._random_replenishment(
                agents, network, isolated_components
            )
            self._execute_replenishment(agents, network)

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

        # 阶段5: TA-RF任务分配（保留模块3）
        if task_subset is not None:
            tasks_to_assign = [tasks.get_task(tid) for tid in task_subset
                               if tasks.get_task(tid) is not None]
        else:
            tasks_to_assign = tasks.get_all_tasks()

        self.task_assignment, self.migration_flows = self._execute_ta_rf(
            agents, network, tasks_to_assign, rpd_matrix
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
        results['risk_field_statistics'] = self.risk_field.get_statistics()
        results['algorithm'] = 'M2'
        return results

    def _random_replenishment(
        self,
        agents: Dict[int, ResilientAgent],
        network: ResilientMultiLayerNetwork,
        isolated_components: Dict[int, Set[int]]
    ) -> Dict[int, Tuple[int, Role]]:
        """随机补位策略"""
        replenishment_plan = {}

        if self.random_seed is not None:
            np.random.seed(self.random_seed + 100)

        functional_agents = [aid for aid, agent in agents.items() if agent.is_functional]

        for failed_id in self.failure_model.failed_agents:
            failed_agent = agents[failed_id]
            failed_role = failed_agent.current_role

            if functional_agents:
                replacement_id = np.random.choice(functional_agents)
                replenishment_plan[failed_id] = (replacement_id, failed_role)

        return replenishment_plan

    def _execute_replenishment(
        self,
        agents: Dict[int, ResilientAgent],
        network: ResilientMultiLayerNetwork
    ):
        """执行补位操作"""
        for failed_id, (replacement_id, role) in self.replenishment_plan.items():
            replacement_agent = agents[replacement_id]
            failed_agent = agents[failed_id]

            # 执行角色切换（修复：使用switch_agent_role而不是add_role）
            network.switch_agent_role(replacement_id, role, save_original=True)

            role_layers = network.role_to_layers.get(role, set())
            for layer_id in role_layers:
                if layer_id in network.layers:
                    layer = network.layers[layer_id]

                    if failed_id in layer.graph:
                        for neighbor in list(layer.graph.neighbors(failed_id)):
                            if neighbor != replacement_id and agents[neighbor].is_functional:
                                layer.add_edge(replacement_id, neighbor, weight=1.0)

    def _execute_ta_rf(
        self,
        agents: Dict[int, ResilientAgent],
        network: ResilientMultiLayerNetwork,
        tasks_to_assign: List[Task],
        rpd_matrix: Dict[Tuple[int, int], float]
    ) -> Tuple[Dict[int, int], Dict[Tuple[int, int, int], int]]:
        """TA-RF任务分配（与RTM-RPF相同）"""
        task_assignment = {}
        migration_flows = {}

        functional_agents = {aid: agent for aid, agent in agents.items() if agent.is_functional}
        if not functional_agents:
            return task_assignment, migration_flows

        rpd_max = self.risk_field.get_max_rpd()
        if rpd_max <= 0:
            rpd_max = 1.0

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

                resilient_fitness = self._compute_resilient_fitness(
                    agent, task, agent_loads[agent_id],
                    rpd_matrix, rpd_max, L_max
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

    def _compute_resilient_fitness(
        self,
        agent: ResilientAgent,
        task: Task,
        current_load: float,
        rpd_matrix: Dict[Tuple[int, int], float],
        rpd_max: float,
        L_max: float
    ) -> float:
        """计算韧性适配度（与RTM-RPF相同）"""
        fitness = agent.compute_role_task_fitness(task.requirements)

        layer_loads = self.problem.network.get_layer_loads()
        agent_layer_loads = {
            lid: layer_loads.get(lid, 0.0)
            for lid in agent.network_layers
        }

        base_survival = agent.compute_survival_probability(agent_layer_loads)
        survival = base_survival * (1.0 - agent.exposure_risk)

        source_id = task.current_agent
        if source_id is not None and source_id != agent.agent_id:
            rpd = rpd_matrix.get((source_id, agent.agent_id), float('inf'))
            if rpd < float('inf'):
                normalized_rpd = rpd / rpd_max
            else:
                normalized_rpd = 1.0
        else:
            normalized_rpd = 0.0

        predicted_load = (current_load + task.workload) / L_max

        resilient_fitness = (
            self.alpha1 * fitness +
            self.alpha2 * survival -
            self.alpha3 * normalized_rpd -
            self.alpha4 * predicted_load
        )

        return resilient_fitness

    def reset(self):
        """重置算法状态"""
        self.role_assignment.clear()
        self.task_assignment.clear()
        self.migration_flows.clear()
        self.replenishment_plan.clear()
        self.path_reliability.clear()
        self.failure_model.reset()
        self.risk_field.reset()

    def __repr__(self):
        return f"AblationM2(agents={len(self.problem.agents)}, tasks={len(self.problem.tasks)})"
