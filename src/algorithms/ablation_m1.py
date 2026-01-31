"""
M1: 消融实验 - 最弱基线算法
完全随机分配（无补位、无路径规划、随机选择）

用作性能对比的最弱基线：
- 模块1：不执行补位（失效节点保持失效）
- 模块2：无路径规划（不计算距离，不检查可达性）
- 模块3：随机分配（检查负载和适配度约束，在满足约束的智能体中随机选择）

注意：M1仍然遵守问题的基本约束条件（负载约束和适配度约束），
但不进行优化，只是随机选择满足约束的智能体，以此作为性能对比的基线。
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
from core.problem import RTMONFProblem


class AblationM1:
    """
    M1: 最弱基线算法
    - 不执行补位：失效节点保持失效状态
    - 无路径规划：不计算距离矩阵，不检查可达性
    - 随机分配：检查负载和适配度约束，在满足约束的智能体中随机选择
    """

    def __init__(self,
                 problem: RTMONFProblem,
                 random_seed: Optional[int] = None):
        self.problem = problem
        self.random_seed = random_seed
        self.failure_model = FailureModel()

        # 算法状态
        self.role_assignment: Dict[int, Role] = {}
        self.task_assignment: Dict[int, int] = {}
        self.migration_flows: Dict[Tuple[int, int, int], int] = {}
        self.replenishment_plan: Dict[int, Tuple[int, Role]] = {}

    def solve(self, execute_failure: bool = True, task_subset: Optional[List[int]] = None) -> Dict:
        """执行M1算法（最弱基线）"""
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

        # 阶段3: 不执行补位（最弱基线特征）
        # 失效节点保持失效状态，不进行角色补位

        # 更新角色分配
        self.role_assignment = {aid: agent.current_role for aid, agent in agents.items()}

        # 阶段4: 随机任务分配（只检查负载约束）
        if task_subset is not None:
            tasks_to_assign = [tasks.get_task(tid) for tid in task_subset
                               if tasks.get_task(tid) is not None]
        else:
            tasks_to_assign = tasks.get_all_tasks()

        self.task_assignment, self.migration_flows = self._random_allocation(
            agents, tasks_to_assign
        )

        # 评估解
        results = self.problem.evaluate_solution(
            self.task_assignment,
            self.role_assignment,
            self.migration_flows,
            self.replenishment_plan
        )

        # 计算新任务的额外迁移成本
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
        results['algorithm'] = 'M1'
        return results

    def _random_allocation(
        self,
        agents: Dict[int, ResilientAgent],
        tasks_to_assign: List[Task]
    ) -> Tuple[Dict[int, int], Dict[Tuple[int, int, int], int]]:
        """随机分配（Random Allocation）- 检查负载和适配度约束，随机选择满足约束的智能体"""
        task_assignment = {}
        migration_flows = {}

        functional_agents = {aid: agent for aid, agent in agents.items() if agent.is_functional}
        if not functional_agents:
            return task_assignment, migration_flows

        # 设置随机种子
        if self.random_seed is not None:
            np.random.seed(self.random_seed + 200)

        # 初始化智能体负载
        agent_loads = {aid: agent.load for aid, agent in functional_agents.items()}

        # 随机分配：在满足负载和适配度约束的智能体中随机选择
        for task in tasks_to_assign:
            # 找出所有满足负载约束和适配度约束的智能体
            capable_agents = []
            fallback_agents = []  # 只满足负载约束的智能体（作为fallback）

            for aid, agent in functional_agents.items():
                if agent_loads[aid] + task.workload <= self.problem.L_crit:
                    # 计算角色-任务适配度
                    role_fitness = agent.compute_role_task_fitness(task.requirements)
                    if role_fitness >= self.problem.eta_phi:
                        capable_agents.append(aid)
                    else:
                        fallback_agents.append(aid)

            # 优先从满足适配度的智能体中随机选择，否则使用fallback
            assigned_agent = None
            if capable_agents:
                assigned_agent = np.random.choice(capable_agents)
            elif fallback_agents:
                assigned_agent = np.random.choice(fallback_agents)

            if assigned_agent is not None:
                task_assignment[task.task_id] = assigned_agent
                agent_loads[assigned_agent] += task.workload

                # 记录迁移流
                # 如果任务有原始位置，使用原始位置作为源（即使原节点已失效）
                # 如果任务没有原始位置（新任务），使用目标节点作为源（稍后会补偿成本）
                source_agent = task.current_agent
                if source_agent is None:
                    # 新任务，暂时使用目标节点作为源（避免破坏其他代码）
                    source_agent = assigned_agent
                # 否则，即使原节点失效，也记录真实的迁移流
                migration_flows[(source_agent, assigned_agent, task.task_id)] = 1

                # 更新任务状态
                task.mark_migrated(assigned_agent)

        return task_assignment, migration_flows

    def _compute_virtual_migration_cost(
        self,
        migration_flows: Dict[Tuple[int, int, int], int],
        tasks: TaskSet
    ) -> float:
        """
        计算新任务的初始化成本

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
                    virtual_cost += avg_migration_cost * task.workload

        return virtual_cost

    def reset(self):
        """重置算法状态"""
        self.role_assignment.clear()
        self.task_assignment.clear()
        self.migration_flows.clear()
        self.replenishment_plan.clear()
        self.failure_model.reset()

    def __repr__(self):
        return f"AblationM1(agents={len(self.problem.agents)}, tasks={len(self.problem.tasks)})"
