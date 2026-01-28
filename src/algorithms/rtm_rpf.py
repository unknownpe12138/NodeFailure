"""
RTM-RPF算法实现
基于风险势能场的韧性任务迁移算法
(Resilient Task Migration based on Risk Potential Field)

包含三个核心子算法：
1. GD-RER: 基于补位效能比的贪心决策 (算法3-1)
2. RPF-PP: 风险势能场路径规划 (算法3-2)
3. TA-RF: 基于韧性适配度的任务分配 (算法3-3)
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


class RTM_RPF:
    """
    基于风险势能场的韧性任务迁移算法
    实现论文第3.3节的完整算法流程
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
                 # 其他参数
                 random_seed: Optional[int] = None):
        """
        Args:
            problem: RTM-ONF问题实例
            alpha_risk: 最大风险上限系数
            eta: 风险增长指数
            gamma: 风险敏感指数
            kappa_task: 任务影响权重系数
            eta_rer: 补位效能比阈值
            kappa_link: 单位连接建立代价系数
            L_max: 最大负载
            alpha1-alpha4: 韧性适配度函数权重
            random_seed: 随机种子
        """
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

        self.risk_field = RiskPotentialField(
            gamma=gamma
        )

        # 任务分配参数
        self.alpha1 = alpha1  # 适配度权重
        self.alpha2 = alpha2  # 存活率权重
        self.alpha3 = alpha3  # 路径风险权重
        self.alpha4 = alpha4  # 负载权重

        # 关联子模块到问题实例
        self.problem.set_failure_model(self.failure_model)
        self.problem.set_replenishment_mechanism(self.replenishment)
        self.problem.set_risk_field(self.risk_field)

        # 算法状态
        self.role_assignment: Dict[int, Role] = {}
        self.task_assignment: Dict[int, int] = {}
        self.migration_flows: Dict[Tuple[int, int, int], int] = {}
        self.replenishment_plan: Dict[int, Tuple[int, Role]] = {}
        self.path_reliability: Dict[int, float] = {}

    def solve(self, execute_failure: bool = True) -> Dict:
        """
        执行完整的RTM-RPF算法

        Args:
            execute_failure: 是否执行失效判定（用于测试）

        Returns:
            解的字典，包含各项指标和分配方案
        """
        agents = self.problem.agents
        network = self.problem.network
        tasks = self.problem.tasks

        # 保存初始角色
        self.initial_roles = {
            aid: agent.current_role
            for aid, agent in agents.items()
        }

        # ==================== 阶段0: 初始化 ====================
        # 更新所有节点的失效概率
        self.failure_model.update_all_failure_probabilities(agents, network)

        # ==================== 阶段1: 执行蒙特卡洛致死判定 ====================
        if execute_failure:
            newly_failed = self.failure_model.execute_monte_carlo_death(
                agents, self.random_seed
            )
        else:
            newly_failed = set()

        # ==================== 阶段2: 识别级联失效和中断任务 ====================
        isolated_agents, isolated_components = self.failure_model.identify_cascade_failures(
            agents, network
        )
        interrupted_tasks = self.failure_model.identify_interrupted_tasks(agents, tasks)

        # ==================== 阶段3: 基于风险-代价比的角色补位 (GD-RER) ====================
        if self.failure_model.failed_agents:
            # 按失效影响度排序失效节点
            failed_sorted = self.failure_model.get_failed_agents_sorted_by_impact(agents, tasks)

            # 执行GD-RER算法
            self.replenishment_plan = self.replenishment.execute_gd_rer(
                failed_sorted,
                agents,
                network,
                isolated_components,
                self.problem.all_roles,
                network.role_to_layers
            )

            # ==================== 阶段4: 执行补位并重建网络拓扑 ====================
            self.replenishment.execute_replenishment(
                agents, network, network.role_to_layers
            )

        # 更新角色分配
        self.role_assignment = {aid: agent.current_role for aid, agent in agents.items()}

        # ==================== 阶段5: 基于风险势能场的路径规划 (RPF-PP) ====================
        # 重新更新失效概率（补位后网络结构变化）
        self.failure_model.update_all_failure_probabilities(agents, network)

        # 获取待迁移任务的源节点
        migration_task_ids = self.failure_model.get_migration_task_set(tasks)
        source_agents = set()
        for task_id in migration_task_ids:
            task = tasks.get_task(task_id)
            if task and task.current_agent is not None:
                source_agents.add(task.current_agent)

        # 添加所有功能有效节点作为源
        for aid, agent in agents.items():
            if agent.is_functional:
                source_agents.add(aid)

        # 执行RPF-PP算法
        rpd_matrix = self.risk_field.execute_rpf_pp(agents, network, source_agents)

        # ==================== 阶段6: 基于韧性适配度的任务分配 (TA-RF) ====================
        self.task_assignment, self.migration_flows = self._execute_ta_rf(
            agents, network, tasks, rpd_matrix
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

        # ==================== 评估解 ====================
        results = self.problem.evaluate_solution(
            self.task_assignment,
            self.role_assignment,
            self.migration_flows,
            self.replenishment_plan,
            self.path_reliability
        )

        # 添加额外统计信息
        results['failure_statistics'] = self.failure_model.get_statistics()
        results['replenishment_statistics'] = self.replenishment.get_replenishment_statistics()
        results['risk_field_statistics'] = self.risk_field.get_statistics()

        return results

    def _execute_ta_rf(self,
                       agents: Dict[int, ResilientAgent],
                       network: ResilientMultiLayerNetwork,
                       tasks: TaskSet,
                       rpd_matrix: Dict[Tuple[int, int], float]) -> Tuple[Dict[int, int], Dict[Tuple[int, int, int], int]]:
        """
        算法3-3: 基于韧性适配度的任务分配 (TA-RF)

        Args:
            agents: 智能体字典
            network: 多重网络
            tasks: 任务集合
            rpd_matrix: 风险势能距离矩阵

        Returns:
            (任务分配方案, 迁移流)
        """
        task_assignment = {}
        migration_flows = {}

        # 获取功能有效的智能体
        functional_agents = {aid: agent for aid, agent in agents.items() if agent.is_functional}

        if not functional_agents:
            return task_assignment, migration_flows

        # 计算RPD最大值（用于归一化）
        rpd_max = self.risk_field.get_max_rpd()
        if rpd_max <= 0:
            rpd_max = 1.0

        # 计算最大负载（用于归一化）
        L_max = max(agent.load for agent in functional_agents.values()) + 1.0

        # 按优先级排序任务
        sorted_tasks = tasks.sort_by_priority()

        # 初始化智能体负载
        agent_loads = {aid: agent.load for aid, agent in functional_agents.items()}

        # 贪心分配任务
        for task in sorted_tasks:
            best_agent = None
            best_fitness = -float('inf')
            fallback_agent = None
            fallback_fitness = -float('inf')

            for agent_id, agent in functional_agents.items():
                # 检查负载约束
                if agent_loads[agent_id] + task.workload > self.problem.L_crit:
                    continue

                # 计算韧性适配度
                resilient_fitness = self._compute_resilient_fitness(
                    agent, task, agent_loads[agent_id],
                    rpd_matrix, rpd_max, L_max
                )

                # 检查适配度阈值
                role_fitness = agent.compute_role_task_fitness(task.requirements)

                if role_fitness >= self.problem.eta_phi:
                    if resilient_fitness > best_fitness:
                        best_fitness = resilient_fitness
                        best_agent = agent_id
                else:
                    # 降级候选
                    if resilient_fitness > fallback_fitness:
                        fallback_fitness = resilient_fitness
                        fallback_agent = agent_id

            # 选择最优智能体
            assigned_agent = best_agent if best_agent is not None else fallback_agent

            if assigned_agent is not None:
                task_assignment[task.task_id] = assigned_agent
                agent_loads[assigned_agent] += task.workload

                # 记录迁移流
                # 如果源节点是失效节点或不存在，使用目标节点作为源（本地执行）
                source_agent = task.current_agent
                if source_agent is None or source_agent not in functional_agents:
                    source_agent = assigned_agent
                migration_flows[(source_agent, assigned_agent, task.task_id)] = 1

                # 更新任务状态
                task.mark_migrated(assigned_agent)

        return task_assignment, migration_flows

    def _compute_resilient_fitness(self,
                                   agent: ResilientAgent,
                                   task: Task,
                                   current_load: float,
                                   rpd_matrix: Dict[Tuple[int, int], float],
                                   rpd_max: float,
                                   L_max: float) -> float:
        """
        计算韧性适配度函数 F_{i,k}^res - 定义3.15

        F = α1·Φ + α2·(1-p_fail) - α3·RPD/RPD_max - α4·(L+q)/L_max

        符合文档公式：
        - 式(2-11): π_i(τ) = (1 - Ψ_α(负载)) × ∏(1 - Ψ_β(层负载)) × (1 - μ_i^ξ)
        - 式(3-27): p_i^fail = 1 - π_i × (1 - μ_i^M)

        Args:
            agent: 智能体
            task: 任务
            current_load: 当前负载
            rpd_matrix: RPD矩阵
            rpd_max: 最大RPD
            L_max: 最大负载

        Returns:
            韧性适配度值
        """
        # 角色-任务适配度 Φ(ρ_i, τ_k) - 式(2-2)
        fitness = agent.compute_role_task_fitness(task.requirements)

        # 动态计算存活率 - 符合文档式(2-11)和式(3-27)
        # 获取网络层负载
        layer_loads = self.problem.network.get_layer_loads()
        agent_layer_loads = {
            lid: layer_loads.get(lid, 0.0)
            for lid in agent.network_layers
        }

        # 计算正常工作概率 π_i(τ) - 式(2-11)
        base_survival = agent.compute_survival_probability(agent_layer_loads)

        # 综合存活率 = π_i × (1 - μ_i^M) - 式(3-27)
        # agent.exposure_risk 是多重度中心性暴露风险 μ_i^M
        survival = base_survival * (1.0 - agent.exposure_risk)

        # 归一化路径风险 RPD/RPD_max
        source_id = task.current_agent
        if source_id is not None and source_id != agent.agent_id:
            rpd = rpd_matrix.get((source_id, agent.agent_id), float('inf'))
            if rpd < float('inf'):
                normalized_rpd = rpd / rpd_max
            else:
                normalized_rpd = 1.0  # 不可达给最大惩罚
        else:
            normalized_rpd = 0.0  # 无需迁移

        # 预测负载率 (L_i + q_k) / L_max
        predicted_load = (current_load + task.workload) / L_max

        # 韧性适配度 - 定义3.15
        resilient_fitness = (
            self.alpha1 * fitness +
            self.alpha2 * survival -
            self.alpha3 * normalized_rpd -
            self.alpha4 * predicted_load
        )

        return resilient_fitness

    def get_solution_summary(self) -> Dict:
        """获取解的摘要信息"""
        return {
            'algorithm': 'RTM-RPF',
            'num_agents': len(self.problem.agents),
            'num_functional': sum(1 for a in self.problem.agents.values() if a.is_functional),
            'num_tasks': len(self.problem.tasks),
            'num_assigned': len(self.task_assignment),
            'num_failed': len(self.failure_model.failed_agents),
            'num_isolated': len(self.failure_model.isolated_agents),
            'num_replenished': len(self.replenishment_plan),
            'num_interrupted': len(self.failure_model.interrupted_tasks)
        }

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
        return f"RTM_RPF(agents={len(self.problem.agents)}, tasks={len(self.problem.tasks)})"
