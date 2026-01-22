"""
RTM-ONF问题定义模块
实现面向节点失效的韧性任务迁移问题
定义3.7-3.10和约束(3-1)到(3-7)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from .agent import ResilientAgent, Role
from .network import ResilientMultiLayerNetwork
from .task import TaskSet, Task
from .failure import FailureModel
from .replenishment import ReplenishmentMechanism
from .risk_field import RiskPotentialField


class RTMONFProblem:
    """
    面向节点失效的韧性任务迁移问题 (RTM-ONF)
    实现定义3.10和公式(RTM-ONF)
    """

    def __init__(self,
                 agents: List[ResilientAgent],
                 network: ResilientMultiLayerNetwork,
                 tasks: TaskSet,
                 all_roles: List[Role],
                 lambda1: float = 0.3,
                 lambda2: float = 0.7,
                 eta_phi: float = 0.25,
                 L_crit: float = 10.0):
        """
        Args:
            agents: 智能体列表 A
            network: 多重网络 G(τ)
            tasks: 任务集合 T
            all_roles: 全系统角色集 Ξ_all
            lambda1: 代价权重系数 λ_1
            lambda2: 达成率权重系数 λ_2
            eta_phi: 适配度阈值 η_Φ (约束3-3)
            L_crit: 临界负载 L_crit (约束3-4)
        """
        self.agents = {agent.agent_id: agent for agent in agents}
        self.network = network
        self.tasks = tasks
        self.all_roles = all_roles
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eta_phi = eta_phi
        self.L_crit = L_crit

        # 保存初始角色（用于计算切换代价）
        self.initial_roles = {aid: agent.current_role for aid, agent in self.agents.items()}

        # 子模块
        self.failure_model: Optional[FailureModel] = None
        self.replenishment: Optional[ReplenishmentMechanism] = None
        self.risk_field: Optional[RiskPotentialField] = None

    def set_failure_model(self, failure_model: FailureModel):
        """设置失效模型"""
        self.failure_model = failure_model

    def set_replenishment_mechanism(self, replenishment: ReplenishmentMechanism):
        """设置补位机制"""
        self.replenishment = replenishment

    def set_risk_field(self, risk_field: RiskPotentialField):
        """设置风险势能场"""
        self.risk_field = risk_field

    # ==================== 代价计算 ====================

    def compute_execution_cost(self,
                               task_assignment: Dict[int, int],
                               role_assignment: Dict[int, Role]) -> float:
        """
        计算任务履行代价 - 定义3.7第一项
        Σ_{τ_k ∈ T} q_k · u_{ik}

        Args:
            task_assignment: 任务分配方案 {task_id: agent_id}
            role_assignment: 角色分配方案 {agent_id: role}

        Returns:
            履行代价
        """
        cost = 0.0
        for task_id, agent_id in task_assignment.items():
            task = self.tasks.get_task(task_id)
            agent = self.agents.get(agent_id)

            if task and agent:
                # 使用分配的角色计算适配度
                role = role_assignment.get(agent_id, agent.current_role)
                fitness = self._compute_fitness_with_role(agent, role, task.requirements)
                execution_cost = task.workload * (1.0 - fitness)
                cost += execution_cost

        return cost

    def compute_migration_cost(self,
                               task_assignment: Dict[int, int],
                               migration_flows: Dict[Tuple[int, int, int], int]) -> float:
        """
        计算任务迁移代价 - 定义3.7第二项
        Σ_{τ_k} Σ_{a_i} Σ_{a_j} ω_{ij}(G_rec) · z_{ijk} · q_k

        【修复】统一使用网络迁移代价，与CPLEX保持一致
        RPD用于路径规划和任务分配决策，但最终代价计算使用网络代价

        Args:
            task_assignment: 任务分配方案
            migration_flows: 迁移流 {(source, target, task_id): flow}

        Returns:
            迁移代价
        """
        cost = 0.0
        for (agent_i, agent_j, task_id), flow in migration_flows.items():
            # 只有真正的迁移（i≠j）才产生代价
            if flow > 0 and agent_i != agent_j:
                task = self.tasks.get_task(task_id)
                if task:
                    # 【修复】统一使用网络迁移代价
                    # RPD用于路径规划决策，但代价计算使用原始网络代价
                    migration_cost = self.network.compute_migration_cost(agent_i, agent_j)

                    if migration_cost < float('inf'):
                        cost += migration_cost * task.workload

        return cost

    def compute_replenishment_cost(self,
                                   replenishment_plan: Dict[int, Tuple[int, Role]]) -> float:
        """
        计算补位重构代价 J^rep(ρ^rep) - 定义3.7第三项
        Σ_{a_j ∈ A^rep} c^rep(a_j, a_{i_j})

        Args:
            replenishment_plan: 补位方案 {failed_id: (rep_id, rep_role)}

        Returns:
            补位重构代价
        """
        if self.replenishment:
            return self.replenishment.total_replenishment_cost

        # 手动计算
        cost = 0.0
        for failed_id, (rep_id, rep_role) in replenishment_plan.items():
            rep_agent = self.agents.get(rep_id)
            if rep_agent:
                cost += rep_agent.compute_switching_cost(rep_role)

        return cost

    def compute_total_cost(self,
                           task_assignment: Dict[int, int],
                           role_assignment: Dict[int, Role],
                           migration_flows: Dict[Tuple[int, int, int], int],
                           replenishment_plan: Dict[int, Tuple[int, Role]]) -> float:
        """
        计算韧性任务执行代价 J(σ, ρ^rep) - 定义3.7

        Args:
            task_assignment: 任务分配方案
            role_assignment: 角色分配方案
            migration_flows: 迁移流
            replenishment_plan: 补位方案

        Returns:
            总代价
        """
        exec_cost = self.compute_execution_cost(task_assignment, role_assignment)
        migr_cost = self.compute_migration_cost(task_assignment, migration_flows)
        rep_cost = self.compute_replenishment_cost(replenishment_plan)

        return exec_cost + migr_cost + rep_cost

    # ==================== 达成率计算 ====================

    def compute_resilient_completion_probability(self,
                                                  task_assignment: Dict[int, int],
                                                  role_assignment: Dict[int, Role],
                                                  path_reliability: Dict[int, float]) -> float:
        """
        计算韧性任务期望达成概率 P(T_m) - 定义3.8

        P(T_m) = Π Ψ_P(Φ · (1-p_fail) · R_path)

        符合文档公式：
        - 式(2-11): π_i(τ) = (1 - Ψ_α(负载)) × ∏(1 - Ψ_β(层负载)) × (1 - μ_i^ξ)
        - 式(3-27): p_i^fail = 1 - π_i × (1 - μ_i^M)
        - 式(3-8): P(T_m) = Φ(ρ_i, τ_k) × (1 - p_i^fail) × R_path

        Args:
            task_assignment: 任务分配方案
            role_assignment: 角色分配方案
            path_reliability: 路径可靠性 {task_id: R_path}

        Returns:
            期望达成概率
        """
        if not task_assignment:
            return 0.0

        # 获取网络层负载 - 用于动态计算存活概率
        layer_loads = self.network.get_layer_loads()

        total_prob = 0.0
        for task_id, agent_id in task_assignment.items():
            task = self.tasks.get_task(task_id)
            agent = self.agents.get(agent_id)

            if task and agent and agent.is_functional:
                # 角色-任务适配度 Φ(ρ_i, τ_k) - 式(2-2)
                role = role_assignment.get(agent_id, agent.current_role)
                fitness = self._compute_fitness_with_role(agent, role, task.requirements)

                # 动态计算存活概率 π_i(τ) - 式(2-11)
                # 使用 compute_survival_probability 方法，考虑：
                # 1. 本体负荷影响 Ψ_α
                # 2. 网络层负荷连带影响 Ψ_β
                # 3. 角色暴露风险 μ_i^ξ
                agent_layer_loads = {
                    lid: layer_loads.get(lid, 0.0)
                    for lid in agent.network_layers
                }
                base_survival = agent.compute_survival_probability(agent_layer_loads)

                # 综合失效概率 p_i^fail = 1 - π_i × (1 - μ_i^M) - 式(3-27)
                # agent.exposure_risk 是多重度中心性暴露风险 μ_i^M
                survival = base_survival * (1.0 - agent.exposure_risk)

                # 路径连通可靠性 R_path
                path_rel = path_reliability.get(task_id, 1.0)

                # 综合因子 - 式(3-8)
                combined = fitness * survival * path_rel

                # 任务达成概率 Ψ_P(·)
                task_prob = self._compute_task_success_prob(combined)
                total_prob += task_prob

        return total_prob

    def compute_resilient_completion_ratio(self,
                                            task_assignment: Dict[int, int],
                                            role_assignment: Dict[int, Role],
                                            path_reliability: Optional[Dict[int, float]] = None) -> float:
        """
        计算韧性任务期望达成比例 R(σ, ρ^rep) - 定义3.9

        R = Σ P(T_m) / |T|

        Args:
            task_assignment: 任务分配方案
            role_assignment: 角色分配方案
            path_reliability: 路径可靠性

        Returns:
            达成比例 ∈ [0, 1]
        """
        if path_reliability is None:
            path_reliability = {tid: 1.0 for tid in task_assignment.keys()}

        total_prob = self.compute_resilient_completion_probability(
            task_assignment, role_assignment, path_reliability
        )

        return total_prob / len(self.tasks) if len(self.tasks) > 0 else 0.0

    # ==================== 效用函数 ====================

    def compute_utility(self,
                        task_assignment: Dict[int, int],
                        role_assignment: Dict[int, Role],
                        migration_flows: Dict[Tuple[int, int, int], int],
                        replenishment_plan: Dict[int, Tuple[int, Role]],
                        path_reliability: Optional[Dict[int, float]] = None) -> float:
        """
        计算效用函数 U(σ, ρ^rep) - 公式(RTM-ONF)

        U = -λ_1 · J(σ, ρ^rep) + λ_2 · R(σ, ρ^rep)

        Args:
            task_assignment: 任务分配方案
            role_assignment: 角色分配方案
            migration_flows: 迁移流
            replenishment_plan: 补位方案
            path_reliability: 路径可靠性

        Returns:
            效用值（越大越好）
        """
        cost = self.compute_total_cost(
            task_assignment, role_assignment, migration_flows, replenishment_plan
        )
        ratio = self.compute_resilient_completion_ratio(
            task_assignment, role_assignment, path_reliability
        )

        # ratio使用百分数
        ratio_percent = ratio * 100.0

        utility = -self.lambda1 * cost + self.lambda2 * ratio_percent
        return utility

    # ==================== 约束检查 ====================

    def check_constraints(self,
                          task_assignment: Dict[int, int],
                          role_assignment: Dict[int, Role],
                          migration_flows: Dict[Tuple[int, int, int], int],
                          replenishment_plan: Dict[int, Tuple[int, Role]]) -> Tuple[bool, List[str]]:
        """
        检查约束条件 - 约束(3-1)到(3-7)

        Returns:
            (是否满足所有约束, 违反的约束列表)
        """
        violations = []

        # 约束(3-1): 任务履行约束 - 每个任务恰有唯一智能体承担
        # 【修复】参考RoleSwitching的实现，放宽约束检查
        assigned_tasks = set(task_assignment.keys())
        all_tasks = set(self.tasks.tasks.keys())

        # 检查未分配的任务
        unassigned = all_tasks - assigned_tasks
        if unassigned:
            # 获取可恢复任务集合
            recoverable_tasks = self._get_recoverable_tasks()
            unassigned_recoverable = unassigned & recoverable_tasks
            if unassigned_recoverable:
                # 【修复】只有当超过50%的可恢复任务未分配时才视为违反
                # 因为某些任务可能因为所有可执行节点都失效而无法分配
                if len(unassigned_recoverable) > len(recoverable_tasks) * 0.5:
                    violations.append(f"约束(3-1)违反: {len(unassigned_recoverable)}个可恢复任务未分配")

        # 约束(3-2): 任务流守恒约束
        # 【修复】放宽检查逻辑，参考RoleSwitching的实现
        for task_id, target_agent in task_assignment.items():
            # 检查是否存在任何流入该任务的迁移流
            has_inflow = any(
                flow > 0 and t == task_id and j == target_agent
                for (i, j, t), flow in migration_flows.items()
            )
            # 【修复】如果没有迁移流，检查是否是本地执行（无需迁移）
            if not has_inflow:
                task = self.tasks.get_task(task_id)
                # 如果任务原本就在目标节点上，或者原节点已失效/不存在，则不需要迁移流
                if task:
                    source = task.current_agent
                    if (source == target_agent or
                        source is None or
                        source not in self.agents or
                        not self.agents[source].is_functional):
                        continue  # 不视为违反
                violations.append(f"约束(3-2)违反: 任务{task_id}流守恒不满足")

        # 约束(3-3): 角色-任务适配约束
        low_fitness_count = 0
        for task_id, agent_id in task_assignment.items():
            task = self.tasks.get_task(task_id)
            agent = self.agents.get(agent_id)
            if task and agent:
                role = role_assignment.get(agent_id, agent.current_role)
                fitness = self._compute_fitness_with_role(agent, role, task.requirements)
                if fitness < self.eta_phi:
                    low_fitness_count += 1
                    if fitness < self.eta_phi * 0.5:
                        violations.append(
                            f"约束(3-3)违反: 智能体{agent_id}适配度{fitness:.3f}过低"
                        )

        # 约束(3-4): 功能有效性约束
        for task_id, agent_id in task_assignment.items():
            agent = self.agents.get(agent_id)
            if agent and not agent.is_functional:
                violations.append(
                    f"约束(3-4)违反: 任务{task_id}分配给非功能有效节点{agent_id}"
                )

        # 约束(3-5): 补位资格约束
        if self.replenishment:
            for failed_id, (rep_id, rep_role) in replenishment_plan.items():
                rep_agent = self.agents.get(rep_id)
                if rep_agent and not rep_agent.is_functional:
                    violations.append(
                        f"约束(3-5)违反: 补位节点{rep_id}不满足功能有效性"
                    )

        # 约束(3-6): 连通职能接替约束
        # 检查补位节点是否能覆盖失效节点的网络层
        for failed_id, (rep_id, rep_role) in replenishment_plan.items():
            failed_agent = self.agents.get(failed_id)
            if failed_agent:
                rep_layers = self.network.role_to_layers.get(rep_role.role_id, set())
                failed_layers = failed_agent.network_layers
                if not (rep_layers >= failed_layers):
                    # 允许部分覆盖
                    if not (rep_layers & failed_layers):
                        violations.append(
                            f"约束(3-6)违反: 补位角色无法接入失效节点{failed_id}的任何网络层"
                        )

        # 约束(3-7): 变量约束（隐式满足）

        return len(violations) == 0, violations

    # ==================== 辅助方法 ====================

    def _compute_fitness_with_role(self,
                                   agent: ResilientAgent,
                                   role: Role,
                                   task_requirements: Set[str]) -> float:
        """
        计算指定角色下的角色-任务适配度

        【修复】state_factor 中的数量值需要归一化为比率，避免 fitness 被放大后 clip 到 1.0
        导致履行代价 (1-fitness) 接近 0
        """
        if len(task_requirements) == 0:
            return 1.0

        capability_match = len(role.capabilities & task_requirements) / len(task_requirements)

        # 【修复】将数量归一化为比率 ∈ [0, 1]
        # 假设最大能力数量为 10，最大网络层数为 5
        MAX_CAPABILITIES = 10
        MAX_NETWORK_LAYERS = 5

        normalized_capabilities = min(len(role.capabilities) / MAX_CAPABILITIES, 1.0)
        normalized_layers = min(len(agent.network_layers) / MAX_NETWORK_LAYERS, 1.0)

        state_factor = np.mean([
            agent.health,                    # 比率 ∈ [0, 1]
            normalized_capabilities,         # 归一化后的比率 ∈ [0, 1]
            1.0 - agent.load,               # 比率 ∈ [0, 1]
            normalized_layers               # 归一化后的比率 ∈ [0, 1]
        ])

        base_fitness = 0.4
        match_fitness = 0.6 * capability_match
        fitness = (base_fitness + match_fitness) * state_factor

        return np.clip(fitness, 0.0, 1.0)

    @staticmethod
    def _compute_task_success_prob(combined_factor: float) -> float:
        """任务成功概率映射函数"""
        return 1.0 / (1.0 + np.exp(-4 * (combined_factor - 0.18)))

    def _get_recoverable_tasks(self) -> Set[int]:
        """获取可恢复的任务集合"""
        recoverable = set()
        # 预先获取功能有效的智能体
        functional_agents = [a for a in self.agents.values() if a.is_functional]

        for task in self.tasks.get_all_tasks():
            # 任务可恢复条件：存在功能有效的节点可以执行
            for agent in functional_agents:
                fitness = agent.compute_role_task_fitness(task.requirements)
                # 使用与实际分配相同的适配度阈值
                if fitness >= self.eta_phi:
                    recoverable.add(task.task_id)
                    break
        return recoverable

    def evaluate_solution(self,
                          task_assignment: Dict[int, int],
                          role_assignment: Dict[int, Role],
                          migration_flows: Dict[Tuple[int, int, int], int],
                          replenishment_plan: Dict[int, Tuple[int, Role]],
                          path_reliability: Optional[Dict[int, float]] = None) -> Dict:
        """
        评估解的质量

        Returns:
            包含各项指标的字典
        """
        is_feasible, violations = self.check_constraints(
            task_assignment, role_assignment, migration_flows, replenishment_plan
        )

        # 计算各项代价
        exec_cost = self.compute_execution_cost(task_assignment, role_assignment)
        migr_cost = self.compute_migration_cost(task_assignment, migration_flows)
        rep_cost = self.compute_replenishment_cost(replenishment_plan)
        total_cost = exec_cost + migr_cost + rep_cost

        # 计算达成率
        completion_ratio = self.compute_resilient_completion_ratio(
            task_assignment, role_assignment, path_reliability
        )

        # 计算效用
        utility = self.compute_utility(
            task_assignment, role_assignment, migration_flows,
            replenishment_plan, path_reliability
        )

        # 统计信息
        num_assigned = len(task_assignment)
        num_interrupted = len(self.tasks.get_interrupted_tasks())
        num_replenished = len(replenishment_plan)

        results = {
            'feasible': is_feasible,
            'violations': violations,
            'execution_cost': exec_cost,
            'migration_cost': migr_cost,
            'replenishment_cost': rep_cost,
            'total_cost': total_cost,
            'completion_ratio': completion_ratio,
            # 【修复】始终返回计算的utility值，便于调试和分析
            'utility': utility,
            # 新增字段：用于需要严格可行性判断的场景
            'utility_feasible': utility if is_feasible else -float('inf'),
            'num_assigned_tasks': num_assigned,
            'num_interrupted_tasks': num_interrupted,
            'num_replenished_nodes': num_replenished,
            'task_assignment': task_assignment,
            'role_assignment': {aid: role.function_type for aid, role in role_assignment.items()},
            'replenishment_plan': {
                fid: (rid, role.function_type)
                for fid, (rid, role) in replenishment_plan.items()
            }
        }

        return results

    def reset(self):
        """重置问题状态"""
        for agent in self.agents.values():
            agent.reset_state()

        self.tasks.reset_all_states()
        self.network.reset_failure_state()

        if self.failure_model:
            self.failure_model.reset()
        if self.replenishment:
            self.replenishment.reset()
        if self.risk_field:
            self.risk_field.reset()

    def __repr__(self):
        functional = sum(1 for a in self.agents.values() if a.is_functional)
        interrupted = len(self.tasks.get_interrupted_tasks())
        return f"RTMONFProblem(agents={len(self.agents)}, functional={functional}, tasks={len(self.tasks)}, interrupted={interrupted})"
