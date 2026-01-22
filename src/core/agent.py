"""
韧性智能体模型模块
实现面向节点失效的智能体模型，包含失效概率、风险势能等属性
基于研究点一的Agent类进行扩展
"""
import numpy as np
from typing import List, Set, Dict, Optional


class Role:
    """
    角色类 - 定义2.1: ξ_i = ⟨e_i, g_i⟩
    与研究点一相同
    """

    def __init__(self, role_id: int, capabilities: Set[str],
                 function_type: str, exposure_risk: float = 0.5):
        """
        Args:
            role_id: 角色编号
            capabilities: 能力特征集合 e_i
            function_type: 职能属性 g_i (如'scout', 'striker', 'relay'等)
            exposure_risk: 角色暴露风险 μ^ξ (定义2.5)
        """
        self.role_id = role_id
        self.capabilities = capabilities  # e_i
        self.function_type = function_type  # g_i
        self.exposure_risk = exposure_risk  # μ^ξ

    def __repr__(self):
        return f"Role(id={self.role_id}, type={self.function_type}, caps={len(self.capabilities)})"

    def __eq__(self, other):
        if not isinstance(other, Role):
            return False
        return self.role_id == other.role_id

    def __hash__(self):
        return hash(self.role_id)


class ResilientAgent:
    """
    韧性智能体类
    在研究点一Agent基础上扩展节点失效相关属性和方法
    实现定义3.1-3.3的失效机制
    """

    def __init__(self, agent_id: int, feasible_roles: List[Role],
                 current_role: Optional[Role] = None,
                 initial_health: float = 1.0):
        """
        Args:
            agent_id: 智能体编号
            feasible_roles: 可行角色集合 Ξ_i (定义2.2)
            current_role: 当前角色 ξ_i
            initial_health: 初始健康度
        """
        # === 继承研究点一的基础属性 ===
        self.agent_id = agent_id
        self.feasible_roles = feasible_roles  # Ξ_i
        self.current_role = current_role or feasible_roles[0]  # ξ_i(τ)
        self.health = initial_health  # 健康度
        self.assigned_tasks: List[int] = []  # 分配的任务ID列表
        self.load = 0.0  # 当前负载
        self.network_layers: Set[int] = set()  # 隶属的网络层集合 Λ_i(τ)

        # === 新增：节点失效相关属性 ===
        self.physical_state = 1       # s_i^phy(τ): 物理存亡状态 {0, 1}
        self.functional_state = 1     # s_i^func(τ): 功能有效状态 {0, 1}
        self.multiplex_degree = 0.0   # d_i^M(τ): 多重度中心性
        self.exposure_risk = 0.0      # μ_i^M(τ): 暴露风险
        self.failure_prob = 0.0       # p_i^fail(τ): 综合失效概率
        self.risk_potential = 1.0     # U_i(τ): 风险势能

        # 补位相关
        self.is_replenishment_node = False  # 是否为补位节点
        self.replenished_for: Optional[int] = None  # 补位的目标失效节点ID
        self.original_role: Optional[Role] = None  # 补位前的原始角色

    @property
    def is_alive(self) -> bool:
        """节点是否物理存活"""
        return self.physical_state == 1

    @property
    def is_functional(self) -> bool:
        """节点是否功能有效（物理存活且在主连通分量内）"""
        return self.functional_state == 1

    @property
    def capabilities(self) -> Set[str]:
        """当前能力集合 e_i(τ)"""
        return self.current_role.capabilities

    @property
    def function_type(self) -> str:
        """当前职能属性 g_i"""
        return self.current_role.function_type

    def get_role_state_vector(self) -> np.ndarray:
        """
        获取角色状态向量 h_i(τ)
        综合反映能力状况、设备健康度等
        """
        return np.array([
            self.health,
            len(self.capabilities),
            1.0 - self.load,  # 剩余容量
            len(self.network_layers)
        ])

    def compute_multiplex_degree_centrality(self,
                                            layer_degrees: Dict[int, float],
                                            layer_weights: Dict[int, float]) -> float:
        """
        计算多重度中心性 d_i^M(τ) - 定义3.1
        d_i^M(τ) = Σ_{ℓ∈Λ_i} β^(ℓ) · d_i^(ℓ)(τ)

        Args:
            layer_degrees: 各层度中心性 {layer_id: degree}
            layer_weights: 各层暴露敏感度权重 β^(ℓ)

        Returns:
            多重度中心性值
        """
        self.multiplex_degree = sum(
            layer_weights.get(lid, 1.0) * layer_degrees.get(lid, 0.0)
            for lid in self.network_layers
        )
        return self.multiplex_degree

    def compute_exposure_risk(self, max_multiplex_degree: float,
                              alpha_risk: float = 0.8, eta: float = 1.5) -> float:
        """
        计算暴露风险 μ_i^M(τ) - 定义3.1
        μ_i^M(τ) = α_risk · (d_i^M / d_max^M)^η

        Args:
            max_multiplex_degree: 全网最大多重度中心性 d_max^M(τ)
            alpha_risk: 最大风险上限系数 α_risk ∈ (0, 1]
            eta: 风险增长指数 η

        Returns:
            暴露风险值
        """
        if max_multiplex_degree <= 0:
            self.exposure_risk = 0.0
        else:
            normalized = self.multiplex_degree / max_multiplex_degree
            self.exposure_risk = alpha_risk * (normalized ** eta)
        return self.exposure_risk

    def compute_survival_probability(self, network_loads: Dict[int, float]) -> float:
        """
        计算正常工作概率 π_i(τ) - 公式(2-11)
        复用研究点一的计算逻辑

        Args:
            network_loads: 各网络层的负载字典 {layer_id: Q^(ℓ)(τ)}

        Returns:
            存活概率 π_i(τ)
        """
        # 本体负荷影响 Ψ_α - 使用tanh函数
        load_factor = 1.0 - np.tanh(self.load * 0.15)

        # 网络层负荷连带影响 Ψ_β - 使用tanh函数
        layer_factor = 1.0
        for layer_id in self.network_layers:
            if layer_id in network_loads:
                # 归一化：假设平均每层15个智能体
                normalized_load = network_loads[layer_id] / 15.0
                layer_factor *= (1.0 - np.tanh(normalized_load * 0.03))

        # 角色暴露风险 μ^ξ
        role_exposure_factor = 1.0 - self.current_role.exposure_risk

        # 综合存活概率
        survival_prob = load_factor * layer_factor * role_exposure_factor
        return np.clip(survival_prob, 0.0, 1.0)

    def compute_failure_probability(self, network_loads: Dict[int, float]) -> float:
        """
        计算综合失效概率 p_i^fail(τ) - 公式(3-49)
        p_i^fail(τ) = 1 - π_i(τ) · (1 - μ_i^M(τ))

        结合正常工作概率和多重度中心性暴露风险

        Args:
            network_loads: 各网络层的负载字典

        Returns:
            综合失效概率
        """
        survival_prob = self.compute_survival_probability(network_loads)
        self.failure_prob = 1.0 - survival_prob * (1.0 - self.exposure_risk)
        return self.failure_prob

    def compute_risk_potential(self, gamma: float = 2.0, epsilon: float = 1e-6) -> float:
        """
        计算风险势能 U_i(τ) - 定义3.13
        U_i(τ) = 1 / (max(ε, 1 - p_i^fail))^γ

        Args:
            gamma: 风险敏感指数 γ > 0，控制势能场陡峭程度
            epsilon: 数值稳定常数 ε > 0

        Returns:
            风险势能值
        """
        survival = max(epsilon, 1.0 - self.failure_prob)
        self.risk_potential = 1.0 / (survival ** gamma)
        return self.risk_potential

    def monte_carlo_death_check(self, random_value: Optional[float] = None) -> bool:
        """
        蒙特卡洛随机致死判定 - 定义3.2

        设 ε_i(τ) ~ U(0, 1)，若 ε_i(τ) < p_i^fail(τ) 则物理死亡
        物理死亡状态不可逆

        Args:
            random_value: 可选的随机值（用于可重复实验）

        Returns:
            True if node dies, False otherwise
        """
        if self.physical_state == 0:
            # 已死亡，状态不可逆
            return True

        epsilon = random_value if random_value is not None else np.random.uniform(0, 1)

        if epsilon < self.failure_prob:
            self.physical_state = 0
            self.functional_state = 0
            return True
        return False

    def set_isolated(self):
        """
        设置为孤岛节点（物理存活但功能失效）
        对应定义3.3中的级联功能失效
        """
        if self.physical_state == 1:
            self.functional_state = 0

    def recover_from_isolation(self):
        """
        从孤岛状态恢复
        当补位操作恢复连通性后调用
        """
        if self.physical_state == 1:
            self.functional_state = 1

    def compute_role_task_fitness(self, task_requirements: Set[str]) -> float:
        """
        计算角色-任务适配度 Φ(ξ_i, τ_k) - 定义2.1 公式(2-2)
        复用研究点一的计算逻辑

        Args:
            task_requirements: 任务能力需求集合 D_k

        Returns:
            适配度值 ∈ [0, 1]
        """
        if len(task_requirements) == 0:
            return 1.0

        # 能力满足率
        capability_match = len(self.capabilities & task_requirements) / len(task_requirements)

        # 角色状态影响因子
        state_vector = self.get_role_state_vector()
        state_factor = np.mean(state_vector)

        # 综合适配度
        base_fitness = 0.4
        match_fitness = 0.6 * capability_match
        fitness = (base_fitness + match_fitness) * state_factor

        return np.clip(fitness, 0.0, 1.0)

    def compute_switching_cost(self, target_role: Role) -> float:
        """
        计算角色切换代价 c(ξ_i, ρ_i) - 定义2.3 公式(2-4)
        复用研究点一的计算逻辑

        Args:
            target_role: 目标角色 ρ_i

        Returns:
            切换代价
        """
        if target_role == self.current_role:
            return 0.0

        # 角色差异度（能力集合对称差）
        cap_diff = len(self.capabilities ^ target_role.capabilities)
        base_cost = cap_diff * 0.02

        # 考虑当前负载的影响
        load_penalty = self.load * 0.2

        return base_cost + load_penalty

    def switch_role(self, target_role: Role, save_original: bool = False) -> float:
        """
        执行角色切换

        Args:
            target_role: 目标角色
            save_original: 是否保存原始角色（用于补位场景）

        Returns:
            切换代价
        """
        cost = self.compute_switching_cost(target_role)

        if save_original and self.original_role is None:
            self.original_role = self.current_role

        self.current_role = target_role
        return cost

    def reset_state(self):
        """重置智能体状态（用于新一轮仿真）"""
        self.physical_state = 1
        self.functional_state = 1
        self.multiplex_degree = 0.0
        self.exposure_risk = 0.0
        self.failure_prob = 0.0
        self.risk_potential = 1.0
        self.is_replenishment_node = False
        self.replenished_for = None
        self.assigned_tasks = []
        self.load = 0.0

        if self.original_role is not None:
            self.current_role = self.original_role
            self.original_role = None

    def __repr__(self):
        status = "ALIVE" if self.is_alive else "DEAD"
        func = "FUNC" if self.is_functional else "ISOL"
        return f"ResilientAgent(id={self.agent_id}, {status}/{func}, p_fail={self.failure_prob:.3f})"

    def __hash__(self):
        return hash(self.agent_id)

    def __eq__(self, other):
        if not isinstance(other, ResilientAgent):
            return False
        return self.agent_id == other.agent_id
