"""
角色补位机制
实现定义3.4-3.6和算法3-1 GD-RER
基于补位效能比的贪心决策
"""
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from .agent import ResilientAgent, Role
from .network import ResilientMultiLayerNetwork
from .task import TaskSet


class ReplenishmentMechanism:
    """
    角色补位机制
    实现基于风险-代价比的角色补位算法 (GD-RER)
    """

    def __init__(self,
                 alpha_risk: float = 0.8,
                 eta_rer: float = 0.1,
                 kappa_link: float = 0.1,
                 L_max: float = 10.0):
        """
        Args:
            alpha_risk: 风险惩罚系数 α_risk
            eta_rer: 补位效能比阈值 η_RER
            kappa_link: 单位连接建立代价系数 κ_link
            L_max: 最大负载 L_max
        """
        self.alpha_risk = alpha_risk
        self.eta_rer = eta_rer
        self.kappa_link = kappa_link
        self.L_max = L_max

        # 补位方案
        self.replenishment_plan: Dict[int, Tuple[int, Role]] = {}  # {failed_id: (rep_id, rep_role)}
        self.unrecoverable_nodes: Set[int] = set()  # 不可恢复的失效节点
        self.total_replenishment_cost: float = 0.0  # 总补位代价

    def get_target_role_requirements(self,
                                     failed_agent: ResilientAgent,
                                     all_roles: List[Role],
                                     role_to_layers: Dict[int, Set[int]]) -> Set[Role]:
        """
        获取目标角色需求集 Ξ_i^req - 定义3.4

        目标角色需求集定义为所有能够接入失效节点原有网络层的角色集合

        Args:
            failed_agent: 失效节点
            all_roles: 全系统可行角色集 Ξ_all
            role_to_layers: 角色-层级映射 H_Λ

        Returns:
            能够接入失效节点原有网络层的角色集合
        """
        failed_layers = failed_agent.network_layers
        required_roles = set()

        for role in all_roles:
            role_layers = role_to_layers.get(role.role_id, set())
            # 存在交集则满足条件
            if role_layers & failed_layers:
                required_roles.add(role)

        return required_roles

    def check_replenishment_qualification(self,
                                          candidate: ResilientAgent,
                                          failed_agent: ResilientAgent,
                                          required_roles: Set[Role]) -> bool:
        """
        补位资格判定 Q_{j→i}(τ) - 定义3.5

        补位资格条件：
        1. 功能有效性约束：候选者物理存活且位于主连通分量内
        2. 角色可行性约束：候选者的可行角色集与目标角色需求集存在非空交集

        Args:
            candidate: 候选补位节点
            failed_agent: 失效节点
            required_roles: 目标角色需求集

        Returns:
            True if candidate is qualified
        """
        # 功能有效性约束
        if not candidate.is_functional:
            return False

        # 角色可行性约束
        candidate_roles = set(candidate.feasible_roles)
        if not (candidate_roles & required_roles):
            return False

        return True

    def get_candidate_set(self,
                          failed_agent: ResilientAgent,
                          agents: Dict[int, ResilientAgent],
                          required_roles: Set[Role],
                          available_agents: Set[int]) -> Set[int]:
        """
        获取补位候选集 C_i^cand(τ)

        Args:
            failed_agent: 失效节点
            agents: 所有智能体
            required_roles: 目标角色需求集
            available_agents: 可用智能体集合（尚未被选为补位者）

        Returns:
            候选智能体ID集合
        """
        candidates = set()

        for aid in available_agents:
            if aid == failed_agent.agent_id:
                continue

            candidate = agents.get(aid)
            if candidate and self.check_replenishment_qualification(
                candidate, failed_agent, required_roles
            ):
                candidates.add(aid)

        return candidates

    def select_optimal_replenishment_role(self,
                                          candidate: ResilientAgent,
                                          failed_layers: Set[int],
                                          role_to_layers: Dict[int, Set[int]]) -> Optional[Role]:
        """
        为候选节点选择最优补位角色 ρ_j^rep

        选择策略：
        1. 优先选择能够覆盖最多失效层的角色
        2. 其次选择切换代价最低的角色

        Args:
            candidate: 候选补位节点
            failed_layers: 失效节点原有的网络层集合
            role_to_layers: 角色-层级映射

        Returns:
            最优补位角色，若无则返回None
        """
        best_role = None
        max_coverage = 0
        min_switch_cost = float('inf')

        for role in candidate.feasible_roles:
            role_layers = role_to_layers.get(role.role_id, set())
            coverage = len(role_layers & failed_layers)

            if coverage > 0:
                switch_cost = candidate.compute_switching_cost(role)

                # 优先选择覆盖更多层的角色，其次选择切换代价更低的
                if coverage > max_coverage or (coverage == max_coverage and switch_cost < min_switch_cost):
                    max_coverage = coverage
                    min_switch_cost = switch_cost
                    best_role = role

        return best_role

    def compute_replenishment_cost(self,
                                   candidate: ResilientAgent,
                                   failed_agent: ResilientAgent,
                                   target_role: Role,
                                   network: ResilientMultiLayerNetwork) -> float:
        """
        计算补位角色切换代价 c^rep - 定义3.6

        c^rep(a_j, a_i) = c(ξ_j, ρ_j^rep) + κ_link · Σ |N_i^(ℓ) ∩ {存活节点}|

        Args:
            candidate: 候选补位节点
            failed_agent: 失效节点
            target_role: 目标补位角色
            network: 多重网络

        Returns:
            补位代价
        """
        # 基础角色切换代价
        base_cost = candidate.compute_switching_cost(target_role)

        # 连接建立代价
        link_cost = 0.0
        for layer_id in failed_agent.network_layers:
            # 获取失效节点在该层的存活邻居数量
            surviving_neighbors = network.get_surviving_neighbors(
                failed_agent.agent_id, layer_id
            )
            link_cost += len(surviving_neighbors)

        total_cost = base_cost + self.kappa_link * link_cost
        return total_cost

    def compute_replenishment_efficiency_ratio(self,
                                               candidate: ResilientAgent,
                                               isolated_size: int,
                                               rep_cost: float) -> float:
        """
        计算补位效能比 RER_{j→i} - 定义3.12

        RER = [|C_iso| · (1 - L_j/L_max)] / [c^rep · (1 + α_risk · μ_j^M)]

        分子：预期净收益（孤岛恢复规模 × 负载余量）
        分母：风险调整代价（补位代价 × 风险惩罚因子）

        Args:
            candidate: 候选补位节点
            isolated_size: 孤岛区域大小
            rep_cost: 补位代价

        Returns:
            补位效能比
        """
        # 分子：预期净收益
        load_margin = max(0.0, 1.0 - candidate.load / self.L_max)
        benefit = max(1, isolated_size) * load_margin  # 至少为1，确保有收益

        # 分母：风险调整代价
        risk_penalty = 1.0 + self.alpha_risk * candidate.exposure_risk
        adjusted_cost = max(0.01, rep_cost) * risk_penalty  # 避免除零

        rer = benefit / adjusted_cost
        return rer

    def execute_gd_rer(self,
                       failed_agents_sorted: List[Tuple[int, float]],
                       agents: Dict[int, ResilientAgent],
                       network: ResilientMultiLayerNetwork,
                       isolated_components: Dict[int, Set[int]],
                       all_roles: List[Role],
                       role_to_layers: Dict[int, Set[int]]) -> Dict[int, Tuple[int, Role]]:
        """
        算法3-1: 基于补位效能比的贪心决策 (GD-RER)

        按失效影响度降序处理失效节点，为每个失效节点选择最优补位者

        Args:
            failed_agents_sorted: 按FID降序排列的失效节点列表 [(agent_id, FID), ...]
            agents: 所有智能体
            network: 多重网络
            isolated_components: 各失效节点导致的孤岛区域 {failed_id: isolated_set}
            all_roles: 全系统角色集
            role_to_layers: 角色-层级映射

        Returns:
            补位方案 {failed_id: (replenishment_id, replenishment_role)}
        """
        self.replenishment_plan.clear()
        self.unrecoverable_nodes.clear()
        self.total_replenishment_cost = 0.0

        # 可用智能体集合（功能有效且尚未被选为补位者）
        available_agents = {
            aid for aid, agent in agents.items()
            if agent.is_functional
        }

        for failed_id, fid in failed_agents_sorted:
            failed_agent = agents.get(failed_id)
            if not failed_agent:
                continue

            # 获取目标角色需求集
            required_roles = self.get_target_role_requirements(
                failed_agent, all_roles, role_to_layers
            )

            if not required_roles:
                self.unrecoverable_nodes.add(failed_id)
                continue

            # 获取候选集
            candidates = self.get_candidate_set(
                failed_agent, agents, required_roles, available_agents
            )

            if not candidates:
                self.unrecoverable_nodes.add(failed_id)
                continue

            # 计算各候选节点的补位效能比
            best_candidate = None
            best_role = None
            best_rer = -float('inf')
            best_cost = 0.0

            isolated_size = len(isolated_components.get(failed_id, set()))

            for cand_id in candidates:
                candidate = agents[cand_id]

                # 选择最优补位角色
                target_role = self.select_optimal_replenishment_role(
                    candidate, failed_agent.network_layers, role_to_layers
                )

                if target_role is None:
                    continue

                # 计算补位代价
                rep_cost = self.compute_replenishment_cost(
                    candidate, failed_agent, target_role, network
                )

                # 计算补位效能比
                rer = self.compute_replenishment_efficiency_ratio(
                    candidate, isolated_size, rep_cost
                )

                if rer > best_rer:
                    best_rer = rer
                    best_candidate = cand_id
                    best_role = target_role
                    best_cost = rep_cost

            # 检查是否满足效能比阈值
            if best_candidate is not None and best_rer >= self.eta_rer:
                self.replenishment_plan[failed_id] = (best_candidate, best_role)
                self.total_replenishment_cost += best_cost

                # 从可用集合中移除已选补位者
                available_agents.discard(best_candidate)

                # 标记补位节点
                agents[best_candidate].is_replenishment_node = True
                agents[best_candidate].replenished_for = failed_id
            else:
                self.unrecoverable_nodes.add(failed_id)

        return self.replenishment_plan

    def execute_replenishment(self,
                              agents: Dict[int, ResilientAgent],
                              network: ResilientMultiLayerNetwork,
                              role_to_layers: Dict[int, Set[int]]):
        """
        执行补位操作

        对每个补位方案：
        1. 执行角色切换
        2. 重建网络拓扑
        3. 恢复孤岛节点状态

        Args:
            agents: 智能体字典
            network: 多重网络
            role_to_layers: 角色-层级映射
        """
        for failed_id, (rep_id, rep_role) in self.replenishment_plan.items():
            failed_agent = agents.get(failed_id)
            rep_agent = agents.get(rep_id)

            if not failed_agent or not rep_agent:
                continue

            # 执行角色切换
            network.switch_agent_role(rep_id, rep_role, save_original=True)

            # 获取需要接入的网络层
            target_layers = failed_agent.network_layers.copy()

            # 重建网络拓扑
            network.rebuild_topology_after_replenishment(
                rep_id, failed_id, target_layers
            )

        # 更新所有节点的功能状态
        network.update_functional_states()

        # 恢复孤岛节点
        for aid, agent in agents.items():
            if aid in network.main_component and not agent.is_functional:
                agent.recover_from_isolation()

    def get_replenishment_statistics(self) -> Dict:
        """获取补位统计信息"""
        return {
            'num_replenished': len(self.replenishment_plan),
            'num_unrecoverable': len(self.unrecoverable_nodes),
            'total_cost': self.total_replenishment_cost,
            'replenishment_plan': {
                failed_id: (rep_id, role.function_type)
                for failed_id, (rep_id, role) in self.replenishment_plan.items()
            },
            'unrecoverable_nodes': list(self.unrecoverable_nodes)
        }

    def reset(self):
        """重置补位状态"""
        self.replenishment_plan.clear()
        self.unrecoverable_nodes.clear()
        self.total_replenishment_cost = 0.0

    def __repr__(self):
        return f"ReplenishmentMechanism(replenished={len(self.replenishment_plan)}, unrecoverable={len(self.unrecoverable_nodes)})"
