"""
风险势能场模型
实现定义3.13-3.14和算法3-2 RPF-PP
基于风险势能场的路径规划
"""
import numpy as np
import networkx as nx
from typing import Dict, Set, List, Tuple, Optional
from .agent import ResilientAgent
from .network import ResilientMultiLayerNetwork


class RiskPotentialField:
    """
    风险势能场模型
    实现基于风险势能场的路径规划算法 (RPF-PP)
    """

    def __init__(self,
                 gamma: float = 2.0,
                 epsilon: float = 1e-6):
        """
        Args:
            gamma: 风险敏感指数 γ > 0，控制势能场陡峭程度
            epsilon: 数值稳定常数 ε > 0
        """
        self.gamma = gamma
        self.epsilon = epsilon

        # 风险势能场数据
        self.risk_potentials: Dict[int, float] = {}  # {agent_id: U_i}
        self.rpd_matrix: Dict[Tuple[int, int], float] = {}  # 风险势能距离矩阵
        self.rpd_paths: Dict[Tuple[int, int], List[int]] = {}  # 最短路径记录

    def build_risk_potential_field(self,
                                   agents: Dict[int, ResilientAgent]):
        """
        构建风险势能场

        计算每个存活节点的风险势能 U_i(τ) - 定义3.13
        U_i(τ) = 1 / (max(ε, 1 - p_i^fail))^γ

        Args:
            agents: 智能体字典
        """
        self.risk_potentials.clear()

        for aid, agent in agents.items():
            if agent.is_functional:
                # 计算风险势能
                survival = max(self.epsilon, 1.0 - agent.failure_prob)
                potential = 1.0 / (survival ** self.gamma)
                self.risk_potentials[aid] = potential
                agent.risk_potential = potential

    def compute_edge_rpd_weight(self,
                                source_id: int,
                                target_id: int,
                                original_weight: float,
                                agents: Dict[int, ResilientAgent]) -> float:
        """
        计算边风险势能权重 w_ij^RPD - 公式(3-13)
        w_ij^RPD(τ) = ω_ij(τ) · U_j(τ)

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            original_weight: 原始迁移代价 ω_ij
            agents: 智能体字典

        Returns:
            风险势能权重
        """
        target_agent = agents.get(target_id)
        if not target_agent or not target_agent.is_functional:
            return float('inf')

        target_potential = self.risk_potentials.get(target_id, 1.0)
        return original_weight * target_potential

    def compute_rpd_matrix(self,
                           agents: Dict[int, ResilientAgent],
                           network: ResilientMultiLayerNetwork,
                           source_agents: Optional[Set[int]] = None) -> Dict[Tuple[int, int], float]:
        """
        计算风险势能距离矩阵 - 定义3.14 和 算法3-2 RPF-PP

        RPD(u, v, τ) = min_{path ∈ P(u,v)} Σ w_ij^RPD(τ)

        Args:
            agents: 智能体字典
            network: 多重网络
            source_agents: 源节点集合（默认为所有功能有效节点）

        Returns:
            风险势能距离矩阵 {(source, target): RPD}
        """
        self.rpd_matrix.clear()
        self.rpd_paths.clear()

        # 构建风险势能场
        self.build_risk_potential_field(agents)

        # 获取功能有效的节点
        functional_agents = {aid for aid, agent in agents.items() if agent.is_functional}

        if source_agents is None:
            source_agents = functional_agents

        # 构建带风险势能权重的聚合图
        rpd_graph = self._build_rpd_graph(agents, network)

        # 对每个源节点计算到所有目标节点的RPD
        for source_id in source_agents:
            if source_id not in rpd_graph:
                continue

            try:
                # 使用Dijkstra算法计算最短路径
                distances, paths = nx.single_source_dijkstra(
                    rpd_graph, source_id, weight='rpd_weight'
                )

                for target_id, dist in distances.items():
                    if target_id in functional_agents:
                        self.rpd_matrix[(source_id, target_id)] = dist
                        if target_id in paths:
                            self.rpd_paths[(source_id, target_id)] = paths[target_id]

            except nx.NetworkXError:
                # 源节点不可达
                continue

        return self.rpd_matrix

    def _build_rpd_graph(self,
                         agents: Dict[int, ResilientAgent],
                         network: ResilientMultiLayerNetwork) -> nx.Graph:
        """
        构建带风险势能权重的聚合图

        Args:
            agents: 智能体字典
            network: 多重网络

        Returns:
            带RPD权重的NetworkX图
        """
        rpd_graph = nx.Graph()

        # 添加功能有效的节点
        for aid, agent in agents.items():
            if agent.is_functional:
                rpd_graph.add_node(aid, risk_potential=self.risk_potentials.get(aid, 1.0))

        # 添加所有层的边，使用RPD权重
        for layer in network.layers.values():
            for u, v, data in layer.graph.edges(data=True):
                if rpd_graph.has_node(u) and rpd_graph.has_node(v):
                    original_weight = data.get('weight', 1.0)

                    # 计算RPD权重（使用目标节点的风险势能）
                    # 对于无向图，取两个方向的平均
                    rpd_weight_uv = self.compute_edge_rpd_weight(u, v, original_weight, agents)
                    rpd_weight_vu = self.compute_edge_rpd_weight(v, u, original_weight, agents)
                    avg_rpd_weight = (rpd_weight_uv + rpd_weight_vu) / 2

                    # 如果边已存在，取较小的权重
                    if rpd_graph.has_edge(u, v):
                        existing_weight = rpd_graph[u][v]['rpd_weight']
                        if avg_rpd_weight < existing_weight:
                            rpd_graph[u][v]['rpd_weight'] = avg_rpd_weight
                    else:
                        rpd_graph.add_edge(u, v, rpd_weight=avg_rpd_weight, original_weight=original_weight)

        return rpd_graph

    def get_rpd(self, source_id: int, target_id: int) -> float:
        """
        获取两节点间的风险势能距离

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID

        Returns:
            风险势能距离，不可达返回inf
        """
        if source_id == target_id:
            return 0.0
        return self.rpd_matrix.get((source_id, target_id), float('inf'))

    def get_rpd_path(self, source_id: int, target_id: int) -> Optional[List[int]]:
        """
        获取两节点间的最短RPD路径

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID

        Returns:
            路径节点列表，不可达返回None
        """
        if source_id == target_id:
            return [source_id]
        return self.rpd_paths.get((source_id, target_id))

    def get_max_rpd(self) -> float:
        """获取最大RPD值（用于归一化）"""
        if not self.rpd_matrix:
            return 1.0
        finite_values = [v for v in self.rpd_matrix.values() if v < float('inf')]
        return max(finite_values) if finite_values else 1.0

    def compute_path_reliability(self,
                                 path: List[int],
                                 agents: Dict[int, ResilientAgent]) -> float:
        """
        计算路径连通可靠性 R_path

        路径可靠性 = 路径上所有中间节点的联合存活概率

        Args:
            path: 路径节点列表
            agents: 智能体字典

        Returns:
            路径可靠性 ∈ [0, 1]
        """
        if not path or len(path) <= 1:
            return 1.0

        reliability = 1.0
        # 中间节点（不包括源和目标）
        for node_id in path[1:-1]:
            agent = agents.get(node_id)
            if agent and agent.is_functional:
                reliability *= (1.0 - agent.failure_prob)
            else:
                return 0.0  # 路径不可达

        return reliability

    def get_path_reliabilities(self,
                               task_locations: Dict[int, int],
                               task_assignments: Dict[int, int],
                               agents: Dict[int, ResilientAgent]) -> Dict[int, float]:
        """
        计算所有任务迁移路径的可靠性

        Args:
            task_locations: 任务当前位置 {task_id: agent_id}
            task_assignments: 任务分配方案 {task_id: target_agent_id}
            agents: 智能体字典

        Returns:
            {task_id: path_reliability}
        """
        reliabilities = {}

        for task_id, target_id in task_assignments.items():
            source_id = task_locations.get(task_id)

            if source_id is None or source_id == target_id:
                reliabilities[task_id] = 1.0
                continue

            path = self.get_rpd_path(source_id, target_id)
            if path:
                reliabilities[task_id] = self.compute_path_reliability(path, agents)
            else:
                reliabilities[task_id] = 0.0

        return reliabilities

    def execute_rpf_pp(self,
                       agents: Dict[int, ResilientAgent],
                       network: ResilientMultiLayerNetwork,
                       source_agents: Optional[Set[int]] = None) -> Dict[Tuple[int, int], float]:
        """
        算法3-2: 风险势能场路径规划 (RPF-PP)

        完整执行流程：
        1. 构建风险势能场
        2. 构建RPD权重图
        3. 计算RPD矩阵

        Args:
            agents: 智能体字典
            network: 多重网络
            source_agents: 源节点集合

        Returns:
            风险势能距离矩阵
        """
        return self.compute_rpd_matrix(agents, network, source_agents)

    def get_statistics(self) -> Dict:
        """获取风险势能场统计信息"""
        if not self.risk_potentials:
            return {
                'num_nodes': 0,
                'avg_potential': 0.0,
                'max_potential': 0.0,
                'min_potential': 0.0,
                'num_rpd_pairs': 0,
                'max_rpd': 0.0
            }

        potentials = list(self.risk_potentials.values())
        finite_rpds = [v for v in self.rpd_matrix.values() if v < float('inf')]

        return {
            'num_nodes': len(self.risk_potentials),
            'avg_potential': np.mean(potentials),
            'max_potential': max(potentials),
            'min_potential': min(potentials),
            'num_rpd_pairs': len(self.rpd_matrix),
            'max_rpd': max(finite_rpds) if finite_rpds else 0.0,
            'avg_rpd': np.mean(finite_rpds) if finite_rpds else 0.0
        }

    def reset(self):
        """重置风险势能场"""
        self.risk_potentials.clear()
        self.rpd_matrix.clear()
        self.rpd_paths.clear()

    def __repr__(self):
        return f"RiskPotentialField(nodes={len(self.risk_potentials)}, rpd_pairs={len(self.rpd_matrix)})"
