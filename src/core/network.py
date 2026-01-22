"""
韧性多重网络模型模块
实现面向节点失效的多重网络架构，包含失效传播、连通性分析等功能
基于研究点一的MultiLayerNetwork类进行扩展
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from .agent import ResilientAgent, Role


class NetworkLayer:
    """
    网络层类
    与研究点一相同，管理单个网络层的拓扑结构
    """

    def __init__(self, layer_id: int, layer_type: str):
        """
        Args:
            layer_id: 网络层编号 ℓ
            layer_type: 网络层类型 (如'communication', 'sensing', 'fire_coordination')
        """
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.graph = nx.Graph()  # 网络层图 G^(ℓ)(τ)
        self.agents: Set[int] = set()  # 该层智能体集合 A^(ℓ)(τ)
        self.total_load = 0.0  # 网络层总负载 Q^(ℓ)(τ)

    def add_agent(self, agent: ResilientAgent, connection_prob: float = 0.4):
        """
        向网络层添加智能体，并与现有智能体建立连接

        Args:
            agent: 智能体对象
            connection_prob: 与现有智能体建立连接的概率
        """
        self.agents.add(agent.agent_id)

        # 如果节点不存在则添加
        is_new_node = not self.graph.has_node(agent.agent_id)
        if is_new_node:
            self.graph.add_node(agent.agent_id)

        # 与该层现有智能体建立随机连接
        existing_agents = [aid for aid in self.agents if aid != agent.agent_id]
        for other_id in existing_agents:
            if (self.graph.has_node(other_id) and
                not self.graph.has_edge(agent.agent_id, other_id) and
                    np.random.random() < connection_prob):
                weight = np.random.uniform(0.5, 2.0)
                self.graph.add_edge(agent.agent_id, other_id, weight=weight)

    def remove_agent(self, agent_id: int):
        """从网络层移除智能体"""
        self.agents.discard(agent_id)
        if self.graph.has_node(agent_id):
            self.graph.remove_node(agent_id)

    def add_edge(self, agent_i: int, agent_j: int, weight: float = 1.0):
        """添加边 b_{ij}^(ℓ)(τ)"""
        if agent_i in self.agents and agent_j in self.agents:
            self.graph.add_edge(agent_i, agent_j, weight=weight)

    def get_degree_centrality(self, agent_id: int) -> float:
        """计算智能体在该层的度中心性 d_i^ℓ"""
        if agent_id not in self.graph:
            return 0.0
        return self.graph.degree(agent_id)

    def get_neighbors(self, agent_id: int) -> Set[int]:
        """获取智能体在该层的邻居集合"""
        if agent_id not in self.graph:
            return set()
        return set(self.graph.neighbors(agent_id))

    def update_load(self, agent_loads: Dict[int, float]):
        """更新网络层总负载"""
        self.total_load = sum(agent_loads.get(aid, 0.0) for aid in self.agents)

    def is_connected(self) -> bool:
        """检查网络层是否连通"""
        if len(self.agents) == 0:
            return True
        return nx.is_connected(self.graph)

    def get_connected_components(self) -> List[Set[int]]:
        """获取所有连通分量"""
        return [set(c) for c in nx.connected_components(self.graph)]

    def __repr__(self):
        return f"Layer(id={self.layer_id}, type={self.layer_type}, agents={len(self.agents)})"


class ResilientMultiLayerNetwork:
    """
    韧性多重网络类
    在研究点一MultiLayerNetwork基础上扩展失效传播和连通性分析功能
    """

    def __init__(self, num_layers: int = 3):
        """
        Args:
            num_layers: 初始网络层数量 L
        """
        self.layers: Dict[int, NetworkLayer] = {}  # 网络层字典
        self.agents: Dict[int, ResilientAgent] = {}  # 智能体字典
        self.migration_cost_matrix: Dict[Tuple[int, int], float] = {}  # 迁移代价矩阵

        # 角色-层级映射函数 H_Λ (定义2.4)
        self.role_to_layers: Dict[int, Set[int]] = {}

        # 失效相关状态
        self.failed_agents: Set[int] = set()  # 物理失效节点集合
        self.isolated_agents: Set[int] = set()  # 孤岛节点集合
        self.main_component: Set[int] = set()  # 主连通分量

        # 保存失效前的邻居信息（用于补位）
        self.pre_failure_neighbors: Dict[int, Dict[int, Set[int]]] = {}  # {agent_id: {layer_id: neighbors}}

    def add_layer(self, layer: NetworkLayer):
        """添加网络层"""
        self.layers[layer.layer_id] = layer

    def add_agent(self, agent: ResilientAgent):
        """添加智能体并根据角色确定层级隶属"""
        self.agents[agent.agent_id] = agent
        self._update_agent_layers(agent)

    def _update_agent_layers(self, agent: ResilientAgent):
        """
        根据角色更新智能体的网络层隶属
        实现角色-结构耦合 Λ_i(τ) = H_Λ(ρ_i) - 公式(2-6)
        """
        # 获取该角色对应的网络层集合
        target_layers = self.role_to_layers.get(agent.current_role.role_id, set())

        # 计算隶属变更
        current_layers = agent.network_layers
        layers_to_add = target_layers - current_layers
        layers_to_remove = current_layers - target_layers

        # 执行层级变更
        for layer_id in layers_to_remove:
            if layer_id in self.layers:
                self.layers[layer_id].remove_agent(agent.agent_id)

        for layer_id in layers_to_add:
            if layer_id in self.layers:
                self.layers[layer_id].add_agent(agent)

        # 更新智能体的网络层隶属集合
        agent.network_layers = target_layers.copy()

    def set_role_layer_mapping(self, role_id: int, layer_ids: Set[int]):
        """
        设置角色到网络层的映射关系 H_Λ

        Args:
            role_id: 角色ID
            layer_ids: 该角色应隶属的网络层ID集合
        """
        self.role_to_layers[role_id] = layer_ids

    def switch_agent_role(self, agent_id: int, target_role: Role,
                          save_original: bool = False) -> Tuple[float, Set[int], Set[int]]:
        """
        执行智能体角色切换并触发网络拓扑重组

        Args:
            agent_id: 智能体ID
            target_role: 目标角色
            save_original: 是否保存原始角色

        Returns:
            (切换代价, 新接入的网络层, 脱离的网络层)
        """
        agent = self.agents[agent_id]
        old_layers = agent.network_layers.copy()

        # 执行角色切换
        switch_cost = agent.switch_role(target_role, save_original)

        # 更新网络层隶属（触发拓扑重组）
        self._update_agent_layers(agent)
        new_layers = agent.network_layers

        layers_added = new_layers - old_layers
        layers_removed = old_layers - new_layers

        # 清除迁移代价缓存
        self.clear_migration_cache()

        return switch_cost, layers_added, layers_removed

    def build_layer_topology(self, layer_id: int, connection_prob: float = 0.3):
        """
        构建网络层内拓扑

        Args:
            layer_id: 网络层ID
            connection_prob: 连接概率
        """
        layer = self.layers[layer_id]
        agents_in_layer = list(layer.agents)

        # 构建随机图
        for i, aid_i in enumerate(agents_in_layer):
            for aid_j in agents_in_layer[i + 1:]:
                if np.random.random() < connection_prob:
                    weight = np.random.uniform(0.5, 2.0)
                    layer.add_edge(aid_i, aid_j, weight)

    # ==================== 失效相关方法 ====================

    def save_pre_failure_state(self):
        """
        保存失效前的网络状态
        用于补位时恢复连接
        """
        self.pre_failure_neighbors.clear()
        for aid in self.agents:
            self.pre_failure_neighbors[aid] = {}
            for lid in self.agents[aid].network_layers:
                if lid in self.layers:
                    self.pre_failure_neighbors[aid][lid] = self.layers[lid].get_neighbors(aid)

    def remove_failed_agent(self, agent_id: int):
        """
        从网络中移除失效节点

        Args:
            agent_id: 失效节点ID
        """
        if agent_id not in self.agents:
            return

        agent = self.agents[agent_id]
        self.failed_agents.add(agent_id)

        # 从所有网络层中移除
        for layer_id in list(agent.network_layers):
            if layer_id in self.layers:
                self.layers[layer_id].remove_agent(agent_id)

        # 清除迁移代价缓存
        self.clear_migration_cache()

    def get_main_connected_component(self) -> Set[int]:
        """
        获取主连通分量 C_main(τ)
        定义为包含最多功能有效节点的连通分量

        Returns:
            主连通分量中的节点ID集合
        """
        # 构建聚合图（所有层的并集）
        aggregate_graph = nx.Graph()

        # 添加所有功能有效的节点
        for aid, agent in self.agents.items():
            if agent.is_alive and aid not in self.failed_agents:
                aggregate_graph.add_node(aid)

        # 添加所有层的边
        for layer in self.layers.values():
            for u, v in layer.graph.edges():
                if aggregate_graph.has_node(u) and aggregate_graph.has_node(v):
                    aggregate_graph.add_edge(u, v)

        # 找到最大连通分量
        if len(aggregate_graph.nodes()) == 0:
            self.main_component = set()
            return self.main_component

        components = list(nx.connected_components(aggregate_graph))
        if components:
            self.main_component = max(components, key=len)
        else:
            self.main_component = set()

        return self.main_component

    def get_isolated_component_by_failure(self, failed_agent_id: int) -> Set[int]:
        """
        获取因某节点失效导致的孤岛区域 C_iso^(i)(τ)

        Args:
            failed_agent_id: 失效节点ID

        Returns:
            孤岛区域中的节点ID集合
        """
        # 获取失效前该节点的邻居
        pre_neighbors = self.pre_failure_neighbors.get(failed_agent_id, {})

        isolated = set()
        for layer_id, neighbors in pre_neighbors.items():
            for neighbor_id in neighbors:
                if neighbor_id in self.agents:
                    agent = self.agents[neighbor_id]
                    if agent.is_alive and neighbor_id not in self.main_component:
                        isolated.add(neighbor_id)

        return isolated

    def get_surviving_neighbors(self, agent_id: int, layer_id: int) -> Set[int]:
        """
        获取节点在指定层的存活邻居

        Args:
            agent_id: 节点ID
            layer_id: 网络层ID

        Returns:
            存活邻居ID集合
        """
        # 优先使用失效前保存的邻居信息
        if agent_id in self.pre_failure_neighbors:
            pre_neighbors = self.pre_failure_neighbors[agent_id].get(layer_id, set())
            return {n for n in pre_neighbors
                    if n in self.agents and self.agents[n].is_alive}

        # 否则使用当前网络状态
        if layer_id not in self.layers:
            return set()

        layer = self.layers[layer_id]
        if agent_id not in layer.graph:
            return set()

        neighbors = set(layer.graph.neighbors(agent_id))
        return {n for n in neighbors
                if n in self.agents and self.agents[n].is_alive}

    def rebuild_topology_after_replenishment(self, replenishment_agent_id: int,
                                             failed_agent_id: int,
                                             target_layers: Set[int],
                                             connection_prob: float = 0.6):
        """
        补位后重建网络拓扑
        补位节点接替失效节点的跨层桥接功能

        Args:
            replenishment_agent_id: 补位节点ID
            failed_agent_id: 失效节点ID
            target_layers: 需要接入的网络层集合
            connection_prob: 与原邻居建立连接的概率
        """
        rep_agent = self.agents.get(replenishment_agent_id)
        if not rep_agent:
            return

        # 获取失效节点在各层的原邻居
        pre_neighbors = self.pre_failure_neighbors.get(failed_agent_id, {})

        for layer_id in target_layers:
            if layer_id not in self.layers:
                continue

            layer = self.layers[layer_id]

            # 确保补位节点在该层
            if replenishment_agent_id not in layer.agents:
                layer.agents.add(replenishment_agent_id)
                if not layer.graph.has_node(replenishment_agent_id):
                    layer.graph.add_node(replenishment_agent_id)

            # 与失效节点的原存活邻居建立连接
            original_neighbors = pre_neighbors.get(layer_id, set())
            for neighbor_id in original_neighbors:
                if (neighbor_id in self.agents and
                    self.agents[neighbor_id].is_alive and
                    neighbor_id in layer.agents and
                        np.random.random() < connection_prob):
                    weight = np.random.uniform(0.5, 2.0)
                    layer.add_edge(replenishment_agent_id, neighbor_id, weight)

        # 更新补位节点的网络层隶属
        rep_agent.network_layers = rep_agent.network_layers | target_layers

        # 清除迁移代价缓存
        self.clear_migration_cache()

    def update_functional_states(self):
        """
        更新所有节点的功能状态
        基于主连通分量判定孤岛节点
        """
        self.get_main_connected_component()
        self.isolated_agents.clear()

        for aid, agent in self.agents.items():
            if agent.is_alive:
                if aid in self.main_component:
                    agent.functional_state = 1
                else:
                    agent.functional_state = 0
                    self.isolated_agents.add(aid)

    # ==================== 迁移代价计算 ====================

    def compute_migration_cost(self, agent_i: int, agent_j: int) -> float:
        """
        计算任务迁移代价 ω_{ij}(τ)
        考虑多重网络的最短路径

        Args:
            agent_i: 源智能体
            agent_j: 目标智能体

        Returns:
            迁移代价 (无穷大表示不连通)
        """
        if agent_i == agent_j:
            return 0.0

        # 检查缓存
        if (agent_i, agent_j) in self.migration_cost_matrix:
            return self.migration_cost_matrix[(agent_i, agent_j)]

        # 检查节点是否功能有效
        ai = self.agents.get(agent_i)
        aj = self.agents.get(agent_j)

        if not ai or not aj or not ai.is_functional or not aj.is_functional:
            return float('inf')

        # 计算所有层的最短路径，取最小值
        min_cost = float('inf')
        common_layers = ai.network_layers & aj.network_layers

        for layer_id in common_layers:
            layer = self.layers[layer_id]
            if layer.graph.has_node(agent_i) and layer.graph.has_node(agent_j):
                try:
                    path_length = nx.shortest_path_length(
                        layer.graph, agent_i, agent_j, weight='weight'
                    )
                    min_cost = min(min_cost, path_length)
                except nx.NetworkXNoPath:
                    continue

        # 缓存结果
        self.migration_cost_matrix[(agent_i, agent_j)] = min_cost
        return min_cost

    def compute_network_distance(self, agent_i: int, agent_j: int) -> float:
        """
        计算多重网络中两智能体间的跳数距离

        Returns:
            最短跳数距离，如果不连通返回inf
        """
        if agent_i == agent_j:
            return 0.0

        ai = self.agents.get(agent_i)
        aj = self.agents.get(agent_j)

        if not ai or not aj:
            return float('inf')

        common_layers = ai.network_layers & aj.network_layers

        min_dist = float('inf')
        for layer_id in common_layers:
            layer = self.layers.get(layer_id)
            if layer and layer.graph.has_node(agent_i) and layer.graph.has_node(agent_j):
                try:
                    dist = nx.shortest_path_length(layer.graph, agent_i, agent_j)
                    min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    continue

        return min_dist

    def get_shortest_path(self, agent_i: int, agent_j: int) -> Optional[List[int]]:
        """
        获取两节点间的最短路径

        Returns:
            路径节点列表，如果不连通返回None
        """
        if agent_i == agent_j:
            return [agent_i]

        ai = self.agents.get(agent_i)
        aj = self.agents.get(agent_j)

        if not ai or not aj:
            return None

        common_layers = ai.network_layers & aj.network_layers

        best_path = None
        min_length = float('inf')

        for layer_id in common_layers:
            layer = self.layers.get(layer_id)
            if layer and layer.graph.has_node(agent_i) and layer.graph.has_node(agent_j):
                try:
                    path = nx.shortest_path(layer.graph, agent_i, agent_j)
                    if len(path) < min_length:
                        min_length = len(path)
                        best_path = path
                except nx.NetworkXNoPath:
                    continue

        return best_path

    # ==================== 辅助方法 ====================

    def get_cross_layer_agents(self) -> List[int]:
        """
        获取跨层桥接智能体
        返回隶属于多个网络层的智能体ID列表
        """
        cross_layer = []
        for aid, agent in self.agents.items():
            if agent.is_functional and len(agent.network_layers) > 1:
                cross_layer.append(aid)
        return cross_layer

    def get_layer_loads(self) -> Dict[int, float]:
        """获取各网络层的负载字典"""
        agent_loads = {aid: agent.load for aid, agent in self.agents.items()}

        loads = {}
        for layer_id, layer in self.layers.items():
            layer.update_load(agent_loads)
            loads[layer_id] = layer.total_load

        return loads

    def get_functional_agents(self) -> Dict[int, ResilientAgent]:
        """获取所有功能有效的智能体"""
        return {aid: agent for aid, agent in self.agents.items()
                if agent.is_functional}

    def clear_migration_cache(self):
        """清除迁移代价缓存（拓扑变化后需要调用）"""
        self.migration_cost_matrix.clear()

    def reset_failure_state(self):
        """重置失效状态（用于新一轮仿真）"""
        self.failed_agents.clear()
        self.isolated_agents.clear()
        self.main_component.clear()
        self.pre_failure_neighbors.clear()
        self.clear_migration_cache()

        for agent in self.agents.values():
            agent.reset_state()

    def __repr__(self):
        functional = sum(1 for a in self.agents.values() if a.is_functional)
        return f"ResilientMultiLayerNetwork(layers={len(self.layers)}, agents={len(self.agents)}, functional={functional})"
