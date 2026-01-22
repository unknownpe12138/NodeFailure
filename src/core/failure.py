"""
节点失效与级联传播模型
实现定义3.1-3.3的失效机制，包括：
- 多重度中心性暴露风险
- 蒙特卡洛随机致死机制
- 级联功能失效
"""
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from .agent import ResilientAgent
from .network import ResilientMultiLayerNetwork
from .task import TaskSet


class FailureModel:
    """
    节点失效模型
    管理节点的失效判定、级联传播和孤岛识别
    实现定义3.1-3.3和定义3.11
    """

    def __init__(self,
                 alpha_risk: float = 0.8,
                 eta: float = 1.5,
                 gamma: float = 2.0,
                 kappa_task: float = 0.5,
                 layer_weights: Optional[Dict[int, float]] = None):
        """
        Args:
            alpha_risk: 最大风险上限系数 α_risk ∈ (0, 1]
            eta: 风险增长指数 η
            gamma: 风险敏感指数 γ
            kappa_task: 任务影响权重系数 κ_task
            layer_weights: 各层暴露敏感度权重 β^(ℓ)
        """
        self.alpha_risk = alpha_risk
        self.eta = eta
        self.gamma = gamma
        self.kappa_task = kappa_task
        self.layer_weights = layer_weights or {}

        # 失效记录
        self.failed_agents: Set[int] = set()           # 物理失效节点 A^fail(τ)
        self.isolated_agents: Set[int] = set()         # 孤岛节点
        self.interrupted_tasks: Set[int] = set()       # 中断任务 T_i^int
        self.isolated_components: Dict[int, Set[int]] = {}  # {失效节点ID: 孤岛区域}
        self.failure_impact_degrees: Dict[int, float] = {}  # {失效节点ID: FID}

    def update_all_failure_probabilities(self,
                                         agents: Dict[int, ResilientAgent],
                                         network: ResilientMultiLayerNetwork):
        """
        更新所有节点的失效概率

        步骤:
        1. 计算各节点在各层的度中心性
        2. 计算多重度中心性 d_i^M(τ)
        3. 计算暴露风险 μ_i^M(τ)
        4. 计算综合失效概率 p_i^fail(τ)
        5. 计算风险势能 U_i(τ)

        Args:
            agents: 智能体字典
            network: 多重网络
        """
        # 获取网络层负载
        layer_loads = network.get_layer_loads()

        # 计算各节点在各层的度中心性
        layer_degrees: Dict[int, Dict[int, float]] = {}
        for aid in agents:
            layer_degrees[aid] = {}
            agent = agents[aid]
            if agent.is_alive:
                for lid in agent.network_layers:
                    if lid in network.layers:
                        layer_degrees[aid][lid] = network.layers[lid].get_degree_centrality(aid)

        # 计算多重度中心性
        for aid, agent in agents.items():
            if agent.is_alive:
                agent.compute_multiplex_degree_centrality(
                    layer_degrees.get(aid, {}),
                    self.layer_weights
                )

        # 获取最大多重度中心性
        max_multiplex = max(
            (a.multiplex_degree for a in agents.values() if a.is_alive),
            default=1.0
        )
        if max_multiplex <= 0:
            max_multiplex = 1.0

        # 计算暴露风险、综合失效概率和风险势能
        for aid, agent in agents.items():
            if agent.is_alive:
                agent.compute_exposure_risk(max_multiplex, self.alpha_risk, self.eta)
                agent.compute_failure_probability(layer_loads)
                agent.compute_risk_potential(self.gamma)

    def execute_monte_carlo_death(self,
                                  agents: Dict[int, ResilientAgent],
                                  random_seed: Optional[int] = None) -> Set[int]:
        """
        执行蒙特卡洛随机致死判定 - 定义3.2

        对每个存活节点，生成随机数 ε_i ~ U(0,1)
        若 ε_i < p_i^fail 则判定为物理死亡

        Args:
            agents: 智能体字典
            random_seed: 随机种子（用于可重复实验）

        Returns:
            本轮新死亡的节点ID集合
        """
        newly_failed = set()

        if random_seed is not None:
            np.random.seed(random_seed)

        for aid, agent in agents.items():
            if agent.is_alive:
                # 生成随机数
                epsilon = np.random.uniform(0, 1)

                if agent.monte_carlo_death_check(epsilon):
                    newly_failed.add(aid)
                    self.failed_agents.add(aid)

        return newly_failed

    def execute_deterministic_failure(self,
                                      agents: Dict[int, ResilientAgent],
                                      failure_threshold: float = 0.5) -> Set[int]:
        """
        执行确定性失效判定（用于测试）

        当 p_i^fail > threshold 时判定为失效

        Args:
            agents: 智能体字典
            failure_threshold: 失效阈值

        Returns:
            新失效的节点ID集合
        """
        newly_failed = set()

        for aid, agent in agents.items():
            if agent.is_alive and agent.failure_prob > failure_threshold:
                agent.physical_state = 0
                agent.functional_state = 0
                newly_failed.add(aid)
                self.failed_agents.add(aid)

        return newly_failed

    def identify_cascade_failures(self,
                                  agents: Dict[int, ResilientAgent],
                                  network: ResilientMultiLayerNetwork) -> Tuple[Set[int], Dict[int, Set[int]]]:
        """
        识别级联功能失效 - 定义3.3

        物理节点死亡后，更新网络拓扑，识别孤岛节点
        孤岛节点：物理存活但不在主连通分量内

        Args:
            agents: 智能体字典
            network: 多重网络

        Returns:
            (孤岛节点集合, {失效节点ID: 其导致的孤岛区域节点集合})
        """
        # 保存失效前的网络状态
        network.save_pre_failure_state()

        # 从网络中移除物理失效节点
        for aid in self.failed_agents:
            network.remove_failed_agent(aid)

        # 获取主连通分量
        main_component = network.get_main_connected_component()

        # 识别孤岛节点
        self.isolated_agents.clear()
        self.isolated_components.clear()

        for aid, agent in agents.items():
            if agent.is_alive and aid not in main_component:
                agent.set_isolated()
                self.isolated_agents.add(aid)

        # 为每个失效节点计算其导致的孤岛区域
        for failed_id in self.failed_agents:
            isolated_by_this = network.get_isolated_component_by_failure(failed_id)
            if isolated_by_this:
                self.isolated_components[failed_id] = isolated_by_this

        return self.isolated_agents, self.isolated_components

    def identify_interrupted_tasks(self,
                                   agents: Dict[int, ResilientAgent],
                                   tasks: TaskSet) -> Set[int]:
        """
        识别中断任务

        任务中断条件：
        1. 执行节点物理死亡
        2. 执行节点功能失效（孤岛）

        Args:
            agents: 智能体字典
            tasks: 任务集合

        Returns:
            中断任务ID集合
        """
        self.interrupted_tasks.clear()

        for task in tasks.get_all_tasks():
            current_agent_id = task.current_agent or task.assigned_agent
            if current_agent_id is not None:
                agent = agents.get(current_agent_id)
                if agent and not agent.is_functional:
                    task.mark_interrupted()
                    self.interrupted_tasks.add(task.task_id)

        return self.interrupted_tasks

    def compute_failure_impact_degree(self,
                                      failed_agent_id: int,
                                      agents: Dict[int, ResilientAgent],
                                      tasks: TaskSet) -> float:
        """
        计算失效影响度 FID_i - 定义3.11
        FID_i = |C_iso^(i)| · (1 + κ_task · |T_i^int|/|T|)

        Args:
            failed_agent_id: 失效节点ID
            agents: 智能体字典
            tasks: 任务集合

        Returns:
            失效影响度
        """
        # 孤岛区域大小
        isolated_size = len(self.isolated_components.get(failed_agent_id, set()))

        # 计算因该节点失效而中断的任务数量
        # 包括该节点承载的任务和孤岛区域内节点承载的任务
        interrupted_count = 0

        # 该节点承载的任务
        for task in tasks.get_all_tasks():
            if task.current_agent == failed_agent_id or task.assigned_agent == failed_agent_id:
                interrupted_count += 1

        # 孤岛区域内节点承载的任务
        isolated_nodes = self.isolated_components.get(failed_agent_id, set())
        for task in tasks.get_all_tasks():
            agent_id = task.current_agent or task.assigned_agent
            if agent_id in isolated_nodes:
                interrupted_count += 1

        # 计算FID
        total_tasks = len(tasks)
        if total_tasks == 0:
            task_ratio = 0.0
        else:
            task_ratio = interrupted_count / total_tasks

        fid = isolated_size * (1.0 + self.kappa_task * task_ratio)

        # 如果孤岛区域为空但有中断任务，仍给予一定影响度
        if isolated_size == 0 and interrupted_count > 0:
            fid = 1.0 + self.kappa_task * task_ratio

        self.failure_impact_degrees[failed_agent_id] = fid
        return fid

    def get_failed_agents_sorted_by_impact(self,
                                           agents: Dict[int, ResilientAgent],
                                           tasks: TaskSet) -> List[Tuple[int, float]]:
        """
        按失效影响度降序排列失效节点

        Args:
            agents: 智能体字典
            tasks: 任务集合

        Returns:
            [(agent_id, FID), ...] 按FID降序排列
        """
        impact_scores = []

        for aid in self.failed_agents:
            fid = self.compute_failure_impact_degree(aid, agents, tasks)
            impact_scores.append((aid, fid))

        impact_scores.sort(key=lambda x: x[1], reverse=True)
        return impact_scores

    def get_migration_task_set(self, tasks: TaskSet) -> List[int]:
        """
        获取待迁移任务集合 T_mig(τ)

        包括：
        1. 中断任务
        2. 失效节点的待处理任务

        Args:
            tasks: 任务集合

        Returns:
            待迁移任务ID列表
        """
        migration_tasks = set(self.interrupted_tasks)

        # 添加失效节点的所有任务
        for task in tasks.get_all_tasks():
            agent_id = task.current_agent or task.assigned_agent
            if agent_id in self.failed_agents or agent_id in self.isolated_agents:
                migration_tasks.add(task.task_id)

        return list(migration_tasks)

    def get_statistics(self) -> Dict:
        """获取失效统计信息"""
        return {
            'num_failed': len(self.failed_agents),
            'num_isolated': len(self.isolated_agents),
            'num_interrupted_tasks': len(self.interrupted_tasks),
            'failed_agents': list(self.failed_agents),
            'isolated_agents': list(self.isolated_agents),
            'failure_impact_degrees': self.failure_impact_degrees.copy()
        }

    def reset(self):
        """重置失效状态"""
        self.failed_agents.clear()
        self.isolated_agents.clear()
        self.interrupted_tasks.clear()
        self.isolated_components.clear()
        self.failure_impact_degrees.clear()

    def __repr__(self):
        return f"FailureModel(failed={len(self.failed_agents)}, isolated={len(self.isolated_agents)}, interrupted={len(self.interrupted_tasks)})"
