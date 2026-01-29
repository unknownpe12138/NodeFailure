"""
场景生成器
自动生成测试场景，包括智能体、多重网络、任务等
支持节点失效场景的生成
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import ResilientAgent, Role
from core.network import ResilientMultiLayerNetwork, NetworkLayer
from core.task import Task, TaskSet
from core.problem import RTMONFProblem


class ScenarioGenerator:
    """场景生成器"""

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)

        # 能力池
        self.capability_pool = {
            'sensing', 'communication', 'computing',
            'surveillance', 'strike', 'reconnaissance',
            'jamming', 'relay', 'navigation', 'targeting'
        }

        # 角色类型
        self.role_types = [
            'scout',      # 侦察
            'striker',    # 打击
            'relay',      # 中继
            'jammer',     # 干扰
            'commander'   # 指挥
        ]

    def generate_scenario(self,
                          num_agents: int = 10,
                          num_layers: int = 3,
                          num_tasks: int = 15,
                          num_roles_per_agent: int = 3,
                          connection_prob: float = 0.4,
                          failure_rate: float = 0.2,
                          num_capabilities: int = 10,
                          capability_coverage: float = 0.35,
                          lambda1: float = 0.3,
                          lambda2: float = 0.7) -> RTMONFProblem:
        """
        生成完整场景

        Args:
            num_agents: 智能体数量
            num_layers: 网络层数量
            num_tasks: 任务数量
            num_roles_per_agent: 每个智能体的可行角色数量
            connection_prob: 网络连接概率
            failure_rate: 基础失效率（影响节点健康度）
            num_capabilities: 能力数量
            capability_coverage: 能力覆盖率（每个角色拥有的能力比例）
            lambda1: 代价权重系数
            lambda2: 达成率权重系数

        Returns:
            RTM-ONF问题实例
        """
        print(f"生成场景: {num_agents}个智能体, {num_layers}个网络层, {num_tasks}个任务")

        # 1. 生成能力池
        self._generate_capability_pool(num_capabilities)

        # 2. 生成角色库
        roles = self._generate_roles(num_roles=num_capabilities, capability_coverage=capability_coverage,
                                     failure_rate=failure_rate)

        # 3. 生成智能体
        agents = self._generate_agents(num_agents, roles, num_roles_per_agent, failure_rate)

        # 4. 生成多重网络
        network = self._generate_network(num_layers, agents, roles, connection_prob)

        # 5. 生成任务
        tasks = self._generate_tasks(num_tasks, agents)

        # 6. 创建问题实例
        problem = RTMONFProblem(
            agents=list(agents.values()),
            network=network,
            tasks=tasks,
            all_roles=roles,
            lambda1=lambda1,
            lambda2=lambda2,
            eta_phi=0.25,
            L_crit=10.0
        )

        print(f"场景生成完成!")
        return problem

    def _generate_capability_pool(self, num_capabilities: int = 10):
        """
        生成能力池

        Args:
            num_capabilities: 能力数量
        """
        base_capabilities = [
            'sensing', 'communication', 'computing',
            'surveillance', 'strike', 'reconnaissance',
            'jamming', 'relay', 'navigation', 'targeting',
            'logistics', 'repair', 'defense', 'intelligence',
            'coordination', 'transport', 'medical', 'engineering'
        ]

        if num_capabilities <= len(base_capabilities):
            self.capability_pool = set(base_capabilities[:num_capabilities])
        else:
            self.capability_pool = set(base_capabilities)
            for i in range(num_capabilities - len(base_capabilities)):
                self.capability_pool.add(f'capability_{i}')

    def _generate_roles(self, num_roles: int = 10, capability_coverage: float = 0.35,
                        failure_rate: float = 0.1) -> list:
        """
        生成角色库

        Args:
            num_roles: 角色数量
            capability_coverage: 能力覆盖率（每个角色拥有的能力比例）
            failure_rate: 失效率（影响角色暴露风险，0.1为基准值）
        """
        roles = []
        pool_size = len(self.capability_pool)

        # 根据覆盖率计算每个角色的能力数量
        base_num_caps = max(1, int(pool_size * capability_coverage))

        # 失效率影响因子：将 failure_rate 映射到暴露风险乘数
        # failure_rate=0.05 时，risk_multiplier=0.75（降低风险）
        # failure_rate=0.1 时，risk_multiplier=1.0（基准，保持原有风险）
        # failure_rate=0.15 时，risk_multiplier=1.1
        # failure_rate=0.2 时，risk_multiplier=1.2
        # failure_rate=0.25 时，risk_multiplier=1.3
        # 使用分段线性映射
        if failure_rate <= 0.1:
            # 0.05 -> 0.75, 0.1 -> 1.0
            risk_multiplier = 0.75 + (failure_rate - 0.05) / 0.05 * 0.25
        else:
            # 0.1 -> 1.0, 0.25 -> 1.3
            risk_multiplier = 1.0 + (failure_rate - 0.1) / 0.15 * 0.3

        for i in range(num_roles):
            role_type = self.role_types[i % len(self.role_types)]

            # 根据覆盖率选择能力数量（允许一定波动）
            num_caps = max(1, base_num_caps + np.random.randint(-1, 2))
            num_caps = min(num_caps, pool_size)

            capabilities = set(np.random.choice(
                list(self.capability_pool),
                size=num_caps,
                replace=False
            ))

            # 暴露风险：不同角色有不同风险
            if role_type == 'striker':
                base_risk = np.random.uniform(0.6, 0.8)
            elif role_type == 'scout':
                base_risk = np.random.uniform(0.5, 0.7)
            elif role_type == 'relay':
                base_risk = np.random.uniform(0.3, 0.5)
            elif role_type == 'commander':
                base_risk = np.random.uniform(0.7, 0.9)  # 指挥节点高风险
            else:
                base_risk = np.random.uniform(0.4, 0.6)

            # 应用 failure_rate 影响，确保不超过 0.95
            exposure_risk = min(0.95, base_risk * risk_multiplier)

            role = Role(
                role_id=i,
                capabilities=capabilities,
                function_type=role_type,
                exposure_risk=exposure_risk
            )
            roles.append(role)

        return roles

    def _generate_agents(self,
                         num_agents: int,
                         roles: list,
                         num_roles_per_agent: int,
                         failure_rate: float) -> dict:
        """生成智能体"""
        agents = {}

        for i in range(num_agents):
            # 随机选择可行角色
            feasible_roles = list(np.random.choice(
                roles,
                size=min(num_roles_per_agent, len(roles)),
                replace=False
            ))

            # 随机选择初始角色
            current_role = feasible_roles[0]

            # 初始健康度：受失效率影响
            base_health = 1.0 - failure_rate * 0.3
            health = np.random.uniform(base_health - 0.1, min(1.0, base_health + 0.1))

            agent = ResilientAgent(
                agent_id=i,
                feasible_roles=feasible_roles,
                current_role=current_role,
                initial_health=health
            )

            # 初始负载为0，与RoleSwitching保持一致
            # agent.load 默认为 0.0

            agents[i] = agent

        return agents

    def _generate_network(self,
                          num_layers: int,
                          agents: dict,
                          roles: list,
                          connection_prob: float) -> ResilientMultiLayerNetwork:
        """生成多重网络"""
        network = ResilientMultiLayerNetwork(num_layers=num_layers)

        # 网络层类型
        layer_types = ['communication', 'sensing', 'fire_coordination', 'command', 'intelligence']

        # 创建网络层
        for layer_id in range(num_layers):
            layer_type = layer_types[layer_id % len(layer_types)]
            layer = NetworkLayer(layer_id=layer_id, layer_type=layer_type)
            network.add_layer(layer)

        # 设置角色-层级映射 H_Λ
        for role in roles:
            # 每个角色随机隶属于1-3个网络层
            num_assigned_layers = np.random.randint(1, min(4, num_layers + 1))
            assigned_layers = set(np.random.choice(
                range(num_layers),
                size=num_assigned_layers,
                replace=False
            ))
            network.set_role_layer_mapping(role.role_id, assigned_layers)

        # 添加智能体到网络
        for agent in agents.values():
            network.add_agent(agent)

        # 构建网络层内拓扑
        for layer_id in range(num_layers):
            network.build_layer_topology(layer_id, connection_prob)

        return network

    def _generate_tasks(self,
                        num_tasks: int,
                        agents: dict) -> TaskSet:
        """生成任务集合"""
        tasks = []
        agent_ids = list(agents.keys())

        # 收集所有智能体的能力
        all_agent_capabilities = set()
        for agent in agents.values():
            for role in agent.feasible_roles:
                all_agent_capabilities.update(role.capabilities)

        if not all_agent_capabilities:
            all_agent_capabilities = self.capability_pool

        for i in range(num_tasks):
            # 任务需求数量
            num_reqs = np.random.randint(1, 4)

            # 随机选择能力需求
            requirements = set(np.random.choice(
                list(all_agent_capabilities),
                size=min(num_reqs, len(all_agent_capabilities)),
                replace=False
            ))

            # 工作量
            workload = np.random.uniform(0.5, 2.0)

            # 紧迫度
            urgency = np.random.uniform(0.0, 1.0)

            # 优先级
            priority = np.random.uniform(0.5, 1.5)

            # 随机分配初始位置
            has_initial = np.random.random() < 0.7
            current_agent = np.random.choice(agent_ids) if has_initial else None

            task = Task(
                task_id=i,
                requirements=requirements,
                workload=workload,
                urgency=urgency,
                current_agent=current_agent,
                priority=priority
            )
            tasks.append(task)

        return TaskSet(tasks)

    # ==================== 预设场景 ====================

    def generate_small_scenario(self) -> RTMONFProblem:
        """生成小规模测试场景"""
        return self.generate_scenario(
            num_agents=5,
            num_layers=2,
            num_tasks=8,
            num_roles_per_agent=3,
            connection_prob=0.6,
            failure_rate=0.15
        )

    def generate_medium_scenario(self) -> RTMONFProblem:
        """生成中等规模场景"""
        return self.generate_scenario(
            num_agents=15,
            num_layers=3,
            num_tasks=25,
            num_roles_per_agent=3,
            connection_prob=0.4,
            failure_rate=0.2
        )

    def generate_large_scenario(self) -> RTMONFProblem:
        """生成大规模场景"""
        return self.generate_scenario(
            num_agents=30,
            num_layers=4,
            num_tasks=50,
            num_roles_per_agent=4,
            connection_prob=0.3,
            failure_rate=0.25
        )

    def generate_high_failure_scenario(self) -> RTMONFProblem:
        """生成高失效率场景"""
        return self.generate_scenario(
            num_agents=20,
            num_layers=3,
            num_tasks=30,
            num_roles_per_agent=3,
            connection_prob=0.4,
            failure_rate=0.4
        )

    def generate_adversarial_scenario(self) -> RTMONFProblem:
        """
        生成对抗场景
        特点：高暴露风险、低连通性、高任务压力
        """
        np.random.seed(self.seed)

        num_agents = 20
        num_layers = 3
        num_tasks = 35

        # 生成高风险角色
        roles = []
        for i in range(10):
            role_type = self.role_types[i % len(self.role_types)]
            num_caps = np.random.randint(2, 4)
            capabilities = set(np.random.choice(
                list(self.capability_pool),
                size=num_caps,
                replace=False
            ))
            # 高暴露风险
            exposure_risk = np.random.uniform(0.5, 0.9)

            role = Role(
                role_id=i,
                capabilities=capabilities,
                function_type=role_type,
                exposure_risk=exposure_risk
            )
            roles.append(role)

        # 生成智能体（低健康度）
        agents = {}
        for i in range(num_agents):
            feasible_roles = list(np.random.choice(
                roles,
                size=min(3, len(roles)),
                replace=False
            ))
            current_role = feasible_roles[0]
            health = np.random.uniform(0.6, 0.85)  # 较低健康度

            agent = ResilientAgent(
                agent_id=i,
                feasible_roles=feasible_roles,
                current_role=current_role,
                initial_health=health
            )
            # 初始负载为0，与RoleSwitching保持一致
            agents[i] = agent

        # 生成低连通性网络
        network = self._generate_network(num_layers, agents, roles, connection_prob=0.25)

        # 生成高压力任务
        tasks = self._generate_tasks(num_tasks, agents)

        problem = RTMONFProblem(
            agents=list(agents.values()),
            network=network,
            tasks=tasks,
            all_roles=roles,
            lambda1=0.3,
            lambda2=0.7,
            eta_phi=0.25,
            L_crit=10.0
        )

        print(f"对抗场景生成完成: {num_agents}个智能体, {num_tasks}个任务")
        return problem

    # ==================== 参数化场景生成 ====================

    def generate_scalability_scenarios(self,
                                       agent_counts: list = [10, 20, 30, 40, 50],
                                       task_ratio: float = 1.5) -> list:
        """
        生成可扩展性测试场景

        Args:
            agent_counts: 智能体数量列表
            task_ratio: 任务数/智能体数比例

        Returns:
            问题实例列表
        """
        scenarios = []
        for num_agents in agent_counts:
            num_tasks = int(num_agents * task_ratio)
            num_layers = min(4, 2 + num_agents // 15)

            problem = self.generate_scenario(
                num_agents=num_agents,
                num_layers=num_layers,
                num_tasks=num_tasks,
                num_roles_per_agent=3,
                connection_prob=0.4,
                failure_rate=0.2
            )
            scenarios.append(problem)

        return scenarios

    def generate_failure_rate_scenarios(self,
                                        failure_rates: list = [0.1, 0.2, 0.3, 0.4, 0.5],
                                        num_agents: int = 20,
                                        num_tasks: int = 30) -> list:
        """
        生成不同失效率的测试场景

        Args:
            failure_rates: 失效率列表
            num_agents: 智能体数量
            num_tasks: 任务数量

        Returns:
            问题实例列表
        """
        scenarios = []
        for rate in failure_rates:
            problem = self.generate_scenario(
                num_agents=num_agents,
                num_layers=3,
                num_tasks=num_tasks,
                num_roles_per_agent=3,
                connection_prob=0.4,
                failure_rate=rate
            )
            scenarios.append(problem)

        return scenarios
