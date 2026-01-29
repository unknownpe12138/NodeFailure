"""
任务模型模块
实现任务的能力需求、负载、适配度等属性
基于研究点一的Task类，扩展中断状态管理
"""
import numpy as np
from typing import Set, List, Dict, Optional


class Task:
    """
    任务类
    扩展研究点一的Task，新增中断状态管理
    """

    def __init__(self, task_id: int, requirements: Set[str],
                 workload: float, urgency: float = 0.5,
                 current_agent: Optional[int] = None,
                 priority: float = 1.0):
        """
        Args:
            task_id: 任务编号 τ_k
            requirements: 能力需求集合 D_k
            workload: 任务工作量 q_k
            urgency: 任务紧迫度 (用于任务排序)
            current_agent: 当前持有该任务的智能体ID
            priority: 任务优先级
        """
        self.task_id = task_id
        self.requirements = requirements  # D_k
        self.workload = workload  # q_k
        self.urgency = urgency
        self.priority = priority
        self.current_agent = current_agent  # 任务当前位置 loc(τ_k)
        self.assigned_agent: Optional[int] = None  # 最终分配的智能体

        # 新增：中断状态
        self.is_interrupted = False  # 是否因节点失效而中断
        self.original_agent: Optional[int] = None  # 中断前的原执行节点

    def mark_interrupted(self):
        """标记任务为中断状态"""
        self.is_interrupted = True
        self.original_agent = self.current_agent

    def mark_migrated(self, new_agent: int):
        """标记任务已迁移到新节点"""
        self.is_interrupted = False
        self.current_agent = new_agent
        self.assigned_agent = new_agent

    def reset_state(self):
        """重置任务状态"""
        self.is_interrupted = False
        self.original_agent = None
        self.assigned_agent = None

    def __repr__(self):
        status = "INT" if self.is_interrupted else "OK"
        return f"Task(id={self.task_id}, reqs={len(self.requirements)}, load={self.workload:.2f}, {status})"

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other):
        if not isinstance(other, Task):
            return False
        return self.task_id == other.task_id


class TaskSet:
    """
    任务集合类
    扩展研究点一的TaskSet，新增中断任务管理
    """

    def __init__(self, tasks: List[Task]):
        """
        Args:
            tasks: 任务列表 T
        """
        self.tasks = {task.task_id: task for task in tasks}
        self.num_tasks = len(tasks)

    def get_task(self, task_id: int) -> Optional[Task]:
        """获取任务"""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        """获取所有任务列表"""
        return list(self.tasks.values())

    def sort_by_urgency(self) -> List[Task]:
        """按紧迫度降序排序任务"""
        return sorted(self.tasks.values(), key=lambda t: t.urgency, reverse=True)

    def sort_by_priority(self) -> List[Task]:
        """按优先级降序排序任务"""
        return sorted(self.tasks.values(), key=lambda t: t.priority, reverse=True)

    def get_total_workload(self) -> float:
        """获取总工作量"""
        return sum(task.workload for task in self.tasks.values())

    def get_interrupted_tasks(self) -> List[Task]:
        """获取所有中断任务"""
        return [task for task in self.tasks.values() if task.is_interrupted]

    def get_tasks_by_agent(self, agent_id: int) -> List[Task]:
        """获取某智能体承载的所有任务"""
        return [task for task in self.tasks.values()
                if task.current_agent == agent_id or task.assigned_agent == agent_id]

    def mark_tasks_interrupted_by_agent(self, agent_id: int) -> List[int]:
        """
        标记某智能体承载的所有任务为中断状态

        Args:
            agent_id: 失效智能体ID

        Returns:
            中断任务ID列表
        """
        interrupted_ids = []
        for task in self.tasks.values():
            if task.current_agent == agent_id or task.assigned_agent == agent_id:
                task.mark_interrupted()
                interrupted_ids.append(task.task_id)
        return interrupted_ids

    def compute_completion_probability(self, agents: Dict[int, 'ResilientAgent']) -> float:
        """
        计算任务集的期望达成概率 - 公式(2-14)

        Args:
            agents: 智能体字典

        Returns:
            任务达成概率
        """
        if not self.tasks:
            return 1.0

        total_prob = 0.0
        for task in self.tasks.values():
            if task.assigned_agent is not None:
                agent = agents.get(task.assigned_agent)
                if agent and agent.is_functional:
                    # 计算适配度
                    fitness = agent.compute_role_task_fitness(task.requirements)

                    # 计算存活概率
                    survival = 1.0 - agent.failure_prob

                    # 单个任务达成概率
                    task_prob = self._compute_task_success_prob(fitness * survival)
                    total_prob += task_prob

        return total_prob / self.num_tasks if self.num_tasks > 0 else 0.0

    def compute_resilient_completion_probability(self,
                                                  agents: Dict[int, 'ResilientAgent'],
                                                  path_reliability: Dict[int, float]) -> float:
        """
        计算韧性任务期望达成概率 - 定义3.8
        考虑角色-任务适配度、智能体存活概率与路径连通可靠性

        Args:
            agents: 智能体字典
            path_reliability: 各任务迁移路径的连通可靠性 {task_id: R_path}

        Returns:
            韧性任务达成概率
        """
        if not self.tasks:
            return 1.0

        total_prob = 0.0
        for task in self.tasks.values():
            if task.assigned_agent is not None:
                agent = agents.get(task.assigned_agent)
                if agent and agent.is_functional:
                    # 角色-任务适配度
                    fitness = agent.compute_role_task_fitness(task.requirements)

                    # 智能体存活概率
                    survival = 1.0 - agent.failure_prob

                    # 路径连通可靠性
                    path_rel = path_reliability.get(task.task_id, 1.0)

                    # 综合因子
                    combined = fitness * survival * path_rel

                    # 任务达成概率
                    task_prob = self._compute_task_success_prob(combined)
                    total_prob += task_prob

        return total_prob / self.num_tasks if self.num_tasks > 0 else 0.0

    @staticmethod
    def _compute_task_success_prob(combined_factor: float) -> float:
        """
        任务成功概率映射函数 Ψ_P - 公式(2-14)
        使用sigmoid函数作为单调递增凸函数

        Args:
            combined_factor: Φ(ρ_i, τ_k) * π_i(τ) * R_path

        Returns:
            成功概率
        """
        return 1.0 / (1.0 + np.exp(-4 * (combined_factor - 0.18)))

    def assign_task(self, task_id: int, agent_id: int):
        """分配任务给智能体"""
        if task_id in self.tasks:
            self.tasks[task_id].assigned_agent = agent_id
            self.tasks[task_id].is_interrupted = False

    def get_assignment_vector(self) -> Dict[int, int]:
        """
        获取任务分配向量

        Returns:
            {task_id: agent_id} 字典
        """
        return {tid: task.assigned_agent
                for tid, task in self.tasks.items()
                if task.assigned_agent is not None}

    def get_migration_tasks(self) -> List[Task]:
        """
        获取待迁移任务集合 T_mig(τ)
        包括中断任务和失效节点的待处理任务
        """
        return [task for task in self.tasks.values()
                if task.is_interrupted or task.assigned_agent is None]

    def reset_all_states(self):
        """重置所有任务状态"""
        for task in self.tasks.values():
            task.reset_state()

    def split_into_batches(self, num_batches: int, strategy: str = 'random') -> List[List[int]]:
        """
        将任务集分成多个批次

        Args:
            num_batches: 批次数量
            strategy: 分批策略 ('random', 'priority', 'urgency')

        Returns:
            List[List[int]]: 任务ID批次列表
        """
        all_task_ids = list(self.tasks.keys())

        if strategy == 'random':
            # 随机打乱后均匀分批
            np.random.shuffle(all_task_ids)
        elif strategy == 'priority':
            # 按优先级排序后均匀分批
            sorted_tasks = sorted(self.tasks.values(), key=lambda t: t.priority, reverse=True)
            all_task_ids = [t.task_id for t in sorted_tasks]
        elif strategy == 'urgency':
            # 按紧迫度排序后均匀分批
            sorted_tasks = sorted(self.tasks.values(), key=lambda t: t.urgency, reverse=True)
            all_task_ids = [t.task_id for t in sorted_tasks]

        # 均匀分批
        batch_size = len(all_task_ids) // num_batches
        batches = []
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size if i < num_batches - 1 else len(all_task_ids)
            batches.append(all_task_ids[start:end])

        return batches

    def __repr__(self):
        interrupted = sum(1 for t in self.tasks.values() if t.is_interrupted)
        return f"TaskSet(num_tasks={self.num_tasks}, interrupted={interrupted}, total_load={self.get_total_workload():.2f})"

    def __len__(self):
        return self.num_tasks

    def __iter__(self):
        return iter(self.tasks.values())
