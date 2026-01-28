"""
CPLEX 精确求解器 - RTM-ONF问题
面向节点失效的韧性任务迁移问题最优解求解

决策变量：
  - x[i,k]: 智能体i是否执行任务k
  - y[i,r]: 智能体i是否选择角色r（用于补位）
  - z[j,i]: 智能体j是否补位失效节点i
"""
import sys
import os
import numpy as np
from typing import Dict, List, Set, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problem import RTMONFProblem
from core.agent import ResilientAgent, Role
from core.network import ResilientMultiLayerNetwork
from core.task import TaskSet, Task
from core.failure import FailureModel

try:
    from docplex.mp.model import Model
    _HAS_CPLEX = True
except ImportError:
    _HAS_CPLEX = False


class CPLEX_RTMONF_Solver:
    """
    基于CPLEX的RTM-ONF精确求解器

    求解面向节点失效的韧性任务迁移问题的最优解
    """

    def __init__(self,
                 problem: RTMONFProblem,
                 failure_model: Optional[FailureModel] = None,
                 time_limit: Optional[float] = None,
                 mip_gap: float = 0.01):
        """
        Args:
            problem: RTM-ONF问题实例
            failure_model: 失效模型（用于获取失效节点信息）
            time_limit: 求解时间限制（秒）
            mip_gap: MIP相对间隙
        """
        if not _HAS_CPLEX:
            raise ImportError(
                "docplex 未安装。请先执行: pip install docplex"
            )

        self.problem = problem
        self.failure_model = failure_model
        self.time_limit = time_limit
        self.mip_gap = mip_gap

        self.model: Optional[Model] = None
        self.x_vars = {}  # (agent_id, task_id) -> binary var
        self.y_vars = {}  # (agent_id, role_id) -> binary var
        self.z_vars = {}  # (rep_agent_id, failed_agent_id) -> binary var
        self.u_vars = {}  # task_id -> binary var (未分配标志)

    def _get_failed_and_functional_agents(self) -> Tuple[Set[int], Set[int]]:
        """获取失效节点和功能有效节点"""
        failed_agents = set()
        functional_agents = set()

        for aid, agent in self.problem.agents.items():
            if agent.is_functional:
                functional_agents.add(aid)
            else:
                failed_agents.add(aid)

        # 如果有失效模型，使用其记录
        if self.failure_model:
            failed_agents = self.failure_model.failed_agents.copy()
            functional_agents = {
                aid for aid in self.problem.agents.keys()
                if aid not in failed_agents and self.problem.agents[aid].is_alive
            }

        return failed_agents, functional_agents

    def _build_model(self):
        """构建RTM-ONF的MILP模型"""
        p = self.problem
        m = Model(name="RTMONF_CPLEX")

        if self.time_limit is not None:
            m.set_time_limit(self.time_limit)

        m.parameters.mip.tolerances.mipgap = self.mip_gap

        # 获取失效和功能有效节点
        failed_agents, functional_agents = self._get_failed_and_functional_agents()

        agent_ids = list(functional_agents)  # 只考虑功能有效的节点
        tasks = p.tasks.get_all_tasks()
        task_ids = [t.task_id for t in tasks]

        print(f"  功能有效节点: {len(agent_ids)}, 失效节点: {len(failed_agents)}, 任务: {len(task_ids)}")

        # 构建角色字典
        agent_roles = {}
        all_roles = {}
        for aid in agent_ids:
            agent = p.agents[aid]
            agent_roles[aid] = [r.role_id for r in agent.feasible_roles]
            for role in agent.feasible_roles:
                all_roles[role.role_id] = role

        # ==================== 决策变量 ====================

        # 决策变量1: x[i,k] - 任务分配
        x = {(i, k): m.binary_var(name=f"x_{i}_{k}")
             for i in agent_ids for k in task_ids}
        self.x_vars = x

        # 决策变量2: y[i,r] - 角色选择（用于补位）
        y = {(i, r): m.binary_var(name=f"y_{i}_{r}")
             for i in agent_ids for r in agent_roles[i]}
        self.y_vars = y

        # 决策变量3: z[j,i] - 补位决策（j补位失效节点i）
        z = {}
        if failed_agents:
            for j in agent_ids:
                for i in failed_agents:
                    z[(j, i)] = m.binary_var(name=f"z_{j}_{i}")
        self.z_vars = z

        # 决策变量4: u[k] - 任务未分配标志
        u = {k: m.binary_var(name=f"u_{k}") for k in task_ids}
        self.u_vars = u

        print(f"  决策变量: {len(x)}个任务分配 + {len(y)}个角色选择 + {len(z)}个补位决策")

        # ==================== 约束条件 ====================

        # 约束(3-1): 任务履行约束 - 每个任务要么分配给一个智能体，要么未分配
        for k in task_ids:
            m.add_constraint(
                m.sum(x[(i, k)] for i in agent_ids) + u[k] == 1,
                ctname=f"task_assignment_{k}"
            )

        # 约束: 每个智能体必须选择恰好一个角色
        for i in agent_ids:
            m.add_constraint(
                m.sum(y[(i, r)] for r in agent_roles[i]) == 1,
                ctname=f"role_selection_{i}"
            )

        # 约束: 智能体负载约束（基于工作量）
        # 【修复】使用工作量而非任务数量来限制负载
        task_workloads = {t.task_id: t.workload for t in tasks}
        for i in agent_ids:
            agent = p.agents[i]
            # 当前负载 + 新分配任务的工作量 <= 临界负载
            current_load = agent.load
            available_capacity = max(0.1, p.L_crit - current_load)
            m.add_constraint(
                m.sum(task_workloads[k] * x[(i, k)] for k in task_ids) <= available_capacity,
                ctname=f"agent_load_{i}"
            )

        # 约束(3-5): 补位资格约束 - 每个失效节点最多被一个节点补位
        for i in failed_agents:
            m.add_constraint(
                m.sum(z[(j, i)] for j in agent_ids if (j, i) in z) <= 1,
                ctname=f"replenishment_unique_{i}"
            )

        # 约束: 每个节点最多补位一个失效节点
        for j in agent_ids:
            m.add_constraint(
                m.sum(z[(j, i)] for i in failed_agents if (j, i) in z) <= 1,
                ctname=f"replenishment_limit_{j}"
            )

        # ==================== 预计算系数 ====================

        exec_cost_coeff = {}
        migr_cost_coeff = {}
        completion_prob_coeff = {}
        switch_cost_coeff = {}
        rep_cost_coeff = {}

        # 获取网络层负载 - 用于计算动态存活概率
        # 符合文档式(2-11): π_i(τ) = (1 - Ψ_α(负载)) × ∏(1 - Ψ_β(层负载)) × (1 - μ_i^ξ)
        layer_loads = p.network.get_layer_loads()

        for i in agent_ids:
            agent = p.agents[i]

            # 角色切换代价
            for r in agent_roles[i]:
                target_role = all_roles[r]
                switch_cost_coeff[(i, r)] = agent.compute_switching_cost(target_role)

            # 计算该智能体的动态存活概率 - 符合文档式(2-11)
            # 使用 compute_survival_probability 方法，与 RoleSwitching 保持一致
            agent_layer_loads = {
                lid: layer_loads.get(lid, 0.0)
                for lid in agent.network_layers
            }
            base_survival = agent.compute_survival_probability(agent_layer_loads)

            for k in task_ids:
                task = tasks[task_ids.index(k)]

                # 对每个可能的角色计算执行代价和完成概率
                for r in agent_roles[i]:
                    role = all_roles[r]

                    # 计算该角色的适配度 - 符合文档式(2-2)
                    match_ratio = len(task.requirements & role.capabilities) / len(task.requirements) if len(task.requirements) > 0 else 1.0
                    role_state_factor = np.mean([
                        agent.health,
                        len(role.capabilities),
                        1.0 - agent.load,
                        len(agent.network_layers)
                    ])
                    fitness = (0.4 + 0.6 * match_ratio) * role_state_factor

                    # 执行代价 - 符合文档式(2-12)履行代价部分
                    exec_c = task.workload * (1.0 - fitness)

                    # 迁移代价 - 符合文档式(2-12)迁移代价部分
                    # 使用网络迁移代价 ω_ij，与problem.py保持一致
                    if task.current_agent is not None and task.current_agent in functional_agents:
                        w_ij = p.network.compute_migration_cost(task.current_agent, i)
                        if w_ij >= float('inf'):
                            w_ij = 1e4
                    else:
                        w_ij = 0.0
                    migr_c = task.workload * w_ij

                    # 完成概率 - 符合文档式(3-8): Φ(ρ_i, τ_k) × (1 - p_i^fail) × R_path
                    # 使用动态存活概率，考虑：
                    # 1. 本体负荷影响 Ψ_α
                    # 2. 网络层负荷连带影响 Ψ_β
                    # 3. 角色暴露风险 μ^ξ
                    # 4. 多重度中心性暴露风险 μ_i^M (已在 failure_prob 中体现)

                    # 综合失效概率 p_i^fail = 1 - π_i × (1 - μ_i^M)
                    # 这里 base_survival 已经是 π_i，agent.exposure_risk 是 μ_i^M
                    adjusted_survival = base_survival * (1.0 - agent.exposure_risk)

                    # 路径可靠性 R_path 简化为 1.0（精确求解中不考虑路径风险）
                    path_reliability = 1.0

                    # 韧性期望达成概率
                    combined = fitness * adjusted_survival * path_reliability
                    prob = TaskSet._compute_task_success_prob(combined)

                    exec_cost_coeff[(i, r, k)] = exec_c
                    migr_cost_coeff[(i, k)] = migr_c
                    completion_prob_coeff[(i, r, k)] = prob

        # 补位代价系数
        for j in agent_ids:
            agent_j = p.agents[j]
            for i in failed_agents:
                if (j, i) not in z:
                    continue
                agent_i = p.agents[i]

                # 基础补位代价：角色切换代价 + 连接建立代价
                min_switch_cost = float('inf')
                for r in agent_roles[j]:
                    role = all_roles[r]
                    cost = agent_j.compute_switching_cost(role)
                    if cost < min_switch_cost:
                        min_switch_cost = cost

                # 连接建立代价
                link_cost = len(agent_i.network_layers) * 0.1

                rep_cost_coeff[(j, i)] = min_switch_cost + link_cost

        # ==================== 辅助变量（线性化） ====================

        # w[i,r,k] = x[i,k] * y[i,r]
        w_vars = {}
        for i in agent_ids:
            for r in agent_roles[i]:
                for k in task_ids:
                    w = m.continuous_var(lb=0, ub=1, name=f"w_{i}_{r}_{k}")
                    w_vars[(i, r, k)] = w

                    m.add_constraint(w <= x[(i, k)])
                    m.add_constraint(w <= y[(i, r)])
                    m.add_constraint(w >= x[(i, k)] + y[(i, r)] - 1)

        # ==================== 目标函数 ====================

        # 执行代价
        exec_cost_expr = m.sum(
            exec_cost_coeff[(i, r, k)] * w_vars[(i, r, k)]
            for i in agent_ids for r in agent_roles[i] for k in task_ids
        )

        # 迁移代价
        migr_cost_expr = m.sum(
            migr_cost_coeff[(i, k)] * x[(i, k)]
            for i in agent_ids for k in task_ids
        )

        # 角色切换代价
        switch_cost_expr = m.sum(
            switch_cost_coeff[(i, r)] * y[(i, r)]
            for i in agent_ids for r in agent_roles[i]
            if r != p.agents[i].current_role.role_id
        )

        # 补位代价
        rep_cost_expr = 0
        if z:
            rep_cost_expr = m.sum(
                rep_cost_coeff[(j, i)] * z[(j, i)]
                for (j, i) in z.keys()
            )

        # 总代价
        total_cost_expr = exec_cost_expr + migr_cost_expr + switch_cost_expr + rep_cost_expr

        # 完成率
        if len(task_ids) > 0:
            completion_expr = (1.0 / len(task_ids)) * m.sum(
                completion_prob_coeff[(i, r, k)] * w_vars[(i, r, k)]
                for i in agent_ids for r in agent_roles[i] for k in task_ids
            )
        else:
            completion_expr = 0

        # 【修复】负载集中惩罚：当智能体承担更多任务时，降低完成率
        # 【修复】移除惩罚项，优化纯 utility
        # 原来的惩罚项会干扰优化，导致 CPLEX 不是真正优化 utility
        # 负载约束和任务分配约束应该通过约束条件实现，而不是惩罚项

        # 目标：最小化 -utility = λ1 * Cost - λ2 * Completion * 100
        # 等价于最大化 utility = -λ1 * Cost + λ2 * Completion * 100
        objective = p.lambda1 * total_cost_expr - p.lambda2 * completion_expr * 100

        m.minimize(objective)

        self.model = m
        self.w_vars = w_vars
        self.all_roles = all_roles
        self.agent_roles = agent_roles
        self.agent_ids = agent_ids
        self.task_ids = task_ids
        self.failed_agents = failed_agents

        return m

    def solve(self) -> Dict:
        """求解模型"""
        print("  构建CPLEX模型（RTM-ONF）...")
        m = self._build_model()

        print("  开始求解...")
        sol = m.solve(log_output=False)

        if sol is None:
            print("  未找到可行解")
            return {
                'feasible': False,
                'status': str(m.solve_details.status),
                'completion_ratio': 0.0,
                'utility': float('-inf'),
                'total_cost': float('inf')
            }

        # 提取解
        task_assignment = {}
        role_assignment = {}
        replenishment_plan = {}
        unassigned_tasks = []

        # 任务分配
        for (i, k), var in self.x_vars.items():
            if sol.get_value(var) > 0.5:
                task_assignment[k] = i

        # 未分配任务
        for k, var in self.u_vars.items():
            if sol.get_value(var) > 0.5:
                unassigned_tasks.append(k)

        # 角色分配
        for (i, r), var in self.y_vars.items():
            if sol.get_value(var) > 0.5:
                role_assignment[i] = self.all_roles[r]

        # 补位方案
        for (j, i), var in self.z_vars.items():
            if sol.get_value(var) > 0.5:
                # 找到补位节点选择的角色
                rep_role = role_assignment.get(j, self.problem.agents[j].current_role)
                replenishment_plan[i] = (j, rep_role)

        # 构建迁移流
        migration_flows = {}
        functional_agent_ids = {aid for aid, agent in self.problem.agents.items() if agent.is_functional}

        for tid, aid in task_assignment.items():
            task = self.problem.tasks.get_task(tid)
            if task:
                source = task.current_agent
                # 如果源节点是失效节点或不存在，使用目标节点作为源（本地执行）
                if source is None or source not in functional_agent_ids:
                    source = aid
                migration_flows[(source, aid, tid)] = 1

        # 计算路径可靠性
        path_reliability = {}
        for tid in task_assignment.keys():
            path_reliability[tid] = 1.0  # 简化处理

        # 使用问题实例评估解
        eval_results = self.problem.evaluate_solution(
            task_assignment,
            role_assignment,
            migration_flows,
            replenishment_plan,
            path_reliability
        )

        # 打印违反的约束
        if not eval_results['feasible'] and eval_results.get('violations'):
            print(f"  约束违反: {len(eval_results['violations'])}个")
            for v in eval_results['violations'][:3]:
                print(f"    - {v}")

        return {
            'feasible': eval_results['feasible'],
            'status': str(m.solve_details.status),
            'completion_ratio': eval_results['completion_ratio'],
            'utility': eval_results['utility'],
            'total_cost': eval_results['total_cost'],
            'execution_cost': eval_results['execution_cost'],
            'migration_cost': eval_results['migration_cost'],
            'replenishment_cost': eval_results['replenishment_cost'],
            'task_assignment': task_assignment,
            'role_assignment': {aid: role.function_type for aid, role in role_assignment.items()},
            'replenishment_plan': {
                fid: (rid, role.function_type)
                for fid, (rid, role) in replenishment_plan.items()
            },
            'num_assigned_tasks': len(task_assignment),
            'num_unassigned_tasks': len(unassigned_tasks),
            'num_replenished': len(replenishment_plan),
            'objective_value': eval_results['utility'],  # 使用utility作为objective_value，与RTM-RPF保持一致
            'cplex_objective': sol.get_objective_value(),  # 保留CPLEX原始目标值用于调试
            'violations': eval_results.get('violations', []),
            'mip_gap': m.solve_details.mip_relative_gap if hasattr(m.solve_details, 'mip_relative_gap') else None
        }


def check_cplex_available() -> bool:
    """检查CPLEX是否可用"""
    return _HAS_CPLEX
