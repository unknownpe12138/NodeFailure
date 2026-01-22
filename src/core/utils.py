"""
工具函数模块
提供通用的辅助函数和评估指标
"""
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    归一化值到[0, 1]区间

    Args:
        value: 原始值
        min_val: 最小值
        max_val: 最大值

    Returns:
        归一化后的值
    """
    if max_val <= min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def compute_jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    计算两个集合的Jaccard相似度

    Args:
        set1: 集合1
        set2: 集合2

    Returns:
        Jaccard相似度 ∈ [0, 1]
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def compute_capability_coverage(agent_capabilities: Set[str],
                                task_requirements: Set[str]) -> float:
    """
    计算能力覆盖率

    Args:
        agent_capabilities: 智能体能力集合
        task_requirements: 任务需求集合

    Returns:
        覆盖率 ∈ [0, 1]
    """
    if not task_requirements:
        return 1.0

    covered = len(agent_capabilities & task_requirements)
    return covered / len(task_requirements)


def sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    """
    Sigmoid函数

    Args:
        x: 输入值
        k: 斜率参数
        x0: 中心点

    Returns:
        Sigmoid值 ∈ (0, 1)
    """
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    """
    Softmax函数

    Args:
        values: 输入值列表
        temperature: 温度参数

    Returns:
        概率分布
    """
    if not values:
        return []

    values = np.array(values) / temperature
    exp_values = np.exp(values - np.max(values))  # 数值稳定性
    return (exp_values / np.sum(exp_values)).tolist()


# ==================== 评估指标 ====================

def compute_task_completion_rate(assigned_tasks: int, total_tasks: int) -> float:
    """
    计算任务完成率

    Args:
        assigned_tasks: 已分配任务数
        total_tasks: 总任务数

    Returns:
        完成率 ∈ [0, 1]
    """
    if total_tasks == 0:
        return 1.0
    return assigned_tasks / total_tasks


def compute_network_connectivity(main_component_size: int, total_nodes: int) -> float:
    """
    计算网络连通性

    Args:
        main_component_size: 主连通分量大小
        total_nodes: 总节点数

    Returns:
        连通性 ∈ [0, 1]
    """
    if total_nodes == 0:
        return 1.0
    return main_component_size / total_nodes


def compute_failure_rate(failed_nodes: int, total_nodes: int) -> float:
    """
    计算节点失效率

    Args:
        failed_nodes: 失效节点数
        total_nodes: 总节点数

    Returns:
        失效率 ∈ [0, 1]
    """
    if total_nodes == 0:
        return 0.0
    return failed_nodes / total_nodes


def compute_isolation_rate(isolated_nodes: int, surviving_nodes: int) -> float:
    """
    计算孤岛率

    Args:
        isolated_nodes: 孤岛节点数
        surviving_nodes: 存活节点数

    Returns:
        孤岛率 ∈ [0, 1]
    """
    if surviving_nodes == 0:
        return 0.0
    return isolated_nodes / surviving_nodes


def compute_replenishment_success_rate(replenished: int, failed: int) -> float:
    """
    计算补位成功率

    Args:
        replenished: 成功补位数
        failed: 失效节点数

    Returns:
        补位成功率 ∈ [0, 1]
    """
    if failed == 0:
        return 1.0
    return replenished / failed


def compute_load_balance_index(loads: List[float]) -> float:
    """
    计算负载均衡指数

    使用变异系数的倒数，值越大表示越均衡

    Args:
        loads: 各节点负载列表

    Returns:
        负载均衡指数
    """
    if not loads or len(loads) < 2:
        return 1.0

    mean_load = np.mean(loads)
    if mean_load == 0:
        return 1.0

    std_load = np.std(loads)
    cv = std_load / mean_load  # 变异系数

    # 转换为均衡指数（CV越小越均衡）
    return 1.0 / (1.0 + cv)


def compute_risk_distribution_index(failure_probs: List[float]) -> float:
    """
    计算风险分布指数

    衡量风险在节点间的分布均匀程度

    Args:
        failure_probs: 各节点失效概率列表

    Returns:
        风险分布指数
    """
    if not failure_probs:
        return 1.0

    # 使用熵来衡量分布均匀程度
    probs = np.array(failure_probs)
    probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones_like(probs) / len(probs)

    # 计算熵
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # 归一化（最大熵为log(n)）
    max_entropy = np.log(len(failure_probs))
    if max_entropy == 0:
        return 1.0

    return entropy / max_entropy


# ==================== 统计函数 ====================

def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    计算统计量

    Args:
        values: 数值列表

    Returns:
        统计量字典
    """
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values))
    }


def compute_percentile(values: List[float], percentile: float) -> float:
    """
    计算百分位数

    Args:
        values: 数值列表
        percentile: 百分位 (0-100)

    Returns:
        百分位数值
    """
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


# ==================== 格式化函数 ====================

def format_percentage(value: float, decimals: int = 2) -> str:
    """格式化为百分比字符串"""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 4) -> str:
    """格式化数字"""
    return f"{value:.{decimals}f}"


def format_dict(d: Dict, indent: int = 2) -> str:
    """格式化字典为可读字符串"""
    lines = []
    for key, value in d.items():
        if isinstance(value, float):
            lines.append(f"{' ' * indent}{key}: {value:.4f}")
        elif isinstance(value, dict):
            lines.append(f"{' ' * indent}{key}:")
            lines.append(format_dict(value, indent + 2))
        else:
            lines.append(f"{' ' * indent}{key}: {value}")
    return '\n'.join(lines)


# ==================== 验证函数 ====================

def validate_assignment(task_assignment: Dict[int, int],
                        task_ids: Set[int],
                        agent_ids: Set[int]) -> Tuple[bool, List[str]]:
    """
    验证任务分配方案的有效性

    Args:
        task_assignment: 任务分配方案
        task_ids: 有效任务ID集合
        agent_ids: 有效智能体ID集合

    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []

    for task_id, agent_id in task_assignment.items():
        if task_id not in task_ids:
            errors.append(f"无效任务ID: {task_id}")
        if agent_id not in agent_ids:
            errors.append(f"无效智能体ID: {agent_id}")

    return len(errors) == 0, errors


def validate_replenishment_plan(replenishment_plan: Dict[int, Tuple[int, Any]],
                                failed_ids: Set[int],
                                functional_ids: Set[int]) -> Tuple[bool, List[str]]:
    """
    验证补位方案的有效性

    Args:
        replenishment_plan: 补位方案
        failed_ids: 失效节点ID集合
        functional_ids: 功能有效节点ID集合

    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []

    for failed_id, (rep_id, role) in replenishment_plan.items():
        if failed_id not in failed_ids:
            errors.append(f"非失效节点被补位: {failed_id}")
        if rep_id not in functional_ids:
            errors.append(f"补位节点非功能有效: {rep_id}")
        if rep_id == failed_id:
            errors.append(f"节点自我补位: {rep_id}")

    return len(errors) == 0, errors


# ==================== 随机生成函数 ====================

def generate_random_capabilities(capability_pool: Set[str],
                                 min_caps: int = 2,
                                 max_caps: int = 5) -> Set[str]:
    """
    随机生成能力集合

    Args:
        capability_pool: 能力池
        min_caps: 最小能力数
        max_caps: 最大能力数

    Returns:
        能力集合
    """
    pool_list = list(capability_pool)
    num_caps = np.random.randint(min_caps, min(max_caps + 1, len(pool_list) + 1))
    return set(np.random.choice(pool_list, size=num_caps, replace=False))


def generate_random_workload(min_load: float = 0.5,
                             max_load: float = 2.0) -> float:
    """
    随机生成工作量

    Args:
        min_load: 最小工作量
        max_load: 最大工作量

    Returns:
        工作量
    """
    return np.random.uniform(min_load, max_load)


def generate_random_probability(alpha: float = 2.0, beta: float = 5.0) -> float:
    """
    随机生成概率值（使用Beta分布）

    Args:
        alpha: Beta分布参数α
        beta: Beta分布参数β

    Returns:
        概率值 ∈ [0, 1]
    """
    return np.random.beta(alpha, beta)
