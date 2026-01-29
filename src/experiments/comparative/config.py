"""
实验配置模块
定义对比实验的默认参数、变量范围和评估指标
"""

# 算法列表
ALGORITHMS = ['CPLEX', 'RTM-RPF', 'SPTM', 'LBTM']

# 默认参数
DEFAULT_PARAMS = {
    'num_tasks': 40,           # 任务数量
    'num_agents': 25,          # 智能体数量
    'num_layers': 5,           # 网络层数
    'failure_rate': 0.1,       # 基础失效率（风险系数）
    'lambda1': 0.3,            # 代价权重系数
    'lambda2': 0.7,            # 达成率权重系数
    'connection_prob': 0.4,    # 网络连接概率
    'num_capabilities': 10,    # 能力数量
    'capability_coverage': 0.35,  # 能力覆盖率
    'num_roles_per_agent': 3,  # 每个智能体的可行角色数
}

# 变量范围
VARIABLE_RANGES = {
    'num_tasks': [10, 20, 30, 40, 50],
    'num_agents': [15, 20, 25, 30, 35],
    'failure_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
    'num_layers': [3, 4, 5, 6],
    'weight_config': ['completion_focused', 'balanced', 'cost_focused'],  # 权重配置实验
}

# 权重配置映射
WEIGHT_CONFIGS = {
    'completion_focused': {'lambda1': 0.25, 'lambda2': 0.75},
    'balanced': {'lambda1': 0.5, 'lambda2': 0.5},
    'cost_focused': {'lambda1': 0.75, 'lambda2': 0.25},
}

# 权重配置实验的固定参数
WEIGHT_CONFIG_FIXED_PARAMS = {
    'num_agents': 25,
    'num_tasks': [10, 20, 30, 40, 50],  # 任务数量作为变量
}

# 每个实验重复次数
NUM_RUNS = 100

# 评估指标
METRICS = [
    'objective_value',           # 目标函数值（效用）
    'task_cost',                 # 任务完成成本
    'expected_completion_rate',  # 任务期望完成率
    'runtime_seconds',           # 运行时间
]

# 指标映射（从算法结果到CSV列名）
METRIC_MAPPING = {
    'objective_value': 'utility',
    'task_cost': 'total_cost',
    'expected_completion_rate': 'completion_ratio',
    'runtime_seconds': 'execution_time',
}

# 变量目录映射
VARIABLE_DIR_MAP = {
    'num_tasks': 'task_count',
    'num_agents': 'agent_count',
    'failure_rate': 'failure_rate',
    'num_layers': 'network_layers',
    'weight_config': 'weight_config',
}

# 汇总文件名
SUMMARY_FILE = 'summary.csv'
