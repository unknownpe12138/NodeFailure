"""
权重配置实验结果可视化脚本
绘制不同权重配置下各算法的性能对比图
绘制6张独立子图：
- 3张任务完成成本图（柱状图，x轴为任务数量）
- 3张期望完成率图（折线图，x轴为任务数量）
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 算法列表和颜色
ALGORITHMS = ['CPLEX', 'RTM-RPF', 'SPTM', 'LBTM']
COLORS = {
    'CPLEX': '#E74C3C',
    'RTM-RPF': '#3498DB',
    'SPTM': '#2ECC71',
    'LBTM': '#F39C12'
}
MARKERS = {
    'CPLEX': 'o',
    'RTM-RPF': 's',
    'SPTM': '^',
    'LBTM': 'D'
}

# 权重配置
WEIGHT_CONFIGS = {
    'completion_focused': {'lambda1': 0.25, 'lambda2': 0.75, 'name': 'Completion-Focused'},
    'balanced': {'lambda1': 0.5, 'lambda2': 0.5, 'name': 'Balanced'},
    'cost_focused': {'lambda1': 0.75, 'lambda2': 0.25, 'name': 'Cost-Focused'},
}


def std_to_ci95(std: np.ndarray, n: int) -> np.ndarray:
    """将标准差转换为95%置信区间"""
    return 1.96 * std / np.sqrt(n)


def load_results(results_dir):
    """加载实验结果"""
    summary_file = os.path.join(results_dir, 'weight_config_summary.csv')

    if not os.path.exists(summary_file):
        print(f"Error: Result file not found {summary_file}")
        return None

    df = pd.read_csv(summary_file)

    # 检查是否有CI95列，如果没有则尝试从单独的算法文件加载
    if 'task_cost_ci95' not in df.columns:
        print("Warning: CI95 columns not found in summary file, trying to load from individual algorithm files...")
        return load_results_with_ci95(results_dir)

    return df


def load_results_with_ci95(results_dir):
    """从单独的算法文件加载包含CI95的结果"""
    all_data = []

    for algo in ALGORITHMS:
        filepath = os.path.join(results_dir, f'{algo}_results.csv')
        if os.path.exists(filepath):
            algo_df = pd.read_csv(filepath)
            algo_df['algorithm'] = algo
            all_data.append(algo_df)

    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)
    return df


def plot_all_metrics(results_dir: str, output_dir: str = None):
    """绘制2行3列的综合对比图"""
    if output_dir is None:
        output_dir = results_dir

    df = load_results(results_dir)
    if df is None:
        return

    # 任务数量范围
    task_counts = sorted(df['num_tasks'].unique())
    lambda_configs = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    algorithms = ALGORITHMS
    n_algos = len(algorithms)

    # 对于每个λ配置绘制一列
    for col, (l1, l2) in enumerate(lambda_configs):
        # 上行：任务完成成本（柱状图）
        ax_cost = axes[0, col]

        x = np.arange(len(task_counts))
        width = 0.18

        for i, algo_name in enumerate(algorithms):
            algo_df = df[df['algorithm'] == algo_name]
            costs = []
            errors = []

            for num_tasks in task_counts:
                row = algo_df[(algo_df['lambda1'] == l1) &
                             (algo_df['lambda2'] == l2) &
                             (algo_df['num_tasks'] == num_tasks)]
                if len(row) > 0:
                    costs.append(row['task_cost'].values[0])
                    # 读取置信区间
                    if 'task_cost_ci95' in row.columns:
                        errors.append(row['task_cost_ci95'].values[0])
                    else:
                        errors.append(0)
                else:
                    costs.append(0)
                    errors.append(0)

            offset = (i - n_algos/2 + 0.5) * width
            ax_cost.bar(x + offset, costs, width,
                       label=algo_name,
                       color=COLORS.get(algo_name, 'gray'),
                       yerr=errors, capsize=3)

        ax_cost.set_xlabel('Number of Tasks', fontsize=12)
        ax_cost.set_ylabel('Task Completion Cost', fontsize=12)
        ax_cost.set_xticks(x)
        ax_cost.set_xticklabels(task_counts)
        ax_cost.legend(loc='upper left', fontsize=10)
        ax_cost.grid(True, alpha=0.3, axis='y')
        ax_cost.set_title(f'λ1={l1}, λ2={l2}', fontsize=12)

        # 下行：期望完成率（折线图）
        ax_ratio = axes[1, col]

        for algo_name in algorithms:
            algo_df = df[df['algorithm'] == algo_name]
            ratios = []
            errors = []

            for num_tasks in task_counts:
                row = algo_df[(algo_df['lambda1'] == l1) &
                             (algo_df['lambda2'] == l2) &
                             (algo_df['num_tasks'] == num_tasks)]
                if len(row) > 0:
                    ratios.append(row['expected_completion_rate'].values[0] * 100)
                    # 读取置信区间
                    if 'expected_completion_rate_ci95' in row.columns:
                        errors.append(row['expected_completion_rate_ci95'].values[0] * 100)
                    else:
                        errors.append(0)
                else:
                    ratios.append(0)
                    errors.append(0)

            ax_ratio.errorbar(task_counts, ratios, yerr=errors,
                         label=algo_name,
                         color=COLORS.get(algo_name, 'gray'),
                         marker=MARKERS.get(algo_name, 'o'),
                         markersize=8, linewidth=2, capsize=3)

        ax_ratio.set_xlabel('Number of Tasks', fontsize=12)
        ax_ratio.set_ylabel('Task Completion Ratio (%)', fontsize=12)
        ax_ratio.legend(loc='upper right', fontsize=10)
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.set_title(f'λ1={l1}, λ2={l2}', fontsize=12)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'comparison_all.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def plot_separate_figures(results_dir: str, output_dir: str = None):
    """分别绘制6张独立的图"""
    if output_dir is None:
        output_dir = results_dir

    df = load_results(results_dir)
    if df is None:
        return

    task_counts = sorted(df['num_tasks'].unique())
    lambda_configs = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]
    algorithms = ALGORITHMS
    n_algos = len(algorithms)

    # 为每个λ配置生成独立的成本图和完成率图
    for idx, (l1, l2) in enumerate(lambda_configs):
        # 任务完成成本图（柱状图）
        fig, ax = plt.subplots(figsize=(6, 5))
        x = np.arange(len(task_counts))
        width = 0.15

        for i, algo_name in enumerate(algorithms):
            algo_df = df[df['algorithm'] == algo_name]
            costs = []
            errors = []

            for num_tasks in task_counts:
                row = algo_df[(algo_df['lambda1'] == l1) &
                             (algo_df['lambda2'] == l2) &
                             (algo_df['num_tasks'] == num_tasks)]
                if len(row) > 0:
                    costs.append(row['task_cost'].values[0])
                    # 读取置信区间
                    if 'task_cost_ci95' in row.columns:
                        errors.append(row['task_cost_ci95'].values[0])
                    else:
                        errors.append(0)
                else:
                    costs.append(0)
                    errors.append(0)

            offset = (i - n_algos/2 + 0.5) * width
            ax.bar(x + offset, costs, width,
                   label=algo_name,
                   color=COLORS.get(algo_name, 'gray'),
                   yerr=errors, capsize=3)

        ax.set_xlabel('Number of Tasks', fontsize=22)
        ax.set_ylabel('Task Completion Cost', fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(task_counts, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = f'fig_cost_lambda1_{str(l1).replace(".", "_")}_lambda2_{str(l2).replace(".", "_")}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

        # 期望完成率图（折线图）
        fig, ax = plt.subplots(figsize=(6, 5))
        x_pos = np.arange(len(task_counts))

        for algo_name in algorithms:
            algo_df = df[df['algorithm'] == algo_name]
            ratios = []
            errors = []

            for num_tasks in task_counts:
                row = algo_df[(algo_df['lambda1'] == l1) &
                             (algo_df['lambda2'] == l2) &
                             (algo_df['num_tasks'] == num_tasks)]
                if len(row) > 0:
                    ratios.append(row['expected_completion_rate'].values[0] * 100)
                    # 读取置信区间
                    if 'expected_completion_rate_ci95' in row.columns:
                        errors.append(row['expected_completion_rate_ci95'].values[0] * 100)
                    else:
                        errors.append(0)
                else:
                    ratios.append(0)
                    errors.append(0)

            ax.errorbar(x_pos, ratios, yerr=errors,
                   label=algo_name,
                   color=COLORS.get(algo_name, 'gray'),
                   marker=MARKERS.get(algo_name, 'o'),
                   markersize=8, linewidth=2, capsize=3)

        ax.set_xlabel('Number of Tasks', fontsize=22)
        ax.set_ylabel('Task Completion Ratio (%)', fontsize=22)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(task_counts, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'fig_ratio_lambda1_{str(l1).replace(".", "_")}_lambda2_{str(l2).replace(".", "_")}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")



def main():
    """主函数"""
    import argparse

    # 获取结果目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir

    parser = argparse.ArgumentParser(description='Plot weight config experiment results')
    parser.add_argument('--results-dir', type=str, default=results_dir,
                       help='Directory containing result CSV files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for figures')
    parser.add_argument('--separate', action='store_true',
                       help='Generate separate figures instead of combined')

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory does not exist {args.results_dir}")
        print("Please run the experiment first: python run_weight_config_exp.py")
        return

    print("="*70)
    print("Weight Configuration Experiment Visualization")
    print("="*70)
    print(f"Results directory: {args.results_dir}")

    # 加载结果
    df = load_results(args.results_dir)
    if df is None:
        return

    print(f"Loaded data: {len(df)} records")
    print(f"Algorithms: {df['algorithm'].unique().tolist()}")
    print(f"Lambda configs: {df[['lambda1', 'lambda2']].drop_duplicates().values.tolist()}")

    # 绘制图表
    print("\nGenerating plots...")

    if args.separate:
        plot_separate_figures(args.results_dir, args.output_dir)
    else:
        plot_all_metrics(args.results_dir, args.output_dir)

    print("\n" + "="*70)
    print("Plotting completed!")
    print(f"Figures saved in: {args.output_dir or args.results_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
