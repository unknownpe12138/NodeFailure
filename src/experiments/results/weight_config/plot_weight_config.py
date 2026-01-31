"""
权重配置(λ₁, λ₂)与任务数量联合实验结果可视化脚本
绘制独立子图：
- 任务完成成本图（柱状图，x轴为任务数量）
- 期望完成率图（折线图，x轴为任务数量）
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 算法名称映射
ALGORITHM_NAMES = {
    'CPLEX_results.csv': 'CPLEX',
    'RTM-RPF_results.csv': 'RTM-RPF',
    'LBTM_results.csv': 'LBTM',
    'SPTM_results.csv': 'SPTM'
}

# 算法颜色
COLORS = {
    'CPLEX': '#E74C3C',
    'RTM-RPF': '#3498DB',
    'LBTM': '#2ECC71',
    'SPTM': '#F39C12'
}

# 算法标记
MARKERS = {
    'CPLEX': 'o',
    'RTM-RPF': 's',
    'LBTM': '^',
    'SPTM': 'D'
}


def load_data(results_dir: str) -> dict:
    """加载所有算法的数据"""
    data = {}

    for filename, algo_name in ALGORITHM_NAMES.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                data[algo_name] = df
                print(f"Loaded {algo_name}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")

    return data


def plot_all_metrics(results_dir: str, output_dir: str = None):
    """绘制2行3列的子图"""
    if output_dir is None:
        output_dir = results_dir

    data = load_data(results_dir)

    if not data:
        print("No data loaded!")
        return

    # λ参数配置
    lambda_configs = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]
    task_counts = [10, 20, 30, 40, 50]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    algorithms = list(data.keys())
    n_algos = len(algorithms)

    # 对于每个λ配置绘制一列
    for col, (l1, l2) in enumerate(lambda_configs):
        # 上行：任务完成成本（柱状图）
        ax_cost = axes[0, col]

        x = np.arange(len(task_counts))
        width = 0.18

        for i, algo_name in enumerate(algorithms):
            df = data[algo_name]
            costs = []
            errors = []

            for num_tasks in task_counts:
                row = df[(df['lambda1'] == l1) & (df['lambda2'] == l2) & (df['num_tasks'] == num_tasks)]
                if len(row) > 0:
                    costs.append(row['task_cost'].values[0])
                    errors.append(row['task_cost_ci95'].values[0])
                else:
                    costs.append(0)
                    errors.append(0)

            offset = (i - n_algos/2 + 0.5) * width
            ax_cost.bar(x + offset, costs, width,
                       label=algo_name,
                       color=COLORS.get(algo_name, 'gray'),
                       yerr=errors, capsize=2)

        ax_cost.set_xlabel('Number of Tasks', fontsize=12)
        ax_cost.set_ylabel('Task Completion Cost', fontsize=12)
        ax_cost.set_xticks(x)
        ax_cost.set_xticklabels(task_counts)
        ax_cost.legend(loc='upper left', fontsize=10)
        ax_cost.grid(True, alpha=0.3, axis='y')
        ax_cost.set_title(f'λ₁={l1}, λ₂={l2}', fontsize=12)

        # 下行：期望完成率（折线图）
        ax_ratio = axes[1, col]

        for algo_name in algorithms:
            df = data[algo_name]
            ratios = []
            errors = []

            for num_tasks in task_counts:
                row = df[(df['lambda1'] == l1) & (df['lambda2'] == l2) & (df['num_tasks'] == num_tasks)]
                if len(row) > 0:
                    ratios.append(row['expected_completion_rate'].values[0] * 100)
                    errors.append(row['expected_completion_rate_ci95'].values[0] * 100)
                else:
                    ratios.append(0)
                    errors.append(0)

            ax_ratio.errorbar(task_counts, ratios, yerr=errors,
                            label=algo_name,
                            color=COLORS.get(algo_name, 'gray'),
                            marker=MARKERS.get(algo_name, 'o'),
                            markersize=8, linewidth=2, capsize=3)

        ax_ratio.set_xlabel('Number of Tasks', fontsize=12)
        ax_ratio.set_ylabel('Expected Completion Rate (%)', fontsize=12)
        ax_ratio.legend(loc='upper right', fontsize=10)
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.set_title(f'λ₁={l1}, λ₂={l2}', fontsize=12)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'comparison_all.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def plot_separate_figures(results_dir: str, output_dir: str = None):
    """分别绘制6张独立的图"""
    if output_dir is None:
        output_dir = results_dir

    data = load_data(results_dir)

    if not data:
        print("No data loaded!")
        return

    lambda_configs = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]
    task_counts = [10, 20, 30, 40, 50]
    algorithms = list(data.keys())
    n_algos = len(algorithms)

    # 为每个λ配置生成独立的成本图和完成率图
    for idx, (l1, l2) in enumerate(lambda_configs):
        # 任务完成成本图（柱状图）
        fig, ax = plt.subplots(figsize=(6, 5))
        x = np.arange(len(task_counts))
        width = 0.15

        for i, algo_name in enumerate(algorithms):
            df = data[algo_name]
            costs = []
            errors = []

            for num_tasks in task_counts:
                row = df[(df['lambda1'] == l1) & (df['lambda2'] == l2) & (df['num_tasks'] == num_tasks)]
                if len(row) > 0:
                    costs.append(row['task_cost'].values[0])
                    errors.append(row['task_cost_ci95'].values[0])
                else:
                    costs.append(0)
                    errors.append(0)

            offset = (i - n_algos/2 + 0.5) * width
            ax.bar(x + offset, costs, width,
                   label=algo_name,
                   color=COLORS.get(algo_name, 'gray'),
                   yerr=errors, capsize=2)

        ax.set_xlabel('Number of Tasks', fontsize=22)
        ax.set_ylabel('Task Completion Cost', fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(task_counts, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = f'fig_cost_lambda1_{l1}_lambda2_{l2}.png'.replace('.', '_').replace('_png', '.png')
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

        # 期望完成率图（折线图）
        fig, ax = plt.subplots(figsize=(6, 5))
        x_pos = np.arange(len(task_counts))

        for algo_name in algorithms:
            df = data[algo_name]
            ratios = []
            errors = []

            for num_tasks in task_counts:
                row = df[(df['lambda1'] == l1) & (df['lambda2'] == l2) & (df['num_tasks'] == num_tasks)]
                if len(row) > 0:
                    ratios.append(row['expected_completion_rate'].values[0] * 100)
                    errors.append(row['expected_completion_rate_ci95'].values[0] * 100)
                else:
                    ratios.append(0)
                    errors.append(0)

            ax.errorbar(x_pos, ratios, yerr=errors,
                       label=algo_name,
                       color=COLORS.get(algo_name, 'gray'),
                       marker=MARKERS.get(algo_name, 'o'),
                       markersize=8, linewidth=2, capsize=3)

        ax.set_xlabel('Number of Tasks', fontsize=22)
        ax.set_ylabel('Expected Completion Rate (%)', fontsize=22)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(task_counts, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'fig_ratio_lambda1_{l1}_lambda2_{l2}.png'.replace('.', '_').replace('_png', '.png')
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


if __name__ == '__main__':
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Plot weight config experiment results')
    parser.add_argument('--results-dir', type=str, default=script_dir,
                       help='Directory containing result CSV files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for figures')
    parser.add_argument('--separate', action='store_true',
                       help='Generate separate figures instead of combined')

    args = parser.parse_args()

    print(f"Results directory: {args.results_dir}")

    if args.separate:
        plot_separate_figures(args.results_dir, args.output_dir)
    else:
        plot_all_metrics(args.results_dir, args.output_dir)
