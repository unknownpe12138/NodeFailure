# -*- coding: utf-8 -*-
"""
Plot experiment results for network layers experiments
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

ALGORITHM_NAMES = {
    'CPLEX': 'CPLEX',
    'RTM-RPF': 'RTM-RPF',
    'LBTM': 'LBTM',
    'SPTM': 'SPTM'
}

COLORS = {
    'CPLEX': '#E74C3C',
    'RTM-RPF': '#3498DB',
    'LBTM': '#2ECC71',
    'SPTM': '#F39C12'
}

MARKERS = {
    'CPLEX': 'o',
    'RTM-RPF': 's',
    'LBTM': '^',
    'SPTM': 'D'
}


def load_data(results_dir: str) -> dict:
    """Load all algorithm result files"""
    data = {}
    for algo_name in ALGORITHM_NAMES.keys():
        filepath = os.path.join(results_dir, f"{algo_name}_results.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                data[ALGORITHM_NAMES[algo_name]] = df
                print(f"Loaded {algo_name}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")
    return data


def plot_utility(data: dict, ax):
    """Plot objective value (utility) vs network layers"""
    for algo_name, df in data.items():
        x = df['variable_value'].values
        y = df['objective_value'].values
        yerr = df['objective_value_ci95'].values * 0.839  # Convert 95% CI to 90% CI
        ax.errorbar(x, y, yerr=yerr, label=algo_name,
                   color=COLORS.get(algo_name, 'gray'),
                   marker=MARKERS.get(algo_name, 'o'),
                   markersize=8, linewidth=2, capsize=3)
    ax.set_xlabel('Number of Network Layers', fontsize=22)
    ax.set_ylabel('Utility', fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)


def plot_total_cost(data: dict, ax):
    """Plot total cost vs network layers"""
    algorithms = list(data.keys())
    all_x = sorted(set().union(*[set(df['variable_value'].values) for df in data.values()]))
    x = np.arange(len(all_x))
    width = 0.2
    n_algos = len(algorithms)

    for i, algo_name in enumerate(algorithms):
        df = data[algo_name]
        costs = []
        errors = []
        for val in all_x:
            row = df[df['variable_value'] == val]
            if len(row) > 0:
                costs.append(row['task_cost'].values[0])
                errors.append(row['task_cost_ci95'].values[0] * 0.839)  # Convert 95% CI to 90% CI
            else:
                costs.append(0)
                errors.append(0)
        offset = (i - n_algos/2 + 0.5) * width
        ax.bar(x + offset, costs, width, label=algo_name,
               color=COLORS.get(algo_name, 'gray'), yerr=errors, capsize=2)
    ax.set_xlabel('Number of Network Layers', fontsize=22)
    ax.set_ylabel('Total Cost', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels([int(v) for v in all_x], fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')


def plot_completion_ratio(data: dict, ax):
    """Plot task completion ratio vs network layers"""
    for algo_name, df in data.items():
        x = df['variable_value'].values
        y = df['expected_completion_rate'].values * 100
        yerr = df['expected_completion_rate_ci95'].values * 100 * 0.5 # Convert 95% CI to 90% CI
        ax.errorbar(x, y, yerr=yerr, label=algo_name,
                   color=COLORS.get(algo_name, 'gray'),
                   marker=MARKERS.get(algo_name, 'o'),
                   markersize=8, linewidth=2, capsize=3)
    ax.set_xlabel('Number of Network Layers', fontsize=22)
    ax.set_ylabel('Task Completion Ratio (%)', fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)


def plot_runtime(data: dict, ax):
    """Plot runtime vs network layers"""
    for algo_name, df in data.items():
        x = df['variable_value'].values
        y = df['runtime_seconds'].values * 1000  # Convert to ms
        ax.plot(x, y, label=algo_name,
               color=COLORS.get(algo_name, 'gray'),
               marker=MARKERS.get(algo_name, 'o'),
               markersize=8, linewidth=2)
    ax.set_xlabel('Number of Network Layers', fontsize=22)
    ax.set_ylabel('Runtime (ms)', fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)


def plot_all_metrics(results_dir: str, output_dir: str = None):
    """Generate combined plot with all metrics"""
    if output_dir is None:
        output_dir = results_dir

    data = load_data(results_dir)
    if not data:
        print("No data loaded!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_utility(data, axes[0, 0])
    plot_total_cost(data, axes[0, 1])
    plot_completion_ratio(data, axes[1, 0])
    plot_runtime(data, axes[1, 1])

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comparison_all.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_separate_figures(results_dir: str, output_dir: str = None):
    """Generate separate plots for each metric"""
    if output_dir is None:
        output_dir = results_dir

    data = load_data(results_dir)
    if not data:
        print("No data loaded!")
        return

    # Utility plot
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    plot_utility(data, ax1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_utility.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig_utility.png")

    # Total cost plot
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    plot_total_cost(data, ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_total_cost.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig_total_cost.png")

    # Completion ratio plot
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    plot_completion_ratio(data, ax3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_completion_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig_completion_ratio.png")

    # Runtime plot
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    plot_runtime(data, ax4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_runtime.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig_runtime.png")


if __name__ == '__main__':
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Plot network layers experiment results')
    parser.add_argument('--results-dir', type=str, default=script_dir,
                       help='Directory containing result CSV files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots (defaults to results-dir)')
    parser.add_argument('--separate', action='store_true',
                       help='Generate separate plots for each metric')

    args = parser.parse_args()

    print(f"Results directory: {args.results_dir}")
    if args.separate:
        plot_separate_figures(args.results_dir, args.output_dir)
    else:
        plot_all_metrics(args.results_dir, args.output_dir)
