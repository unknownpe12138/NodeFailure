"""
RTM-RPF算法演示脚本
面向节点失效的韧性任务迁移问题
"""
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiments.scenario_generator import ScenarioGenerator
from experiments.runner import ExperimentRunner
from algorithms.rtm_rpf import RTM_RPF


def demo_basic():
    """基础演示：运行RTM-RPF算法"""
    print("\n" + "=" * 70)
    print("RTM-RPF算法基础演示")
    print("面向节点失效的韧性任务迁移问题")
    print("=" * 70)

    # 1. 生成场景
    print("\n[1] 生成测试场景...")
    generator = ScenarioGenerator(seed=42)
    problem = generator.generate_medium_scenario()

    print(f"    智能体数量: {len(problem.agents)}")
    print(f"    网络层数量: {len(problem.network.layers)}")
    print(f"    任务数量: {len(problem.tasks)}")

    # 2. 运行算法
    print("\n[2] 运行RTM-RPF算法...")
    runner = ExperimentRunner()
    results = runner.run_algorithm(
        problem=problem,
        algorithm_name="RTM-RPF",
        execute_failure=True,
        random_seed=42
    )

    # 3. 打印结果
    print("\n[3] 算法结果:")
    runner.print_results(results)

    # 4. 打印摘要
    if 'summary' in results:
        print("\n[4] 解摘要:")
        summary = results['summary']
        for key, value in summary.items():
            print(f"    {key}: {value}")

    return results


def demo_comparison():
    """对比演示：比较多个算法"""
    print("\n" + "=" * 70)
    print("算法对比演示")
    print("=" * 70)

    # 1. 生成场景
    print("\n[1] 生成测试场景...")
    generator = ScenarioGenerator(seed=42)
    problem = generator.generate_medium_scenario()

    # 2. 运行对比实验
    print("\n[2] 运行对比实验...")
    runner = ExperimentRunner()
    comparison_results = runner.run_comparison(
        problem=problem,
        algorithms=["RTM-RPF", "GREEDY", "RANDOM"],
        num_runs=3,
        execute_failure=True
    )

    # 3. 打印对比结果
    print("\n[3] 对比结果:")
    runner.print_comparison(comparison_results)

    return comparison_results


def demo_high_failure():
    """高失效率场景演示"""
    print("\n" + "=" * 70)
    print("高失效率场景演示")
    print("=" * 70)

    # 1. 生成高失效率场景
    print("\n[1] 生成高失效率场景...")
    generator = ScenarioGenerator(seed=42)
    problem = generator.generate_high_failure_scenario()

    print(f"    智能体数量: {len(problem.agents)}")
    print(f"    任务数量: {len(problem.tasks)}")

    # 2. 运行算法
    print("\n[2] 运行RTM-RPF算法...")
    runner = ExperimentRunner()
    results = runner.run_algorithm(
        problem=problem,
        algorithm_name="RTM-RPF",
        execute_failure=True,
        random_seed=42
    )

    # 3. 打印结果
    print("\n[3] 算法结果:")
    runner.print_results(results)

    # 4. 打印失效统计
    if 'failure_statistics' in results:
        print("\n[4] 失效统计:")
        fs = results['failure_statistics']
        print(f"    物理失效节点: {fs.get('num_failed', 0)}")
        print(f"    孤岛节点: {fs.get('num_isolated', 0)}")
        print(f"    中断任务: {fs.get('num_interrupted_tasks', 0)}")

    # 5. 打印补位统计
    if 'replenishment_statistics' in results:
        print("\n[5] 补位统计:")
        rs = results['replenishment_statistics']
        print(f"    成功补位: {rs.get('num_replenished', 0)}")
        print(f"    不可恢复: {rs.get('num_unrecoverable', 0)}")
        print(f"    补位代价: {rs.get('total_cost', 0.0):.4f}")

    return results


def demo_adversarial():
    """对抗场景演示"""
    print("\n" + "=" * 70)
    print("对抗场景演示")
    print("=" * 70)

    # 1. 生成对抗场景
    print("\n[1] 生成对抗场景...")
    generator = ScenarioGenerator(seed=42)
    problem = generator.generate_adversarial_scenario()

    # 2. 运行对比实验
    print("\n[2] 运行对比实验...")
    runner = ExperimentRunner()
    comparison_results = runner.run_comparison(
        problem=problem,
        algorithms=["RTM-RPF", "GREEDY"],
        num_runs=3,
        execute_failure=True
    )

    # 3. 打印对比结果
    print("\n[3] 对比结果:")
    runner.print_comparison(comparison_results)

    return comparison_results


def demo_batch_allocation():
    """分批分配模式演示"""
    print("\n" + "=" * 70)
    print("分批分配模式演示")
    print("=" * 70)

    # 1. 生成场景
    print("\n[1] 生成测试场景...")
    generator = ScenarioGenerator(seed=42)
    problem = generator.generate_medium_scenario()

    print(f"    智能体数量: {len(problem.agents)}")
    print(f"    网络层数量: {len(problem.network.layers)}")
    print(f"    任务数量: {len(problem.tasks)}")

    # 2. 运行分批分配
    print("\n[2] 运行分批分配模式...")
    runner = ExperimentRunner()
    results = runner.run_batch_allocation(
        problem=problem,
        algorithm_name="RTM-RPF",
        batch_ratio=0.2,  # 每批20%的任务
        batch_strategy='random',
        random_seed=42
    )

    # 3. 打印结果
    print("\n[3] 最终结果:")
    print(f"    可行性: {results.get('feasible', False)}")
    print(f"    完成率: {results.get('completion_ratio', 0.0):.2%}")
    print(f"    总成本: {results.get('total_cost', 0.0):.2f}")
    print(f"    效用值: {results.get('utility', 0.0):.2f}")
    print(f"    失效节点数: {results.get('total_failed_agents', 0)}")
    print(f"    执行时间: {results.get('execution_time', 0.0):.2f}s")

    return results


def demo_batch_comparison():
    """分批模式与原有模式对比"""
    print("\n" + "=" * 70)
    print("分批模式 vs 原有模式对比")
    print("=" * 70)

    # 1. 生成场景
    print("\n[1] 生成测试场景...")
    generator = ScenarioGenerator(seed=42)
    problem = generator.generate_medium_scenario()

    runner = ExperimentRunner()

    # 2. 运行原有模式
    print("\n[2] 运行原有模式（一次性分配）...")
    problem.reset()
    results_original = runner.run_algorithm(
        problem=problem,
        algorithm_name="RTM-RPF",
        execute_failure=True,
        random_seed=42
    )

    # 3. 运行分批模式
    print("\n[3] 运行分批模式（5批次）...")
    problem.reset()
    results_batch = runner.run_batch_allocation(
        problem=problem,
        algorithm_name="RTM-RPF",
        batch_ratio=0.2,
        batch_strategy='random',
        random_seed=42
    )

    # 4. 打印对比结果
    print("\n[4] 对比结果:")
    print("\n" + "-" * 70)
    print(f"{'指标':<20} {'原有模式':<20} {'分批模式':<20} {'差异':<15}")
    print("-" * 70)

    metrics = [
        ('完成率', 'completion_ratio', '{:.2%}'),
        ('总成本', 'total_cost', '{:.2f}'),
        ('效用值', 'utility', '{:.2f}'),
        ('失效节点数', 'total_failed_agents', '{:.0f}'),
        ('执行时间(s)', 'execution_time', '{:.2f}')
    ]

    for label, key, fmt in metrics:
        val_orig = results_original.get(key, 0.0)
        val_batch = results_batch.get(key, 0.0)
        diff = val_batch - val_orig

        if key == 'completion_ratio':
            diff_str = f"{diff:+.2%}"
        elif key == 'total_failed_agents':
            diff_str = f"{diff:+.0f}"
        else:
            diff_str = f"{diff:+.2f}"

        print(f"{label:<20} {fmt.format(val_orig):<20} {fmt.format(val_batch):<20} {diff_str:<15}")

    print("-" * 70)

    return results_original, results_batch


def demo_no_failure():
    """无失效场景演示（用于验证基础功能）"""
    print("\n" + "=" * 70)
    print("无失效场景演示（验证基础功能）")
    print("=" * 70)

    # 1. 生成场景
    print("\n[1] 生成测试场景...")
    generator = ScenarioGenerator(seed=42)
    problem = generator.generate_small_scenario()

    print(f"    智能体数量: {len(problem.agents)}")
    print(f"    任务数量: {len(problem.tasks)}")

    # 2. 运行算法（不执行失效判定）
    print("\n[2] 运行RTM-RPF算法（无失效）...")
    runner = ExperimentRunner()
    results = runner.run_algorithm(
        problem=problem,
        algorithm_name="RTM-RPF",
        execute_failure=False,  # 不执行失效判定
        random_seed=42
    )

    # 3. 打印结果
    print("\n[3] 算法结果:")
    runner.print_results(results)

    return results


def main():
    """主函数"""
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "RTM-RPF 算法演示程序" + " " * 20 + "#")
    print("#" + " " * 10 + "面向节点失效的韧性任务迁移问题" + " " * 10 + "#")
    print("#" * 70)

    # 运行各个演示
    print("\n选择演示模式:")
    print("  1. 基础演示")
    print("  2. 算法对比")
    print("  3. 高失效率场景")
    print("  4. 对抗场景")
    print("  5. 无失效场景（验证）")
    print("  6. 分批分配模式")
    print("  7. 分批模式对比")
    print("  8. 运行所有演示")

    try:
        choice = input("\n请输入选择 (1-8, 默认1): ").strip() or "1"

        if choice == "1":
            demo_basic()
        elif choice == "2":
            demo_comparison()
        elif choice == "3":
            demo_high_failure()
        elif choice == "4":
            demo_adversarial()
        elif choice == "5":
            demo_no_failure()
        elif choice == "6":
            demo_batch_allocation()
        elif choice == "7":
            demo_batch_comparison()
        elif choice == "8":
            demo_basic()
            demo_comparison()
            demo_high_failure()
            demo_adversarial()
            demo_no_failure()
            demo_batch_allocation()
            demo_batch_comparison()
        else:
            print("无效选择，运行基础演示...")
            demo_basic()

    except KeyboardInterrupt:
        print("\n\n演示已取消")
    except Exception as e:
        print(f"\n演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#" * 70)
    print("#" + " " * 25 + "演示结束" + " " * 25 + "#")
    print("#" * 70)


if __name__ == "__main__":
    main()
