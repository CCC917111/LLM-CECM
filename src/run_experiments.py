#!/usr/bin/env python3
"""
LLM-CECM 统一实验入口

用于运行论文中的所有实验，支持命令行参数选择实验类型。

实验列表：
- exp1: 智能体行为模式验证（RB vs RL vs IBR-ADF）
- exp2: ADF架构效率与精度评估（IBR vs IBR-ADF）
- exp3: IBR-CR机制博弈有效性验证（ADF vs IBR-ADF）
- all: 运行所有实验

使用示例:
    python run_experiments.py --experiment exp1 --variant RB
    python run_experiments.py --experiment exp2 --variant with_adf
    python run_experiments.py --experiment exp3
    python run_experiments.py --experiment all
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from experiments.experiment_configs import EXPERIMENT_VARIANTS, describe_experiment


def main(argv=None):
    """主函数"""
    parser = argparse.ArgumentParser(
        description="LLM-CECM 实验管理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验说明:
  exp1  智能体行为模式验证
        - 对比三种智能体类型：RB（规则）、RL（强化学习）、IBR-ADF（LLM驱动）
        - 使用真实澳洲电价和碳价数据
        - 验证策略性行为（如燃气机组在价格冲击前抬价）
        
  exp2  ADF架构效率与精度评估
        - 对比IBR-Agent（无ADF）vs IBR-ADF-Agent（有ADF）
        - 评估指标：LLM调用次数、运行时间、价格/利润偏差
        
  exp3  IBR-CR机制博弈有效性验证
        - 对比ADF-Agent（无IBR-CR）vs IBR-ADF-Agent（有IBR-CR）
        - 评估指标：智能体利润、策略稳定性、市场价格波动

变体选项:
  exp1: RB | RL | IBR-ADF (默认: IBR-ADF)
  exp2: with_adf | without_adf (默认: with_adf)
  exp3: with_ibr_cr | without_ibr_cr (默认: with_ibr_cr)

示例:
  # 运行实验一的所有变体
  python run_experiments.py --experiment exp1 --variant RB
  python run_experiments.py --experiment exp1 --variant RL
  python run_experiments.py --experiment exp1 --variant IBR-ADF
  
  # 运行实验二
  python run_experiments.py --experiment exp2 --variant without_adf
  python run_experiments.py --experiment exp2 --variant with_adf

  # δ_perf 灵敏度分析（论文 Table VI）
  python run_experiments.py --experiment exp2 --variant with_adf --delta-perf 0.05
  python run_experiments.py --experiment exp2 --variant with_adf --delta-perf 0.25
  
  # 运行所有实验
  python run_experiments.py --experiment all
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        required=True,
        choices=['exp1', 'exp2', 'exp3', 'all'],
        help='实验类型: exp1, exp2, exp3, 或 all'
    )
    
    parser.add_argument(
        '--variant', '-v',
        type=str,
        default=None,
        help='实验变体（具体选项见实验说明）'
    )
    
    parser.add_argument(
        '--delta-perf',
        type=float,
        default=None,
        help='仅exp2：ADF业绩驱动阈值δ_perf（论文Table VI：0.05/0.15/0.25，默认0.15）'
    )
    
    horizon = parser.add_mutually_exclusive_group()
    horizon.add_argument(
        '--quick-test',
        action='store_true',
        help='快速测试模式（减少仿真轮次，仅用于验证）'
    )
    
    horizon.add_argument(
        '--rounds', type=int, default=None,
        help='显式缩短轮数；所有输入序列必须覆盖指定时长，all时应用于每个变体'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='离线输出有效配置、输入覆盖及已知复现缺口，不初始化模型或写结果'
    )
    args = parser.parse_args(argv)
    if args.experiment == 'all' and args.variant is not None:
        parser.error('--variant cannot be combined with --experiment all')
    if args.experiment == 'all' and args.delta_perf is not None:
        parser.error('--delta-perf cannot be combined with --experiment all; select exp2 with_adf')

    config_kwargs = {}
    if args.delta_perf is not None:
        config_kwargs['delta_perf'] = args.delta_perf
    if args.quick_test:
        config_kwargs['num_rounds'] = 3
    elif args.rounds is not None:
        config_kwargs['num_rounds'] = args.rounds

    try:
        if args.experiment == 'all':
            plans = [
                describe_experiment(experiment, variant, **config_kwargs)
                for experiment, variants in EXPERIMENT_VARIANTS.items()
                for variant in variants
            ]
        else:
            plans = [describe_experiment(args.experiment, args.variant, **config_kwargs)]
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        preview = {'experiments': plans} if args.experiment == 'all' else plans[0]
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0

    # Keep the preview and argument-validation path free of model / torch imports.
    from experiments.experiment_runner import run_experiment, run_all_experiments

    print("LLM-CECM: running released experiment configurations")
    try:
        if args.experiment == 'all':
            results = run_all_experiments(**config_kwargs)
            print(f"Completed {len(results)} experiment variants")
        else:
            result = run_experiment(args.experiment, variant=args.variant, **config_kwargs)
            print(f"Completed: {result['experiment_name']}")
            print(f"Elapsed: {result['elapsed_time']/60:.2f} minutes")
            print(f"Results: {result['results_dir']}")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断实验")
        return 130
        
    except Exception as e:
        print(f"\n\n❌ 实验执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
