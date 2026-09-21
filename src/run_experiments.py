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
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from experiments.experiment_runner import run_experiment, run_all_experiments


def main():
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
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='快速测试模式（减少仿真轮次，仅用于验证）'
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("="*80)
    print("🎯 LLM-CECM 统一实验入口")
    print("   基于论文大纲的电碳耦合市场仿真实验")
    print("="*80)
    
    try:
        if args.experiment == 'all':
            # 运行所有实验
            print("\n🚀 运行所有实验...")
            if args.variant:
                print("⚠️  警告: 运行所有实验时，--variant参数将被忽略")
            
            results = run_all_experiments()
            
            print("\n" + "="*80)
            print("✅ 所有实验完成")
            print(f"   完成实验数: {len(results)}")
            print("="*80)
            
        else:
            # 运行单个实验
            print(f"\n🚀 运行实验: {args.experiment}")
            if args.variant:
                print(f"   变体: {args.variant}")
            
            config_kwargs = {}
            if args.delta_perf is not None:
                if args.experiment != 'exp2':
                    print("⚠️  --delta-perf 仅对 exp2 生效，已忽略")
                else:
                    print(f"   δ_perf = {args.delta_perf}")
                    config_kwargs['delta_perf'] = args.delta_perf
            if args.quick_test:
                print("   ⚡ 快速测试模式")
                config_kwargs['num_rounds'] = 3  # 减少轮次用于快速测试
            
            result = run_experiment(
                args.experiment,
                variant=args.variant,
                **config_kwargs
            )
            
            print("\n" + "="*80)
            print("✅ 实验完成")
            print(f"   实验名称: {result['experiment_name']}")
            print(f"   耗时: {result['elapsed_time']/60:.2f} 分钟")
            print(f"   结果目录: {result['results_dir']}")
            print("="*80)
        
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
