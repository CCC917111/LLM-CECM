#!/usr/bin/env python3
"""
实验执行模块

提供运行单个或多个实验的高层接口
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.simulator import MarketSimulator
from utils.logger import setup_logging
from utils.visualization import ResultVisualizer
from agents.rl_agent_enhanced import EnhancedRLAgent
from agents.zi_agent import ZIAgent
import json


class ExperimentRunner:
    """实验执行器"""
    
    def __init__(self, config: Dict[str, Any], experiment_name: str):
        """
        初始化实验执行器
        
        Args:
            config: 实验配置字典
            experiment_name: 实验名称（用于日志和结果标识）
        """
        self.config = config
        self.experiment_name = experiment_name
        self.results_dir = Path(config["logging"]["results_file_csv"]).parent
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        setup_logging(
            log_file=config["logging"]["log_file"],
            log_level=logging.INFO
        )
        
    def prepare_agents(self) -> Optional[Dict]:
        """
        根据配置准备智能体
        
        Returns:
            智能体字典，如果使用默认（LLM）则返回None
        """
        agent_type = self.config.get("default_agent_type", "LLM")
        
        # 加载智能体配置
        agents_config_file = "true_digital_twin_agents.json"
        if not Path(agents_config_file).exists():
            raise FileNotFoundError(f"智能体配置文件不存在: {agents_config_file}")
        
        with open(agents_config_file, 'r', encoding='utf-8') as f:
            agents_config = json.load(f)
        
        if agent_type == "RB":
            # 规则基础智能体
            logging.info("使用RB（规则基础）智能体")
            agents = {}
            for agent_cfg in agents_config:
                aid = agent_cfg['agent_id']
                config = agent_cfg['config']
                agents[aid] = ZIAgent(agent_id=aid, config=config)
            return agents
            
        elif agent_type == "RL":
            # 强化学习智能体
            logging.info("使用RL（强化学习）智能体")
            rl_model_dir = self.config.get("rl_model_dir", "models/multi_agent_high_price")
            
            # 查找默认模型
            default_model_path = None
            if os.path.isdir(rl_model_dir):
                for fname in os.listdir(rl_model_dir):
                    if fname.endswith("_model.zip"):
                        default_model_path = os.path.join(rl_model_dir, fname)
                        break
            
            if not default_model_path:
                raise FileNotFoundError(f"RL模型目录中未找到模型: {rl_model_dir}")
            
            agents = {}
            all_agents_info = [a['config'] for a in agents_config]
            
            for agent_cfg in agents_config:
                aid = agent_cfg['agent_id']
                config = agent_cfg['config']
                # 尝试查找专用模型，否则使用默认模型
                specific_model = os.path.join(rl_model_dir, f"{aid}_model.zip")
                model_path = specific_model if os.path.exists(specific_model) else default_model_path
                agents[aid] = EnhancedRLAgent(
                    agent_id=aid,
                    config=config,
                    model_path=model_path,
                    all_agents_info=all_agents_info
                )
            return agents
            
        elif agent_type == "LLM":
            # LLM智能体（默认，由simulator自动创建）
            logging.info("使用LLM智能体（含IBR-CR和ADF机制）")
            # 更新agents配置中的llm_config（合并而非整体替换，保留每个智能体自带的字段）
            llm_config = self.config.get("llm_config", {})
            # ADF参数需要单独下发到 agent_cfg["adf_config"]，
            # simulator._init_agents() 从这里读取并传给 AdaptiveDecisionFrequency
            adf_config = self.config.get("adf_config")
            for agent_cfg in agents_config:
                agent_cfg["llm_config"] = {**agent_cfg.get("llm_config", {}), **llm_config}
                if adf_config is not None:
                    agent_cfg["adf_config"] = dict(adf_config)
            if adf_config is not None and llm_config.get("enable_adf", True):
                logging.info(
                    "ADF参数: delta_perf=%s -> profit_threshold=%s, "
                    "price_volatility_threshold=%s, strategic_review_cycle=%s",
                    adf_config.get("delta_perf"), adf_config.get("profit_threshold"),
                    adf_config.get("price_volatility_threshold"), adf_config.get("strategic_review_cycle"),
                )
            self.config["agents"] = agents_config
            return None
        
        else:
            raise ValueError(f"不支持的智能体类型: {agent_type}")
    
    def run(self) -> Dict[str, Any]:
        """
        运行实验
        
        Returns:
            实验结果字典
        """
        start_time = time.time()
        
        logging.info("="*80)
        logging.info(f"开始实验: {self.experiment_name}")
        logging.info(f"仿真名称: {self.config['simulation_name']}")
        logging.info(f"仿真轮次: {self.config['num_rounds']}")
        logging.info("="*80)
        
        # 准备智能体
        external_agents = self.prepare_agents()
        
        # 创建并运行仿真器
        if external_agents:
            simulator = MarketSimulator(self.config, external_agents=external_agents)
        else:
            simulator = MarketSimulator(self.config)
        
        results = simulator.run_coupled()
        
        elapsed_time = time.time() - start_time
        
        # 记录结果摘要
        logging.info("="*80)
        logging.info(f"实验完成: {self.experiment_name}")
        logging.info(f"总耗时: {elapsed_time/60:.2f} 分钟")
        
        # 统计智能体利润
        total_profit = sum(r['total_profit'] for r in results.values())
        avg_profit = total_profit / len(results) if results else 0
        
        logging.info(f"智能体数量: {len(results)}")
        logging.info(f"总利润: {total_profit:.2f}")
        logging.info(f"平均利润: {avg_profit:.2f}")
        logging.info(f"结果已保存至: {self.results_dir}")
        logging.info("="*80)
        
        # 生成可视化
        try:
            self._generate_visualizations()
        except Exception as e:
            logging.error(f"生成可视化时出错: {e}", exc_info=True)
        
        return {
            "experiment_name": self.experiment_name,
            "elapsed_time": elapsed_time,
            "total_profit": total_profit,
            "avg_profit": avg_profit,
            "num_agents": len(results),
            "results_dir": str(self.results_dir)
        }
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        logging.info("正在生成可视化图表...")
        
        results_csv = self.config["logging"]["results_file_csv"]
        
        # 获取真实数据用于对比
        real_data = self.config.get("real_data_dynamics", {})
        ground_truth_rrp = real_data.get("ground_truth_rrp", [])
        ground_truth_carbon = real_data.get("ground_truth_carbon_price", [])
        
        visualizer = ResultVisualizer(
            results_file=results_csv,
            agents_config_path="true_digital_twin_agents.json",
            ground_truth_prices=ground_truth_rrp,
            ground_truth_carbon_prices=ground_truth_carbon,
            output_dir=str(self.results_dir / "visualization")
        )
        
        generated_files = visualizer.generate_all_visualizations()
        logging.info(f"已生成 {len(generated_files)} 个可视化图表")


def run_experiment(experiment_id: str, variant: str = None, **config_kwargs) -> Dict[str, Any]:
    """
    运行单个实验
    
    Args:
        experiment_id: 实验ID ("exp1", "exp2", "exp3")
        variant: 实验变体（具体含义依实验而定）
        **config_kwargs: 传递给配置函数的额外参数
    
    Returns:
        实验结果字典
    """
    from experiments.experiment_configs import get_experiment_config
    
    # 根据实验ID和变体获取配置
    if experiment_id == "exp1":
        # variant: "RB" | "RL" | "IBR-ADF"
        agent_type = variant or "IBR-ADF"
        config = get_experiment_config("exp1", agent_type=agent_type)
        experiment_name = f"Exp1_BehaviorValidation_{agent_type}"
        
    elif experiment_id == "exp2":
        # variant: "with_adf" | "without_adf"
        enable_adf = (variant != "without_adf")
        exp2_kwargs = {"enable_adf": enable_adf}
        delta_perf = config_kwargs.pop("delta_perf", None)
        if delta_perf is not None:
            exp2_kwargs["delta_perf"] = float(delta_perf)
        config = get_experiment_config("exp2", **exp2_kwargs)
        experiment_name = f"Exp2_ADF_Efficiency_{'with' if enable_adf else 'without'}_ADF"
        if delta_perf is not None and enable_adf:
            experiment_name += f"_dperf{float(delta_perf):g}"
        
    elif experiment_id == "exp3":
        # variant: "with_ibr_cr" | "without_ibr_cr"
        enable_ibr_cr = (variant != "without_ibr_cr")
        config = get_experiment_config("exp3", enable_ibr_cr=enable_ibr_cr)
        experiment_name = f"Exp3_IBR_CR_Effectiveness_{'with' if enable_ibr_cr else 'without'}_IBR_CR"
        
    else:
        raise ValueError(f"无效的实验ID: {experiment_id}")
    
    # delta_perf 只对 exp2 有意义，其余实验忽略
    config_kwargs.pop("delta_perf", None)

    # 应用config_kwargs（如quick-test的num_rounds）
    for key, value in config_kwargs.items():
        if key in config:
            config[key] = value
    
    # 创建并运行实验
    runner = ExperimentRunner(config, experiment_name)
    return runner.run()


def run_all_experiments() -> List[Dict[str, Any]]:
    """
    运行所有实验
    
    Returns:
        所有实验结果的列表
    """
    all_results = []
    
    print("="*80)
    print("开始运行论文所有实验")
    print("="*80)
    
    # 实验一：三种智能体类型
    print("\n📊 实验一：智能体行为模式验证")
    for agent_type in ["RB", "RL", "IBR-ADF"]:
        print(f"\n运行实验一 - {agent_type} 智能体...")
        result = run_experiment("exp1", variant=agent_type)
        all_results.append(result)
    
    # 实验二：ADF效率评估
    print("\n\n⚡ 实验二：ADF架构效率与精度评估")
    for variant in ["without_adf", "with_adf"]:
        print(f"\n运行实验二 - {variant}...")
        result = run_experiment("exp2", variant=variant)
        all_results.append(result)
    
    # 实验三：IBR-CR机制验证
    print("\n\n🎯 实验三：IBR-CR机制博弈有效性验证")
    for variant in ["without_ibr_cr", "with_ibr_cr"]:
        print(f"\n运行实验三 - {variant}...")
        result = run_experiment("exp3", variant=variant)
        all_results.append(result)
    
    print("\n" + "="*80)
    print("✅ 所有实验完成！")
    print("="*80)
    
    # 打印总结
    print("\n实验总结:")
    total_time = sum(r['elapsed_time'] for r in all_results)
    print(f"总耗时: {total_time/3600:.2f} 小时")
    print(f"完成实验数: {len(all_results)}")
    print("\n各实验详情:")
    for i, result in enumerate(all_results, 1):
        print(f"{i}. {result['experiment_name']}")
        print(f"   耗时: {result['elapsed_time']/60:.2f} 分钟")
        print(f"   结果目录: {result['results_dir']}")
    
    return all_results
