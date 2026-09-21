# llm_market_sim/simulation/simulator.py
import logging
import time
import csv
import os
import json
import re
import random
import copy
import traceback
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 修改相对导入为绝对导入
from market.market_environment import MarketEnvironment
from agents.base_agent import BaseAgent
from agents.llm_agent import LLMAgent
from agents.rl_agent import RLAgent
from llm_interface.llm_client import LLMClient
from utils.validation import BidValidator
from market.carbon_market import CarbonMarket
from market.coupled_market import CoupledMarket
from simulation.event_manager import EventManager
from simulation.behavior_analyzer import BehaviorAnalyzer


class MarketSimulator:
    """
    电力市场仿真器。
    负责初始化市场环境和智能体，并管理整个仿真过程。
    """
    def __init__(self, config: Dict[str, Any], external_agents: Optional[Dict[str, BaseAgent]] = None, results_dir: Optional[str] = None):
        """
        初始化仿真器。
        
        Args:
            config: 仿真配置，包含市场参数、智能体配置等
            external_agents: 可选，外部传入的已初始化智能体字典 {agent_id: BaseAgent 实例}
            results_dir: 可选，结果文件输出目录；不传则使用配置中的默认路径
        """
        self.config = config
        self.num_rounds = config["num_rounds"]
        
        # 初始化市场环境
        market_config = config.get("market", {})
        self.market_environment = MarketEnvironment(market_config)
        
        # 初始化LLM客户端
        llm_config = config.get("llm_client", {})
        self.llm_client = LLMClient(llm_config)
        
        # 初始化智能体
        self.agents: Dict[str, BaseAgent] = {}
        if external_agents is not None:
            # 直接使用外部传入的智能体集合（用于RL训练环境等场景）
            self.agents = external_agents
            logging.info(f"使用外部传入的智能体，共 {len(self.agents)} 个")
        else:
            # 从配置构造智能体
            self._init_agents()

        # 初始化事件管理器
        market_events = config.get("market_events", [])
        self.event_manager = EventManager(market_events)

        # 初始化行为分析器
        self.behavior_analyzer = BehaviorAnalyzer()
        self.behavior_analyzer.classify_agents(list(self.agents.values()))

        # 实时数据动态配置
        real_cfg = self.config.get("real_data_dynamics", {})
        self.use_realdata_dynamics = bool(real_cfg.get("enabled", True))
        self.gas_prices_by_round = list(real_cfg.get("gas_prices_by_round", [10, 11, 12, 14, 16, 18, 20, 24, 27, 29]))
        self.coal_outage_rates_by_round = list(real_cfg.get("coal_outage_rates_by_round", [0, 0, 0.05, 0.08, 0.12, 0.16, 0.20, 0.22, 0.18, 0.15]))
        self.demand_increase_by_round = list(real_cfg.get("demand_increase_by_round", [0, 20, 40, 60, 80, 100, 103, 103, 103, 103]))
        # 记录需求基线（用于每轮在基线上叠加增长值）
        self._base_demand_profile = list(self.market_environment.demand_profile)
        self.random_seed = int(real_cfg.get("random_seed", 42))
        random.seed(self.random_seed)
        self._realdata_prev_outage_agents = set()

        # 为可再生能源出力系数机制备份原始容量
        self.solar_capacity_factors = list(real_cfg.get("solar_capacity_factors", []))
        self.wind_capacity_factors = list(real_cfg.get("wind_capacity_factors", []))
        self._renewable_base_capacities = {}
        if self.use_realdata_dynamics and (self.solar_capacity_factors or self.wind_capacity_factors):
            for agent_id, agent in self.agents.items():
                fuel_cat = agent.private_info.get('fuel_category', '')
                if fuel_cat in ['Solar', 'Wind']:
                    self._renewable_base_capacities[agent_id] = agent.private_info.get('max_capacity', 0)

        # 避免事件倍数与真实数据动态同时叠加
        try:
            self.event_manager.disable_fuel_price_multiplier = bool(self.use_realdata_dynamics)
        except Exception:
            pass

        # 准备结果记录
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            base_name = os.path.basename(config["logging"]["results_file_csv"])
            results_file = os.path.join(results_dir, base_name)
        else:
            results_file = config["logging"]["results_file_csv"]
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        self.results_file = results_file
        self.results: Dict[str, Dict[str, Any]] = {}
        
        # 初始化碳市场
        carbon_config = config.get("carbon_market", {})
        carbon_config['agent_ids'] = list(self.agents.keys())  # 确保包含智能体列表
        self.carbon_market = CarbonMarket(carbon_config)
        # 初始化耦合市场
        self.coupled_market = CoupledMarket(self.market_environment, self.carbon_market, self.config)
        
        logging.info(f"仿真器初始化完成，包含 {len(self.agents)} 个智能体")
    
    def _init_agents(self) -> None:
        """
        初始化所有智能体。
        支持两种模式：
        1. 直接使用config.py中定义的智能体配置
        2. 使用LLM批量生成智能体配置（需在config中设置auto_generate_agents=True）
        """
        # 检查是否启用自动生成智能体
        auto_generate = self.config.get("auto_generate_agents", False)
        
        if auto_generate:
            # 使用LLM批量生成智能体配置
            agent_configs = self._generate_agents_with_llm()
            logging.info(f"使用LLM批量生成了 {len(agent_configs)} 个智能体配置")
        else:
            # 使用配置文件中的智能体配置
            agent_configs = self.config.get("agents", [])
            logging.info(f"从配置文件加载了 {len(agent_configs)} 个智能体配置")
        
        # 获取所有智能体ID，用于构建竞争对手列表
        all_agent_ids = [agent_config["agent_id"] for agent_config in agent_configs]
        
        for agent_config in agent_configs:
            agent_id = agent_config["agent_id"]
            agent_type = agent_config["agent_type"]
            
            # 构建该智能体的竞争对手列表
            competitor_ids = [aid for aid in all_agent_ids if aid != agent_id]
            
            if agent_type == "LLM":
                # 对于LLM智能体，需要传递额外的配置
                belief_params = agent_config.get("belief_config", {})
                memory_capacity = agent_config.get("memory_config", {}).get("capacity", 100)
                llm_model = agent_config.get("llm_config", {}).get("model", "gemma-3-27b-it")
                
                # 获取角色描述（如果有）
                role_description = agent_config.get("role_description", "一家普通的发电企业")
                
                agent = LLMAgent(
                    agent_id=agent_id,
                    config=agent_config["config"],
                    llm_client=self.llm_client,
                    competitor_ids=competitor_ids,
                    initial_belief_params=belief_params,
                    memory_capacity=memory_capacity,
                    role_description=role_description,
                    adf_config=agent_config.get("adf_config", {}),
                    llm_config=agent_config.get("llm_config", {})
                )
                
                logging.info(f"创建LLM智能体: {agent_id}, 模型: {llm_model}")
                self.agents[agent_id] = agent
                
            elif agent_type == "RL":
                # 预训练RL智能体（推理模式）。
                # 期望在agent_config中可选提供 model_path；未提供则RLAgent将使用fallback策略。
                model_path = agent_config.get("model_path")
                try:
                    agent = RLAgent(
                        agent_id=agent_id,
                        config=agent_config["config"],
                        model_path=model_path
                    )
                    logging.info(f"创建RL智能体: {agent_id}, model_path={model_path}")
                    self.agents[agent_id] = agent
                except ImportError as e:
                    logging.error(f"创建RL智能体 {agent_id} 失败（依赖未安装）: {e}")
                except Exception as e:
                    logging.error(f"创建RL智能体 {agent_id} 失败: {e}")

            # 可以在这里添加其他类型的智能体初始化逻辑
            # elif agent_type == "RuleBased":
            #     ...
            else:
                logging.warning(f"未知的智能体类型 '{agent_type}'，跳过 {agent_id}")
                
    def _apply_realdata_gas_cost(self, round_number: int) -> None:
        if not self.use_realdata_dynamics:
            return
        if 0 <= round_number < len(self.gas_prices_by_round):
            gas_price_gj = float(self.gas_prices_by_round[round_number])
            gas_marginal_cost = gas_price_gj * 8.0
            updated = 0
            for agent in self.agents.values():
                fuel = str(agent.private_info.get('fuel_category', ''))
                if 'gas' in fuel.lower():
                    agent.config['marginal_cost_energy'] = gas_marginal_cost
                    agent.private_info['marginal_cost_energy'] = gas_marginal_cost
                    reserve_cost = gas_marginal_cost * 0.3
                    agent.config['marginal_cost_reserve'] = reserve_cost
                    agent.private_info['marginal_cost_reserve'] = reserve_cost
                    updated += 1
            if updated:
                logging.info(f"t={round_number}: 更新燃气机组边际成本为 {gas_marginal_cost:.1f} 元/MWh，受影响机组 {updated} 个")

    def _apply_realdata_coal_outage(self, round_number: int) -> None:
        if not self.use_realdata_dynamics:
            return
        rate = 0.0
        if 0 <= round_number < len(self.coal_outage_rates_by_round):
            rate = float(self.coal_outage_rates_by_round[round_number])
        # 恢复上轮停机机组
        if self._realdata_prev_outage_agents:
            for aid in list(self._realdata_prev_outage_agents):
                agent = self.agents.get(aid)
                if agent is None:
                    self._realdata_prev_outage_agents.discard(aid)
                    continue
                original_capacity = agent.private_info.get('original_max_capacity', agent.config.get('max_capacity', 0))
                agent.private_info['max_capacity'] = original_capacity
                agent.config['max_capacity'] = original_capacity
                if 'original_min_output' in agent.private_info:
                    agent.private_info['min_output'] = agent.private_info['original_min_output']
                    agent.config['min_output'] = agent.private_info['original_min_output']
                agent.private_info['is_outage'] = False
                self._realdata_prev_outage_agents.discard(aid)
        # 选择本轮需要停机的煤机
        coal_agents = [a for a in self.agents.values() if 'coal' in str(a.private_info.get('fuel_category', '')).lower()]
        num_outage = int(round(len(coal_agents) * rate))
        if num_outage > 0 and coal_agents:
            k = min(num_outage, len(coal_agents))
            selected = random.sample(coal_agents, k=k)
            for agent in selected:
                if 'original_max_capacity' not in agent.private_info:
                    agent.private_info['original_max_capacity'] = agent.private_info.get('max_capacity', agent.config.get('max_capacity', 0))
                if 'original_min_output' not in agent.private_info:
                    agent.private_info['original_min_output'] = agent.private_info.get('min_output', agent.config.get('min_output', 0))
                agent.private_info['max_capacity'] = 0
                agent.config['max_capacity'] = 0
                agent.private_info['min_output'] = 0
                agent.config['min_output'] = 0
                agent.private_info['is_outage'] = True
                self._realdata_prev_outage_agents.add(agent.agent_id)
                logging.info(f"t={round_number}: 煤炭机组 {agent.agent_id} 计划外停机 (比例 {rate*100:.0f}%)")
        else:
            logging.info(f"t={round_number}: 本轮煤炭停机比例 {rate*100:.0f}%，无停机机组")

    def _apply_realdata_demand_adjustment(self, round_number: int) -> None:
        if not self.use_realdata_dynamics:
            return
        if 0 <= round_number < len(self._base_demand_profile):
            base = self._base_demand_profile[round_number]
            inc = 0
            if 0 <= round_number < len(self.demand_increase_by_round):
                inc = int(self.demand_increase_by_round[round_number])
            new_demand = int(base + inc)
            self.market_environment.demand_profile[round_number] = new_demand
            # 备用需求按8%自动调整（若reserve_profile存在） - 当前实验中不启用
            # try:
            #     self.market_environment.reserve_profile[round_number] = int(new_demand * 0.08)
            # except Exception:
            #     pass
            logging.info(f"t={round_number}: 需求调整 基线={base}MW + 增长={inc}MW => {new_demand}MW (备用≈{int(new_demand*0.08)}MW)")

    def _apply_renewable_capacity_factors(self, round_number: int) -> None:
        """根据预设的出力系数，动态调整风能和太阳能机组的可用容量。"""
        if not self.use_realdata_dynamics or not self._renewable_base_capacities:
            return

        for agent_id, base_capacity in self._renewable_base_capacities.items():
            agent = self.agents.get(agent_id)
            if not agent:
                continue

            fuel_cat = agent.private_info.get('fuel_category', '')
            factor = 1.0

            if fuel_cat == 'Solar' and 0 <= round_number < len(self.solar_capacity_factors):
                factor = self.solar_capacity_factors[round_number]
            elif fuel_cat == 'Wind' and 0 <= round_number < len(self.wind_capacity_factors):
                factor = self.wind_capacity_factors[round_number]
            
            new_capacity = base_capacity * factor
            # 更新智能体的 private_info，出清时会使用
            agent.private_info['max_capacity'] = new_capacity
            # 同时更新 min_output，假设按比例缩放
            if 'min_output' in agent.private_info and base_capacity > 0:
                original_min_output_ratio = agent.private_info.get('original_min_output_ratio', agent.private_info['min_output'] / base_capacity)
                agent.private_info['original_min_output_ratio'] = original_min_output_ratio # 备份原始比例
                agent.private_info['min_output'] = new_capacity * original_min_output_ratio

            logging.info(f"t={round_number}: {fuel_cat}机组 {agent_id} 容量调整: 原始={base_capacity:.0f}MW * 系数={factor:.2f} => 当前可用={new_capacity:.0f}MW")

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        运行整个仿真过程。
        
        Returns:
            仿真结果字典，键是智能体ID，值是结果统计数据
        """
        start_time = time.time()
        
        # 初始化结果记录
        for agent_id in self.agents.keys():
            self.results[agent_id] = {
                "rounds_data": [],
                "total_profit": 0.0,
                "avg_profit": 0.0,
                "max_profit": float('-inf'),
                "min_profit": float('inf')
            }
        
        # 准备CSV结果文件
        with open(self.results_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'round', 'timestamp', 'energy_price', 'reserve_price', 'energy_demand', 'reserve_demand',
                # 碳市场整体数据
                'carbon_market_price', 'carbon_clearing_price', 'carbon_total_volume', 'carbon_num_trades',
                'carbon_supply_demand_imbalance', 'carbon_market_maker_inventory',
                'carbon_total_buy_demand', 'carbon_total_sell_supply', 'carbon_market_tightness', 'carbon_price_change'
            ]
            
            # 为每个智能体添加字段
            for agent_id in self.agents.keys():
                agent_fields = [
                    # 电力市场字段
                    f"{agent_id}_energy_bid_price",
                    f"{agent_id}_energy_bid_quantity",
                    f"{agent_id}_reserve_bid_price", 
                    f"{agent_id}_reserve_bid_quantity",
                    f"{agent_id}_cleared_energy",
                    f"{agent_id}_cleared_reserve",
                    f"{agent_id}_profit",
                    # 碳市场分析所需字段
                    f"{agent_id}_dispatch_energy", # 实际发电量
                    f"{agent_id}_emissions",       # 实际排放量
                    # 碳市场字段
                    f"{agent_id}_carbon_cost",
                    f"{agent_id}_carbon_quota",
                    f"{agent_id}_carbon_funds",
                    f"{agent_id}_carbon_buy_orders",
                    f"{agent_id}_carbon_sell_orders",
                    # ADF决策相关字段
                    f"{agent_id}_decision_type",
                    f"{agent_id}_llm_trigger_reason",
                    f"{agent_id}_total_tokens",
                    f"{agent_id}_prompt_tokens",
                    f"{agent_id}_completion_tokens"
                ]
                fieldnames.extend(agent_fields)
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # 运行每一轮仿真
            for round_number in range(self.num_rounds):
                round_results = self._run_round(round_number)
                
                # 写入本轮结果到CSV
                round_record = {
                    'round': round_number,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'energy_price': round_results['market_results']['energy_price'],
                    'reserve_price': round_results['market_results']['reserve_price'],
                    'energy_demand': self.market_environment.demand_profile[round_number],
                    'reserve_demand': self.market_environment.reserve_profile[round_number],
                    # 碳市场整体数据
                    'carbon_market_price': round_results['market_results'].get('carbon_market', {}).get('current_price', 0),
                    'carbon_clearing_price': round_results['market_results'].get('carbon_market', {}).get('clearing_price', 0),
                    'carbon_total_volume': round_results['market_results'].get('carbon_market', {}).get('total_volume', 0),
                    'carbon_num_trades': len(round_results['market_results'].get('carbon_market', {}).get('trades', [])),
                    'carbon_supply_demand_imbalance': round_results['market_results'].get('carbon_market', {}).get('supply_demand_imbalance', 0),
                    'carbon_market_maker_inventory': round_results['market_results'].get('carbon_market', {}).get('market_maker_inventory', 0),
                    'carbon_total_buy_demand': round_results['market_results'].get('carbon_market', {}).get('total_buy_demand', 0),
                    'carbon_total_sell_supply': round_results['market_results'].get('carbon_market', {}).get('total_sell_supply', 0),
                    'carbon_market_tightness': round_results['market_results'].get('carbon_market', {}).get('market_tightness', 0),
                    'carbon_price_change': round_results['market_results'].get('carbon_market', {}).get('price_change', 0),
                }
                
                # 添加各智能体的详细结果
                for agent_id, agent_result in round_results['agent_results'].items():
                    bid = agent_result.get('submitted_bid', {})
                    energy_bid = bid.get('energy_bid', {})
                    reserve_bid = bid.get('reserve_bid', {})
                    carbon_bid = bid.get('carbon', {})
                    
                    # 电力市场数据
                    round_record[f"{agent_id}_energy_bid_price"] = energy_bid.get('price', 0)
                    round_record[f"{agent_id}_energy_bid_quantity"] = energy_bid.get('quantity', 0)
                    round_record[f"{agent_id}_reserve_bid_price"] = reserve_bid.get('price', 0)
                    round_record[f"{agent_id}_reserve_bid_quantity"] = reserve_bid.get('quantity', 0)
                    cleared_energy = agent_result.get('cleared_energy', 0)
                    round_record[f"{agent_id}_cleared_energy"] = cleared_energy
                    round_record[f"{agent_id}_cleared_reserve"] = agent_result.get('cleared_reserve', 0)
                    round_record[f"{agent_id}_profit"] = agent_result.get('profit', 0)

                    # 写入碳市场分析所需的新数据
                    round_record[f"{agent_id}_dispatch_energy"] = cleared_energy # 实际发电量就是出清的能量
                    round_record[f"{agent_id}_emissions"] = agent_result.get('emissions', 0) # 从agent_outcomes获取实际排放
                    
                    # 碳市场数据
                    carbon_result = agent_result.get('carbon', {}) or agent_result.get('carbon_result', {})
                    round_record[f"{agent_id}_carbon_cost"] = carbon_result.get('cost', 0)
                    round_record[f"{agent_id}_carbon_quota"] = agent_result.get('initial_quota', 0) # 改为记录初始配额，更有分析价值
                    round_record[f"{agent_id}_carbon_funds"] = carbon_result.get('final_funds', 0)
                    
                    # 碳市场订单数据（转换为字符串格式）
                    submitted_carbon = agent_result.get('submitted_carbon_bid', {})
                    buy_orders = submitted_carbon.get('buy_orders', [])
                    sell_orders = submitted_carbon.get('sell_orders', [])
                    round_record[f"{agent_id}_carbon_buy_orders"] = len(buy_orders)
                    round_record[f"{agent_id}_carbon_sell_orders"] = len(sell_orders)
                    
                    # ADF决策数据
                    decision_info = agent_result.get('decision_info', {})
                    round_record[f"{agent_id}_decision_type"] = decision_info.get('decision_type', 'unknown')
                    round_record[f"{agent_id}_llm_trigger_reason"] = decision_info.get('trigger_reason', '')
                    
                    # Token使用数据
                    token_info = agent_result.get('token_usage', {})
                    round_record[f"{agent_id}_total_tokens"] = token_info.get('total_tokens', 0)
                    round_record[f"{agent_id}_prompt_tokens"] = token_info.get('prompt_tokens', 0)
                    round_record[f"{agent_id}_completion_tokens"] = token_info.get('completion_tokens', 0)
                    
                    # 更新智能体结果统计
                    profit = agent_result.get('profit', 0)
                    self.results[agent_id]['rounds_data'].append(agent_result)
                    self.results[agent_id]['total_profit'] += profit
                    self.results[agent_id]['max_profit'] = max(self.results[agent_id]['max_profit'], profit)
                    self.results[agent_id]['min_profit'] = min(self.results[agent_id]['min_profit'], profit)
                
                writer.writerow(round_record)
                csvfile.flush()  # 确保实时写入
        
        # 计算平均利润
        for agent_id, result in self.results.items():
            result['avg_profit'] = result['total_profit'] / self.num_rounds
        
        elapsed_time = time.time() - start_time
        logging.info(f"仿真完成，总耗时: {elapsed_time:.2f} 秒")
        
        return self.results
    
    def _run_round(self, round_number: int) -> Dict[str, Any]:
        """
        运行单轮仿真。
        
        Args:
            round_number: 当前轮次
            
        Returns:
            本轮的结果字典
        """
        logging.info(f"\n===== 开始第 {round_number} 轮仿真 =====")
        
        # 获取市场信息
        market_info = self.market_environment.get_market_info(
            round_number=round_number,
            num_competitors=len(self.agents) - 1  # 竞争对手数量
        )
        logging.info(f"市场需求: 能源 {market_info['total_demand']:.2f} MW, 备用 {market_info['reserve_requirement']:.2f} MW")
        
        
        # 收集所有智能体的报价决策 (并行化优化)
        def make_agent_decision(agent_id, agent):
            """单个智能体决策的包装函数,用于并行执行"""
            try:
                # 获取电碳耦合报价（若不支持则回退到仅电力报价）
                if hasattr(agent, "decide_coupled_bid"):
                    cm_info = self.carbon_market.get_market_info()
                    cm_info = {**cm_info, 'current_quota': float(self.carbon_market.agent_quotas.get(agent_id, 0.0))}
                    combined_info = {
                        'electricity_market': market_info,
                        'carbon_market': cm_info
                    }
                    actions = agent.decide_coupled_bid(round_number, combined_info)
                    elec_bid = actions.get('electricity', {})
                else:
                    elec_bid = agent.decide_bid(round_number, market_info)
                    actions = {'electricity': elec_bid, 'carbon': {}}
                
                decision_time = time.time()
                logging.info(f"智能体 {agent_id} 决策完成")
                
                # 验证报价
                if not self._validate_bid(agent_id, elec_bid, agent.get_private_info()):
                    logging.warning(f"智能体 {agent_id} 提交了无效报价，将跳过")
                    return {'success': False, 'agent_id': agent_id, 'error': '报价无效'}
                
                return {
                    'agent_id': agent_id,
                    'actions': actions,
                    'success': True
                }
            except Exception as e:
                return {
                    'agent_id': agent_id,
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # 使用通用的并行执行方法
        agent_actions = self._execute_agents_parallel(make_agent_decision, max_workers=10, task_desc="报价决策")


        
        # 如果没有有效报价，返回空结果
        if not agent_actions:
            logging.error("本轮没有有效报价，无法进行市场出清")
            return {"market_results": {}, "agent_results": {}}
        
        # 获取所有智能体的私有信息(用于市场结算)
        agent_private_info = {
            agent_id: agent.get_private_info() 
            for agent_id, agent in self.agents.items()
        }
        
        # 执行市场出清 (调用耦合市场的电碳耦合出清)
        market_results, agent_outcomes = self.coupled_market.run_coupled_clearing(
            agent_actions=agent_actions,
            round_number=round_number,
            agent_private_info=agent_private_info
        )
        
        # 更新每个智能体的状态
        for agent_id, agent in self.agents.items():
            if agent_id in agent_outcomes:
                try:
                    agent.update_state(
                        round_number=round_number,
                        market_results=market_results,
                        my_bid_result=agent_outcomes[agent_id]
                    )
                except Exception as e:
                    logging.error(f"智能体 {agent_id} 状态更新失败: {e}", exc_info=True)
        
        logging.info(f"===== 结束第 {round_number} 轮仿真 =====\n")
        
        return {
            "market_results": market_results,
            "agent_results": agent_outcomes
        }
    
    def _validate_bid(self, agent_id: str, bid: Dict, private_info: Dict) -> bool:
        """
        验证智能体的报价是否有效。
        使用BidValidator工具类进行验证。
        
        Args:
            agent_id: 智能体ID
            bid: 智能体的报价
            private_info: 智能体的私有信息
            
        Returns:
            报价是否有效
        """
        return BidValidator.validate_bid(agent_id, bid, private_info)
        
    def _generate_agents_with_llm(self) -> List[Dict[str, Any]]:
        """
        使用LLM批量生成智能体配置。
        
        Returns:
            生成的智能体配置列表
        """
        # 从配置中获取自动生成的参数
        gen_config = self.config.get("agent_generation", {})
        num_agents = gen_config.get("num_agents", 5)  # 默认生成5个智能体
        num_batches = gen_config.get("num_batches", 1)  # 批次数，默认为1
        template_agent = gen_config.get("template_agent", None)  # 模板智能体配置
        name_prefix = gen_config.get("name_prefix", "GenCo_LLM")  # 智能体ID前缀
        perturb_range = gen_config.get("perturb_range", 0.2)  # 扰动范围（百分比）
        use_llm = gen_config.get("use_llm", True)  # 是否使用LLM生成参数
        generate_roles = gen_config.get("generate_roles", True)  # 是否生成角色描述
        
        # 如果没有提供模板智能体，使用默认模板
        if not template_agent:
            template_agent = {
                "agent_type": "LLM",
                "config": {
                    "marginal_cost_energy": 25.0,
                    "marginal_cost_reserve": 5.0,
                    "max_capacity": 100.0,
                    "min_output": 0.0,
                },
                "llm_config": {
                    "model": "gemma-3-27b-it",
                },
                "belief_config": {
                    "avg_cost_estimation_mean": 30.0,
                    "avg_cost_estimation_std": 10.0,
                    "update_aggressiveness": 0.1
                },
                "memory_config": {
                    "capacity": 100  # 增加到100轮，足够72轮仿真使用
                },
                "role_description": "一家传统火力发电企业，注重稳定收益，倾向于保守策略。" # 默认角色描述
            }
        
        # 生成智能体配置列表
        agents = []
        
        if use_llm and self.llm_client:
            # 使用LLM生成智能体参数（按批次循环）
            logging.info(f"开始使用LLM按批次生成智能体配置：每批 {num_agents} 个，共 {num_batches} 批，总计 {num_agents * num_batches} 个")
            try:
                # 构建提示词
                prompt_template = f"""
                作为一个电力市场仿真系统的辅助工具，请帮我生成 {num_agents} 个具有多样性的发电商智能体的配置参数。
                目标是模拟真实市场中不同技术类型、成本结构、容量和行为特征的发电主体。

                    请为每个智能体首先确定一个主要的技术类型（例如：大型燃煤火电、CCGT燃气联合循环、OCGT燃气调峰、大型水电、中小型水电、风电场、光伏电站等）。
                    然后，根据其技术类型和市场定位，生成以下参数。请确保参数之间逻辑合理且彼此不同：

                1.  **Agent Name**: 一个符合其角色和技术类型的名称 (例如: "华能XX电厂", "协鑫光伏", "大唐XX燃气", "三峡水电")。
                2.  **Technology Type**: 明确的技术类型字符串。
                3.  **能源生产边际成本 (marginal_cost_energy)**:
                    *   燃煤火电: 通常在 20-40 元/MWh。
                    *   CCGT燃气: 通常在 30-50 元/MWh。
                    *   OCGT燃气/调峰: 通常较高，40-70 元/MWh 或更高。
                    *   水电: 成本较低，通常 < 20 元/MWh (若有水)。
                    *   风电/光伏: 成本极低，通常 < 15 元/MWh (若有资源)。
                4.  **备用提供边际成本 (marginal_cost_reserve)**:
                    *   应低于能源成本，并反映提供备用的机会成本和灵活性。
                    *   通常是能源成本的 10%-30%。
                    *   灵活机组（如水电、燃气）相对成本较低，基荷机组（如大型燃煤）提供备用机会成本较高。新能源提供备用能力可能受限或成本模式不同（若配备储能）。
                5.  **最大总容量 (max_capacity)**:
                    *   应与技术类型和规模匹配 (例如：大型火电/水电/新能源 100-200 MW，CCGT 80-200 MW，OCGT/中小型水电/生物质 30-100 MW)。请确保总容量在 30-300 MW 范围内有多样性。
                6.  **最小发电出力 (min_output)**:
                    *   必须 <= 最大容量。
                    *   大型燃煤火电通常有较高的最小出力 (如最大容量的 20%-40%)。
                    *   CCGT燃气可能有中等最小出力 (如最大容量的 10%-30%)。
                    *   水电、OCGT燃气、风电、光伏通常为 0 MW 或非常低 (< 5% 容量)。
                7.  **初始信念均值 (belief_mean)**: 对竞争对手平均能源边际成本的初始估计值。应在所有智能体实际成本的平均值附近随机生成，以模拟不完全信息。
                8.  **初始信念标准差 (belief_std)**: 对上述估计的不确定性程度，通常是均值的10%-20%。数值越大表示越不确定。
                9.  **信念更新速率 (update_rate)**: 智能体根据市场信息更新其信念的速度，通常是一个 0 到 1 之间的小数 (例如 0.1-0.3)。(如果LLM难以生成，可默认为0.1)
                10. **角色描述 (role_description)**:
                    *   **必须**与生成的技术类型、成本、容量、最小出力等参数保持一致。
                    *   详细描述该发电商的类型、规模（大型、中型、小型）、所有制（国有、民营等）、市场定位（基荷、调峰、新能源）、风险偏好（保守、中性、激进）、决策风格（稳健、灵活、机会主义）、主要目标（稳定收益、短期利润最大化、市场份额）等，使其行为特征鲜明。

                请确保生成的 {num_agents} 个智能体在技术类型、成本、容量、灵活性（最小出力）、角色定位等方面具有显著的多样性。

                请以JSON格式返回结果，格式如下：
                {{"agents": [
                {{"agent_name": "名称", "technology_type": "技术类型",
                "marginal_cost_energy": 值, "marginal_cost_reserve": 值,
                "max_capacity": 值, "min_output": 值,
                "belief_mean": 值, "belief_std": 值, "update_rate": 值,
                "role_description": "角色描述文本"}},
                ... (重复 {num_agents} 次)
                ]}}

                请仔细检查生成的参数是否符合逻辑和要求。
                """

                # 初始化技术类型计数器（跨批次累加）
                tech_counters = {}
                
                # 按批次调用LLM并解析
                agent_index = 1  # 全局连续编号（跨批次不重置）
                for b in range(num_batches):
                    logging.info(f"正在调用LLM API生成第 {b+1}/{num_batches} 批智能体参数...")
                    response = self.llm_client.chat(prompt_template, return_json=True)
                    logging.info(f"LLM API调用完成（批次 {b+1}），响应类型: {type(response)}, 响应长度: {len(str(response)) if response else 0}")

                    # 解析LLM返回的结果
                    if response is None:
                        logging.error("LLM API调用返回None，可能是网络错误或API服务不可用")
                        llm_agents_params = None
                    elif isinstance(response, dict):
                        if "agents" in response:
                            llm_agents_params = response["agents"]
                            logging.info(f"成功从LLM响应中解析出 {len(llm_agents_params)} 个智能体配置（直接JSON格式）")
                        elif "text" in response:
                            logging.info("LLM返回text格式响应，尝试解析JSON...")
                            try:
                                text_content = response["text"]
                                json_str = None
                                json_match = re.search(r'```json\s*\n(.*?)\n```', text_content, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(1)
                                    logging.info(f"从```json代码块中提取到JSON字符串，长度: {len(json_str)}")
                                elif re.search(r'```\s*\n.*?\n```', text_content, re.DOTALL):
                                    code_match = re.search(r'```\s*\n(.*?)\n```', text_content, re.DOTALL)
                                    if code_match:
                                        json_str = code_match.group(1)
                                        logging.info(f"从```代码块中提取到JSON字符串，长度: {len(json_str)}")
                                elif '{' in text_content and '}' in text_content:
                                    start_idx = text_content.find('{')
                                    end_idx = text_content.rfind('}')
                                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                        json_str = text_content[start_idx:end_idx+1]
                                        logging.info(f"从text中提取到JSON对象，长度: {len(json_str)}")
                                else:
                                    json_str = text_content.strip()
                                    logging.info(f"尝试直接解析整个text内容，长度: {len(json_str)}")

                                if json_str:
                                    try:
                                        parsed_json = json.loads(json_str)
                                        if "agents" in parsed_json:
                                            llm_agents_params = parsed_json["agents"]
                                            logging.info(f"✅ 成功从text格式中解析出 {len(llm_agents_params)} 个智能体配置")
                                        else:
                                            logging.warning("解析的JSON中没有找到'agents'键")
                                            llm_agents_params = None
                                    except json.JSONDecodeError as e:
                                        logging.warning(f"JSON解析失败: {e}")
                                        llm_agents_params = None
                                else:
                                    logging.warning("无法从text内容中提取JSON")
                                    llm_agents_params = None
                            except (json.JSONDecodeError, AttributeError) as e:
                                logging.warning(f"解析LLM返回的JSON时出错: {e}")
                                llm_agents_params = None
                        else:
                            logging.warning(f"LLM响应格式不支持，响应键: {list(response.keys()) if isinstance(response, dict) else 'N/A'}")
                            llm_agents_params = None
                    else:
                        logging.error(f"LLM返回了非字典类型的响应: {type(response)}")
                        llm_agents_params = None

                    if llm_agents_params is not None:
                        # 使用LLM生成的参数创建智能体配置（当前批次）
                        for i, params in enumerate(llm_agents_params[:num_agents]):
                            agent = {}
                            
                            # 生成基于技术类型的ID
                            tech_type = params.get("technology_type", "未指定")
                            tech_key = self._get_tech_key(tech_type)
                            if tech_key not in tech_counters:
                                tech_counters[tech_key] = 0
                            tech_counters[tech_key] += 1
                            agent["agent_id"] = f"{tech_key}-{tech_counters[tech_key]}"
                            
                            agent["agent_type"] = template_agent["agent_type"]

                            agent["config"] = {
                                "marginal_cost_energy": params.get("marginal_cost_energy", template_agent["config"]["marginal_cost_energy"]),
                                "marginal_cost_reserve": params.get("marginal_cost_reserve", template_agent["config"]["marginal_cost_reserve"]),
                                "max_capacity": params.get("max_capacity", template_agent["config"]["max_capacity"]),
                                "min_output": params.get("min_output", template_agent["config"]["min_output"]),
                                "technology_type": tech_type
                            }

                            if "emission_factor" in template_agent["config"]:
                                agent["config"]["emission_factor"] = params.get("emission_factor", template_agent["config"]["emission_factor"])

                            agent["llm_config"] = dict(template_agent.get("llm_config", {}))

                            agent["belief_config"] = {
                                "avg_cost_estimation_mean": params.get("belief_mean", template_agent["belief_config"]["avg_cost_estimation_mean"]),
                                "avg_cost_estimation_std": params.get("belief_std", template_agent["belief_config"]["avg_cost_estimation_std"]),
                                "update_aggressiveness": params.get("update_rate", template_agent["belief_config"]["update_aggressiveness"])
                            }

                            agent["memory_config"] = dict(template_agent.get("memory_config", {}))
                            agent["role_description"] = params.get("role_description", template_agent.get("role_description", "一家普通的发电企业"))

                            agents.append(agent)
                            agent_index += 1

                    else:
                        logging.warning("❌ LLM解析失败，本批次将跳过并继续下一批或最终回退到随机生成")

                if len(agents) > 0:
                    # 打印每个智能体的配置信息
                    logging.info("===== 生成的智能体配置信息（LLM） =====")
                    for i, agent in enumerate(agents):
                        logging.info(f"智能体 {i+1}: {agent['agent_id']}")
                        logging.info(f"  类型: {agent['agent_type']}")
                        logging.info(f"  能源边际成本: {agent['config']['marginal_cost_energy']:.2f} 元/MWh")
                        logging.info(f"  备用边际成本: {agent['config']['marginal_cost_reserve']:.2f} 元/MWh")
                        logging.info(f"  最大容量: {agent['config']['max_capacity']:.2f} MW")
                        logging.info(f"  最小出力: {agent['config']['min_output']:.2f} MW")
                        logging.info(f"  角色描述: {agent['role_description']}")
                        logging.info("----------------------------")

                    logging.info(f"✅ 成功使用LLM生成了 {len(agents)} 个智能体配置（共 {num_batches} 批）")
                    return agents
                else:
                    logging.warning("❌ 所有批次的LLM解析均失败，将使用随机生成方法作为fallback")
            except Exception as e:
                logging.error(f"❌ 使用LLM生成智能体配置时出错: {type(e).__name__}: {e}")
                logging.error("将使用随机生成方法作为fallback")
        else:
            if not use_llm:
                logging.info("配置中禁用了LLM生成，将使用随机生成方法")
            elif not self.llm_client:
                logging.warning("LLM客户端未初始化，将使用随机生成方法")
        
        # 如果LLM生成失败或未启用，使用随机生成方法
        total_agents = num_agents * num_batches
        logging.info(f"🔄 开始使用随机扰动方法生成 {total_agents} 个智能体配置（每批 {num_agents}，共 {num_batches} 批）...")
        
        # 预定义的角色描述列表，用于随机分配
        # 定义技术类型和对应的角色描述
        tech_types_and_roles = [
            # 燃煤电厂类型
            ("燃煤火电", "一家大型国有火力发电集团，拥有丰富的市场经验，倾向于稳健策略，注重长期收益。"),
            ("天然气发电", "一家中型民营燃气发电企业，灵活性高，愿意承担适度风险以获取更高利润。"),
            ("风力发电", "一家新兴的可再生能源公司，主要经营风电和光伏，边际成本低，但出力不稳定，倾向于积极竞价。"),
            ("水力发电", "一家老牌水力发电企业，发电成本低，但受水资源限制，策略较为保守。"),
            ("综合能源", "一家综合性能源公司，同时拥有火电和新能源资产，决策平衡且灵活。"),
            ("燃气发电", "一家小型独立燃气发电商，资源有限但决策灵活，倾向于寻找市场机会。"),
            ("燃煤发电", "一家以煤电为主的传统发电企业，成本结构较高，但运行稳定可靠。"),
            ("天然气联合循环", "一家新进入市场的外资能源公司，技术先进，愿意采取创新策略。")
        ]
        
        
        # 如果没有传入tech_counters，则初始化（支持跨批次累加）
        if not hasattr(self, '_global_tech_counters'):
            self._global_tech_counters = {}
        tech_counters = self._global_tech_counters
        
        for i in range(total_agents):
            agent = copy.deepcopy(template_agent)  # 深拷贝模板智能体
            
            # 分配技术类型和角色描述
            tech_type, role_desc = tech_types_and_roles[i % len(tech_types_and_roles)]
            agent["config"]["technology_type"] = tech_type  # 添加技术类型字段
            agent["role_description"] = role_desc
            
            # 生成基于技术类型的ID
            tech_key = self._get_tech_key(tech_type)
            if tech_key not in tech_counters:
                tech_counters[tech_key] = 0
            tech_counters[tech_key] += 1
            agent["agent_id"] = f"{tech_key}-{tech_counters[tech_key]}"

            # 根据技术类型调整成本参数
            if "天然气" in tech_type or "燃气" in tech_type:
                # 燃气机组：中等成本，高灵活性
                agent["config"]["marginal_cost_energy"] *= random.uniform(1.5, 2.0)  # 燃气成本较高
                agent["config"]["min_output"] *= random.uniform(0.3, 0.5)  # 最小出力较低（灵活性高）
            elif "燃煤" in tech_type or "火电" in tech_type:
                # 燃煤机组：低成本，低灵活性
                agent["config"]["marginal_cost_energy"] *= random.uniform(0.8, 1.2)  # 燃煤成本较低
                agent["config"]["min_output"] *= random.uniform(0.6, 0.8)  # 最小出力较高（灵活性低）
            elif "水力" in tech_type or "水电" in tech_type:
                # 水电：极低成本，中等灵活性
                agent["config"]["marginal_cost_energy"] *= random.uniform(0.3, 0.6)  # 水电成本很低
            elif "风力" in tech_type or "风电" in tech_type:
                # 风电：极低成本，但不稳定
                agent["config"]["marginal_cost_energy"] *= random.uniform(0.2, 0.4)  # 风电成本很低
                agent["config"]["min_output"] *= random.uniform(0.1, 0.3)  # 最小出力很低

            # 为其他数值参数添加随机扰动
            for key, value in agent["config"].items():
                if key != "technology_type" and isinstance(value, (int, float)) and value != 0:
                    perturb = random.uniform(-perturb_range, perturb_range)
                    agent["config"][key] = value * (1 + perturb)

            # 为信念参数添加随机扰动
            for key, value in agent["belief_config"].items():
                if isinstance(value, (int, float)) and value != 0:
                    perturb = random.uniform(-perturb_range/2, perturb_range/2)
                    agent["belief_config"][key] = value * (1 + perturb)
            
            agents.append(agent)
        
        # 打印每个智能体的配置信息
        logging.info("===== 生成的智能体配置信息 =====")
        for i, agent in enumerate(agents):
            logging.info(f"智能体 {i+1}: {agent['agent_id']}")
            logging.info(f"  类型: {agent['agent_type']}")
            logging.info(f"  技术类型: {agent['config'].get('technology_type', '未指定')}")
            logging.info(f"  能源边际成本: {agent['config']['marginal_cost_energy']:.2f} 元/MWh")
            logging.info(f"  备用边际成本: {agent['config']['marginal_cost_reserve']:.2f} 元/MWh")
            logging.info(f"  最大容量: {agent['config']['max_capacity']:.2f} MW")
            logging.info(f"  最小出力: {agent['config']['min_output']:.2f} MW")
            logging.info(f"  角色描述: {agent['role_description']}")
            logging.info("----------------------------")
        
        logging.info(f"✅ 使用随机扰动方法生成了 {len(agents)} 个智能体配置")
        return agents
    
    def _get_tech_key(self, tech_type):
        """将技术类型转换为简短英文标识"""
        if not tech_type:
            return "other"
        
        tech_type_lower = tech_type.lower()
        
        # 更全面的技术类型映射，支持LLM生成的各种表述
        if any(keyword in tech_type_lower for keyword in ["燃煤", "煤电", "火电", "coal"]):
            return "coal"
        elif any(keyword in tech_type_lower for keyword in ["燃气", "天然气", "gas", "ccgt", "ocgt", "联合循环"]):
            return "gas"
        elif any(keyword in tech_type_lower for keyword in ["水电", "水力", "hydro", "水电站"]):
            return "hydro"
        elif any(keyword in tech_type_lower for keyword in ["风电", "风力", "wind", "风电场"]):
            return "wind"
        elif any(keyword in tech_type_lower for keyword in ["光伏", "太阳能", "solar", "pv", "光电"]):
            return "solar"
        elif any(keyword in tech_type_lower for keyword in ["核电", "nuclear", "核能"]):
            return "nuclear"
        elif any(keyword in tech_type_lower for keyword in ["生物质", "biomass", "生物能"]):
            return "biomass"
        elif any(keyword in tech_type_lower for keyword in ["抽水蓄能", "储能", "storage"]):
            return "storage"
        else:
            return "other"
    
    def _execute_agents_parallel(self, decision_func, max_workers=10, task_desc="决策"):
        """
        通用的并行执行智能体决策的方法
        
        Args:
            decision_func: 决策函数,接收(agent_id, agent)参数,返回{success, agent_id, result/error}
            max_workers: 最大并发线程数
            task_desc: 任务描述,用于日志
        
        Returns:
            成功执行的结果字典 {agent_id: result}
        """
        max_workers = min(len(self.agents), max_workers)
        results = {}
        
        logging.info(f"🚀 使用 {max_workers} 个并发线程执行 {len(self.agents)} 个智能体的{task_desc}...")
        parallel_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_agent = {
                executor.submit(decision_func, agent_id, agent): agent_id
                for agent_id, agent in self.agents.items()
            }
            
            # 收集结果
            for future in as_completed(future_to_agent):
                result = future.result()
                
                if result.get('success'):
                    agent_id = result['agent_id']
                    results[agent_id] = result.get('result') or result.get('action') or result.get('actions')
                else:
                    logging.error(f"智能体 {result.get('agent_id', 'unknown')} {task_desc}失败: {result.get('error')}")
                    if 'traceback' in result:
                        logging.debug(f"错误堆栈:\n{result['traceback']}")
        
        parallel_time = time.time() - parallel_start_time
        logging.info(f"✅ 所有智能体并行{task_desc}完成，总耗时: {parallel_time:.2f} 秒")
        
        return results

    def run_coupled(self) -> Dict[str, Dict[str, Any]]:
        """
        电碳市场耦合仿真主流程
        """
        start_time = time.time()
        
        # 确保每个智能体都有technology_type字段
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'update_technology_type'):
                agent.update_technology_type()
        
        # 重新初始化行为分析器，确保正确分类
        self.behavior_analyzer = BehaviorAnalyzer()
        self.behavior_analyzer.classify_agents(list(self.agents.values()))
        logging.info("已重新初始化行为分析器并分类智能体")
        
        # 可选：仿真开始前的RL预训练（几十步），与在线微调分开
        try:
            online_cfg = self.config.get("online_rl", {})
            pretrain_steps = int(online_cfg.get("pretrain_timesteps", 0)) if online_cfg.get("enabled") else 0
            if pretrain_steps > 0:
                rl_agent_ids = [aid for aid, a in self.agents.items() if isinstance(a, RLAgent)]
                if rl_agent_ids:
                    training_agent_id = rl_agent_ids[0]
                    rl_agent_obj = self.agents[training_agent_id]
                    logging.info(f"准备对RL智能体 {training_agent_id} 进行仿真前预训练，步数={pretrain_steps}")
                    from train_rl_agent import MarketEnv
                    opponents = {aid: a for aid, a in self.agents.items() if aid != training_agent_id}
                    env = MarketEnv(config=self.config, training_agent_id=training_agent_id, opponent_agents=opponents)
                    # 如果模型不存在，则新建一个PPO模型再训练
                    if getattr(rl_agent_obj, 'model', None) is None:
                        from stable_baselines3 import PPO
                        try:
                            import torch
                            device = (
                                "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                                else ("cuda" if torch.cuda.is_available() else "cpu")
                            )
                        except Exception:
                            device = "cpu"
                        rl_agent_obj.model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./rl_tensorboard_logs/", device=device)
                        logging.info(f"未检测到预训练模型，已新建PPO模型(device={device})用于预训练。")
                    else:
                        # 复用现有模型
                        rl_agent_obj.model.set_env(env)
                    logging.info(f"开始预训练RL智能体 {training_agent_id}...")
                    rl_agent_obj.model.learn(total_timesteps=pretrain_steps, reset_num_timesteps=True, progress_bar=False)
                    # 保存预训练模型（如配置提供）
                    pretrain_save_path = online_cfg.get("pretrain_save_path")
                    if isinstance(pretrain_save_path, str) and pretrain_save_path:
                        os.makedirs(os.path.dirname(pretrain_save_path), exist_ok=True)
                        rl_agent_obj.model.save(pretrain_save_path)
                        logging.info(f"预训练完成，模型已保存到: {pretrain_save_path}")
                        # 广播预训练模型到所有RL智能体，使其在仿真开始即使用相同的最佳策略
                        try:
                            from stable_baselines3 import PPO as _PPO
                            for aid, a in self.agents.items():
                                if isinstance(a, RLAgent):
                                    if aid == training_agent_id:
                                        # 训练主体已拥有最新模型，可跳过或从文件重新加载以确保一致
                                        continue
                                    a.model = _PPO.load(pretrain_save_path)
                            logging.info("已将预训练模型广播到所有RL智能体。")
                        except Exception as be:
                            logging.error(f"广播预训练模型失败: {be}", exc_info=True)
                else:
                    logging.warning("预训练已启用，但未发现RL智能体，跳过")
        except Exception as e:
            logging.error(f"仿真前预训练失败: {e}", exc_info=True)

        for agent_id in self.agents.keys():
            self.results[agent_id] = {
                "rounds_data": [],
                "total_profit": 0.0,
                "avg_profit": 0.0,
                "max_profit": float('-inf'),
                "min_profit": float('inf')
            }

        # 准备CSV结果文件
        with open(self.results_file, 'w', newline='', encoding='utf-8') as csvfile:
            # 定义CSV文件的表头
            fieldnames = [
                'round', 'timestamp', 'energy_price', 'reserve_price', 
                'carbon_market_price', 'carbon_clearing_price', 'carbon_total_volume', 
                'carbon_num_trades', 'carbon_total_buy_demand', 'carbon_total_sell_supply',
                'carbon_external_supply', 'carbon_price_change', 'carbon_market_tightness',
                'energy_demand', 'reserve_demand'
            ]
            for agent_id in self.agents.keys():
                agent_fields = [
                    f"{agent_id}_energy_bid_price", f"{agent_id}_energy_bid_quantity",
                    f"{agent_id}_adjusted_energy_bid_price",  # 添加调整后的报价（含碳成本）
                    f"{agent_id}_reserve_bid_price", f"{agent_id}_reserve_bid_quantity",
                    f"{agent_id}_cleared_energy", f"{agent_id}_cleared_reserve",
                    f"{agent_id}_carbon_buy_quantity", f"{agent_id}_carbon_buy_price",
                    f"{agent_id}_carbon_sell_quantity", f"{agent_id}_carbon_sell_price",
                    f"{agent_id}_carbon_cost", f"{agent_id}_carbon_quota",
                    f"{agent_id}_profit", f"{agent_id}_prompt_tokens",
                    f"{agent_id}_completion_tokens", f"{agent_id}_total_tokens"
                ]
                # 添加智能体分类（技术类型）列，便于可视化分组
                agent_fields.append(f"{agent_id}_category")
                fieldnames.extend(agent_fields)
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # 新增历史记录
            history = {"market_prices": [], "awards": {aid: [] for aid in self.agents.keys()}}
            
            # 在线RL微调控制参数
            online_cfg = self.config.get("online_rl", {})
            updates_done = 0
            update_interval = int(online_cfg.get("update_interval", 0)) if online_cfg.get("enabled") else 0
            max_updates = int(online_cfg.get("max_updates", 0)) if online_cfg.get("enabled") else 0
            
            for round_number in range(self.num_rounds):
                logging.info(f"\n===== 开始第 {round_number} 轮耦合仿真 =====")

                # A. 基于真实数据的每轮动态：先调整需求、再更新燃气成本、再应用煤炭停机
                try:
                    self._apply_realdata_demand_adjustment(round_number)
                    self._apply_realdata_gas_cost(round_number)
                    self._apply_realdata_coal_outage(round_number)
                except Exception as re_e:
                    logging.error(f"真实数据动态更新失败: {re_e}", exc_info=True)

                # A.0. 应用可再生能源出力系数
                try:
                    self._apply_renewable_capacity_factors(round_number)
                except Exception as re_e:
                    logging.error(f"可再生能源出力系数应用失败: {re_e}", exc_info=True)

                # 0. 处理市场事件
                events_info = self.event_manager.process_round_events(round_number)
                market_events_info = self.event_manager.get_market_information_for_agents(round_number)

                # 更新智能体成本（如果有事件生效）
                if events_info["taking_effect"]:
                    self.event_manager.update_agent_costs(list(self.agents.values()), round_number)

                # 1. 智能体联合决策 (并行化优化)
                def make_coupled_decision(agent_id, agent):
                    """单个智能体的耦合决策包装函数,用于并行执行"""
                    try:
                        # 构造电力市场公开信息，并加入事件特征供RL观察
                        elec_info = self.market_environment.get_public_info(round_number)

                        # 事件数值特征
                        try:
                            # 已公告但未生效的燃料价格冲击
                            pending_fuel_events = [
                                e for e in getattr(self.event_manager, 'events', [])
                                if getattr(e, 'event_type', '') == 'fuel_price_shock' and getattr(e, 'announced', False) and not getattr(e, 'in_effect', False)
                            ]
                            is_event_pending = 1.0 if pending_fuel_events else 0.0
                            time_to_effect = 0.0
                            if pending_fuel_events:
                                time_to_effect = float(min(max(0, e.effect_time - round_number) for e in pending_fuel_events))
                            tech_type_str = str(agent.private_info.get('technology_type', '')).lower()
                            affected_fuel_flag = 0.0
                            for e in getattr(self.event_manager, 'events', []):
                                if getattr(e, 'event_type', '') == 'fuel_price_shock' and (getattr(e, 'announced', False) or getattr(e, 'in_effect', False)):
                                    if e.affected_fuel.lower() in tech_type_str:
                                        affected_fuel_flag = 1.0
                                        break
                            elec_info.update({
                                'is_event_pending': is_event_pending,
                                'time_to_effect': time_to_effect,
                                'affected_fuel_flag': affected_fuel_flag
                            })
                        except Exception:
                            # 防御性处理，避免事件特征计算影响主流程
                            elec_info.update({'is_event_pending': 0.0, 'time_to_effect': 0.0, 'affected_fuel_flag': 0.0})

                        market_info = {
                            'electricity_market': elec_info,
                            'carbon_market': {
                                'current_price': self.carbon_market.current_price,
                                'opening_price': self.carbon_market.opening_price,
                                'current_quota': self.carbon_market.agent_quotas.get(agent_id, 0),
                                'current_funds': self.carbon_market.agent_funds.get(agent_id, 100000.0)
                            },
                            'market_events_info': market_events_info  # 添加市场事件信息
                        }
                        
                        action = agent.decide_coupled_bid(round_number, market_info, history={
                            "market_prices": history["market_prices"],
                            "awards": history["awards"].get(agent_id, [])
                        }, market_events_info=market_events_info)
                        
                        return {
                            'agent_id': agent_id,
                            'action': action,
                            'success': True
                        }
                    except Exception as e:
                        return {
                            'agent_id': agent_id,
                            'success': False,
                            'error': str(e),
                            'traceback': traceback.format_exc()
                        }
                
                # 使用通用的并行执行方法
                agent_actions = self._execute_agents_parallel(make_coupled_decision, max_workers=10, task_desc="耦合决策")



                # 2. 联合结算
                # 获取智能体私有信息
                agent_private_info = {
                    agent_id: agent.get_private_info()
                    for agent_id, agent in self.agents.items()
                }
                market_results, agent_outcomes = self.coupled_market.run_coupled_clearing(
                    agent_actions, round_number, agent_private_info
                )

                # 更新公开信息中的"上一轮出清价"（耦合流程不经过 market_environment.run_round）
                try:
                    self.market_environment.last_market_prices = {
                        'energy_price': float(market_results.get('energy_price', 0) or 0),
                        'reserve_price': float(market_results.get('reserve_price', 0) or 0),
                    }
                except (TypeError, ValueError):
                    pass
                
                # 提取碳市场信息
                carbon_data = market_results.get('carbon_market', {})
                trades = carbon_data.get('trades', [])
                
                # 准备写入CSV的单行记录
                round_record = {
                    'round': round_number,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'energy_price': market_results.get('energy_price', 0),
                    'reserve_price': market_results.get('reserve_price', 0),
                    'carbon_market_price': carbon_data.get('current_price', 0),
                    'carbon_clearing_price': carbon_data.get('clearing_price', 0),
                    'carbon_total_volume': carbon_data.get('total_volume', 0),
                    'carbon_num_trades': len(trades),
                    'carbon_total_buy_demand': sum(t['quantity'] for t in trades if 'buyer' in t),
                    'carbon_total_sell_supply': sum(t['quantity'] for t in trades if 'seller' in t),
                    'carbon_external_supply': 0,  # 暂时设为0，需要时可添加
                    'carbon_price_change': carbon_data.get('current_price', 50) - 50,  # 相对于初始价格的变化
                    'carbon_market_tightness': 0,  # 暂时设为0，需要时可添加
                    'energy_demand': self.market_environment.demand_profile[round_number] if round_number < len(self.market_environment.demand_profile) else 0,
                    'reserve_demand': self.market_environment.reserve_profile[round_number] if round_number < len(self.market_environment.reserve_profile) else 0,
                }

                # 3. 记录结果
                market_price = market_results.get('energy_price', 0)
                history["market_prices"].append(market_price)

                for agent_id in self.agents.keys():
                    # 电力市场结果
                    elec_result = agent_outcomes.get(agent_id, {})
                    cleared_energy = elec_result.get('cleared_energy', 0)
                    cleared_reserve = elec_result.get('cleared_reserve', 0)
                    
                    # 获取智能体成本信息
                    agent_energy_cost = self.agents[agent_id].config['marginal_cost_energy']
                    agent_reserve_cost = self.agents[agent_id].config['marginal_cost_reserve']
                    
                    # 获取市场价格
                    energy_price = market_results.get('energy_price', 0)
                    reserve_price = market_results.get('reserve_price', 0)
                    
                    # 计算电力市场收入和成本
                    energy_revenue = cleared_energy * energy_price
                    reserve_revenue = cleared_reserve * reserve_price
                    energy_cost = cleared_energy * agent_energy_cost
                    reserve_cost = cleared_reserve * agent_reserve_cost
                    
                    # 电力市场利润
                    electricity_profit = (energy_revenue + reserve_revenue) - (energy_cost + reserve_cost)
                    
                    # 碳市场结果
                    carbon_result = agent_outcomes.get(agent_id, {}).get('carbon', {})
                    carbon_cost = carbon_result.get('cost', 0)
                    
                    # 总利润 = 电力市场利润 - 碳市场成本
                    profit = electricity_profit - carbon_cost
                    
                    # 更新智能体利润统计
                    self.results[agent_id]['rounds_data'].append({
                        'electricity': elec_result,
                        'carbon': carbon_result,
                        'profit': profit
                    })
                    self.results[agent_id]['total_profit'] += profit
                    self.results[agent_id]['max_profit'] = max(self.results[agent_id]['max_profit'], profit)
                    self.results[agent_id]['min_profit'] = min(self.results[agent_id]['min_profit'], profit)
                    
                    # 更新历史记录
                    history["awards"].setdefault(agent_id, []).append(cleared_energy > 0)

                    # 填充CSV记录
                    action = agent_actions.get(agent_id, {})
                    elec_bid = action.get('electricity', {})
                    energy_bid = elec_bid.get('energy_bid', {})
                    reserve_bid = elec_bid.get('reserve_bid', {})
                    carbon_bid = action.get('carbon', {})
                    
                    # 记录原始报价
                    round_record[f"{agent_id}_energy_bid_price"] = energy_bid.get('price', 0)
                    round_record[f"{agent_id}_energy_bid_quantity"] = energy_bid.get('quantity', 0)
                    
                    # 记录调整后的报价（含碳成本）
                    agent_outcome = agent_outcomes.get(agent_id, {})
                    adjusted_bid = agent_outcome.get('adjusted_bid', {})
                    adjusted_energy_bid = adjusted_bid.get('energy_bid', {})
                    adjusted_energy_price = adjusted_energy_bid.get('price', energy_bid.get('price', 0))
                    round_record[f"{agent_id}_adjusted_energy_bid_price"] = adjusted_energy_price
                    
                    round_record[f"{agent_id}_reserve_bid_price"] = reserve_bid.get('price', 0)
                    round_record[f"{agent_id}_reserve_bid_quantity"] = reserve_bid.get('quantity', 0)
                    round_record[f"{agent_id}_cleared_energy"] = cleared_energy
                    round_record[f"{agent_id}_cleared_reserve"] = elec_result.get('cleared_reserve', 0)
                    round_record[f"{agent_id}_carbon_buy_quantity"] = carbon_bid.get('buy_quantity', 0)
                    round_record[f"{agent_id}_carbon_buy_price"] = carbon_bid.get('buy_price', 0)
                    round_record[f"{agent_id}_carbon_sell_quantity"] = carbon_bid.get('sell_quantity', 0)
                    round_record[f"{agent_id}_carbon_sell_price"] = carbon_bid.get('sell_price', 0)
                    round_record[f"{agent_id}_carbon_cost"] = carbon_cost
                    round_record[f"{agent_id}_carbon_quota"] = carbon_result.get('quota', 0)
                    round_record[f"{agent_id}_profit"] = profit

                    # 添加token使用统计（LLM智能体的token_usage位于报价顶层）
                    token_usage = action.get('token_usage') or elec_bid.get('token_usage', {}) or {}
                    round_record[f"{agent_id}_prompt_tokens"] = token_usage.get('prompt_tokens', 0)
                    round_record[f"{agent_id}_completion_tokens"] = token_usage.get('completion_tokens', 0)
                    round_record[f"{agent_id}_total_tokens"] = token_usage.get('total_tokens', 0)

                    # 写入分类类别（gas/coal/hydro/...），从行为分析器读取；若无则为other
                    cat = 'other'
                    try:
                        cat = self.behavior_analyzer.agent_classifications.get(agent_id, {}).get('category', 'other')
                    except Exception:
                        cat = 'other'
                    round_record[f"{agent_id}_category"] = cat
                
                writer.writerow(round_record)
                csvfile.flush()

                # 4. 学习与记忆更新（论文 III-A 第五阶段）：把本轮出清结果反馈给LLM智能体，
                #    更新其信念（数值层）、短期工作记忆、以及ADF业绩判断所需的上一轮利润。
                #    RL/规则基线智能体使用各自固定的策略，不在此处更新。
                for agent_id, agent in self.agents.items():
                    if not isinstance(agent, LLMAgent) or agent_id not in agent_outcomes:
                        continue
                    try:
                        total_profit = self.results[agent_id]['rounds_data'][-1]['profit']
                        feedback = {**agent_outcomes[agent_id], 'profit': total_profit}
                        agent.update_state(
                            round_number=round_number,
                            market_results=market_results,
                            my_bid_result=feedback
                        )
                    except Exception as e:
                        logging.error(f"智能体 {agent_id} 状态更新失败: {e}", exc_info=True)

                # 记录行为数据
                self.behavior_analyzer.record_round_behavior(
                    round_number, agent_actions, round_record, events_info
                )

                # 可选：在线强化学习微调（边跑边训）
                try:
                    if online_cfg.get("enabled") and update_interval > 0 and (round_number + 1) % update_interval == 0 and updates_done < max_updates:
                        # 找到第一个RL智能体
                        rl_agent_ids = [aid for aid, a in self.agents.items() if isinstance(a, RLAgent)]
                        if rl_agent_ids:
                            training_agent_id = rl_agent_ids[0]
                            rl_agent_obj = self.agents[training_agent_id]
                            logging.info(f"准备对RL智能体 {training_agent_id} 执行在线微调，步数={online_cfg.get('timesteps_per_update', 0)}")
                            # 延迟导入以避免潜在循环依赖
                            from train_rl_agent import MarketEnv
                            opponents = {aid: a for aid, a in self.agents.items() if aid != training_agent_id}
                            env = MarketEnv(config=self.config, training_agent_id=training_agent_id, opponent_agents=opponents)
                            timesteps = int(online_cfg.get("timesteps_per_update", 0))
                            if timesteps > 0:
                                # 如果模型不存在，则新建一个PPO模型再训练
                                if getattr(rl_agent_obj, 'model', None) is None:
                                    from stable_baselines3 import PPO
                                    try:
                                        import torch
                                        device = (
                                            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                                            else ("cuda" if torch.cuda.is_available() else "cpu")
                                        )
                                    except Exception:
                                        device = "cpu"
                                    rl_agent_obj.model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./rl_tensorboard_logs/", device=device)
                                    logging.info(f"未检测到预训练模型，已新建PPO模型(device={device})用于在线训练。")
                                else:
                                    # 复用现有模型
                                    rl_agent_obj.model.set_env(env)
                                logging.info(f"开始在线微调RL智能体 {training_agent_id}...")
                                rl_agent_obj.model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)
                                # 保存更新后的模型
                                save_path = online_cfg.get("save_path")
                                if isinstance(save_path, str) and save_path:
                                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                    rl_agent_obj.model.save(save_path)
                                    logging.info(f"在线微调完成，模型已保存到: {save_path}")
                                    # 广播在线微调后的模型到所有RL智能体，确保保持一致的最佳策略
                                    try:
                                        from stable_baselines3 import PPO as _PPO
                                        for aid, a in self.agents.items():
                                            if isinstance(a, RLAgent):
                                                # 对训练主体也执行一次从磁盘加载，确保与落盘权重完全一致
                                                a.model = _PPO.load(save_path)
                                        logging.info("已将在线微调后的模型广播到所有RL智能体。")
                                    except Exception as be:
                                        logging.error(f"广播在线微调模型失败: {be}", exc_info=True)
                                updates_done += 1
                        else:
                            logging.warning("在线微调已启用，但未发现RL智能体，跳过")
                except Exception as e:
                    logging.error(f"在线RL微调失败: {e}", exc_info=True)

                logging.info(f"===== 结束第 {round_number} 轮耦合仿真 =====\n")

        for agent_id, result in self.results.items():
            if self.num_rounds > 0:
                result['avg_profit'] = result['total_profit'] / self.num_rounds
            else:
                result['avg_profit'] = 0

        elapsed_time = time.time() - start_time
        logging.info(f"电碳市场耦合仿真完成，总耗时: {elapsed_time:.2f} 秒")

        # 生成行为分析报告
        if self.config.get("market_events"):
            # 查找燃气价格冲击事件
            gas_price_event = None
            for event in self.config["market_events"]:
                if event.get("event_type") == "fuel_price_shock" and "gas" in event.get("affected_fuel", "").lower():
                    gas_price_event = event
                    break
            
            if gas_price_event:
                announce_time = gas_price_event["announce_time"]
                effect_time = gas_price_event["effect_time"]
                
                logging.info(f"找到燃气价格冲击事件：公告时间={announce_time}，生效时间={effect_time}")
                
                # 生成并输出行为分析报告
                behavior_report = self.behavior_analyzer.generate_behavior_report(announce_time, effect_time)
                logging.info(f"\n{behavior_report}")
                
                # 保存行为数据
                behavior_file = self.results_file.replace('.csv', '_behavior.json')
                self.behavior_analyzer.save_behavior_data(behavior_file)
                logging.info(f"行为分析数据已保存到: {behavior_file}")
            else:
                logging.warning("未找到燃气价格冲击事件，无法生成行为分析报告")
        else:
            logging.warning("配置中未定义市场事件，无法生成行为分析报告")

        return self.results