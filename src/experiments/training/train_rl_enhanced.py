#!/usr/bin/env python3
"""
中度增强版RL智能体训练脚本
观测空间: 39维 | 动作空间: 7维 | 奖励函数: 7个维度
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from simulation.simulator import MarketSimulator
from simulation.config import get_config
from agents.zi_agent import ZIAgent
from agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TrainingPlaceholderAgent(BaseAgent):
    def decide_bid(self, round_number: int, market_info: Dict) -> Dict:
        return {}
    def update_state(self, round_number: int, market_results: Dict, my_bid_result: Dict):
        pass


class EnhancedMarketEnv(gym.Env):
    """
    中度增强版电力市场环境
    - 观测空间: 39维 (基础7 + 历史5 + 自身10 + 竞争6 + 碳5 + 供需3 + 事件3)
    - 动作空间: 7维 (价格策略2 + 数量策略2 + 风险1 + 时机1 + 碳对冲1)
    - 奖励函数: 7个维度综合评估
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, config: Dict[str, Any], training_agent_id: str, opponent_agents: Dict[str, BaseAgent]):
        super(EnhancedMarketEnv, self).__init__()
        
        self.config = config
        self.training_agent_id = training_agent_id
        self.opponent_agents = opponent_agents
        
        # 组合所有智能体
        self.all_agents = self.opponent_agents.copy()
        training_agent_config = next(item for item in config["agents"] if item["agent_id"] == training_agent_id)
        self.all_agents[self.training_agent_id] = TrainingPlaceholderAgent(
            self.training_agent_id, training_agent_config['config']
        )
        
        self.simulator = MarketSimulator(self.config, external_agents=self.all_agents, results_dir=None)
        self._current_round = 0
        self._max_rounds = self.config.get("num_rounds", 100)
        
        # 历史数据缓存
        self.price_history = deque(maxlen=5)
        self.profit_history = deque(maxlen=5)
        self.cleared_history = deque(maxlen=5)
        self.competitor_price_history = deque(maxlen=3)  # 对手的平均价格（dict）
        self.my_bid_history = deque(maxlen=5)  # 自己的报价历史（float）
        
        # 观测空间: 44维（增加5维身份特征）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        
        # 动作空间: 7维
        self.action_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        
        # 统计信息
        self.cumulative_profit = 0
        self.cumulative_cleared = 0
        self.market_share_history = []

    def _get_observation(self) -> np.ndarray:
        """构建39维观测向量"""
        market_info = self.simulator.market_environment.get_market_info(
            self._current_round, len(self.simulator.agents) - 1
        )
        agent_private_info = self.simulator.agents[self.training_agent_id].get_private_info()
        
        # 1. 基础市场信息 (7维)
        total_demand = market_info.get('total_demand', 0)
        reserve_requirement = market_info.get('reserve_requirement', 0)
        last_energy_price = market_info.get('last_energy_price', 0)
        last_reserve_price = market_info.get('last_reserve_price', 0)
        my_marginal_cost_energy = agent_private_info['marginal_cost_energy']
        my_marginal_cost_reserve = agent_private_info['marginal_cost_reserve']
        my_max_capacity = agent_private_info['max_capacity']
        
        base_obs = [
            total_demand / 10000.0,
            reserve_requirement / 1000.0,
            last_energy_price / 200.0,
            last_reserve_price / 50.0,
            my_marginal_cost_energy / 200.0,
            my_marginal_cost_reserve / 50.0,
            my_max_capacity / 1000.0
        ]
        
        # 2. 历史价格趋势 (5维)
        price_trend = list(self.price_history) if len(self.price_history) > 0 else [0.0] * 5
        while len(price_trend) < 5:
            price_trend.insert(0, last_energy_price / 200.0)
        
        # 3. 自身历史表现 (10维: 5轮利润 + 5轮出清量)
        profit_trend = list(self.profit_history) if len(self.profit_history) > 0 else [0.0] * 5
        while len(profit_trend) < 5:
            profit_trend.insert(0, 0.0)
        
        cleared_trend = list(self.cleared_history) if len(self.cleared_history) > 0 else [0.0] * 5
        while len(cleared_trend) < 5:
            cleared_trend.insert(0, 0.0)
        
        self_history = profit_trend + cleared_trend
        
        # 4. 竞争对手行为 (6维: 3轮平均价格 + 3轮平均出清量)
        competitor_avg_price = []
        competitor_avg_cleared = []
        
        # 使用my_bid_history获取自己最近的报价
        if len(self.my_bid_history) > 0:
            recent_my_prices = list(self.my_bid_history)[-3:]
            for price in recent_my_prices:
                competitor_avg_price.append(price / 200.0)
        
        # 从cleared_history获取对手的出清情况
        if len(self.cleared_history) > 0:
            recent_cleared = list(self.cleared_history)[-3:]
            for cleared in recent_cleared:
                competitor_avg_cleared.append(cleared / 1000.0)
        
        while len(competitor_avg_price) < 3:
            competitor_avg_price.insert(0, last_energy_price / 200.0)
        while len(competitor_avg_cleared) < 3:
            competitor_avg_cleared.insert(0, 0.5)
        
        competitor_obs = competitor_avg_price[:3] + competitor_avg_cleared[:3]
        
        # 5. 碳市场信息 (5维)
        carbon_price = market_info.get('carbon_price', 50.0) / 200.0
        carbon_quota = agent_private_info.get('carbon_quota', 0) / 10000.0
        emission_factor = agent_private_info.get('emission_factor', 0.8)
        expected_emissions = (my_max_capacity * 0.8 * emission_factor) / 10000.0
        carbon_deficit = max(0, expected_emissions - carbon_quota)
        
        carbon_obs = [carbon_price, carbon_quota, emission_factor, expected_emissions, carbon_deficit]
        
        # 6. 供需平衡 (3维)
        total_supply_capacity = sum(
            a.get_private_info().get('max_capacity', 0) for a in self.simulator.agents.values()
        )
        supply_demand_ratio = total_supply_capacity / max(total_demand, 1.0)
        reserve_ratio = reserve_requirement / max(total_demand, 1.0)
        congestion_indicator = 1.0 if supply_demand_ratio < 1.1 else 0.0
        
        supply_demand_obs = [
            min(supply_demand_ratio, 2.0) / 2.0,
            reserve_ratio,
            congestion_indicator
        ]
        
        # 7. 事件预警 (3维)
        is_event_pending = 0.0
        time_to_effect = 0.0
        affected_fuel_flag = 0.0
        
        try:
            tech_type = str(agent_private_info.get('technology_type', '')).lower()
            pending_times = []
            affected_flag = 0.0
            
            for evt in self.config.get('market_events', []):
                if evt.get('event_type') == 'fuel_price_shock':
                    announce_time = int(evt.get('announce_time', 0))
                    effect_time = int(evt.get('effect_time', 0))
                    if announce_time <= self._current_round < effect_time:
                        pending_times.append(float(effect_time - self._current_round))
                        if str(evt.get('affected_fuel', '')).lower() in tech_type:
                            affected_flag = 1.0
            
            if pending_times:
                is_event_pending = 1.0
                time_to_effect = min(pending_times) / 10.0
            affected_fuel_flag = affected_flag
        except Exception:
            pass
        
        event_obs = [is_event_pending, time_to_effect, affected_fuel_flag]
        
        # 8. 身份特征 (6维) - 关键改进！让RL知道"我是谁"
        # 计算所有智能体的成本和容量分布
        all_costs = []
        all_capacities = []
        for agent in self.simulator.agents.values():
            info = agent.get_private_info()
            all_costs.append(info.get('marginal_cost_energy', 50))
            all_capacities.append(info.get('max_capacity', 100))
        
        all_costs.sort()
        all_capacities.sort()
        
        # 我的成本排名（0-1，0=最低成本，1=最高成本）
        my_cost_percentile = all_costs.index(my_marginal_cost_energy) / max(len(all_costs) - 1, 1)
        
        # 我的容量排名（0-1，0=最小容量，1=最大容量）
        my_capacity_percentile = all_capacities.index(my_max_capacity) / max(len(all_capacities) - 1, 1)
        
        # 燃料类型one-hot编码（5类：煤炭、天然气、水电、风电、太阳能）
        fuel_category = agent_private_info.get('fuel_category', 'Unknown')
        is_coal = 1.0 if 'coal' in fuel_category.lower() else 0.0
        is_gas = 1.0 if 'gas' in fuel_category.lower() else 0.0
        is_hydro = 1.0 if 'hydro' in fuel_category.lower() else 0.0
        is_wind = 1.0 if 'wind' in fuel_category.lower() else 0.0
        is_solar = 1.0 if 'solar' in fuel_category.lower() else 0.0
        
        # 如果都不是，归为其他类（用煤炭代替）
        if is_coal + is_gas + is_hydro + is_wind + is_solar == 0:
            is_coal = 1.0
        
        identity_obs = [
            my_cost_percentile,
            my_capacity_percentile,
            is_coal,
            is_gas,
            is_hydro + is_wind + is_solar  # 可再生能源合并为一类
        ]
        
        # 组合所有观测（39+5=44维）
        obs = np.array(
            base_obs + price_trend + self_history + competitor_obs + 
            carbon_obs + supply_demand_obs + event_obs + identity_obs,
            dtype=np.float32
        )
        
        return obs

    def _action_to_bid(self, action: np.ndarray) -> Dict:
        """将7维动作转换为报价"""
        agent_private_info = self.simulator.agents[self.training_agent_id].get_private_info()
        
        # 解析动作
        energy_price_strategy = action[0]
        reserve_price_strategy = action[1]
        energy_quantity_strategy = action[2]
        reserve_quantity_strategy = action[3]
        risk_preference = action[4]
        market_timing = action[5]
        carbon_hedging = action[6]
        
        # 价格策略
        base_energy_price = agent_private_info['marginal_cost_energy']
        base_reserve_price = agent_private_info['marginal_cost_reserve']
        
        # 根据风险偏好调整价格倍数（扩大价格范围，增加差异化）
        if risk_preference < 0.5:  # 保守
            energy_multiplier = 1.0 + energy_price_strategy * 1.5  # 1.0-2.5倍
            reserve_multiplier = 1.0 + reserve_price_strategy * 1.5
        else:  # 激进
            energy_multiplier = 1.5 + energy_price_strategy * 2.5  # 1.5-4.0倍
            reserve_multiplier = 1.5 + reserve_price_strategy * 2.5
        
        # 考虑历史价格趋势
        if len(self.price_history) >= 3:
            recent_prices = list(self.price_history)[-3:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / max(recent_prices[0], 0.01)
            if price_trend > 0.1:
                energy_multiplier *= 1.1
        
        energy_price = base_energy_price * energy_multiplier
        reserve_price = base_reserve_price * reserve_multiplier
        
        # 数量策略
        total_capacity = agent_private_info['max_capacity']
        
        if risk_preference < 0.5:  # 保守
            energy_fraction = 0.3 + energy_quantity_strategy * 0.4
            reserve_fraction = 0.1 + reserve_quantity_strategy * 0.3
        else:  # 激进
            energy_fraction = 0.7 + energy_quantity_strategy * 0.3
            reserve_fraction = 0.1 + reserve_quantity_strategy * 0.2
        
        energy_quantity = total_capacity * energy_fraction
        reserve_quantity = total_capacity * reserve_fraction
        
        # 确保总量不超过容量
        total_quantity_bid = energy_quantity + reserve_quantity
        if total_quantity_bid > total_capacity:
            scale = total_capacity / total_quantity_bid
            energy_quantity *= scale
            reserve_quantity *= scale
        
        return {
            "energy_bid": {"price": float(energy_price), "quantity": float(energy_quantity)},
            "reserve_bid": {"price": float(reserve_price), "quantity": float(reserve_quantity)}
        }

    def _calculate_reward(self, my_outcome: Dict, action: np.ndarray, market_results: Dict) -> float:
        """
        改进的7维度奖励函数
        重点：鼓励价格差异化和动态竞争策略
        """
        profit = my_outcome.get('profit', 0.0)
        cleared_energy = my_outcome.get('cleared_energy', 0.0)
        cleared_reserve = my_outcome.get('cleared_reserve', 0.0)
        revenue = my_outcome.get('revenue', 0.0)
        
        agent_private_info = self.simulator.agents[self.training_agent_id].get_private_info()
        max_capacity = agent_private_info.get('max_capacity', 1.0)
        marginal_cost = agent_private_info.get('marginal_cost_energy', 50.0)
        
        # 获取报价信息
        my_bid = self._action_to_bid(action)
        my_energy_price = my_bid['energy_bid']['price']
        market_energy_price = market_results.get('energy_price', my_energy_price)
        total_demand = market_results.get('total_demand', 1.0)
        
        reward = 0.0
        
        # 1. 利润奖励（权重降低到20%）
        normalized_profit = profit / 1000.0  # 归一化
        reward += normalized_profit * 0.2
        
        # 2. 市场份额奖励（权重提高到25%）
        market_share = cleared_energy / max(total_demand, 1.0)
        self.market_share_history.append(market_share)
        
        # 鼓励获得合理市场份额（不是越多越好）
        if 0.01 <= market_share <= 0.05:
            reward += 10.0 * 0.25  # 小型机组
        elif 0.05 < market_share <= 0.15:
            reward += 15.0 * 0.25  # 中型机组
        elif 0.15 < market_share <= 0.25:
            reward += 12.0 * 0.25  # 大型机组
        elif market_share > 0.25:
            reward += 5.0 * 0.25   # 过大惩罚
        
        # 3. 价格竞争力奖励（权重提高到30%）- 关键改进
        if cleared_energy > 0:
            # 鼓励报价接近但略高于市场价
            price_diff = my_energy_price - market_energy_price
            price_ratio = my_energy_price / max(market_energy_price, 1.0)
            
            # 最优：报价在市场价的0.95-1.15倍之间
            if 0.95 <= price_ratio <= 1.15:
                reward += 20.0 * 0.3
            # 次优：报价在市场价的0.85-0.95或1.15-1.3倍
            elif 0.85 <= price_ratio < 0.95 or 1.15 < price_ratio <= 1.3:
                reward += 10.0 * 0.3
            # 惩罚：报价过高（超过市场价1.5倍）
            elif price_ratio > 1.5:
                reward -= 15.0 * 0.3
            # 惩罚：报价过低（低于成本）
            elif my_energy_price < marginal_cost * 0.9:
                reward -= 10.0 * 0.3
        else:
            # 未出清时，根据报价与市场价的关系给予反馈
            price_ratio = my_energy_price / max(market_energy_price, 1.0)
            if price_ratio > 1.3:
                reward -= 8.0 * 0.3  # 报价太高导致未出清
        
        # 4. 价格动态性奖励（新增，权重10%）- 鼓励根据市场变化调价
        if len(self.price_history) >= 2:
            # 检查市场价格变化
            recent_market_prices = list(self.price_history)[-2:]
            market_price_change = recent_market_prices[-1] - recent_market_prices[0]
            
            # 检查自己的报价变化
            if len(self.my_bid_history) >= 2:
                my_prev_price = self.my_bid_history[-2] if len(self.my_bid_history) >= 2 else my_energy_price
                my_price_change = my_energy_price - my_prev_price
                
                # 如果市场价上涨，自己也上涨 -> 奖励
                if market_price_change > 5 and my_price_change > 0:
                    reward += 8.0 * 0.1
                # 如果市场价下跌，自己也下跌 -> 奖励
                elif market_price_change < -5 and my_price_change < 0:
                    reward += 8.0 * 0.1
                # 如果市场价变化，但自己不变 -> 轻微惩罚
                elif abs(market_price_change) > 10 and abs(my_price_change) < 2:
                    reward -= 3.0 * 0.1
        
        # 5. 容量利用率奖励（权重10%）
        utilization = cleared_energy / max(max_capacity, 1.0)
        if 0.5 <= utilization <= 0.95:
            reward += 10.0 * 0.1
        elif utilization > 0.95:
            reward += 5.0 * 0.1  # 满负荷运行也不错，但不是最优
        
        # 6. 收益成本比奖励（权重5%）
        if cleared_energy > 0:
            cost = marginal_cost * cleared_energy
            if cost > 0:
                profit_margin = (revenue - cost) / cost
                if profit_margin > 0.1:
                    reward += 5.0 * 0.05
                elif profit_margin > 0.3:
                    reward += 10.0 * 0.05
        
        # 7. 惩罚无效策略
        if cleared_energy == 0 and profit == 0:
            reward -= 10.0
        
        # 记录当前报价用于下一轮对比
        self.my_bid_history.append(my_energy_price)
        
        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._current_round = 0
        
        # 重置历史记录
        self.price_history.clear()
        self.profit_history.clear()
        self.cleared_history.clear()
        self.competitor_price_history.clear()
        self.my_bid_history.clear()
        self.market_share_history.clear()
        self.cumulative_profit = 0
        self.cumulative_cleared = 0
        
        # 重置市场环境
        self.simulator.market_environment.reset()
        
        # 重置对手智能体
        for agent in self.opponent_agents.values():
            if hasattr(agent, 'reset'):
                agent.reset()
        
        initial_obs = self._get_observation()
        info = {}
        
        return initial_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 1. 收集所有智能体的报价
        agent_bids = []
        competitor_prices = []
        competitor_cleared = []
        
        for agent_id, agent in self.simulator.agents.items():
            if agent_id == self.training_agent_id:
                bid = self._action_to_bid(action)
            else:
                market_info = self.simulator.market_environment.get_market_info(
                    self._current_round, len(self.simulator.agents) - 1
                )
                decision = agent.decide_bid(self._current_round, market_info)
                
                # 兼容不同返回格式
                bid = None
                if isinstance(decision, dict):
                    if 'bid' in decision and isinstance(decision['bid'], dict):
                        bid = decision['bid']
                    elif 'energy_bid' in decision and 'reserve_bid' in decision:
                        bid = {'energy_bid': decision['energy_bid'], 'reserve_bid': decision['reserve_bid']}
                
                if not isinstance(bid, dict) or 'energy_bid' not in bid or 'reserve_bid' not in bid:
                    raise ValueError(f"对手智能体 {agent_id} 返回了无效的决策结构。")
                
                # 记录竞争对手报价
                competitor_prices.append(bid['energy_bid'].get('price', 0))
            
            agent_bids.append({'agent_id': agent_id, 'bid': bid})
        
        # 2. 运行市场清算
        agent_private_info = {aid: a.get_private_info() for aid, a in self.simulator.agents.items()}
        market_results, agent_outcomes = self.simulator.market_environment.run_round(
            self._current_round, agent_bids, agent_private_info
        )
        
        # 3. 记录竞争对手出清情况
        for agent_id, outcome in agent_outcomes.items():
            if agent_id != self.training_agent_id:
                competitor_cleared.append(outcome.get('cleared_energy', 0))
        
        # 4. 计算奖励
        my_outcome = agent_outcomes.get(self.training_agent_id, {})
        reward = self._calculate_reward(my_outcome, action, market_results)
        
        # 5. 更新历史记录
        energy_price = market_results.get('energy_price', 0)
        self.price_history.append(energy_price / 200.0)
        self.profit_history.append(my_outcome.get('profit', 0) / 1000.0)
        self.cleared_history.append(my_outcome.get('cleared_energy', 0) / 1000.0)
        
        if len(competitor_prices) > 0:
            self.competitor_price_history.append({
                'avg_price': np.mean(competitor_prices),
                'avg_cleared': np.mean(competitor_cleared) if len(competitor_cleared) > 0 else 0
            })
        
        self.cumulative_profit += my_outcome.get('profit', 0)
        self.cumulative_cleared += my_outcome.get('cleared_energy', 0)
        
        # 6. 更新轮次
        self._current_round += 1
        
        # 7. 检查是否结束
        terminated = self._current_round == self._max_rounds
        truncated = False
        
        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._get_observation()
        
        # 8. 更新对手智能体状态
        for agent_id, agent in self.simulator.agents.items():
            if agent_id != self.training_agent_id and agent_id in agent_outcomes:
                agent.update_state(self._current_round - 1, market_results, agent_outcomes[agent_id])
        
        info = my_outcome
        
        return observation, reward, terminated, truncated, info


def train_enhanced_rl_agent(
    config: Dict[str, Any],
    training_agent_id: str,
    opponent_agents: Dict[str, BaseAgent],
    total_timesteps: int,
    save_path: str
) -> str:
    """训练增强版RL智能体"""
    logging.info("="*60)
    logging.info("🚀 开始训练增强版RL智能体")
    logging.info(f"   训练智能体: {training_agent_id}")
    logging.info(f"   对手数量: {len(opponent_agents)}")
    logging.info(f"   总训练步数: {total_timesteps:,}")
    logging.info(f"   观测空间: 39维")
    logging.info(f"   动作空间: 7维")
    logging.info("="*60)
    
    # 创建环境
    env = EnhancedMarketEnv(config, training_agent_id, opponent_agents)
    
    # 跳过环境检查（太慢了，因为OPF出清每轮需要1-2秒）
    logging.info("⚠️  跳过环境检查以加速训练（环境已在开发中验证）")
    # check_env(env, warn=True)  # 注释掉，节省时间
    logging.info("✅ 环境已创建")
    
    # 创建模型
    logging.info("创建PPO模型...")
    # 由于OPF环境很慢，减少n_steps从2048到512，加快训练
    # 每个episode是10轮，每轮约1.5秒，所以512步约需要13分钟收集
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,  # 减少到512，加快训练（原2048）
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # 设置评估回调
    eval_env = EnhancedMarketEnv(config, training_agent_id, opponent_agents)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path),
        log_path=os.path.dirname(save_path),
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    logging.info("▶️  开始训练...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    # 保存模型
    model.save(save_path)
    logging.info(f"✅ 模型已保存到: {save_path}")
    
    return save_path


def main():
    """主函数"""
    print("="*80)
    print("🚀 增强版RL智能体训练程序")
    print("   观测空间: 39维 | 动作空间: 7维 | 奖励函数: 7维度")
    print("="*80)
    
    # 获取配置
    config = get_config()
    config['num_rounds'] = 10
    
    # 创建对手智能体（使用ZI智能体）
    opponent_agents = {}
    for i in range(5):  # 5个对手
        agent_id = f"opponent_{i}"
        agent_config = {
            'marginal_cost_energy': 50 + i * 10,
            'marginal_cost_reserve': 10 + i * 2,
            'max_capacity': 500 + i * 100,
            'technology_type': 'coal',
            'emission_factor': 0.8
        }
        opponent_agents[agent_id] = ZIAgent(agent_id, agent_config)
        config['agents'].append({'agent_id': agent_id, 'config': agent_config})
    
    # 添加训练智能体配置
    training_agent_id = "rl_trainee"
    training_config = {
        'marginal_cost_energy': 60,
        'marginal_cost_reserve': 12,
        'max_capacity': 600,
        'technology_type': 'coal',
        'emission_factor': 0.8
    }
    config['agents'].append({'agent_id': training_agent_id, 'config': training_config})
    
    # 训练
    save_path = "models/enhanced_rl_agent"
    os.makedirs("models", exist_ok=True)
    
    train_enhanced_rl_agent(
        config=config,
        training_agent_id=training_agent_id,
        opponent_agents=opponent_agents,
        total_timesteps=100000,
        save_path=save_path
    )
    
    print("\n" + "="*80)
    print("✅ 训练完成！")
    print(f"   模型保存位置: {save_path}.zip")
    print("="*80)


if __name__ == "__main__":
    main()
