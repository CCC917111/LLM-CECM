"""
增强版RL智能体
与train_rl_enhanced.py中的环境配置完全一致
观测空间: 39维 | 动作空间: 7维
"""
import logging
from typing import Dict, Optional, Any
from collections import deque
import os
import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    PPO = Any

from .base_agent import BaseAgent


class EnhancedRLAgent(BaseAgent):
    """
    增强版RL智能体，使用更复杂的观测和动作空间
    """
    model: Optional[PPO]

    def __init__(self, agent_id: str, config: Dict[str, Any], model_path: Optional[str] = None,
                 all_agents_info: Optional[list] = None) -> None:
        super().__init__(agent_id, config)

        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("需要安装 stable-baselines3 和 gymnasium")

        # 历史数据缓存
        self.price_history = deque(maxlen=5)
        self.profit_history = deque(maxlen=5)
        self.cleared_history = deque(maxlen=5)
        self.competitor_price_history = deque(maxlen=3)
        self.market_share_history = []
        
        # 保存所有智能体信息（用于计算身份特征）
        self.all_agents_info = all_agents_info or []
        
        # 加载模型
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = PPO.load(model_path)
                logging.info(f"EnhancedRLAgent {self.agent_id} 已加载模型: {model_path}")
            except Exception as e:
                logging.error(f"加载模型失败 {self.agent_id}: {e}", exc_info=True)
                self.model = None
        else:
            logging.warning(f"EnhancedRLAgent {self.agent_id} 未找到模型，将使用fallback策略")

    def _construct_observation(self, market_info: Dict[str, Any], 
                               competitor_info: Optional[Dict] = None) -> np.ndarray:
        """
        构建39维观测向量
        必须与训练环境完全一致
        """
        # 1. 基础市场信息 (7维)
        total_demand = market_info.get('total_demand', 0)
        reserve_requirement = market_info.get('reserve_requirement', 0)
        last_energy_price = market_info.get('last_energy_price', 0)
        last_reserve_price = market_info.get('last_reserve_price', 0)
        my_marginal_cost_energy = self.private_info['marginal_cost_energy']
        my_marginal_cost_reserve = self.private_info['marginal_cost_reserve']
        my_max_capacity = self.private_info['max_capacity']
        
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
        
        # 3. 自身历史表现 (10维)
        profit_trend = list(self.profit_history) if len(self.profit_history) > 0 else [0.0] * 5
        while len(profit_trend) < 5:
            profit_trend.insert(0, 0.0)
        
        cleared_trend = list(self.cleared_history) if len(self.cleared_history) > 0 else [0.0] * 5
        while len(cleared_trend) < 5:
            cleared_trend.insert(0, 0.0)
        
        self_history = profit_trend + cleared_trend
        
        # 4. 竞争对手行为 (6维)
        competitor_avg_price = []
        competitor_avg_cleared = []
        
        if len(self.competitor_price_history) > 0:
            for hist in self.competitor_price_history:
                competitor_avg_price.append(hist.get('avg_price', 0) / 200.0)
                competitor_avg_cleared.append(hist.get('avg_cleared', 0) / 1000.0)
        
        while len(competitor_avg_price) < 3:
            competitor_avg_price.insert(0, last_energy_price / 200.0)
            competitor_avg_cleared.insert(0, 0.5)
        
        competitor_obs = competitor_avg_price + competitor_avg_cleared
        
        # 5. 碳市场信息 (5维)
        carbon_price = market_info.get('carbon_price', 50.0) / 200.0
        carbon_quota = self.private_info.get('carbon_quota', 0) / 10000.0
        emission_factor = self.private_info.get('emission_factor', 0.8)
        expected_emissions = (my_max_capacity * 0.8 * emission_factor) / 10000.0
        carbon_deficit = max(0, expected_emissions - carbon_quota)
        
        carbon_obs = [carbon_price, carbon_quota, emission_factor, expected_emissions, carbon_deficit]
        
        # 6. 供需平衡 (3维)
        # 注意：在实际仿真中，我们无法获取所有智能体的容量
        # 这里使用市场信息中的估计值
        total_supply_capacity = market_info.get('total_supply_capacity', total_demand * 1.2)
        supply_demand_ratio = total_supply_capacity / max(total_demand, 1.0)
        reserve_ratio = reserve_requirement / max(total_demand, 1.0)
        congestion_indicator = 1.0 if supply_demand_ratio < 1.1 else 0.0
        
        supply_demand_obs = [
            min(supply_demand_ratio, 2.0) / 2.0,
            reserve_ratio,
            congestion_indicator
        ]
        
        # 7. 事件预警 (3维)
        is_event_pending = float(market_info.get('is_event_pending', 0))
        time_to_effect = float(market_info.get('time_to_effect', 0))
        affected_fuel_flag = float(market_info.get('affected_fuel_flag', 0))
        
        event_obs = [is_event_pending, time_to_effect, affected_fuel_flag]
        
        # 8. 身份特征 (5维) - 让RL知道"我是谁"
        if len(self.all_agents_info) > 0:
            # 计算所有智能体的成本和容量分布
            all_costs = [info.get('marginal_cost_energy', 50) for info in self.all_agents_info]
            all_capacities = [info.get('max_capacity', 100) for info in self.all_agents_info]
            
            all_costs.sort()
            all_capacities.sort()
            
            # 我的成本排名
            try:
                my_cost_percentile = all_costs.index(my_marginal_cost_energy) / max(len(all_costs) - 1, 1)
            except ValueError:
                my_cost_percentile = 0.5
            
            # 我的容量排名
            try:
                my_capacity_percentile = all_capacities.index(my_max_capacity) / max(len(all_capacities) - 1, 1)
            except ValueError:
                my_capacity_percentile = 0.5
        else:
            my_cost_percentile = 0.5
            my_capacity_percentile = 0.5
        
        # 燃料类型
        fuel_category = self.private_info.get('fuel_category', 'Unknown')
        is_coal = 1.0 if 'coal' in fuel_category.lower() else 0.0
        is_gas = 1.0 if 'gas' in fuel_category.lower() else 0.0
        is_renewable = 1.0 if any(x in fuel_category.lower() for x in ['hydro', 'wind', 'solar']) else 0.0
        
        if is_coal + is_gas + is_renewable == 0:
            is_coal = 1.0
        
        identity_obs = [
            my_cost_percentile,
            my_capacity_percentile,
            is_coal,
            is_gas,
            is_renewable
        ]
        
        # 组合所有观测（39+5=44维）
        obs = np.array(
            base_obs + price_trend + self_history + competitor_obs + 
            carbon_obs + supply_demand_obs + event_obs + identity_obs,
            dtype=np.float32
        )
        
        return obs

    def _action_to_bid(self, action: np.ndarray) -> Dict[str, Any]:
        """将7维动作转换为报价"""
        # 解析动作
        energy_price_strategy = action[0]
        reserve_price_strategy = action[1]
        energy_quantity_strategy = action[2]
        reserve_quantity_strategy = action[3]
        risk_preference = action[4]
        market_timing = action[5]
        carbon_hedging = action[6]
        
        # 价格策略
        base_energy_price = self.private_info['marginal_cost_energy']
        base_reserve_price = self.private_info['marginal_cost_reserve']
        
        # 根据风险偏好调整价格倍数（大幅扩大价格范围，允许策略性高价）
        # 必须与train_rl_enhanced.py完全一致！
        if risk_preference < 0.5:  # 保守
            energy_multiplier = 1.0 + energy_price_strategy * 3.0  # 1.0-4.0倍
            reserve_multiplier = 1.0 + reserve_price_strategy * 3.0
        else:  # 激进
            energy_multiplier = 2.0 + energy_price_strategy * 6.0  # 2.0-8.0倍
            reserve_multiplier = 2.0 + reserve_price_strategy * 6.0
        
        # 考虑历史价格趋势
        if len(self.price_history) >= 3:
            recent_prices = list(self.price_history)[-3:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / max(recent_prices[0], 0.01)
            if price_trend > 0.1:
                energy_multiplier *= 1.1
        
        energy_price = base_energy_price * energy_multiplier
        reserve_price = base_reserve_price * reserve_multiplier
        
        # 数量策略
        total_capacity = self.private_info['max_capacity']
        
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

    def decide_bid(self, round_number: int, market_info: Dict[str, Any]) -> Dict[str, Any]:
        """使用增强版RL模型生成报价（返回简单格式供validator验证）"""
        if self.model is None:
            logging.warning(f"EnhancedRLAgent {self.agent_id} 无模型，使用fallback策略")
            return self._fallback_bid()

        obs = self._construct_observation(market_info)
        action, _states = self.model.predict(obs, deterministic=True)
        
        return self._action_to_bid(action)
    
    def _fallback_bid(self) -> Dict[str, Any]:
        """Fallback策略"""
        energy_price = self.private_info['marginal_cost_energy'] * 1.15
        reserve_price = self.private_info['marginal_cost_reserve'] * 1.15
        
        total_capacity = self.private_info['max_capacity']
        energy_quantity = total_capacity * 0.7
        reserve_quantity = total_capacity * 0.2
        
        return {
            "energy_bid": {"price": float(energy_price), "quantity": float(energy_quantity)},
            "reserve_bid": {"price": float(reserve_price), "quantity": float(reserve_quantity)}
        }

    def update_state(self, round_number: int, market_results: Dict, my_bid_result: Dict):
        """更新智能体状态和历史记录"""
        try:
            # 更新历史记录
            energy_price = market_results.get('energy_price', 0)
            profit = my_bid_result.get('profit', 0)
            cleared_energy = my_bid_result.get('cleared_energy', 0)
            
            self.price_history.append(energy_price / 200.0)
            self.profit_history.append(profit / 1000.0)
            self.cleared_history.append(cleared_energy / 1000.0)
            
            # 计算市场份额
            total_demand = market_results.get('total_demand', 1.0)
            market_share = cleared_energy / max(total_demand, 1.0)
            self.market_share_history.append(market_share)
            
            logging.debug(
                f"EnhancedRLAgent {self.agent_id} round={round_number} "
                f"profit={profit:.2f} cleared={cleared_energy:.2f} share={market_share:.2%}"
            )
        except Exception as e:
            logging.warning(f"更新状态失败 {self.agent_id}: {e}")

    def decide_coupled_bid(self, round_number: int, market_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """电碳耦合市场决策"""
        elec_market_info = market_info.get('electricity_market') if isinstance(market_info, dict) else None
        if isinstance(elec_market_info, dict):
            elec_bid = self.decide_bid(round_number, elec_market_info)
        else:
            elec_bid = self.decide_bid(round_number, market_info if isinstance(market_info, dict) else {})
        
        # 简单的碳市场策略
        carbon_bid: Dict[str, Any] = {}
        try:
            cm = market_info.get('carbon_market', {}) if isinstance(market_info, dict) else {}
            current_price = float(cm.get('current_price', 50.0))
            opening_price = float(cm.get('opening_price', current_price))
            quota = float(cm.get('current_quota', 0.0))
            
            energy_qty = float(elec_bid.get('energy_bid', {}).get('quantity', 0.0))
            emission_factor = float(self.private_info.get('emission_factor', 0.8))
            expected_emissions = energy_qty * emission_factor
            deficit = expected_emissions - quota
            
            band_low = 0.9 * opening_price
            band_high = 1.1 * opening_price
            
            carbon_bid = {'buy_orders': [], 'sell_orders': []}
            lot = max(5.0, min(20.0, max(self.private_info.get('max_capacity', 50.0) * 0.05, 10.0)))
            
            if deficit > 1e-6:
                buy_price = min(band_high, max(band_low, current_price * 1.05))
                carbon_bid['buy_orders'].append({'quantity': float(min(deficit, lot)), 'price': float(buy_price)})
            else:
                surplus = -deficit
                if surplus > 1e-6:
                    sell_price = max(band_low, min(band_high, current_price * 0.95))
                    carbon_bid['sell_orders'].append({'quantity': float(min(surplus, lot)), 'price': float(sell_price)})
        except Exception:
            carbon_bid = {}
        
        return {
            'electricity': elec_bid,
            'carbon': carbon_bid
        }
