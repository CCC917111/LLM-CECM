import logging
from typing import Dict, Optional, Any
import os
import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    PPO = Any # Define PPO as Any if import fails, for type hinting

from .base_agent import BaseAgent


class RLAgent(BaseAgent):
    """
    一个由强化学习模型驱动的智能体。
    该智能体加载一个预先训练好的模型来进行决策。
    """
    model: Optional[PPO]

    def __init__(self, agent_id: str, config: Dict[str, Any], model_path: Optional[str] = None) -> None:
        """
        初始化RL智能体。

        Args:
            agent_id: 智能体ID。
            config: 智能体配置 (成本、容量等)。
            model_path: 预训练模型的路径。如果为None，将无法决策。
        """
        super().__init__(agent_id, config)

        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("Reinforcement Learning libraries (stable-baselines3, gymnasium) are not installed.")

        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = PPO.load(model_path)
                logging.info(f"RLAgent {self.agent_id} a rechargé le modèle depuis {model_path}")
            except Exception as e:
                logging.error(f"Erreur lors du chargement du modèle RL pour l'agent {self.agent_id}: {e}", exc_info=True)
                self.model = None
        else:
            logging.warning(f"Aucun chemin de modèle fourni pour RLAgent {self.agent_id} ou le chemin n'existe pas. L'agent utilisera une stratégie de repli.")

    def _construct_observation(self, market_info: Dict[str, Any]) -> np.ndarray:
        """
        根据市场信息构建模型的输入观测值。
        
        Note: The observation space must match the one used during training.
        """
        # Observation space (extended with event features):
        # [total_demand, reserve_requirement, last_energy_price, last_reserve_price,
        #  my_marginal_cost_energy, my_marginal_cost_reserve, my_max_capacity,
        #  is_event_pending, time_to_effect, affected_fuel_flag]
        is_event_pending = float(market_info.get('is_event_pending', 0))
        time_to_effect = float(market_info.get('time_to_effect', 0))
        affected_fuel_flag = float(market_info.get('affected_fuel_flag', 0))
        obs = np.array([
            market_info.get('total_demand', 0),
            market_info.get('reserve_requirement', 0),
            market_info.get('last_energy_price', 0),
            market_info.get('last_reserve_price', 0),
            self.private_info['marginal_cost_energy'],
            self.private_info['marginal_cost_reserve'],
            self.private_info['max_capacity'],
            is_event_pending,
            time_to_effect,
            affected_fuel_flag
        ], dtype=np.float32)
        return obs

    def decide_bid(self, round_number: int, market_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用预训练的RL模型生成报价。
        """
        if self.model is None:
            logging.warning(f"RLAgent {self.agent_id} ne dispose pas de modèle, il utilise la stratégie de repli (offre au coût marginal).")
            return self._fallback_bid()

        obs = self._construct_observation(market_info)
        
        # The action space should also match the one from training.
        # Example action space (4-dimensional continuous box):
        # [energy_price_multiplier, reserve_price_multiplier, energy_quantity_fraction, reserve_quantity_fraction]
        action, _states = self.model.predict(obs, deterministic=True)

        # 将标准化动作映射为可解释的报价（与训练环境保持一致）
        # 训练时动作空间：Box([0,0,0,0],[1,1,1,1])，含义：
        # [energy_price_multiplier, reserve_price_multiplier, energy_quantity_fraction, reserve_quantity_fraction]
        
        # Clamp price multipliers to be non-negative to avoid bidding below cost
        energy_price_multiplier = 1 + max(0, action[0]) # Ensures price >= cost
        reserve_price_multiplier = 1 + max(0, action[1])

        energy_price = self.private_info['marginal_cost_energy'] * energy_price_multiplier
        reserve_price = self.private_info['marginal_cost_reserve'] * reserve_price_multiplier
        
        # --- 关键修复：确保此处的动作到报价的转换逻辑与训练环境(train_rl_agent.py)中的逻辑完全一致 ---
        
        # 1. 将动作直接解释为容量的百分比
        total_capacity = self.private_info['max_capacity']
        # 使用max(0,...)来防止因为浮点数精度问题导致的微小负值
        energy_quantity = total_capacity * max(0, action[2])
        reserve_quantity = total_capacity * max(0, action[3])

        # 2. 仅在总报价量超过总容量时，才按比例缩减，与训练环境保持一致
        total_quantity_bid = energy_quantity + reserve_quantity
        if total_quantity_bid > total_capacity:
            scale = total_capacity / total_quantity_bid if total_quantity_bid > 0 else 0
            energy_quantity *= scale
            reserve_quantity *= scale

        # 统一输出格式：返回根级 energy_bid/reserve_bid（与MarketSimulator/BidValidator一致）
        return {
            "energy_bid": {"price": float(energy_price), "quantity": float(energy_quantity)},
            "reserve_bid": {"price": float(reserve_price), "quantity": float(reserve_quantity)}
        }
    
    def _fallback_bid(self) -> Dict[str, Any]:
        """ A simple fallback bid used when no model is available. """
        energy_price = self.private_info['marginal_cost_energy'] * 1.1
        reserve_price = self.private_info['marginal_cost_reserve'] * 1.1
        
        total_capacity = self.private_info['max_capacity']
        energy_quantity = total_capacity * 0.8
        reserve_quantity = total_capacity * 0.2

        # 统一输出格式（根级）
        return {
            "energy_bid": {"price": float(energy_price), "quantity": float(energy_quantity)},
            "reserve_bid": {"price": float(reserve_price), "quantity": float(reserve_quantity)}
        }

    def update(self, market_results: Dict, my_result: Dict, all_bids: Dict):
        """
        更新RL智能体的状态。
        在SB3中，学习（learn）通常是在一个单独的训练循环中完成的，
        这里我们只记录必要的信息，或者如果需要，可以在这里准备下一次的观测。
        对于一个在线学习的RL Agent，这里可能会更复杂。
        目前，我们假设模型是预训练的，主要用于推理。
        """
        # 在非训练模式下，RL Agent通常是无状态的，或者其状态更新逻辑
        # 会在专门的训练循环中处理。这里保持简单。
        pass

    def update_state(self, round_number: int, market_results: Dict, my_bid_result: Dict):
        """
        与仿真器接口保持一致的状态更新方法。
        当前推理解码阶段不维护内部学习，仅记录关键信息以便调试。
        """
        try:
            profit = my_bid_result.get('profit', None)
            cleared_e = my_bid_result.get('cleared_energy', None)
            cleared_r = my_bid_result.get('cleared_reserve', None)
            logging.debug(
                f"RLAgent {self.agent_id} round={round_number} profit={profit} cleared(E={cleared_e}, R={cleared_r})"
            )
        except Exception:
            # 保持健壮性，避免影响主流程
            pass

    # 兼容电碳耦合流程：提供电力与碳市场的联合决策接口
    def decide_coupled_bid(self, round_number: int, market_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        返回与耦合市场兼容的联合报价结构：
        {
          'electricity': { 'energy_bid': {...}, 'reserve_bid': {...} },
          'carbon': {...}
        }
        碳市场部分默认不主动下单（返回空结构，系统会处理为0数量）。
        """
        # 在耦合仿真中，market_info 形如 {'electricity_market': {...}, 'carbon_market': {...}, ...}
        # 优先从 'electricity_market' 提取电力市场公开信息以构造观测
        elec_market_info = market_info.get('electricity_market') if isinstance(market_info, dict) else None
        if isinstance(elec_market_info, dict):
            elec_bid = self.decide_bid(round_number, elec_market_info)
        else:
            # 兼容旧结构：直接将整个 market_info 传入（若缺少字段则使用默认值0）
            elec_bid = self.decide_bid(round_number, market_info if isinstance(market_info, dict) else {})
        carbon_bid: Dict[str, Any] = {}
        # --- 简单碳市场启发式：按预期排放与配额缺口买入/卖出，报价贴近做市商价带 ---
        try:
            cm = market_info.get('carbon_market', {}) if isinstance(market_info, dict) else {}
            current_price = float(cm.get('current_price', 50.0))
            opening_price = float(cm.get('opening_price', current_price))
            quota = float(cm.get('current_quota', 0.0))

            energy_qty = float(elec_bid.get('energy_bid', {}).get('quantity', 0.0))
            emission_factor = float(self.private_info.get('emission_factor', 0.8))
            expected_emissions = energy_qty * emission_factor
            deficit = expected_emissions - quota  # >0 需要买入

            band_low = 0.9 * opening_price
            band_high = 1.1 * opening_price

            carbon_bid = { 'buy_orders': [], 'sell_orders': [] }
            lot = max(5.0, min(20.0, max(self.private_info.get('max_capacity', 50.0) * 0.05, 10.0)))
            if deficit > 1e-6:
                buy_price = min(band_high, max(band_low, current_price * 1.05))
                carbon_bid['buy_orders'].append({ 'quantity': float(min(deficit, lot)), 'price': float(buy_price) })
            else:
                surplus = -deficit
                if surplus > 1e-6:
                    sell_price = max(band_low, min(band_high, current_price * 0.95))
                    carbon_bid['sell_orders'].append({ 'quantity': float(min(surplus, lot)), 'price': float(sell_price) })
        except Exception:
            carbon_bid = {}

        return {
            'electricity': elec_bid,
            'carbon': carbon_bid
        }

    def _get_observation(self, market_info: Dict) -> np.ndarray:
        """
        根据市场信息构建模型的输入观测值。
        
        Note: The observation space must match the one used during training.
        """
        # Example observation space:
        # [total_demand, reserve_requirement, last_energy_price, last_reserve_price,
        #  my_marginal_cost_energy, my_marginal_cost_reserve, my_max_capacity]
        obs = np.array([
            market_info.get('total_demand', 0),
            market_info.get('reserve_requirement', 0),
            market_info.get('last_energy_price', 0),
            market_info.get('last_reserve_price', 0),
            self.private_info['marginal_cost_energy'],
            self.private_info['marginal_cost_reserve'],
            self.private_info['max_capacity']
        ], dtype=np.float32)
        return obs 