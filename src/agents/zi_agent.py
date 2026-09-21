import random
from typing import Dict, Any

from .base_agent import BaseAgent


class ZIAgent(BaseAgent):
    """
    简单的零智能(Zero-Intelligence)对手代理，用于RL训练的陪练。
    - 价格：在边际成本基础上添加随机加成
    - 数量：按容量进行简单分配，并根据是否有备用市场进行调整
    返回根级结构：{"energy_bid": {...}, "reserve_bid": {...}}
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]) -> None:
        super().__init__(agent_id, config)
        self._rng = random.Random(42)

    def reset(self) -> None:
        # 可选：重置内部状态/随机数（这里固定种子以稳定训练）
        self._rng.seed(42)

    def decide_bid(self, round_number: int, market_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成报价（返回耦合格式）"""
        cost_e = float(self.private_info.get('marginal_cost_energy', 30.0))
        cost_r = float(self.private_info.get('marginal_cost_reserve', 5.0))
        capacity = float(self.private_info.get('max_capacity', 100.0))

        # 价格：成本的(1.0 ~ 1.5)倍，避免过低或负数
        energy_price = cost_e * (1.0 + 0.5 * self._rng.random())
        reserve_price = cost_r * (1.0 + 0.5 * self._rng.random())

        # 数量分配：基础 80% 能源 + 20% 备用，加入少量噪声
        frac_e = min(max(0.75 + 0.1 * (self._rng.random() - 0.5), 0.0), 1.0)
        frac_r = min(max(0.25 + 0.1 * (self._rng.random() - 0.5), 0.0), 1.0)

        # 若无备用市场，将备用置零
        reserve_requirement = float(market_info.get('reserve_requirement', 0.0) or 0.0)
        if reserve_requirement <= 0:
            frac_r = 0.0

        energy_quantity = capacity * frac_e
        reserve_quantity = capacity * frac_r

        # 容量约束：若总量超限则按比例缩放
        total_q = energy_quantity + reserve_quantity
        if total_q > capacity and total_q > 0:
            scale = capacity / total_q
            energy_quantity *= scale
            reserve_quantity *= scale

        # 返回简单格式（供validator验证）
        return {
            "energy_bid": {"price": float(energy_price), "quantity": float(energy_quantity)},
            "reserve_bid": {"price": float(reserve_price), "quantity": float(reserve_quantity)}
        }

    def update_state(self, round_number: int, market_results: Dict, my_bid_result: Dict):
        # 简单对手不维护状态
        pass

    def decide_coupled_bid(self, round_number: int, market_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """电碳耦合决策：在原有电力报价基础上，按排放缺口生成简易碳订单。"""
        elec_market = market_info.get('electricity_market', market_info)
        elec_bid = self.decide_bid(round_number, elec_market)

        # 读取碳市场公开信息
        cm = market_info.get('carbon_market', {}) if isinstance(market_info, dict) else {}
        current_price = float(cm.get('current_price', 50.0))
        opening_price = float(cm.get('opening_price', current_price))
        quota = float(cm.get('current_quota', 0.0))

        # 估算本轮排放
        energy_qty = float(elec_bid.get('energy_bid', {}).get('quantity', 0.0))
        ef = float(self.private_info.get('emission_factor', 0.8))
        expected_emissions = energy_qty * ef
        deficit = expected_emissions - quota

        # 价格带（相对当前价±10%）
        band_low = current_price * 0.9
        band_high = current_price * 1.1

        # 手数（容量的2%~5%，至少5吨，至多30吨）
        cap = float(self.private_info.get('max_capacity', 100.0))
        lot = max(5.0, min(30.0, cap * 0.03))

        carbon_bid: Dict[str, Any] = {'buy_orders': [], 'sell_orders': []}
        if deficit > 1e-6:
            buy_price = min(band_high, max(band_low, current_price * 1.05))
            carbon_bid['buy_orders'].append({'quantity': float(min(deficit, lot)), 'price': float(buy_price)})
        else:
            surplus = -deficit
            if surplus > 1e-6:
                sell_price = max(band_low, min(band_high, current_price * 0.95))
                carbon_bid['sell_orders'].append({'quantity': float(min(surplus, lot)), 'price': float(sell_price)})

        return {
            'electricity': elec_bid,
            'carbon': carbon_bid
        }
