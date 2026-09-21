from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """
    智能体基类 (Abstract Base Class)。
    定义所有发电商智能体必须实现的接口。
    """
    def __init__(self, agent_id: str, config: Dict):
        """
        初始化智能体。

        Args:
            agent_id: 智能体的唯一标识符。
            config: 包含智能体私有信息和配置的字典。
                    例如: {'marginal_cost_energy': float, 'marginal_cost_reserve': float,
                           'max_capacity': float, 'min_output': float, ...}
        """
        self.agent_id = agent_id
        self.config = config
        self.private_info = {
            'marginal_cost_energy': config.get('marginal_cost_energy', 30.0),
            'marginal_cost_reserve': config.get('marginal_cost_reserve', 5.0), # 备用成本通常较低
            'max_capacity': config.get('max_capacity', 100.0),
            'min_output': config.get('min_output', 0.0),
            'emission_factor': config.get('emission_factor', 0.8),
            'fuel_category': config.get('fuel_category', ''),
            'ramp_up_mw': config.get('ramp_up_mw', 0.0),
            'ramp_down_mw': config.get('ramp_down_mw', 0.0),
            'technology_type': config.get('technology_type', '未指定')  # 添加technology_type字段
        }

    @abstractmethod
    def decide_bid(self, round_number: int, market_info: Dict) -> Dict:
        """
        智能体的核心决策逻辑，用于生成当前轮次的报价。

        Args:
            round_number: 当前市场轮次。
            market_info: 当前市场的公开信息。
                         例如: {'total_demand': float, 'reserve_requirement': float,
                                'num_competitors': int, 'last_energy_price': float, ...}

        Returns:
            一个包含能源和备用报价的字典。
            例如: {'energy_bid': {'price': P_e, 'quantity': Q_e},
                   'reserve_bid': {'price': P_r, 'quantity': Q_r}}
        """
        pass

    @abstractmethod
    def update_state(self, round_number: int, market_results: Dict, my_bid_result: Dict):
        """
        在市场出清后更新智能体的内部状态（信念、记忆等）。

        Args:
            round_number: 当前市场轮次。
            market_results: 市场的总体出清结果，包含价格等。
                            {'energy_price': float, 'reserve_price': float, ...}
            my_bid_result: 该智能体在本轮的具体中标结果和利润。
                           {'cleared_energy': float, 'cleared_reserve': float, 'profit': float, ...}
                           (注意：profit需要由外部计算后传入)
        """
        pass

    def get_id(self) -> str:
        """返回智能体ID"""
        return self.agent_id

    def get_private_info(self) -> Dict:
        """返回智能体的私有信息"""
        return self.private_info 