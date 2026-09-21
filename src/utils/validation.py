# llm_market_sim/utils/validation.py
import logging
from typing import Dict, Any, Optional

class BidValidator:
    """
    报价验证工具类，用于集中处理报价验证逻辑，避免代码重复。
    """
    
    @staticmethod
    def validate_bid(agent_id: str, bid: Dict, private_info: Dict) -> bool:
        """
        验证智能体的报价是否有效。
        
        Args:
            agent_id: 智能体ID
            bid: 智能体的报价
            private_info: 智能体的私有信息
            
        Returns:
            报价是否有效
        """
        # 检查必要字段是否存在
        if not isinstance(bid, dict):
            logging.error(f"智能体 {agent_id} 的报价不是有效的字典")
            return False
            
        if 'energy_bid' not in bid or 'reserve_bid' not in bid:
            logging.error(f"智能体 {agent_id} 的报价缺少必要字段 energy_bid 或 reserve_bid")
            return False
            
        # 提取报价信息
        energy_bid = bid.get('energy_bid', {})
        reserve_bid = bid.get('reserve_bid', {})
        
        # 检查数量和价格字段
        if 'price' not in energy_bid or 'quantity' not in energy_bid:
            logging.error(f"智能体 {agent_id} 的能源报价缺少价格或数量字段")
            return False
            
        if 'price' not in reserve_bid or 'quantity' not in reserve_bid:
            logging.error(f"智能体 {agent_id} 的备用报价缺少价格或数量字段")
            return False
            
        # 获取报价数量
        energy_quantity = energy_bid.get('quantity', 0)
        reserve_quantity = reserve_bid.get('quantity', 0)
        
        # 检查数量是否为非负数
        if energy_quantity < 0 or reserve_quantity < 0:
            logging.error(f"智能体 {agent_id} 的报价数量不能为负数")
            return False
            
        # 检查价格是否在合理范围内 (允许小幅负价格，如真实市场)
        energy_price = energy_bid.get('price', 0)
        reserve_price = reserve_bid.get('price', 0)
        
        # 允许-10到15000的价格范围 (基于真实电力市场)
        if energy_price < -10 or energy_price > 15000:
            logging.error(f"智能体 {agent_id} 的能源报价价格超出合理范围: {energy_price} 元/MWh")
            return False
            
        if reserve_price < -5 or reserve_price > 5000:
            logging.error(f"智能体 {agent_id} 的备用报价价格超出合理范围: {reserve_price} 元/MWh")
            return False
            
        # 检查总报价量是否超过容量
        max_capacity = private_info.get('max_capacity', 0)
        total_bid_quantity = energy_quantity + reserve_quantity
        
        if total_bid_quantity > max_capacity * 1.001:  # 允许0.1%的容差
            logging.error(f"智能体 {agent_id} 的总报价量 {total_bid_quantity:.2f} MW 超过最大容量 {max_capacity:.2f} MW")
            return False
            
        return True
    
    @staticmethod
    def validate_bid_for_clearing(bid: Dict, agent_id: Optional[str] = None) -> bool:
        """
        验证用于市场出清的报价是否有效。
        
        Args:
            bid: 报价字典
            agent_id: 智能体ID（可选，用于日志记录）
            
        Returns:
            报价是否有效
        """
        agent_str = f"Agent {agent_id}" if agent_id else "报价"
        
        # 基本验证，检查必要字段
        if not (bid and 'agent_id' in bid and 'energy_bid' in bid and 'reserve_bid' in bid and 'max_capacity' in bid):
            logging.warning(f"{agent_str} 提交了无效或不完整的报价，将被忽略。")
            return False
            
        # 确保价格和数量是数值
        e_price = bid['energy_bid'].get('price')
        e_quant = bid['energy_bid'].get('quantity')
        r_price = bid['reserve_bid'].get('price')
        r_quant = bid['reserve_bid'].get('quantity')
        max_cap = bid.get('max_capacity')
        
        if not all(isinstance(v, (int, float)) for v in [e_price, e_quant, r_price, r_quant, max_cap]):
            logging.warning(f"{agent_str} 报价包含非数值: E({e_price},{e_quant}), R({r_price},{r_quant}), Cap({max_cap}). 跳过。")
            return False
            
        # 确保报价量非负
        if e_quant < 0 or r_quant < 0:
            logging.warning(f"{agent_str} 报价数量为负。将数量置为0。")
            return False
            
        # 确保总报价量不超过容量
        total_bid_quant = e_quant + r_quant
        if total_bid_quant > max_cap * 1.001:  # 允许极小的误差
            logging.warning(f"{agent_str} 总报价量 {total_bid_quant:.2f} > 容量 {max_cap:.2f}. 超出容量。")
            return False
            
        return True