"""
市场事件管理器
负责处理市场冲击事件的公告、生效和影响
"""

import logging
from typing import Dict, List, Any, Optional

class MarketEvent:
    """市场事件类"""
    
    def __init__(self, event_config: Dict[str, Any]):
        self.announce_time = event_config["announce_time"]
        self.effect_time = event_config["effect_time"]
        self.event_type = event_config["event_type"]
        self.affected_fuel = event_config.get("affected_fuel", "")
        self.price_multiplier = event_config.get("price_multiplier", 1.0)
        self.description = event_config.get("description", "")
        self.announcement = event_config.get("announcement", "")
        
        # 容量停机事件特有参数
        self.outage_mw = event_config.get("outage_mw", 0)
        self.carbon_price_multiplier = event_config.get("carbon_price_multiplier", 1.0)
        
        # 供应紧急状态事件参数
        self.supply_shortage_factor = event_config.get("supply_shortage_factor", 0.0)
        
        # 事件状态
        self.announced = False
        self.in_effect = False
        
    def should_announce(self, current_round: int) -> bool:
        """检查是否应该在当前轮次公告"""
        return current_round == self.announce_time and not self.announced
    
    def should_take_effect(self, current_round: int) -> bool:
        """检查是否应该在当前轮次生效"""
        return current_round == self.effect_time and not self.in_effect
    
    def is_announced_but_not_effective(self, current_round: int) -> bool:
        """检查事件是否已公告但未生效（预期期间）"""
        return (self.announced and 
                current_round >= self.announce_time and 
                current_round < self.effect_time)

class EventManager:
    """事件管理器"""
    
    def __init__(self, event_configs: List[Dict[str, Any]]):
        self.events = [MarketEvent(config) for config in event_configs]
        self.announced_events = []  # 已公告但未生效的事件
        self.active_events = []     # 已生效的事件
        # 当由外部（如Simulator）以真实数据动态更新燃气成本时，避免重复应用燃料价格倍数
        self.disable_fuel_price_multiplier = False
        
        logging.info(f"事件管理器初始化完成，加载了 {len(self.events)} 个事件")
        for event in self.events:
            logging.info(f"  事件: {event.description} (公告时间: t={event.announce_time}, 生效时间: t={event.effect_time})")
    
    def process_round_events(self, current_round: int) -> Dict[str, Any]:
        """处理当前轮次的事件，返回事件信息"""
        round_events = {
            "new_announcements": [],
            "taking_effect": [],
            "announced_pending": [],
            "active_events": self.active_events.copy()
        }
        
        for event in self.events:
            # 检查新公告
            if event.should_announce(current_round):
                event.announced = True
                self.announced_events.append(event)
                round_events["new_announcements"].append(event)
                logging.info(f"🔔 t={current_round}: 新事件公告 - {event.announcement}")
                logging.info(f"🔔 事件详情: {event.description} (将在t={event.effect_time}生效，距离现在{event.effect_time-current_round}轮)")
            
            # 检查生效
            if event.should_take_effect(current_round):
                event.in_effect = True
                if event in self.announced_events:
                    self.announced_events.remove(event)
                self.active_events.append(event)
                round_events["taking_effect"].append(event)
                logging.info(f"⚡ t={current_round}: 事件生效 - {event.description}")
                logging.info(f"⚡ 事件详情: 公告于t={event.announce_time}，影响燃料: {event.affected_fuel}，价格倍数: {event.price_multiplier:.2f}")
            
            # 收集已公告但未生效的事件
            if event.is_announced_but_not_effective(current_round):
                round_events["announced_pending"].append(event)
                # 每5轮提醒一次未生效的事件
                if current_round % 5 == 0:
                    logging.info(f"⏳ t={current_round}: 待生效事件 - {event.description} (将在t={event.effect_time}生效，还有{event.effect_time-current_round}轮)")
        
        return round_events
    
    def get_fuel_cost_multiplier(self, fuel_type: str) -> float:
        """获取特定燃料类型的成本倍数"""
        multiplier = 1.0
        for event in self.active_events:
            if (event.event_type == "fuel_price_shock" and 
                event.affected_fuel.lower() in fuel_type.lower()):
                # 可选：当外部已按真实数据逐轮更新燃气成本时，不再叠加事件倍数
                if getattr(self, 'disable_fuel_price_multiplier', False):
                    continue
                multiplier *= event.price_multiplier
            elif (event.event_type == "carbon_price_shock" and 
                  event.affected_fuel.lower() in fuel_type.lower()):
                multiplier *= event.carbon_price_multiplier
        return multiplier
    
    def get_capacity_outage_mw(self, fuel_type: str) -> float:
        """获取特定燃料类型的停机容量 (MW)"""
        outage_mw = 0.0
        for event in self.active_events:
            if (event.event_type == "capacity_outage" and 
                event.affected_fuel.lower() in fuel_type.lower()):
                outage_mw += event.outage_mw
        return outage_mw
    
    def is_agent_affected_by_outage(self, agent_fuel_type: str, agent_capacity: float) -> bool:
        """检查智能体是否受到停机事件影响"""
        for event in self.active_events:
            if (event.event_type == "capacity_outage" and 
                event.affected_fuel.lower() in agent_fuel_type.lower()):
                return True
        return False
    
    def get_market_information_for_agents(self, current_round: int) -> str:
        """为智能体生成市场信息字符串"""
        pending_events = []
        active_events = []
        
        # 已公告但未生效的事件
        for event in self.announced_events:
            if event.is_announced_but_not_effective(current_round):
                remaining_rounds = event.effect_time - current_round
                announce_age = current_round - event.announce_time
                pending_events.append({
                    "description": event.description,
                    "announcement": event.announcement,
                    "announce_time": event.announce_time,
                    "effect_time": event.effect_time,
                    "remaining_rounds": remaining_rounds,
                    "announce_age": announce_age,
                    "affected_fuel": event.affected_fuel,
                    "price_multiplier": event.price_multiplier
                })
        
        # 已生效的事件
        for event in self.active_events:
            active_since = current_round - event.effect_time
            active_events.append({
                "description": event.description,
                "announcement": event.announcement,
                "announce_time": event.announce_time,
                "effect_time": event.effect_time,
                "active_since": active_since,
                "affected_fuel": event.affected_fuel,
                "price_multiplier": event.price_multiplier
            })
        
        if not pending_events and not active_events:
            return "📊 当前无特殊市场事件。"
        
        # 构建结构化的信息字符串
        info_parts = []
        
        # 待生效事件
        if pending_events:
            info_parts.append("📣 待生效的市场事件:")
            for evt in pending_events:
                info_parts.append(f"  • {evt['announcement']}")
                info_parts.append(f"    - 公告时间: t={evt['announce_time']} ({evt['announce_age']}轮前)")
                info_parts.append(f"    - 生效时间: t={evt['effect_time']} (还有{evt['remaining_rounds']}轮)")
                info_parts.append(f"    - 影响燃料: {evt['affected_fuel']}")
                info_parts.append(f"    - 价格影响: x{evt['price_multiplier']:.2f}")
        
        # 已生效事件
        if active_events:
            if pending_events:
                info_parts.append("")
            info_parts.append("📈 当前生效的市场事件:")
            for evt in active_events:
                info_parts.append(f"  • {evt['description']}")
                info_parts.append(f"    - 公告时间: t={evt['announce_time']}")
                info_parts.append(f"    - 生效时间: t={evt['effect_time']} ({evt['active_since']}轮前)")
                info_parts.append(f"    - 影响燃料: {evt['affected_fuel']}")
                info_parts.append(f"    - 价格影响: x{evt['price_multiplier']:.2f}")
        
        return "\n".join(info_parts)
    
    def update_agent_costs_and_capacity(self, agents: List[Any], current_round: int):
        """更新智能体的燃料成本和可用容量（基于生效的事件）"""
        updated_cost_count = 0
        outage_count = 0
        
        # 添加事件状态日志
        for event in self.events:
            if event.in_effect:
                logging.info(f"t={current_round}: 事件 '{event.description}' 已生效，影响燃料: {event.affected_fuel}")
                if event.event_type == "capacity_outage":
                    logging.info(f"  停机容量: {event.outage_mw} MW")
        
        # 统计需要停机的总容量
        outage_by_fuel = {}
        for event in self.active_events:
            if event.event_type == "capacity_outage":
                fuel_type = event.affected_fuel.lower()
                if fuel_type not in outage_by_fuel:
                    outage_by_fuel[fuel_type] = 0
                outage_by_fuel[fuel_type] += event.outage_mw
        
        # 为每种燃料类型选择停机的机组
        outage_agents = {}
        for fuel_type, total_outage_mw in outage_by_fuel.items():
            # 找到该燃料类型的所有机组
            fuel_agents = []
            for agent in agents:
                if (hasattr(agent, 'private_info') and 
                    'fuel_category' in agent.private_info and
                    fuel_type.lower() in agent.private_info['fuel_category'].lower()):
                    fuel_agents.append(agent)
                    logging.debug(f"找到匹配的{fuel_type}机组: {agent.agent_id} (fuel_category: {agent.private_info.get('fuel_category', 'N/A')})")
            
            # 按容量排序，优先停机大容量机组（模拟现实中的经济调度）
            fuel_agents.sort(key=lambda a: a.private_info.get('max_capacity', 0), reverse=True)
            
            # 选择停机机组
            remaining_outage = total_outage_mw
            selected_agents = []
            for agent in fuel_agents:
                if remaining_outage <= 0:
                    break
                capacity = agent.private_info.get('max_capacity', 0)
                selected_agents.append(agent)
                remaining_outage -= capacity
                logging.info(f"🔴 选择停机: {agent.agent_id} ({capacity:.1f} MW)")
            
            outage_agents[fuel_type] = selected_agents
        
        # 更新所有智能体
        for agent in agents:
            if hasattr(agent, 'private_info') and 'fuel_category' in agent.private_info:
                fuel_category = agent.private_info['fuel_category']
                tech_type = agent.private_info.get('technology_type', fuel_category)
                original_cost = agent.private_info.get('original_marginal_cost', 
                                                     agent.private_info['marginal_cost_energy'])
                
                # 1. 检查是否需要停机
                is_outage = False
                for fuel_type, outage_agent_list in outage_agents.items():
                    if agent in outage_agent_list:
                        # 设置停机状态
                        agent.private_info['is_outage'] = True
                        agent.private_info['original_max_capacity'] = agent.private_info.get('original_max_capacity', 
                                                                                           agent.private_info.get('max_capacity', 0))
                        agent.private_info['max_capacity'] = 0  # 停机
                        is_outage = True
                        outage_count += 1
                        logging.info(f"🔴 智能体 {agent.agent_id} ({fuel_category}) 停机生效")
                        break
                
                # 如果之前停机但现在事件结束，恢复运行
                if not is_outage and agent.private_info.get('is_outage', False):
                    original_capacity = agent.private_info.get('original_max_capacity', 0)
                    if original_capacity > 0:
                        agent.private_info['max_capacity'] = original_capacity
                        agent.private_info['is_outage'] = False
                        logging.info(f"🟢 智能体 {agent.agent_id} ({fuel_category}) 恢复运行，容量: {original_capacity:.1f} MW")
                
                # 2. 更新燃料成本 (基于fuel_category)
                cost_multiplier = self.get_fuel_cost_multiplier(fuel_category)
                
                # 为燃气机组添加额外日志
                if "natural_gas" in fuel_category.lower() or "gas" in fuel_category.lower():
                    logging.info(f"t={current_round}: 燃气机组 {agent.agent_id} 当前成本倍数: {cost_multiplier:.2f}, "
                               f"原始成本: {original_cost:.2f}, 当前成本: {agent.private_info['marginal_cost_energy']:.2f}")
                
                if cost_multiplier != 1.0:
                    # 保存原始成本（如果还没保存）
                    if 'original_marginal_cost' not in agent.private_info:
                        agent.private_info['original_marginal_cost'] = agent.private_info['marginal_cost_energy']
                    
                    # 更新成本
                    new_cost = original_cost * cost_multiplier
                    old_cost = agent.private_info['marginal_cost_energy']
                    agent.private_info['marginal_cost_energy'] = new_cost
                    
                    if abs(new_cost - old_cost) > 0.01:  # 只记录有显著变化的
                        logging.info(f"💰 智能体 {agent.agent_id} ({fuel_category}) 成本更新: "
                                   f"{old_cost:.2f} → {new_cost:.2f} 元/MWh (倍数: {cost_multiplier:.2f})")
                        updated_cost_count += 1
        
        if updated_cost_count > 0:
            logging.info(f"📊 本轮共更新了 {updated_cost_count} 个智能体的燃料成本")
        if outage_count > 0:
            logging.info(f"🔴 本轮共有 {outage_count} 个智能体停机")
    
    # 保持向后兼容
    def update_agent_costs(self, agents: List[Any], current_round: int):
        """向后兼容的方法名"""
        return self.update_agent_costs_and_capacity(agents, current_round)
