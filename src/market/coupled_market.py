from typing import Dict
from .clearing_mechanism import opf_joint_clearing
import logging

class CoupledMarket:
    def __init__(self, electricity_market, carbon_market, config: Dict):
        self.electricity_market = electricity_market
        self.carbon_market = carbon_market
        self.config = config

    def run_round(self, round_number: int, agent_bids, agent_private_info: dict):
        """
        运行一轮耦合市场出清（包装器方法，兼容simulator调用）
        agent_bids: list格式 [{'agent_id': ..., 'bid': ...}]
        bid格式: {'energy_bid': {...}, 'reserve_bid': {...}}
        """
        # 将列表格式转换为字典格式，并转换为耦合格式
        if isinstance(agent_bids, list):
            agent_actions = {}
            for item in agent_bids:
                agent_id = item['agent_id']
                bid = item['bid']
                # 转换为耦合格式
                agent_actions[agent_id] = {
                    'electricity': bid,
                    'carbon': {'buy_orders': [], 'sell_orders': []}
                }
        else:
            agent_actions = agent_bids
        
        return self.run_coupled_clearing(agent_actions, round_number, agent_private_info)

    def run_coupled_clearing(self, agent_actions, round_number: int, agent_private_info: dict):
        """
        电碳耦合市场出清
        agent_actions: {agent_id: {'electricity': {...}, 'carbon': {...}}}
        agent_private_info: {agent_id: {'max_capacity': ..., ...}}
        返回: {'carbon': ..., 'electricity': ...}
        """
        logging.debug(f"\n=== 第 {round_number} 轮耦合市场出清开始 ===")
        
        # 1. 先结算碳市场，得到碳成本和价格信息
        carbon_bids = {}
        for aid, acts in agent_actions.items():
            carbon_action = acts.get('carbon', {})
            elec_bid = acts.get('electricity', {}) or {}
            logging.debug(f"智能体 {aid} 原始碳市场行动: {carbon_action}")

            # 直接传递碳市场行动，支持新的buy_orders/sell_orders格式
            if isinstance(carbon_action, dict) and ('buy_orders' in carbon_action or 'sell_orders' in carbon_action):
                # 若两侧都为空，则生成基于需求的后备订单
                buy_orders = list(carbon_action.get('buy_orders', []) or [])
                sell_orders = list(carbon_action.get('sell_orders', []) or [])
                if not buy_orders and not sell_orders:
                    # 生成后备订单
                    max_cap = 100.0
                    ef = 0.8
                    cfg_agents = self.config.get('agents', [])
                    cfg_entry = next((a for a in cfg_agents if a.get('agent_id') == aid), None)
                    if cfg_entry:
                        max_cap = float(cfg_entry.get('config', {}).get('max_capacity', max_cap))
                        ef = float(cfg_entry.get('config', {}).get('emission_factor', ef))
                    energy_qty = float((elec_bid.get('energy_bid', {}) or {}).get('quantity', 0.0))
                    expected_emissions = energy_qty * ef
                    quota = float(self.carbon_market.agent_quotas.get(aid, 0.0))
                    deficit = expected_emissions - quota
                    lot = float(min(30.0, max(max_cap * 0.02, 5.0)))
                    # 安全获取碳价，避免None导致的错误
                    current_carbon_price = self.carbon_market.current_price if self.carbon_market.current_price is not None else 150.0
                    if deficit > 1e-6:
                        buy_price = float(current_carbon_price) * 1.05
                        buy_orders.append({'quantity': float(min(deficit, lot)), 'price': buy_price})
                    elif deficit < -1e-6:
                        surplus = -deficit
                        sell_price = float(current_carbon_price) * 0.95
                        sell_orders.append({'quantity': float(min(surplus, lot)), 'price': sell_price})
                carbon_bids[aid] = {'buy_orders': buy_orders, 'sell_orders': sell_orders}
                logging.debug(f"智能体 {aid} 使用新格式碳市场报价（含后备生成）")
            else:
                # 兼容旧格式：转换为新格式，若仍为空则生成后备订单
                buy_q = float(carbon_action.get('buy_quantity', carbon_action.get('buy', 0)) or 0)
                sell_q = float(carbon_action.get('sell_quantity', carbon_action.get('sell', 0)) or 0)
                # 安全获取碳价，避免None导致的错误
                current_carbon_price = self.carbon_market.current_price if self.carbon_market.current_price is not None else 150.0
                buy_p = float(carbon_action.get('buy_price', current_carbon_price * 1.1))
                sell_p = float(carbon_action.get('sell_price', current_carbon_price * 0.9))
                buy_orders = ([{'quantity': buy_q, 'price': buy_p}] if buy_q > 0 else [])
                sell_orders = ([{'quantity': sell_q, 'price': sell_p}] if sell_q > 0 else [])
                if not buy_orders and not sell_orders:
                    max_cap = 100.0
                    ef = 0.8
                    cfg_agents = self.config.get('agents', [])
                    cfg_entry = next((a for a in cfg_agents if a.get('agent_id') == aid), None)
                    if cfg_entry:
                        max_cap = float(cfg_entry.get('config', {}).get('max_capacity', max_cap))
                        ef = float(cfg_entry.get('config', {}).get('emission_factor', ef))
                    energy_qty = float((elec_bid.get('energy_bid', {}) or {}).get('quantity', 0.0))
                    expected_emissions = energy_qty * ef
                    quota = float(self.carbon_market.agent_quotas.get(aid, 0.0))
                    deficit = expected_emissions - quota
                    lot = float(min(30.0, max(max_cap * 0.02, 5.0)))
                    # 安全获取碳价，避免None导致的错误
                    current_carbon_price = self.carbon_market.current_price if self.carbon_market.current_price is not None else 150.0
                    if deficit > 1e-6:
                        buy_p = float(current_carbon_price) * 1.05
                        buy_orders.append({'quantity': float(min(deficit, lot)), 'price': buy_p})
                    elif deficit < -1e-6:
                        sell_p = float(current_carbon_price) * 0.95
                        sell_orders.append({'quantity': float(min(-deficit, lot)), 'price': sell_p})
                carbon_bids[aid] = {'buy_orders': buy_orders, 'sell_orders': sell_orders}
                logging.debug(f"智能体 {aid} 使用旧格式碳市场报价（含后备生成）")
        
        logging.info(f"向碳市场传递 {len(carbon_bids)} 个智能体的报价")
        carbon_results = self.carbon_market.process_bids(carbon_bids)
        
        # 提取碳市场信息
        trades = carbon_results.get('trades', [])
        carbon_clearing_price = carbon_results.get('current_price', self.carbon_market.current_price)
        carbon_new_price = carbon_results.get('current_price', self.carbon_market.current_price)
        
        # 构建智能体碳交易结果
        agent_carbon_results = {}
        for agent_id in carbon_bids.keys():
            quota_change = 0
            total_cost = 0
            
            # 计算每个智能体的碳交易结果
            for trade in trades:
                if trade['buyer'] == agent_id:
                    quota_change += trade['quantity']
                    total_cost += trade['quantity'] * trade['price']
                elif trade['seller'] == agent_id:
                    quota_change -= trade['quantity']
                    total_cost -= trade['quantity'] * trade['price']
            
            agent_carbon_results[agent_id] = {
                'quota': self.carbon_market.agent_quotas[agent_id],
                'cost': total_cost,
                'quota_change': quota_change,
                'clearing_price': carbon_clearing_price
            }
        
        logging.info(f"碳市场结算完成 - 清算价: {carbon_clearing_price:.2f}, 新市价: {carbon_new_price:.2f}")
        
        # 2. 将碳成本反馈到电力市场报价
        # 构造bids列表，并记录调整后的报价
        bids = []
        adjusted_bids_map = {}  # 记录每个智能体的调整后报价
        
        for aid, acts in agent_actions.items():
            elec_bid = acts.get('electricity', {})
            # 获取智能体配置信息
            # 首先尝试从agent_private_info获取（这包含了LLM生成的智能体信息）
            max_capacity = 100  # 默认值
            agent_config = None  # 初始化agent_config

            if aid in agent_private_info:
                max_capacity = agent_private_info[aid].get('max_capacity', 100)
            else:
                # 如果agent_private_info中没有，再尝试从配置文件获取
                private_info = self.config.get('agents', [])
                agent_config = next((a for a in private_info if a['agent_id'] == aid), None)
                if agent_config:
                    max_capacity = agent_config['config']['max_capacity']

            # 获取碳成本，调整电力报价
            carbon_cost_per_mwh = self._calculate_carbon_cost_per_mwh(aid, agent_carbon_results.get(aid, {}), agent_config)
            
            # 调整电力报价以反映碳成本
            original_energy_bid = elec_bid.get('energy_bid', {})
            adjusted_energy_bid = original_energy_bid.copy()
            if 'price' in adjusted_energy_bid:
                adjusted_energy_bid['price'] += carbon_cost_per_mwh
                logging.debug(f"智能体 {aid} 电力报价调整: {original_energy_bid.get('price', 0):.2f} + {carbon_cost_per_mwh:.2f} = {adjusted_energy_bid['price']:.2f}")

            # 保存调整后的报价
            adjusted_bids_map[aid] = {
                'energy_bid': adjusted_energy_bid,
                'reserve_bid': elec_bid.get('reserve_bid', {}),
                'carbon_cost_per_mwh': carbon_cost_per_mwh
            }

            bid = {
                'agent_id': aid,
                'energy_bid': adjusted_energy_bid,
                'reserve_bid': elec_bid.get('reserve_bid', {}),
                'max_capacity': max_capacity,
                'carbon_cost_per_mwh': carbon_cost_per_mwh  # 记录碳成本
            }
            bids.append(bid)
            
        # 获取需求参数
        total_demand = self.electricity_market.demand_profile[round_number]
        reserve_requirement = self.electricity_market.reserve_profile[round_number]
        
        # 3. 电力市场出清
        opf_options = self.config.get("market", {}).get("opf_options", {})
        
        market_prices, agent_results = opf_joint_clearing(
            bids=bids,
            total_demand=total_demand,
            reserve_requirement=reserve_requirement,
            opf_options=opf_options
        )

        logging.info(f"电力市场出清完成 - 能源价: {market_prices.get('energy_price', 0):.2f}")

        # 4. 构建完整的返回结果
        # 返回格式：(market_results, agent_outcomes)
        market_results = {
            'energy_price': market_prices.get('energy_price', 0),
            'reserve_price': market_prices.get('reserve_price', 0),
            'carbon_market': {
                'current_price': carbon_new_price,
                'clearing_price': carbon_clearing_price,
                'trades': trades,
                'total_volume': carbon_results.get('total_volume', 0),
                'market_maker_inventory': carbon_results.get('market_maker_inventory', 0),
                'supply_demand_imbalance': carbon_results.get('supply_demand_imbalance', 0),
                'total_buy_demand': carbon_results.get('total_buy_demand', 0),
                'total_sell_supply': carbon_results.get('total_sell_supply', 0),
                'market_tightness': carbon_results.get('market_tightness', 0),
                'price_change': carbon_results.get('price_change', 0)
            }
        }
        
        # 合并电力和碳市场的智能体结果，并计算利润
        agent_outcomes = {}
        for agent_id in agent_results.keys():
            # 获取原始报价和私有信息
            original_bid = agent_actions.get(agent_id, {}).get('electricity', {})
            private_info = agent_private_info.get(agent_id, {})
            
            # 获取出清结果
            elec_result = agent_results[agent_id]
            cleared_energy = elec_result.get('cleared_energy', 0)
            cleared_reserve = elec_result.get('cleared_reserve', 0)
            
            # 计算收入
            energy_price = market_prices.get('energy_price', 0)
            reserve_price = market_prices.get('reserve_price', 0)
            revenue = cleared_energy * energy_price + cleared_reserve * reserve_price
            
            # 计算成本
            cost_energy = private_info.get('marginal_cost_energy', 0)
            cost_reserve = private_info.get('marginal_cost_reserve', 0)
            cost = cleared_energy * cost_energy + cleared_reserve * cost_reserve

            # 计算并扣减碳排放对应配额
            emission_factor = 0.8
            cfg_agents = self.config.get('agents', [])
            cfg_entry = next((a for a in cfg_agents if a.get('agent_id') == agent_id), None)
            if cfg_entry:
                emission_factor = float(cfg_entry.get('config', {}).get('emission_factor', emission_factor))
            else:
                emission_factor = float(private_info.get('emission_factor', emission_factor))
            emissions = float(cleared_energy) * emission_factor
            initial_quota = float(self.carbon_market.agent_quotas.get(agent_id, 0.0))
            self.carbon_market.agent_quotas[agent_id] = max(0.0, initial_quota - emissions)
            
            # 计算利润
            profit = revenue - cost
            
            agent_outcomes[agent_id] = {
                **agent_results[agent_id],  # 电力市场结果
                'carbon': agent_carbon_results.get(agent_id, {}),  # 碳市场结果
                'submitted_bid': original_bid,  # 原始电力报价
                'adjusted_bid': adjusted_bids_map.get(agent_id, {}),  # 调整后的电力报价（含碳成本）
                'submitted_carbon_bid': carbon_bids.get(agent_id, {}),  # 原始碳市场订单
                'emissions': emissions,
                'initial_quota': initial_quota,
                'profit': profit,  # 添加利润
                'revenue': revenue,
                'cost': cost
            }
        
        return market_results, agent_outcomes
    
    def _calculate_carbon_cost_per_mwh(self, agent_id: str, carbon_result: Dict, agent_config: Dict) -> float:
        """
        计算每MWh电力的碳成本
        """
        if not agent_config:
            return 0.0
        
        # 获取智能体的排放因子（吨CO2/MWh）
        emission_factor = agent_config['config'].get('emission_factor', 0.8)  # 默认0.8吨CO2/MWh
        
        # 获取碳成本
        carbon_cost_total = carbon_result.get('cost', 0)
        quota_change = carbon_result.get('quota_change', 0)
        
        # 如果智能体购买了碳配额，将成本分摊到发电量上
        if carbon_cost_total > 0 and quota_change > 0:
            # 根据配额变化计算每吨CO2的实际成本
            cost_per_ton_co2 = carbon_cost_total / quota_change
            carbon_cost_per_mwh = cost_per_ton_co2 * emission_factor
        else:
            # 使用当前碳价格作为机会成本，安全获取避免None
            current_carbon_price = self.carbon_market.current_price if self.carbon_market.current_price is not None else 150.0
            carbon_cost_per_mwh = current_carbon_price * emission_factor
        
        return carbon_cost_per_mwh