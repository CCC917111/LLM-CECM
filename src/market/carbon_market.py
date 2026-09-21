import logging
import numpy as np
from typing import Dict, List, Tuple, Any
import random


class CarbonMarket:
    def __init__(self, config):
        """
        增强版碳市场，引入做市商机制和价格动态调整
        """
        self.agent_ids = config.get('agent_ids', [])
        
        # 检查是否启用分层配额系统
        tiered_quota = config.get('tiered_quota', {})
        if tiered_quota.get('enabled', False):
            # 使用分层配额系统
            agent_quota_mapping = tiered_quota.get('agent_quota_mapping', {})
            self.agent_quotas = {}
            for aid in self.agent_ids:
                self.agent_quotas[aid] = agent_quota_mapping.get(aid, config.get('init_quota', 80.0))
            print(f"🏭 分层配额系统已启用，{len(self.agent_ids)}个智能体使用差异化配额")
        else:
            # 使用统一配额
            self.agent_quotas = {aid: config.get('init_quota', 80.0) for aid in self.agent_ids}
            
        self.agent_funds = {aid: config.get('init_funds', 100000.0) for aid in self.agent_ids}
        
        # 市场价格参数
        self.opening_price = config.get('init_carbon_price', 50.0)  # S
        self.current_price = self.opening_price  # C
        self.min_price = config.get('min_carbon_price', 20.0)
        self.max_price = config.get('max_carbon_price', 120.0)
        
        # 做市商机制参数
        self.enable_market_maker = config.get('enable_market_maker', True)
        self.spread_limit = config.get('spread_limit', 0.2)  # 20%价差限制
        self.market_maker_inventory = 0  # 做市商库存
        self.market_maker_funds = config.get('market_maker_funds', 500000.0)
        
        # 多层级订单簿增加市场深度
        self.order_book_levels = config.get('order_book_levels', 5)
        self.buy_orders = []  # [(price, quantity, agent_id)]
        self.sell_orders = []  # [(price, quantity, agent_id)]
        
        # 交易激励机制
        self.volume_bonus = config.get('volume_bonus', 0.01)  # 交易量奖励
        self.liquidity_bonus = config.get('liquidity_bonus', 0.005)  # 流动性奖励
        
        # 价格波动监控
        self.price_history = [self.current_price]
        self.volatility_threshold = config.get('volatility_threshold', 0.05)
        
        # === 新增价格动态调整机制 ===
        # 参考文献: Feng et al. (2009), Alberola et al. (2008)
        self.enable_price_dynamics = config.get('enable_price_dynamics', True)
        self.price_adjustment_speed = config.get('price_adjustment_speed', 0.1)  # 价格调整速度
        self.supply_demand_sensitivity = config.get('supply_demand_sensitivity', 0.05)  # 供需敏感度
        self.volatility_damping = config.get('volatility_damping', 0.8)  # 波动衰减因子
        self.external_shock_probability = config.get('external_shock_probability', 0.1)  # 外部冲击概率
        self.shock_magnitude = config.get('shock_magnitude', 0.15)  # 冲击幅度
        
        # 记录供需失衡历史
        self.supply_demand_history = []
        self.volume_history = []
        
        # 统计信息
        self.total_volume = 0
        self.total_trades = 0
        self.successful_rounds = 0
        
        logging.info(f"碳市场初始化完成，启用做市商机制：{self.enable_market_maker}，价格动态调整：{self.enable_price_dynamics}")

    def calculate_supply_demand_imbalance(self, buy_orders, sell_orders):
        """
        计算供需失衡度
        参考文献: Feng et al. (2009) - 基于EU ETS的供需失衡分析
        """
        total_buy_volume = sum(qty for _, qty, _ in buy_orders)
        total_sell_volume = sum(qty for _, qty, _ in sell_orders)
        
        if total_buy_volume + total_sell_volume == 0:
            return 0.0
            
        # 供需失衡度 = (需求 - 供给) / (需求 + 供给)
        # 正值表示供不应求，负值表示供过于求
        imbalance = (total_buy_volume - total_sell_volume) / (total_buy_volume + total_sell_volume)
        
        logging.debug(f"供需失衡度: {imbalance:.3f} (买单:{total_buy_volume:.1f}, 卖单:{total_sell_volume:.1f})")
        
        return imbalance



    def dynamic_price_update(self, trades, supply_demand_imbalance):
        """
        简化的价格动态调整机制
        基于简化版学术文献模型
        """
        if not self.enable_price_dynamics:
            return
            
        # 基于供需失衡的线性价格调整 (Feng et al., 2009)
        # ΔC = α × Imbalance × C + ε
        alpha = 0.05  # 价格敏感度参数
        epsilon = random.uniform(-0.01, 0.01) * self.current_price  # 随机冲击项
        
        # 主要价格调整：基于供需失衡
        price_adjustment = alpha * supply_demand_imbalance * self.current_price + epsilon
        
        # 应用调整幅度限制：|ΔC| ≤ 0.1 × C
        max_adjustment = 0.1 * self.current_price
        price_adjustment = max(-max_adjustment, min(max_adjustment, price_adjustment))
        
        # 更新价格
        new_price = self.current_price + price_adjustment
        new_price = max(self.min_price, min(self.max_price, new_price))  # 价格边界
        
        if abs(new_price - self.current_price) > 0.01:  # 超过1分钱才记录
            logging.info(f"价格调整: {self.current_price:.2f} → {new_price:.2f} 元/吨 (失衡度: {supply_demand_imbalance:.3f})")
                
        self.current_price = new_price
        
        # 更新历史记录
        self.supply_demand_history.append(supply_demand_imbalance)
        if trades:
            self.volume_history.append(sum(trade['quantity'] for trade in trades))
        else:
            self.volume_history.append(0)
            
        # 保持历史记录不超过20期
        if len(self.supply_demand_history) > 20:
            self.supply_demand_history.pop(0)
        if len(self.volume_history) > 20:
            self.volume_history.pop(0)

    def generate_market_maker_orders(self):
        """
        做市商自动生成双边报价，增加交易机会
        """
        if not self.enable_market_maker:
            return
            
        # 计算市场波动率
        volatility = self._calculate_volatility()
        
        # 动态调整价差 - 波动率越高价差越大，但设置更激进的价差
        dynamic_spread = min(self.spread_limit, 0.08 + volatility * 0.3)  # 降低基础价差
        
        # 计算做市商报价
        bid_price = self.current_price * (1 - dynamic_spread/2)
        ask_price = self.current_price * (1 + dynamic_spread/2)
        
        # 确保价格在合理范围内
        bid_price = max(bid_price, self.min_price)
        ask_price = min(ask_price, self.max_price)
        
        # 增加做市商的报价量，提供更多流动性
        base_quantity = 30 + random.uniform(-10, 10)  # 增加基础报价量
        
        # 根据库存调整报价量 - 更激进的库存管理
        if self.market_maker_inventory > 30:  # 库存过多，积极卖出
            sell_quantity = base_quantity * 2.0  # 更激进
            buy_quantity = base_quantity * 0.5
        elif self.market_maker_inventory < -30:  # 库存不足，积极买入
            sell_quantity = base_quantity * 0.5
            buy_quantity = base_quantity * 2.0  # 更激进
        else:  # 库存平衡
            sell_quantity = buy_quantity = base_quantity
            
        # 添加做市商订单
        if self.market_maker_funds >= bid_price * buy_quantity:
            self.buy_orders.append((bid_price, buy_quantity, 'market_maker'))
            
        self.sell_orders.append((ask_price, sell_quantity, 'market_maker'))
        
        # 增加多层做市商报价以提供更深的市场深度
        for i in range(1, 3):  # 额外2层报价
            spread_multiplier = 1 + i * 0.02  # 逐层增加价差
            deep_bid = bid_price * (1 - i * 0.01)
            deep_ask = ask_price * (1 + i * 0.01)
            
            if deep_bid >= self.min_price and self.market_maker_funds >= deep_bid * base_quantity * 0.7:
                self.buy_orders.append((deep_bid, base_quantity * 0.7, 'market_maker'))
                
            if deep_ask <= self.max_price:
                self.sell_orders.append((deep_ask, base_quantity * 0.7, 'market_maker'))
        
        logging.debug(f"做市商多层报价：买入 {buy_quantity:.1f}吨@{bid_price:.2f}元/吨，"
                     f"卖出 {sell_quantity:.1f}吨@{ask_price:.2f}元/吨")

    def add_liquidity_incentives(self):
        """
        增加流动性激励机制，鼓励更多交易
        """
        # 为所有智能体提供流动性奖励（增加概率和金额）
        for agent_id in self.agent_ids:
            if random.random() < 0.5:  # 50%概率获得奖励（增加）
                bonus = self.liquidity_bonus * self.current_price * random.uniform(2, 8)  # 增加奖励金额
                self.agent_funds[agent_id] += bonus
                logging.debug(f"智能体 {agent_id} 获得流动性奖励 {bonus:.2f} 元")
                
        # 增加额外的交易激励：配额轻微失衡
        if random.random() < 0.3:  # 30%概率进行配额调整
            selected_agents = random.sample(self.agent_ids, min(3, len(self.agent_ids)))
            for agent_id in selected_agents:
                if random.random() < 0.5:  # 减少配额，促进购买
                    quota_reduction = random.uniform(5, 15)
                    self.agent_quotas[agent_id] = max(0, self.agent_quotas[agent_id] - quota_reduction)
                    logging.debug(f"智能体 {agent_id} 配额减少 {quota_reduction:.1f}吨，促进买入")
                else:  # 增加配额，促进销售
                    quota_increase = random.uniform(5, 10)
                    self.agent_quotas[agent_id] += quota_increase
                    logging.debug(f"智能体 {agent_id} 配额增加 {quota_increase:.1f}吨，促进卖出")

    def implement_volume_discounts(self, trades):
        """
        实施大额交易折扣，鼓励大量交易
        """
        for trade in trades:
            volume = trade['quantity']
            if volume >= 50:  # 大额交易
                discount = min(0.1, volume * 0.001)  # 最高10%折扣
                refund = trade['price'] * volume * discount
                
                # 给买方返还部分资金
                if trade['buyer'] != 'market_maker':
                    self.agent_funds[trade['buyer']] += refund
                    logging.debug(f"大额交易折扣：{trade['buyer']} 获得 {refund:.2f} 元返还")

    def process_bids(self, agent_bids):
        """
        处理智能体报价，增强版包含多种促进交易的机制
        """
        # 清空订单簿
        self.buy_orders = []
        self.sell_orders = []
        
        # 添加流动性激励
        self.add_liquidity_incentives()
        
        # 生成做市商订单
        self.generate_market_maker_orders()
        
        # 处理智能体订单
        for agent_id, bid in agent_bids.items():
            self._process_agent_bid(agent_id, bid)
        
        # 计算供需失衡与供需总量
        supply_demand_imbalance = self.calculate_supply_demand_imbalance(self.buy_orders, self.sell_orders)
        total_buy_demand = float(sum(qty for _, qty, _ in self.buy_orders))
        total_sell_supply = float(sum(qty for _, qty, _ in self.sell_orders))
        market_tightness = 0.0
        if (total_buy_demand + total_sell_supply) > 0:
            market_tightness = total_buy_demand / (total_buy_demand + total_sell_supply)  # 映射到[0,1]

        # 执行撮合
        prev_price = float(self.current_price)
        trades = self._execute_matching()
        
        # 应用价格动态调整机制
        self.dynamic_price_update(trades, supply_demand_imbalance)
        # 价格变动率（相对上一时点）
        price_change = 0.0
        try:
            if prev_price > 0:
                price_change = (float(self.current_price) - prev_price) / prev_price
        except Exception:
            price_change = 0.0
        
        # 实施交易激励
        if trades:
            self.implement_volume_discounts(trades)
            
        # 更新统计信息
        if trades:
            self.successful_rounds += 1
            self.total_trades += len(trades)
            total_round_volume = sum(trade['quantity'] for trade in trades)
            self.total_volume += total_round_volume
            
            logging.info(f"碳市场出清完成，本轮成交 {len(trades)} 笔交易，"
                        f"总量 {total_round_volume:.1f} 吨，成交价 {self.current_price:.2f} 元/吨")
        else:
            logging.info("碳市场出清完成，本轮无成交")
            
        return {
            'trades': trades,
            'current_price': self.current_price,
            'clearing_price': self.current_price,  # 简化：撮合后价格作为清算价
            'total_volume': sum(trade['quantity'] for trade in trades) if trades else 0,
            'market_maker_inventory': self.market_maker_inventory,
            'supply_demand_imbalance': supply_demand_imbalance,
            'total_buy_demand': total_buy_demand,
            'total_sell_supply': total_sell_supply,
            'market_tightness': market_tightness,
            'price_change': price_change
        }

    def _process_agent_bid(self, agent_id, bid):
        """
        处理单个智能体的报价
        """
        # 支持新格式（buy_orders/sell_orders数组）和旧格式（直接字段）
        buy_orders = bid.get('buy_orders', [])
        sell_orders = bid.get('sell_orders', [])
        
        # 兼容旧格式
        if not buy_orders and not sell_orders:
            buy_quantity = bid.get('buy_quantity', 0)
            buy_price = bid.get('buy_price', 0)
            sell_quantity = bid.get('sell_quantity', 0) 
            sell_price = bid.get('sell_price', 0)
            
            # 转换为新格式
            if buy_quantity > 0 and buy_price > 0:
                buy_orders = [{'quantity': buy_quantity, 'price': buy_price}]
            if sell_quantity > 0 and sell_price > 0:
                sell_orders = [{'quantity': sell_quantity, 'price': sell_price}]
        
        # 价格约束：随当前价动态调整，放宽至±15%
        price_lower = float(self.current_price) * 0.85
        price_upper = float(self.current_price) * 1.15
        
        # 处理买单数组
        for buy_order in buy_orders:
            buy_quantity = buy_order.get('quantity', 0)
            buy_price = buy_order.get('price', 0)
            
            if buy_quantity > 0 and price_lower <= buy_price <= price_upper:
                # 资金约束：p_i * q_i <= M_i
                if buy_price * buy_quantity <= self.agent_funds[agent_id]:
                    self.buy_orders.append((buy_price, buy_quantity, agent_id))
                    logging.debug(f"智能体 {agent_id} 买单：{buy_quantity:.1f}吨@{buy_price:.2f}元/吨")
                else:
                    logging.debug(f"智能体 {agent_id} 买单被拒：资金不足")
                    
        # 处理卖单数组
        for sell_order in sell_orders:
            sell_quantity = sell_order.get('quantity', 0)
            sell_price = sell_order.get('price', 0)
            
            if sell_quantity > 0 and price_lower <= sell_price <= price_upper:
                # 配额约束：|q_i| <= CEA_i
                if sell_quantity <= self.agent_quotas[agent_id]:
                    self.sell_orders.append((sell_price, sell_quantity, agent_id))
                    logging.debug(f"智能体 {agent_id} 卖单：{sell_quantity:.1f}吨@{sell_price:.2f}元/吨")
                else:
                    logging.debug(f"智能体 {agent_id} 卖单被拒：配额不足")

    def _execute_matching(self):
        """
        执行集中撮合，支持多层级匹配
        """
        trades = []
        
        if not self.buy_orders or not self.sell_orders:
            return trades
            
        # 排序：买单按价格降序，卖单按价格升序
        self.buy_orders.sort(key=lambda x: x[0], reverse=True)
        self.sell_orders.sort(key=lambda x: x[0])
        
        # 选取前N个最优订单进行撮合
        top_buy_orders = self.buy_orders[:self.order_book_levels]
        top_sell_orders = self.sell_orders[:self.order_book_levels]
        
        # 逐级撮合
        for buy_order in top_buy_orders:
            buy_price, buy_qty, buyer = buy_order
            
            for i, sell_order in enumerate(top_sell_orders):
                sell_price, sell_qty, seller = sell_order
                
                # 撮合条件：买价 >= 卖价
                if buy_price >= sell_price and buy_qty > 0 and sell_qty > 0:
                    # 成交数量：取较小值
                    trade_qty = min(buy_qty, sell_qty)
                     
                    # 成交价格：三方中位数定价
                    # P_trade = median(P_b, P_s, C)
                    trade_price = np.median([buy_price, sell_price, self.current_price])
                    
                    # 成交后更新市场价格，使其更接近clearing price
                    self.current_price = trade_price
                     
                    # 执行交易
                    if self._execute_trade(buyer, seller, trade_qty, trade_price):
                        trades.append({
                            'buyer': buyer,
                            'seller': seller, 
                            'quantity': trade_qty,
                            'price': trade_price
                        })
                        
                        # 更新订单数量
                        buy_qty -= trade_qty
                        sell_qty -= trade_qty
                        top_sell_orders[i] = (sell_price, sell_qty, seller)
                        
                        logging.debug(f"撮合成功：{buyer} 从 {seller} 买入 {trade_qty:.1f}吨@{trade_price:.2f}元/吨")
                        
                        if buy_qty <= 0:
                            break
            
            # 更新买单
            buy_order = (buy_price, buy_qty, buyer)
                    
        # 如果有成交，价格已在dynamic_price_update中更新，这里不再重复更新
        if trades:
            self.price_history.append(self.current_price)
            
        return trades

    def _execute_trade(self, buyer, seller, quantity, price):
        """
        执行单笔交易
        """
        try:
            total_cost = quantity * price
            
            # 处理买方
            if buyer == 'market_maker':
                if self.market_maker_funds >= total_cost:
                    self.market_maker_funds -= total_cost
                    self.market_maker_inventory += quantity
                else:
                    return False
            else:
                if self.agent_funds[buyer] >= total_cost:
                    self.agent_funds[buyer] -= total_cost
                    self.agent_quotas[buyer] += quantity
                else:
                    return False
                    
            # 处理卖方
            if seller == 'market_maker':
                self.market_maker_funds += total_cost
                self.market_maker_inventory -= quantity
            else:
                if self.agent_quotas[seller] >= quantity:
                    self.agent_quotas[seller] -= quantity
                    self.agent_funds[seller] += total_cost
                else:
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"交易执行失败：{e}")
            return False

    def _calculate_volatility(self):
        """
        计算价格波动率
        """
        if len(self.price_history) < 2:
            return 0.0
            
        # 使用最近5个价格计算波动率
        recent_prices = self.price_history[-5:]
        if len(recent_prices) < 2:
            return 0.0
            
        returns = []
        for i in range(1, len(recent_prices)):
            ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            returns.append(ret)
            
        return np.std(returns) if returns else 0.0

    def get_market_info(self):
        """
        获取市场信息
        """
        avg_volume_per_round = self.total_volume / max(1, self.successful_rounds)
        activity_rate = self.successful_rounds / max(1, len(self.price_history) - 1)
        
        return {
            'current_price': self.current_price,
            'opening_price': self.opening_price,
            'total_volume': self.total_volume,
            'total_trades': self.total_trades,
            'successful_rounds': self.successful_rounds,
            'avg_volume_per_round': avg_volume_per_round,
            'activity_rate': activity_rate,
            'market_maker_inventory': self.market_maker_inventory,
            'current_volatility': self._calculate_volatility(),
            'price_dynamics_enabled': self.enable_price_dynamics
        }

    def get_market_statistics(self) -> Dict[str, Any]:
        """
        获取碳市场统计信息
        """
        if len(self.price_history) < 2:
            return {'current_price': self.current_price}
        
        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history) if self.volume_history else np.array([0])
        
        return {
            'current_price': self.current_price,
            'price_volatility': np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0,
            'price_trend': (prices[-1] - prices[0]) / prices[0] * 100,  # 总体趋势%
            'avg_volume': np.mean(volumes),
            'total_volume': np.sum(volumes),
            'price_range': {'min': np.min(prices), 'max': np.max(prices)},
            'num_rounds': len(self.price_history) - 1
        }
    
    def get_agent_quota_summary(self) -> Dict[str, float]:
        """
        获取所有智能体的配额汇总
        """
        return self.agent_quotas.copy()
    
    def get_agent_funds_summary(self) -> Dict[str, float]:
        """
        获取所有智能体的资金汇总
        """
        return self.agent_funds.copy() 