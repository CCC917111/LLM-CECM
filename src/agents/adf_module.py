# agents/adf_module.py
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import statistics
import copy
import random

class AdaptiveDecisionFrequency:
    """
    自适应决策频率（ADF）模块
    
    实现企业决策中的"常规运营"与"战略复盘"平衡，
    通过三种触发机制决定是否启用深度LLM决策：
    1. 业绩驱动：近期利润显著低于历史移动平均值
    2. 事件驱动：市场价格波动率或关键外部信息超过阈值
    3. 周期驱动：固定的长周期例行复盘
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict] = None):
        """
        初始化ADF模块
        
        Args:
            agent_id: 智能体ID
            config: ADF配置参数
        """
        self.agent_id = agent_id
        
        # 默认配置参数
        default_config = {
            # 初始化参数
            'min_llm_rounds': 3,  # 前几轮强制使用LLM建立基线
            
            # 业绩驱动参数
            'profit_window_size': 10,  # 利润移动平均窗口大小
            'profit_threshold': 0.9,   # 利润阈值（当前利润/历史均值）- 提高到0.9，更容易触发
            
            # 事件驱动参数
            'price_volatility_threshold': 0.08,  # 价格波动率阈值 - 降低到0.08，更敏感
            'price_window_size': 5,              # 价格波动计算窗口
            'event_keywords': ['政策', '公告', '变化', '调整', '新规'],  # 事件关键词
            
            # 周期驱动参数
            'strategic_review_cycle': 7,  # 战略复盘周期（轮次）
            
            # 启发式决策参数
            'markup_adjustment_rate': 0.05,  # 加价率调整幅度
            'capacity_allocation_inertia': 0.9,  # 容量分配惯性系数

            # 触发节流&冷却（可选，默认不启用以保持兼容）
            'llm_cooldown_rounds': 0,                 # LLM触发后的冷却轮数（0表示不冷却）
            'extreme_volatility_threshold': None,      # 极端波动阈值（例如0.35），超过则可突破冷却
            'extreme_profit_ratio_threshold': None,    # 极端利润比阈值（例如0.5），低于则可突破冷却
        }
        
        # 对无法识别的配置键给出告警，避免参数因键名写错而被静默忽略
        informational_keys = {'delta_perf'}
        unknown_keys = set(config or {}) - set(default_config) - informational_keys
        if unknown_keys:
            logging.warning(
                f"智能体 {agent_id} ADF配置中存在无法识别的键(将被忽略): {sorted(unknown_keys)}"
            )
        self.config = {**default_config, **(config or {})}
        logging.info(
            f"智能体 {agent_id} ADF生效参数: profit_threshold={self.config['profit_threshold']}, "
            f"price_volatility_threshold={self.config['price_volatility_threshold']}, "
            f"strategic_review_cycle={self.config['strategic_review_cycle']}"
        )
        
        # 历史数据存储
        self.profit_history = deque(maxlen=self.config['profit_window_size'])
        self.price_history = deque(maxlen=self.config['price_window_size'])
        self.last_llm_decision_round = 0
        
        # 决策统计
        self.llm_decision_count = 0  # LLM决策次数
        self.heuristic_decision_count = 0  # 启发式决策次数
        
        # 启发式决策状态
        self.last_llm_strategy = None  # 上次LLM决策的策略
        self.current_markup_rate = 0.1  # 当前加价率
        self.current_capacity_split = {'energy': 0.8, 'reserve': 0.2}  # 当前容量分配
        
        # 碳市场策略状态 - 初始为None，等待第一次LLM决策后学习
        self.current_carbon_strategy = None
        self.carbon_strategy_initialized = False  # 标记是否已从LLM学习到策略
        
        logging.info(f"ADF模块初始化完成 - 智能体 {agent_id}")
        logging.info("碳市场策略将从第一次LLM决策中学习")
    
    def should_trigger_llm_decision(self, round_number: int, market_info: Dict, 
                                  my_profit: Optional[float] = None, 
                                  market_events_info: Optional[str] = None) -> Tuple[bool, str]:
        """
        判断是否应该触发LLM深度决策
        
        Args:
            round_number: 当前轮次
            market_info: 市场信息
            my_profit: 当前轮次利润
            market_events_info: 市场事件信息
            
        Returns:
            Tuple[是否触发, 触发原因]
        """
        trigger_reasons = []
        
        # 1. 业绩驱动触发
        if self._check_performance_trigger(my_profit):
            trigger_reasons.append("业绩驱动")
        
        # 2. 事件驱动触发
        if self._check_event_trigger(market_info, market_events_info):
            trigger_reasons.append("事件驱动")
        
        # 3. 周期驱动触发
        if self._check_cycle_trigger(round_number):
            trigger_reasons.append("周期驱动")
        
        should_trigger = len(trigger_reasons) > 0
        reason = ", ".join(trigger_reasons) if trigger_reasons else "无触发条件"

        # 冷却控制（若开启）
        cooldown = max(0, int(self.config.get('llm_cooldown_rounds', 0) or 0))
        if should_trigger and cooldown > 0:
            rounds_since_last = round_number - self.last_llm_decision_round
            if rounds_since_last < cooldown:
                # 判断是否有极端条件允许突破冷却
                extreme_trigger = False

                # 极端价格波动
                extreme_vol_th = self.config.get('extreme_volatility_threshold')
                if extreme_vol_th is not None and len(self.price_history) >= 3:
                    prices = list(self.price_history)
                    mean_price = np.mean(prices) if len(prices) > 0 else 0
                    volatility = np.std(prices) / mean_price if mean_price > 0 else 0
                    if volatility > float(extreme_vol_th):
                        extreme_trigger = True

                # 极端利润下滑
                if (not extreme_trigger) and (self.config.get('extreme_profit_ratio_threshold') is not None) and (my_profit is not None):
                    min_history_for_trigger = max(5, self.config['min_llm_rounds'] + 2)
                    if len(self.profit_history) >= min_history_for_trigger:
                        hist_avg = statistics.mean(list(self.profit_history))
                        if hist_avg > 0:
                            profit_ratio = my_profit / hist_avg
                            if profit_ratio < float(self.config['extreme_profit_ratio_threshold']):
                                extreme_trigger = True

                if not extreme_trigger:
                    remaining = cooldown - rounds_since_last
                    logging.debug(
                        f"智能体 {self.agent_id} ADF冷却中，阻止LLM触发 - round={round_number}, 剩余冷却={remaining}"
                    )
                    should_trigger = False
                    reason = f"冷却中(剩余{remaining}轮); 触发来源: {reason}"
                else:
                    logging.info(
                        f"智能体 {self.agent_id} ADF极端条件触发，突破冷却限制 - 原因: {reason}"
                    )

        if should_trigger:
            self.last_llm_decision_round = round_number
            self.llm_decision_count += 1
            logging.info(f"智能体 {self.agent_id} ADF触发LLM决策 - 原因: {reason}")
        else:
            self.heuristic_decision_count += 1
            logging.debug(f"智能体 {self.agent_id} ADF使用启发式决策 - 轮次 {round_number}")
        
        return should_trigger, reason
    
    def _check_performance_trigger(self, current_profit: Optional[float]) -> bool:
        """检查业绩驱动触发条件"""
        if current_profit is None:
            return False
        
        # 需要足够的历史数据才能进行业绩比较
        min_history_for_trigger = max(5, self.config['min_llm_rounds'] + 2)
        if len(self.profit_history) < min_history_for_trigger:
            self.profit_history.append(current_profit)
            return False
        
        # 计算历史利润移动平均（不包括当前轮次）
        historical_avg = statistics.mean(list(self.profit_history))
        
        # 判断当前利润是否显著低于历史平均
        if historical_avg > 0:
            profit_ratio = current_profit / historical_avg
            if profit_ratio < self.config['profit_threshold']:
                logging.info(f"智能体 {self.agent_id} 业绩触发: 当前利润 {current_profit:.2f} < 历史均值 {historical_avg:.2f} * {self.config['profit_threshold']}")
                # 更新利润历史
                self.profit_history.append(current_profit)
                return True
        
        # 更新利润历史
        self.profit_history.append(current_profit)
        return False
    
    def _check_event_trigger(self, market_info: Dict, market_events_info: Optional[str]) -> bool:
        """检查事件驱动触发条件"""
        # 检查价格波动率
        # 优先读取 current_price；耦合市场流程中公开信息提供的是上一轮出清价 last_energy_price
        elec_info = market_info.get('electricity_market', {}) or {}
        current_price = elec_info.get('current_price') or elec_info.get('last_energy_price') or 0
        if current_price > 0:
            self.price_history.append(current_price)
            
            if len(self.price_history) >= 3:
                prices = list(self.price_history)
                volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                
                if volatility > self.config['price_volatility_threshold']:
                    logging.info(f"智能体 {self.agent_id} 价格波动触发: 波动率 {volatility:.3f} > 阈值 {self.config['price_volatility_threshold']}")
                    return True
        
        # 检查市场事件关键词
        if market_events_info:
            for keyword in self.config['event_keywords']:
                if keyword in market_events_info:
                    logging.info(f"智能体 {self.agent_id} 市场事件触发: 检测到关键词 '{keyword}'")
                    return True
        
        return False
    
    def _check_cycle_trigger(self, round_number: int) -> bool:
        """检查周期驱动触发条件"""
        rounds_since_last = round_number - self.last_llm_decision_round
        
        if rounds_since_last >= self.config['strategic_review_cycle']:
            logging.info(f"智能体 {self.agent_id} 周期触发: 距离上次LLM决策 {rounds_since_last} 轮 >= 周期 {self.config['strategic_review_cycle']}")
            return True
        
        return False
    
    def generate_heuristic_bid(self, round_number: int, market_info: Dict, 
                             private_info: Dict) -> Dict:
        """
        生成启发式报价（基于上次LLM策略的微调版本）
        
        策略：如果有上次LLM决策，则基于该决策进行小幅调整；
        否则使用简单的成本加成策略
        
        Args:
            round_number: 当前轮次
            market_info: 市场信息
            private_info: 智能体私有信息
            
        Returns:
            启发式报价字典
        """
        logging.debug(f"智能体 {self.agent_id} 生成启发式报价 - 轮次 {round_number}")
        
        # 如果有上次LLM策略，基于该策略进行微调
        if self.last_llm_strategy and 'electricity' in self.last_llm_strategy:
            result = self._generate_strategy_based_bid(market_info, private_info)
            try:
                last_elec = self.last_llm_strategy.get('electricity', {})
                last_energy = last_elec.get('energy_bid', {})
                last_reserve = last_elec.get('reserve_bid', {})
                new_energy = result['electricity']['energy_bid']
                new_reserve = result['electricity']['reserve_bid']
                logging.debug(
                    "ADF启发式(基于LLM) - 上次LLM vs 本次启发式: "
                    f"energy_price {last_energy.get('price', 0):.4f} -> {new_energy.get('price', 0):.4f}, "
                    f"energy_qty {last_energy.get('quantity', 0):.4f} -> {new_energy.get('quantity', 0):.4f}; "
                    f"reserve_price {last_reserve.get('price', 0):.4f} -> {new_reserve.get('price', 0):.4f}, "
                    f"reserve_qty {last_reserve.get('quantity', 0):.4f} -> {new_reserve.get('quantity', 0):.4f}; "
                    f"markup_rate={self.current_markup_rate:.4f}, capacity_split={self.current_capacity_split}"
                )
            except Exception as e:
                logging.debug(f"ADF启发式差异记录失败: {e}")
            return result
        else:
            # 没有LLM基线策略时，使用简单成本加成
            logging.debug(
                f"ADF启发式(简单成本) - 无上次LLM策略或结构缺失，改用简单成本加成。"
            )
            return self._generate_simple_cost_bid(market_info, private_info)
    
    def _generate_strategy_based_bid(self, market_info: Dict, private_info: Dict) -> Dict:
        """基于上次LLM策略生成智能化启发式报价"""
        last_elec = self.last_llm_strategy['electricity']
        
        # 获取上次的报价作为基准
        last_energy_price = last_elec['energy_bid']['price']
        last_energy_qty = last_elec['energy_bid']['quantity']
        last_reserve_price = last_elec['reserve_bid']['price']
        last_reserve_qty = last_elec['reserve_bid']['quantity']
        
        # 智能化市场分析
        energy_price, energy_quantity = self._intelligent_energy_pricing(
            last_energy_price, last_energy_qty, market_info, private_info)
        reserve_price, reserve_quantity = self._intelligent_reserve_pricing(
            last_reserve_price, last_reserve_qty, market_info, private_info)
        
        # 生成碳市场报价
        carbon_bid = self._generate_heuristic_carbon_bid(market_info)
        
        return {
            'electricity': {
                'energy_bid': {'price': energy_price, 'quantity': energy_quantity},
                'reserve_bid': {'price': reserve_price, 'quantity': reserve_quantity}
            },
            'carbon': carbon_bid,
            'decision_type': 'heuristic',
            'markup_rate': self.current_markup_rate,
            'capacity_split': self.current_capacity_split.copy()
        }
    
    def _intelligent_energy_pricing(self, last_price: float, last_qty: float, 
                                   market_info: Dict, private_info: Dict) -> tuple:
        """智能化能源报价策略 - 直接基于LLM学习价格进行微调"""
        energy_cost = private_info['marginal_cost_energy']
        max_capacity = private_info['max_capacity']
        
        # 市场需求分析
        total_demand = market_info.get('electricity_market', {}).get('total_demand', 4000)
        
        # 直接基于LLM学习到的价格进行小幅微调，而不是重新计算
        # 这是关键：保持LLM决策的价格水平，只做微小调整
        base_price = last_price  # 直接使用LLM决策的价格作为基础
        
        # 只进行非常小的随机微调，保持LLM决策的主要方向
        import random
        
        # 微调幅度控制在±2%以内，保持LLM决策的主要趋势
        micro_adjustment = random.uniform(0.98, 1.02)
        
        # 根据极端市场条件进行轻微调整（但不覆盖LLM策略）
        market_adjustment = 1.0
        if total_demand > 5000:  # 极高需求
            market_adjustment = 1.01  # 轻微提价1%
        elif total_demand < 3000:  # 极低需求  
            market_adjustment = 0.99  # 轻微降价1%
        
        # 计算最终价格：优先保持LLM策略，只做微调
        adjusted_price = base_price * micro_adjustment * market_adjustment
        
        # 成本保护：根据智能体成本水平动态调整保护策略
        # 高成本智能体需要更大的降价空间来保持竞争力
        if energy_cost > 50:  # 高成本智能体
            cost_protection_ratio = 0.75  # 允许25%亏损以保持竞争力
            logging.debug(f"智能体 {self.agent_id} 高成本保护: cost={energy_cost:.2f}, 允许25%亏损")
        elif energy_cost > 30:  # 中等成本智能体
            cost_protection_ratio = 0.85  # 允许15%亏损
            logging.debug(f"智能体 {self.agent_id} 中等成本保护: cost={energy_cost:.2f}, 允许15%亏损")
        else:  # 低成本智能体
            cost_protection_ratio = 0.90  # 允许10%亏损
            logging.debug(f"智能体 {self.agent_id} 低成本保护: cost={energy_cost:.2f}, 允许10%亏损")
        
        min_acceptable_price = energy_cost * cost_protection_ratio
        
        if adjusted_price >= min_acceptable_price:
            energy_price = adjusted_price  # 使用微调后的LLM价格
        else:
            # 只在严重亏损时才调整，并记录日志
            energy_price = min_acceptable_price
            logging.warning(f"智能体 {self.agent_id} ADF价格保护触发: {adjusted_price:.2f} -> {energy_price:.2f} (成本保护比例={cost_protection_ratio:.2f})")
        
        # 数量调整：基于LLM数量进行小幅调整
        quantity_adjustment = random.uniform(0.95, 1.05)  # ±5%的数量微调
        energy_quantity = min(max_capacity * 0.9, 
                             max(last_qty * 0.8,  # 最低保持80%
                                 last_qty * quantity_adjustment))
        
        # 调试日志：显示保持LLM策略的效果
        logging.info(f"智能体 {self.agent_id} ADF电力定价(保持LLM策略): {last_price:.2f} -> {energy_price:.2f}")
        logging.info(f"智能体 {self.agent_id} ADF微调因子: micro={micro_adjustment:.3f}, market={market_adjustment:.3f}")
        
        return energy_price, energy_quantity
    
    def _intelligent_reserve_pricing(self, last_price: float, last_qty: float,
                                   market_info: Dict, private_info: Dict) -> tuple:
        """智能化备用报价策略"""
        reserve_cost = private_info['marginal_cost_reserve']
        max_capacity = private_info['max_capacity']
        reserve_requirement = market_info.get('reserve_requirement', 0)
        
        if reserve_requirement <= 0:
            return 0, 0
        
        # 备用市场竞争分析
        reserve_scarcity = reserve_requirement / max(1, max_capacity * 0.3)
        
        if reserve_scarcity > 1.5:  # 备用稀缺
            price_premium = 1.15
            quantity_factor = 1.2
        elif reserve_scarcity < 0.5:  # 备用充足
            price_premium = 0.95
            quantity_factor = 0.8
        else:
            price_premium = 1.0
            quantity_factor = 1.0
        
        reserve_price = max(reserve_cost * 1.05, 
                           last_price * price_premium)
        reserve_quantity = min(max_capacity * 0.4, 
                              last_qty * quantity_factor)
        
        return reserve_price, reserve_quantity
    
    def _generate_heuristic_carbon_bid(self, market_info: Dict) -> Dict:
        """
        生成启发式碳市场报价，基于学习到的策略
        增加随机性以避免固定交易模式，同时更好地保持LLM学习的策略
        """
        import random
        
        current_carbon_price = market_info.get('carbon_price', 50.0)
        current_funds = market_info.get('agent_funds', {}).get(self.agent_id, 1000.0)
        current_quota = market_info.get('agent_quotas', {}).get(self.agent_id, 100.0)
        
        # 获取当前策略，如果没有则使用默认值
        strategy = self.current_carbon_strategy
        
        if strategy is None:
            # 动态随机基础数量，避免固定模式
            buy_quantity_base = random.uniform(5, 15)  # 随机基础买入量
            sell_quantity_base = random.uniform(5, 15)  # 随机基础卖出量
            buy_price_factor = random.uniform(0.98, 1.04)   # 随机买入价格因子
            sell_price_factor = random.uniform(0.96, 1.02)  # 随机卖出价格因子
            logging.info(f"智能体 {self.agent_id} ADF碳市场: 使用随机初始策略")
        else:
            # 基于学习到的策略，但增加适度随机性
            buy_quantity_base = strategy['buy_quantity_base'] * random.uniform(0.8, 1.2)
            sell_quantity_base = strategy['sell_quantity_base'] * random.uniform(0.8, 1.2)
            buy_price_factor = strategy['buy_price_factor'] * random.uniform(0.98, 1.02)
            sell_price_factor = strategy['sell_price_factor'] * random.uniform(0.98, 1.02)
            logging.info(f"智能体 {self.agent_id} ADF碳市场: 基于学习策略 buy_base={strategy['buy_quantity_base']:.1f}, sell_base={strategy['sell_quantity_base']:.1f}")
            
        # 增加智能体个性化随机因子
        agent_randomness = hash(self.agent_id) % 1000 / 1000.0  # 基于智能体ID的固定随机性
        personal_buy_factor = 0.7 + agent_randomness * 0.6  # 0.7-1.3范围
        personal_sell_factor = 0.7 + (1 - agent_randomness) * 0.6  # 反向关联
        
        # 每轮随机调整因子
        round_randomness = random.uniform(0.8, 1.2)
        
        # 根据资金和配额情况智能调整
        funds_ratio = current_funds / max(1000, current_carbon_price * 20)  # 资金充裕度
        quota_ratio = current_quota / max(50, current_quota + 10)  # 配额充裕度
        
        # 资金充裕时更积极买入，配额充裕时更积极卖出
        if funds_ratio > 1.5:  # 资金充裕
            buy_quantity_base *= 1.3
            sell_quantity_base *= 0.8
        elif funds_ratio < 0.5:  # 资金紧张
            buy_quantity_base *= 0.5
            sell_quantity_base *= 1.2
            
        if quota_ratio > 0.8:  # 配额充裕
            sell_quantity_base *= 1.4
            buy_quantity_base *= 0.7
        elif quota_ratio < 0.3:  # 配额不足
            buy_quantity_base *= 1.5
            sell_quantity_base *= 0.6
        
        # 计算最终数量（加入多层随机性和智能调整）
        final_buy_qty = buy_quantity_base * personal_buy_factor * round_randomness
        final_sell_qty = sell_quantity_base * personal_sell_factor * round_randomness
        
        # 根据价格趋势调整策略
        if len(self.price_history) >= 3:
            # price_history 为 deque，先转为 list 再切片
            recent_prices = list(self.price_history)[-3:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
            
            if price_trend > 0.05:  # 价格上涨趋势，更积极买入
                buy_price_factor *= 1.02
                final_buy_qty *= 1.2
            elif price_trend < -0.05:  # 价格下跌趋势，更积极卖出
                sell_price_factor *= 0.98
                final_sell_qty *= 1.2
        
        # 确保价格因子在合理范围内
        buy_price_factor = max(0.95, min(1.08, buy_price_factor))
        sell_price_factor = max(0.92, min(1.05, sell_price_factor))
        
        # 随机调整买卖价格关系，但大部分时候确保买价高于卖价
        if random.random() < 0.8:  # 80%概率确保买价高于卖价
            if buy_price_factor < sell_price_factor:
                avg_factor = (buy_price_factor + sell_price_factor) / 2
                random_spread = random.uniform(0.005, 0.02)
                buy_price_factor = avg_factor + random_spread
                sell_price_factor = avg_factor - random_spread
        
        # 资金约束检查
        safety_margin = random.uniform(0.85, 0.95)  # 随机安全边际
        proposed_cost = final_buy_qty * current_carbon_price * buy_price_factor
        if proposed_cost > current_funds:
            final_buy_qty = current_funds / (current_carbon_price * buy_price_factor) * safety_margin
        
        # 返回与LLM决策格式匹配的碳市场报价
        carbon_bid = {
            'buy_orders': [],
            'sell_orders': []
        }
        
        # 调试日志：输出关键参数
        logging.info(f"智能体 {self.agent_id} ADF碳市场报价生成:")
        logging.info(f"  final_buy_qty: {final_buy_qty:.2f}, final_sell_qty: {final_sell_qty:.2f}")
        logging.info(f"  buy_price_factor: {buy_price_factor:.3f}, sell_price_factor: {sell_price_factor:.3f}")
        logging.info(f"  funds_ratio: {funds_ratio:.2f}, quota_ratio: {quota_ratio:.2f}")
        
        # 智能买单生成条件
        buy_threshold = random.uniform(0.5, 1.0)  # 随机买单阈值
        buy_cost_required = final_buy_qty * current_carbon_price * buy_price_factor
        buy_condition1 = final_buy_qty > buy_threshold
        buy_condition2 = current_funds > buy_cost_required
        buy_condition3 = random.random() > 0.1  # 90%概率生成买单
        
        if buy_condition1 and buy_condition2 and buy_condition3:
            buy_order = {
                'quantity': round(final_buy_qty, 1),
                'price': round(current_carbon_price * buy_price_factor, 2)
            }
            carbon_bid['buy_orders'].append(buy_order)
            logging.info(f"  生成买单: {buy_order}")
        else:
            logging.info(f"  跳过买单生成 - 条件: qty={buy_condition1}, funds={buy_condition2}, random={buy_condition3}")
        
        # 智能卖单生成条件
        sell_threshold = random.uniform(0.5, 1.0)  # 随机卖单阈值
        sell_condition1 = final_sell_qty > sell_threshold
        sell_condition2 = current_quota > 0
        sell_condition3 = random.random() > 0.1  # 90%概率生成卖单
        sell_quantity = min(final_sell_qty, current_quota)
        
        if sell_condition1 and sell_condition2 and sell_condition3:
            sell_order = {
                'quantity': round(sell_quantity, 1),
                'price': round(current_carbon_price * sell_price_factor, 2)
            }
            carbon_bid['sell_orders'].append(sell_order)
            logging.info(f"  生成卖单: {sell_order}")
        else:
            logging.info(f"  跳过卖单生成 - 条件: qty={sell_condition1}, quota={sell_condition2}, random={sell_condition3}")
        
        return carbon_bid
    
    def _generate_simple_cost_bid(self, market_info: Dict, private_info: Dict) -> Dict:
        """简单成本加成报价（无LLM基线时使用）"""
        energy_cost = private_info['marginal_cost_energy']
        reserve_cost = private_info['marginal_cost_reserve']
        max_capacity = private_info['max_capacity']
        
        # 简单加成策略
        energy_price = energy_cost * 1.15  # 15%加成
        reserve_price = reserve_cost * 1.20  # 20%加成
        
        # 简单容量分配
        reserve_requirement = market_info.get('reserve_requirement', 0)
        if reserve_requirement > 0:
            energy_quantity = max_capacity * 0.7
            reserve_quantity = max_capacity * 0.3
        else:
            energy_quantity = max_capacity
            reserve_quantity = 0
        
        carbon_bid = self._generate_heuristic_carbon_bid(market_info)
        
        return {
            'electricity': {
                'energy_bid': {'price': energy_price, 'quantity': energy_quantity},
                'reserve_bid': {'price': reserve_price, 'quantity': reserve_quantity}
            },
            'carbon': carbon_bid,
            'decision_type': 'heuristic',
            'markup_rate': self.current_markup_rate,
            'capacity_split': self.current_capacity_split.copy()
        }
    
    def update_llm_strategy(self, llm_decision: Dict, private_info: Dict = None):
        """
        更新上次LLM决策的策略信息，用于后续启发式决策参考
        
        Args:
            llm_decision: LLM决策结果
            private_info: 智能体私有信息（用于计算加价率）
        """
        # 使用深拷贝避免外部修改影响ADF内部基线
        self.last_llm_strategy = copy.deepcopy(llm_decision)
        
        # 从LLM决策中提取策略参数
        if 'electricity' in llm_decision:
            elec_bid = llm_decision['electricity']
            energy_bid = elec_bid.get('energy_bid', {})
            reserve_bid = elec_bid.get('reserve_bid', {})
            
            # 更新加价率（基于LLM决策的价格）
            energy_price = energy_bid.get('price', 0)
            if energy_price > 0 and private_info:
                cost = private_info.get('marginal_cost_energy', 0)
                if cost > 0:
                    self.current_markup_rate = max(0.02, (energy_price / cost) - 1)
            
            # 更新容量分配比例
            energy_qty = energy_bid.get('quantity', 0)
            reserve_qty = reserve_bid.get('quantity', 0)
            total_qty = energy_qty + reserve_qty
            
            if total_qty > 0:
                self.current_capacity_split = {
                    'energy': energy_qty / total_qty,
                    'reserve': reserve_qty / total_qty
                }
        
        # 学习碳市场策略参数
        if 'carbon' in llm_decision:
            carbon_bid = llm_decision['carbon']
            
            # 兼容两种格式：新格式(buy_orders/sell_orders)和旧格式(buy_quantity/sell_quantity)
            buy_orders = carbon_bid.get('buy_orders', [])
            sell_orders = carbon_bid.get('sell_orders', [])
            
            # 如果没有buy_orders格式，尝试从旧格式转换
            if not buy_orders and 'buy_quantity' in carbon_bid and 'buy_price' in carbon_bid:
                buy_qty = carbon_bid.get('buy_quantity', 0)
                buy_price = carbon_bid.get('buy_price', 0)
                if buy_qty > 0 and buy_price > 0:
                    buy_orders = [{'quantity': buy_qty, 'price': buy_price}]
            
            if not sell_orders and 'sell_quantity' in carbon_bid and 'sell_price' in carbon_bid:
                sell_qty = carbon_bid.get('sell_quantity', 0)
                sell_price = carbon_bid.get('sell_price', 0)
                if sell_qty > 0 and sell_price > 0:
                    sell_orders = [{'quantity': sell_qty, 'price': sell_price}]
            
            # 学习买入策略
            if buy_orders and len(buy_orders) > 0:
                # 计算总买入量和平均价格
                total_buy_qty = sum(order.get('quantity', 0) for order in buy_orders)
                total_buy_value = sum(order.get('quantity', 0) * order.get('price', 0) for order in buy_orders)
                avg_buy_price = total_buy_value / total_buy_qty if total_buy_qty > 0 else 0
                
                # 保守学习买入量（小学习率，避免剧烈变化）
                if total_buy_qty > 0:
                    alpha = 0.1  # 降低学习率
                    new_buy_qty = max(1.0, min(15.0, total_buy_qty))  # 限制范围
                    if not self.carbon_strategy_initialized:
                        self.current_carbon_strategy = {
                            'buy_quantity_base': new_buy_qty,
                            'sell_quantity_base': 5.0,
                            'buy_price_factor': 0.98,
                            'sell_price_factor': 1.02,
                            'quantity_adjustment': 1.0,
                            'price_spread': 0.02
                        }
                        self.carbon_strategy_initialized = True
                    else:
                        self.current_carbon_strategy['buy_quantity_base'] = (
                            self.current_carbon_strategy['buy_quantity_base'] * (1 - alpha) + 
                            new_buy_qty * alpha
                        )
                
                # 学习买入价格策略
                current_carbon_price = 50.0  # 使用固定参考价格，避免价格波动影响
                if avg_buy_price > 0 and current_carbon_price > 0:
                    buy_price_factor = avg_buy_price / current_carbon_price
                    # 确保价格因子在合理范围内
                    buy_price_factor = max(0.92, min(1.03, buy_price_factor))
                    alpha = 0.1
                    if not self.carbon_strategy_initialized:
                        # 如果策略未初始化，需要先创建完整策略
                        if self.current_carbon_strategy is None:
                            self.current_carbon_strategy = {
                                'buy_quantity_base': 5.0,
                                'sell_quantity_base': 5.0,
                                'buy_price_factor': buy_price_factor,
                                'sell_price_factor': 1.02,
                                'quantity_adjustment': 1.0,
                                'price_spread': 0.02
                            }
                            self.carbon_strategy_initialized = True
                        else:
                            self.current_carbon_strategy['buy_price_factor'] = buy_price_factor
                    else:
                        self.current_carbon_strategy['buy_price_factor'] = (
                            self.current_carbon_strategy['buy_price_factor'] * (1 - alpha) + 
                            buy_price_factor * alpha
                        )
            
            # 学习卖出策略
            if sell_orders and len(sell_orders) > 0:
                # 计算总卖出量和平均价格
                total_sell_qty = sum(order.get('quantity', 0) for order in sell_orders)
                total_sell_value = sum(order.get('quantity', 0) * order.get('price', 0) for order in sell_orders)
                avg_sell_price = total_sell_value / total_sell_qty if total_sell_qty > 0 else 0
                
                # 保守学习卖出量
                if total_sell_qty > 0:
                    alpha = 0.1
                    new_sell_qty = max(1.0, min(15.0, total_sell_qty))  # 限制范围
                    if not self.carbon_strategy_initialized:
                        # 确保策略字典已初始化
                        if self.current_carbon_strategy is None:
                            self.current_carbon_strategy = {
                                'buy_quantity_base': 5.0,
                                'sell_quantity_base': new_sell_qty,
                                'buy_price_factor': 0.98,
                                'sell_price_factor': 1.02,
                                'quantity_adjustment': 1.0,
                                'price_spread': 0.02
                            }
                            self.carbon_strategy_initialized = True
                        else:
                            self.current_carbon_strategy['sell_quantity_base'] = new_sell_qty
                    else:
                        self.current_carbon_strategy['sell_quantity_base'] = (
                            self.current_carbon_strategy['sell_quantity_base'] * (1 - alpha) + 
                            new_sell_qty * alpha
                        )
                
                # 学习卖出价格策略
                current_carbon_price = 50.0  # 使用固定参考价格
                if avg_sell_price > 0 and current_carbon_price > 0:
                    sell_price_factor = avg_sell_price / current_carbon_price
                    # 确保价格因子在合理范围内
                    sell_price_factor = max(0.97, min(1.08, sell_price_factor))
                    alpha = 0.1
                    if not self.carbon_strategy_initialized:
                        # 确保策略字典已初始化
                        if self.current_carbon_strategy is None:
                            self.current_carbon_strategy = {
                                'buy_quantity_base': 5.0,
                                'sell_quantity_base': 5.0,
                                'buy_price_factor': 0.98,
                                'sell_price_factor': sell_price_factor,
                                'quantity_adjustment': 1.0,
                                'price_spread': 0.02
                            }
                            self.carbon_strategy_initialized = True
                        else:
                            self.current_carbon_strategy['sell_price_factor'] = sell_price_factor
                    else:
                        self.current_carbon_strategy['sell_price_factor'] = (
                            self.current_carbon_strategy['sell_price_factor'] * (1 - alpha) + 
                            sell_price_factor * alpha
                        )
            
            # 最终参数安全检查
            if self.carbon_strategy_initialized:
                self.current_carbon_strategy['buy_quantity_base'] = max(2.0, min(12.0, 
                    self.current_carbon_strategy['buy_quantity_base']))
                self.current_carbon_strategy['sell_quantity_base'] = max(2.0, min(12.0, 
                    self.current_carbon_strategy['sell_quantity_base']))
                self.current_carbon_strategy['buy_price_factor'] = max(0.95, min(1.02, 
                    self.current_carbon_strategy['buy_price_factor']))
                self.current_carbon_strategy['sell_price_factor'] = max(0.98, min(1.05, 
                    self.current_carbon_strategy['sell_price_factor']))
                
                logging.info(f"智能体 {self.agent_id} 学习碳市场策略: "
                            f"买入量={self.current_carbon_strategy['buy_quantity_base']:.1f}, "
                            f"卖出量={self.current_carbon_strategy['sell_quantity_base']:.1f}, "
                            f"买入因子={self.current_carbon_strategy['buy_price_factor']:.3f}, "
                            f"卖出因子={self.current_carbon_strategy['sell_price_factor']:.3f}")
            else:
                logging.info(f"智能体 {self.agent_id} 碳市场策略尚未初始化，等待有效的LLM碳市场决策")
        
        logging.debug(
            "ADF更新LLM基线 - 电力: "
            f"energy_price={energy_bid.get('price', 0) if 'electricity' in llm_decision else 0:.4f}, "
            f"energy_qty={energy_bid.get('quantity', 0) if 'electricity' in llm_decision else 0:.4f}; "
            f"reserve_price={reserve_bid.get('price', 0) if 'electricity' in llm_decision else 0:.4f}, "
            f"reserve_qty={reserve_bid.get('quantity', 0) if 'electricity' in llm_decision else 0:.4f}; "
            f"computed_markup_rate={self.current_markup_rate:.4f}, capacity_split={self.current_capacity_split}"
        )
        
        logging.info(f"智能体 {self.agent_id} 更新LLM策略基线: 加价率{self.current_markup_rate:.3f}, "
                     f"容量分配{self.current_capacity_split}")
    
    def update_performance_data(self, round_number: int, profit: float, market_results: Dict):
        """
        更新性能数据，用于后续ADF判断
        
        Args:
            round_number: 轮次
            profit: 当前轮次利润
            market_results: 市场结果
        """
        # 利润数据在should_trigger_llm_decision中已更新
        
        # 更新价格历史
        if 'electricity_market' in market_results:
            clearing_price = market_results['electricity_market'].get('clearing_price', 0)
            if clearing_price > 0:
                self.price_history.append(clearing_price)
    
    def get_adf_status(self) -> Dict:
        """
        获取ADF模块当前状态信息
        
        Returns:
            ADF状态字典
        """
        total_decisions = self.llm_decision_count + self.heuristic_decision_count
        llm_decision_ratio = self.llm_decision_count / total_decisions if total_decisions > 0 else 0
        
        return {
            'last_llm_round': self.last_llm_decision_round,
            'llm_decision_count': self.llm_decision_count,
            'heuristic_decision_count': self.heuristic_decision_count,
            'llm_decision_ratio': llm_decision_ratio,
            'current_markup_rate': self.current_markup_rate,
            'capacity_split': self.current_capacity_split.copy(),
            'profit_history_size': len(self.profit_history),
            'price_history_size': len(self.price_history),
            'config': self.config.copy()
        }
