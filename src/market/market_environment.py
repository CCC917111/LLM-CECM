# llm_market_sim/market/market_environment.py
import logging
from typing import List, Dict, Tuple, Optional

# 导入选择的出清机制
from .clearing_mechanism import simple_joint_clearing, opf_joint_clearing
# 导入验证工具
from utils.validation import BidValidator

class MarketEnvironment:
    """
    模拟电力现货与备用联合市场的环境。
    负责接收报价、执行市场出清、计算利润并返回结果。
    """
    def __init__(self, config: Dict):
        """
        初始化市场环境。

        Args:
            config: 市场配置字典。例如:
                {
                    "demand_profile": [float], # 每轮的需求列表
                    "reserve_profile": [float], # 每轮的备用需求列表
                    "clearing_mechanism": "simple" # 或 "milp"
                }
        """
        self.config = config
        self.demand_profile = config.get("demand_profile", [500] * 10) # 默认10轮，每轮500MW
        self.reserve_profile = config.get("reserve_profile", [50] * 10) # 默认10轮，每轮50MW
        self.num_rounds = len(self.demand_profile)

        if len(self.demand_profile) != len(self.reserve_profile):
            raise ValueError("需求曲线和备用需求曲线的长度必须一致。")

        # 选择出清算法
        clearing_func_name = config.get("clearing_mechanism", "simple")
        if clearing_func_name == "simple":
            self.clear_market = simple_joint_clearing
            logging.info("市场环境使用 'simple_joint_clearing' 出清机制。")
        elif clearing_func_name == "opf":
            self.clear_market = opf_joint_clearing
            logging.info("市场环境使用 'opf_joint_clearing' 出清机制（考虑网络约束）。")
        # elif clearing_func_name == "milp":
        #     self.clear_market = milp_joint_clearing
        #     logging.info("市场环境使用 'milp_joint_clearing' 出清机制。")
        else:
            logging.warning(f"未知的出清机制: {clearing_func_name}，将使用默认的simple出清机制")
            self.clear_market = simple_joint_clearing

        # 存储上一轮的市场价格，供智能体参考
        self.last_market_prices = {'energy_price': 0.0, 'reserve_price': 0.0}

    def reset(self) -> None:
        """
        重置市场环境的内部状态。
        当前仅重置上一轮市场价格记录，以便训练环境在每个episode开始时有一致的初始观测。
        """
        self.last_market_prices = {'energy_price': 0.0, 'reserve_price': 0.0}

    def get_market_info(self, round_number: int, num_competitors: int) -> Dict:
        """
        获取当前轮次的市场公开信息。

        Args:
            round_number: 当前轮次 (从0开始)。
            num_competitors: 竞争对手数量 (供智能体参考)。

        Returns:
            包含市场信息的字典。
        """
        if round_number < 0 or round_number >= self.num_rounds:
             raise IndexError(f"请求的轮次 {round_number} 超出范围 [0, {self.num_rounds - 1}]。")

        return {
            "round_number": round_number,
            "total_demand": self.demand_profile[round_number],
            "reserve_requirement": self.reserve_profile[round_number],
            "num_competitors": num_competitors,
            "last_energy_price": self.last_market_prices['energy_price'],
            "last_reserve_price": self.last_market_prices['reserve_price'],
        }

    def get_public_info(self, round_number: int) -> Dict:
        """
        获取当前轮次的市场公开信息（不含竞争对手数量，兼容耦合市场流程）。
        Args:
            round_number: 当前轮次 (从0开始)。
        Returns:
            包含市场信息的字典。
        """
        if round_number < 0 or round_number >= self.num_rounds:
            raise IndexError(f"请求的轮次 {round_number} 超出范围 [0, {self.num_rounds - 1}]。")
        return {
            "round_number": round_number,
            "total_demand": self.demand_profile[round_number],
            "reserve_requirement": self.reserve_profile[round_number],
            "last_energy_price": self.last_market_prices['energy_price'],
            "last_reserve_price": self.last_market_prices['reserve_price'],
        }

    def run_round(self, round_number: int, agent_bids: List[Dict], agent_private_info: Dict[str, Dict]) -> Tuple[Dict, Dict]:
        """
        运行一轮市场模拟。

        Args:
            round_number: 当前轮次。
            agent_bids: 所有智能体提交的报价列表。
                        每个元素是 {'agent_id': str, 'bid': {'energy_bid':..., 'reserve_bid':...}}
            agent_private_info: 包含所有智能体私有信息的字典，键是 agent_id。
                                用于计算利润。{'agent_id': {'marginal_cost_energy': ..., 'marginal_cost_reserve': ...}}

        Returns:
            一个元组 (market_results, agent_outcomes):
            - market_results: 包含市场价格的字典 {'energy_price': float, 'reserve_price': float}
            - agent_outcomes: 包含每个智能体结果的字典，键是 agent_id。
                             {'agent_id': {'cleared_energy': float, 'cleared_reserve': float, 'profit': float, 'submitted_bid': dict}}
        """
        if round_number < 0 or round_number >= self.num_rounds:
             raise IndexError(f"运行的轮次 {round_number} 超出范围 [0, {self.num_rounds - 1}]。")

        current_demand = self.demand_profile[round_number]
        current_reserve = self.reserve_profile[round_number]
        logging.info(f"\n--- 市场环境：开始第 {round_number} 轮 ---")
        logging.info(f"本轮需求: {current_demand:.2f} MW, 备用需求: {current_reserve:.2f} MW")
        logging.info(f"收到 {len(agent_bids)} 个智能体的报价。")

        # --- 准备出清所需的数据 ---
        bids_for_clearing = []
        submitted_bids_map = {} # 存储原始提交的报价，用于返回给智能体
        for agent_bid_info in agent_bids:
            agent_id = agent_bid_info['agent_id']
            bid = agent_bid_info['bid']
            # 获取该智能体的容量信息
            max_capacity = agent_private_info.get(agent_id, {}).get('max_capacity', 0)
            if max_capacity <= 0:
                 logging.warning(f"智能体 {agent_id} 容量为0或未提供，无法参与市场。")
                 continue

            # 使用BidValidator验证报价
            if not BidValidator.validate_bid(agent_id, bid, {'max_capacity': max_capacity}):
                logging.error(f"智能体 {agent_id} 提交了无效的报价，将无法参与本轮市场。")
                continue

            bid_entry = {
                'agent_id': agent_id,
                'energy_bid': bid['energy_bid'],
                'reserve_bid': bid['reserve_bid'],
                'max_capacity': max_capacity
            }
            bids_for_clearing.append(bid_entry)
            # 存储原始报价（包括可能存在的belief状态）
            submitted_bids_map[agent_id] = bid


        # --- 执行市场出清 ---
        try:
            clearing_prices, agent_clearance = self.clear_market(
                bids=bids_for_clearing,
                total_demand=current_demand,
                reserve_requirement=current_reserve
            )
            # 更新上一轮市场价格记录
            self.last_market_prices = clearing_prices.copy()

        except Exception as e:
            logging.error(f"市场出清过程中发生错误: {e}", exc_info=True)
            # 出错时，返回默认结果，避免仿真中断
            clearing_prices = {'energy_price': 0, 'reserve_price': 0}
            agent_clearance = {bid['agent_id']: {'cleared_energy': 0, 'cleared_reserve': 0, 'total_cleared': 0} for bid in bids_for_clearing}


        # --- 计算每个智能体的利润并整理结果 ---
        agent_outcomes = {}
        total_cleared_energy_check = 0
        total_cleared_reserve_check = 0

        for agent_id, clearance_result in agent_clearance.items():
            cleared_energy = clearance_result.get('cleared_energy', 0)
            cleared_reserve = clearance_result.get('cleared_reserve', 0)
            total_cleared_check = clearance_result.get('total_cleared', 0) # 来自出清模块的总量

            # 再次校验总量是否超容量
            agent_cap = agent_private_info.get(agent_id, {}).get('max_capacity', 0)
            if cleared_energy + cleared_reserve > agent_cap * 1.001:
                 logging.error(f"FATAL: Agent {agent_id} 出清总量 {cleared_energy + cleared_reserve:.2f} 超过容量 {agent_cap:.2f} 在利润计算前发现！")
                 # 这里可能需要修正出清量，但简化版假设出清模块已处理好

            total_cleared_energy_check += cleared_energy
            total_cleared_reserve_check += cleared_reserve

            # 获取成本信息
            costs = agent_private_info.get(agent_id, {})
            cost_energy = costs.get('marginal_cost_energy', 9999) # 若无成本信息则设为高价
            cost_reserve = costs.get('marginal_cost_reserve', 999)

            # 计算收入
            revenue_energy = cleared_energy * clearing_prices['energy_price']
            revenue_reserve = cleared_reserve * clearing_prices['reserve_price']
            total_revenue = revenue_energy + revenue_reserve

            # 计算成本
            # 注意：备用成本通常指提供备用能力的成本，或者是因为提供备用而未能发电的机会成本。
            # 简化计算：成本 = 出清能源量 * 能源成本 + 出清备用量 * 备用成本
            # 更精确的模型可能只对出清的备用容量收费，或者基于能源价格计算备用的机会成本。
            total_cost = (cleared_energy * cost_energy) + (cleared_reserve * cost_reserve)

            # 计算利润
            profit = total_revenue - total_cost

            agent_outcomes[agent_id] = {
                'cleared_energy': cleared_energy,
                'cleared_reserve': cleared_reserve,
                'profit': profit,
                'revenue': total_revenue,
                'cost': total_cost,
                'submitted_bid': submitted_bids_map.get(agent_id) # 将原始报价放回结果中
            }
            logging.debug(f"  智能体 {agent_id}: 清(E:{cleared_energy:.2f}, R:{cleared_reserve:.2f}), "
                          f"价(E:{clearing_prices['energy_price']:.2f}, R:{clearing_prices['reserve_price']:.2f}), "
                          f"成本(E:{cost_energy:.2f}, R:{cost_reserve:.2f}), "
                          f"收入:{total_revenue:.2f}, 成本:{total_cost:.2f}, 利润:{profit:.2f}")

        logging.info(f"本轮出清价格: 能源={clearing_prices['energy_price']:.2f}, 备用={clearing_prices['reserve_price']:.2f}")
        logging.info(f"核对: 总能源出清={total_cleared_energy_check:.2f}, 总备用出清={total_cleared_reserve_check:.2f}")
        logging.info(f"--- 市场环境：结束第 {round_number} 轮 ---")

        return clearing_prices, agent_outcomes