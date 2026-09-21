# llm_market_sim/agents/belief_module.py
import logging
from typing import Dict, List, Any, Optional, Tuple

_ZERO_TOKENS = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


class BeliefModule:
    """
    管理智能体对竞争对手与市场趋势的信念 B_i = {θ̂_{-i}, λ̂_market}（论文 Eq.(7)）。

    两层更新：
    1. 数值层（每轮，零LLM成本）：市场出清后按启发式规则微调对竞争对手平均成本的估计，
       同时把本轮观测暂存到 pending_observations。
    2. 语义层（论文中的语义转移算子 T: B × Ω_pub → B）：智能体进入深度决策（Deep Mode）
       前调用 revise_with_llm()，由LLM结合上一版信念、暂存的观测和公开信息（含政策公告）
       生成新的信念。与ADF的"理性疏忽"一致，只有深度决策时才付出这部分认知成本；
       Fast Mode 下仅做数值层更新。LLM调用失败时保留数值层结果，不影响仿真。
    """

    def __init__(self, agent_id: str, competitor_ids: List[str], initial_belief_params: Optional[Dict] = None,
                 llm_client: Any = None, enable_llm_update: bool = False, max_pending_observations: int = 10):
        """
        Args:
            agent_id: 当前智能体的ID。
            competitor_ids: 竞争对手的ID列表。
            initial_belief_params: 初始信念参数，例如:
                {
                    "avg_cost_estimation_mean": 40.0, # 初始估计竞争对手平均成本的均值
                    "avg_cost_estimation_std": 15.0,  # 初始估计竞争对手平均成本的标准差
                    "update_aggressiveness": 0.1      # 数值层更新的学习率/幅度
                }
            llm_client: 用于语义层信念转移的LLM客户端。
            enable_llm_update: 是否启用语义层（LLM）信念转移。
            max_pending_observations: 两次语义更新之间最多保留的观测条数。
        """
        self.agent_id = agent_id
        self.competitor_ids = competitor_ids
        self.params = initial_belief_params or {}
        self.learning_rate = self.params.get("update_aggressiveness", 0.1)

        # 核心信念：对竞争对手平均边际成本的估计 (θ̂_{-i})
        self.estimated_competitor_avg_cost_mean = self.params.get("avg_cost_estimation_mean", 40.0)
        self.estimated_competitor_avg_cost_std = self.params.get("avg_cost_estimation_std", 15.0)
        # 对市场趋势的判断 (λ̂_market)
        self.market_trend_estimate = "尚无判断"

        self.belief_summary_text = f"初始估计：竞争对手平均成本可能在 {self.estimated_competitor_avg_cost_mean - self.estimated_competitor_avg_cost_std:.1f} 到 {self.estimated_competitor_avg_cost_mean + self.estimated_competitor_avg_cost_std:.1f} 范围内。"

        # 语义层（LLM）配置与状态
        self.llm_client = llm_client
        self.enable_llm_update = bool(enable_llm_update and llm_client is not None)
        self.max_pending_observations = max(1, int(max_pending_observations))
        self.pending_observations: List[Dict] = []
        self.last_llm_revision_round = -1
        self.llm_revision_count = 0

        logging.info(f"智能体 {agent_id} 的信念模块初始化完成（LLM语义转移: {'启用' if self.enable_llm_update else '禁用'}）。初始信念: {self.belief_summary_text}")

    def get_belief_summary(self) -> str:
        """获取当前信念的文本摘要（供决策Prompt使用）"""
        return self.belief_summary_text

    def get_estimated_competitor_costs(self) -> Dict:
        """返回当前对竞争对手成本的数值估计"""
        return {
            "mean": self.estimated_competitor_avg_cost_mean,
            "std": self.estimated_competitor_avg_cost_std
        }

    def get_belief_state(self) -> Dict:
        """返回完整的信念状态（数值 + 趋势 + 文本），便于记录与分析"""
        return {
            "competitor_cost_mean": self.estimated_competitor_avg_cost_mean,
            "competitor_cost_std": self.estimated_competitor_avg_cost_std,
            "market_trend": self.market_trend_estimate,
            "summary": self.belief_summary_text,
            "llm_revision_count": self.llm_revision_count,
        }

    # ------------------------------------------------------------------
    # 数值层：每轮市场出清后调用
    # ------------------------------------------------------------------
    def update_beliefs(self, round_number: int, market_results: Dict, my_bid_result: Dict):
        """
        根据市场结果做数值层（启发式）信念更新，并暂存本轮观测供语义层使用。

        Args:
            round_number: 当前轮次。
            market_results: 包含市场出清价格等的字典。
                {'energy_price': float, 'reserve_price': float, 'carbon_market': {...}, ...}
            my_bid_result: 当前智能体在本轮的报价和中标结果。
                {'energy_bid': {'price': P_e, 'quantity': Q_e}, 'reserve_bid': {...},
                 'cleared_energy': float, 'cleared_reserve': float, 'profit': float}
        """
        energy_price = market_results.get('energy_price')
        reserve_price = market_results.get('reserve_price')
        carbon_price = (market_results.get('carbon_market') or {}).get('current_price')
        my_energy_bid = my_bid_result.get('energy_bid', {}) or {}
        my_energy_bid_price = my_energy_bid.get('price')
        my_cleared_energy = my_bid_result.get('cleared_energy', 0) or 0
        my_energy_bid_quantity = my_energy_bid.get('quantity', 0) or 0

        # 暂存观测（语义层使用）
        self.pending_observations.append({
            "round": round_number,
            "energy_price": energy_price,
            "reserve_price": reserve_price,
            "carbon_price": carbon_price,
            "my_energy_bid_price": my_energy_bid_price,
            "my_energy_bid_quantity": my_energy_bid_quantity,
            "my_cleared_energy": my_cleared_energy,
            "profit": my_bid_result.get('profit'),
        })
        if len(self.pending_observations) > self.max_pending_observations:
            self.pending_observations = self.pending_observations[-self.max_pending_observations:]

        if None in [energy_price, my_energy_bid_price]:
            logging.warning(f"智能体 {self.agent_id} 信念更新跳过：缺少必要的市场价格或自身报价信息。")
            return

        logging.debug(f"智能体 {self.agent_id} 开始更新信念 (轮次 {round_number})...")
        logging.debug(f"  市场能源价格: {energy_price:.2f}, 我的能源报价: {my_energy_bid_price:.2f}, 中标量: {my_cleared_energy:.2f}/{my_energy_bid_quantity:.2f}")

        # --- 启发式更新逻辑 ---
        # 场景1: 市场价格显著高于我的报价，且我中标了 -> 倾向于认为对手成本高
        if my_cleared_energy > 0 and energy_price > my_energy_bid_price * 1.1:
            cost_adjustment_factor = 1.0
        # 场景2: 市场价格低于我的报价，导致未中标或部分中标 -> 倾向于认为对手成本低
        elif my_cleared_energy < my_energy_bid_quantity * 0.9 and energy_price < my_energy_bid_price:
            cost_adjustment_factor = -1.0
        # 场景3: 市场价格略高于我的报价，且我中标 -> 轻微倾向对手成本较高
        elif my_cleared_energy > 0 and my_energy_bid_price <= energy_price <= my_energy_bid_price * 1.1:
            cost_adjustment_factor = 0.2
        # 场景4: 其他情况
        else:
            cost_adjustment_factor = 0.0

        old_mean = self.estimated_competitor_avg_cost_mean
        self.estimated_competitor_avg_cost_mean += cost_adjustment_factor * self.estimated_competitor_avg_cost_std * self.learning_rate
        self.estimated_competitor_avg_cost_mean = max(5.0, self.estimated_competitor_avg_cost_mean)  # 假设成本至少为5

        logging.info(f"智能体 {self.agent_id} 更新后信念：估计竞争对手平均成本均值从 {old_mean:.2f} 调整为 {self.estimated_competitor_avg_cost_mean:.2f}")

        # 只有在尚未得到LLM语义信念时，才用规则生成文本摘要；
        # 已有LLM信念时保留其文字判断，只在末尾附上最新数值，避免覆盖语义信息
        numeric_note = (f"(最新数值估计: 竞争对手平均成本约 {max(0, self.estimated_competitor_avg_cost_mean - self.estimated_competitor_avg_cost_std):.1f}"
                        f"-{self.estimated_competitor_avg_cost_mean + self.estimated_competitor_avg_cost_std:.1f} 元/MWh，第{round_number}轮能源价 {energy_price:.1f})")
        if self.llm_revision_count == 0:
            self.belief_summary_text = f"基于第{round_number}轮结果(能源价:{energy_price:.1f})，更新估计：竞争对手平均成本可能在 {max(0, self.estimated_competitor_avg_cost_mean - self.estimated_competitor_avg_cost_std):.1f} 到 {self.estimated_competitor_avg_cost_mean + self.estimated_competitor_avg_cost_std:.1f} 范围内。"
        else:
            base = self.belief_summary_text.split("\n(最新数值估计")[0]
            self.belief_summary_text = f"{base}\n{numeric_note}"

    # ------------------------------------------------------------------
    # 语义层：论文 Eq.(7) B_{i,t} ← T(B_{i,t-1}, Ω_t^pub, Outcome_{t-1})
    # ------------------------------------------------------------------
    def revise_with_llm(self, round_number: int, public_info: Dict, private_info: Dict,
                        market_events_info: Optional[str] = None) -> Tuple[Dict, int]:
        """
        在深度决策前调用LLM完成语义信念转移。

        Returns:
            (token_info, llm_calls)。未启用、无新观测或调用失败时返回 (零token, 实际调用次数)。
        """
        if not self.enable_llm_update:
            return dict(_ZERO_TOKENS), 0
        # EventManager 在无事件时返回占位文本，这里视为没有公告
        if market_events_info and "当前无特殊市场事件" in market_events_info:
            market_events_info = None
        # 没有新的市场结果、也没有政策公告，就不必重新推理
        if not self.pending_observations and not market_events_info:
            return dict(_ZERO_TOKENS), 0

        from llm_interface.prompt_manager import PromptManager
        prompt = PromptManager.format_belief_revision_prompt(
            agent_id=self.agent_id,
            round_number=round_number,
            private_info=private_info,
            previous_belief=self.get_belief_state(),
            observations=self.pending_observations,
            public_info=public_info,
            market_events_info=market_events_info,
            num_competitors=len(self.competitor_ids),
        )
        try:
            response, token_info = self.llm_client.chat(prompt, return_json=True, track_tokens=True)
        except Exception as e:
            logging.warning(f"智能体 {self.agent_id} LLM信念转移调用失败，保留数值层信念: {e}")
            return dict(_ZERO_TOKENS), 1

        parsed = self._parse_revision(response)
        if parsed is None:
            logging.warning(f"智能体 {self.agent_id} LLM信念转移结果无法解析，保留数值层信念。原始响应: {str(response)[:300]}")
            return token_info or dict(_ZERO_TOKENS), 1

        old_mean = self.estimated_competitor_avg_cost_mean
        if parsed.get("competitor_cost_mean") is not None:
            self.estimated_competitor_avg_cost_mean = max(0.0, parsed["competitor_cost_mean"])
        if parsed.get("competitor_cost_std") is not None:
            self.estimated_competitor_avg_cost_std = max(0.5, parsed["competitor_cost_std"])
        if parsed.get("market_trend"):
            self.market_trend_estimate = parsed["market_trend"]
        summary = parsed.get("belief_summary") or self.belief_summary_text
        self.belief_summary_text = (
            f"{summary}\n市场趋势判断: {self.market_trend_estimate}；"
            f"竞争对手平均成本估计 {self.estimated_competitor_avg_cost_mean:.1f}±{self.estimated_competitor_avg_cost_std:.1f} 元/MWh。"
        )
        self.pending_observations = []
        self.last_llm_revision_round = round_number
        self.llm_revision_count += 1
        logging.info(
            f"智能体 {self.agent_id} LLM语义信念转移完成(第{round_number}轮): 对手成本均值 {old_mean:.2f} -> "
            f"{self.estimated_competitor_avg_cost_mean:.2f}, 趋势: {self.market_trend_estimate}"
        )
        return token_info or dict(_ZERO_TOKENS), 1

    @staticmethod
    def _to_float(value) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_revision(self, response: Any) -> Optional[Dict]:
        """解析LLM返回的信念JSON；兼容已解析dict或 {'text': ...} 形式"""
        import json
        import re
        data = None
        if isinstance(response, dict):
            if 'text' in response and len(response) == 1:
                text = re.sub(r"^```json|^```|```$", "", str(response['text']).strip(), flags=re.IGNORECASE).strip()
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    m = re.search(r"\{.*\}", text, re.DOTALL)
                    if m:
                        try:
                            data = json.loads(m.group(0))
                        except json.JSONDecodeError:
                            data = None
            elif response.get("default_response") or response.get("error"):
                data = None
            else:
                data = response
        if not isinstance(data, dict):
            return None
        result = {
            "belief_summary": str(data.get("belief_summary", "")).strip() or None,
            "competitor_cost_mean": self._to_float(data.get("competitor_cost_mean")),
            "competitor_cost_std": self._to_float(data.get("competitor_cost_std")),
            "market_trend": str(data.get("market_trend", "")).strip() or None,
        }
        if not any(v is not None for v in result.values()):
            return None
        return result
