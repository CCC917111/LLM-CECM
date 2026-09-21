"""
统一的Prompt管理模块
集中管理所有LLM交互的prompt模板,便于维护和版本控制
"""

import json
from typing import Dict, Optional, List, Any


class PromptManager:
    """统一管理所有Prompt模板的类"""
    
    # ========================================
    # 耦合市场联合决策 Prompt
    # ========================================
    COUPLED_MARKET_PROMPT_TEMPLATE = """你是一个名为 {agent_id} 的发电企业智能体，目标是在电力和碳市场中最大化长期利润。你是一个理性的市场参与者。

### 你的内部参数:
{private_info_str}

### 当前市场信息 (轮次 {round_number}):
{market_info_str}
{market_events_str}{tech_guidance}{history_str}{cognitive_state_str}
### 🌱 碳市场交易机制
- 当前碳价格: {current_carbon_price:.2f} 元/吨CO2
- 有效报价范围: [{carbon_price_min:.2f} - {carbon_price_max:.2f}] 元/吨
- 你当前持有配额: {current_quota:.1f} 吨CO2
- 建议交易量: 5-20 吨 (即使小额交易也有流动性奖励)
- 🔥 强烈建议每轮都参与碳市场交易以获得额外收益！

### 决策指导原则
**利润最大化策略：**
1. **成本效益分析**: 你的边际成本是制定报价的基准。能源成本: {marginal_cost_energy:.2f} 元/MWh，备用成本: {marginal_cost_reserve:.2f} 元/MWh
2. **预期行动**: 如果收到未来市场变化的公告，应在变化实际发生前适当调整报价，反映预期影响
3. **竞争定价**: 考虑历史价格趋势。报价应在成本基础上加合理利润，但要保持竞争力
4. **容量优化**: 总容量 {max_capacity:.2f} MW，合理分配能源和备用报价以最大化收入
5. **市场分析**: 观察市场价格趋势，根据市场情况调整报价策略
6. **碳配额管理**: 预估发电导致的碳排放，合理买卖配额

**重要约束:**
1. **容量约束**: 能源 + 备用报价总量 ≤ {max_capacity:.2f} MW
2. **成本约束**: 报价应 ≥ 边际成本 + 合理利润边际
3. **预期影响**: 如有市场事件公告，应考虑其对成本和市场的潜在影响
4. **碳排放**: 考虑排放因子 {emission_factor:.2f}，预估配额需求
5. **碳价格约束**: 碳报价必须在 [{carbon_price_min:.2f}, {carbon_price_max:.2f}] 范围内
6. **备用市场状态**: 本轮备用需求为 {reserve_requirement:.2f} MW {reserve_market_status}

**报价建议：**
- 能源报价: 在边际成本基础上加合理利润
{reserve_bid_guidance}- 数量分配: 根据市场情况合理分配能源和备用容量
- 碳交易策略: 根据预期排放和持有配额决定买入或卖出

请基于以上分析，制定理性的报价策略。输出严格的JSON格式，不要添加其他文字。

```json
{{
  "reasoning": "在这里简要描述你的决策原因，比如你对市场价格的判断、你的成本、以及你的报价策略是如何结合这些因素的。",
  "electricity_bid": {{
    "energy_bid": {{"price": <你的能源报价价格>, "quantity": <你的能源报价数量>}}{reserve_bid_format}
  }},
  "carbon_bid": {{
    "buy_quantity": <你希望购买的碳配额数量(吨CO2)>,
    "buy_price": <你的碳配额买入价格(元/吨)>,
    "sell_quantity": <你希望卖出的碳配额数量(吨CO2)>,
    "sell_price": <你的碳配额卖出价格(元/吨)>
  }}
}}
```
"""

    # ========================================
    # 认知状态（信念 + 双层记忆）注入块
    # ========================================
    COGNITIVE_STATE_TEMPLATE = """
### 🧠 你的认知状态
**你对竞争对手和市场的当前信念:**
{belief_summary}

**短期工作记忆（最近 {working_memory_k} 轮你的策略、出清价格与利润）:**
{working_memory}

**长期经验总结:**
{semantic_memory}
"""

    # ========================================
    # 信念语义转移 Prompt（论文 Eq.(7) 的 T 算子）
    # ========================================
    BELIEF_REVISION_TEMPLATE = """作为发电商智能体 {agent_id}，你需要在第 {round_number} 轮决策前更新你对竞争对手和市场的信念。
你的技术类型: {technology_type}；能源边际成本: {marginal_cost_energy:.2f} 元/MWh；排放因子: {emission_factor:.2f} 吨CO2/MWh。
市场中共有 {num_competitors} 个竞争对手。

# 你之前的信念:
{previous_belief_summary}
(数值: 竞争对手平均边际成本约 {prev_mean:.2f} ± {prev_std:.2f} 元/MWh；市场趋势判断: {prev_trend})

# 上次更新信念以来观察到的市场结果:
{observations}

# 本轮公开信息:
{public_info}
{market_events_block}
# 任务
结合上述观察和公开信息（尤其是政策公告等非结构化信息），推断：
1. 竞争对手的平均边际成本及其不确定性（他们是否在策略性抬价或压价？）；
2. 电力与碳市场未来几轮的走势。
如果新信息不足以改变判断，请保持原有信念并说明原因。

# 输出格式
只输出如下JSON，不要添加其他文字：
```json
{{
  "belief_summary": "用2-4句话描述你更新后的信念及理由",
  "competitor_cost_mean": <竞争对手平均边际成本估计，元/MWh>,
  "competitor_cost_std": <估计的不确定性（标准差），元/MWh>,
  "market_trend": "对未来几轮电价和碳价走势的简短判断"
}}
```
"""

    # ========================================
    # 长期语义记忆（经验总结）Prompt
    # ========================================
    EXPERIENCE_SUMMARY_TEMPLATE = """作为发电商智能体 {agent_id}{profile_str}，你需要总结过去 {num_rounds_to_summarize} 轮在电力与碳市场中的经验，形成可以指导未来决策的长期记忆。

# 历史记录:
{historical_data_snippet}
(列含义: round=轮次, energy_bid_price/quantity=能源报价价/量, reserve_bid_*=备用报价, cleared_*=中标量,
market_energy_price/market_reserve_price=出清价, profit=利润, belief_*=决策时信念, market_carbon_price=碳价,
carbon_buy/sell_quantity=碳买卖量)
{previous_summary_block}
# 任务
识别成功的模式和失败的教训，以及报价、碳交易与利润之间的关系。
用不超过5条要点给出简明的经验总结和对未来策略的建议，例如"在高峰期提高加价率因水电竞争而失败"。

请直接输出经验总结文本，不要输出JSON。
"""

    # ========================================
    # IBR-CR 相关 Prompts
    # ========================================
    IBR_CR_INITIAL_BID_SUFFIX = """

### IBR-CR 第0步：初始意向报价
请生成你的初始报价意向，这将作为后续策略优化的基础。"""

    IBR_CR_COMPETITOR_RESPONSE_TEMPLATE = """你现在需要模拟一个理性的竞争对手的视角。假设你是市场中的一个典型竞争对手，观察到智能体 {agent_id} 提交了以下初始报价：

### 观察到的初始报价:
{initial_bid_str}

### 当前市场信息 (轮次 {round_number}):
{market_info_str}

### 备用市场状态: 本轮备用需求为 {reserve_requirement:.2f} MW {reserve_market_status}

### 竞争对手分析任务
作为一个理性的竞争对手，请分析这个初始报价，并预测其他竞争对手（包括你自己）会如何反应。
考虑以下因素：
1. 这个报价的竞争力如何？价格是否过高或过低？
2. 如果你是竞争对手，你会如何调整自己的报价来应对？
3. 市场中其他参与者可能的反应是什么？
4. 这种报价策略可能导致什么样的市场结果？

请输出一个典型竞争对手的预期报价策略：

```json
{{
  "analysis": "对初始报价的分析和竞争对手可能反应的推理",
  "expected_competitor_strategy": {{
    "electricity_bid": {{
      "energy_bid": {{"price": <预期竞争对手能源报价>, "quantity": <预期数量>}}{reserve_bid_format}
    }},
    "carbon_bid": {{
      "buy_quantity": <预期竞争对手碳配额买入量>,
      "buy_price": <预期买入价格>,
      "sell_quantity": <预期竞争对手碳配额卖出量>,
      "sell_price": <预期卖出价格>
    }}
  }},
  "market_impact": "预期这种竞争格局对市场价格和清算结果的影响"
}}
```
"""

    IBR_CR_BID_OPTIMIZATION_TEMPLATE = """你是智能体 {agent_id}，现在需要基于对竞争对手反应的预期来优化你的最终报价策略。

### 你的初始报价意向:
{initial_bid_str}

### 预期的竞争对手反应:
{competitor_str}

### 竞争分析:
{analysis}

### 市场影响预期:
{market_impact}

### 当前市场信息 (轮次 {round_number}):
{market_info_str}

### 策略优化任务
基于以上信息，请重新优化你的报价策略。考虑：
1. **反事实推理**: 如果竞争对手按预期反应，你的初始报价是否仍然最优？
2. **策略调整**: 是否需要调整价格或数量来应对预期的竞争？
3. **市场定位**: 在预期的竞争格局中，如何定位自己的报价？
4. **风险管理**: 如何平衡收益最大化和中标概率？
5. **动态适应**: 如何利用对竞争对手的预期来获得竞争优势？

### 你的私有信息提醒
- 边际成本: 能源 {marginal_cost_energy:.2f} 元/MWh, 备用 {marginal_cost_reserve:.2f} 元/MWh
- 最大容量: {max_capacity:.2f} MW
- 技术类型: {technology_type}

请输出你的最终优化报价：

```json
{{
  "optimization_reasoning": "基于竞争对手预期的策略优化推理过程",
  "strategy_changes": "相比初始报价的主要调整和原因",
  "electricity_bid": {{
    "energy_bid": {{"price": <最终能源报价>, "quantity": <最终能源数量>}},
    "reserve_bid": {{"price": <最终备用报价>, "quantity": <最终备用数量>}}
  }},
  "carbon_bid": {{
    "buy_quantity": <最终碳配额买入量>,
    "buy_price": <最终买入价格>,
    "sell_quantity": <最终碳配额卖出量>,
    "sell_price": <最终卖出价格>
  }},
  "expected_outcome": "对这个优化策略预期效果的评估"
}}
```
"""

    # ========================================
    # 市场事件信息模板
    # ========================================
    MARKET_EVENTS_TEMPLATE = """
### 🚨 重要市场事件公告
{market_events_info}

⚠️ 市场事件响应指南:
- 分析事件对你的成本结构的直接影响，特别是当事件涉及你使用的燃料类型
- 对于价格变化公告：考虑在实际变化前反映未来成本，同时评估市场接受度
- 评估市场价格趋势变化，在价格实际变动前适度调整报价策略
- 考虑优化容量分配，根据预期的市场变化调整能源和备用的比例
- 预判其他市场参与者的可能反应，尤其是使用相同或替代燃料类型的竞争对手
"""

    # ========================================
    # 技术类型特定指导
    # ========================================
    TECH_GUIDANCE = {
        'gas': """
### 📊 燃气机组特定策略指导
- 你是燃气发电机组，对燃料价格变化较为敏感
- 关注燃气市场动态，及时调整报价策略
- 根据市场情况灵活分配能源和备用容量
- 密切关注市场出清价格，根据中标情况调整策略
""",
        'coal': """
### 📊 煤电机组特定策略指导
- 你是煤电机组，对燃料价格变化有一定敏感度
- 关注煤炭市场动态，及时调整报价策略
- 煤电机组通常作为基荷运行，可以保持较高的能源报价比例
""",
        'hydro': """
### 📊 水电机组特定策略指导
- 你是水电机组，边际成本较低，对燃料价格变化不敏感
- 关注市场整体价格趋势，适时调整报价策略
- 水电机组灵活性高，可以在备用市场获得较好收益
""",
        'wind': """
### 📊 风电机组特定策略指导
- 你是风电机组，边际成本极低，对燃料价格变化不敏感
- 关注市场整体价格趋势，适时调整报价策略
- 风电出力不稳定，需要合理安排能源和备用容量
""",
        'solar': """
### 📊 光伏机组特定策略指导
- 你是光伏机组，边际成本极低，对燃料价格变化不敏感
- 关注市场整体价格趋势，适时调整报价策略
- 光伏出力受天气影响，需要合理安排能源和备用容量
"""
    }

    @classmethod
    def get_tech_guidance(cls, tech_type: str) -> str:
        """根据技术类型获取特定指导"""
        tech_type_lower = tech_type.lower()
        for key, guidance in cls.TECH_GUIDANCE.items():
            if key in tech_type_lower:
                return guidance
        return ""

    @classmethod
    def format_coupled_market_prompt(cls,
                                     agent_id: str,
                                     round_number: int,
                                     private_info: Dict,
                                     market_info: Dict,
                                     history: Optional[Dict] = None,
                                     market_events_info: Optional[str] = None,
                                     belief_summary: Optional[str] = None,
                                     working_memory: Optional[str] = None,
                                     semantic_memory: Optional[str] = None,
                                     working_memory_k: int = 5) -> str:
        """
        格式化耦合市场联合决策prompt
        
        Args:
            agent_id: 智能体ID
            round_number: 轮次
            private_info: 私有信息
            market_info: 市场信息
            history: 历史信息
            market_events_info: 市场事件信息
            belief_summary: 信念摘要 B_i（为None时不注入认知状态块）
            working_memory: 短期工作记忆文本
            semantic_memory: 长期语义记忆文本
            working_memory_k: 短期工作记忆的轮数K（仅用于显示）
        
        Returns:
            格式化后的prompt字符串
        """
        # 构建历史信息字符串
        history_str = ""
        if history and (history.get('market_prices') or history.get('awards')):
            prices = history.get('market_prices', [])
            awards = history.get('awards', [])
            history_str = (
                "\n### 历史参考信息\n"
                f"- 最近几轮电力市场出清价: {prices[-5:] if prices else '无'}\n"
                f"- 你在最近几轮的电力市场中标情况 (True=中标): {awards[-5:] if awards else '无'}\n"
            )
        
        # 获取碳市场信息
        carbon_market = market_info.get('carbon_market', {})
        current_carbon_price = carbon_market.get('current_price', 50.0)
        current_quota = carbon_market.get('current_quota', 80.0)
        
        # 检查备用市场需求
        reserve_requirement = market_info.get('reserve_requirement', 0)
        has_reserve_market = reserve_requirement > 0
        
        # 构建市场事件信息
        market_events_str = ""
        if market_events_info:
            market_events_str = cls.MARKET_EVENTS_TEMPLATE.format(
                market_events_info=market_events_info
            )
        
        # 获取技术类型特定指导
        tech_type = private_info.get('technology_type', '')
        tech_guidance = cls.get_tech_guidance(tech_type)
        
        # 格式化私有信息和市场信息为JSON字符串
        private_info_str = json.dumps(private_info, ensure_ascii=False, indent=2)
        market_info_str = json.dumps(market_info, ensure_ascii=False, indent=2)
        
        # 根据是否有备用市场调整prompt格式
        reserve_bid_format = (
            ',\n    "reserve_bid": {"price": <你的备用报价价格>, "quantity": <你的备用报价数量>}'
            if has_reserve_market else
            ',\n    "reserve_bid": {"price": 0, "quantity": 0}'
        )
        
        reserve_bid_guidance = (
            "- 备用报价: 考虑备用机会成本，通常低于能源报价\n"
            if has_reserve_market else
            "- 备用报价: 本轮无备用需求，设置为0或不报价\n"
        )
        
        reserve_market_status = "(有备用市场)" if has_reserve_market else "(无备用市场)"

        # 认知状态（信念 + 双层记忆）
        cognitive_state_str = ""
        if belief_summary is not None or working_memory is not None or semantic_memory is not None:
            cognitive_state_str = cls.COGNITIVE_STATE_TEMPLATE.format(
                belief_summary=belief_summary or "你目前对竞争对手的信息了解有限。",
                working_memory_k=working_memory_k,
                working_memory=working_memory or "暂无。",
                semantic_memory=semantic_memory or "暂无足够的历史经验。",
            )
        
        # 填充模板
        return cls.COUPLED_MARKET_PROMPT_TEMPLATE.format(
            cognitive_state_str=cognitive_state_str,
            agent_id=agent_id,
            round_number=round_number,
            private_info_str=private_info_str,
            market_info_str=market_info_str,
            market_events_str=market_events_str,
            tech_guidance=tech_guidance,
            history_str=history_str,
            current_carbon_price=current_carbon_price,
            carbon_price_min=current_carbon_price * 0.9,
            carbon_price_max=current_carbon_price * 1.1,
            current_quota=current_quota,
            marginal_cost_energy=private_info['marginal_cost_energy'],
            marginal_cost_reserve=private_info['marginal_cost_reserve'],
            max_capacity=private_info['max_capacity'],
            emission_factor=private_info.get('emission_factor', 0.8),
            reserve_requirement=reserve_requirement,
            reserve_market_status=reserve_market_status,
            reserve_bid_guidance=reserve_bid_guidance,
            reserve_bid_format=reserve_bid_format
        )

    @classmethod
    def format_belief_revision_prompt(cls,
                                      agent_id: str,
                                      round_number: int,
                                      private_info: Dict,
                                      previous_belief: Dict,
                                      observations: List[Dict],
                                      public_info: Dict,
                                      market_events_info: Optional[str] = None,
                                      num_competitors: int = 0) -> str:
        """格式化信念语义转移prompt（B_{t-1}, Ω_pub, Outcome -> B_t）"""
        def fmt(v, nd=2):
            try:
                return f"{float(v):.{nd}f}"
            except (TypeError, ValueError):
                return "-"
        if observations:
            obs_lines = ["轮次 | 电价 | 碳价 | 我的能源报价 | 我的报价量 | 我的中标量 | 我的利润"]
            for o in observations:
                obs_lines.append(
                    f"{o.get('round')} | {fmt(o.get('energy_price'))} | {fmt(o.get('carbon_price'))} | "
                    f"{fmt(o.get('my_energy_bid_price'))} | {fmt(o.get('my_energy_bid_quantity'), 1)} | "
                    f"{fmt(o.get('my_cleared_energy'), 1)} | {fmt(o.get('profit'))}"
                )
            observations_str = "\n".join(obs_lines)
        else:
            observations_str = "上次更新后没有新的出清结果。"
        market_events_block = (
            f"\n# 市场公告/事件:\n{market_events_info}\n" if market_events_info else ""
        )
        return cls.BELIEF_REVISION_TEMPLATE.format(
            agent_id=agent_id,
            round_number=round_number,
            technology_type=private_info.get('technology_type', '未指定'),
            marginal_cost_energy=float(private_info.get('marginal_cost_energy', 0.0) or 0.0),
            emission_factor=float(private_info.get('emission_factor', 0.0) or 0.0),
            num_competitors=num_competitors,
            previous_belief_summary=previous_belief.get('summary', '初始信念。'),
            prev_mean=float(previous_belief.get('competitor_cost_mean', 0.0) or 0.0),
            prev_std=float(previous_belief.get('competitor_cost_std', 0.0) or 0.0),
            prev_trend=previous_belief.get('market_trend', '尚无判断'),
            observations=observations_str,
            public_info=json.dumps(public_info, ensure_ascii=False, indent=2),
            market_events_block=market_events_block,
        )

    @classmethod
    def format_experience_summary_prompt(cls,
                                         agent_id: str,
                                         historical_data_snippet: str,
                                         num_rounds_to_summarize: int,
                                         previous_summary: Optional[str] = None,
                                         agent_profile: Optional[Dict] = None) -> str:
        """格式化长期语义记忆（经验总结）prompt"""
        profile_str = ""
        if agent_profile:
            profile_str = (f"（{agent_profile.get('technology_type', '未指定')}，能源边际成本 "
                           f"{float(agent_profile.get('marginal_cost_energy', 0.0) or 0.0):.2f} 元/MWh）")
        previous_summary_block = (
            f"\n# 你之前的经验总结（请在其基础上修订）:\n{previous_summary}\n" if previous_summary else ""
        )
        return cls.EXPERIENCE_SUMMARY_TEMPLATE.format(
            agent_id=agent_id,
            profile_str=profile_str,
            num_rounds_to_summarize=num_rounds_to_summarize,
            historical_data_snippet=historical_data_snippet,
            previous_summary_block=previous_summary_block,
        )

    @classmethod
    def format_ibr_cr_initial_bid_prompt(cls, base_prompt: str) -> str:
        """为IBR-CR初始报价添加后缀"""
        return base_prompt + cls.IBR_CR_INITIAL_BID_SUFFIX

    @classmethod
    def format_ibr_cr_competitor_response_prompt(cls,
                                                 agent_id: str,
                                                 round_number: int,
                                                 initial_bid: Dict,
                                                 market_info: Dict) -> str:
        """格式化IBR-CR竞争对手反应prompt"""
        initial_bid_str = json.dumps({
            'electricity': initial_bid.get('electricity', {}),
            'carbon': initial_bid.get('carbon', {})
        }, ensure_ascii=False, indent=2)
        
        market_info_str = json.dumps(market_info, ensure_ascii=False, indent=2)
        
        reserve_requirement = market_info.get('reserve_requirement', 0)
        has_reserve_market = reserve_requirement > 0
        
        reserve_bid_format = (
            ',\n      "reserve_bid": {"price": <预期竞争对手备用报价>, "quantity": <预期数量>}'
            if has_reserve_market else
            ',\n      "reserve_bid": {"price": 0, "quantity": 0}'
        )
        
        reserve_market_status = "(有备用市场)" if has_reserve_market else "(无备用市场)"
        
        return cls.IBR_CR_COMPETITOR_RESPONSE_TEMPLATE.format(
            agent_id=agent_id,
            round_number=round_number,
            initial_bid_str=initial_bid_str,
            market_info_str=market_info_str,
            reserve_requirement=reserve_requirement,
            reserve_market_status=reserve_market_status,
            reserve_bid_format=reserve_bid_format
        )

    @classmethod
    def format_ibr_cr_optimization_prompt(cls,
                                         agent_id: str,
                                         round_number: int,
                                         initial_bid: Dict,
                                         competitor_response: Dict,
                                         market_info: Dict,
                                         private_info: Dict) -> str:
        """格式化IBR-CR报价优化prompt"""
        initial_bid_str = json.dumps({
            'electricity': initial_bid.get('electricity', {}),
            'carbon': initial_bid.get('carbon', {})
        }, ensure_ascii=False, indent=2)
        
        competitor_str = json.dumps(
            competitor_response.get('expected_competitor_strategy', {}),
            ensure_ascii=False, indent=2
        )
        
        market_info_str = json.dumps(market_info, ensure_ascii=False, indent=2)
        
        return cls.IBR_CR_BID_OPTIMIZATION_TEMPLATE.format(
            agent_id=agent_id,
            round_number=round_number,
            initial_bid_str=initial_bid_str,
            competitor_str=competitor_str,
            analysis=competitor_response.get('analysis', '无分析'),
            market_impact=competitor_response.get('market_impact', '无预期'),
            market_info_str=market_info_str,
            marginal_cost_energy=private_info['marginal_cost_energy'],
            marginal_cost_reserve=private_info['marginal_cost_reserve'],
            max_capacity=private_info['max_capacity'],
            technology_type=private_info.get('technology_type', '未指定')
        )


# ========================================
# 向后兼容: 旧的决策Prompt函数 (来自prompts.py)
# ========================================

DECISION_MAKING_PROMPT_TEMPLATE = """
作为一个理性的发电商智能体，你的目标是在当前的电力现货与备用联合市场中最大化你的利润。
你正在参与第 {round_number} 轮市场报价。

# 你的角色描述:
{role_description}

# 你的私有信息:
- 发电机组ID: {agent_id}
- 技术类型: {technology_type}
- 能源生产边际成本: {marginal_cost_energy:.2f} 元/MWh
- 备用提供边际成本: {marginal_cost_reserve:.2f} 元/MWh (通常低于能源成本，代表机会成本或准备成本)
- 最大总容量 (能源+备用): {max_capacity:.2f} MW
- 最小发电出力 (如有): {min_output:.2f} MW (简化模型中可能不使用)
- 爬坡限制等 (简化模型中可能不使用)

# 当前市场公开信息:
- 预测系统总负荷: {total_demand:.2f} MW
- 预测系统所需备用: {reserve_requirement:.2f} MW
- 竞争对手数量: {num_competitors}
- (可选) 上一轮市场出清价格 (如果适用):
    - 现货能源价格: {last_energy_price:.2f} 元/MWh
    - 备用价格: {last_reserve_price:.2f} 元/MWh

# 你对竞争对手的当前信念 (基于历史观察的估计):
{belief_summary}
(信念说明: 这里总结了你对其他竞争对手可能的成本或行为的看法。例如，估计他们的平均成本范围，或者他们是否倾向于激进报价等。)

# 你的历史经验总结 (最近几轮的表现和反思):
{experience_summary}
(经验说明: 回顾过去的报价、中标情况和利润。例如，\"在第N轮，报价X，利润Y，原因是...\"、\"当市场价格高于某值时，某策略似乎更有效。\")

# 市场事件和预期行为指南:
- 对市场公告的各类事件（如燃料价格变化、供应中断等）及时做出反应
- 当收到未来市场变化的公告时，应提前调整报价策略，而不是等到变化实际发生
- 预期行为是理性的市场策略，提前反映未来变化可以帮助你获得更好的利润
- 根据你的技术类型，采取相应的应对策略:
  * 燃料依赖型机组: 对相关燃料价格变化较为敏感，应及时调整报价
  * 可再生能源机组: 虽然对燃料价格不敏感，但可以根据市场整体变化调整策略

# 任务与要求:
根据以上所有信息，进行推理和博弈分析，为当前这一轮制定一个有利可图的战略性报价。
你需要同时报出现货能源的价格和数量，以及备用的价格和数量。
你的总报价数量 (能源 + 备用) 不能超过你的最大容量。
报价需要平衡争取中标概率和获得高利润之间的关系。考虑到不完全信息和竞争对手可能的反应。

## 战略思考指南:
1. **市场事件快速响应**: 对市场公告的各类事件立即做出反应。特别是当事件影响你的成本结构时，应提前调整策略。
2. **预期行动**: 如果收到未来市场变化的公告，应在变化实际发生前适当调整报价，反映预期影响。
3. **差异化策略**:
   - 如果你的成本结构将受到影响：提前调整报价，同时考虑容量分配策略
   - 如果竞争对手的成本结构将受到影响：准备调整你的策略以应对市场结构变化
4. **理性预期行为**:
   - 预期行为是理性的市场策略，可以帮助你在市场变化前获得更好的利润
   - 根据市场信息及时调整策略，避免错失利润机会

# 输出格式要求:
请严格按照以下JSON格式返回你的报价决策，不要包含任何额外的解释或说明文字。
```json
{{
  "energy_bid": {{
    "price": <你的能源报价价格 (元/MWh)>,
    "quantity": <你的能源报价数量 (MW)>
  }},
  "reserve_bid": {{
    "price": <你的备用报价价格 (元/MWh)>,
    "quantity": <你的备用报价数量 (MW)>
  }},
  "reasoning": "<对你报价策略的简短说明，例如基于什么考虑，预期市场情况等 (可选但推荐)>"
}}
```

请现在生成你的报价决策。
"""

def format_decision_prompt(round_number, agent_id, private_info, market_info, belief_summary, experience_summary, role_description=None):
    """填充决策Prompt模板 (向后兼容函数)"""
    return DECISION_MAKING_PROMPT_TEMPLATE.format(
        round_number=round_number,
        agent_id=agent_id,
        technology_type=private_info.get('technology_type', '未指定'),
        marginal_cost_energy=private_info.get('marginal_cost_energy', 0.0),
        marginal_cost_reserve=private_info.get('marginal_cost_reserve', 0.0),
        max_capacity=private_info.get('max_capacity', 100.0),
        min_output=private_info.get('min_output', 0.0),
        total_demand=market_info.get('total_demand', 500.0),
        reserve_requirement=market_info.get('reserve_requirement', 50.0),
        num_competitors=market_info.get('num_competitors', 3),
        last_energy_price=market_info.get('last_energy_price', 0.0),
        last_reserve_price=market_info.get('last_reserve_price', 0.0),
        belief_summary=belief_summary if belief_summary else "你目前对竞争对手的信息了解有限。",
        experience_summary=experience_summary if experience_summary else "你暂时还没有足够的历史经验可供参考。",
        role_description=role_description if role_description else "一家普通的发电企业"
    )

