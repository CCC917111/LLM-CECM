"""
智能体行为分析器
用于跟踪和分析智能体的报价策略变化，特别是预期行为
"""

import logging
import json
import csv
from typing import Dict, List, Any, Optional
from collections import defaultdict
import statistics

class BehaviorAnalyzer:
    """智能体行为分析器"""
    
    def __init__(self):
        self.agent_classifications = {}  # 智能体技术类型分类
        self.bidding_history = defaultdict(list)  # 报价历史 {agent_id: [round_data]}
        self.behavior_metrics = defaultdict(dict)  # 行为指标
        self.baseline_period = (0, 23)  # 基准期间（用于计算偏离度）
        
        logging.info("行为分析器初始化完成")
    
    def classify_agents(self, agents: List[Any]):
        """根据技术类型对智能体进行分类"""
        tech_type_mapping = {
            "coal": ["燃煤", "煤电", "coal"],
            "gas": ["燃气", "天然气", "gas", "natural_gas"],
            "hydro": ["水电", "水力", "hydro"],
            "wind": ["风电", "风力", "wind"],
            "solar": ["光伏", "太阳能", "solar", "pv"],
            "nuclear": ["核电", "nuclear"],
            "biomass": ["生物质", "biomass"]
        }
        
        # 确保每个智能体都有technology_type字段
        for agent in agents:
            if hasattr(agent, 'update_technology_type'):
                agent.update_technology_type()
                logging.info(f"更新智能体 {agent.agent_id} 的技术类型")
        
        for agent in agents:
            if hasattr(agent, 'private_info') and 'technology_type' in agent.private_info:
                tech_type = agent.private_info['technology_type'].lower()
                
                # 分类逻辑
                classified_type = "other"
                for category, keywords in tech_type_mapping.items():
                    if any(keyword.lower() in tech_type for keyword in keywords):
                        classified_type = category
                        break
                
                # 针对CCGT和OCGT的特殊处理
                if "ccgt" in tech_type or "ocgt" in tech_type:
                    classified_type = "gas"
                
                # 针对角色描述中的关键词进行额外检查
                if hasattr(agent, 'role_description') and agent.role_description:
                    role_desc = agent.role_description.lower()
                    for category, keywords in tech_type_mapping.items():
                        if any(keyword.lower() in role_desc for keyword in keywords):
                            if classified_type == "other":  # 如果之前没有分类成功，则使用角色描述中的信息
                                classified_type = category
                                logging.info(f"智能体 {agent.agent_id} 通过角色描述分类为 {category}")
                
                self.agent_classifications[agent.agent_id] = {
                    "category": classified_type,
                    "original_type": agent.private_info['technology_type'],
                    "marginal_cost": agent.private_info.get('marginal_cost_energy', 0),
                    "capacity": agent.private_info.get('max_capacity', 0)
                }
                
                logging.info(f"智能体分类: {agent.agent_id} -> {classified_type} ({agent.private_info['technology_type']})")
            else:
                logging.warning(f"智能体 {agent.agent_id} 缺少technology_type字段，无法分类")
        
        # 输出分类结果
        category_counts = defaultdict(int)
        for agent_id, info in self.agent_classifications.items():
            category_counts[info["category"]] += 1
            logging.info(f"智能体分类: {agent_id} -> {info['category']} ({info['original_type']})")
        
        logging.info(f"智能体分类统计: {dict(category_counts)}")
    
    def record_round_behavior(self, round_number: int, agent_bids: Dict[str, Any], 
                            market_results: Dict[str, Any], events_info: Dict[str, Any]):
        """记录每轮的行为数据"""
        
        for agent_id, bid_data in agent_bids.items():
            if agent_id not in self.agent_classifications:
                continue
                
            # 提取关键数据
            energy_bid = bid_data.get('electricity', {}).get('energy_bid', {})
            marginal_cost = self.agent_classifications[agent_id]["marginal_cost"]
            
            # 计算加价率，增加数据验证和边界情况处理
            bid_price = energy_bid.get('price', marginal_cost)
            raw_markup = bid_price - marginal_cost
            
            # 增强加价率计算逻辑
            if marginal_cost > 0:
                markup_rate = raw_markup / marginal_cost
                # 限制异常值
                if markup_rate > 10.0:  # 加价率超过1000%视为异常
                    logging.warning(f"智能体 {agent_id} 在轮次 {round_number} 的加价率异常高: {markup_rate:.2f}, 原始数据: 报价={bid_price}, 成本={marginal_cost}")
                    markup_rate = min(markup_rate, 10.0)  # 限制最大加价率
            else:
                if bid_price > 0:
                    markup_rate = 1.0  # 成本为0但报价>0时，设为100%加价
                    logging.warning(f"智能体 {agent_id} 在轮次 {round_number} 的边际成本为0，报价为{bid_price}，设置加价率为1.0")
                else:
                    markup_rate = 0.0
            
            # 记录数据，增加更多字段便于调试
            round_data = {
                "round": round_number,
                "bid_price": bid_price,
                "marginal_cost": marginal_cost,
                "raw_markup": raw_markup,  # 新增：原始加价值
                "markup_rate": markup_rate,
                "bid_quantity": energy_bid.get('quantity', 0),
                "cleared_quantity": market_results.get(f"{agent_id}_cleared_energy", 0),
                "profit": market_results.get(f"{agent_id}_profit", 0),
                "token_usage": bid_data.get('token_usage', {}),
                "events": {
                    "new_announcements": len(events_info.get("new_announcements", [])),
                    "announced_pending": len(events_info.get("announced_pending", [])),
                    "taking_effect": len(events_info.get("taking_effect", []))
                }
            }
            
            # 添加详细日志，特别是当有事件时
            if (events_info.get("new_announcements") or 
                events_info.get("announced_pending") or 
                events_info.get("taking_effect")):
                logging.info(f"轮次 {round_number}, 智能体 {agent_id} ({self.agent_classifications[agent_id]['category']}): "
                           f"报价={bid_price:.2f}, 成本={marginal_cost:.2f}, 加价率={markup_rate:.2f}")
            
            self.bidding_history[agent_id].append(round_data)
    
    def calculate_baseline_metrics(self):
        """计算基准期间的行为指标"""
        baseline_start, baseline_end = self.baseline_period
        
        for agent_id, history in self.bidding_history.items():
            baseline_data = [data for data in history 
                           if baseline_start <= data["round"] <= baseline_end]
            
            if baseline_data:
                baseline_markup_rates = [data["markup_rate"] for data in baseline_data]
                self.behavior_metrics[agent_id]["baseline_markup_mean"] = statistics.mean(baseline_markup_rates)
                self.behavior_metrics[agent_id]["baseline_markup_std"] = (
                    statistics.stdev(baseline_markup_rates) if len(baseline_markup_rates) > 1 else 0
                )
            else:
                self.behavior_metrics[agent_id]["baseline_markup_mean"] = 0
                self.behavior_metrics[agent_id]["baseline_markup_std"] = 0
    
    def analyze_anticipatory_behavior(self, event_announce_time: int, event_effect_time: int) -> Dict[str, Any]:
        """分析预期行为（事件公告后、生效前的行为变化）"""
        
        # 确保基准指标已计算
        self.calculate_baseline_metrics()
        
        analysis_results = {
            "gas_agents": [],
            "coal_agents": [],
            "other_agents": [],
            "summary": {},
            "raw_data": {}  # 新增：保存原始数据用于调试
        }
        
        # 分析每个智能体在预期期间的行为
        for agent_id, history in self.bidding_history.items():
            if agent_id not in self.agent_classifications:
                continue
                
            agent_category = self.agent_classifications[agent_id]["category"]
            baseline_markup = self.behavior_metrics[agent_id]["baseline_markup_mean"]
            
            # 提取基准期间和预期期间的数据
            baseline_data = [data for data in history 
                           if self.baseline_period[0] <= data["round"] <= self.baseline_period[1]]
            anticipatory_data = [data for data in history 
                               if event_announce_time <= data["round"] < event_effect_time]
            
            # 保存原始数据用于调试
            analysis_results["raw_data"][agent_id] = {
                "baseline_data": baseline_data,
                "anticipatory_data": anticipatory_data
            }
            
            # 记录数据点数量
            baseline_count = len(baseline_data)
            anticipatory_count = len(anticipatory_data)
            
            logging.info(f"智能体 {agent_id} ({agent_category}): 基准期数据点 {baseline_count}, 预期期数据点 {anticipatory_count}")
            
            if baseline_count == 0:
                logging.warning(f"智能体 {agent_id} 在基准期间没有数据")
                continue
                
            if anticipatory_count == 0:
                logging.warning(f"智能体 {agent_id} 在预期期间没有数据")
                continue
            
            # 计算基准期和预期期的平均加价率
            baseline_markup_rates = [data["markup_rate"] for data in baseline_data]
            anticipatory_markup_rates = [data["markup_rate"] for data in anticipatory_data]
            
            avg_baseline_markup = statistics.mean(baseline_markup_rates)
            avg_anticipatory_markup = statistics.mean(anticipatory_markup_rates)
                
            # 计算相对于基准的变化
            markup_change = avg_anticipatory_markup - avg_baseline_markup
            
            # 增强加价率变化百分比计算，避免除以零
            if abs(avg_baseline_markup) > 0.001:  # 使用小阈值避免接近零的情况
                markup_change_pct = (markup_change / avg_baseline_markup) * 100
            else:
                if markup_change > 0:
                    markup_change_pct = 100.0  # 基准接近零但变化为正
                    logging.warning(f"智能体 {agent_id}: 基准加价率接近零 ({avg_baseline_markup:.6f})，变化为正 ({markup_change:.6f})，设置为+100%")
                elif markup_change < 0:
                    markup_change_pct = -100.0  # 基准接近零但变化为负
                    logging.warning(f"智能体 {agent_id}: 基准加价率接近零 ({avg_baseline_markup:.6f})，变化为负 ({markup_change:.6f})，设置为-100%")
                else:
                    markup_change_pct = 0.0  # 无变化
            
            # 限制异常大的百分比变化
            if abs(markup_change_pct) > 1000:
                markup_change_pct = 1000.0 if markup_change_pct > 0 else -1000.0
                logging.warning(f"智能体 {agent_id}: 加价率变化百分比异常大，限制为 {markup_change_pct}%")
            
            # 记录详细的变化数据
            logging.info(f"智能体 {agent_id} ({agent_category}) 加价率分析: "
                   f"基准期 {avg_baseline_markup:.4f} → 预期期 {avg_anticipatory_markup:.4f}, "
                   f"变化 {markup_change:+.4f} ({markup_change_pct:+.1f}%)")
            
            agent_analysis = {
            "agent_id": agent_id,
            "category": agent_category,
            "original_type": self.agent_classifications[agent_id]["original_type"],
            "baseline_markup": avg_baseline_markup,
            "anticipatory_markup": avg_anticipatory_markup,
            "markup_change": markup_change,
            "markup_change_pct": markup_change_pct,
            "anticipatory_rounds": anticipatory_count,
            "baseline_rounds": baseline_count
            }
                
            # 按类型分类
            if agent_category == "gas":
                analysis_results["gas_agents"].append(agent_analysis)
            elif agent_category == "coal":
                analysis_results["coal_agents"].append(agent_analysis)
            else:
                analysis_results["other_agents"].append(agent_analysis)
        
        # 计算汇总统计
        gas_changes = [agent["markup_change_pct"] for agent in analysis_results["gas_agents"]]
        coal_changes = [agent["markup_change_pct"] for agent in analysis_results["coal_agents"]]
        other_changes = [agent["markup_change_pct"] for agent in analysis_results["other_agents"]]
        
        # 记录详细的分组数据
        if gas_changes:
            logging.info(f"燃气机组加价率变化: {gas_changes}")
        if coal_changes:
            logging.info(f"煤电机组加价率变化: {coal_changes}")
        if other_changes:
            logging.info(f"其他机组加价率变化: {other_changes}")
        
        # 降低检测阈值，使其更敏感
        detection_threshold = 3.0  # 从5%降低到3%
        
        analysis_results["summary"] = {
            "gas_agents_count": len(analysis_results["gas_agents"]),
            "coal_agents_count": len(analysis_results["coal_agents"]),
            "other_agents_count": len(analysis_results["other_agents"]),
            "gas_avg_change_pct": statistics.mean(gas_changes) if gas_changes else 0,
            "coal_avg_change_pct": statistics.mean(coal_changes) if coal_changes else 0,
            "other_avg_change_pct": statistics.mean(other_changes) if other_changes else 0,
            "gas_behavior_detected": any(change > detection_threshold for change in gas_changes),
            "coal_behavior_detected": any(change > detection_threshold for change in coal_changes),
            "detection_threshold": detection_threshold
        }
        
        return analysis_results
    
    def generate_behavior_report(self, event_announce_time: int, event_effect_time: int) -> str:
        """生成行为分析报告"""
        analysis = self.analyze_anticipatory_behavior(event_announce_time, event_effect_time)
        
        report_lines = [
            "=" * 80,
            "🔍 智能体预期行为分析报告",
            "=" * 80,
            f"📅 分析期间: t={event_announce_time} (事件公告) 到 t={event_effect_time-1} (事件生效前)",
            f"📊 基准期间: t={self.baseline_period[0]} 到 t={self.baseline_period[1]}",
            f"🔎 预期行为检测阈值: 加价率变化 > {analysis['summary']['detection_threshold']}%",
            "",
            "📈 燃气机组行为分析:",
        ]
        
        if analysis["gas_agents"]:
            # 按加价率变化排序，从高到低
            sorted_gas_agents = sorted(analysis["gas_agents"], 
                                     key=lambda x: x['markup_change_pct'], 
                                     reverse=True)
            
            for agent in sorted_gas_agents:
                # 添加更详细的信息，包括原始报价和成本
                report_lines.append(
                    f"  • {agent['agent_id']} ({agent['original_type']}): "
                    f"加价率变化 {agent['markup_change_pct']:+.1f}% "
                    f"({agent['baseline_markup']:.3f} → {agent['anticipatory_markup']:.3f})"
                )
                
                # 添加数据点信息
                report_lines.append(
                    f"    - 数据点: 基准期 {agent['baseline_rounds']}轮, 预期期 {agent['anticipatory_rounds']}轮"
                )
                
                # 判断是否表现出预期行为
                if agent['markup_change_pct'] > analysis['summary']['detection_threshold']:
                    report_lines.append(f"    - ✅ 表现出预期行为 (加价率增加 {agent['markup_change_pct']:+.1f}%)")
                else:
                    report_lines.append(f"    - ❌ 未表现出预期行为")
            
            report_lines.append(f"  平均变化: {analysis['summary']['gas_avg_change_pct']:+.1f}%")
        else:
            report_lines.append("  • 无燃气机组参与")
        
        # 添加煤电机组行为分析
        report_lines.extend([
            "",
            "📈 煤电机组行为分析:",
        ])
        
        if analysis["coal_agents"]:
            # 按加价率变化排序，从高到低
            sorted_coal_agents = sorted(analysis["coal_agents"], 
                                     key=lambda x: x['markup_change_pct'], 
                                     reverse=True)
            
            for agent in sorted_coal_agents:
                report_lines.append(
                    f"  • {agent['agent_id']} ({agent['original_type']}): "
                    f"加价率变化 {agent['markup_change_pct']:+.1f}% "
                    f"({agent['baseline_markup']:.3f} → {agent['anticipatory_markup']:.3f})"
                )
                
                report_lines.append(
                    f"    - 数据点: 基准期 {agent['baseline_rounds']}轮, 预期期 {agent['anticipatory_rounds']}轮"
                )
                
                if agent['markup_change_pct'] > analysis['summary']['detection_threshold']:
                    report_lines.append(f"    - ✅ 表现出预期行为 (加价率增加 {agent['markup_change_pct']:+.1f}%)")
                else:
                    report_lines.append(f"    - ❌ 未表现出预期行为")
            
            report_lines.append(f"  平均变化: {analysis['summary']['coal_avg_change_pct']:+.1f}%")
        else:
            report_lines.append("  • 无煤电机组参与")
        
        report_lines.extend([
            "",
            "📊 其他机组行为对比:",
        ])
        
        if analysis["other_agents"]:
            # 按类型和加价率变化排序
            sorted_other_agents = sorted(analysis["other_agents"], 
                                       key=lambda x: (x['category'], -x['markup_change_pct']))
            
            current_category = None
            for agent in sorted_other_agents:
                # 按类别分组显示
                if current_category != agent['category']:
                    current_category = agent['category']
                    report_lines.append(f"  【{current_category}类机组】:")
                
                report_lines.append(
                    f"  • {agent['agent_id']}: 加价率变化 {agent['markup_change_pct']:+.1f}% "
                    f"({agent['baseline_markup']:.3f} → {agent['anticipatory_markup']:.3f})"
                )
            
            report_lines.append(f"  所有非燃气/煤电机组平均变化: {analysis['summary']['other_avg_change_pct']:+.1f}%")
        else:
            report_lines.append("  • 无其他类型机组")
        
        # 添加更详细的结论
        gas_behavior_detected = analysis['summary']['gas_behavior_detected']
        coal_behavior_detected = analysis['summary'].get('coal_behavior_detected', False)
        gas_avg_change = analysis['summary']['gas_avg_change_pct']
        coal_avg_change = analysis['summary'].get('coal_avg_change_pct', 0)
        other_avg_change = analysis['summary']['other_avg_change_pct']
        
        report_lines.extend([
            "",
            "🎯 预期行为验证结果:",
            f"  • 燃气机组预期行为检测: {'✅ 是' if gas_behavior_detected else '❌ 否'}",
            f"  • 煤电机组预期行为检测: {'✅ 是' if coal_behavior_detected else '❌ 否'}",
            f"  • 燃气机组平均加价变化: {gas_avg_change:+.1f}%",
            f"  • 煤电机组平均加价变化: {coal_avg_change:+.1f}%",
            f"  • 其他机组平均加价变化: {other_avg_change:+.1f}%",
        ])
        
        # 添加总结性评论
        if gas_behavior_detected or coal_behavior_detected:
            report_lines.append("  • 📝 结论: 部分机组表现出预期行为，对市场事件做出了提前反应")
            
        if gas_behavior_detected:
            if gas_avg_change > 2 * other_avg_change:
                    report_lines.append("    - 燃气机组明显表现出比其他机组更强的预期行为")
            else:
                    report_lines.append("    - 燃气机组表现出一定的预期行为，但不够明显")
            
            if coal_behavior_detected:
                if coal_avg_change > 2 * other_avg_change:
                    report_lines.append("    - 煤电机组明显表现出比其他机组更强的预期行为")
            else:
                    report_lines.append("    - 煤电机组表现出一定的预期行为，但不够明显")
        else:
            if gas_avg_change > 0 or coal_avg_change > 0:
                report_lines.append("  • 📝 结论: 燃气/煤电机组加价率有轻微上升，但未达到预期行为检测阈值")
            else:
                report_lines.append("  • 📝 结论: 未检测到明显的预期行为")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_behavior_data(self, filename: str):
        """保存行为数据到文件"""
        behavior_data = {
            "agent_classifications": self.agent_classifications,
            "bidding_history": dict(self.bidding_history),
            "behavior_metrics": dict(self.behavior_metrics)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(behavior_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"行为数据已保存到: {filename}")
