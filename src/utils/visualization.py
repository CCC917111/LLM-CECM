# utils/visualization.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional


class ResultVisualizer:
    """
    Simulation Result Visualization Tool.
    Responsible for reading CSV result files, generating various visualization charts, and saving them to the specified directory.
    """
    
    def __init__(self, results_file: str, output_dir: str = None, agents_config_path: Optional[str] = None, ground_truth_prices: List[float] = None, ground_truth_carbon_prices: List[float] = None):
        """
        Initialize the visualization tool.
        
        Args:
            results_file: CSV result file path
            output_dir: Chart output directory, defaults to the visualization subdirectory under the results_file directory
            agents_config_path: Path to the agent configuration JSON file to load agent categories.
            ground_truth_prices: A list of ground truth electricity prices for comparison.
            ground_truth_carbon_prices: A list of ground truth carbon prices for comparison.
        """
        self.results_file = results_file
        self.ground_truth_prices = ground_truth_prices
        self.ground_truth_carbon_prices = ground_truth_carbon_prices
        
        # 设置输出目录
        if output_dir is None:
            base_dir = os.path.dirname(results_file)
            self.output_dir = os.path.join(base_dir, 'visualization')
        else:
            self.output_dir = output_dir
            
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            self.df = pd.read_csv(self.results_file, on_bad_lines='skip')
            self.data = self.df # 保持旧的data属性以兼容其他可能使用它的函数
        except Exception as e:
            print(f"警告：读取结果文件 {self.results_file} 失败: {e}")
            # 创建空的DataFrame以避免后续错误
            self.df = pd.DataFrame()
            self.data = self.df
        
        # Set visualization style
        sns.set(style="whitegrid")
        
        # --- Load agent categories directly from the provided agent config file ---
        self.agent_categories: Dict[str, str] = {}
        if agents_config_path and os.path.exists(agents_config_path):
            try:
                with open(agents_config_path, 'r', encoding='utf-8') as f:
                    agents_config = json.load(f)
                for agent_data in agents_config:
                    agent_id = agent_data.get('agent_id')
                    fuel_category = agent_data.get('config', {}).get('fuel_category')
                    if agent_id and fuel_category:
                        self.agent_categories[agent_id] = fuel_category
                print(f"成功从 {agents_config_path} 加载 {len(self.agent_categories)} 个智能体的分类信息。")
            except Exception as e:
                print(f"警告：从 {agents_config_path} 加载智能体分类失败: {e}")

        # 如果上述加载失败，执行旧的、不可靠的推断逻辑作为后备
        if not self.agent_categories:
            print("警告：未提供有效的智能体配置文件，将尝试从结果文件中推断分类。")
            try:
                # 收集可能的智能体ID
                detected_ids = set()
                suffixes = ['_profit', '_cleared_energy', '_energy_bid_price', '_carbon_cost', '_carbon_quota']
                for col in self.data.columns:
                    for suf in suffixes:
                        if col.endswith(suf):
                            detected_ids.add(col[:-len(suf)])
                # 为每个智能体寻找其_category列的第一个非空值
                for aid in detected_ids:
                    if aid in self.agent_categories:
                        continue
                    cat_col = f"{aid}_category"
                    if cat_col in self.data.columns:
                        series = self.data[cat_col]
                        val = series.dropna().iloc[0] if not series.dropna().empty else None
                        if isinstance(val, str) and val.strip():
                            self.agent_categories[aid] = val.strip()
            except Exception:
                pass
        
        # --- Global color maps ---
        # Technology color map (default colors)
        self.tech_colors = {
            'Black_Coal': '#8B4513', # 黑煤
            'Brown_Coal': '#A0522D', # 褐煤 (使用不同于黑煤的棕色)
            'Natural_Gas': '#FF6B35', # 天然气
            'Hydro': '#4A90E2',    # 水电
            'Wind': '#7ED321',     # 风电
            'Solar': '#F5A623',    # 太阳能
            'nuclear': '#9013FE',
            'biomass': '#50E3C2',
            'storage': '#BD10E0',
            'other': '#9B9B9B'
        }
        
        # 创建标准化的技术类型映射
        self.tech_type_mapping = {
            'Black_Coal': 'Black_Coal',
            'Brown_Coal': 'Brown_Coal', 
            'Natural_Gas': 'Natural_Gas',
            'Hydro': 'Hydro',
            'Wind': 'Wind',
            'Solar': 'Solar',
            'nuclear': 'nuclear',
            'biomass': 'biomass',
            'storage': 'storage',
            'other': 'other'
        }
        
        # Build a per-agent color map so agents without tech prefix are not all gray
        self.agent_colors = {}
        # Collect agent ids from common suffixes present in CSV columns
        detected_agent_ids = set()
        suffixes = ['_profit', '_cleared_energy', '_energy_bid_price', '_carbon_cost', '_carbon_quota']
        for col in self.data.columns:
            for suf in suffixes:
                if col.endswith(suf):
                    detected_agent_ids.add(col[:-len(suf)])
        # Fallback palette
        palette = sns.color_palette('tab20', n_colors=max(10, len(detected_agent_ids)))
        pal_idx = 0
        for aid in sorted(detected_agent_ids):
            tech = self.agent_categories.get(aid, (aid.split('-')[0] if '-' in aid else 'other'))
            if tech == 'other':
                # assign a unique palette color
                self.agent_colors[aid] = palette[pal_idx % len(palette)]
                pal_idx += 1
            else:
                self.agent_colors[aid] = self.tech_colors.get(tech, '#9B9B9B')
    
    def generate_all_visualizations(self) -> List[str]:
        """
        Generate all visualization charts.
        
        Returns:
            List of generated chart file paths
        """
        generated_files = []
        
        # 生成各类图表
        profit_chart = self.visualize_agent_profits()
        generated_files.append(profit_chart)
        
        price_chart = self.visualize_market_prices()
        generated_files.append(price_chart)
        
        clearing_chart = self.visualize_market_clearing()
        generated_files.append(clearing_chart)
        
        bid_strategy_chart = self.visualize_bid_strategies()
        generated_files.append(bid_strategy_chart)
        
        # 添加碳市场可视化
        try:
            carbon_price_chart = self.visualize_carbon_market()
            generated_files.append(carbon_price_chart)
            
            carbon_trading_chart = self.visualize_carbon_trading()
            generated_files.append(carbon_trading_chart)
            
            carbon_quota_chart = self.visualize_carbon_quotas()
            generated_files.append(carbon_quota_chart)
        except Exception as e:
            print(f"警告：碳市场数据可视化失败，可能是因为数据格式不兼容: {e}")

        # 生成LMP节点电价分析图表
        try:
            lmp_chart = self.visualize_lmp_analysis()
            generated_files.append(lmp_chart)
        except Exception as e:
            print(f"警告：LMP电价分析可视化失败: {e}")

        # 生成Token使用统计图表
        try:
            token_chart = self.visualize_token_usage()
            generated_files.append(token_chart)
        except Exception as e:
            print(f"警告：Token使用统计可视化失败: {e}")

        return generated_files
    
    def visualize_agent_profits(self) -> str:
        """
        Visualize the profit situation of each agent.
        
        Returns:
            Generated chart file path
        """
        # 提取智能体ID列表
        agent_ids = [col.split('_profit')[0] for col in self.data.columns if col.endswith('_profit')]
        
        # 创建图表，增加高度
        plt.figure(figsize=(12, 10))
        
        # 使用全局技术颜色映射
        tech_colors = self.tech_colors
        
        # 定义线型样式
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
        
        # 定义线宽
        line_widths = [2, 2.5, 3, 1.5, 2.2, 2.8, 1.8]
        
        # 按技术类型分组智能体
        tech_groups = {}
        for agent_id in agent_ids:
            # 优先使用行为分析器分类，其次使用ID前缀
            tech_type = self.agent_categories.get(agent_id, 'other')
            # 标准化技术类型名称
            tech_type = self.tech_type_mapping.get(tech_type, tech_type)
            
            if tech_type not in tech_groups:
                tech_groups[tech_type] = []
            tech_groups[tech_type].append(agent_id)
        
        # 绘制每个智能体的利润曲线
        for tech_type, agents in tech_groups.items():
            for i, agent_id in enumerate(agents):
                # 为无技术前缀的智能体分配独立颜色
                color = self.agent_colors.get(agent_id, tech_colors.get(tech_type, '#9B9B9B'))
                profit_col = f"{agent_id}_profit"
                
                # 选择线型和线宽
                style_idx = i % len(line_styles)
                width_idx = i % len(line_widths)
                
                plt.plot(self.data['round'], self.data[profit_col], 
                        color=color,
                        linestyle=line_styles[style_idx],
                        linewidth=line_widths[width_idx],
                        marker='o', 
                        markersize=4,
                        label=f"{agent_id} ({tech_type})",
                        alpha=0.8)
        
        # 计算每轮的总利润
        profit_cols = [f"{agent_id}_profit" for agent_id in agent_ids]
        self.data['total_profit'] = self.data[profit_cols].sum(axis=1)
        plt.plot(self.data['round'], self.data['total_profit'], 'k--', linewidth=3, label='Total Profit')
        
        # 设置y轴范围，确保线条分离清晰
        all_profits = []
        for agent_id in agent_ids:
            all_profits.extend(self.data[f"{agent_id}_profit"].values)
        all_profits.extend(self.data['total_profit'].values)
        
        y_min = min(all_profits) * 0.9  # 留10%的下边距
        y_max = max(all_profits) * 1.1  # 留10%的上边距
        plt.ylim(y_min, y_max)
        
        # Set chart properties
        plt.title('Agent Profit by Round', fontsize=16)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Profit', fontsize=14)
        
        # 将图例放在图下方，避免遮挡
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5, fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.output_dir, 'agent_profits.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        # 创建累计利润柱状图
        plt.figure(figsize=(12, 8))
        
        # 计算每个智能体的累计利润
        total_profits = {agent_id: self.data[f"{agent_id}_profit"].sum() for agent_id in agent_ids}
        
        # 按利润从大到小排序
        sorted_agents = sorted(total_profits.items(), key=lambda x: x[1], reverse=True)
        agents = [item[0] for item in sorted_agents]
        profits = [item[1] for item in sorted_agents]
        
        # 为每个智能体分配颜色（优先使用每智能体颜色映射）
        colors = []
        for agent_id in agents:
            # 优先使用行为分析器分类，其次使用ID前缀
            tech_type = self.agent_categories.get(agent_id) or (agent_id.split('-')[0] if '-' in agent_id else 'other')
            color = self.agent_colors.get(agent_id, tech_colors.get(tech_type, '#9B9B9B'))
            colors.append(color)
        
        # 绘制柱状图
        bars = plt.bar(agents, profits, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在柱子上方显示数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(profits)*0.01,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 添加技术类型图例
        legend_elements = []
        displayed_types = set()
        for agent_id in agents:
            tech_type = self.agent_categories.get(agent_id) or (agent_id.split('-')[0] if '-' in agent_id else 'other')
            
            if tech_type not in displayed_types:
                legend_elements.append(plt.Rectangle((0,0),1,1, 
                                                   facecolor=tech_colors.get(tech_type, '#9B9B9B'), 
                                                   alpha=0.8, 
                                                   label=tech_type))
                displayed_types.add(tech_type)
        
        plt.title('Agent Total Profit Comparison (Sorted by Profit)', fontsize=16)
        plt.xlabel('Agent (Sorted by Total Profit)', fontsize=14)
        plt.ylabel('Total Profit', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # 保存图表
        output_file_bar = os.path.join(self.output_dir, 'agent_total_profits.png')
        plt.savefig(output_file_bar, dpi=300)
        plt.close()
        
        return output_file
    
    def visualize_market_prices(self) -> str:
        """
        Visualize market price trends.
        
        Returns:
            Generated chart file path
        """
        plt.figure(figsize=(12, 6))
        
        # Plot energy price and reserve price
        plt.plot(self.data['round'], self.data['energy_price'], 'b-', marker='o', linewidth=2, label='Energy Price')
        plt.plot(self.data['round'], self.data['reserve_price'], 'r-', marker='s', linewidth=2, label='Reserve Price')
        
        # Set chart properties
        plt.title('Market Price Trends', fontsize=16)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.output_dir, 'market_prices.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        return output_file
    
    def visualize_market_clearing(self) -> str:
        """
        Visualize market clearing situation.
        
        Returns:
            Generated chart file path
        """
        # 提取智能体ID列表
        agent_ids = [col.split('_cleared_energy')[0] for col in self.data.columns if col.endswith('_cleared_energy')]
        
        # 定义技术类型颜色映射（与其他图保持一致）
        tech_colors = {
            'coal': '#8B4513',    # 棕色 - 燃煤
            'gas': '#FF6B35',     # 橙红色 - 燃气
            'hydro': '#4A90E2',   # 蓝色 - 水电
            'wind': '#7ED321',    # 绿色 - 风电
            'solar': '#F5A623',   # 黄色 - 光伏
            'nuclear': '#9013FE', # 紫色 - 核电
            'biomass': '#50E3C2', # 青绿色 - 生物质
            'storage': '#BD10E0', # 品红色 - 储能
            'other': '#9B9B9B'    # 灰色 - 其他
        }
        
        # 按技术类型分组智能体
        tech_groups = {}
        for agent_id in agent_ids:
            tech_type = self.agent_categories.get(agent_id, 'other')
            # 标准化技术类型名称
            tech_type = self.tech_type_mapping.get(tech_type, tech_type)
            
            if tech_type not in tech_groups:
                tech_groups[tech_type] = []
            tech_groups[tech_type].append(agent_id)
        
        # 对每个技术类型内的智能体按序号排序
        for tech_type in tech_groups:
            tech_groups[tech_type].sort(key=lambda x: int(x.split('-')[1]) if '-' in x and x.split('-')[1].isdigit() else 0)
        
        # 按技术类型排序，创建最终的有序智能体列表
        tech_order = ['Black_Coal', 'Brown_Coal', 'Natural_Gas', 'Hydro', 'Wind', 'Solar', 'nuclear', 'biomass', 'storage', 'other']
        ordered_agent_ids = []
        ordered_colors = []
        
        for tech_type in tech_order:
            if tech_type in tech_groups:
                color = self.tech_colors.get(tech_type, '#9B9B9B')
                for agent_id in tech_groups[tech_type]:
                    ordered_agent_ids.append(agent_id)
                    ordered_colors.append(color)
        
        # 创建单个图表
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Energy clearing volume - 按排序后的顺序创建堆叠图
        energy_data = []
        for agent_id in ordered_agent_ids:
            energy_data.append(self.data[f"{agent_id}_cleared_energy"].values)
        
        # 创建堆叠柱状图 - 按技术类型分组但显示每个智能体
        if ordered_agent_ids:
            bottom = np.zeros(len(self.data['round']))
            
            # 按技术类型顺序显示每个智能体
            for tech_type in tech_order:
                if tech_type in tech_groups:
                    color = self.tech_colors.get(tech_type, '#9B9B9B')
                    # 为同一技术类型的智能体生成相近但可区分的颜色
                    agents_in_type = tech_groups[tech_type]
                    
                    for i, agent_id in enumerate(agents_in_type):
                        agent_data = self.data[f"{agent_id}_cleared_energy"].values
                        if np.sum(agent_data) > 0:  # 只显示有数据的智能体
                            # 为同类型智能体创建颜色变化
                            if len(agents_in_type) > 1:
                                # 通过调整亮度创建颜色变化
                                import matplotlib.colors as mcolors
                                base_color = mcolors.to_rgb(color)
                                # 创建亮度变化 (0.7 到 1.0)
                                brightness = 0.7 + 0.3 * i / max(1, len(agents_in_type) - 1)
                                varied_color = tuple(min(1.0, c * brightness) for c in base_color)
                            else:
                                varied_color = color
                            
                            ax.bar(self.data['round'], agent_data, bottom=bottom, 
                                   color=varied_color, label=f'{agent_id}', alpha=0.8)
                            bottom += agent_data
        
        # Add demand line
        ax.plot(self.data['round'], self.data['energy_demand'], 'r--', linewidth=2, label='Energy Demand')
        
        ax.set_title('Energy Market Clearing Volume', fontsize=14)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Clearing Volume (MW)', fontsize=12)
        # 创建分组图例
        legend_elements = []
        for tech_type in tech_order:
            if tech_type in tech_groups and len(tech_groups[tech_type]) > 0:
                color = self.tech_colors.get(tech_type, '#9B9B9B')
                legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, 
                                                 label=f'{tech_type} ({len(tech_groups[tech_type])} agents)'))
        
        # 为市场出清图创建详细图例，显示每个智能体
        detailed_legend = []
        for tech_type in tech_order:
            if tech_type in tech_groups:
                agents_in_type = tech_groups[tech_type]
                base_color = self.tech_colors.get(tech_type, '#9B9B9B')
                
                for i, agent_id in enumerate(agents_in_type):
                    agent_data = self.data[f"{agent_id}_cleared_energy"].values
                    if np.sum(agent_data) > 0:  # 只为有数据的智能体创建图例
                        # 为同类型智能体创建颜色变化
                        if len(agents_in_type) > 1:
                            import matplotlib.colors as mcolors
                            base_rgb = mcolors.to_rgb(base_color)
                            brightness = 0.7 + 0.3 * i / max(1, len(agents_in_type) - 1)
                            varied_color = tuple(min(1.0, c * brightness) for c in base_rgb)
                        else:
                            varied_color = base_color
                        
                        detailed_legend.append(plt.Line2D([0], [0], color=varied_color, lw=4, label=agent_id))
        
        # 分两列显示图例
        ax.legend(handles=detailed_legend, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2)
        ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.output_dir, 'market_clearing.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        return output_file
    
    def visualize_bid_strategies(self) -> str:
        """
        Visualize agent bidding strategies.
        
        Returns:
            Generated chart file path
        """
        # 提取智能体ID列表（排除adjusted列，避免重复）
        agent_ids = [col.replace('_energy_bid_price', '') for col in self.data.columns 
                     if col.endswith('_energy_bid_price') and not 'adjusted' in col]
        
        # 创建单个图表（只显示Energy bid price）
        plt.figure(figsize=(12, 8))
        
        # 定义技术类型颜色映射（与利润图保持一致）
        tech_colors = {
            'coal': '#8B4513',    # 棕色 - 燃煤
            'gas': '#FF6B35',     # 橙红色 - 燃气
            'hydro': '#4A90E2',   # 蓝色 - 水电
            'wind': '#7ED321',    # 绿色 - 风电
            'solar': '#F5A623',   # 黄色 - 光伏
            'nuclear': '#9013FE', # 紫色 - 核电
            'biomass': '#50E3C2', # 青绿色 - 生物质
            'storage': '#BD10E0', # 品红色 - 储能
            'other': '#9B9B9B'    # 灰色 - 其他
        }
        
        # 定义线型样式
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
        
        # 定义线宽
        line_widths = [2, 2.5, 3, 1.5, 2.2, 2.8, 1.8]
        
        # 按技术类型分组智能体（优先使用行为分析分类）
        tech_groups = {}
        for agent_id in agent_ids:
            # 使用我们权威的分类
            tech_type = self.agent_categories.get(agent_id, 'other')
            # 标准化技术类型名称
            tech_type = self.tech_type_mapping.get(tech_type, tech_type)
            
            if tech_type not in tech_groups:
                tech_groups[tech_type] = []
            tech_groups[tech_type].append(agent_id)
        
        # Energy bid price (使用调整后的报价，含碳成本)
        for tech_type, agents in tech_groups.items():
            color = self.tech_colors.get(tech_type, '#9B9B9B')
            
            for i, agent_id in enumerate(agents):
                # 选择线型和线宽
                style_idx = i % len(line_styles)
                width_idx = i % len(line_widths)
                
                # 优先使用调整后的报价，如果不存在则回退到原始报价
                price_column = f"{agent_id}_adjusted_energy_bid_price"
                if price_column not in self.data.columns:
                    price_column = f"{agent_id}_energy_bid_price"
                
                plt.plot(self.data['round'], self.data[price_column], 
                        color=color,
                        linestyle=line_styles[style_idx],
                        linewidth=line_widths[width_idx],
                        marker='o', 
                        markersize=4,
                        label=f"{agent_id} ({tech_type})",
                        alpha=0.8)
        
        # Add market clearing price
        plt.plot(self.data['round'], self.data['energy_price'], 'k--', linewidth=3, label='Market Clearing Price', zorder=3)
        
        # 绘制Ground Truth RRP
        if self.ground_truth_prices and len(self.ground_truth_prices) == len(self.df['round'].unique()):
            plt.plot(self.df['round'].unique(), self.ground_truth_prices, color='green', linestyle='-', linewidth=3, label='Ground Truth RRP', zorder=4, alpha=0.8)
        
        plt.title('Energy Bid Price Strategy', fontsize=16)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        
        # 将图例放在图表底部
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5, fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.output_dir, 'bid_strategies.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        return output_file
    
    def visualize_carbon_market(self) -> str:
        """
        Visualize carbon market price trends and market information
        
        Returns:
            Generated chart file path
        """
        plt.figure(figsize=(15, 10))
        
        # Create 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Carbon price trends
        if 'carbon_market_price' in self.data.columns:
            ax1.plot(self.data['round'], self.data['carbon_market_price'], 'g-', 
                    marker='o', linewidth=2, label='Market Price')
        if 'carbon_clearing_price' in self.data.columns:
            ax1.plot(self.data['round'], self.data['carbon_clearing_price'], 'b--', 
                    marker='s', linewidth=2, label='Clearing Price')
        
        # 绘制Ground Truth碳价
        if self.ground_truth_carbon_prices and len(self.ground_truth_carbon_prices) == len(self.df['round'].unique()):
            ax1.plot(self.df['round'].unique(), self.ground_truth_carbon_prices, color='orange', 
                    linestyle='-', linewidth=3, marker='D', label='Ground Truth ACCU Price', zorder=4, alpha=0.8)
        
        ax1.set_title('Carbon Price Trends', fontsize=14)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Price (CNY/tCO2)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True)
        
        # 2. Carbon trading volume
        if 'carbon_total_volume' in self.data.columns:
            ax2.bar(self.data['round'], self.data['carbon_total_volume'], 
                   color='lightgreen', alpha=0.7, label='Total Volume')
        if 'carbon_num_trades' in self.data.columns:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(self.data['round'], self.data['carbon_num_trades'], 
                         'r-', marker='o', linewidth=2, label='Number of Trades')
            ax2_twin.set_ylabel('Number of Trades', fontsize=12, color='r')
            ax2_twin.tick_params(axis='y', labelcolor='r')
        
        ax2.set_title('Carbon Trading Volume Statistics', fontsize=14)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Trading Volume (tCO2)', fontsize=12)
        ax2.legend(fontsize=10, loc='upper left')
        ax2.grid(True)
        
        # 3. Supply and demand relationship
        if 'carbon_total_buy_demand' in self.data.columns and 'carbon_total_sell_supply' in self.data.columns:
            ax3.plot(self.data['round'], self.data['carbon_total_buy_demand'], 
                    'r-', marker='o', linewidth=2, label='Buy Demand')
            ax3.plot(self.data['round'], self.data['carbon_total_sell_supply'], 
                    'b-', marker='s', linewidth=2, label='Sell Supply')
            if 'carbon_external_supply' in self.data.columns:
                ax3.plot(self.data['round'], self.data['carbon_external_supply'], 
                        'g--', linewidth=2, label='External Supply')
        
        ax3.set_title('Carbon Market Supply & Demand', fontsize=14)
        ax3.set_xlabel('Round', fontsize=12)
        ax3.set_ylabel('Quantity (tCO2)', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True)
        
        # 4. Market tightness and price changes
        if 'carbon_market_tightness' in self.data.columns:
            ax4.plot(self.data['round'], self.data['carbon_market_tightness'], 
                    'purple', marker='o', linewidth=2, label='Market Tightness')
        if 'carbon_price_change' in self.data.columns:
            ax4_twin = ax4.twinx()
            ax4_twin.bar(self.data['round'], self.data['carbon_price_change'], 
                        alpha=0.5, color='orange', label='Price Change Rate')
            ax4_twin.set_ylabel('Price Change Rate (%)', fontsize=12, color='orange')
            ax4_twin.tick_params(axis='y', labelcolor='orange')
        
        ax4.set_title('Market Tightness & Price Change', fontsize=14)
        ax4.set_xlabel('Round', fontsize=12)
        ax4.set_ylabel('Market Tightness', fontsize=12)
        ax4.legend(fontsize=10, loc='upper left')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.output_dir, 'carbon_market_overview.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        return output_file
    
    def visualize_carbon_trading(self) -> str:
        """
        Visualize carbon trading details
        
        Returns:
            Generated chart file path
        """
        plt.figure(figsize=(12, 8))
        
        # Extract agent ID list
        agent_ids = []
        for col in self.data.columns:
            if col.endswith('_carbon_cost'):
                agent_ids.append(col.replace('_carbon_cost', ''))
        
        if not agent_ids:
            # If no carbon cost data, return empty chart
            plt.text(0.5, 0.5, 'No Carbon Trading Data', ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
            plt.title('Carbon Trading Cost Analysis', fontsize=16)
            output_file = os.path.join(self.output_dir, 'carbon_trading_analysis.png')
            plt.savefig(output_file, dpi=300)
            plt.close()
            return output_file
        
        # 定义技术类型颜色映射（与其他图保持一致）
        tech_colors = {
            'coal': '#8B4513',    # 棕色 - 燃煤
            'gas': '#FF6B35',     # 橙红色 - 燃气
            'hydro': '#4A90E2',   # 蓝色 - 水电
            'wind': '#7ED321',    # 绿色 - 风电
            'solar': '#F5A623',   # 黄色 - 光伏
            'nuclear': '#9013FE', # 紫色 - 核电
            'biomass': '#50E3C2', # 青绿色 - 生物质
            'storage': '#BD10E0', # 品红色 - 储能
            'other': '#9B9B9B'    # 灰色 - 其他
        }
        
        # 定义线型样式
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
        
        # 定义线宽
        line_widths = [2, 2.5, 3, 1.5, 2.2, 2.8, 1.8]
        
        # 按技术类型分组智能体（优先使用行为分析分类）
        tech_groups = {}
        for agent_id in agent_ids:
            tech_type = self.agent_categories.get(agent_id, 'other')
            # 标准化技术类型名称
            tech_type = self.tech_type_mapping.get(tech_type, tech_type)
            
            if tech_type not in tech_groups:
                tech_groups[tech_type] = []
            tech_groups[tech_type].append(agent_id)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # 1. Agent carbon trading costs
        for tech_type, agents in tech_groups.items():
            base_color = self.tech_colors.get(tech_type, '#9B9B9B')
            
            for i, agent_id in enumerate(agents):
                cost_col = f"{agent_id}_carbon_cost"
                if cost_col in self.data.columns:
                    # 为同类型智能体创建颜色变化
                    if len(agents) > 1:
                        import matplotlib.colors as mcolors
                        base_rgb = mcolors.to_rgb(base_color)
                        brightness = 0.7 + 0.3 * i / max(1, len(agents) - 1)
                        varied_color = tuple(min(1.0, c * brightness) for c in base_rgb)
                    else:
                        varied_color = base_color
                    
                    # 选择线型和线宽
                    style_idx = i % len(line_styles)
                    width_idx = i % len(line_widths)
                    
                    ax1.plot(self.data['round'], self.data[cost_col], 
                            color=varied_color,
                            linestyle=line_styles[style_idx],
                            linewidth=line_widths[width_idx],
                            marker='o', 
                            markersize=4,
                            label=f'{agent_id} ({tech_type})',
                            alpha=0.8)
        
        ax1.set_title('Agent Carbon Trading Costs', fontsize=14)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Carbon Cost (CNY)', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # 2. Agent quota changes
        for tech_type, agents in tech_groups.items():
            base_color = self.tech_colors.get(tech_type, '#9B9B9B')
            
            for i, agent_id in enumerate(agents):
                quota_col = f"{agent_id}_carbon_quota"
                if quota_col in self.data.columns:
                    # 为同类型智能体创建颜色变化
                    if len(agents) > 1:
                        import matplotlib.colors as mcolors
                        base_rgb = mcolors.to_rgb(base_color)
                        brightness = 0.7 + 0.3 * i / max(1, len(agents) - 1)
                        varied_color = tuple(min(1.0, c * brightness) for c in base_rgb)
                    else:
                        varied_color = base_color
                    
                    # 选择线型和线宽
                    style_idx = i % len(line_styles)
                    width_idx = i % len(line_widths)
                    
                    ax2.plot(self.data['round'], self.data[quota_col], 
                            color=varied_color,
                            linestyle=line_styles[style_idx],
                            linewidth=line_widths[width_idx],
                            marker='s', 
                            markersize=4,
                            label=f'{agent_id} ({tech_type})',
                            alpha=0.8)
        
        ax2.set_title('Agent Carbon Quota Changes', fontsize=14)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Carbon Quota (tCO2)', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.output_dir, 'carbon_trading_analysis.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        return output_file
    
    def visualize_carbon_quotas(self) -> str:
        """
        Visualize carbon quota allocation and usage
        
        Returns:
            Generated chart file path
        """
        plt.figure(figsize=(12, 8))
        
        # Extract agent ID list
        agent_ids = []
        for col in self.data.columns:
            if col.endswith('_carbon_quota'):
                agent_ids.append(col.replace('_carbon_quota', ''))
        
        if not agent_ids:
            # If no quota data, create example chart
            plt.text(0.5, 0.5, 'No Carbon Quota Data', ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
            plt.title('Carbon Quota Analysis', fontsize=16)
            output_file = os.path.join(self.output_dir, 'carbon_quota_analysis.png')
            plt.savefig(output_file, dpi=300)
            plt.close()
            return output_file
        
        # 定义技术类型颜色映射（与其他图保持一致）
        tech_colors = {
            'coal': '#8B4513',    # 棕色 - 燃煤
            'gas': '#FF6B35',     # 橙红色 - 燃气
            'hydro': '#4A90E2',   # 蓝色 - 水电
            'wind': '#7ED321',    # 绿色 - 风电
            'solar': '#F5A623',   # 黄色 - 光伏
            'nuclear': '#9013FE', # 紫色 - 核电
            'biomass': '#50E3C2', # 青绿色 - 生物质
            'storage': '#BD10E0', # 品红色 - 储能
            'other': '#9B9B9B'    # 灰色 - 其他
        }
        
        # 定义技术类型顺序
        tech_order = ['Black_Coal', 'Brown_Coal', 'Natural_Gas', 'Hydro', 'Wind', 'Solar', 'nuclear', 'biomass', 'storage', 'other']
        
        # 按技术类型分组智能体（优先行为分析分类）
        tech_groups = {}
        for agent_id in agent_ids:
            tech_type = self.agent_categories.get(agent_id, 'other')
            # 标准化技术类型名称
            tech_type = self.tech_type_mapping.get(tech_type, tech_type)
            
            if tech_type not in tech_groups:
                tech_groups[tech_type] = []
            tech_groups[tech_type].append(agent_id)
        
        # Create subplots with adjusted spacing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        plt.subplots_adjust(left=0.05, right=0.75, wspace=0.3)

        # 1. Final quota distribution (pie chart)
        final_quotas = {}
        for agent_id in agent_ids:
            quota_col = f"{agent_id}_carbon_quota"
            if quota_col in self.data.columns and len(self.data) > 0:
                final_quotas[agent_id] = self.data[quota_col].iloc[-1]
        
        if final_quotas:
            # 按技术类型重新排序所有智能体，不限制数量
            tech_agent_quotas = {}
            for tech_type in tech_order:
                tech_agent_quotas[tech_type] = []
            
            for agent_id, quota in final_quotas.items():
                if quota > 0:  # 只显示有配额的智能体
                    tech_type = self.agent_categories.get(agent_id, 'other')
                    tech_type = self.tech_type_mapping.get(tech_type, tech_type)
                    tech_agent_quotas[tech_type].append((agent_id, quota))
            
            # 按配额大小对每个技术类型内的智能体排序
            for tech_type in tech_agent_quotas:
                tech_agent_quotas[tech_type].sort(key=lambda x: x[1], reverse=True)
            
            # 创建饼图数据
            labels = []
            sizes = []
            colors = []
            
            for tech_type in tech_order:
                if tech_agent_quotas[tech_type]:
                    base_color = self.tech_colors.get(tech_type, '#9B9B9B')
                    agents_in_type = tech_agent_quotas[tech_type]
                    
                    for i, (agent_id, quota) in enumerate(agents_in_type):
                        labels.append(f'{agent_id}')
                        sizes.append(quota)
                        
                        # 为同类型智能体创建颜色变化
                        if len(agents_in_type) > 1:
                            import matplotlib.colors as mcolors
                            base_rgb = mcolors.to_rgb(base_color)
                            brightness = 0.7 + 0.3 * i / max(1, len(agents_in_type) - 1)
                            varied_color = tuple(min(1.0, c * brightness) for c in base_rgb)
                            colors.append(varied_color)
                        else:
                            colors.append(base_color)

            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 6})
            ax1.set_title('Final Carbon Quota Distribution (All Agents)', fontsize=14)
            ax1.axis('equal')

        # 2. Agent quota change trends (stacked area chart)
        quota_data = []
        ordered_agent_ids = []
        ordered_colors = []
        
        # 按技术类型分组显示每个智能体的配额变化
        tech_order = ['Black_Coal', 'Brown_Coal', 'Natural_Gas', 'Hydro', 'Wind', 'Solar', 'nuclear', 'biomass', 'storage', 'other']
        
        for tech_type in tech_order:
            if tech_type in tech_groups:
                base_color = self.tech_colors.get(tech_type, '#9B9B9B')
                agents_in_type = tech_groups[tech_type]
                
                for i, agent_id in enumerate(agents_in_type):
                    quota_col = f"{agent_id}_carbon_quota"
                    if quota_col in self.data.columns:
                        agent_quota_data = self.data[quota_col].values
                        if np.sum(agent_quota_data) > 0:  # 只显示有数据的智能体
                            # 为同类型智能体创建颜色变化
                            if len(agents_in_type) > 1:
                                import matplotlib.colors as mcolors
                                base_rgb = mcolors.to_rgb(base_color)
                                brightness = 0.7 + 0.3 * i / max(1, len(agents_in_type) - 1)
                                varied_color = tuple(min(1.0, c * brightness) for c in base_rgb)
                            else:
                                varied_color = base_color
                            
                            quota_data.append(agent_quota_data)
                            ordered_agent_ids.append(agent_id)
                            ordered_colors.append(varied_color)
        
        if quota_data:
            quota_data = np.row_stack(quota_data)
            ax2.stackplot(self.data['round'], *quota_data, labels=ordered_agent_ids, colors=ordered_colors, alpha=0.8)

        ax2.set_title('Carbon Quota Change Trends (Grouped by Technology)', fontsize=14)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Quota (tCO2)', fontsize=12)
        # 创建分组图例
        legend_elements = []
        for tech_type in tech_order:
            if tech_type in tech_groups and len(tech_groups[tech_type]) > 0:
                # 检查该技术类型是否有数据
                has_data = False
                for agent_id in tech_groups[tech_type]:
                    quota_col = f"{agent_id}_carbon_quota"
                    if quota_col in self.data.columns and np.sum(self.data[quota_col].values) > 0:
                        has_data = True
                        break
                
                if has_data:
                    color = self.tech_colors.get(tech_type, '#9B9B9B')
                    legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, 
                                                     label=f'{tech_type} ({len(tech_groups[tech_type])} agents)'))
        
        # 显示技术类型分组图例
        ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.output_dir, 'carbon_quota_analysis.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        return output_file

    def visualize_lmp_analysis(self) -> str:
        """
        可视化LMP节点电价分析，包括平均、最高、最低电价统计

        Returns:
            Generated chart file path
        """
        plt.figure(figsize=(15, 10))

        # 创建2x2子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 检查是否有LMP相关数据
        lmp_columns = [col for col in self.data.columns if 'lmp' in col.lower() or 'energy_price' in col.lower()]

        if 'energy_price' not in self.data.columns:
            # 如果没有电价数据，创建示例图表
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No LMP Data Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
            plt.suptitle('LMP Node Price Analysis (No Data)', fontsize=16)
            output_file = os.path.join(self.output_dir, 'lmp_analysis.png')
            plt.savefig(output_file, dpi=300)
            plt.close()
            return output_file

        # 1. 电价趋势图（左上）
        ax1.plot(self.data['round'], self.data['energy_price'], linestyle='-', marker='o',
                linewidth=2, label='Energy Price (LMP)', color='#2E86AB')
        if 'reserve_price' in self.data.columns:
            ax1.plot(self.data['round'], self.data['reserve_price'], linestyle='--', marker='s',
                    linewidth=2, label='Reserve Price', color='#A23B72')

        ax1.set_title('LMP Price Trends', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Price (CNY/MWh)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. 电价统计分析（右上）
        energy_prices = self.data['energy_price'].values
        price_stats = {
            'Average': np.mean(energy_prices),
            'Maximum': np.max(energy_prices),
            'Minimum': np.min(energy_prices),
            'Std Dev': np.std(energy_prices)
        }

        bars = ax2.bar(range(len(price_stats)), list(price_stats.values()),
                      color=['#F18F01', '#C73E1D', '#2E86AB', '#A23B72'])
        ax2.set_title('LMP Price Statistics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Price (CNY/MWh)', fontsize=12)
        ax2.set_xticks(range(len(price_stats)))
        ax2.set_xticklabels(list(price_stats.keys()), rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加数值标签
        for bar, value in zip(bars, price_stats.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)

        # 3. 电价分布直方图（左下）
        ax3.hist(energy_prices, bins=15, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax3.axvline(np.mean(energy_prices), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(energy_prices):.2f}')
        ax3.axvline(np.median(energy_prices), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(energy_prices):.2f}')
        ax3.set_title('LMP Price Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Price (CNY/MWh)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. 电价波动性分析（右下）
        # 计算滚动标准差（波动性）
        window_size = min(5, len(energy_prices))
        if len(energy_prices) >= window_size:
            rolling_std = pd.Series(energy_prices).rolling(window=window_size).std()
            ax4.plot(self.data['round'], rolling_std, linestyle='-', marker='o', linewidth=2,
                    color='#A23B72', label=f'{window_size}-Round Rolling Std')
            ax4.fill_between(self.data['round'], rolling_std, alpha=0.3, color='#A23B72')

        # 计算价格变化率
        price_changes = np.diff(energy_prices) / energy_prices[:-1] * 100
        if len(price_changes) > 0:
            ax4_twin = ax4.twinx()
            ax4_twin.bar(self.data['round'][1:], price_changes, alpha=0.5,
                        color='orange', label='Price Change %')
            ax4_twin.set_ylabel('Price Change (%)', fontsize=12, color='orange')
            ax4_twin.tick_params(axis='y', labelcolor='orange')

        ax4.set_title('LMP Price Volatility Analysis', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Round', fontsize=12)
        ax4.set_ylabel('Rolling Std Dev', fontsize=12, color='#A23B72')
        ax4.tick_params(axis='y', labelcolor='#A23B72')
        ax4.legend(loc='upper left', fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('LMP Node Price Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.98)

        # 保存图表
        output_file = os.path.join(self.output_dir, 'lmp_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    def visualize_token_usage(self) -> str:
        """
        可视化LLM Token使用统计分析

        Returns:
            Generated chart file path
        """
        plt.figure(figsize=(16, 12))

        # 创建2x2子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 提取智能体ID列表和token相关列
        agent_ids = []
        token_columns = []
        for col in self.data.columns:
            if col.endswith('_total_tokens'):
                agent_id = col.replace('_total_tokens', '')
                agent_ids.append(agent_id)
                token_columns.extend([
                    f"{agent_id}_prompt_tokens",
                    f"{agent_id}_completion_tokens",
                    f"{agent_id}_total_tokens"
                ])

        if not agent_ids:
            # 如果没有token数据，创建示例图表
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No Token Usage Data Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
            plt.suptitle('LLM Token Usage Analysis (No Data)', fontsize=16)
            output_file = os.path.join(self.output_dir, 'token_usage_analysis.png')
            plt.savefig(output_file, dpi=300)
            plt.close()
            return output_file

        # 1. 每轮总Token使用趋势（左上）
        total_tokens_per_round = []
        for _, row in self.data.iterrows():
            # 添加NaN检查，将NaN值视为0
            round_total = sum(0 if np.isnan(row.get(f"{agent_id}_total_tokens", 0)) else row.get(f"{agent_id}_total_tokens", 0) for agent_id in agent_ids)
            total_tokens_per_round.append(round_total)

        # 确保没有NaN值
        total_tokens_per_round = [0 if np.isnan(x) else x for x in total_tokens_per_round]
        
        ax1.plot(self.data['round'], total_tokens_per_round, linestyle='-', marker='o',
                linewidth=3, markersize=6, color='#2E86AB')
        ax1.fill_between(self.data['round'], total_tokens_per_round, alpha=0.3, color='#2E86AB')
        ax1.set_title('Total Token Usage Per Round', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Total Tokens', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 添加平均线，确保处理NaN值
        avg_tokens = np.nanmean(total_tokens_per_round) if total_tokens_per_round else 0
        if not np.isnan(avg_tokens):
            ax1.axhline(y=avg_tokens, color='red', linestyle='--', linewidth=2,
                      label=f'Average: {int(avg_tokens)} tokens')
            ax1.legend(fontsize=10)

        # 2. 各智能体平均Token使用对比（右上）
        agent_avg_tokens = []
        agent_labels = []
        for agent_id in agent_ids[:10]:  # 只显示前10个智能体，避免图表过于拥挤
            avg_tokens = self.data[f"{agent_id}_total_tokens"].mean()
            # 添加NaN检查
            if not np.isnan(avg_tokens):
                agent_avg_tokens.append(avg_tokens)
                # 处理新的ID格式，如coal-1, gas-2等
                if 'GenCo_LLM_' in agent_id:
                    agent_labels.append(agent_id.replace('GenCo_LLM_', 'Agent '))
                else:
                    agent_labels.append(agent_id)

        bars = ax2.bar(range(len(agent_avg_tokens)), agent_avg_tokens,
                      color=plt.cm.Set3(np.linspace(0, 1, len(agent_avg_tokens))))
        ax2.set_title('Average Token Usage by Agent', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Tokens per Round', fontsize=12)
        ax2.set_xticks(range(len(agent_labels)))
        ax2.set_xticklabels(agent_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加数值标签
        for bar, value in zip(bars, agent_avg_tokens):
            height = bar.get_height()
            # 确保height不是NaN
            if not np.isnan(height):
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(value) if not np.isnan(value) else 0}', ha='center', va='bottom', fontsize=9)

        # 3. Token类型分布（左下）
        # 计算总的prompt tokens和completion tokens，添加NaN检查
        total_prompt_tokens = sum(0 if np.isnan(self.data[f"{agent_id}_prompt_tokens"].sum()) else self.data[f"{agent_id}_prompt_tokens"].sum() for agent_id in agent_ids)
        total_completion_tokens = sum(0 if np.isnan(self.data[f"{agent_id}_completion_tokens"].sum()) else self.data[f"{agent_id}_completion_tokens"].sum() for agent_id in agent_ids)

        token_types = ['Prompt Tokens', 'Completion Tokens']
        token_counts = [total_prompt_tokens, total_completion_tokens]
        
        # 确保没有NaN或者零值导致pie图绘制失败
        if sum(token_counts) > 0 and all(not np.isnan(x) for x in token_counts):
            colors = ['#F18F01', '#C73E1D']
            wedges, texts, autotexts = ax3.pie(token_counts, labels=token_types, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
        else:
            # 如果没有有效的token数据，显示提示信息
            ax3.text(0.5, 0.5, 'No Token Type Data Available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Token Type Distribution', fontsize=14, fontweight='bold')

        # 4. Token使用效率分析（右下）
        # 计算每个智能体的token效率（利润/token）
        efficiency_data = []
        efficiency_labels = []

        for agent_id in agent_ids[:8]:  # 显示前8个智能体
            total_tokens = self.data[f"{agent_id}_total_tokens"].sum()
            total_profit = self.data[f"{agent_id}_profit"].sum()

            # 添加NaN检查和处理
            if total_tokens > 0 and not np.isnan(total_tokens) and not np.isnan(total_profit):
                efficiency = total_profit / total_tokens  # 每token的利润
                efficiency_data.append(efficiency)
                # 处理新的ID格式，如coal-1, gas-2等
                if 'GenCo_LLM_' in agent_id:
                    efficiency_labels.append(agent_id.replace('GenCo_LLM_', 'A'))
                else:
                    efficiency_labels.append(agent_id)

        if efficiency_data:
            bars = ax4.bar(range(len(efficiency_data)), efficiency_data,
                          color=plt.cm.viridis(np.linspace(0, 1, len(efficiency_data))))
            ax4.set_title('Token Efficiency (Profit per Token)', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Profit per Token (CNY/Token)', fontsize=12)
            ax4.set_xlabel('Agent', fontsize=12)
            ax4.set_xticks(range(len(efficiency_labels)))
            ax4.set_xticklabels(efficiency_labels, rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')

            # 添加零线
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

            # 在柱状图上添加数值标签
            for bar, value in zip(bars, efficiency_data):
                height = bar.get_height()
                # 确保height不是NaN
                if not np.isnan(height):
                    ax4.text(bar.get_x() + bar.get_width()/2.,
                            height + (0.01 if height >= 0 else -0.01),
                            f'{value:.3f}', ha='center',
                            va='bottom' if height >= 0 else 'top', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'No Efficiency Data', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)

        plt.tight_layout()
        plt.suptitle('LLM Token Usage Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.98)

        # 保存图表
        output_file = os.path.join(self.output_dir, 'token_usage_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file