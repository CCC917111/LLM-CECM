# generate_true_digital_twin_agents.py
import pandas as pd
import json
import random
import numpy as np
from collections import defaultdict

# --- 核心配置 ---

# 1. 目标供应结构 (来自您图片中的 Q3 2022 数据)
q3_2022_supply_mix = {
    'Black_Coal': 44.0,
    'Brown_Coal': 14.6,
    'Natural_Gas': 7.1, # 脚本内部会处理 'Gas' -> 'Natural_Gas'
    'Hydro': 8.7,
    'Wind': 13.6,
    'Solar': 4.5 + 7.1, # 合并 Grid Solar 和 Distributed PV
    'Other': 0.3
}

# 2. 边际成本估算范围 (元/MWh) - 这是我们唯一需要合理估算的部分
COST_RANGES = {
    'Black_Coal': (60, 95),
    'Brown_Coal': (50, 65),
    'Natural_Gas': (80, 150), # 天然气成本会被动态价格覆盖，但基础成本仍需设定
    'Hydro': (15, 30),
    'Wind': (0, 5),
    'Solar': (0, 5),
    'Other': (70, 100)
}

# 3. 智能体总数
TOTAL_AGENTS = 50

# 4. 数据文件路径
SCADA_FILE = 'downloads/PUBLIC_DVD_DISPATCH_UNIT_SCADA_202204010000.CSV'
EXCEL_FILE = 'downloads/NEM Generation Information July 2025 (1).xlsx'
DYNAMIC_RATIOS_FILE = 'tools/fuel_dynamic_ratios.json'

# --- 辅助函数 (部分来自 classify_scada_duids.py) ---

EXACT = { 'NPS': 'Black_Coal', 'DDPS1': 'Natural_Gas', 'MPP_1': 'Black_Coal', 'MPP_2': 'Black_Coal', 'TNPS1': 'Natural_Gas', 'TALWA1': 'Natural_Gas', 'ER01': 'Black_Coal', 'ER02': 'Black_Coal', 'ER03': 'Black_Coal', 'ER04': 'Black_Coal', 'BW01': 'Black_Coal', 'BW02': 'Black_Coal', 'BW03': 'Black_Coal', 'BW04': 'Black_Coal', 'VP5': 'Black_Coal', 'VP6': 'Black_Coal', 'LYA1': 'Brown_Coal', 'LYA2': 'Brown_Coal', 'LYA3': 'Brown_Coal', 'LYA4': 'Brown_Coal', 'LOYYB1': 'Brown_Coal', 'LOYYB2': 'Brown_Coal', 'YWPS1': 'Brown_Coal', 'YWPS2': 'Brown_Coal', 'YWPS3': 'Brown_Coal', 'YWPS4': 'Brown_Coal', 'CALL_B_1': 'Black_Coal', 'CALL_B_2': 'Black_Coal', 'MP1': 'Black_Coal', 'KPP_1': 'Black_Coal', 'LD01': 'Black_Coal', 'LD02': 'Black_Coal', 'LD03': 'Black_Coal', 'LD04': 'Black_Coal', 'MACARTH1': 'Wind', 'VBBG1': 'Battery'}
PREFIX = [('STAN-', 'Black_Coal'), ('TARONG', 'Black_Coal'), ('GSTONE', 'Black_Coal'), ('CALL', 'Black_Coal'), ('MP', 'Black_Coal'), ('KPP', 'Black_Coal'), ('LYA', 'Brown_Coal'), ('LOYYB', 'Brown_Coal'), ('YWPS', 'Brown_Coal'), ('BRAEMAR', 'Natural_Gas'), ('TORRENS', 'Natural_Gas'), ('TNPS', 'Natural_Gas'), ('PPCCGT', 'Natural_Gas'), ('OSBORNE', 'Natural_Gas'), ('ROMA', 'Natural_Gas'), ('OAKEY', 'Natural_Gas'), ('SWANBANK', 'Natural_Gas'), ('URANQ', 'Natural_Gas'), ('COLONGRA', 'Natural_Gas'), ('JEERALANG', 'Natural_Gas'), ('LAVERTON', 'Natural_Gas'), ('NEWPORT', 'Natural_Gas'), ('SOMERTON', 'Natural_Gas'), ('TALWA', 'Natural_Gas'), ('TUMUT', 'Hydro'), ('MURRAY', 'Hydro'), ('UPPTUMUT', 'Hydro'), ('SNOWYP', 'Hydro'), ('GORDON', 'Hydro'), ('BASTYAN', 'Hydro'), ('TREVALLYN', 'Hydro'), ('COOPGWF', 'Wind'), ('SAPHWF', 'Wind'), ('COLWF', 'Wind'), ('MACARTH', 'Wind'), ('WF', 'Wind'), ('WIND', 'Wind'), ('DARLSF', 'Solar'), ('LIMOSF', 'Solar'), ('SUNRSF', 'Solar'), ('SF', 'Solar'), ('SOLAR', 'Solar'), ('PV', 'Solar')]
EXCEL_MAP = {}

def _norm_fuel_category(raw: str) -> str:
    s = str(raw).lower()
    if 'black coal' in s or ('coal' in s and 'brown' not in s): return 'Black_Coal'
    if 'brown coal' in s: return 'Brown_Coal'
    if 'gas' in s: return 'Natural_Gas'
    if 'hydro' in s or 'water' in s: return 'Hydro'
    if 'wind' in s: return 'Wind'
    if 'solar' in s: return 'Solar'
    if 'battery' in s: return 'Battery'
    return 'Other'

def _load_excel_mapping(excel_path: str) -> int:
    global EXCEL_MAP
    try:
        df = pd.read_excel(excel_path, sheet_name='ExistingGeneration&NewDevs', header=1)
        c_asset = next((c for c in df.columns if 'asset type' in str(c).lower()), None)
        c_duid = next((c for c in df.columns if 'duid' in str(c).lower()), None)
        c_fuel = next((c for c in df.columns if 'fuel type' in str(c).lower()), None)
        c_status = next((c for c in df.columns if 'unit status' in str(c).lower()), None)
        use = df[df[c_asset].astype(str).str.contains('Existing', case=False, na=False)]
        use = use[use[c_status].astype(str).str.contains('Service', case=False, na=False)]
        use = use[[c_duid, c_fuel]].dropna()
        use[c_duid] = use[c_duid].astype(str).str.upper()
        mapped = {duid: _norm_fuel_category(fuel) for duid, fuel in zip(use[c_duid], use[c_fuel])}
        EXCEL_MAP.update(mapped)
        return len(mapped)
    except Exception as e:
        print(f"Excel加载失败: {e}")
        return 0

def classify_duid(duid: str) -> str:
    d = duid.upper()
    if d in EXCEL_MAP: return EXCEL_MAP[d]
    if d in EXACT: return EXACT[d]
    for pref, cat in PREFIX:
        if d.startswith(pref) or (pref in d): return cat
    if d.endswith('WF1') or d.endswith('WF2'): return 'Wind'
    if d.endswith('SF1') or d.endswith('SF2'): return 'Solar'
    return 'Other'

# --- 主逻辑 ---

def main():
    print("--- 终极智能体生成脚本 V2 ---")
    
    # 1. 加载所有数据源
    print(f"正在加载动态比率 from {DYNAMIC_RATIOS_FILE}...")
    with open(DYNAMIC_RATIOS_FILE, 'r') as f:
        fuel_ratios = json.load(f)

    print(f"正在加载Excel映射 from {EXCEL_FILE}...")
    _load_excel_mapping(EXCEL_FILE)

    print(f"正在从SCADA文件 {SCADA_FILE} 提取真实机组参数...")
    # 从SCADA文件中流式读取，获取每个DUID的最大容量
    real_units = defaultdict(lambda: {'max_capacity': 0.0})
    with open(SCADA_FILE, 'r') as f:
        for line in f:
            if not line.startswith('D,DISPATCH,UNIT_SCADA'): continue
            parts = line.strip().split(',')
            if len(parts) < 7: continue
            duid = parts[5].strip('"').upper()
            try:
                val = abs(float(parts[6]))
                if val > real_units[duid]['max_capacity']:
                    real_units[duid]['max_capacity'] = val
            except ValueError:
                continue

    # 过滤掉容量过小的机组并进行分类
    classified_units = defaultdict(list)
    for duid, data in real_units.items():
        if data['max_capacity'] > 1.0: # 至少1MW
            fuel_type = classify_duid(duid)
            if fuel_type != 'Other' and fuel_type != 'Battery':
                data['duid'] = duid
                data['fuel_category'] = fuel_type
                classified_units[fuel_type].append(data)
    
    print("真实机组数据提取和分类完成。")

    # 2. 根据供应结构，计算每种燃料类型需要采样的智能体数量
    total_supply_pct = sum(q3_2022_supply_mix.values())
    shares = {k: v / total_supply_pct for k, v in q3_2022_supply_mix.items()}
    
    # 使用最大余数法分配智能体数量
    agent_counts = {k: int(v * TOTAL_AGENTS) for k, v in shares.items()}
    remainder = {k: (v * TOTAL_AGENTS) - agent_counts[k] for k, v in shares.items()}
    assigned = sum(agent_counts.values())
    to_assign = TOTAL_AGENTS - assigned
    for k, _ in sorted(remainder.items(), key=lambda item: item[1], reverse=True):
        if to_assign <= 0: break
        agent_counts[k] += 1
        to_assign -= 1

    print("根据Q3 2022供应结构，确定的智能体采样数量：")
    for fuel, count in agent_counts.items():
        print(f"  - {fuel}: {count} 个")

    # 3. 从真实机组池中进行采样，并创建智能体
    final_agents = []
    for fuel_type, count in agent_counts.items():
        if count == 0: continue
        
        unit_pool = classified_units.get(fuel_type, [])
        if not unit_pool:
            print(f"警告: 燃料类型 {fuel_type} 的真实机组池为空，无法采样！")
            continue
        
        # 进行有放回的随机采样
        sampled_units = random.choices(unit_pool, k=count)
        
        for i, unit in enumerate(sampled_units):
            # 获取动态比率
            ratios = fuel_ratios.get(fuel_type, {})
            pmin_ratio = ratios.get('pmin_ratio', 0.1)
            ramp_up_ratio = ratios.get('ramp_up_ratio', 0.5)
            ramp_down_ratio = ratios.get('ramp_down_ratio', 0.5)

            # 估算有差异的成本
            cost_min, cost_max = COST_RANGES.get(fuel_type, (50, 100))
            # 引入与容量相关的微小扰动
            base_cost = random.uniform(cost_min, cost_max)
            capacity_factor = (unit['max_capacity'] - 50) / 1000 # 假设容量越大成本越高
            final_cost = base_cost + capacity_factor * 10
            final_cost = max(cost_min, min(cost_max, final_cost))

            # 确定bidding_personality
            personality = 'default'
            if fuel_type in ['Black_Coal', 'Brown_Coal']:
                personality = 'baseload'
            elif fuel_type == 'Natural_Gas':
                personality = 'peaker'
            elif fuel_type in ['Wind', 'Solar']:
                personality = 'renewable'

            agent_config = {
                'agent_id': f"{unit['duid']}_{i}", # 保证ID唯一性
                'agent_type': 'LLM',
                'config': {
                    'marginal_cost_energy': round(final_cost, 2),
                    'marginal_cost_reserve': round(final_cost * 0.3, 2),
                    'max_capacity': round(unit['max_capacity'] * 0.85 if fuel_type == 'Wind' else unit['max_capacity'], 2),
                    'fuel_category': fuel_type,
                    'min_output': round(unit['max_capacity'] * pmin_ratio, 2),
                    'ramp_up_mw': round(unit['max_capacity'] * ramp_up_ratio, 2),
                    'ramp_down_mw': round(unit['max_capacity'] * ramp_down_ratio, 2),
                    'emission_factor': 0.8 if 'Coal' in fuel_type else (0.4 if 'Gas' in fuel_type else 0.0)
                },
                'llm_config': {
                    'model': 'gemma-3-27b-it',
                    'enable_adf': False,
                    'enable_ibr_cr': True,
                    'bidding_personality': personality
                }
            }
            final_agents.append(agent_config)

    # 4. 保存到文件
    output_filename = 'true_digital_twin_agents.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_agents, f, indent=2, ensure_ascii=False)

    print(f"\n--- 成功 ---")
    print(f"已生成终极智能体配置文件: {output_filename}")
    print(f"总计 {len(final_agents)} 个智能体。请修改 main.py 以使用此文件并重新运行仿真。")

if __name__ == '__main__':
    main()
