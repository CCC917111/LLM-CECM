import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from utils.validation import BidValidator
import time
# 尝试导入PyPower
try:
    # 避免导入有问题的pypower.api，只导入需要的常量模块
    from pypower.idx_bus import PD, QD, LAM_P, VM, VA
    from pypower.idx_gen import PG, QG, PMAX, PMIN, GEN_BUS, GEN_STATUS, QMAX, QMIN
    from pypower.idx_cost import COST, MODEL, NCOST, PW_LINEAR
    # 如果OPF功能需要，可以单独导入runopf等函数
    try:
        from pypower.runopf import runopf
        from pypower.ppoption import ppoption
        # 导入IEEE测试系统
        try:
            import pypower.api as pp
            PYPOWER_OPF_AVAILABLE = True
        except ImportError:
            # 如果api导入失败，尝试直接导入
            from pypower import case9, case14, case30, case39, case57, case118
            class PyPowerCases:
                @staticmethod
                def case9(): return case9()
                @staticmethod
                def case14(): return case14()
                @staticmethod
                def case30(): return case30()
                @staticmethod
                def case39(): return case39()
                @staticmethod
                def case57(): return case57()
                @staticmethod
                def case118(): return case118()
            pp = PyPowerCases()
            PYPOWER_OPF_AVAILABLE = True
    except ImportError:
        PYPOWER_OPF_AVAILABLE = False
        pp = None
        logging.warning("PyPower OPF功能导入失败，将仅使用基础功能。")
    PYPOWER_AVAILABLE = True
except ImportError:
    PYPOWER_AVAILABLE = False
    PYPOWER_OPF_AVAILABLE = False
    logging.warning("PyPower库未安装，考虑网络约束的出清功能将不可用。")

def simple_joint_clearing(bids: List[Dict], total_demand: float, reserve_requirement: float) -> Tuple[Dict, Dict]:
    """
    简化的现货与备用联合市场出清算法。

    基本逻辑:
    1. 将所有智能体的能源和备用报价收集起来。
    2. 按能源报价从低到高排序。
    3. 依次接受能源报价，直到满足总需求 (total_demand)。最后一个被接受的能源报价（或部分接受）的价格决定能源市场价格。
    4. 在满足能源需求的基础上，从剩余的容量（或未被能源完全选中的机组的剩余容量）中，按备用报价从低到高选择，直到满足备用需求。
    5. 最后一个被接受的备用报价决定备用市场价格。
    6. 考虑每个智能体的总容量限制。

    Args:
        bids: 一个列表，每个元素是一个字典，代表一个智能体的报价:
              {'agent_id': str,
               'energy_bid': {'price': float, 'quantity': float},
               'reserve_bid': {'price': float, 'quantity': float},
               'max_capacity': float} # 需要智能体的最大容量信息
        total_demand: 当前总的能源需求 (MW)。
        reserve_requirement: 当前的备用需求 (MW)。

    Returns:
        一个元组包含两个字典:
        1. clearing_prices: {'energy_price': float, 'reserve_price': float}
        2. agent_results: 一个字典，键是 agent_id，值是该智能体的出清结果:
           {'cleared_energy': float, 'cleared_reserve': float, 'total_cleared': float}
    """
    logging.debug(f"开始市场出清... 总需求: {total_demand:.2f}, 备用需求: {reserve_requirement:.2f}")
    if not bids:
        logging.warning("没有收到任何报价，无法出清。")
        return {'energy_price': 0, 'reserve_price': 0}, {}

    # --- 数据准备 ---
    # 将 bids 转换为 DataFrame 更易于处理
    bid_data = []
    for bid in bids:
        # 使用BidValidator验证报价
        if not BidValidator.validate_bid_for_clearing(bid):
            continue
            
        # 提取报价信息
        agent_id = bid.get('agent_id')
        e_price = bid['energy_bid'].get('price')
        e_quant = bid['energy_bid'].get('quantity')
        r_price = bid['reserve_bid'].get('price')
        r_quant = bid['reserve_bid'].get('quantity')
        max_cap = bid.get('max_capacity')
        
        # 处理负值报价
        if e_quant < 0 or r_quant < 0:
            logging.warning(f"报价数量为负: Agent {agent_id}. 将数量置为0。")
            e_quant = max(0, e_quant)
            r_quant = max(0, r_quant)
            
        # 处理超容量报价
        total_bid_quant = e_quant + r_quant
        if total_bid_quant > max_cap * 1.001: # 允许极小的误差
            logging.warning(f"Agent {agent_id} 总报价量 {total_bid_quant:.2f} > 容量 {max_cap:.2f}. 按比例缩减.")
            if total_bid_quant > 0:
                scale = max_cap / total_bid_quant
                e_quant *= scale
                r_quant *= scale
            else: # 如果总量为0，则无需缩减
                e_quant = 0
                r_quant = 0

        bid_data.append({
             'agent_id': bid['agent_id'],
             'energy_price': e_price,
             'energy_quantity': e_quant,
             'reserve_price': r_price,
             'reserve_quantity': r_quant,
             'max_capacity': max_cap,
             'cleared_energy': 0.0, # 初始化出清量
             'cleared_reserve': 0.0,
             'available_for_reserve': max_cap # 初始可用于备用的容量等于最大容量
         })

    if not bid_data:
        logging.warning("所有报价均无效，无法出清。")
        return {'energy_price': 0, 'reserve_price': 0}, {}

    bids_df = pd.DataFrame(bid_data)
    bids_df = bids_df.replace([np.inf, -np.inf], np.nan).dropna() # 移除包含inf或nan的行

    if bids_df.empty:
        logging.warning("处理和过滤后无有效报价，无法出清。")
        return {'energy_price': 0, 'reserve_price': 0}, {}

    # 显式设置相关列的数据类型为float，避免FutureWarning
    float_cols = ['cleared_energy', 'cleared_reserve', 'available_for_reserve']
    for col in float_cols:
        if col in bids_df.columns:
            bids_df[col] = bids_df[col].astype(float)
            
    # --- 能源出清 (按能源价格排序) ---
    bids_df_sorted_energy = bids_df.sort_values(by='energy_price').reset_index(drop=True)
    energy_cleared_so_far = 0.0
    energy_price = 0.0 # 默认为0，如果没有机组出清

    logging.debug("\n--- 能源出清过程 ---")
    logging.debug(f"排序后的能源报价:\n{bids_df_sorted_energy[['agent_id', 'energy_price', 'energy_quantity', 'max_capacity']].to_string()}")

    for index, row in bids_df_sorted_energy.iterrows():
        if energy_cleared_so_far >= total_demand:
            break # 需求已满足

        agent_id = row['agent_id']
        # 本次可供能源量 = min(该智能体能源报价量, 该智能体最大容量)
        # (因为智能体不能提供超过其总容量的能源)
        potential_energy = min(row['energy_quantity'], row['max_capacity'])

        # 还需要多少能源
        needed_energy = total_demand - energy_cleared_so_far

        # 本次实际能出清的能源量
        cleared_energy_this_step = min(potential_energy, needed_energy)

        if cleared_energy_this_step > 0.001: # 避免浮点数误差导致微小量出清
            # 更新该智能体的出清量
            bids_df.loc[bids_df['agent_id'] == agent_id, 'cleared_energy'] += cleared_energy_this_step
            # 更新已满足的能源需求
            energy_cleared_so_far += cleared_energy_this_step
            # 更新能源价格 (边际价格)
            energy_price = row['energy_price']
            # 更新该智能体可用于备用的容量
            # 可用备用容量 = 最大容量 - 已出清的能源量
            # 这里使用 += 是因为一个agent_id可能在bids_df中只有一行
            current_cleared_energy = bids_df.loc[bids_df['agent_id'] == agent_id, 'cleared_energy'].iloc[0]
            bids_df.loc[bids_df['agent_id'] == agent_id, 'available_for_reserve'] = max(0.0, row['max_capacity'] - current_cleared_energy)

            logging.debug(f"  Agent {agent_id}: 报价(P:{row['energy_price']:.2f}, Q:{row['energy_quantity']:.2f}), "
                          f"可供:{potential_energy:.2f}, 需求剩余:{needed_energy:.2f}, "
                          f"本次出清:{cleared_energy_this_step:.2f}, "
                          f"累计出清:{energy_cleared_so_far:.2f}, "
                          f"当前能源价:{energy_price:.2f}, "
                          f"剩余备用容量:{bids_df.loc[bids_df['agent_id'] == agent_id, 'available_for_reserve'].iloc[0]:.2f}")


    # 如果需求未满足，价格可能需要设为最高报价或惩罚价 (简化：使用最后一个出价)
    if energy_cleared_so_far < total_demand * 0.999: # 允许微小误差
        logging.warning(f"能源需求 {total_demand:.2f} 未完全满足，仅满足 {energy_cleared_so_far:.2f}。能源价格可能偏低。")
        # 可以设置一个非常高的价格，或者使用最后一个报价的价格（如果存在）
        if not bids_df_sorted_energy.empty:
             energy_price = bids_df_sorted_energy['energy_price'].iloc[-1] # 使用最高报价作为价格
             logging.warning(f"  将能源价格设置为最高报价: {energy_price:.2f}")
        else:
             energy_price = 9999 # 惩罚价

    logging.debug(f"--- 能源出清结束 --- 最终能源价格: {energy_price:.2f}, 总能源出清量: {energy_cleared_so_far:.2f}")


    # --- 备用出清 (在剩余容量上，按备用价格排序) ---
    # 筛选出还有备用容量且报了备用价的机组
    bids_df_reserve_candidates = bids_df[
        (bids_df['available_for_reserve'] > 0.001) & # 必须有剩余容量
        (bids_df['reserve_quantity'] > 0.001) &     # 必须报了备用量
        (bids_df['reserve_price'].notna())          # 必须报了备用价
    ].copy() # 使用 .copy() 避免 SettingWithCopyWarning

    # 按备用价格排序
    bids_df_reserve_sorted = bids_df_reserve_candidates.sort_values(by='reserve_price').reset_index(drop=True)

    reserve_cleared_so_far = 0.0
    reserve_price = 0.0 # 默认为0

    logging.debug("\n--- 备用出清过程 ---")
    logging.debug(f"排序后的备用候选报价:\n{bids_df_reserve_sorted[['agent_id', 'reserve_price', 'reserve_quantity', 'available_for_reserve']].to_string()}")


    for index, row in bids_df_reserve_sorted.iterrows():
        if reserve_cleared_so_far >= reserve_requirement:
            break # 备用需求已满足

        agent_id = row['agent_id']
        # 本次可供备用量 = min(该智能体备用报价量, 该智能体当前剩余可用容量)
        potential_reserve = min(row['reserve_quantity'], row['available_for_reserve'])

        # 还需要多少备用
        needed_reserve = reserve_requirement - reserve_cleared_so_far

        # 本次实际能出清的备用量
        cleared_reserve_this_step = min(potential_reserve, needed_reserve)

        if cleared_reserve_this_step > 0.001: # 避免微小量
             # 更新该智能体的备用出清量
             # 注意：这里更新的是原始的 bids_df，因为 cleared_reserve 列存在于 bids_df 中
             bids_df.loc[bids_df['agent_id'] == agent_id, 'cleared_reserve'] += cleared_reserve_this_step
             # 更新已满足的备用需求
             reserve_cleared_so_far += cleared_reserve_this_step
             # 更新备用价格 (边际价格)
             reserve_price = row['reserve_price']
             # 更新该智能体最终剩余的备用容量 (虽然在此算法中不再使用，但可以记录)
             # 这不是必需的，因为我们不会再从该容量中减去任何东西
             # bids_df.loc[bids_df['agent_id'] == agent_id, 'available_for_reserve'] -= cleared_reserve_this_step

             logging.debug(f"  Agent {agent_id}: 报价(P:{row['reserve_price']:.2f}, Q:{row['reserve_quantity']:.2f}), "
                           f"可用容量:{row['available_for_reserve']:.2f}, 可供备用:{potential_reserve:.2f}, "
                           f"需求剩余:{needed_reserve:.2f}, 本次出清:{cleared_reserve_this_step:.2f}, "
                           f"累计出清:{reserve_cleared_so_far:.2f}, 当前备用价:{reserve_price:.2f}")


    # 如果备用需求未满足
    if reserve_cleared_so_far < reserve_requirement * 0.999:
        logging.warning(f"备用需求 {reserve_requirement:.2f} 未完全满足，仅满足 {reserve_cleared_so_far:.2f}。备用价格可能偏低。")
        # 设置高价或使用最后一个报价
        if not bids_df_reserve_sorted.empty:
             reserve_price = bids_df_reserve_sorted['reserve_price'].iloc[-1]
             logging.warning(f"  将备用价格设置为最高有效报价: {reserve_price:.2f}")
        else:
            # 如果一开始就没有备用候选，备用价格可以为0或一个基础机会成本价，这里设高价表示短缺
            reserve_price = 999 # 备用短缺惩罚价

    logging.debug(f"--- 备用出清结束 --- 最终备用价格: {reserve_price:.2f}, 总备用出清量: {reserve_cleared_so_far:.2f}")


    # --- 整理结果 ---
    clearing_prices = {
        'energy_price': energy_price,
        'reserve_price': reserve_price
    }

    agent_results = {}
    bids_df['total_cleared'] = bids_df['cleared_energy'] + bids_df['cleared_reserve']

    # 检查是否有智能体的总出清量超过其容量 (理论上不应发生，但作为校验)
    over_capacity_agents = bids_df[bids_df['total_cleared'] > bids_df['max_capacity'] * 1.001]
    if not over_capacity_agents.empty:
        logging.error(f"错误：出清后发现智能体总出清量超过容量！\n{over_capacity_agents[['agent_id', 'cleared_energy', 'cleared_reserve', 'total_cleared', 'max_capacity']]}")
        # 这里应该进行错误处理或修正，但简化版暂不处理

    final_results_df = bids_df[['agent_id', 'cleared_energy', 'cleared_reserve', 'total_cleared']].copy()
    agent_results = final_results_df.set_index('agent_id').to_dict('index')

    logging.info(f"市场出清完成。能源价格: {energy_price:.2f}, 备用价格: {reserve_price:.2f}")
    logging.debug(f"各智能体出清结果:\n{final_results_df.to_string()}")

    return clearing_prices, agent_results

def run_opf_clearing(ppc: Dict, opf_options: Dict) -> Tuple[Optional[Dict], str]:
    """
    运行Pypower的最优潮流计算。
    首先尝试AC OPF，如果失败则尝试DC OPF作为备选方案。
    """
    try:
        start_time = time.time()
        
        # 定义 Pypower 求解器选项
        if PYPOWER_OPF_AVAILABLE:
            # 首先尝试 AC OPF
            ppopt = ppoption(
                VERBOSE=0, 
                OUT_ALL=0, 
                OPF_ALG=opf_options.get('OPF_ALG', 560), # 默认使用 PDIPM (AC OPF)
            )
            
            # 执行最优潮流计算 (AC OPF)
            results = runopf(ppc, ppopt)
            elapsed_time = time.time() - start_time
            
            # 检查是否成功收敛
            if results['success']:
                logging.info(f"AC OPF 求解成功，耗时 {elapsed_time:.4f} 秒。")
                return results, "success"
            else:
                logging.warning(f"AC OPF 求解器在 {elapsed_time:.4f} 秒内未能收敛，尝试 DC OPF...")
                
                # 尝试 DC OPF 作为备选方案
                ppopt_dc = ppoption(
                    VERBOSE=0, 
                    OUT_ALL=0, 
                    OPF_ALG_DC=200, # DC OPF 默认算法
                    PF_DC=1,        # 启用 DC 潮流
                )
                
                # 为 DC OPF 简化网络模型
                ppc_dc = ppc.copy()
                # DC OPF 不需要无功功率，将其设为0
                ppc_dc['gen'][:, 2] = 0  # Qg = 0
                ppc_dc['bus'][:, 3] = 0  # Qd = 0
                
                start_time_dc = time.time()
                results_dc = runopf(ppc_dc, ppopt_dc)
                elapsed_time_dc = time.time() - start_time_dc
                
                if results_dc['success']:
                    logging.info(f"DC OPF 求解成功，耗时 {elapsed_time_dc:.4f} 秒。")
                    # 将 DC 结果映射回原始格式
                    results_dc['opf_type'] = 'DC'  # 标记为 DC OPF 结果
                    return results_dc, "success_dc"
                else:
                    logging.warning(f"DC OPF 求解器在 {elapsed_time_dc:.4f} 秒内也未能收敛。")
                    return None, "convergence_fail"
        else:
            raise ImportError("PyPower OPF功能不可用")
            
    except Exception as e:
        logging.error(f"执行 OPF 计算时发生未知错误: {e}", exc_info=True)
        return None, f"exception: {e}"

def dynamic_mapping_opf(base_ppc, bids, opf_options):
    """
    动态映射发电机到负荷节点，尝试解决OPF收敛问题。
    """
    num_buses = base_ppc['bus'].shape[0]
    all_buses = list(range(num_buses))
    
    # 尝试不同的映射策略
    # 策略1: 均匀分布
    for i in range(num_buses):
        ppc = base_ppc.copy()
        
        # 将发电机均匀映射到母线上
        for j, gen in enumerate(ppc['gen']):
            bus_idx = (i + j) % num_buses
            ppc['gen'][j, GEN_BUS] = all_buses[bus_idx] + 1 # PyPower母线编号从1开始
            
        # 更新成本
        ppc['gencost'] = build_gencost(bids)
        
        results, status = run_opf_clearing(ppc, opf_options)
        if status in ["success", "success_dc"]:
            opf_type = "AC" if status == "success" else "DC"
            logging.info(f"动态映射成功 (策略: 均匀分布, 起始母线: {i+1}, 使用 {opf_type} OPF)")
            return results, status
    
    # 如果所有尝试都失败
    logging.warning("所有动态映射策略均未能使 OPF 收敛。")
    return None, "dynamic_mapping_fail"
    
def opf_joint_clearing_simple_backup(bids: List[Dict], total_demand: float, reserve_requirement: float, opf_options: Dict) -> Tuple[Dict, Dict]:
    """
    简化版OPF出清（原有实现的备份）
    """
    if not PYPOWER_AVAILABLE or not PYPOWER_OPF_AVAILABLE:
        logging.warning("PyPower OPF功能不可用，无法执行 OPF 出清。将回退到简化模式。")
        return simple_joint_clearing(bids, total_demand, reserve_requirement)

def opf_joint_clearing(bids: List[Dict], total_demand: float, reserve_requirement: float, opf_options: Dict = None) -> Tuple[Dict, Dict]:
    """
    主要的OPF出清函数，优先使用IEEE系统实现
    """
    # 如果PyPower可用且有IEEE系统支持，使用IEEE版本
    if PYPOWER_AVAILABLE and pp is not None:
        try:
            logging.info("使用IEEE系统进行OPF出清...")
            return opf_joint_clearing_ieee(bids, total_demand, reserve_requirement)
        except Exception as e:
            logging.warning(f"IEEE系统OPF失败: {e}，回退到简化版本")
            return opf_joint_clearing_simple_backup(bids, total_demand, reserve_requirement, opf_options or {})
    else:
        # 回退到简化版本
        logging.warning("PyPower或IEEE系统不可用，使用简化版本")
        return opf_joint_clearing_simple_backup(bids, total_demand, reserve_requirement, opf_options or {})

def opf_joint_clearing_simple_backup_impl(bids: List[Dict], total_demand: float, reserve_requirement: float, opf_options: Dict) -> Tuple[Dict, Dict]:
    """
    简化版OPF出清的具体实现
    """
    if not PYPOWER_AVAILABLE or not PYPOWER_OPF_AVAILABLE:
        logging.warning("PyPower OPF功能不可用，无法执行 OPF 出清。将回退到简化模式。")
        return simple_joint_clearing(bids, total_demand, reserve_requirement)

    # 1. 构建基础的 Pypower Case (ppc)
    ppc = build_base_ppc(len(bids), total_demand, reserve_requirement)

    # 2. 根据报价动态生成发电机和成本数据
    generators = []
    cleaned_bids = []
    for bid in bids:
        e_quant = bid['energy_bid'].get('quantity', 0)
        r_quant = bid['reserve_bid'].get('quantity', 0)
        max_cap = bid.get('max_capacity', 0)
        
        total_q = e_quant + r_quant
        if total_q > max_cap:
            logging.warning(f"OPF清算中发现智能体 {bid['agent_id']} 报价总量 {total_q:.2f} > 容量 {max_cap:.2f}，将按比例缩减")
            if total_q > 0:
                scale = max_cap / total_q
                bid['energy_bid']['quantity'] = e_quant * scale
                bid['reserve_bid']['quantity'] = r_quant * scale
        
        # 仅当最大容量大于0时才认为是有效发电机
        if max_cap > 0:
            gen = [
                1,              # bus index (will be dynamically mapped, start with bus 1)
                e_quant,        # P (active power) - initial guess
                0,              # Q (reactive power)
                1000, -1000,    # Qmax, Qmin
                1.0,            # Vg (voltage magnitude setpoint)
                100,            # mBase
                1,              # status (1 = in service)
                max_cap,        # Pmax
                0,              # Pmin
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 # placeholders
            ]
            generators.append(gen)
            cleaned_bids.append(bid)
    
    if not generators:
        logging.warning("没有有效的发电机报价，无法执行OPF，回退到简化模式。")
        return simple_joint_clearing(bids, total_demand, reserve_requirement)

    ppc['gen'] = np.array(generators)
    ppc['gencost'] = build_gencost(cleaned_bids)

    # 3. 运行 OPF (仅处理能源市场)
    results, status = run_opf_clearing(ppc, opf_options)
    
    # 4. 如果OPF失败，尝试动态映射
    if status not in ["success", "success_dc"]:
        logging.warning(f"初始OPF求解失败 ({status})，尝试动态映射策略...")
        results, status = dynamic_mapping_opf(ppc, cleaned_bids, opf_options)

    # 5. 如果仍然失败，回退到简化机制
    if status not in ["success", "success_dc"]:
        logging.warning(f"最优潮流计算失败或未成功 (已尝试动态映射)，将使用简化版出清机制替代。")
        return simple_joint_clearing(bids, total_demand, reserve_requirement)

    # 6. 解析OPF结果（能源市场）
    load_bus_idx = 0 
    energy_price = results['bus'][load_bus_idx, LAM_P]
    
    # 初始化智能体结果
    agent_results = {}
    for i, bid in enumerate(cleaned_bids):
        agent_id = bid['agent_id']
        cleared_energy = results['gen'][i, PG]
        
        agent_results[agent_id] = {
            'cleared_energy': cleared_energy,
            'cleared_reserve': 0,  # 将在备用出清中更新
            'total_cleared': cleared_energy,  # 将在备用出清后更新
            'remaining_capacity': bid['max_capacity'] - cleared_energy  # 剩余容量用于备用
        }

    # 7. 执行备用市场出清（使用简化算法）
    reserve_price = clear_reserve_market(cleaned_bids, agent_results, reserve_requirement)
    
    # 8. 更新最终结果
    for agent_id in agent_results:
        agent_results[agent_id]['total_cleared'] = (
            agent_results[agent_id]['cleared_energy'] + 
            agent_results[agent_id]['cleared_reserve']
        )
        # 移除临时字段
        agent_results[agent_id].pop('remaining_capacity', None)

    clearing_prices = {
        'energy_price': energy_price,
        'reserve_price': reserve_price
    }

    # 确定使用的OPF类型并记录日志
    opf_type = "DC" if results.get('opf_type') == 'DC' or status == "success_dc" else "AC"
    logging.info(f"{opf_type} OPF市场出清完成。能源价格: {clearing_prices['energy_price']:.2f}, 备用价格: {clearing_prices['reserve_price']:.2f}")

    return clearing_prices, agent_results

def opf_joint_clearing_ieee(bids: List[Dict], total_demand: float, reserve_requirement: float) -> Tuple[Dict, Dict]:
    """
    考虑网络约束的现货与备用联合市场出清算法 (使用 PyPower OPF 和动态映射)。

    基本逻辑:
    1. 根据智能体数量选择合适的IEEE测试系统。
    2. **动态映射**: 根据智能体能源报价成本和预定义的节点优先级，将智能体映射到发电机。
    3. 将智能体参数（容量、成本）应用到映射的发电机，调整负荷。
    4. 运行 PyPower AC OPF 求解能源市场出清。
    5. 若 OPF 成功：
        a. 提取能源出清价格 (节点边际电价 LMP)。
        b. 提取各智能体的能源出清量（根据映射关系）。
        c. 基于 OPF 后的剩余容量，按备用报价从低到高分配备用，确定备用出清量和价格。
    6. 若 OPF 失败，则回退到 `simple_joint_clearing`。

    Args:
        bids: 智能体报价列表 (同 `simple_joint_clearing`)。
        total_demand: 当前总能源需求 (MW)。
        reserve_requirement: 当前备用需求 (MW)。

    Returns:
        一个元组包含两个字典:
        1. clearing_prices: {'energy_price': float, 'reserve_price': float}
        2. agent_results: 键是 agent_id，值是出清结果字典
           {'cleared_energy': float, 'cleared_reserve': float, 'total_cleared': float}
    """
    logging.debug(f"开始考虑网络约束的市场出清 (动态映射)... 总需求: {total_demand:.2f}, 备用需求: {reserve_requirement:.2f}")

    # 检查PyPower是否可用
    if not PYPOWER_AVAILABLE:
        logging.warning("PyPower库未安装或导入失败，将使用简化版出清机制替代。")
        return simple_joint_clearing(bids, total_demand, reserve_requirement)

    if not bids:
        logging.warning("没有收到任何报价，无法出清。")
        return {'energy_price': 0, 'reserve_price': 0}, {}

    try:
        # --- 第一步：数据预处理和验证 ---
        bid_data = []
        for bid in bids:
            agent_id = bid['agent_id']
            e_price = bid['energy_bid'].get('price', 0)
            e_quant = bid['energy_bid'].get('quantity', 0)
            r_price = bid['reserve_bid'].get('price', 0)
            r_quant = bid['reserve_bid'].get('quantity', 0)
            max_cap = bid.get('max_capacity', 0)

            # 数据验证
            if e_price < 0 or r_price < 0:
                logging.warning(f"Agent {agent_id} 报价为负数，跳过。")
                continue
            if e_quant < 0 or r_quant < 0:
                logging.warning(f"Agent {agent_id} 报价量为负数，跳过。")
                continue
            if max_cap <= 0:
                logging.warning(f"Agent {agent_id} 容量 <= 0，跳过。")
                continue

            # 处理超容量报价
            total_bid_quant = e_quant + r_quant
            if total_bid_quant > max_cap * 1.001: # 允许极小的误差
                logging.warning(f"Agent {agent_id} 总报价量 {total_bid_quant:.2f} > 容量 {max_cap:.2f}. 按比例缩减.")
                if total_bid_quant > 0:
                    scale = max_cap / total_bid_quant
                    e_quant *= scale
                    r_quant *= scale
                else: # 如果总量为0，则无需缩减
                    e_quant = 0
                    r_quant = 0

            bid_data.append({
                 'agent_id': agent_id,
                 'energy_price': e_price,
                 'energy_quantity': e_quant,
                 'reserve_price': r_price,
                 'reserve_quantity': r_quant,
                 'max_capacity': max_cap,
                 'cleared_energy': 0.0, # 初始化出清量
                 'cleared_reserve': 0.0,
                 'available_for_reserve': max_cap # 初始可用于备用的容量等于最大容量
             })

        if not bid_data:
            logging.warning("所有报价均无效，无法出清。")
            return {'energy_price': 0, 'reserve_price': 0}, {}

        bids_df = pd.DataFrame(bid_data)
        bids_df = bids_df.replace([np.inf, -np.inf], np.nan).dropna() # 移除包含inf或nan的行

        if bids_df.empty:
            logging.warning("处理和过滤后无有效报价，无法出清。")
            return {'energy_price': 0, 'reserve_price': 0}, {}

        num_agents = len(bids_df)
        logging.info(f"有效智能体数量: {num_agents}")

        # --- 第二步：选择合适的IEEE测试系统 ---
        gen_counts = {
            "case9": pp.case9()['gen'].shape[0] if PYPOWER_AVAILABLE else 3,
            "case14": pp.case14()['gen'].shape[0] if PYPOWER_AVAILABLE else 5,
            "case30": pp.case30()['gen'].shape[0] if PYPOWER_AVAILABLE else 6,
            "case39": pp.case39()['gen'].shape[0] if PYPOWER_AVAILABLE else 10,
            "case57": pp.case57()['gen'].shape[0] if PYPOWER_AVAILABLE else 7,
            "case118": pp.case118()['gen'].shape[0] if PYPOWER_AVAILABLE else 54,
        }

        selected_case_func = None
        if num_agents <= gen_counts["case9"]:
            selected_case_func = pp.case9
            case_name = "IEEE Case 9"
            num_gens_in_case = gen_counts["case9"]
        elif num_agents <= gen_counts["case14"]:
            selected_case_func = pp.case14
            case_name = "IEEE Case 14"
            num_gens_in_case = gen_counts["case14"]
        elif num_agents <= gen_counts["case30"]:
            selected_case_func = pp.case30
            case_name = "IEEE Case 30"
            num_gens_in_case = gen_counts["case30"]
        elif num_agents <= gen_counts["case39"]:
            selected_case_func = pp.case39
            case_name = "IEEE Case 39"
            num_gens_in_case = gen_counts["case39"]
        elif num_agents <= gen_counts["case57"]:
            selected_case_func = pp.case57
            case_name = "IEEE Case 57"
            num_gens_in_case = gen_counts["case57"]
        elif num_agents <= gen_counts["case118"]:
             selected_case_func = pp.case118
             case_name = "IEEE Case 118"
             num_gens_in_case = gen_counts["case118"]
        else:
            selected_case_func = pp.case118
            case_name = "IEEE Case 118 (截断)"
            num_gens_in_case = gen_counts["case118"]
            # 截断将在映射后进行

        logging.info(f"选择的IEEE系统: {case_name} (发电机数量: {num_gens_in_case})")

        # 加载选定的IEEE测试系统
        case = selected_case_func()
        case_copy = case.copy()  # 创建副本以避免修改原始数据

        # --- 第三步：动态映射智能体到发电机 ---
        # 根据能源报价对智能体进行排序
        bids_df_sorted = bids_df.sort_values(by='energy_price').reset_index()

        # 执行映射：成本最低的智能体映射到优先级最高的发电机
        # 定义每个IEEE系统的发电机优先级（基于重要性，如Slack Bus优先）
        if case_name == "IEEE Case 9":
            priority_gen_indices = [0, 1, 2]  # 所有发电机
        elif case_name == "IEEE Case 14":
            priority_gen_indices = [0, 1, 2, 3, 4]  # 所有发电机
        elif case_name == "IEEE Case 30":
            priority_gen_indices = [0, 1, 2, 3, 4, 5]  # 所有发电机
        elif case_name == "IEEE Case 39":
            priority_gen_indices = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # 反向优先级
        elif case_name == "IEEE Case 57":
            priority_gen_indices = [0, 1, 2, 3, 4, 5, 6]  # 所有发电机
        elif "IEEE Case 118" in case_name:
            # IEEE 118系统的发电机优先级：Slack Bus (69, idx 18) 优先
            slack_bus_gen_idx = 18  # 对应Bus 69的发电机
            priority_gen_indices = [slack_bus_gen_idx] + [i for i in range(num_gens_in_case) if i != slack_bus_gen_idx]
        else:
            # 默认优先级
            priority_gen_indices = list(range(num_gens_in_case))

        # 如果智能体数量超过发电机数量，截断智能体列表
        if num_agents > num_gens_in_case:
            logging.warning(f"智能体数量 ({num_agents}) 超过发电机数量 ({num_gens_in_case})，将截断到前 {num_gens_in_case} 个智能体。")
            bids_df_sorted = bids_df_sorted.head(num_gens_in_case)
            num_agents = num_gens_in_case

        # 创建映射关系
        agent_to_gen_mapping = {}
        for i in range(num_agents):
            agent_id = bids_df_sorted.iloc[i]['agent_id']
            gen_idx = priority_gen_indices[i]
            agent_to_gen_mapping[agent_id] = gen_idx

        logging.info(f"智能体到发电机的映射关系: {agent_to_gen_mapping}")

        # --- 第四步：调整系统参数 ---
        # 调整负荷：将总需求按比例分配到各负荷节点
        original_total_load = np.sum(case_copy['bus'][:, PD])
        if original_total_load > 0:
            load_scale_factor = total_demand / original_total_load
            case_copy['bus'][:, PD] *= load_scale_factor
            logging.info(f"负荷调整：原始总负荷 {original_total_load:.2f} MW，调整后 {total_demand:.2f} MW，缩放因子 {load_scale_factor:.4f}")
        else:
            # 如果原始负荷为0，将总需求分配到第一个负荷节点
            case_copy['bus'][0, PD] = total_demand
            logging.warning(f"原始系统无负荷，将总需求 {total_demand:.2f} MW 分配到Bus 1")

        # 调整发电机参数：将智能体的容量和成本映射到对应的发电机
        for i, row in bids_df_sorted.iterrows():
            agent_id = row['agent_id']
            gen_idx = agent_to_gen_mapping[agent_id]

            # 更新发电机容量
            case_copy['gen'][gen_idx, PMAX] = row['max_capacity']
            case_copy['gen'][gen_idx, PMIN] = 0  # 最小出力设为0

            # 更新发电机初始出力（作为初值）
            initial_output = min(row['energy_quantity'], row['max_capacity'])
            case_copy['gen'][gen_idx, PG] = initial_output

        # 构建成本数据
        num_gens = case_copy['gen'].shape[0]
        gencost = np.zeros((num_gens, 7))

        for i in range(num_gens):
            gencost[i, MODEL] = 2  # 多项式成本模型
            gencost[i, NCOST] = 3  # 3个系数（二次函数）
            gencost[i, 4] = 0      # 二次项系数
            gencost[i, 5] = 50     # 一次项系数（默认边际成本）
            gencost[i, 6] = 0      # 常数项

        # 为映射的智能体设置实际成本
        for agent_id, gen_idx in agent_to_gen_mapping.items():
            agent_row = bids_df_sorted[bids_df_sorted['agent_id'] == agent_id].iloc[0]
            gencost[gen_idx, 5] = agent_row['energy_price']  # 边际成本

        case_copy['gencost'] = gencost

        # 调整系统约束（可选）
        # 电压约束
        case_copy['bus'][:, 12] = 0.85  # VMIN
        case_copy['bus'][:, 11] = 1.15  # VMAX

        # 线路容量（可选：增加容量以减少约束）
        if 'branch' in case_copy:
            case_copy['branch'][:, 5] *= 3  # 增加线路容量

        # 检查容量负荷比
        total_gen_capacity = np.sum(case_copy['gen'][:, PMAX])
        total_load = np.sum(case_copy['bus'][:, PD])
        capacity_load_ratio = total_gen_capacity / total_load if total_load > 0 else float('inf')
        logging.info(f"容量负荷比: {capacity_load_ratio:.2f} (总容量: {total_gen_capacity:.2f} MW, 总负荷: {total_load:.2f} MW)")

        if capacity_load_ratio < 1.1:
            logging.warning(f"容量负荷比过低 ({capacity_load_ratio:.2f})，可能导致OPF无解。")

        # --- 第五步：运行OPF ---
        start_time = time.time()

        # 定义 Pypower 求解器选项
        if PYPOWER_OPF_AVAILABLE:
            # 首先尝试 AC OPF
            ppopt = ppoption(
                VERBOSE=0,
                OUT_ALL=0,
                OPF_ALG=560, # PDIPM (AC OPF)
            )

            try:
                results = runopf(case_copy, ppopt)
                opf_solver_used = "AC-OPF (PDIPM)"

                # 检查收敛性
                if results['success']:
                    logging.info(f"AC OPF 求解成功，耗时 {time.time() - start_time:.3f} 秒")
                else:
                    logging.warning("AC OPF 求解失败，尝试 DC OPF...")
                    raise Exception("AC OPF failed")

            except Exception as e:
                logging.warning(f"AC OPF 失败: {e}，尝试 DC OPF...")

                # 尝试 DC OPF
                try:
                    from pypower.rundcopf import rundcopf
                    results = rundcopf(case_copy, ppopt)
                    opf_solver_used = "DC-OPF"

                    if results['success']:
                        logging.info(f"DC OPF 求解成功，耗时 {time.time() - start_time:.3f} 秒")
                        results['opf_type'] = 'DC'  # 标记为DC OPF结果
                    else:
                        logging.error("DC OPF 也失败了")
                        raise Exception("Both AC and DC OPF failed")

                except Exception as dc_e:
                    logging.error(f"DC OPF 也失败: {dc_e}")
                    raise Exception("Both AC and DC OPF failed")
        else:
            raise Exception("PyPower OPF not available")

        # --- 第六步：提取结果 ---
        if results['success']:
            # 提取能源出清价格 (负荷加权平均LMP)
            load_bus_indices = np.where(case_copy['bus'][:, PD] > 0)[0]
            if len(load_bus_indices) > 0:
                lmps = results['bus'][load_bus_indices, LAM_P]
                loads_at_buses = results['bus'][load_bus_indices, PD]
                total_load_at_buses = np.sum(loads_at_buses)
                if total_load_at_buses > 0:
                    energy_price = np.sum(lmps * loads_at_buses) / total_load_at_buses
                else:
                    energy_price = np.mean(lmps)
            else:
                # 如果没有负荷节点，使用所有节点的平均LMP
                energy_price = np.mean(results['bus'][:, LAM_P])

            logging.info(f"能源出清价格 (负荷加权平均LMP): {energy_price:.2f} 元/MWh")

            # 提取各智能体的能源出清量
            bids_df_sorted['cleared_energy'] = 0.0
            bids_df_sorted['cleared_reserve'] = 0.0
            bids_df_sorted['total_cleared'] = 0.0

            for agent_id, gen_idx in agent_to_gen_mapping.items():
                cleared_energy = results['gen'][gen_idx, PG]
                bids_df_sorted.loc[bids_df_sorted['agent_id'] == agent_id, 'cleared_energy'] = cleared_energy
                available_reserve = float(bids_df_sorted.loc[bids_df_sorted['agent_id'] == agent_id, 'max_capacity'].values[0] - cleared_energy)
                bids_df_sorted.loc[bids_df_sorted['agent_id'] == agent_id, 'available_for_reserve'] = available_reserve

            logging.debug(f"各智能体能源出清结果:\n{bids_df_sorted[['agent_id', 'cleared_energy', 'available_for_reserve']].to_string()}")

            # --- 第七步：备用市场出清 ---
            # 基于剩余容量进行备用出清
            reserve_candidates = bids_df_sorted[
                (bids_df_sorted['reserve_quantity'] > 0.001) &
                (bids_df_sorted['available_for_reserve'] > 0.001)
            ].copy()

            if not reserve_candidates.empty and reserve_requirement > 0:
                reserve_candidates = reserve_candidates.sort_values(by='reserve_price')
                reserve_cleared_so_far = 0.0
                reserve_price = 0.0

                logging.debug(f"开始备用出清，需求: {reserve_requirement:.2f} MW")
                logging.debug(f"备用候选智能体:\n{reserve_candidates[['agent_id', 'reserve_price', 'reserve_quantity', 'available_for_reserve']].to_string()}")

                for _, candidate in reserve_candidates.iterrows():
                    if reserve_cleared_so_far >= reserve_requirement:
                        break

                    agent_id = candidate['agent_id']
                    available_quantity = min(candidate['reserve_quantity'], candidate['available_for_reserve'])
                    remaining_need = reserve_requirement - reserve_cleared_so_far

                    # 本次出清量 = min(可用量, 剩余需求)
                    cleared_this_round = min(available_quantity, remaining_need)

                    if cleared_this_round > 0.001:
                        # 更新该智能体的备用出清量
                        bids_df_sorted.loc[bids_df_sorted['agent_id'] == agent_id, 'cleared_reserve'] += cleared_this_round
                        reserve_cleared_so_far += cleared_this_round
                        reserve_price = candidate['reserve_price']  # 边际价格

                        logging.debug(f"  智能体 {agent_id}: 备用价格 {candidate['reserve_price']:.2f}, "
                                     f"可用量 {available_quantity:.2f}, "
                                     f"本次出清 {cleared_this_round:.2f}, "
                                     f"累计出清 {reserve_cleared_so_far:.2f}")

                logging.info(f"备用出清完成，价格: {reserve_price:.2f} 元/MW，总出清量: {reserve_cleared_so_far:.2f} MW")
            else:
                reserve_price = 0.0
                logging.info("无备用需求或无可用备用容量，备用价格为0")

            # 更新总出清量
            bids_df_sorted['total_cleared'] = bids_df_sorted['cleared_energy'] + bids_df_sorted['cleared_reserve']

            # 整理最终结果
            clearing_prices = {
                'energy_price': energy_price,
                'reserve_price': reserve_price
            }

            agent_results = bids_df_sorted.set_index('agent_id')[['cleared_energy', 'cleared_reserve', 'total_cleared']].to_dict('index')

            # 记录电压和相角范围
            vm = results['bus'][:, VM]
            va = results['bus'][:, VA]
            logging.info(f"OPF ({opf_solver_used}) 结果电压幅值范围: {np.min(vm):.4f} - {np.max(vm):.4f} p.u.")
            logging.info(f"OPF ({opf_solver_used}) 结果相角范围: {np.min(va):.2f} - {np.max(va):.2f} 度")

            return clearing_prices, agent_results

        else:
            # --- OPF 失败，回退到简化版 ---
            logging.warning("第三步: AC-OPF 和 DC-OPF 均失败，将使用简化版出清机制替代。")
            return simple_joint_clearing(bids, total_demand, reserve_requirement)

    except NameError as ne:
        logging.error(f"PyPower 相关函数或变量未定义: {ne}。请确保 PyPower 已正确安装并导入。")
        logging.warning("由于 PyPower 错误，将使用简化版出清机制替代。")
        return simple_joint_clearing(bids, total_demand, reserve_requirement)
    except Exception as e:
        logging.error(f"准备 OPF 数据或执行过程中发生意外错误: {e}", exc_info=True)
        logging.warning("由于 OPF 准备或执行阶段异常，将使用简化版出清机制替代。")
        return simple_joint_clearing(bids, total_demand, reserve_requirement)

def clear_reserve_market(bids: List[Dict], agent_results: Dict, reserve_requirement: float) -> float:
    """
    使用简化算法出清备用市场。
    基于能源出清后的剩余容量进行备用出清。
    
    Args:
        bids: 智能体报价列表
        agent_results: 包含能源出清结果的字典（将被修改）
        reserve_requirement: 备用需求
        
    Returns:
        备用出清价格
    """
    if reserve_requirement <= 0:
        logging.info("备用需求为0，跳过备用市场出清")
        return 0.0
    
    # 构建备用候选列表
    reserve_candidates = []
    for bid in bids:
        agent_id = bid['agent_id']
        if agent_id not in agent_results:
            continue
            
        # 计算该智能体可用于备用的容量
        remaining_capacity = agent_results[agent_id]['remaining_capacity']
        reserve_bid_quantity = bid['reserve_bid'].get('quantity', 0)
        reserve_price = bid['reserve_bid'].get('price', 0)
        
        # 可供备用的容量 = min(备用报价量, 剩余容量)
        available_for_reserve = min(reserve_bid_quantity, remaining_capacity)
        
        if available_for_reserve > 0.001:  # 避免微小量
            reserve_candidates.append({
                'agent_id': agent_id,
                'reserve_price': reserve_price,
                'available_quantity': available_for_reserve,
                'bid_quantity': reserve_bid_quantity
            })
    
    if not reserve_candidates:
        logging.warning("没有可用的备用容量候选，备用价格设为0")
        return 0.0
    
    # 按价格排序（从低到高）
    reserve_candidates.sort(key=lambda x: x['reserve_price'])
    
    # 执行备用出清
    reserve_cleared_total = 0.0
    reserve_price = 0.0
    
    logging.debug(f"开始备用市场出清，需求: {reserve_requirement:.2f} MW")
    logging.debug(f"备用候选数量: {len(reserve_candidates)}")
    
    for candidate in reserve_candidates:
        if reserve_cleared_total >= reserve_requirement:
            break
            
        agent_id = candidate['agent_id']
        remaining_need = reserve_requirement - reserve_cleared_total
        
        # 本次出清量 = min(可用量, 剩余需求)
        cleared_this_round = min(candidate['available_quantity'], remaining_need)
        
        if cleared_this_round > 0.001:
            # 更新该智能体的备用出清量
            agent_results[agent_id]['cleared_reserve'] += cleared_this_round
            reserve_cleared_total += cleared_this_round
            reserve_price = candidate['reserve_price']  # 边际价格
            
            logging.debug(f"  智能体 {agent_id}: 备用价格 {candidate['reserve_price']:.2f}, "
                         f"可用量 {candidate['available_quantity']:.2f}, "
                         f"本次出清 {cleared_this_round:.2f}, "
                         f"累计出清 {reserve_cleared_total:.2f}")
    
    # 检查备用需求是否满足
    if reserve_cleared_total < reserve_requirement * 0.99:
        shortage = reserve_requirement - reserve_cleared_total
        logging.warning(f"备用需求未完全满足，短缺 {shortage:.2f} MW")
        
        # 设置短缺惩罚价格
        if reserve_candidates:
            # 使用最高报价作为短缺价格
            max_reserve_price = max(c['reserve_price'] for c in reserve_candidates)
            reserve_price = max(reserve_price, max_reserve_price * 1.5)  # 1.5倍惩罚
        else:
            reserve_price = 100.0  # 默认短缺价格
            
        logging.warning(f"设置备用短缺价格: {reserve_price:.2f}")
    
    logging.info(f"备用市场出清完成。出清价格: {reserve_price:.2f}, 出清量: {reserve_cleared_total:.2f}/{reserve_requirement:.2f}")
    
    return reserve_price

def build_gencost(bids: List[Dict]) -> np.ndarray:
    """
    根据智能体报价构建Pypower的成本矩阵。
    使用多项式成本函数而不是分段线性，因为分段线性在 PYPOWER 中格式复杂。
    对于二次成本函数（POLYNOMIAL），格式为：
    [MODEL, STARTUP, SHUTDOWN, NCOST, c(n-1), c(n-2), ..., c1, c0]
    其中 MODEL=2 表示多项式，NCOST 是系数个数
    """
    cost_list = []
    for bid in bids:
        price = bid['energy_bid'].get('price', 9999)
        quantity = bid.get('max_capacity', 100)
        
        # 使用线性成本函数 f(p) = price * p + 0
        # 这等价于二次函数 f(p) = 0*p^2 + price*p + 0
        row = [
            2,      # MODEL = 2 (多项式)
            0,      # Startup cost
            0,      # Shutdown cost
            3,      # NCOST = 3 (二次函数有3个系数: c2, c1, c0)
            0,      # c2 (二次项系数)
            price,  # c1 (一次项系数，即边际成本)
            0       # c0 (常数项)
        ]
        cost_list.append(row)
    
    return np.array(cost_list)

def build_base_ppc(num_generators: int, total_demand: float, reserve_requirement: float) -> Dict:
    """
    构建一个基础的Pypower Case文件。
    创建一个简单的网络拓扑：一个负荷母线和一个发电机母线。
    """
    # 母线数据 (bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin)
    # type 1=PQ bus (load), 2=PV bus (generation), 3=reference bus
    bus_data = np.array([
        [1, 3, total_demand, 0, 0, 0, 1, 1.0, 0, 100, 1, 1.1, 0.9], # Ref bus, also carries the load
        [2, 2, 0, 0, 0, 0, 1, 1.0, 0, 100, 1, 1.1, 0.9]             # PV bus for generation
    ])

    # 支路数据 (fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax)
    branch_data = np.array([
        [1, 2, 0.001, 0.01, 0.0, 9999, 9999, 9999, 0, 0, 1, -360, 360] # Low impedance line
    ])
    
    ppc = {
        'version': '2',
        'baseMVA': 100.0,
        'bus': bus_data,
        'gen': np.zeros((num_generators, 21)), # Placeholder for generator data
        'branch': branch_data,
        'gencost': np.zeros((num_generators, 7)) # 7列：[MODEL, STARTUP, SHUTDOWN, NCOST, c2, c1, c0]
    }
    return ppc

# --- 主函数/测试代码 (可选) ---
if __name__ == '__main__':
    # 添加测试用例来测试动态映射
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

    print("\n--- 测试用例: 4 个智能体 (Case 14)，测试动态映射 ---")
    bids_test = [
        # 成本顺序: C < A < D < B
        {'agent_id': 'Gen_A', 'energy_bid': {'price': 28.5, 'quantity': 155}, 'reserve_bid': {'price': 6, 'quantity': 25}, 'max_capacity': 180, 'min_output': 25},
        {'agent_id': 'Gen_B', 'energy_bid': {'price': 39.0, 'quantity': 80}, 'reserve_bid': {'price': 9, 'quantity': 40}, 'max_capacity': 120, 'min_output': 0},
        {'agent_id': 'Gen_C', 'energy_bid': {'price': 19.0, 'quantity': 60}, 'reserve_bid': {'price': 5, 'quantity': 15}, 'max_capacity': 75, 'min_output': 0}, # 成本最低
        {'agent_id': 'Gen_D', 'energy_bid': {'price': 34.0, 'quantity': 130}, 'reserve_bid': {'price': 8, 'quantity': 20}, 'max_capacity': 150, 'min_output': 10},
    ]
    total_demand_test = 200
    reserve_req_test = 20

    # 预期映射 (Case 14 优先级 [0, 1, 3, 2, 4]):
    # Gen_C (19.0) -> Gen Index 0 (Bus 1 - Slack)
    # Gen_A (28.5) -> Gen Index 1 (Bus 2)
    # Gen_D (34.0) -> Gen Index 3 (Bus 6)
    # Gen_B (39.0) -> Gen Index 2 (Bus 3)
    # Gen Index 4 (Bus 8) -> 下线

    prices_test, results_test = opf_joint_clearing(bids_test, total_demand_test, reserve_req_test, {})
    print("出清价格 (动态映射测试):", prices_test)
    print("智能体结果 (动态映射测试):")
    for agent, res in results_test.items():
        print(f"  {agent}: {res}")