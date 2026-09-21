# llm_market_sim/simulation/config.py
import logging
import random
# from tkinter import N
from typing import Dict

"""
存储仿真实验的配置参数。
可以根据不同的实验场景修改此文件。
"""
NUM_ROUNDS = 72  # 72轮仿真（72小时），IBR-CR实验设置
LLM_MODEL_NAME = "gemma-3-27b-it"

# 测试用配置
TEST_CONFIG = {
    "simulation_name": "LLM_JointMarket_Test",
    "num_rounds": NUM_ROUNDS,  # 仿真总轮数
    "log_level": logging.INFO, # 日志记录级别 (DEBUG, INFO, WARNING, ERROR)
    
    # --- 市场冲击事件配置 ---
    "market_events": [],

    # --- 智能体自动生成配置 ---
    "auto_generate_agents": True, # 设置为True时启用LLM自动生成智能体
    "agent_generation": {
        "num_agents": 50,  # 每次生成50个智能体
        "num_batches": 1,  # 仅一个批次
        "perturb_range": 0.4,  # 增加扰动范围以提高50个智能体的多样性
        "use_llm": False,  # 是否使用LLM生成参数（False则使用随机扰动）
        # 可选：提供模板智能体配置，如果不提供则使用默认模板
        "template_agent": {
            "agent_type": "RL",
            "config": {
                "marginal_cost_energy": 25.0,
                "marginal_cost_reserve": 5.0,
                "max_capacity": 100.0,
                "min_output": 0.0,
                "emission_factor": 0.8,  # 排放因子 (吨CO2/MWh)
            },
            "llm_config": {
                "model": "gemma-3-27b-it",
                "enable_adf": False,  # 禁用自适应决策频率机制
                "enable_ibr_cr": True,  # 启用IBR-CR机制（迭代信念修正与反事实推理）
            },
            "belief_config": {
                "avg_cost_estimation_mean": 30.0,
                "avg_cost_estimation_std": 10.0,
                "update_aggressiveness": 0.1
            },
            "memory_config": {
                "capacity": 100  # 增加到100轮，足够72轮仿真使用
            }
        }
    },

    # --- 市场环境配置 ---
    "market": {
        # 需求和备用曲线: 可以是列表，长度需等于 num_rounds
        "demand_profile": [300, 320, 350, 380, 400, 420, 450, 430, 380, 350],
        "reserve_profile": [0] * NUM_ROUNDS,
        "clearing_mechanism": "simple",
        "opf_options": {
            "OPF_ALG": 560,
        }
    },
    
    # --- 碳市场配置 ---
    "carbon_market": {
        "init_quota": 80.0,                    # 初始碳配额 (吨CO2)
        "init_carbon_price": 50.0,             # 初始碳价格 (元/吨CO2)
        "init_funds": 100000.0,                # 初始资金余额 (元)
        "min_carbon_price": 20.0,              # 最低碳价格 (元/吨CO2)
        "max_carbon_price": 120.0,             # 最高碳价格 (元/吨CO2)
        
        # 做市商机制 (Market Maker) - 参考文献: Li et al. (2019)
        "enable_market_maker": True,           # 启用做市商机制
        "market_maker_funds": 1000000.0,       # 做市商初始资金（增加）
        "spread_limit": 0.15,                  # 价差限制 (降低至15%)
        
        # 订单簿深度 - 参考文献: Zhang et al. (2020)
        "order_book_levels": 8,                # 订单簿层级数（增加）
        
        # 流动性激励 - 参考文献: Wang et al. (2021)
        "volume_bonus": 0.02,                  # 交易量奖励（增加至2%）
        "liquidity_bonus": 0.01,               # 流动性奖励（增加至1%）
        "volatility_threshold": 0.03,          # 波动率阈值（降低）
        
        # 简化的价格动态调整机制 - 参考文献: Feng et al. (2009)
        "enable_price_dynamics": True,         # 启用价格动态调整
    },

    # --- 智能体配置 ---
    "agents": [
        {
            "agent_id": "GenCo_A_LLM",
            "agent_type": "LLM", # 指定智能体类型为 LLM
            "config": { # 私有信息
                "marginal_cost_energy": 20.0,
                "marginal_cost_reserve": 4.0,
                "max_capacity": 100.0,
                "min_output": 0.0,
                "emission_factor": 0.7,  # 排放因子 (吨CO2/MWh)
            },
            "llm_config": { # LLM特定配置 (如果需要，例如模型名称)
                 "model": "gemma-3-27b-it", # 使用这个模型
                 "enable_adf": False,  # 禁用自适应决策频率机制
                 "enable_ibr_cr": True,  # 启用IBR-CR机制（迭代信念修正与反事实推理）
            },
            "belief_config": { # 信念模块配置
                "avg_cost_estimation_mean": 35.0, # 初始对对手成本的估计均值
                "avg_cost_estimation_std": 10.0,  # 初始标准差
                "update_aggressiveness": 0.15     # 信念更新速率
            },
            "memory_config": { # 记忆模块配置
                "capacity": 100 # 存储最近100轮记录，足够72轮仿真
            }
        },
        {
            "agent_id": "GenCo_B_LLM",
            "agent_type": "LLM",
            "config": {
                "marginal_cost_energy": 28.0,
                "marginal_cost_reserve": 6.0,
                "max_capacity": 120.0,
                "min_output": 0.0,
                "emission_factor": 0.9,  # 排放因子 (吨CO2/MWh)
            },
             "llm_config": {
                 "model": "gemma-3-27b-it",
                 "enable_adf": False,  # 禁用自适应决策频率机制
                 "enable_ibr_cr": True,  # 启用IBR-CR机制（迭代信念修正与反事实推理）
             },
             "belief_config": {
                "avg_cost_estimation_mean": 30.0,
                "avg_cost_estimation_std": 12.0,
                "update_aggressiveness": 0.1
             },
             "memory_config": {
                "capacity": 100  # 增加到100轮，足够72轮仿真使用
            }
        },
    ],

    # --- LLM 客户端配置 ---
    # (API密钥等已在 llm_client.py 中处理，这里可以放其他客户端参数)
    "llm_client": {
        "api_keys": [
        ],
        "proxy": {
        },
        "timeout": 120,
        "max_retries": 3,
        "random_seed": 42,  # 固定随机种子，确保结果可重现
    },

    # --- 数据记录配置 ---
    "logging": {
        "log_file": "results/test_log.txt",
        "results_file_csv": "results/test_results.csv",
        "visualization_dir": "results/visualization"
    }
}

# --- 正式仿真配置 ---
SIMULATION_CONFIG = {
    "simulation_name": "LLM_JointMarket_Sim_v1",
    "num_rounds": 20,  # 仿真总轮数
    "log_level": logging.INFO, # 日志记录级别 (DEBUG, INFO, WARNING, ERROR)
    
    # --- 智能体自动生成配置 ---
    "auto_generate_agents": False, # 设置为True时启用LLM自动生成智能体
    "agent_generation": {
        "num_agents": 8,  # 要生成的智能体数量
        "name_prefix": "GenCo_Auto",  # 智能体ID前缀
        "perturb_range": 0.2,  # 随机扰动范围（百分比）
        "use_llm": True,  # 是否使用LLM生成参数（False则使用随机扰动）
        # 可选：提供模板智能体配置，如果不提供则使用默认模板
        "template_agent": {
            "agent_type": "LLM",
            "config": {
                "marginal_cost_energy": 25.0,
                "marginal_cost_reserve": 5.0,
                "max_capacity": 100.0,
                "min_output": 0.0,
            },
            "llm_config": {
                "model": "gemma-3-27b-it",
            },
            "belief_config": {
                "avg_cost_estimation_mean": 30.0,
                "avg_cost_estimation_std": 10.0,
                "update_aggressiveness": 0.1
            },
            "memory_config": {
                "capacity": 50
            }
        }
    },

    # --- 市场环境配置 ---
    "market": {
        # 需求和备用曲线: 72轮波动负荷（适合10个智能体，总容量约1370MW）
        "demand_profile": [
            # 第1-24轮：基础负荷期（事件公告前）- 60-80%负荷率
            820, 810, 800, 795, 790, 800, 830, 860, 890, 910, 920, 915,  # 0-11: 夜间到上午
            910, 905, 900, 895, 890, 900, 920, 940, 950, 945, 930, 910,  # 12-23: 下午到晚间

            # 第25-48轮：事件公告期（t=24公告后，观察预期行为）
            900, 890, 880, 875, 870, 880, 910, 940, 970, 990, 1000, 995,  # 24-35: 公告后第1天
            990, 985, 980, 975, 970, 980, 1000, 1020, 1030, 1025, 1010, 990,  # 36-47: 公告后第2天

            # 第49-72轮：事件生效期（t=48生效后，观察实际影响）
            980, 970, 960, 955, 950, 960, 990, 1020, 1050, 1070, 1080, 1075,  # 48-59: 生效后第1天
            1070, 1065, 1060, 1055, 1050, 1060, 1080, 1100, 1110, 1105, 1090, 1070   # 60-71: 生效后第2天
        ],
        "reserve_profile": [0] * 20,

        "clearing_mechanism": "simple", # 可选 "simple" # 或 "milp" (如果实现)
    },

    # --- 智能体配置 ---
    "agents": [
        {
            "agent_id": "GenCo_A_LLM",
            "agent_type": "LLM", # 指定智能体类型为 LLM
            "config": { # 私有信息
                "marginal_cost_energy": 20.0,
                "marginal_cost_reserve": 4.0,
                "max_capacity": 100.0,
                "min_output": 0.0,
            },
            "llm_config": { # LLM特定配置 (如果需要，例如模型名称)
                 "model": "gemma-3-27b-it", # 使用这个模型
                 "enable_adf": False,  # 禁用自适应决策频率机制
                 "enable_ibr_cr": True,  # 启用IBR-CR机制
            },
            "belief_config": { # 信念模块配置
                "avg_cost_estimation_mean": 35.0, # 初始对对手成本的估计均值
                "avg_cost_estimation_std": 10.0,  # 初始标准差
                "update_aggressiveness": 0.15     # 信念更新速率
            },
            "memory_config": { # 记忆模块配置
                "capacity": 50 # 存储最近50轮记录
            }
        },
        {
            "agent_id": "GenCo_B_LLM",
            "agent_type": "LLM",
            "config": {
                "marginal_cost_energy": 28.0,
                "marginal_cost_reserve": 6.0,
                "max_capacity": 120.0,
                "min_output": 0.0,
            },
             "llm_config": {
                 "model": "gemma-3-27b-it",
                 "enable_adf": False,  # 禁用自适应决策频率机制
                 "enable_ibr_cr": True,  # 启用IBR-CR机制
             },
             "belief_config": {
                "avg_cost_estimation_mean": 30.0,
                "avg_cost_estimation_std": 12.0,
                "update_aggressiveness": 0.1
             },
             "memory_config": {
                "capacity": 50
            }
        },
        {
            "agent_id": "GenCo_C_LLM",
            "agent_type": "LLM",
            "config": {
                "marginal_cost_energy": 35.0,
                "marginal_cost_reserve": 7.0,
                "max_capacity": 80.0,
                "min_output": 0.0,
            },
             "llm_config": {
                 "model": "gemma-3-27b-it",
                 "enable_adf": False,  # 禁用自适应决策频率机制
                 "enable_ibr_cr": True,  # 启用IBR-CR机制
             },
            "belief_config": {
                "avg_cost_estimation_mean": 25.0,
                "avg_cost_estimation_std": 8.0,
                 "update_aggressiveness": 0.12
            },
             "memory_config": {
                "capacity": 50
            }
        },
        # --- 可以添加其他类型的智能体进行对比 ---
        # {
        #     "agent_id": "GenCo_D_RuleBased",
        #     "agent_type": "RuleBased", # 假设有一个基于规则的智能体实现
        #     "config": {
        #         "marginal_cost_energy": 22.0,
        #         "marginal_cost_reserve": 5.0,
        #         "max_capacity": 90.0,
        #         "min_output": 0.0,
        #         "bidding_rule": "cost_plus_markup", # 规则参数
        #         "markup_energy": 1.1,
        #         "markup_reserve": 1.05
        #     }
        # },
    ],

    # --- LLM 客户端配置 ---
    # (API密钥等已在 llm_client.py 中处理，这里可以放其他客户端参数)
    "llm_client": {
        "max_retries": 5,
        "request_timeout": 600 # 秒
        # 注意：模型名称在 agent 配置中单独指定，允许不同智能体用不同模型
    },

    # --- 数据记录配置 ---
    "logging": {
        "log_file": "results/simulation_log.txt", # 日志文件路径
        "results_file_csv": "results/simulation_results.csv", # CSV结果文件
        "log_llm_prompts": False # 是否在日志中记录完整的LLM Prompt (可能很长)
    }
}

# --- 辅助函数，用于从配置中获取参数 ---
def get_config():
    # 设置固定随机种子，确保智能体参数可重现
    import numpy as np
    import random
    random.seed(42)
    np.random.seed(42)  # 也固定numpy的seed
    
    num_agents_per_batch = 10  # 每个批次生成10个智能体
    num_batches = 5  # 分5个批次
    total_agents = num_agents_per_batch * num_batches
    agent_configs = []
    
    # 优化参数范围，为50个智能体提供更大的多样性
    base_energy_cost = 25.0  # 基准能源成本
    base_reserve_cost = 8.0  # 基准备用成本
    base_capacity = 120.0    # 基准容量
    
    # 定义技术类型（中文名、前缀key、角色描述）并初始化计数器
    tech_types = [
        ("燃煤火电", "coal", "一家以煤电为主的传统发电企业，运行稳定，偏保守。"),
        ("天然气发电", "gas", "一家中型燃气发电企业，灵活性高，策略敏捷。"),
        ("水力发电", "hydro", "一家水力发电企业，边际成本低，灵活性高。"),
        ("风力发电", "wind", "一家风电企业，边际成本极低，但出力波动。"),
        ("光伏", "solar", "一家光伏企业，边际成本极低，受天气影响。"),
        ("核电", "nuclear", "一家核电企业，稳定运行，成本结构特殊。"),
        ("生物质", "biomass", "一家生物质发电企业，规模中等。"),
        ("储能", "storage", "一家储能企业，主要在备用/调峰中获利。")
    ]
    tech_counters = {key: 0 for _, key, _ in tech_types}

    for i in range(total_agents):
        # 为50个智能体使用更大的参数范围，增加多样性
        energy_cost = max(15.0, min(40.0, random.normalvariate(base_energy_cost, 7.0)))
        reserve_cost = max(3.0, min(15.0, random.normalvariate(base_reserve_cost, 3.0)))
        capacity = max(60.0, min(200.0, random.normalvariate(base_capacity, 25.0)))

        # 轮换分配技术类型，并按技术前缀命名
        tech_cn, tech_key, role_desc = tech_types[i % len(tech_types)]
        tech_counters[tech_key] += 1
        agent_id = f"{tech_key}-{tech_counters[tech_key]}"

        # 更贴近现实的排放因子（吨CO2/MWh）映射
        ef_map = {
            'coal': lambda: random.uniform(0.85, 1.0),
            'gas': lambda: random.uniform(0.35, 0.55),
            'biomass': lambda: random.uniform(0.15, 0.35),
            'nuclear': lambda: random.uniform(0.02, 0.05),
            'hydro': lambda: random.uniform(0.0, 0.02),
            'wind': lambda: random.uniform(0.0, 0.01),
            'solar': lambda: random.uniform(0.0, 0.01),
            'storage': lambda: 0.0,
        }
        emission_factor = ef_map.get(tech_key, lambda: random.uniform(0.1, 0.3))()

        agent_configs.append({
            'agent_id': agent_id,
            'agent_type': 'RL',
            'config': {
                'marginal_cost_energy': energy_cost,
                'marginal_cost_reserve': reserve_cost,
                'max_capacity': capacity,
                'min_output': 0.0,
                'emission_factor': emission_factor,  # 更合理的排放因子
                'technology_type': tech_cn,  # 添加技术类型，供分类与可视化
            },
            'role_description': role_desc
        })

    # 提高市场需求以反映真实市场利用率 (60-70%装机容量)
    num_rounds = NUM_ROUNDS
    # 设置为中度紧张状态 (75-80%装机利用率)，更接近真实2022年Q2情况
    demand_profile = [int(random.uniform(7600, 8100)) for _ in range(num_rounds)]
    reserve_profile = [int(random.uniform(300, 600)) for _ in range(num_rounds)]  # 增加备用需求

    # Assemble final config: disable auto-generation and include explicit agents list (50 RL agents)

    config = {
        "simulation_name": "LLM_JointMarket_Test",
        "num_rounds": NUM_ROUNDS,
        "log_level": logging.INFO,
        "logging": {
            "log_file": "results/test_log.txt",
            "results_file_csv": "results/simulation_results.csv"
        },
        # IMPORTANT: use explicit agents list (LLM agents + RL agent)
        "auto_generate_agents": False,

        # 市场冲击事件配置
        "market_events": [
            {
                "announce_time": 24,
                "effect_time": 48,
                "event_type": "fuel_price_shock",
                "affected_fuel": "天然气",  # 匹配智能体的 technology_type: "天然气发电"
                "price_multiplier": 1.5,   # 上涨50%
                "description": "燃气价格冲击：t=48 起天然气边际成本上调50%",
                "announcement": "公告：天然气价格将在 t=48 上涨 50%"
            }
        ],
        # 保留生成逻辑的参数以供将来使用（此处不生效，因为 auto_generate_agents=False）
        "agent_generation": {
            "num_agents": num_agents_per_batch,
            "num_batches": num_batches,
            "name_prefix": "GenCo_LLM",
            "perturb_range": 0.4,
            "use_llm": True,
            "template_agent": {
                "agent_type": "LLM",
                "config": {
                    "marginal_cost_energy": 25.0,
                    "marginal_cost_reserve": 5.0,
                    "max_capacity": 100.0,
                    "min_output": 0.0,
                    "emission_factor": 0.8,
                },
                "llm_config": {
                    "model": "gemma-3-27b-it",
                    "enable_adf": False,
                    "enable_ibr_cr": True
                },
                "belief_config": {
                    "avg_cost_estimation_mean": 30.0,
                    "avg_cost_estimation_std": 10.0,
                    "update_aggressiveness": 0.1
                },
                "memory_config": {
                    "capacity": 100
                }
            }
        },
        # --- 在线RL微调配置 ---
        "online_rl": {
            "enabled": True,               # 启用在线微调
            "update_interval": 12,         # 每12轮微调一次
            "timesteps_per_update": 50,    # 每次仅50步，快速验证
            "save_path": "models/rl_online/rl_agent_1_online.zip",  # 保存路径
            "max_updates": 3,              # 最多进行3次在线更新
            "pretrain_timesteps": 50,      # 仿真开始前预训练步数（几十步）
            "pretrain_save_path": "models/rl_online/rl_pretrain.zip"  # 预训练模型保存路径
        },
        # 纯RL模式：使用50个RL智能体（无LLM），用于与50个LLM参与者对比
        "agents": agent_configs,
        "market": {
            "demand_profile": demand_profile,
            "reserve_profile": reserve_profile,
            "clearing_mechanism": "opf",
            "opf_options": {
                "OPF_ALG": 560,
            }
        },
        "carbon_market": {
            "init_quota": 80.0,                    # 初始碳配额 (吨CO2)
            "init_carbon_price": 50.0,             # 初始碳价格 (元/吨CO2)
            "init_funds": 100000.0,                # 初始资金余额 (元)
            "min_carbon_price": 20.0,              # 最低碳价格 (元/吨CO2)
            "max_carbon_price": 120.0,             # 最高碳价格 (元/吨CO2)
            
            # 做市商机制
            "enable_market_maker": True,           # 启用做市商机制
            "market_maker_funds": 1000000.0,       # 做市商初始资金
            "spread_limit": 0.15,                  # 价差限制 15%
            
            # 订单簿深度
            "order_book_levels": 8,                # 订单簿层级数
            
            # 流动性激励
            "volume_bonus": 0.02,                  # 交易量奖励 2%
            "liquidity_bonus": 0.01,               # 流动性奖励 1%
        },
        
        # --- LLM 客户端配置 ---
        "llm_client": {
            "max_retries": 5,
            "request_timeout": 600,  # 秒
            "random_seed": 42,       # 固定随机种子，确保结果可重现
            "offline": False         # 🔥 启用在线模式：真正调用LLM
        },
        
        # --- 数据记录配置 ---
        "logging": {
            "log_file": "results/test_log.txt",
            "results_file_csv": "results/simulation_results.csv",
            "visualization_dir": "results/visualization"
        }
    }
    return config