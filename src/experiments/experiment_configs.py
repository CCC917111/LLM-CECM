#!/usr/bin/env python3
"""
实验配置模块

定义论文中三大核心实验的配置参数
"""

import math
from numbers import Real
from pathlib import Path
from typing import Dict, Any


EXPERIMENT_VARIANTS = {
    "exp1": ("RB", "RL", "IBR-ADF"),
    "exp2": ("without_adf", "with_adf"),
    "exp3": ("without_ibr_cr", "with_ibr_cr"),
}
DEFAULT_VARIANTS = {
    "exp1": "IBR-ADF", "exp2": "with_adf", "exp3": "with_ibr_cr",
}
DEFAULT_ROUNDS = {"exp1": 10, "exp2": 20, "exp3": 30}
SCENARIO_SERIES = (
    "gas_prices_by_round", "coal_outage_rates_by_round", "demand_increase_by_round",
    "solar_capacity_factors", "wind_capacity_factors",
)
REFERENCE_SERIES = ("ground_truth_rrp", "ground_truth_carbon_price")


def resolve_variant(experiment_id: str, variant: str = None) -> str:
    """Resolve an explicit experiment/variant pair without importing the simulator."""
    if experiment_id not in EXPERIMENT_VARIANTS:
        raise ValueError(f"Unknown experiment: {experiment_id!r}")
    resolved = DEFAULT_VARIANTS[experiment_id] if variant is None else variant
    if resolved not in EXPERIMENT_VARIANTS[experiment_id]:
        choices = ", ".join(EXPERIMENT_VARIANTS[experiment_id])
        raise ValueError(f"Invalid variant {resolved!r} for {experiment_id}; choose {choices}")
    return resolved


def validate_experiment_config(config: Dict[str, Any]) -> Dict[str, int]:
    """Require complete finite input series; never invent values for later rounds."""
    rounds = config.get("num_rounds")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("num_rounds must be a positive integer")

    coverage = {}

    def check_series(path, values):
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{path} must be a list or tuple of finite numbers")
        if len(values) < rounds:
            raise ValueError(
                f"{path} covers {len(values)} rounds, but num_rounds={rounds}; "
                "provide an explicit complete scenario or shorten the run"
            )
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) for value in values):
            raise ValueError(f"{path} must contain only finite numbers")
        coverage[path] = len(values)

    market = config.get("market", {})
    for key in ("demand_profile", "reserve_profile"):
        check_series(f"market.{key}", market.get(key))
    if coverage["market.demand_profile"] != coverage["market.reserve_profile"]:
        raise ValueError("market.demand_profile and market.reserve_profile must have equal lengths")

    dynamics = config.get("real_data_dynamics", {})
    if dynamics.get("enabled", False):
        for key in SCENARIO_SERIES:
            check_series(f"real_data_dynamics.{key}", dynamics.get(key))
    # Reference data are plotted even when operational dynamics are disabled.
    for key in REFERENCE_SERIES:
        if key in dynamics:
            check_series(f"real_data_dynamics.{key}", dynamics[key])
    return coverage


def build_experiment_config(experiment_id: str, variant: str = None,
                            **overrides) -> Dict[str, Any]:
    """Shared configuration contract for the CLI preview and actual execution."""
    variant = resolve_variant(experiment_id, variant)
    delta_perf = overrides.pop("delta_perf", None)
    if delta_perf is not None and (experiment_id != "exp2" or variant != "with_adf"):
        raise ValueError("delta_perf applies only to exp2 with variant with_adf")
    if experiment_id == "exp1":
        config = get_exp1_config(agent_type=variant)
    elif experiment_id == "exp2":
        kwargs = {} if delta_perf is None else {"delta_perf": delta_perf}
        config = get_exp2_config(enable_adf=variant == "with_adf", **kwargs)
    else:
        config = get_exp3_config(enable_ibr_cr=variant == "with_ibr_cr")
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(f"Unknown configuration overrides: {', '.join(sorted(unknown))}")
    config.update(overrides)
    validate_experiment_config(config)
    if "num_rounds" in overrides:
        # Explicit shortened / quick runs must not replace full-run outputs.
        logging_config = dict(config["logging"])
        for key in ("results_file_csv", "results_file_json", "log_file"):
            if key in logging_config:
                path = Path(logging_config[key])
                logging_config[key] = str(path.parent / f"rounds_{config['num_rounds']}" / path.name)
        logging_config["visualization_dir"] = str(
            Path(logging_config["results_file_csv"]).parent / "visualization"
        )
        config["logging"] = logging_config
    return config


def describe_experiment(experiment_id: str, variant: str = None, **overrides) -> Dict[str, Any]:
    """Return a JSON-ready offline plan, without creating agents or output files."""
    variant = resolve_variant(experiment_id, variant)
    config = build_experiment_config(experiment_id, variant, **overrides)
    description = {
        "experiment": experiment_id,
        "variant": variant,
        "effective_rounds": config["num_rounds"],
        "released_default_rounds": DEFAULT_ROUNDS[experiment_id],
        "series_coverage": validate_experiment_config(config),
        "reference_price_unit": config["real_data_dynamics"].get("ground_truth_rrp_unit"),
        "reference_price_source": config["real_data_dynamics"].get("ground_truth_rrp_source"),
        "output_directory": str(Path(config["logging"]["results_file_csv"]).parent),
        "model_calls_made": False,
    }
    if experiment_id == "exp3":
        description["paper_result_horizon"] = 72
        description["reproduction_gap"] = (
            "The released exp3 scenario covers 30 rounds; paper summaries report a "
            "72-round comparison and convergence at round 31. Original per-round "
            "records, the full 72-round scenario and the sub-region convergence "
            "calculation are not supplied here. This plan does not reproduce those results."
        )
    return description


def get_base_config() -> Dict[str, Any]:
    """获取基础配置（从simulation.config导入）"""
    from simulation.config import get_config
    config = get_config()
    # 论文实验中的RL基线为离线预训练模型，仿真过程中不做在线微调
    config.setdefault("online_rl", {})["enabled"] = False
    return config


# ---------------------------------------------------------------------------
# ADF 参数（对应论文 Eq.(9) 与 Table VI）
#
# 论文中的 δ_perf 表示"业绩相对历史均值下滑多少才触发深度决策"：
#   δ_perf=0.05 (Conservative) → 利润下滑超过5%就触发 → LLM调用最多
#   δ_perf=0.25 (Aggressive)   → 利润下滑超过25%才触发 → LLM调用最少
# 而 AdaptiveDecisionFrequency 内部比较的是 profit_ratio < profit_threshold
# （profit_ratio = 当前利润 / 历史均值），因此换算关系为：
#   profit_threshold = 1 - δ_perf
# 注意：键名必须与 agents/adf_module.py 中 default_config 的键名完全一致，
# 并且要放在 config["adf_config"] 中（由 experiment_runner 下发到每个智能体），
# 放进 llm_config 或写错键名都会被静默忽略。
# ---------------------------------------------------------------------------
ADF_BASE_DELTA_PERF = 0.15            # 论文 Table VI 的 Base 设定
ADF_BASE_VOLATILITY_THRESHOLD = 0.10  # 事件驱动阈值 δ_event
ADF_BASE_REVIEW_CYCLE = 5             # 周期驱动 T_cycle（轮）


# 认知架构配置（论文 III-B：信念系统 Eq.(7) 与双层记忆），合并进 llm_config
COGNITIVE_CONFIG = {
    "enable_llm_belief": True,          # 深度决策前由LLM做语义信念转移
    "enable_llm_memory_summary": True,  # 由LLM定期总结长期语义记忆
    "memory_summary_interval": 10,      # 长期记忆最短更新间隔（轮）
    "working_memory_k": 5,              # 短期工作记忆长度K
    "inject_cognitive_state": True,     # 把信念与记忆写入决策Prompt
}


def build_adf_config(delta_perf: float = ADF_BASE_DELTA_PERF,
                     volatility_threshold: float = ADF_BASE_VOLATILITY_THRESHOLD,
                     review_cycle: int = ADF_BASE_REVIEW_CYCLE) -> Dict[str, Any]:
    """构造可被 AdaptiveDecisionFrequency 直接识别的 adf_config。"""
    if not 0.0 < delta_perf < 1.0:
        raise ValueError(f"delta_perf 应在 (0, 1) 区间内，当前为 {delta_perf}")
    return {
        "profit_threshold": round(1.0 - float(delta_perf), 6),
        "price_volatility_threshold": float(volatility_threshold),
        "strategic_review_cycle": int(review_cycle),
        # 仅用于日志/结果追溯，ADF模块不参与计算
        "delta_perf": float(delta_perf),
    }


def get_exp1_config(agent_type: str = "IBR-ADF") -> Dict[str, Any]:
    """
    实验一：智能体行为模式验证
    
    对比三种智能体类型：
    - RB: Rule-Based (规则基础)
    - RL: Reinforcement Learning (强化学习)
    - IBR-ADF: LLM驱动的智能体（含IBR-CR和ADF机制）
    
    Args:
        agent_type: "RB" | "RL" | "IBR-ADF"
    
    Returns:
        实验一配置字典
    """
    resolve_variant("exp1", agent_type)
    config = get_base_config()
    
    # 实验名称
    config["simulation_name"] = f"Exp1_BehaviorValidation_{agent_type}"
    
    # 使用真实澳洲AEMO数据（10轮仿真）
    config["num_rounds"] = 10
    
    # 市场事件配置（用于验证策略性行为）
    config["market_events"] = [
        {
            "announce_time": 3,
            "effect_time": 6,
            "event_type": "fuel_price_shock",
            "affected_fuel": "Gas",
            "price_multiplier": 2.5,
            "description": "天然气价格冲击，验证燃气机组策略性抬价行为",
            "announcement": "🚨 紧急公告：国际天然气供应紧张，价格将在3轮后大幅上涨150%"
        },
        {
            "announce_time": 1,
            "effect_time": 7,
            "event_type": "carbon_price_shock",
            "affected_fuel": "Coal",
            "carbon_price_multiplier": 1.8,
            "description": "碳价格上涨，影响化石燃料发电成本",
            "announcement": "📈 公告：碳市场政策收紧，碳价格将上涨80%"
        }
    ]
    
    # 加载真实AEMO数据
    import json
    real_data_file = Path(__file__).resolve().parents[1] / "data_integration/real_aemo_data_config.json"
    try:
        real_aemo_config = json.loads(real_data_file.read_text(encoding="utf-8"))
        ground_truth_rrp = real_aemo_config["real_data_dynamics"]["ground_truth_rrp"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot load the bundled NSW1 observed price reference: {real_data_file}") from exc
    
    # 真实数据动态配置
    config["real_data_dynamics"] = {
        "enabled": True,
        "gas_prices_by_round": [10, 11, 12, 14, 16, 18, 20, 24, 23, 24],
        "coal_outage_rates_by_round": [0, 0, 0.05, 0.09, 0.11, 0.14, 0.20, 0.22, 0.12, 0.08],
        "demand_increase_by_round": [0, 20, 40, 60, 80, 90, 95, 100, 103, 103],
        "random_seed": 42,
        "solar_capacity_factors": [0.35, 0.40, 0.45, 0.50, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30],
        "wind_capacity_factors": [0.6, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "ground_truth_rrp": ground_truth_rrp,  # 使用真实AEMO数据
        "ground_truth_rrp_unit": "AUD/MWh",
        "ground_truth_rrp_source": "Observed AEMO NSW1 hourly RRP, 15-17 July 2024 (AEST)",
        "ground_truth_carbon_price": [144.0, 146.4, 148.8, 153.6, 158.4, 163.2, 168.0, 170.4, 172.8, 170.4],
    }
    
    # 根据智能体类型设置特定配置
    if agent_type == "RB":
        config["default_agent_type"] = "RB"
        config["agent_behavior_mode"] = "rule_based"
    elif agent_type == "RL":
        config["default_agent_type"] = "RL"
        config["agent_behavior_mode"] = "reinforcement_learning"
        config["rl_model_dir"] = "models/multi_agent_high_price"
    elif agent_type == "IBR-ADF":
        config["default_agent_type"] = "LLM"
        config["llm_client"] = config.get("llm_client", {})
        config["llm_client"]["offline"] = False
        config["llm_config"] = {
            "enable_adf": True,
            "enable_ibr_cr": True,
            "model": "gemma-3-27b-it",
            **COGNITIVE_CONFIG,
        }
        # IBR-ADF-Agent 统一使用论文的 Base ADF 参数
        config["adf_config"] = build_adf_config()
    
    # 结果保存路径
    config["logging"]["results_file_csv"] = f"results/exp1_behavior_validation/{agent_type.lower()}_agent/simulation_results.csv"
    config["logging"]["results_file_json"] = f"results/exp1_behavior_validation/{agent_type.lower()}_agent/simulation_results_behavior.json"
    config["logging"]["log_file"] = f"results/exp1_behavior_validation/{agent_type.lower()}_agent/simulation.log"
    
    return config


def get_exp2_config(enable_adf: bool = True, delta_perf: float = ADF_BASE_DELTA_PERF) -> Dict[str, Any]:
    """
    实验二：ADF架构效率与精度评估
    
    对比两种配置：
    - IBR-Agent: 仅启用IBR-CR机制（无ADF）
    - IBR-ADF-Agent: 同时启用IBR-CR和ADF机制
    
    Args:
        enable_adf: 是否启用ADF机制
        delta_perf: ADF业绩驱动阈值 δ_perf（论文 Table VI：0.05 / 0.15 / 0.25）
    
    Returns:
        实验二配置字典
    """
    config = get_base_config()
    
    agent_variant = "IBR-ADF" if enable_adf else "IBR"
    config["simulation_name"] = f"Exp2_ADF_Efficiency_{agent_variant}"
    
    # 较长的仿真轮次以评估效率
    config["num_rounds"] = 20
    
    # LLM智能体配置
    config["default_agent_type"] = "LLM"
    config["llm_client"] = config.get("llm_client", {})
    config["llm_client"]["offline"] = False
    
    config["llm_config"] = {
        "enable_adf": enable_adf,  # 关键变量
        "enable_ibr_cr": True,      # 两者都启用IBR-CR
        "model": "gemma-3-27b-it",
        **COGNITIVE_CONFIG,
    }
    # ADF触发参数（仅在enable_adf=True时生效），通过 adf_config 下发给每个智能体
    config["adf_config"] = build_adf_config(delta_perf=delta_perf)
    
    # 真实数据动态（使用与实验一相同的配置）
    config["real_data_dynamics"] = {
        "enabled": True,
        "gas_prices_by_round": [10, 11, 12, 14, 16, 18, 20, 24, 23, 24] * 2,  # 重复以匹配20轮
        "coal_outage_rates_by_round": [0, 0, 0.05, 0.09, 0.11, 0.14, 0.20, 0.22, 0.12, 0.08] * 2,
        "demand_increase_by_round": [0, 20, 40, 60, 80, 90, 95, 100, 103, 103] * 2,
        "random_seed": 42,
        "solar_capacity_factors": [0.35, 0.40, 0.45, 0.50, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30] * 2,
        "wind_capacity_factors": [0.6, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] * 2,
        "ground_truth_rrp": [85.3, 92.1, 88.5, 95.4, 105.2, 115.8, 135.7, 180.9, 210.4, 225.6] * 2,
        "ground_truth_rrp_unit": "CNY/MWh",
        "ground_truth_rrp_source": "Synthetic scenario reference; not observed NSW1 RRP",
        "ground_truth_carbon_price": [144.0, 146.4, 148.8, 153.6, 158.4, 163.2, 168.0, 170.4, 172.8, 170.4] * 2,
    }
    
    # 性能追踪配置
    config["tracking"] = {
        "track_llm_calls": True,
        "track_execution_time": True,
        "track_memory_usage": True
    }
    
    # 结果保存路径
    agent_dir = "ibr_adf_agent" if enable_adf else "ibr_agent"
    # 做 δ_perf 灵敏度分析时，非 Base 设定写入单独目录，避免覆盖主实验结果
    if enable_adf and abs(delta_perf - ADF_BASE_DELTA_PERF) > 1e-9:
        agent_dir = f"{agent_dir}_dperf{delta_perf:g}"
        config["simulation_name"] += f"_dperf{delta_perf:g}"
    config["logging"]["results_file_csv"] = f"results/exp2_adf_efficiency/{agent_dir}/simulation_results.csv"
    config["logging"]["results_file_json"] = f"results/exp2_adf_efficiency/{agent_dir}/simulation_results_behavior.json"
    config["logging"]["log_file"] = f"results/exp2_adf_efficiency/{agent_dir}/simulation.log"
    
    return config


def get_exp3_config(enable_ibr_cr: bool = True) -> Dict[str, Any]:
    """
    实验三：IBR-CR机制博弈有效性验证
    
    对比两种配置：
    - ADF-Agent: 仅启用ADF机制（无IBR-CR）
    - IBR-ADF-Agent: 同时启用IBR-CR和ADF机制
    
    Args:
        enable_ibr_cr: 是否启用IBR-CR机制
    
    Returns:
        实验三配置字典
    """
    config = get_base_config()
    
    agent_variant = "IBR-ADF" if enable_ibr_cr else "ADF"
    config["simulation_name"] = f"Exp3_IBR_CR_Effectiveness_{agent_variant}"
    
    # 中等轮次以观察策略收敛
    config["num_rounds"] = 30
    
    # LLM智能体配置
    config["default_agent_type"] = "LLM"
    config["llm_client"] = config.get("llm_client", {})
    config["llm_client"]["offline"] = False
    
    config["llm_config"] = {
        "enable_adf": True,         # 两者都启用ADF
        "enable_ibr_cr": enable_ibr_cr,  # 关键变量
        "model": "gemma-3-27b-it",
        # IBR-CR参数（仅在enable_ibr_cr=True时生效）
        "ibr_cr_depth": 2,  # 迭代深度k=2
        "ibr_cr_temperature": 0.7,
        **COGNITIVE_CONFIG,
    }
    # 两组都启用ADF，使用论文 Base 参数
    config["adf_config"] = build_adf_config()
    
    # 真实数据动态（扩展到30轮）
    base_gas = [10, 11, 12, 14, 16, 18, 20, 24, 23, 24]
    base_outage = [0, 0, 0.05, 0.09, 0.11, 0.14, 0.20, 0.22, 0.12, 0.08]
    base_demand = [0, 20, 40, 60, 80, 90, 95, 100, 103, 103]
    base_solar = [0.35, 0.40, 0.45, 0.50, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
    base_wind = [0.6, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    base_rrp = [85.3, 92.1, 88.5, 95.4, 105.2, 115.8, 135.7, 180.9, 210.4, 225.6]
    base_carbon = [144.0, 146.4, 148.8, 153.6, 158.4, 163.2, 168.0, 170.4, 172.8, 170.4]
    
    config["real_data_dynamics"] = {
        "enabled": True,
        "gas_prices_by_round": base_gas * 3,
        "coal_outage_rates_by_round": base_outage * 3,
        "demand_increase_by_round": base_demand * 3,
        "random_seed": 42,
        "solar_capacity_factors": base_solar * 3,
        "wind_capacity_factors": base_wind * 3,
        "ground_truth_rrp": base_rrp * 3,
        "ground_truth_rrp_unit": "CNY/MWh",
        "ground_truth_rrp_source": "Synthetic scenario reference; not observed NSW1 RRP",
        "ground_truth_carbon_price": base_carbon * 3,
    }
    
    # 策略收敛性追踪
    config["tracking"] = {
        "track_strategy_convergence": True,
        "track_profit_variance": True,
        "track_price_volatility": True
    }
    
    # 结果保存路径
    agent_dir = "ibr_adf_agent" if enable_ibr_cr else "adf_agent"
    config["logging"]["results_file_csv"] = f"results/exp3_ibr_cr_effectiveness/{agent_dir}/simulation_results.csv"
    config["logging"]["results_file_json"] = f"results/exp3_ibr_cr_effectiveness/{agent_dir}/simulation_results_behavior.json"
    config["logging"]["log_file"] = f"results/exp3_ibr_cr_effectiveness/{agent_dir}/simulation.log"
    
    return config


def get_experiment_config(experiment_id: str, **kwargs) -> Dict[str, Any]:
    """
    根据实验ID获取配置
    
    Args:
        experiment_id: 实验标识符 "exp1", "exp2", "exp3"
        **kwargs: 传递给具体实验配置函数的参数
    
    Returns:
        实验配置字典
    
    Raises:
        ValueError: 如果实验ID无效
    """
    exp_map = {
        "exp1": get_exp1_config,
        "exp2": get_exp2_config,
        "exp3": get_exp3_config
    }
    
    if experiment_id not in exp_map:
        raise ValueError(f"无效的实验ID: {experiment_id}. 有效值: {list(exp_map.keys())}")
    
    return exp_map[experiment_id](**kwargs)
