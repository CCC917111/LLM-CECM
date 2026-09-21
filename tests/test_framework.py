"""
Offline tests for the LLM-CECM framework.

  * configuration plumbing of the three paper experiments (ADF / IBR-CR switches, delta_perf)
  * the cognitive modules (belief revision, dual-layer memory) and their failure fallbacks
  * short end-to-end coupled electricity-carbon simulations with a mock LLM
"""
import csv
import logging
import types

import pytest

from experiments.experiment_configs import get_experiment_config
from experiments.experiment_runner import ExperimentRunner
from simulation.simulator import MarketSimulator
from agents.llm_agent import LLMAgent
from agents.belief_module import BeliefModule
from agents.memory_module import MemoryModule


def _build_llm_agents(config):
    """Replicates ExperimentRunner + MarketSimulator agent construction without writing files."""
    ExperimentRunner.prepare_agents(types.SimpleNamespace(config=config))
    holder = types.SimpleNamespace(config=config, llm_client=object(), agents={})
    MarketSimulator._init_agents(holder)
    return holder.agents


def _adf_params(agent):
    cfg = agent.adf_module.config
    return cfg["profit_threshold"], cfg["price_volatility_threshold"], cfg["strategic_review_cycle"]


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("experiment,kwargs,adf,ibr", [
    ("exp1", {"agent_type": "IBR-ADF"}, True, True),
    ("exp2", {"enable_adf": True}, True, True),
    ("exp2", {"enable_adf": False}, False, True),
    ("exp3", {"enable_ibr_cr": True}, True, True),
    ("exp3", {"enable_ibr_cr": False}, True, False),
])
def test_experiment_switches(in_src_dir, experiment, kwargs, adf, ibr):
    agents = _build_llm_agents(get_experiment_config(experiment, **kwargs))
    assert len(agents) == 50
    for agent in agents.values():
        assert agent.enable_adf is adf
        assert agent.enable_ibr_cr is ibr
        if adf:
            assert _adf_params(agent) == (0.85, 0.10, 5)   # delta_perf = 0.15 (paper base setting)


@pytest.mark.parametrize("delta_perf,threshold", [(0.05, 0.95), (0.15, 0.85), (0.25, 0.75)])
def test_delta_perf_sensitivity(in_src_dir, delta_perf, threshold):
    agents = _build_llm_agents(get_experiment_config("exp2", enable_adf=True, delta_perf=delta_perf))
    assert {_adf_params(a)[0] for a in agents.values()} == {threshold}


# ---------------------------------------------------------------------------
# Cognitive modules
# ---------------------------------------------------------------------------
def test_belief_revision_updates_state(mock_llm):
    belief = BeliefModule("A", ["B", "C"], {"avg_cost_estimation_mean": 30, "avg_cost_estimation_std": 10},
                          llm_client=mock_llm, enable_llm_update=True)
    belief.update_beliefs(0, {"energy_price": 80, "carbon_market": {"current_price": 50}},
                          {"energy_bid": {"price": 60, "quantity": 10}, "cleared_energy": 10, "profit": 200})
    _, calls = belief.revise_with_llm(1, {"electricity_market": {}}, {"marginal_cost_energy": 40})
    state = belief.get_belief_state()
    assert calls == 1 and belief.llm_revision_count == 1
    assert state["competitor_cost_mean"] == 60.0
    assert "rising" in state["market_trend"]
    assert belief.pending_observations == []


class _FailingClient:
    def __init__(self, response=None, raise_error=False):
        self.response, self.raise_error = response, raise_error

    def chat(self, *args, **kwargs):
        if self.raise_error:
            raise RuntimeError("network down")
        return self.response, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.mark.parametrize("client", [
    _FailingClient(None), _FailingClient({"text": "not json"}),
    _FailingClient({"default_response": True, "error": "offline"}), _FailingClient(raise_error=True),
])
def test_cognitive_modules_fall_back_on_llm_failure(client):
    belief = BeliefModule("A", ["B"], {}, llm_client=client, enable_llm_update=True)
    belief.update_beliefs(0, {"energy_price": 80}, {"energy_bid": {"price": 60, "quantity": 10}, "cleared_energy": 10})
    before = belief.get_belief_state()
    belief.revise_with_llm(1, {}, {"marginal_cost_energy": 40})
    assert belief.get_belief_state() == before

    memory = MemoryModule("A", 100, llm_client=client, enable_llm_summary=True, summary_interval=1)
    for r in range(4):
        memory.add_record(r, {"energy_bid": {"price": 60, "quantity": 10}},
                          {"cleared_energy": 10, "market_energy_price": 80, "profit": 200}, {})
    memory.update_semantic_memory_with_llm(4)
    assert memory.llm_summary_count == 0


def test_working_memory_contains_strategy_price_profit():
    memory = MemoryModule("A", 100)
    memory.add_record(0, {"energy_bid": {"price": 61.5, "quantity": 90}},
                      {"cleared_energy": 90, "market_energy_price": 80.2, "market_carbon_price": 51.3,
                       "carbon_buy_quantity": 5, "carbon_sell_quantity": 0, "profit": 1683.0}, {})
    text = memory.get_working_memory_text(5)
    for token in ("61.50", "80.20", "51.30", "1683.00"):
        assert token in text


# ---------------------------------------------------------------------------
# End-to-end coupled simulation with a mock LLM
# ---------------------------------------------------------------------------
def _run_mock_simulation(tmp_path, mock_llm, enable_adf, rounds):
    config = get_experiment_config("exp2", enable_adf=enable_adf)
    config["num_rounds"] = rounds
    config["llm_config"]["memory_summary_interval"] = 2
    config["llm_client"]["offline"] = True
    config["logging"]["results_file_csv"] = str(tmp_path / "results.csv")
    ExperimentRunner.prepare_agents(types.SimpleNamespace(config=config))
    simulator = MarketSimulator(config)
    simulator.llm_client = mock_llm
    for agent in simulator.agents.values():
        if isinstance(agent, LLMAgent):
            agent.llm_client = mock_llm
            agent.belief_module.llm_client = mock_llm
            agent.memory_module.llm_client = mock_llm
    simulator.run_coupled()
    return simulator


@pytest.mark.parametrize("enable_adf,rounds", [(True, 6), (False, 4)])
def test_coupled_simulation_end_to_end(in_src_dir, tmp_path, mock_llm, caplog, enable_adf, rounds):
    caplog.set_level(logging.ERROR)
    simulator = _run_mock_simulation(tmp_path, mock_llm, enable_adf, rounds)
    agents = [a for a in simulator.agents.values() if isinstance(a, LLMAgent)]

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert all(len(a.memory_module.history_df) == rounds for a in agents)       # feedback every round
    assert all(a.last_profit is not None for a in agents)
    assert all(a.belief_module.llm_revision_count >= 1 for a in agents)        # Eq. (7) semantic revision
    assert all(a.memory_module.llm_summary_count >= 1 for a in agents)         # long-term memory
    assert any("你的认知状态" in p for p in mock_llm.prompts["bid"])             # beliefs/memory in prompt
    if enable_adf:
        assert any(a.adf_module.heuristic_decision_count > 0 for a in agents)   # Fast Mode used
    else:
        assert mock_llm.calls["rival"] == len(agents) * rounds                  # Deep Mode every round

    with open(tmp_path / "results.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == rounds
    total_tokens = sum(float(v or 0) for row in rows for k, v in row.items() if k.endswith("_total_tokens"))
    assert total_tokens == MockLLM_TOKENS * sum(mock_llm.calls.values())


MockLLM_TOKENS = 120


def test_rule_based_baseline_runs(in_src_dir, tmp_path):
    config = get_experiment_config("exp1", agent_type="RB")
    config["num_rounds"] = 2
    config["llm_client"]["offline"] = True
    config["logging"]["results_file_csv"] = str(tmp_path / "rb.csv")
    agents = ExperimentRunner.prepare_agents(types.SimpleNamespace(config=config))
    simulator = MarketSimulator(config, external_agents=agents)
    simulator.run_coupled()
    with open(tmp_path / "rb.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2
