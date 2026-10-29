"""Zero electricity dispatch must not discard carbon settlement or feedback."""
from types import SimpleNamespace

from agents.llm_agent import LLMAgent
from market.coupled_market import CoupledMarket


def test_zero_capacity_participant_retains_carbon_settlement_and_memory(monkeypatch):
    actions = {
        "active": {
            "electricity": {"energy_bid": {"price": 60.0, "quantity": 10.0}},
            "carbon": {"buy_orders": [], "sell_orders": []},
        },
        "outage": {
            "electricity": {
                "energy_bid": {"price": 45.0, "quantity": 0.0},
                "reserve_bid": {"price": 0.0, "quantity": 0.0},
            },
            "carbon": {"buy_orders": [{"price": 50.0, "quantity": 2.0}], "sell_orders": []},
        },
    }
    private_info = {
        "active": {"max_capacity": 10.0, "marginal_cost_energy": 40.0, "emission_factor": 0.0},
        "outage": {"max_capacity": 0.0, "marginal_cost_energy": 40.0, "emission_factor": 0.8},
        "nonparticipant": {"max_capacity": 10.0, "marginal_cost_energy": 40.0},
    }

    class CarbonSettlement:
        current_price = 50.0

        def __init__(self):
            self.agent_quotas = {"active": 20.0, "outage": 10.0}

        def process_bids(self, bids):
            assert bids["outage"]["buy_orders"] == [{"price": 50.0, "quantity": 2.0}]
            self.agent_quotas["outage"] += 2.0
            return {
                "current_price": 50.0,
                "trades": [{"buyer": "outage", "seller": "market_maker", "quantity": 2.0, "price": 50.0}],
                "total_volume": 2.0,
            }

    def electricity_clearing(**kwargs):
        # Match OPF's omission of a unit whose outage capacity and offer are zero.
        assert {bid["agent_id"] for bid in kwargs["bids"]} == set(actions)
        return {"energy_price": 70.0, "reserve_price": 0.0}, {
            "active": {"cleared_energy": 10.0, "cleared_reserve": 0.0},
        }

    monkeypatch.setattr("market.coupled_market.opf_joint_clearing", electricity_clearing)
    carbon = CarbonSettlement()
    market = CoupledMarket(
        SimpleNamespace(demand_profile=[10.0], reserve_profile=[0.0]), carbon, {})

    market_results, outcomes = market.run_coupled_clearing(actions, 0, private_info)

    assert set(outcomes) == {"active", "outage"}
    outcome = outcomes["outage"]
    assert outcome["submitted_bid"] == actions["outage"]["electricity"]
    assert outcome["submitted_carbon_bid"] == actions["outage"]["carbon"]
    assert outcome["cleared_energy"] == outcome["cleared_reserve"] == 0.0
    assert outcome["carbon"]["cost"] == 100.0
    assert outcome["carbon"]["quota_change"] == 2.0
    assert outcome["carbon"]["quota"] == 12.0
    assert carbon.agent_quotas["outage"] == 12.0  # Zero dispatch consumes no quota.
    assert outcome["emissions"] == outcome["revenue"] == outcome["cost"] == 0.0

    agent = LLMAgent("outage", private_info["outage"], None, ["active"],
                     llm_config={"enable_adf": False})
    # The simulator feeds back electricity profit minus settled carbon cost.
    agent.update_state(0, market_results,
                       {**outcome, "profit": outcome["profit"] - outcome["carbon"]["cost"]})
    assert agent.last_profit == -100.0
    assert agent.memory_module.history_df["round"].tolist() == [0]
    record = agent.memory_module.history_df.iloc[0]
    assert record["energy_bid_quantity"] == 0.0
    assert record["carbon_buy_quantity"] == 2.0
    assert record["profit"] == -100.0
