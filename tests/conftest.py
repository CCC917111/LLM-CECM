"""
Shared fixtures for the LLM-CECM test suite.

The tests run completely offline: a deterministic mock replaces the LLM backend,
so no API key or network access is required.
"""
import os
import re
import sys
import random
import threading
import collections
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class MockLLM:
    """Deterministic stand-in for LLMClient.chat() that recognises every prompt type."""

    TOKENS = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.calls = collections.Counter()
        self.prompts = collections.defaultdict(list)

    def chat(self, prompt, return_json=True, model=None, temperature=0.3, track_tokens=False):
        if "更新你对竞争对手和市场的信念" in prompt:          # belief revision (Eq. 7)
            kind = "belief"
            response = {
                "belief_summary": "Clearing prices stay above my bids; rivals appear to add a markup.",
                "competitor_cost_mean": 60.0,
                "competitor_cost_std": 8.0,
                "market_trend": "electricity price rising, carbon price stable",
            }
        elif "形成可以指导未来决策的长期记忆" in prompt:      # long-term semantic memory
            kind = "memory"
            response = "1) Moderate markups still clear in high-price rounds. 2) Sell surplus quotas."
        elif "模拟一个理性的竞争对手" in prompt:              # IBR-CR level-1 rival simulation
            kind = "rival"
            response = {
                "analysis": "Rivals will undercut slightly.",
                "expected_competitor_strategy": {
                    "electricity_bid": {"energy_bid": {"price": 70, "quantity": 50},
                                        "reserve_bid": {"price": 0, "quantity": 0}},
                    "carbon_bid": {"buy_quantity": 5, "buy_price": 50, "sell_quantity": 0, "sell_price": 0},
                },
                "market_impact": "slightly lower clearing price",
            }
        else:                                                   # bidding decision (IBR-CR L0 / L2)
            kind = "bid"
            with self._lock:
                markup = self._rng.uniform(1.05, 1.6)
            mc_match = re.search(r"能源成本: ([0-9.]+)", prompt)
            cap_match = re.search(r"总容量 ([0-9.]+)", prompt)
            mc = float(mc_match.group(1)) if mc_match else 50.0
            cap = float(cap_match.group(1)) if cap_match else 100.0
            response = {
                "reasoning": "mock decision",
                "electricity_bid": {
                    "energy_bid": {"price": round(mc * markup + 5, 2), "quantity": round(cap * 0.9, 2)},
                    "reserve_bid": {"price": 0, "quantity": 0},
                },
                "carbon_bid": {"buy_quantity": 5, "buy_price": 50, "sell_quantity": 3, "sell_price": 52},
            }
        with self._lock:
            self.calls[kind] += 1
            self.prompts[kind].append(prompt)
        return (response, dict(self.TOKENS)) if track_tokens else response


@pytest.fixture
def in_src_dir():
    """Experiments resolve data files relative to src/, so run each test from there."""
    previous = os.getcwd()
    os.chdir(SRC_DIR)
    try:
        yield SRC_DIR
    finally:
        os.chdir(previous)


@pytest.fixture
def mock_llm():
    return MockLLM()
