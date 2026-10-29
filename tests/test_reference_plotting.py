"""Regression checks for reference units and zero-based round alignment."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from utils.visualization import ResultVisualizer


ELECTRICITY_LABEL = "Scenario price reference (CNY/MWh)"
CARBON_LABEL = "Scenario carbon price reference"


@pytest.fixture
def saved_figures(monkeypatch):
    """Inspect real Matplotlib lines without generating large raster artifacts."""
    figures = []
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: figures.append(plt.gcf()))
    yield figures
    plt.close("all")


def _visualizer(tmp_path, rounds, prices=None, unit=None, carbon_prices=None):
    result_file = tmp_path / "results.csv"
    pd.DataFrame({
        "round": rounds,
        "energy_price": [70.0] * len(rounds),
        "carbon_market_price": [50.0] * len(rounds),
    }).to_csv(result_file, index=False)
    return ResultVisualizer(
        str(result_file),
        ground_truth_prices=prices,
        ground_truth_carbon_prices=carbon_prices,
        ground_truth_price_unit=unit,
    )


def _lines(figure):
    return {line.get_label(): line for axis in figure.axes for line in axis.lines}


@pytest.mark.parametrize("rounds,expected_rounds", [
    (list(range(10)), list(range(10))),
    ([7, 2, 7, 5], [7, 2, 5]),
])
def test_long_reference_aligns_by_round_id(tmp_path, saved_figures, rounds, expected_rounds):
    reference = [100.0 + i for i in range(72)]
    visualizer = _visualizer(tmp_path, rounds, prices=reference, unit="CNY/MWh")

    visualizer.visualize_bid_strategies()

    line = _lines(saved_figures[-1])[ELECTRICITY_LABEL]
    np.testing.assert_array_equal(line.get_xdata(), expected_rounds)
    np.testing.assert_array_equal(line.get_ydata(), [reference[i] for i in expected_rounds])
    assert saved_figures[-1].axes[0].get_ylabel() == "Price (CNY/MWh)"


@pytest.mark.parametrize("unit", [None, "AUD/MWh", "EUR/MWh"])
def test_reference_with_unknown_or_different_currency_is_skipped(
        tmp_path, saved_figures, caplog, unit):
    visualizer = _visualizer(tmp_path, [0, 1], prices=[80.0, 90.0], unit=unit)

    with caplog.at_level(logging.WARNING):
        visualizer.visualize_bid_strategies()

    lines = _lines(saved_figures[-1])
    assert ELECTRICITY_LABEL not in lines
    assert "Market Clearing Price" in lines
    assert "reference unit" in caplog.text
    assert "no currency conversion" in caplog.text


@pytest.mark.parametrize("rounds,reference", [
    ([-1, 0], [80.0, 90.0]),
    ([0, 0.5], [80.0, 90.0]),
    ([0, 2], [80.0, 90.0]),
    ([0, float("nan")], [80.0, 90.0]),
    ([0, 1], [80.0, float("nan")]),
])
def test_invalid_reference_alignment_warns_without_losing_simulation_plot(
        tmp_path, saved_figures, caplog, rounds, reference):
    visualizer = _visualizer(tmp_path, rounds, prices=reference, unit="CNY/MWh")

    with caplog.at_level(logging.WARNING):
        visualizer.visualize_bid_strategies()

    lines = _lines(saved_figures[-1])
    assert ELECTRICITY_LABEL not in lines
    assert "Market Clearing Price" in lines
    assert "Skipping electricity reference overlay" in caplog.text


@pytest.mark.parametrize("reference", [None, []])
def test_missing_optional_reference_still_plots_results(tmp_path, saved_figures, reference):
    visualizer = _visualizer(tmp_path, [0, 1], prices=reference)
    visualizer.visualize_bid_strategies()
    assert set(_lines(saved_figures[-1])) == {"Market Clearing Price"}


def test_carbon_reference_uses_round_ids_and_scenario_label(tmp_path, saved_figures):
    reference = [50.0 + i for i in range(72)]
    visualizer = _visualizer(tmp_path, [2, 5, 9], carbon_prices=reference)

    visualizer.visualize_carbon_market()

    lines = _lines(saved_figures[-1])
    np.testing.assert_array_equal(lines[CARBON_LABEL].get_xdata(), [2, 5, 9])
    np.testing.assert_array_equal(lines[CARBON_LABEL].get_ydata(), [52.0, 55.0, 59.0])
    assert not any("Ground Truth" in label or "ACCU" in label for label in lines)
