"""Configuration previews must obey the same contract as real experiments."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.experiment_configs import (  # noqa: E402
    DEFAULT_ROUNDS, EXPERIMENT_VARIANTS, REFERENCE_SERIES, SCENARIO_SERIES,
    build_experiment_config, describe_experiment, validate_experiment_config,
)


class ExperimentContractTests(unittest.TestCase):
    def test_released_defaults_and_all_variants(self):
        for experiment, variants in EXPERIMENT_VARIANTS.items():
            for variant in variants:
                with self.subTest(experiment=experiment, variant=variant):
                    config = build_experiment_config(experiment, variant)
                    self.assertEqual(config["num_rounds"], DEFAULT_ROUNDS[experiment])
                    self.assertTrue(validate_experiment_config(config))
                    if experiment == "exp1":
                        self.assertEqual(config["default_agent_type"],
                                         "LLM" if variant == "IBR-ADF" else variant)
                    elif experiment == "exp2":
                        self.assertEqual(config["llm_config"]["enable_adf"], variant == "with_adf")
                    else:
                        self.assertEqual(config["llm_config"]["enable_ibr_cr"], variant == "with_ibr_cr")

    def test_invalid_variant_never_selects_a_default(self):
        for experiment in EXPERIMENT_VARIANTS:
            for variant in ("typo", "", "RB" if experiment != "exp1" else "with_adf"):
                with self.subTest(experiment=experiment, variant=variant):
                    with self.assertRaisesRegex(ValueError, "Invalid variant"):
                        build_experiment_config(experiment, variant)
        with self.assertRaisesRegex(ValueError, "Unknown experiment"):
            build_experiment_config("all")

    def test_rounds_must_be_positive_integers(self):
        for rounds in (0, -1, 1.5, True, "3", None):
            with self.subTest(rounds=rounds):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    build_experiment_config("exp3", num_rounds=rounds)

    def test_shortening_does_not_change_the_source_scenario(self):
        config = build_experiment_config("exp3", num_rounds=3)
        self.assertEqual(config["num_rounds"], 3)
        self.assertEqual(len(config["real_data_dynamics"]["gas_prices_by_round"]), 30)
        self.assertEqual(build_experiment_config("exp3")["num_rounds"], 30)

    def test_explicit_rounds_isolate_every_output_path(self):
        for experiment in EXPERIMENT_VARIANTS:
            with self.subTest(experiment=experiment):
                baseline = build_experiment_config(experiment)
                shortened = build_experiment_config(experiment, num_rounds=3)
                for key in ("results_file_csv", "results_file_json", "log_file"):
                    old_path = Path(baseline["logging"][key])
                    expected = old_path.parent / "rounds_3" / old_path.name
                    self.assertEqual(Path(shortened["logging"][key]), expected)
                    self.assertEqual(Path(build_experiment_config(experiment)["logging"][key]), old_path)
                self.assertEqual(Path(shortened["logging"]["visualization_dir"]),
                                 Path(shortened["logging"]["results_file_csv"]).parent / "visualization")
        # An explicit request for the default horizon is also kept separate.
        explicit = build_experiment_config("exp3", num_rounds=30)
        self.assertEqual(Path(explicit["logging"]["log_file"]).parent.name, "rounds_30")

    def test_each_input_series_must_cover_the_requested_horizon(self):
        config = build_experiment_config("exp3")
        for section, keys in (("market", ("demand_profile", "reserve_profile")),
                              ("real_data_dynamics", SCENARIO_SERIES + REFERENCE_SERIES)):
            for key in keys:
                with self.subTest(section=section, key=key):
                    short = copy.deepcopy(config)
                    short[section][key] = short[section][key][:29]
                    with self.assertRaisesRegex(ValueError, f"{section}.{key} covers 29 rounds"):
                        validate_experiment_config(short)
        with self.assertRaisesRegex(ValueError, "covers 30 rounds, but num_rounds=72"):
            build_experiment_config("exp3", num_rounds=72)

    def test_complete_explicit_extension_is_structurally_valid(self):
        config = build_experiment_config("exp3")
        dynamics = copy.deepcopy(config["real_data_dynamics"])
        # An explicitly supplied test scenario is distinct from the paper scenario.
        for key in SCENARIO_SERIES + REFERENCE_SERIES:
            dynamics[key] = [1.0] * 72
        extended = build_experiment_config("exp3", num_rounds=72, real_data_dynamics=dynamics)
        self.assertEqual(extended["num_rounds"], 72)

    def test_malformed_series_and_unknown_overrides_fail(self):
        config = build_experiment_config("exp3")
        for invalid in ("1" * 30, [float("nan")] * 30, [True] * 30):
            broken = copy.deepcopy(config)
            broken["real_data_dynamics"]["wind_capacity_factors"] = invalid
            with self.assertRaisesRegex(ValueError, "finite numbers"):
                validate_experiment_config(broken)
        with self.assertRaisesRegex(ValueError, "Unknown configuration overrides"):
            build_experiment_config("exp3", num_round=3)

    def test_delta_perf_is_never_silently_ignored(self):
        for experiment, variant in (("exp1", "RB"), ("exp3", None), ("exp2", "without_adf")):
            with self.subTest(experiment=experiment, variant=variant):
                with self.assertRaisesRegex(ValueError, "only to exp2"):
                    build_experiment_config(experiment, variant, delta_perf=0.05)
        config = build_experiment_config("exp2", "with_adf", delta_perf=0.05)
        self.assertEqual(config["adf_config"]["profit_threshold"], 0.95)

    def test_reference_currency_and_source_are_explicit(self):
        observed = build_experiment_config("exp1")["real_data_dynamics"]
        self.assertEqual(observed["ground_truth_rrp_unit"], "AUD/MWh")
        self.assertIn("Observed", observed["ground_truth_rrp_source"])
        for experiment in ("exp2", "exp3"):
            synthetic = build_experiment_config(experiment)["real_data_dynamics"]
            self.assertEqual(synthetic["ground_truth_rrp_unit"], "CNY/MWh")
            self.assertIn("Synthetic", synthetic["ground_truth_rrp_source"])

    def test_exp3_preview_discloses_the_reproduction_gap(self):
        preview = describe_experiment("exp3")
        self.assertEqual(preview["effective_rounds"], 30)
        self.assertEqual(preview["paper_result_horizon"], 72)
        self.assertIn("does not reproduce", preview["reproduction_gap"])

    def test_all_quick_test_is_forwarded_to_execution(self):
        import run_experiments
        fake = types.ModuleType("experiments.experiment_runner")
        fake.run_all_experiments = Mock(return_value=[])
        fake.run_experiment = Mock()
        with patch.dict(sys.modules, {"experiments.experiment_runner": fake}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_experiments.main(["--experiment", "all", "--quick-test"]), 0)
        fake.run_all_experiments.assert_called_once_with(num_rounds=3)


class OfflineCliTests(unittest.TestCase):
    # Fail immediately if a preview tries to import a runtime dependency or runner.
    BOOTSTRAP = """
import importlib.abc, runpy, sys
class NoRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('numpy', 'torch', 'requests', 'pypower', 'matplotlib', 'stable_baselines3',
                   'simulation.simulator', 'experiments.experiment_runner')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Runtime import during preview: ' + fullname)
sys.meta_path.insert(0, NoRuntimeImports())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""

    def run_cli(self, *args):
        env = dict(os.environ)
        env.pop("GOOGLE_API_KEY", None)
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-S", "-B", "-c", self.BOOTSTRAP, str(SRC / "run_experiments.py"), *args],
                cwd=temporary, env=env, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(list(Path(temporary).iterdir()), [], "Preview created output files")
        return result

    def test_dry_run_works_without_api_key_or_runtime_imports(self):
        result = self.run_cli("--experiment", "exp1", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        preview = json.loads(result.stdout)
        self.assertEqual(preview["effective_rounds"], 10)
        self.assertEqual(preview["series_coverage"]["real_data_dynamics.ground_truth_rrp"], 72)

    def test_all_preview_contains_every_variant_and_applies_shortening(self):
        result = self.run_cli("--experiment", "all", "--quick-test", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        plans = json.loads(result.stdout)["experiments"]
        self.assertEqual(len(plans), 7)
        self.assertEqual({plan["effective_rounds"] for plan in plans}, {3})
        self.assertTrue(all(Path(plan["output_directory"]).name == "rounds_3" for plan in plans))
        self.assertEqual({(p["experiment"], p["variant"]) for p in plans},
                         {(e, v) for e, variants in EXPERIMENT_VARIANTS.items() for v in variants})

    def test_exp3_dry_run_works_with_site_packages_disabled(self):
        result = self.run_cli("--experiment", "exp3", "--rounds", "3", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        preview = json.loads(result.stdout)
        self.assertEqual(preview["effective_rounds"], 3)
        self.assertEqual(Path(preview["output_directory"]).name, "rounds_3")
        self.assertEqual(preview["paper_result_horizon"], 72)

    def test_invalid_options_fail_before_runtime_import(self):
        cases = (
            ("--experiment", "all", "--variant", "RB"),
            ("--experiment", "all", "--delta-perf", "0.05"),
            ("--experiment", "exp1", "--delta-perf", "0.05"),
            ("--experiment", "exp3", "--variant", "typo"),
            ("--experiment", "exp3", "--rounds", "72"),
            ("--experiment", "exp3", "--rounds", "0"),
            ("--experiment", "exp3", "--rounds", "2", "--quick-test"),
        )
        for args in cases:
            with self.subTest(args=args):
                result = self.run_cli(*args, "--dry-run")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("Runtime import", result.stderr)

    def test_runner_uses_the_contract_before_constructing_or_writing(self):
        # Stub the simulator dependencies to exercise runner plumbing without
        # requiring torch, an API key or a numerical power-system installation.
        script = r'''
import importlib, pathlib, sys, types
sys.path.insert(0, sys.argv[1])
for name, symbol in (
    ('simulation.simulator', 'MarketSimulator'),
    ('utils.visualization', 'ResultVisualizer'),
    ('agents.rl_agent_enhanced', 'EnhancedRLAgent'),
    ('agents.zi_agent', 'ZIAgent'),
):
    module = types.ModuleType(name)
    setattr(module, symbol, object)
    sys.modules[name] = module
runner = importlib.import_module('experiments.experiment_runner')
from experiments.experiment_configs import build_experiment_config
original_runner = runner.ExperimentRunner
bad = build_experiment_config('exp3')
bad['num_rounds'] = 72
bad['logging'] = {'results_file_csv': 'must-not-exist/results.csv'}
try:
    original_runner(bad, 'invalid')
except ValueError:
    pass
else:
    raise AssertionError('Direct constructor accepted incomplete scenario')
assert not pathlib.Path('must-not-exist').exists()
calls = []
class CaptureRunner:
    def __init__(self, config, name):
        calls.append(config['num_rounds'])
        self.name = name
    def run(self):
        return {'experiment_name': self.name, 'elapsed_time': 0, 'results_dir': '.'}
runner.ExperimentRunner = CaptureRunner
for kwargs in ({'num_rounds': 72}, {'variant': 'typo'}):
    try:
        runner.run_experiment('exp3', **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError('Runner accepted invalid configuration')
assert calls == []
try:
    runner.run_all_experiments(num_rounds=11)
except ValueError:
    pass
else:
    raise AssertionError('Batch accepted an uncovered exp1 horizon')
assert calls == [], 'Batch started before validating every scenario'
runner.run_all_experiments(num_rounds=3)
assert calls == [3] * 7, calls
events = []
fake_numpy = types.ModuleType('numpy')
fake_numpy.random = types.SimpleNamespace(seed=lambda value: events.append(('seed', value)))
sys.modules['numpy'] = fake_numpy
class EmptySimulator:
    def __init__(self, config):
        events.append(('simulator',))
    def run_coupled(self):
        return {}
runner.MarketSimulator = EmptySimulator
execution = object.__new__(original_runner)
execution.config = build_experiment_config('exp3')
execution.config['llm_client']['random_seed'] = 73
execution.experiment_name = 'seed-test'
execution.results_dir = pathlib.Path('.')
execution.prepare_agents = lambda: events.append(('prepare_agents',))
execution._generate_visualizations = lambda: None
execution.run()
assert events == [('seed', 73), ('prepare_agents',), ('simulator',)], events
'''
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, "-S", "-B", "-c", script, str(SRC)],
                                    cwd=temporary, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
