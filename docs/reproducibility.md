# Reproducibility ledger

[Project home](../README.md) · [Five-minute case study](case-study.md) · [Input data](data.md)

This ledger separates released software, retained research summaries and the materials needed to reproduce
them. An offline test passing is evidence about the implementation, not confirmation of a reported effect.

## Released configurations

| Experiment | Variants | Default rounds | Status |
|---|---|---:|---|
| exp1 | RB, RL, IBR-ADF | 10 | Released short behavior scenario; exp1 loads the observed NSW1 reference |
| exp2 | without_adf, with_adf | 20 | Released efficiency scenario; comparison arrays are scenario settings |
| exp3 | without_ibr_cr, with_ibr_cr | 30 | Released short reasoning scenario; does not reproduce the 72-round convergence summary |

Inspect the exact effective configuration before a run:

```bash
python3 src/run_experiments.py --experiment exp3 --dry-run
python3 src/run_experiments.py --experiment exp2 --variant with_adf --rounds 3 --dry-run
python3 src/run_experiments.py --experiment all --quick-test --dry-run
```

Inspection has no API or simulation side effects. Invalid variants, non-positive/non-integer round counts and
insufficient series coverage fail before an experiment runner is constructed. Shortening a horizon uses the
configured prefix; it does not remove the need to check whether an event occurs within that shorter run.
Explicit horizon overrides, including the three-round quick test, write CSV, JSON, logs and figures under
the variant's `rounds_N/` subdirectory. They do not overwrite the default horizon's outputs.

All enabled operational arrays, including demand and reserve profiles, must cover the requested rounds.
The shared builder also checks supplied comparison-series coverage. It does not extend a scenario by cycling,
padding or extrapolating it. A user-supplied complete 72-round scenario is a new declared configuration unless
its identity with the original study is independently established.

## Evidence inventory

| Artifact | What is retained | What it establishes |
|---|---|---|
| `src/true_digital_twin_agents.json` | Fixed 50-agent population | Released agent membership and parameters |
| `src/data_integration/` | 72 hourly NSW1 values, timestamps and provenance | An observed reference series; not a completed simulation |
| `results/exp1_behavior_validation/` | One summary CSV and four images | Retained behavior/reference results; no original per-run trajectory |
| `results/exp2_adf_efficiency/` | Three summary CSVs and one image | Separate efficiency, repeated-run and sensitivity summaries |
| `results/exp3_ibr_cr_effectiveness/` | One convergence summary CSV | Reported aggregate values; insufficient to recompute convergence |
| `tests/` | Deterministic offline tests | Configuration, cognitive-module and short simulation behavior |

Together with the agent-setup CSV, `results/` contains six aggregate CSVs and five PNGs.
The original Monte Carlo trajectories, model responses and complete experiment 3 configuration are not supplied.
Table IV and Table V have different runtime and profit totals; neither table reconstructs the other's runs.
The existing PNG distribution does not independently verify the repeated-run population claimed by Table V.

## The 30-round / 72-round gap

The retained experiment 3 summary reports convergence at round 31, with its baseline not converged in 72 rounds.
The released command runs 30 rounds. A run cannot establish a convergence event outside its horizon.

Changing only `num_rounds` previously allowed the simulator to exhaust operational series and use different
fallback behavior for different variables. The run-time configuration check now rejects that incomplete input.
The base configuration's 72-element demand curve and the bundled 72-hour price reference do not supply the
missing gas, outage, renewable and carbon scenario assumptions.

The current `tracking` entries are configuration declarations. They do not compute the study's convergence
criterion. The visualization module's rolling-price plot is not the reported congested-subregion statistic.

Before claiming a replication, obtain and retain:

1. All operational series for the original horizon, agent population and network/constraint settings.
2. The congested-subregion definition and the generator IDs included in the statistic.
3. The rolling-window width, standard-deviation convention, threshold, persistence rule and round numbering.
4. Per-round prices, bids, profits and model-call records for each comparison run.
5. Model/provider settings, seeds, run count and the exact code/configuration used to create each summary.

The reference text mentions a price standard deviation below 3 CNY/MWh for 10 consecutive rounds, but does not
supply the complete rolling-window specification above. This release does not invent the missing definition.

## Reference prices and units

The bundled observed series is 72 hourly NSW1 RRP values for 15–17 July 2024, in **AUD/MWh**.
It is aggregated from 5-minute dispatch-price records; the first hour uses 11 observations and subsequent
hours use 12. Exact timestamps and the original source checksum are recorded in [Input data](data.md).
Experiment 1 loads the full reference alongside its 10-round scenario; its AUD/MWh overlay is skipped.
Experiments 2 and 3 retain
their separately defined scenario comparison arrays, rather than claiming those arrays are observed NSW1 prices.

Simulation figures and retained aggregate tables label simulated prices **CNY/MWh**. No currency conversion
or shared price-scale calibration is provided. The visualizer therefore skips an electricity-reference overlay
when units are missing or do not match CNY/MWh. It uses explicit round indices to align compatible reference
series, including shortened runs. No exchange rate, observed carbon-price provenance or new MAPE is inferred.

The retained MAPE values are reference summaries and have not been re-established against the bundled hourly
series. Carbon comparison arrays are labelled scenario references, not verified observed ACCU prices.
