# Validation record

Validated on 29 October 2026, using a working copy based on commit
`587a33b18efffa926748d3eebb32643dfe418156` plus the accompanying changes.

## Environment

Python 3.12.14, NumPy 1.26.4, pandas 2.2.3, PYPOWER 5.1.21, PyTorch 2.15.0+cpu,
Stable-Baselines3 2.9.2 and pytest 9.2.0. Tests used the deterministic mock LLM;
no paid model calls or original study reruns were performed.

## Checks

| Check | Result |
|---|---|
| `make test` | 48 passed, including the original 17 framework tests |
| `make check-style` | Passed the repository's syntax/undefined-name checks |
| `git diff --check` | Passed |
| `python3 -S -B -m unittest discover -s tests -p test_experiment_contract.py -q` | 17 passed with site packages disabled |
| `python3 -S -B src/run_experiments.py --experiment all --quick-test --dry-run` | Seven variants, each three rounds, each with an isolated `rounds_3` output path |
| Documentation file links and citation metadata | Local targets exist; eight authors and the publication DOI are present |

The full suite emitted non-failing warnings about pandas concatenation, an installed pytest plugin's
deprecated API and legends in minimal plotting fixtures. These checks do not establish compatibility with
every dependency version permitted by `requirements.txt`.

## Outage feedback regression

The original coupled-simulation tests exposed a pre-existing failure when OPF omitted a zero-capacity unit.
The coupled market assembled outcomes only for units returned by electricity clearing, losing that
participant's carbon settlement and subsequent memory feedback.

The fix assembles outcomes from actual submitted actions and gives a missing electricity award zero dispatch.
A direct regression retains a stopped participant's carbon purchase, cost, quota and submitted bid, and verifies
negative profit feedback and a memory record. A nonparticipant is not added. The two existing end-to-end tests
pass with their original assertions unchanged. One failing case was also reproduced in an isolated baseline
worktree before the fix.

## Research-result status

The retained result CSVs and PNGs were not modified or regenerated. Tests confirm software behavior on
synthetic/mocked inputs. Full replication of the 72-round convergence comparison and reconciliation of
AUD/CNY price scales remain open, as detailed in [the reproducibility ledger](reproducibility.md).
