# LLM-CECM helper targets. All simulations run from the src/ directory.
PYTHON ?= python3
VENV   ?= .venv
PY      = $(VENV)/bin/python
SRC     = src

.PHONY: install run exp1 exp2 exp3 all quick sensitivity check-style test clean

## Create a virtual environment and install dependencies
install:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

## Default run: Experiment 1 with the IBR-ADF agent
run:
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp1 --variant IBR-ADF

## Experiment 1: behaviour validation (RB / RL / IBR-ADF)
exp1:
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp1 --variant RB
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp1 --variant RL
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp1 --variant IBR-ADF

## Experiment 2: ADF efficiency (IBR vs IBR-ADF)
exp2:
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp2 --variant without_adf
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp2 --variant with_adf

## Experiment 2 (Table VI): sensitivity of the ADF threshold delta_perf
sensitivity:
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp2 --variant with_adf --delta-perf 0.05
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp2 --variant with_adf --delta-perf 0.15
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp2 --variant with_adf --delta-perf 0.25

## Experiment 3: IBR-CR effectiveness (ADF vs IBR-ADF)
exp3:
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp3 --variant without_ibr_cr
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp3 --variant with_ibr_cr

## Every experiment in the paper
all:
	cd $(SRC) && ../$(PY) run_experiments.py --experiment all

## 3-round smoke run of Experiment 1 (needs an API key)
quick:
	cd $(SRC) && ../$(PY) run_experiments.py --experiment exp1 --variant IBR-ADF --quick-test

## Syntax errors and undefined names
check-style:
	$(PY) -m flake8 $(SRC) tests --count --select=E9,F63,F7,F82 --show-source --statistics

## Offline tests with a mock LLM (no API key required)
test:
	$(PY) -m pytest -q tests

## Remove caches and locally generated simulation outputs
clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache $(SRC)/results $(SRC)/rl_tensorboard_logs
