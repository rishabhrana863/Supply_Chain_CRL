# Supply Chain CRL Reproducibility Package

Code and data for **"Cost-Efficient Healthcare Logistics Recovery under
Concurrent Disruptions: A Causal Reinforcement Learning Approach."**

## What this repository contains

- `paper_experiment/` — the complete experiment pipeline that generates every number in the paper
  - `calibration.py` — derives all environment parameters from the datasets (marginal distributions only)
  - `environment.py` — four-tier supply chain simulation (3 suppliers, 2 corridors, 1 DC, 4 service points). Includes the full structural assumptions table (`ASSUMPTIONS` dict). The environment never knows which agent is acting; LPI affects environmental constraints only.
  - `agents.py` — PPO (from scratch, numpy), CRL (PPO + causal action masking + reward shaping from interventionally estimated ATEs), and the sample-average stochastic lookahead planning baseline
  - `run_experiment.py`: main experiment with 5 training seeds and 200 paired held-out episodes
  - `planner_frontier.py`: cadence, horizon, scenario-count, and expediting sensitivity; generates the observed cost-recovery frontier
  - `sample_efficiency.py`: training-budget by seed study
  - `robustness.py`: ablation and recovery-threshold sensitivity with paired episode outputs
  - `robustness_v2.py`: daily-planner extensions for threshold, tail-severity, and LPI-mapping analyses
  - `round2_analysis.py`, `round3_analysis.py`: censoring, survival, policy-alignment, tail-severity, and LPI-mapping analyses
- `data/`: calibration inputs
  - `GHSC_PSM_Synthetic_Resilience_Dataset_v2_consistent.csv`: **synthetic** dataset with 2,000 records calibrated to figures publicly reported in USAID GHSC-PSM reports. It is not raw transaction data.
  - `International_LPI_from_2007_to_2023.xlsx`: World Bank Logistics Performance Index data. The 2018 and 2023 survey-based editions are used.
- `paper_experiment/results/`: frozen aggregate and per-episode outputs used in the paper

The licensed EM-DAT extract is not redistributed and does not enter numerical
parameter estimation.

## Reproducing the paper's results

```
pip install -r requirements_paper.txt
cd paper_experiment
python run_experiment.py
python planner_frontier.py
python sample_efficiency.py
python robustness.py
python robustness_v2.py
python round2_analysis.py
python round3_analysis.py
```

All analysis scripts use fixed seeds. The planner compares candidate actions
with common random numbers at each replanning point.

## Honesty notes

- No performance difference between models is programmed anywhere. All action effects are mechanistic (capacities, lead times, inventories, and costs).
- Causal effects used by CRL are estimated from 400 randomized-intervention episodes, not hand-specified.
- The planner sensitivity analysis shows a genuine cost-recovery tradeoff. Longer lookahead horizons can recover faster by using substantially more air expediting and incurring higher cost.
- Earlier files in this repository's history, including `comprehensive_comparison.py` and associated proof documents, contained hard-coded comparison outputs and inflated claims. They are superseded and removed. The `src/` DQN prototype predates the paper experiment and is retained for archival purposes only.
