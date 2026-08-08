# Supply Chain CRL — Reproducibility Package

Code and data for **"AI-Driven Supply Chain Resilience under Multi-Source Disruption"**
(Causal Reinforcement Learning framework for healthcare supply chain resilience).

## What this repository contains

- `paper_experiment/` — the complete experiment pipeline that generates every number in the paper
  - `calibration.py` — derives all environment parameters from the datasets (marginal distributions only)
  - `environment.py` — four-tier supply chain simulation (3 suppliers, 2 corridors, 1 DC, 4 service points). Includes the full structural assumptions table (`ASSUMPTIONS` dict). The environment never knows which agent is acting; LPI affects environmental constraints only.
  - `agents.py` — PPO (from scratch, numpy), CRL (PPO + causal action masking + reward shaping from interventionally estimated ATEs), and the sample-average stochastic lookahead planning baseline
  - `run_experiment.py` — main experiment: 5 training seeds, 200 paired held-out episodes → `results/frozen_results.json`, `results/per_episode.csv` (Tables II & IIb)
  - `sample_efficiency.py` — training-budget × seed study (Section 5.1)
  - `robustness.py` — ablation, recovery-threshold sensitivity, tail-severity stress test (Section 5.2)
  - `round2_analysis.py`, `round3_analysis.py` — censoring/survival robustness, P2 trend test, P3 moderation test, P4 inference, LPI mapping-endpoint sensitivity
- `data/` — calibration inputs
  - `GHSC_PSM_Synthetic_Resilience_Dataset_v2_consistent.csv` — **synthetic** dataset (2,000 records) calibrated to figures publicly reported in USAID GHSC-PSM annual reports FY2022–FY2024. This is not raw transaction data; see paper Section 4.2.
  - `International_LPI_from_2007_to_2023.xlsx` — World Bank Logistics Performance Index (2018 & 2023 editions used)
  - `emdat_custom_request_2025-10-23.xlsx` — EM-DAT custom extract (natural-hazard & epidemic events; EM-DAT excludes armed conflict)
- `paper_experiment/results/` — frozen outputs used in the paper

## Reproducing the paper's results

```
pip install -r requirements_paper.txt
cd paper_experiment
python run_experiment.py        # Tables II, IIb  (~5 min, CPU only)
python sample_efficiency.py     # sample-efficiency study
python robustness.py            # ablation, thresholds, stress test
python round2_analysis.py       # censoring, P4 inference, stress diagnostics
python round3_analysis.py       # log-rank, LPI mapping sensitivity
```

All randomness is seeded (seed = 42); results regenerate exactly.

## Honesty notes

- No performance difference between models is programmed anywhere. All action effects are mechanistic (capacities, lead times, inventories, costs).
- Causal effects used by CRL are estimated from 400 randomized-intervention episodes, not hand-specified.
- Earlier files in this repository's history (e.g., `comprehensive_comparison.py` and associated "proof" documents) contained hard-coded comparison outputs and inflated claims; they are superseded and removed. The `src/` DQN prototype predates the paper's experiment and is retained for archival purposes only.
