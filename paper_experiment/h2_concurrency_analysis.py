"""Reproduce the episode-level disruption-concurrency trend test.

The dependent variable is the paired recovery advantage of CRL over RL-only,
defined as RL-only recovery days minus CRL recovery days. The predictor is the
ordered disruption-concurrency level: single=1, dual=2, and compound=3.
Uncertainty is estimated with a stratified episode bootstrap that preserves the
observed number of episodes in each concurrency group.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress


RESULTS = Path(__file__).parent / "results"
INPUT = RESULTS / "per_episode.csv"
OUTPUT_JSON = RESULTS / "h2_concurrency_results.json"
OUTPUT_CSV = RESULTS / "h2_concurrency_per_episode.csv"
BOOTSTRAP_SEED = 20260823
BOOTSTRAP_REPS = 20000
LEVELS = {"single": 1, "dual": 2, "compound": 3}


def slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))


def stratified_bootstrap(frame, repetitions, seed):
    rng = np.random.RandomState(seed)
    groups = [
        frame[frame["scenario_type"] == label].reset_index(drop=True)
        for label in LEVELS
    ]
    slopes = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        samples = []
        for group in groups:
            indices = rng.randint(0, len(group), size=len(group))
            samples.append(group.iloc[indices])
        sample = pd.concat(samples, ignore_index=True)
        slopes[repetition] = slope(
            sample["concurrency_level"], sample["recovery_advantage_days"]
        )
    lower, upper = np.percentile(slopes, [2.5, 97.5])
    lower_tail = (np.count_nonzero(slopes <= 0.0) + 1) / (repetitions + 1)
    upper_tail = (np.count_nonzero(slopes >= 0.0) + 1) / (repetitions + 1)
    return {
        "repetitions": repetitions,
        "seed": seed,
        "bootstrap_95_ci": [round(float(lower), 6), round(float(upper), 6)],
        "two_sided_sign_p": round(float(min(1.0, 2.0 * min(lower_tail, upper_tail))), 10),
        "bootstrap_slope_mean": round(float(slopes.mean()), 6),
        "bootstrap_slope_sd": round(float(slopes.std(ddof=1)), 6),
    }


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "mean": round(float(values.mean()), 6),
        "sd": round(float(values.std(ddof=1)), 6),
        "se": round(float(values.std(ddof=1) / np.sqrt(len(values))), 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    args = parser.parse_args()
    frame = pd.read_csv(INPUT)[
        ["episode_id", "scenario_type", "rl_only_recovery_days", "crl_recovery_days"]
    ].copy()
    unexpected = sorted(set(frame["scenario_type"]) - set(LEVELS))
    if unexpected:
        raise ValueError(f"Unexpected scenario types: {unexpected}")
    frame["concurrency_level"] = frame["scenario_type"].map(LEVELS)
    frame["recovery_advantage_days"] = (
        frame["rl_only_recovery_days"] - frame["crl_recovery_days"]
    )

    regression = linregress(
        frame["concurrency_level"], frame["recovery_advantage_days"]
    )
    bootstrap = stratified_bootstrap(
        frame, args.bootstrap_reps, BOOTSTRAP_SEED
    )
    summary = {
        "design": {
            "unit": "held-out paired episode averaged over five training seeds",
            "n_episodes": int(len(frame)),
            "concurrency_coding": LEVELS,
            "outcome": "RL-only recovery days minus CRL recovery days",
        },
        "episode_level_linear_trend": {
            "intercept": round(float(regression.intercept), 6),
            "slope_days_per_concurrency_level": round(float(regression.slope), 6),
            "standard_error": round(float(regression.stderr), 6),
            "pearson_r": round(float(regression.rvalue), 6),
            "r_squared": round(float(regression.rvalue ** 2), 6),
            "two_sided_ols_p": round(float(regression.pvalue), 10),
            **bootstrap,
        },
        "group_summaries": {
            label: mean_se(
                frame.loc[
                    frame["scenario_type"] == label, "recovery_advantage_days"
                ]
            )
            for label in LEVELS
        },
    }
    frame.to_csv(OUTPUT_CSV, index=False, lineterminator="\n")
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
