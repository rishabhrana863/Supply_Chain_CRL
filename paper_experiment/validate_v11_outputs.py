"""Validate the frozen Version 11 multi-seed result package."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path(__file__).parent / "results"
PLANNER_SEEDS = [42, 43, 44, 45, 46]
PLANNER_CONFIGURATIONS = 7


def close(observed, expected, label, tolerance=1e-5):
    if not np.isclose(observed, expected, atol=tolerance, rtol=0.0):
        raise AssertionError(
            f"{label}: observed={observed}, expected={expected}"
        )


def main():
    frontier = json.loads(
        (RESULTS / "planner_frontier_results.json").read_text()
    )
    if frontier["design"]["planner_seeds"] != PLANNER_SEEDS:
        raise AssertionError("Unexpected planner seeds")
    if frontier["design"]["n_eval"] != 200:
        raise AssertionError("Frontier is not the 200-episode run")

    raw = pd.read_csv(RESULTS / "planner_frontier_per_episode_by_seed.csv")
    averaged = pd.read_csv(RESULTS / "planner_frontier_per_episode.csv")
    expected_raw = 200 * len(PLANNER_SEEDS) * PLANNER_CONFIGURATIONS
    if len(raw) != expected_raw:
        raise AssertionError(f"Expected {expected_raw} raw frontier rows, found {len(raw)}")
    counts = raw.groupby(["configuration", "planner_seed"]).size()
    if not (counts == 200).all():
        raise AssertionError("A configuration-seed frontier cell is incomplete")
    if len(averaged) != 200 * (PLANNER_CONFIGURATIONS + 2):
        raise AssertionError("Unexpected seed-averaged frontier row count")

    for configuration, metrics in frontier["aggregate"].items():
        subset = averaged[averaged["configuration"] == configuration]
        for metric in ("recovery_days", "total_cost", "service_cov"):
            close(
                subset[metric].mean(), metrics[metric]["mean"],
                f"{configuration}.{metric}", tolerance=1e-4,
            )

    robustness = json.loads(
        (RESULTS / "robustness_v2_results.json").read_text()
    )
    if robustness["design"]["planner_seeds"] != PLANNER_SEEDS:
        raise AssertionError("Unexpected robustness planner seeds")
    robustness_raw = pd.read_csv(
        RESULTS / "robustness_v2_per_episode_by_seed.csv"
    )
    robustness_averaged = pd.read_csv(
        RESULTS / "robustness_v2_per_episode.csv"
    )
    if len(robustness_raw) != 6500:
        raise AssertionError("Unexpected raw robustness row count")
    if len(robustness_averaged) != 1300:
        raise AssertionError("Unexpected averaged robustness row count")
    validations = robustness["threshold_sensitivity"]["validation"]
    validations += [robustness["tail_severity"]["validation"]]
    validations += [
        value["validation"]
        for value in robustness["lpi_mapping_sensitivity"].values()
    ]
    if not all(item["passed"] for item in validations):
        raise AssertionError("A frozen weekly validation failed")

    h2 = json.loads((RESULTS / "h2_concurrency_results.json").read_text())
    trend = h2["episode_level_linear_trend"]
    if trend["bootstrap_95_ci"][0] <= 0.0:
        raise AssertionError("H2 bootstrap interval includes zero")
    if trend["two_sided_ols_p"] >= 0.001:
        raise AssertionError("H2 episode-level trend is not p < .001")

    print("Version 11 frozen outputs validated")
    print(f"Frontier rows: {len(raw)} raw, {len(averaged)} seed-averaged")
    print(
        "Robustness rows: "
        f"{len(robustness_raw)} raw, {len(robustness_averaged)} seed-averaged"
    )


if __name__ == "__main__":
    main()
