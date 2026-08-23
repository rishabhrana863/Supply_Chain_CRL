"""Cadence-matched planner extensions for the robustness analyses.

The script adds the daily H=14, K=12 planner to the recovery-threshold,
tail-severity, and LPI-mapping analyses. For each analysis it evaluates planner
seeds 42 through 46 on the original contexts and environment seeds. Seed 42
first reproduces the corrected frozen weekly result. Reported means then
average the five planner seeds within each episode.

Run after ``robustness.py`` from ``paper_experiment``:

    python robustness_v2.py
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import environment as envmod
from agents import StochasticOptAgent
from calibration import Calibration
from environment import N_ACTIONS, SupplyChainEnv


RESULTS = Path(__file__).parent / "results"
OUTPUT_JSON = RESULTS / "robustness_v2_results.json"
OUTPUT_CSV = RESULTS / "robustness_v2_per_episode.csv"
OUTPUT_RAW_CSV = RESULTS / "robustness_v2_per_episode_by_seed.csv"
CALIBRATION_SEED = 42
PLANNER_SEEDS = (42, 43, 44, 45, 46)


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": round(float(values.mean()), 6),
        "sd": round(float(values.std(ddof=1)), 6),
        "se": round(float(values.std(ddof=1) / np.sqrt(len(values))), 6),
    }


def evaluate(planner, calibration, contexts, seed_base, thresholds, planner_seed):
    rows = []
    for episode_id, context in enumerate(contexts):
        env = SupplyChainEnv(
            context, calibration, np.random.RandomState(seed_base + episode_id)
        )
        state = env.reset()
        planner.reset()
        action_counts = np.zeros(N_ACTIONS, dtype=int)
        while True:
            action = planner.act(state, env=env)
            action_counts[action] += 1
            state, _, done, _ = env.step(action)
            if done:
                break
        base_metrics = env.episode_metrics()
        row = {
            "episode_id": episode_id,
            "scenario_type": context["scenario_type"],
            "planner_seed": planner_seed,
            "lpi": context["lpi"],
            "total_severity": context["total_severity"],
            "total_cost": base_metrics["total_cost"],
            "service_cov": base_metrics["service_cov"],
            "air_expedites": int(action_counts[2]),
        }
        for threshold in thresholds:
            row[f"recovery_{threshold:.2f}"] = env.episode_metrics(
                threshold=threshold
            )["recovery_days"]
        rows.append(row)
    return rows


def summarize(rows, thresholds):
    summary = {
        "total_cost": mean_se([row["total_cost"] for row in rows]),
        "service_cov": mean_se([row["service_cov"] for row in rows]),
        "air_expedites": mean_se([row["air_expedites"] for row in rows]),
    }
    summary["recovery_days"] = {
        f"{threshold:.2f}": mean_se(
            [row[f"recovery_{threshold:.2f}"] for row in rows]
        )
        for threshold in thresholds
    }
    return summary


def planner(calibration, cadence, planner_seed):
    return StochasticOptAgent(
        calibration,
        seed=planner_seed,
        replan_every=cadence,
        lookahead=14,
        scenario_count=12,
        name="planner_weekly" if cadence == 7 else "planner_daily",
    )


def evaluate_task(task):
    """Evaluate one cadence and seed, optionally under an LPI variant."""
    cadence, planner_seed, contexts, seed_base, thresholds, changes = task
    base_assumptions = {key: envmod.ASSUMPTIONS[key] for key in changes}
    try:
        envmod.ASSUMPTIONS.update(changes)
        calibration = Calibration(seed=CALIBRATION_SEED)
        rows = evaluate(
            planner(calibration, cadence, planner_seed), calibration, contexts,
            seed_base, thresholds, planner_seed,
        )
    finally:
        envmod.ASSUMPTIONS.update(base_assumptions)
    return cadence, planner_seed, rows


def average_rows(rows):
    """Average planner-seed outcomes within each held-out episode."""
    metadata = ["episode_id", "scenario_type", "lpi", "total_severity"]
    numeric = [key for key in rows[0] if key not in metadata + ["planner_seed"]]
    averaged = []
    episode_ids = sorted({row["episode_id"] for row in rows})
    for episode_id in episode_ids:
        episode_rows = [row for row in rows if row["episode_id"] == episode_id]
        item = {key: episode_rows[0][key] for key in metadata}
        item["planner_seed"] = "mean"
        for key in numeric:
            item[key] = float(np.mean([row[key] for row in episode_rows]))
        averaged.append(item)
    return averaged


def evaluate_seed_set(contexts, seed_base, thresholds, planner_seeds, workers, changes=None):
    changes = changes or {}
    tasks = [
        (cadence, planner_seed, contexts, seed_base, thresholds, changes)
        for cadence in (7, 1)
        for planner_seed in planner_seeds
    ]
    by_seed = {"planner_weekly": {}, "planner_daily": {}}
    if workers == 1:
        completed = [evaluate_task(task) for task in tasks]
    else:
        completed = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_task, task): task for task in tasks}
            for future in as_completed(futures):
                completed.append(future.result())
    for cadence, planner_seed, rows in completed:
        configuration = "planner_weekly" if cadence == 7 else "planner_daily"
        by_seed[configuration][planner_seed] = rows
        print(f"Completed {configuration}, planner seed {planner_seed}", flush=True)
    averaged = {}
    raw = {}
    for configuration in by_seed:
        raw[configuration] = [
            row for seed in planner_seeds for row in by_seed[configuration][seed]
        ]
        averaged[configuration] = average_rows(raw[configuration])
    return averaged, raw, by_seed


def seed_summaries(by_seed, thresholds):
    output = {}
    for configuration, seed_rows in by_seed.items():
        output[configuration] = {
            str(seed): summarize(rows, thresholds)
            for seed, rows in sorted(seed_rows.items())
        }
    return output


def validation(observed, expected, label, tolerance=0.02):
    difference = round(float(observed - expected), 6)
    passed = abs(difference) <= tolerance
    if not passed:
        raise RuntimeError(
            f"{label} weekly reproduction failed: observed={observed:.4f}, "
            f"expected={expected:.4f}, difference={difference:.4f}"
        )
    return {
        "label": label,
        "observed_weekly_mean": round(float(observed), 6),
        "frozen_weekly_mean": round(float(expected), 6),
        "difference": difference,
        "tolerance": tolerance,
        "passed": True,
    }


def append_long(output_rows, analysis, regime, configuration, rows, thresholds):
    for row in rows:
        item = {
            "analysis": analysis,
            "regime": regime,
            "configuration": configuration,
            **row,
        }
        for threshold in (0.90, 0.95, 0.98):
            item.setdefault(f"recovery_{threshold:.2f}", "")
        output_rows.append(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--planner-seeds", type=int, nargs="+", default=list(PLANNER_SEEDS)
    )
    args = parser.parse_args()
    planner_seeds = tuple(args.planner_seeds)
    if len(planner_seeds) < 2:
        raise ValueError("Robustness extension requires at least two planner seeds")
    if 42 not in planner_seeds:
        raise ValueError("Planner seed 42 is required for frozen-result validation")
    robustness = json.loads((RESULTS / "robustness_results.json").read_text())
    round2 = json.loads((RESULTS / "round2_results.json").read_text())
    round3 = json.loads((RESULTS / "round3_results.json").read_text())
    calibration = Calibration(seed=CALIBRATION_SEED)
    output_rows = []
    raw_output_rows = []
    results = {
        "design": {
            "calibration_seed": CALIBRATION_SEED,
            "planner_seeds": list(planner_seeds),
            "weekly_replan_days": 7,
            "daily_replan_days": 1,
            "lookahead_days": 14,
            "scenario_count": 12,
            "common_random_numbers_across_actions": True,
        }
    }

    # Recovery-threshold sensitivity on the original 150 contexts.
    thresholds = (0.90, 0.95, 0.98)
    threshold_contexts = calibration.sample_episode_contexts(
        150, np.random.RandomState(9000)
    )
    threshold_averaged, threshold_raw, threshold_by_seed = evaluate_seed_set(
        threshold_contexts, 63000, thresholds, planner_seeds, args.workers
    )
    threshold_weekly = threshold_averaged["planner_weekly"]
    threshold_daily = threshold_averaged["planner_daily"]
    threshold_summary = {
        "planner_weekly": summarize(threshold_weekly, thresholds),
        "planner_daily": summarize(threshold_daily, thresholds),
    }
    expected_threshold = robustness["threshold_sensitivity"]["stochastic_opt"]
    threshold_validations = []
    for threshold in thresholds:
        observed = summarize(
            threshold_by_seed["planner_weekly"][42], thresholds
        )["recovery_days"][f"{threshold:.2f}"]["mean"]
        expected = float(expected_threshold[str(threshold)])
        threshold_validations.append(
            validation(observed, expected, f"threshold_{threshold:.2f}")
        )
    results["threshold_sensitivity"] = {
        "validation": threshold_validations,
        "summary": threshold_summary,
        "planner_seed_summaries": seed_summaries(threshold_by_seed, thresholds),
    }
    append_long(output_rows, "threshold", "baseline", "planner_weekly", threshold_weekly, thresholds)
    append_long(output_rows, "threshold", "baseline", "planner_daily", threshold_daily, thresholds)
    append_long(raw_output_rows, "threshold", "baseline", "planner_weekly", threshold_raw["planner_weekly"], thresholds)
    append_long(raw_output_rows, "threshold", "baseline", "planner_daily", threshold_raw["planner_daily"], thresholds)

    # Tail-severity distribution shift from round2_analysis.py.
    tail_rng = np.random.RandomState(4321)
    tail_contexts = calibration.sample_episode_contexts(300, tail_rng)
    all_types = list(calibration.disruption_type_probs.keys())
    for context in tail_contexts:
        chosen = list(tail_rng.choice(all_types, size=3, replace=False))
        context["disruption_types"] = chosen
        context["severities"] = {
            disruption_type: int(tail_rng.choice([4, 5]))
            for disruption_type in chosen
        }
        context["total_severity"] = sum(context["severities"].values())
        context["scenario_type"] = "compound"
    tail_averaged, tail_raw, tail_by_seed = evaluate_seed_set(
        tail_contexts, 77000, (0.95,), planner_seeds, args.workers
    )
    tail_weekly = tail_averaged["planner_weekly"]
    tail_daily = tail_averaged["planner_daily"]
    tail_summary = {
        "planner_weekly": summarize(tail_weekly, (0.95,)),
        "planner_daily": summarize(tail_daily, (0.95,)),
    }
    expected_tail = float(
        round2["new_stress_results"]["stochastic_opt"]["recovery_mean"]
    )
    observed_tail = summarize(
        tail_by_seed["planner_weekly"][42], (0.95,)
    )["recovery_days"]["0.95"]["mean"]
    results["tail_severity"] = {
        "validation": validation(observed_tail, expected_tail, "tail_severity"),
        "summary": tail_summary,
        "planner_seed_summaries": seed_summaries(tail_by_seed, (0.95,)),
    }
    append_long(output_rows, "tail_severity", "random_3_of_4", "planner_weekly", tail_weekly, (0.95,))
    append_long(output_rows, "tail_severity", "random_3_of_4", "planner_daily", tail_daily, (0.95,))
    append_long(raw_output_rows, "tail_severity", "random_3_of_4", "planner_weekly", tail_raw["planner_weekly"], (0.95,))
    append_long(raw_output_rows, "tail_severity", "random_3_of_4", "planner_daily", tail_raw["planner_daily"], (0.95,))

    # LPI mapping-endpoint sensitivity from round3_analysis.py.
    variants = {
        "mild": {
            "customs_delay_at_lpi2": 4.0,
            "switch_setup_at_lpi2": 9.0,
            "alt_supplier_p_at_lpi2": 0.65,
            "transport_cv_at_lpi2": 0.25,
        },
        "harsh": {
            "customs_delay_at_lpi2": 8.0,
            "switch_setup_at_lpi2": 15.0,
            "alt_supplier_p_at_lpi2": 0.45,
            "transport_cv_at_lpi2": 0.45,
        },
    }
    lpi_results = {}
    for variant_name, changes in variants.items():
            lpi_contexts = calibration.sample_episode_contexts(
                100, np.random.RandomState(9000)
            )
            lpi_averaged, lpi_raw, lpi_by_seed = evaluate_seed_set(
                lpi_contexts, 63000, (0.95,), planner_seeds, args.workers,
                changes=changes,
            )
            lpi_weekly = lpi_averaged["planner_weekly"]
            lpi_daily = lpi_averaged["planner_daily"]
            lpi_summary = {
                "planner_weekly": summarize(lpi_weekly, (0.95,)),
                "planner_daily": summarize(lpi_daily, (0.95,)),
            }
            expected_lpi = float(
                round3["lpi_mapping_sensitivity"][variant_name]["stochastic_planner"]
            )
            observed_lpi = summarize(
                lpi_by_seed["planner_weekly"][42], (0.95,)
            )["recovery_days"]["0.95"]["mean"]
            lpi_results[variant_name] = {
                "validation": validation(
                    observed_lpi, expected_lpi, f"lpi_{variant_name}"
                ),
                "summary": lpi_summary,
                "planner_seed_summaries": seed_summaries(lpi_by_seed, (0.95,)),
            }
            append_long(
                output_rows, "lpi_mapping", variant_name,
                "planner_weekly", lpi_weekly, (0.95,)
            )
            append_long(
                output_rows, "lpi_mapping", variant_name,
                "planner_daily", lpi_daily, (0.95,)
            )
            append_long(
                raw_output_rows, "lpi_mapping", variant_name,
                "planner_weekly", lpi_raw["planner_weekly"], (0.95,)
            )
            append_long(
                raw_output_rows, "lpi_mapping", variant_name,
                "planner_daily", lpi_raw["planner_daily"], (0.95,)
            )
    results["lpi_mapping_sensitivity"] = lpi_results

    with OUTPUT_JSON.open("w") as handle:
        json.dump(results, handle, indent=2)
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    with OUTPUT_RAW_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(raw_output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(raw_output_rows)
    print(json.dumps(results, indent=2))
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_CSV}")
    print(f"Saved {OUTPUT_RAW_CSV}")


if __name__ == "__main__":
    main()
