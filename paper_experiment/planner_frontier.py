"""Evaluate the empirical cost, recovery, and stability frontier.

The script compares the frozen CRL and RL-only policies with seven planner
configurations on the same 200 held-out contexts and environment seeds. It
reports paired uncertainty, identifies the discrete nondominated set, and
creates the figure used in the manuscript.

Run from ``paper_experiment`` after ``run_experiment.py``:

    python planner_frontier.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from agents import StochasticOptAgent
from calibration import Calibration
from environment import N_ACTIONS, SupplyChainEnv


SEED = 42
N_EVAL = 200
EVAL_CONTEXT_SEED = 9000
ENV_SEED_BASE = EVAL_CONTEXT_SEED * 7
BOOTSTRAP_SEED = 20260823
BOOTSTRAP_REPS = 10000
RESULTS = Path(__file__).parent / "results"
PRIMARY_EPISODES = RESULTS / "per_episode.csv"
OUTPUT_EPISODES = RESULTS / "planner_frontier_per_episode.csv"
OUTPUT_SUMMARY = RESULTS / "planner_frontier_results.json"
FIGURE_STEM = RESULTS / "planner_frontier"


CONFIGS = {
    "planner_weekly_h14": {
        "label": "Weekly H=14",
        "replan_every": 7,
        "lookahead": 14,
        "scenario_count": 12,
    },
    "planner_daily_h14": {
        "label": "Daily H=14",
        "replan_every": 1,
        "lookahead": 14,
        "scenario_count": 12,
    },
    "planner_daily_no_expedite": {
        "label": "Daily, no air",
        "replan_every": 1,
        "lookahead": 14,
        "scenario_count": 12,
        "allowed_actions": (0, 1, 3, 4, 5),
    },
    "planner_daily_expedite_7d": {
        "label": "Daily, air at most 1/7d",
        "replan_every": 1,
        "lookahead": 14,
        "scenario_count": 12,
        "expedite_cooldown": 7,
    },
    "planner_daily_h28": {
        "label": "Daily H=28",
        "replan_every": 1,
        "lookahead": 28,
        "scenario_count": 12,
    },
    "planner_daily_h56": {
        "label": "Daily H=56",
        "replan_every": 1,
        "lookahead": 56,
        "scenario_count": 12,
    },
    "planner_daily_h14_k48": {
        "label": "Daily H=14, K=48",
        "replan_every": 1,
        "lookahead": 14,
        "scenario_count": 48,
    },
}


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": round(float(values.mean()), 6),
        "sd": round(float(values.std(ddof=1)), 6),
        "se": round(float(values.std(ddof=1) / np.sqrt(len(values))), 6),
    }


def bootstrap_mean_ci(differences, seed, reps=BOOTSTRAP_REPS):
    differences = np.asarray(differences, dtype=float)
    rng = np.random.RandomState(seed)
    samples = rng.choice(differences, size=(reps, len(differences)), replace=True)
    means = samples.mean(axis=1)
    return [
        round(float(np.percentile(means, 2.5)), 6),
        round(float(np.percentile(means, 97.5)), 6),
    ]


def paired_comparison(first, second, seed):
    """Return paired statistics for first minus second."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    differences = first - second
    if np.allclose(differences, 0.0):
        p_value = 1.0
    else:
        result = wilcoxon(
            first, second, alternative="two-sided", zero_method="wilcox"
        )
        p_value = float(result.pvalue)
    sd = float(differences.std(ddof=1))
    return {
        "contrast": "first minus second",
        "mean_difference": round(float(differences.mean()), 6),
        "bootstrap_95_ci": bootstrap_mean_ci(differences, seed),
        "paired_wilcoxon_p": round(p_value, 10),
        "paired_dz": round(float(differences.mean() / sd), 6) if sd else 0.0,
        "fraction_first_greater": round(float(np.mean(differences > 0)), 6),
        "fraction_equal": round(float(np.mean(differences == 0)), 6),
    }


def evaluate_planner(planner, contexts):
    rows = []
    for episode_id, context in enumerate(contexts):
        env = SupplyChainEnv(
            context,
            planner.cal,
            np.random.RandomState(ENV_SEED_BASE + episode_id),
        )
        state = env.reset()
        planner.reset()
        action_counts = np.zeros(N_ACTIONS, dtype=int)
        expedite_noops = 0
        while True:
            action = planner.act(state, env=env)
            action_counts[action] += 1
            if action == 2:
                eligible = any(
                    shipment[0] > env.day + 2 and shipment[2] == 1.0
                    for shipment in env.pipeline
                )
                if not eligible:
                    expedite_noops += 1
            state, _, done, _ = env.step(action)
            if done:
                break
        metrics = env.episode_metrics()
        row = {
            "episode_id": episode_id,
            "scenario_type": context["scenario_type"],
            "recovery_days": metrics["recovery_days"],
            "recovered": metrics["recovered"],
            "total_cost": metrics["total_cost"],
            "service_cov": metrics["service_cov"],
            "mean_service": metrics["mean_service"],
            "air_expedites": int(action_counts[2]),
            "air_expedite_noops": int(expedite_noops),
        }
        for action_id in range(N_ACTIONS):
            row[f"action_{action_id}_count"] = int(action_counts[action_id])
        rows.append(row)
    return rows


def nondominated_set(aggregate, x_metric="total_cost", y_metric="recovery_days"):
    names = list(aggregate)
    nondominated = []
    dominated_by = {}
    for candidate in names:
        x_c = aggregate[candidate][x_metric]["mean"]
        y_c = aggregate[candidate][y_metric]["mean"]
        dominators = []
        for other in names:
            if other == candidate:
                continue
            x_o = aggregate[other][x_metric]["mean"]
            y_o = aggregate[other][y_metric]["mean"]
            if x_o <= x_c and y_o <= y_c and (x_o < x_c or y_o < y_c):
                dominators.append(other)
        dominated_by[candidate] = dominators
        if not dominators:
            nondominated.append(candidate)
    nondominated.sort(key=lambda name: aggregate[name][x_metric]["mean"])
    return nondominated, dominated_by


def plot_frontier(aggregate, labels, nondominated_recovery, nondominated_stability):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    colors = {"crl": "#0087B3", "rl_only": "#7B4397"}
    markers = {"crl": "o", "rl_only": "s"}

    planner_names = [name for name in aggregate if name.startswith("planner_")]
    for panel, metric, title, nondominated in [
        (axes[0], "recovery_days", "(a) Cost and recovery", nondominated_recovery),
        (axes[1], "service_cov", "(b) Cost and service stability", nondominated_stability),
    ]:
        for name in ["crl", "rl_only"]:
            panel.errorbar(
                aggregate[name]["total_cost"]["mean"] / 1000,
                aggregate[name][metric]["mean"],
                xerr=aggregate[name]["total_cost"]["se"] / 1000,
                yerr=aggregate[name][metric]["se"],
                fmt=markers[name],
                markersize=9,
                color=colors[name],
                ecolor=colors[name],
                capsize=3,
                label=labels[name],
                zorder=4,
            )
        for index, name in enumerate(planner_names):
            panel.errorbar(
                aggregate[name]["total_cost"]["mean"] / 1000,
                aggregate[name][metric]["mean"],
                xerr=aggregate[name]["total_cost"]["se"] / 1000,
                yerr=aggregate[name][metric]["se"],
                fmt="^",
                markersize=7,
                color="#C96B00",
                ecolor="#D99A5A",
                capsize=3,
                label="Lookahead planner configurations" if index == 0 else None,
                zorder=3,
            )
        envelope_x = [aggregate[name]["total_cost"]["mean"] / 1000 for name in nondominated]
        envelope_y = [aggregate[name][metric]["mean"] for name in nondominated]
        panel.plot(
            envelope_x,
            envelope_y,
            color="#555555",
            linewidth=1.6,
            linestyle="--",
            label="Observed nondominated set",
            zorder=2,
        )
        panel.set_title(title, loc="left")
        panel.set_xlabel("Mean episode cost (US$ thousands)")
        panel.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
        panel.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Mean recovery time (days)")
    axes[1].set_ylabel("Service-level volatility (CoV)")

    offsets = {
        "crl": (12, 8),
        "rl_only": (10, -18),
        "planner_weekly_h14": (10, -18),
        "planner_daily_h14": (10, 6),
        "planner_daily_no_expedite": (-105, 10),
        "planner_daily_expedite_7d": (10, 8),
        "planner_daily_h28": (10, 4),
        "planner_daily_h56": (10, -2),
        "planner_daily_h14_k48": (10, 6),
    }
    for name in aggregate:
        axes[0].annotate(
            labels[name],
            (aggregate[name]["total_cost"]["mean"] / 1000,
             aggregate[name]["recovery_days"]["mean"]),
            xytext=offsets.get(name, (8, 5)),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold" if name == "crl" else "normal",
            color="#222222",
        )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, legend_labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(), unique.keys(), loc="lower center", ncol=4,
        frameon=False, bbox_to_anchor=(0.5, -0.01)
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(f"{FIGURE_STEM}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Evaluate 12 episodes")
    args = parser.parse_args()
    n_eval = 12 if args.quick else N_EVAL

    if not PRIMARY_EPISODES.exists():
        raise FileNotFoundError(
            "Run paper_experiment/run_experiment.py before planner_frontier.py"
        )

    frozen = pd.read_csv(PRIMARY_EPISODES).sort_values("episode_id").iloc[:n_eval]
    if len(frozen) != n_eval:
        raise ValueError(f"Expected {n_eval} frozen episodes, found {len(frozen)}")

    calibration = Calibration(seed=SEED)
    contexts = calibration.sample_episode_contexts(
        n_eval, np.random.RandomState(EVAL_CONTEXT_SEED)
    )

    planner_results = {}
    for name, config in CONFIGS.items():
        kwargs = {key: value for key, value in config.items() if key != "label"}
        planner = StochasticOptAgent(calibration, seed=SEED, name=name, **kwargs)
        print(f"Evaluating {name}", flush=True)
        planner_results[name] = evaluate_planner(planner, contexts)

    long_rows = []
    for episode_id in range(n_eval):
        context = contexts[episode_id]
        for name in ("rl_only", "crl"):
            long_rows.append({
                "episode_id": episode_id,
                "scenario_type": context["scenario_type"],
                "configuration": name,
                "recovery_days": float(frozen.iloc[episode_id][f"{name}_recovery_days"]),
                "recovered": bool(frozen.iloc[episode_id].get(f"{name}_recovered", True)),
                "total_cost": float(frozen.iloc[episode_id][f"{name}_total_cost"]),
                "service_cov": float(frozen.iloc[episode_id][f"{name}_service_cov"]),
                "mean_service": float(frozen.iloc[episode_id][f"{name}_mean_service"]),
                "air_expedites": "",
                "air_expedite_noops": "",
                **{f"action_{action_id}_count": "" for action_id in range(N_ACTIONS)},
            })
        for name in CONFIGS:
            long_rows.append({"configuration": name, **planner_results[name][episode_id]})

    with OUTPUT_EPISODES.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(long_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(long_rows)

    frame = pd.DataFrame(long_rows)
    labels = {"crl": "CRL", "rl_only": "RL-only"}
    labels.update({name: config["label"] for name, config in CONFIGS.items()})
    metrics = [
        "recovery_days", "total_cost", "service_cov", "mean_service",
        "air_expedites", "air_expedite_noops",
    ]
    aggregate = {}
    for name in labels:
        subset = frame[frame["configuration"] == name]
        aggregate[name] = {}
        for metric in metrics:
            numeric = pd.to_numeric(subset[metric], errors="coerce").dropna()
            if len(numeric):
                aggregate[name][metric] = mean_se(numeric)

    comparisons = {}
    comparison_pairs = [
        (name, "crl") for name in CONFIGS
    ] + [
        ("planner_daily_h14_k48", "planner_daily_h14"),
        ("planner_daily_h28", "planner_daily_h14"),
        ("planner_daily_h56", "planner_daily_h28"),
        ("planner_daily_no_expedite", "planner_daily_h14"),
        ("planner_daily_expedite_7d", "planner_daily_h14"),
    ]
    for pair_index, (first_name, second_name) in enumerate(comparison_pairs):
        comparisons[f"{first_name}_minus_{second_name}"] = {}
        first = frame[frame["configuration"] == first_name].sort_values("episode_id")
        second = frame[frame["configuration"] == second_name].sort_values("episode_id")
        for metric_index, metric in enumerate(["recovery_days", "total_cost", "service_cov"]):
            comparisons[f"{first_name}_minus_{second_name}"][metric] = paired_comparison(
                first[metric], second[metric],
                BOOTSTRAP_SEED + pair_index * 10 + metric_index,
            )

    nondominated_recovery, dominated_recovery = nondominated_set(
        aggregate, "total_cost", "recovery_days"
    )
    nondominated_stability, dominated_stability = nondominated_set(
        aggregate, "total_cost", "service_cov"
    )

    scenario_breakdown = {}
    for scenario_type in ["single", "dual", "compound"]:
        scenario_breakdown[scenario_type] = {}
        for name in labels:
            subset = frame[
                (frame["configuration"] == name)
                & (frame["scenario_type"] == scenario_type)
            ]
            scenario_breakdown[scenario_type][name] = {
                metric: mean_se(subset[metric])
                for metric in ["recovery_days", "total_cost", "service_cov"]
            }

    summary = {
        "design": {
            "seed": SEED,
            "n_eval": n_eval,
            "evaluation_context_seed": EVAL_CONTEXT_SEED,
            "environment_seed_base": ENV_SEED_BASE,
            "bootstrap_repetitions": BOOTSTRAP_REPS,
            "common_random_numbers_across_actions": True,
            "planner_configurations": CONFIGS,
        },
        "labels": labels,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "scenario_breakdown": scenario_breakdown,
        "discrete_frontier": {
            "cost_recovery_nondominated": nondominated_recovery,
            "cost_recovery_dominated_by": dominated_recovery,
            "cost_stability_nondominated": nondominated_stability,
            "cost_stability_dominated_by": dominated_stability,
            "note": "Dominance is based on aggregate sample means in the tested discrete set.",
        },
    }
    with OUTPUT_SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)

    plot_frontier(
        aggregate, labels, nondominated_recovery, nondominated_stability
    )
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved {OUTPUT_EPISODES}")
    print(f"Saved {OUTPUT_SUMMARY}")
    print(f"Saved {FIGURE_STEM}.png, .pdf, and .svg")


if __name__ == "__main__":
    main()
