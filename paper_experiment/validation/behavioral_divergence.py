"""Behavioral-divergence analysis for the prospective 30-seed replication.

WHAT THIS REPRODUCES
--------------------
Every number reported in Section 6.3 of the manuscript under "behavioral
divergence", plus the Table 5 aggregate-baseline column and the paired-difference
dispersion figures quoted in the abstract and introduction. It reads only the
public export

    paper_experiment/validation/replication_30seed_per_stream.csv.gz

so each of those numbers can be recomputed from the deposit alone. Values the
manuscript states are asserted at the end of the run; the script exits non-zero
if any of them drifts.

WHAT THIS IS NOT
----------------
It is NOT the training-exposure audit. Exposure -- the share of mechanically
feasible action slots removed by the causal mask, the share of decisions at
which at least one action was removed, the share of training environment steps
carrying a nonzero shaping term, and the size of the shaping contribution
relative to the base reward -- is a property of the TRAINING trajectories and
cannot be recovered from a deployment export. Those figures come from the
separate post hoc replay of the 30 guided-policy training runs; see
EXPOSURE_AUDIT.md in this folder.

The quantity computed here is the total variation distance between the two
converged agents' realized daily action frequencies within a model seed,

    d(s) = 0.5 * sum_k | P_ppo(a_k | s) - P_guided(a_k | s) |

bounded in [0, 1]. It is a behavioral distance, NOT a treatment dose. It
reflects both differences in action choice and differences in the states the
policies visited, and it is a property of the converged policies rather than of
the training signal. The natural reference scale is the same distance computed
between different seeds of the SAME agent, which measures how far ordinary
convergence variation moves the policy.

Rank correlations against seed-level paired outcome differences are reported
signed and in absolute value. The contrast between the two is the informative
part, and neither supports a causal reading: divergence is realized, not
assigned. The correlation with absolute intervention differences is close to
mechanical, because intervention rate is one minus the no-action frequency and
the total variation distance is bounded below by the absolute difference in that
frequency; it is printed for completeness and carries little independent
information.

Run:  python3 paper_experiment/validation/behavioral_divergence.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EXPORT = Path(__file__).resolve().parent / "replication_30seed_per_stream.csv.gz"
ACTIONS = [f"action_{k}_rate" for k in range(6)]
ACTION_NAMES = ["no action", "switch supplier", "air expedite",
                "reroute", "emergency procurement", "ration"]
PRIMARY_PROFILE = "aggregate_baseline"

# ─────────────────────────────────────────────────────────────────────────────
# Tolerances
#
# Every tolerance below is HALF AN ULP OF THE PRECISION THE MANUSCRIPT PRINTS.
# A figure written as 4.64 is checked to +/-0.005, one written as 0.0650 to
# +/-0.00005, one written as "$216 thousand" to +/-500. A tolerance looser than
# that would let a value drift far enough to change the printed digit without
# failing, which would make the check decorative. Where the manuscript states a
# BOUND rather than an estimate, the check is an inequality, not a window.
# ─────────────────────────────────────────────────────────────────────────────

#: Table 5, aggregate baseline: (standard PPO, causally guided PPO, tolerance).
#: All five outcomes; ten values.
PAPER_AGGREGATE = {
    "recovery_days": (13.95, 15.24, 5e-3),
    "total_cost": (787_258, 764_854, 0.5),
    "intervention_rate": (0.0650, 0.0581, 5e-5),
    "service_cov": (0.0550, 0.0592, 5e-5),
    "recovered": (0.9802, 0.9782, 5e-5),
}

#: Point values stated in the manuscript: (expected, tolerance).
PAPER_CLAIMS = {
    "median_paired_divergence": (0.0099, 5e-5),
    "median_seed_divergence": (0.0297, 5e-5),
    "share_below_seed_median_pct": (77.0, 0.5),
    "air_expedite_ppo_pct": (4.64, 5e-3),
    "air_expedite_guided_pct": (4.00, 5e-3),
    "rho_abs_cost": (0.714, 5e-4),
    "rho_abs_intervention": (0.965, 5e-4),
    "rho_abs_recovery": (0.596, 5e-4),
    "rho_signed_cost": (0.219, 5e-4),
    "rho_signed_intervention": (0.320, 5e-4),
    "rho_signed_recovery": (-0.060, 5e-4),
    "high_low_abs_cost_ratio": (4.1, 0.05),
    "abs_cost_high": (230_000, 500),
    "abs_cost_low": (56_000, 500),
    "paired_cost_mean": (22_400, 50),
    "paired_cost_sd": (216_000, 500),
    "paired_cost_min": (-624_000, 500),
    "paired_cost_max": (558_000, 500),
}

#: Bounds stated in the manuscript: (limit, direction). "at most" means the
#: observed value must not exceed the limit.
PAPER_BOUNDS = {
    # Section 6.3: "no other action class differed by more than 0.05 pp".
    # The largest non-air, non-no-action gap is rerouting at 0.0436 pp.
    "max_other_action_gap_pp": (0.05, "at most"),
}


def load_cells(path: Path = EXPORT):
    """Average the three deployment streams into model-episode-agent cells.

    This is the manuscript's analysis unit. Averaging first keeps the stream
    randomness inside the cell rather than letting it masquerade as extra
    independent observations.
    """
    df = pd.read_csv(path)
    keep = {**{a: "mean" for a in ACTIONS},
            "recovery_days": "mean", "total_cost": "mean",
            "intervention_rate": "mean", "service_cov": "mean",
            "episode_reward": "mean", "recovered": "mean",
            "infeasible_selected": "sum"}
    # "recovered" averaged over episodes is the recovery probability in Table 5.
    cells = df.groupby(["profile", "model_seed", "episode_id", "agent"],
                       as_index=False).agg(keep)
    return df, cells


def total_variation(p, q) -> float:
    return float(0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def reproduce_headline(agg: pd.DataFrame, claims: dict) -> None:
    print("REPRODUCTION CHECK -- manuscript Table 5, aggregate baseline "
          "(all five outcomes)\n")
    print(f"  {'outcome':<22}{'PPO':>14}{'guided':>14}"
          f"{'paper PPO':>14}{'paper guided':>14}{'tol':>10}")
    for metric, (pp, pg, tol) in PAPER_AGGREGATE.items():
        a = agg.loc[agg.agent == "ppo", metric].mean()
        b = agg.loc[agg.agent == "crl", metric].mean()
        fmt = (lambda x: f"{x:,.0f}") if metric == "total_cost" else (lambda x: f"{x:.4f}")
        label = "recovery_prob" if metric == "recovered" else metric
        print(f"  {label:<22}{fmt(a):>14}{fmt(b):>14}{fmt(pp):>14}{fmt(pg):>14}"
              f"{tol:>10.5g}")
        claims[f"table5_{label}_ppo"] = (a, pp, tol)
        claims[f"table5_{label}_guided"] = (b, pg, tol)
    print()


def seed_level(agg: pd.DataFrame):
    """Return per-seed action profiles, divergences and paired outcome gaps."""
    profile = agg.groupby(["model_seed", "agent"])[ACTIONS].mean()
    means = agg.groupby(["model_seed", "agent"])[
        ["total_cost", "intervention_rate", "recovery_days"]].mean()
    seeds = sorted(agg.model_seed.unique())

    divergence = np.array([total_variation(profile.loc[(s, "ppo")], profile.loc[(s, "crl")])
                           for s in seeds])
    diffs = {
        "cost": np.array([means.loc[(s, "ppo"), "total_cost"]
                          - means.loc[(s, "crl"), "total_cost"] for s in seeds]),
        "intervention_pp": np.array([100 * (means.loc[(s, "ppo"), "intervention_rate"]
                                            - means.loc[(s, "crl"), "intervention_rate"])
                                     for s in seeds]),
        "recovery_days": np.array([means.loc[(s, "ppo"), "recovery_days"]
                                   - means.loc[(s, "crl"), "recovery_days"] for s in seeds]),
    }
    return seeds, divergence, diffs, profile


def main() -> int:
    observed: dict[str, float] = {}
    checks: dict[str, tuple] = {}

    df, cells = load_cells()
    agg = cells[cells.profile == PRIMARY_PROFILE]

    print(f"\nexport: {len(df):,} stream rows -> {len(cells):,} model-episode-agent cells")
    print(f"seeds {df.model_seed.min()}-{df.model_seed.max()} ({df.model_seed.nunique()}), "
          f"profiles {df.profile.nunique()}, streams/cell {df.policy_stream.nunique()}")
    print(f"infeasible actions selected anywhere in the export: "
          f"{int(df.infeasible_selected.sum())}\n")

    reproduce_headline(agg, checks)

    seeds, divergence, diffs, profile = seed_level(agg)

    # ---- 1. Divergence against its own reference scale -----------------------
    reference = []
    for agent in ("ppo", "crl"):
        reference += [total_variation(profile.loc[(a, agent)], profile.loc[(b, agent)])
                      for a, b in itertools.combinations(seeds, 2)]

    observed["median_paired_divergence"] = float(np.median(divergence))
    observed["median_seed_divergence"] = float(np.median(reference))
    observed["share_below_seed_median_pct"] = float(
        100 * np.mean(divergence < np.median(reference)))

    print("1. BEHAVIORAL DIVERGENCE -- paired contrast against the seed contrast\n")
    print(f"  paired contrast (guided vs PPO, same seed, n={len(divergence)})")
    print(f"     median {np.median(divergence):.4f}   mean {divergence.mean():.4f}   "
          f"range [{divergence.min():.4f}, {divergence.max():.4f}]")
    print(f"  seed contrast (same agent, different seed, n={len(reference)})")
    print(f"     median {np.median(reference):.4f}   mean {np.mean(reference):.4f}   "
          f"range [{min(reference):.4f}, {max(reference):.4f}]")
    print(f"\n  seed / paired, medians: "
          f"{np.median(reference) / np.median(divergence):.2f}x")
    print(f"  {observed['share_below_seed_median_pct']:.0f}% of paired contrasts fall below "
          f"the median seed contrast")
    print("  -> the paired policies differ by less than seeds of the same agent do.\n")

    # ---- 2. Where the displacement lands ------------------------------------
    print("2. WHERE THE DISPLACEMENT LANDS -- mean action probability by agent\n")
    by_agent = agg.groupby("agent")[ACTIONS].mean()
    print(f"  {'action':<24}{'PPO':>10}{'guided':>10}{'diff (pp)':>12}")
    gaps = {}
    for name, col in zip(ACTION_NAMES, ACTIONS):
        p, g = by_agent.loc["ppo", col], by_agent.loc["crl", col]
        gaps[name] = 100 * (p - g)
        print(f"  {name:<24}{100*p:>9.2f}%{100*g:>9.2f}%{100*(p-g):>11.2f}")
    observed["air_expedite_ppo_pct"] = float(100 * by_agent.loc["ppo", "action_2_rate"])
    observed["air_expedite_guided_pct"] = float(100 * by_agent.loc["crl", "action_2_rate"])
    observed["max_other_action_gap_pp"] = float(
        max(abs(v) for k, v in gaps.items() if k not in ("air expedite", "no action")))
    print()

    # ---- 3. Signed against absolute -----------------------------------------
    print("3. DIVERGENCE AND OUTCOMES -- exploratory, and not a causal contrast\n")
    print(f"  {'outcome difference':<22}{'signed rho':>14}{'p':>8}{'|abs| rho':>14}{'p':>8}")
    key = {"cost": "cost", "intervention_pp": "intervention", "recovery_days": "recovery"}
    for label, y in diffs.items():
        rs, ps = stats.spearmanr(divergence, y)
        ra, pa = stats.spearmanr(divergence, np.abs(y))
        observed[f"rho_signed_{key[label]}"] = float(rs)
        observed[f"rho_abs_{key[label]}"] = float(ra)
        print(f"  {label:<22}{rs:>+14.3f}{ps:>8.3f}{ra:>+14.3f}{pa:>8.3f}")

    lo = divergence < np.median(divergence)
    hi = ~lo
    observed["abs_cost_low"] = float(np.abs(diffs["cost"][lo]).mean())
    observed["abs_cost_high"] = float(np.abs(diffs["cost"][hi]).mean())
    observed["high_low_abs_cost_ratio"] = observed["abs_cost_high"] / observed["abs_cost_low"]

    print("\n  split at the median divergence:")
    print(f"  {'':<18}{'n':>4}{'mean cost diff':>18}{'mean |cost diff|':>20}")
    print(f"  {'below median':<18}{lo.sum():>4}{diffs['cost'][lo].mean():>17,.0f}"
          f"{observed['abs_cost_low']:>19,.0f}")
    print(f"  {'above median':<18}{hi.sum():>4}{diffs['cost'][hi].mean():>17,.0f}"
          f"{observed['abs_cost_high']:>19,.0f}")
    print(f"\n  seeds above the median show {observed['high_low_abs_cost_ratio']:.1f}x the "
          f"absolute cost divergence of\n  seeds below it, with no reliable shift in sign.\n")

    # ---- 4. Dispersion, the quantity the paper leads with --------------------
    print("4. DISPERSION -- the paired difference and its spread\n")
    cost = diffs["cost"]
    observed["paired_cost_mean"] = float(cost.mean())
    observed["paired_cost_sd"] = float(cost.std(ddof=1))
    observed["paired_cost_min"] = float(cost.min())
    observed["paired_cost_max"] = float(cost.max())
    print(f"  paired cost difference (PPO - guided), n=30")
    print(f"     mean ${cost.mean():,.0f}   SD ${cost.std(ddof=1):,.0f}   "
          f"range [${cost.min():,.0f}, ${cost.max():,.0f}]")
    print("     This SD is the quantity quoted in the abstract and introduction: the")
    print("     spread of the PAIRED difference, computed within seed.\n")

    print(f"  {'outcome':<20}{'between-seed SD':>18}{'paired mean diff':>20}{'ratio':>8}")
    spec = (("total_cost", "total cost", lambda x: f"${x:,.0f}"),
            ("intervention_rate", "intervention rate", lambda x: f"{100*x:.2f} pp"),
            ("recovery_days", "recovery days", lambda x: f"{x:.2f} d"))
    for metric, label, fmt in spec:
        wide = agg.groupby(["model_seed", "agent"])[metric].mean().unstack()
        between = float(np.std(wide.mean(axis=1), ddof=1))
        effect = float(np.mean(wide["ppo"] - wide["crl"]))
        print(f"  {label:<20}{fmt(between):>18}{fmt(effect):>20}{between/abs(effect):>7.1f}x")
    print("     The between-seed SD column describes the level of the outcome across")
    print("     seeds and is a different quantity from the paired SD above. The")
    print("     manuscript quotes the paired SD.\n")

    # ---- 5. Assertions -------------------------------------------------------
    for name, (expected, tol) in PAPER_CLAIMS.items():
        checks[name] = (observed[name], expected, tol)

    n_checks = len(checks) + len(PAPER_BOUNDS)
    print(f"5. AGREEMENT WITH THE MANUSCRIPT -- {n_checks} stated values checked,")
    print("   each against half an ulp of the precision the manuscript prints\n")

    failures = [(k, got, exp, tol) for k, (got, exp, tol) in checks.items()
                if not abs(got - exp) <= tol]
    for k, (limit, direction) in PAPER_BOUNDS.items():
        got = observed[k]
        ok = got <= limit if direction == "at most" else got >= limit
        print(f"  bound     {k}: {got:.4g} ({direction} {limit:g}) "
              f"{'ok' if ok else 'VIOLATED'}")
        if not ok:
            failures.append((k, got, limit, 0.0))
    print()

    if failures:
        for k, got, exp, tol in failures:
            print(f"  MISMATCH  {k}: recomputed {got:.6g}, manuscript {exp:.6g} "
                  f"(tolerance {tol:.6g})")
        print("\n  This export does not reproduce the reported values.\n")
        return 1
    print("  all values reproduce within tolerance.\n")

    print("SCOPE")
    print("  Behavioral divergence is realized, not assigned: it is a property of the")
    print("  converged policies. The signed-against-absolute contrast is descriptive")
    print("  and was not prespecified. It belongs with the exploratory mechanism")
    print("  diagnostics, not the confirmatory outcomes. Training exposure is a")
    print("  separate quantity and is not computable from this export; see")
    print("  EXPOSURE_AUDIT.md.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
