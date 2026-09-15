"""Recompute the FDA benchmark headline figures from the workbook's raw episode rows.

Reproduces the values reported in FDA_Recovery_Time_Validation_Report.md directly
from the "FDA Product Episodes" sheet, so the summary sheets are never the only
record of how a number was obtained.

Usage:  python verify_fda_benchmark.py [path/to/FDA_Recovery_Time_Validation.xlsx]
"""
import sys
import random
from pathlib import Path

import openpyxl

WORKBOOK = Path(__file__).with_name("FDA_Recovery_Time_Validation.xlsx")
HORIZON = 110


def load_episodes(path):
    rows = list(openpyxl.load_workbook(path, data_only=True)["FDA Product Episodes"].values)
    header = [str(c).strip() if c else "" for c in rows[3]]
    idx = {name: i for i, name in enumerate(header)}
    episodes = []
    for row in rows[4:]:
        if not row or not row[0]:
            continue
        try:
            duration = float(row[idx["Duration days"]])
        except (TypeError, ValueError):
            continue
        episodes.append((
            duration,
            str(row[idx["Event observed"]]).strip().lower() == "true",
            str(row[idx["Generic name"]]),
        ))
    return episodes


def survival(episodes, horizon=None):
    """Kaplan-Meier survival, optionally truncated at a horizon."""
    s = 1.0
    curve = []
    for t in sorted({d for d, _, _ in episodes}):
        if horizon is not None and t > horizon:
            break
        at_risk = sum(1 for d, _, _ in episodes if d >= t)
        events = sum(1 for d, e, _ in episodes if d == t and e)
        if at_risk and events:
            s *= 1 - events / at_risk
        curve.append((t, s))
    return curve


def median(curve):
    return next((t for t, s in curve if s <= 0.5), None)


def recovery_by(curve, day):
    return (1 - next((s for t, s in reversed(curve) if t <= day), 1.0)) * 100


def cluster_bootstrap(episodes, day, draws=600, seed=0):
    """Drug-clustered bootstrap CI: NDCs of the same drug are not independent."""
    clusters = {}
    for ep in episodes:
        clusters.setdefault(ep[2], []).append(ep)
    keys = list(clusters)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = []
        for key in rng.choices(keys, k=len(keys)):
            sample += clusters[key]
        estimates.append(recovery_by(survival(sample, day), day))
    estimates.sort()
    lo = estimates[int(0.025 * draws)]
    hi = estimates[int(0.975 * draws) - 1]
    return lo, hi, len(keys)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else WORKBOOK
    episodes = load_episodes(path)
    curve = survival(episodes)
    events = sum(1 for _, e, _ in episodes if e)
    med = median(curve)

    print(f"Product episodes        : {len(episodes)}")
    print(f"Resolved / censored     : {events} / {len(episodes) - events}")
    print(f"Kaplan-Meier median     : {med:.0f} days ({med / 365.25:.2f} years)")
    print(f"Recovery by day {HORIZON}     : {recovery_by(curve, HORIZON):.2f}%")
    print(f"Unresolved at one year  : {next(s for t, s in reversed(curve) if t <= 365) * 100:.2f}%")

    lo, hi, n_clusters = cluster_bootstrap(episodes, HORIZON)
    print(f"\nDrug clusters           : {n_clusters} distinct generic names")
    print(f"Cluster-robust 95% CI   : {lo:.2f}% - {hi:.2f}%  (day-{HORIZON} recovery)")
    print("The naive binomial CI treats co-listed NDCs of one drug as independent")
    print("and is correspondingly too narrow; the timescale gap is unaffected.")


if __name__ == "__main__":
    main()
