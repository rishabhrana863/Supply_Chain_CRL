"""
Experiment runner: trains PPO and CRL, fits the causal model from randomized
rollouts, evaluates all three agents on IDENTICAL held-out episodes
(same contexts, same environment seeds — fully paired design), and freezes
results to JSON + per-episode CSV.

Usage: python3 run_experiment.py [--quick]
"""

import sys
import json
import csv
import time
import numpy as np
from pathlib import Path

from calibration import Calibration
from environment import SupplyChainEnv, HORIZON, DISRUPTION_ONSET, N_ACTIONS
from agents import PPOAgent, CRLAgent, StochasticOptAgent, fit_causal_model, MLP

SEED = 42
QUICK = "--quick" in sys.argv
N_TRAIN = 60 if QUICK else 1500
N_EVAL = 30 if QUICK else 200
N_CAUSAL = 60 if QUICK else 400
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def train(agent, cal, n_episodes, seed_base, label):
    rng = np.random.RandomState(seed_base)
    contexts = cal.sample_episode_contexts(n_episodes, rng)
    t0 = time.time()
    rewards = []
    for i, ctx in enumerate(contexts):
        env = SupplyChainEnv(ctx, cal, np.random.RandomState(seed_base * 100000 + i))
        s = env.reset()
        ep_r = 0.0
        while True:
            a, logp, v = agent.act(s, ctx)
            s2, r, done, info = env.step(a)
            r_train = agent.shaped(r, s, a, ctx)
            agent.store(s, a, logp, r_train, v, done)
            s = s2
            ep_r += r
            if done:
                break
        rewards.append(ep_r)
        if (i + 1) % 8 == 0:
            agent.update()
        if (i + 1) % 250 == 0:
            print(f"  [{label}] ep {i+1}/{n_episodes} avg_r(100)={np.mean(rewards[-100:]):.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    agent.update()
    return rewards


def evaluate(agents_dict, cal, n_episodes, seed_base=9000):
    """Paired evaluation: every agent sees the same contexts AND same env seeds."""
    rng = np.random.RandomState(seed_base)
    contexts = cal.sample_episode_contexts(n_episodes, rng)
    env_seeds = [seed_base * 7 + i for i in range(n_episodes)]
    rows = []
    for i, ctx in enumerate(contexts):
        row = {"episode_id": i, "country": ctx["country"], "commodity": ctx["commodity"],
               "lpi": round(ctx["lpi"], 3), "scenario_type": ctx["scenario_type"],
               "total_severity": ctx["total_severity"],
               "disruption_types": "+".join(sorted(ctx["disruption_types"]))}
        for name, agent in agents_dict.items():
            env = SupplyChainEnv(ctx, cal, np.random.RandomState(env_seeds[i]))
            s = env.reset()
            if hasattr(agent, "reset"):
                agent.reset()
            actions_taken = []
            while True:
                if isinstance(agent, StochasticOptAgent):
                    a = agent.act(s, env=env)
                else:
                    a, _, _ = agent.act(s, ctx, greedy=True)
                actions_taken.append(a)
                s, r, done, info = env.step(a)
                if done:
                    break
            m = env.episode_metrics()
            row[f"{name}_recovery_days"] = round(m["recovery_days"], 2)
            row[f"{name}_recovered"] = m["recovered"]
            row[f"{name}_dipped"] = m["dipped"]
            row[f"{name}_total_cost"] = round(m["total_cost"], 2)
            row[f"{name}_service_cov"] = round(m["service_cov"], 4)
            row[f"{name}_mean_service"] = round(m["mean_service"], 4)
            row[f"{name}_min_service"] = round(m["min_service"], 4)
            row[f"{name}_switch_used"] = int(any(a == 1 for a in actions_taken))
            row[f"{name}_actions"] = "".join(map(str, actions_taken[:40]))
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  eval {i+1}/{n_episodes}", flush=True)
    return rows


# ── statistics (numpy-only, same as earlier fixed script) ──

def cohens_d(a, b):
    ps = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return float((np.mean(a) - np.mean(b)) / ps) if ps > 0 else 0.0


def wilcoxon(a, b):
    import math
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0, 1.0
    ranks = np.argsort(np.argsort(np.abs(diff))) + 1.0
    w = min(ranks[diff > 0].sum(), ranks[diff < 0].sum())
    mu = n * (n + 1) / 4
    sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sig
    p = math.erfc(-z / math.sqrt(2)) / 2
    return float(w), float(p)


def summarize(rows, agents, key):
    out = {}
    for name in agents:
        vals = np.array([r[f"{name}_{key}"] for r in rows], dtype=float)
        out[name] = {"mean": round(float(vals.mean()), 3),
                     "sd": round(float(vals.std(ddof=1)), 3),
                     "se": round(float(vals.std(ddof=1) / np.sqrt(len(vals))), 3)}
    return out


def explainability_index(agent, cm, cal, n_episodes=40, seed=777, eps=0.01):
    """Causal Alignment Index (redesigned P4, CRL vs RL-only):
    EI = fraction of in-disruption decisions whose chosen action's estimated
    context-conditional ATE is within eps of the best available action's ATE.
    A passive policy scores low whenever an active intervention has a clearly
    higher estimated causal effect. Formal, reproducible; optimization model
    excluded (its transparency is of a different, algebraic kind)."""
    rng = np.random.RandomState(seed)
    contexts = cal.sample_episode_contexts(n_episodes, rng)
    aligned, total = 0, 0
    for i, ctx in enumerate(contexts):
        env = SupplyChainEnv(ctx, cal, np.random.RandomState(seed + i))
        s = env.reset()
        while True:
            a, _, _ = agent.act(s, ctx, greedy=True)
            if env.day > DISRUPTION_ONSET and env.severity_now > 0.5:
                total += 1
                st = cm.stratum(s)
                ates = np.array([0.0] + [cm.ate(st, x) for x in range(1, N_ACTIONS)])
                chosen = ates[a]
                if chosen >= ates.max() - eps:
                    aligned += 1
            s, r, done, info = env.step(a)
            if done:
                break
    return round(aligned / max(total, 1), 4)


def main():
    np.random.seed(SEED)
    print("=" * 70)
    print(f"CRL EXPERIMENT — seed={SEED} quick={QUICK}")
    print(f"  train={N_TRAIN} eval={N_EVAL} causal_rollouts={N_CAUSAL}")
    print("=" * 70, flush=True)

    cal = Calibration(seed=SEED)
    print("calibration:", json.dumps(cal.summary())[:400], flush=True)

    print("\n[1/4] Fitting causal model from randomized interventional rollouts...", flush=True)
    t0 = time.time()
    cm = fit_causal_model(cal, n_rollouts=N_CAUSAL, seed=SEED)
    print(f"  strata learned: {len(cm.table)} ({time.time()-t0:.0f}s)", flush=True)

    N_SEEDS = 1 if QUICK else 5
    print(f"\n[2/4] Training RL-only (PPO) × {N_SEEDS} seeds...", flush=True)
    rls = []
    for sd in range(N_SEEDS):
        a = PPOAgent(state_dim=14, seed=SEED + sd)
        train(a, cal, N_TRAIN, seed_base=SEED + 1 + sd, label=f"rl-s{sd}")
        rls.append(a)

    print(f"\n[3/4] Training CRL (PPO + causal mask/shaping) × {N_SEEDS} seeds...", flush=True)
    crls = []
    for sd in range(N_SEEDS):
        a = CRLAgent(state_dim=14, causal_model=cm, seed=SEED + sd)
        train(a, cal, N_TRAIN, seed_base=SEED + 1 + sd, label=f"crl-s{sd}")  # same contexts as RL
        crls.append(a)

    print("\n[4/4] Paired evaluation on held-out episodes (all seeds)...", flush=True)
    so = StochasticOptAgent(cal, seed=SEED)
    agents = {"stochastic_opt": so}
    for sd in range(N_SEEDS):
        agents[f"rl_s{sd}"] = rls[sd]
        agents[f"crl_s{sd}"] = crls[sd]
    rows = evaluate(agents, cal, N_EVAL)
    # seed-averaged episode-level values become the primary comparison unit
    for r in rows:
        for base in ["recovery_days", "total_cost", "service_cov", "mean_service",
                     "min_service", "switch_used"]:
            r[f"rl_only_{base}"] = round(float(np.mean(
                [r[f"rl_s{sd}_{base}"] for sd in range(N_SEEDS)])), 4)
            r[f"crl_{base}"] = round(float(np.mean(
                [r[f"crl_s{sd}_{base}"] for sd in range(N_SEEDS)])), 4)
    rl, crl = rls[0], crls[0]  # representatives for EI computation below

    # ── aggregate + tests (on seed-averaged episode values) ──
    names = ["stochastic_opt", "rl_only", "crl"]
    results = {"seed": SEED, "n_eval": N_EVAL, "n_train": N_TRAIN,
               "aggregate": {k: summarize(rows, names, k) for k in
                             ["recovery_days", "total_cost", "service_cov", "mean_service"]}}

    rec = {n: np.array([r[f"{n}_recovery_days"] for r in rows], float) for n in names}
    cost = {n: np.array([r[f"{n}_total_cost"] for r in rows], float) for n in names}
    cov = {n: np.array([r[f"{n}_service_cov"] for r in rows], float) for n in names}
    tests = {}
    for a, b in [("crl", "stochastic_opt"), ("crl", "rl_only"), ("rl_only", "stochastic_opt")]:
        for metric, data in [("recovery", rec), ("cost", cost), ("service_cov", cov)]:
            w, p = wilcoxon(data[b], data[a])
            tests[f"{a}_vs_{b}_{metric}"] = {
                "p": round(p, 6), "d": round(cohens_d(data[b], data[a]), 3),
                "mean_diff": round(float(data[b].mean() - data[a].mean()), 3)}
    results["tests"] = tests

    # scenario breakdown (P2)
    sb = {}
    for st in ["single", "dual", "compound"]:
        idx = [i for i, r in enumerate(rows) if r["scenario_type"] == st]
        if not idx:
            continue
        sb[st] = {"n": len(idx)}
        for n in names:
            v = np.array([rows[i][f"{n}_recovery_days"] for i in idx])
            sb[st][n] = {"mean": round(float(v.mean()), 2), "sd": round(float(v.std(ddof=1)), 2)}
        sb[st]["crl_minus_rl"] = round(sb[st]["rl_only"]["mean"] - sb[st]["crl"]["mean"], 2)
    results["scenario_breakdown"] = sb

    # infrastructure analysis (P3) — LPI median split; effect must EMERGE
    lpis = np.array([r["lpi"] for r in rows])
    med = float(np.median(lpis))
    p3 = {}
    for grp, sel in [("low_lpi", lpis < med), ("high_lpi", lpis >= med)]:
        idx = np.where(sel)[0]
        p3[grp] = {"n": int(len(idx)), "median_split_at": round(med, 3)}
        for n in ["crl", "rl_only", "stochastic_opt"]:
            v = np.array([rows[i][f"{n}_recovery_days"] for i in idx])
            p3[grp][n + "_recovery"] = round(float(v.mean()), 2)
        sw = np.array([rows[i]["crl_switch_used"] for i in idx])
        p3[grp]["crl_switch_rate"] = round(float(sw.mean()), 3)
    results["infrastructure_analysis"] = p3

    # explainability (redesigned P4: CRL vs RL-only), averaged over seeds
    print("\ncomputing explainability index...", flush=True)
    results["explainability_index"] = {
        "crl": round(float(np.mean([explainability_index(a, cm, cal) for a in crls])), 4),
        "rl_only": round(float(np.mean([explainability_index(a, cm, cal) for a in rls])), 4),
        "definition": "fraction of in-disruption decisions whose chosen action has "
                      "non-negative estimated context-conditional ATE (causally justifiable decisions)"}

    # ── freeze ──
    with open(OUT / "frozen_results.json", "w") as f:
        json.dump(results, f, indent=2)
    csv_keys = list(rows[0].keys())
    with open(OUT / "per_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys)
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 70)
    print(json.dumps(results["aggregate"], indent=1))
    print("\nTESTS:", json.dumps(tests, indent=1))
    print("\nSCENARIOS:", json.dumps(sb, indent=1))
    print("\nP3:", json.dumps(p3, indent=1))
    print("\nEI:", json.dumps(results["explainability_index"], indent=1))
    print(f"\n✅ frozen → {OUT}/frozen_results.json, per_episode.csv")


if __name__ == "__main__":
    main()
