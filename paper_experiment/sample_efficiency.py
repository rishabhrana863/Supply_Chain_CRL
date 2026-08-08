"""
Sample-efficiency study: CRL vs RL-only across training budgets and seeds.
Tests the honest P1/P2 mechanism: does causal structure make learning more
efficient and more RELIABLE (across random seeds), rather than better at
convergence? Nothing is programmed toward either conclusion — identical
environments, budgets, and hyperparameters; only the causal layer differs.
"""

import json
import time
import numpy as np
from calibration import Calibration
from environment import SupplyChainEnv
from agents import PPOAgent, CRLAgent, fit_causal_model
from run_experiment import train

BUDGETS = [100, 300, 800]
SEEDS = [0, 1, 2, 3, 4]
N_EVAL = 60


def eval_agent(agent, cal, n=N_EVAL, seed_base=5555):
    rng = np.random.RandomState(seed_base)
    contexts = cal.sample_episode_contexts(n, rng)
    recs, costs = [], []
    for i, ctx in enumerate(contexts):
        env = SupplyChainEnv(ctx, cal, np.random.RandomState(seed_base * 3 + i))
        s = env.reset()
        while True:
            a, _, _ = agent.act(s, ctx, greedy=True)
            s, r, done, info = env.step(a)
            if done:
                break
        m = env.episode_metrics()
        recs.append(m["recovery_days"])
        costs.append(m["total_cost"])
    return float(np.mean(recs)), float(np.mean(costs))


def main():
    cal = Calibration(seed=42)
    cm = fit_causal_model(cal, n_rollouts=400, seed=42)
    results = {"budgets": BUDGETS, "seeds": SEEDS, "n_eval": N_EVAL, "cells": []}
    t0 = time.time()
    for budget in BUDGETS:
        for seed in SEEDS:
            for kind in ["rl_only", "crl"]:
                if kind == "rl_only":
                    ag = PPOAgent(state_dim=14, seed=seed)
                else:
                    ag = CRLAgent(state_dim=14, causal_model=cm, seed=seed)
                train(ag, cal, budget, seed_base=seed * 17 + 3, label=f"{kind}-b{budget}-s{seed}")
                rec, cost = eval_agent(ag, cal)
                results["cells"].append({"agent": kind, "budget": budget, "seed": seed,
                                         "recovery_days": round(rec, 2),
                                         "total_cost": round(cost, 1)})
                print(f"budget={budget:4d} seed={seed} {kind:8s} → recovery={rec:6.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    with open("results/sample_efficiency.json", "w") as f:
        json.dump(results, f, indent=2)

    # summary table
    print("\nbudget | RL mean±sd (5 seeds) | CRL mean±sd | Δ")
    for b in BUDGETS:
        rl = [c["recovery_days"] for c in results["cells"] if c["agent"] == "rl_only" and c["budget"] == b]
        cr = [c["recovery_days"] for c in results["cells"] if c["agent"] == "crl" and c["budget"] == b]
        print(f"{b:6d} | {np.mean(rl):6.2f}±{np.std(rl):5.2f} | {np.mean(cr):6.2f}±{np.std(cr):5.2f} "
              f"| {np.mean(rl)-np.mean(cr):+.2f}")


if __name__ == "__main__":
    main()
