"""
Robustness suite — produces the actual evidence behind Section 5.2:
  A) Ablation: CRL-full vs mask-only vs shaping-only vs RL-only (3 seeds)
  B) Recovery-threshold sensitivity: 90% / 95% / 98% (same trajectories)
  C) Out-of-distribution stress test: 300 forced compound high-severity
     scenarios (severity 4-5 on three concurrent types)
Outputs: results/robustness_results.json
"""

import json
import time
import numpy as np
from calibration import Calibration
from environment import SupplyChainEnv
from agents import PPOAgent, CRLAgent, StochasticOptAgent, fit_causal_model
from run_experiment import train, wilcoxon, cohens_d

SEEDS = [0, 1, 2]
N_TRAIN = 1500
N_EVAL = 150
OUT = "results/robustness_results.json"


def eval_multi(agent_lists, cal, contexts, seed_base, thresholds=(0.90, 0.95, 0.98)):
    """agent_lists: {name: [agents across seeds]} → seed-averaged per-episode metrics
    at each recovery threshold, computed from the SAME trajectories."""
    out = {name: {th: {"rec": [], "cost": []} for th in thresholds} for name in agent_lists}
    for i, ctx in enumerate(contexts):
        for name, agents in agent_lists.items():
            recs = {th: [] for th in thresholds}
            costs = []
            for agent in agents:
                env = SupplyChainEnv(ctx, cal, np.random.RandomState(seed_base + i))
                s = env.reset()
                if hasattr(agent, "reset"):
                    agent.reset()
                while True:
                    if isinstance(agent, StochasticOptAgent):
                        a = agent.act(s, env=env)
                    else:
                        a, _, _ = agent.act(s, ctx, greedy=True)
                    s, r, done, info = env.step(a)
                    if done:
                        break
                for th in thresholds:
                    recs[th].append(env.episode_metrics(threshold=th)["recovery_days"])
                costs.append(env.episode_metrics()["total_cost"])
            for th in thresholds:
                out[name][th]["rec"].append(float(np.mean(recs[th])))
            cost_th = thresholds[1] if len(thresholds) > 1 else thresholds[0]
            out[name][cost_th]["cost"].append(float(np.mean(costs)))
    return out


def main():
    t0 = time.time()
    cal = Calibration(seed=42)
    cm = fit_causal_model(cal, n_rollouts=400, seed=42)

    # ── train ablation variants (3 seeds each) ──
    variants = {}
    for name, mask, shape in [("crl_full", True, True), ("crl_mask_only", True, False),
                              ("crl_shape_only", False, True), ("rl_only", False, False)]:
        agents = []
        for sd in SEEDS:
            if mask or shape:
                a = CRLAgent(state_dim=14, causal_model=cm, seed=42 + sd,
                             causal_lambda=0.15 if shape else 0.0)
                if not mask:
                    a.mask = lambda s, c: np.ones(6)  # disable masking
            else:
                a = PPOAgent(state_dim=14, seed=42 + sd)
            train(a, cal, N_TRAIN, seed_base=43 + sd, label=f"{name}-s{sd}")
            agents.append(a)
        variants[name] = agents
        print(f"trained {name} ({time.time()-t0:.0f}s)", flush=True)

    so = StochasticOptAgent(cal, seed=42)
    variants["stochastic_opt"] = [so]

    # ── A+B: ablation & threshold sensitivity on held-out episodes ──
    rng = np.random.RandomState(9000)
    contexts = cal.sample_episode_contexts(N_EVAL, rng)
    res = eval_multi(variants, cal, contexts, seed_base=63000)
    ablation = {}
    threshold_sens = {}
    for name in variants:
        ablation[name] = {
            "recovery_mean": round(float(np.mean(res[name][0.95]["rec"])), 2),
            "recovery_se": round(float(np.std(res[name][0.95]["rec"], ddof=1) / np.sqrt(N_EVAL)), 2),
            "cost_mean": round(float(np.mean(res[name][0.95]["cost"])), 1),
        }
        threshold_sens[name] = {str(th): round(float(np.mean(res[name][th]["rec"])), 2)
                                for th in (0.90, 0.95, 0.98)}
    # ranking preserved check
    order = {th: sorted(["crl_full", "rl_only", "stochastic_opt"],
                        key=lambda n: threshold_sens[n][str(th)])
             for th in (0.90, 0.95, 0.98)}
    rank_preserved = all(order[th] == order[0.95] for th in (0.90, 0.98))

    # significance for ablation deltas vs crl_full
    abl_tests = {}
    base_rec = np.array(res["crl_full"][0.95]["rec"])
    for name in ["crl_mask_only", "crl_shape_only", "rl_only", "stochastic_opt"]:
        v = np.array(res[name][0.95]["rec"])
        w, p = wilcoxon(v, base_rec)
        abl_tests[name] = {"p": round(p, 6), "d": round(cohens_d(v, base_rec), 3),
                           "delta_days": round(float(v.mean() - base_rec.mean()), 2)}

    # ── C: OOD stress test — forced compound high-severity scenarios ──
    print(f"stress test... ({time.time()-t0:.0f}s)", flush=True)
    rng2 = np.random.RandomState(1234)
    stress_ctx = cal.sample_episode_contexts(300, rng2)
    all_types = list(cal.disruption_type_probs.keys())
    for c in stress_ctx:  # force compound, high severity (beyond training distribution mix)
        c["disruption_types"] = all_types[:3]
        c["severities"] = {t: int(rng2.choice([4, 5])) for t in c["disruption_types"]}
        c["total_severity"] = sum(c["severities"].values())
        c["scenario_type"] = "compound"
    stress_agents = {"crl_full": variants["crl_full"], "rl_only": variants["rl_only"],
                     "stochastic_opt": [so]}
    sres = eval_multi(stress_agents, cal, stress_ctx, seed_base=77000, thresholds=(0.95,))
    stress = {}
    for name in stress_agents:
        v = np.array(sres[name][0.95]["rec"])
        stress[name] = {"recovery_mean": round(float(v.mean()), 2),
                        "recovery_sd": round(float(v.std(ddof=1)), 2),
                        "recovery_se": round(float(v.std(ddof=1) / np.sqrt(len(v))), 2)}
    w, p = wilcoxon(np.array(sres["rl_only"][0.95]["rec"]),
                    np.array(sres["crl_full"][0.95]["rec"]))
    stress["crl_vs_rl_p"] = round(p, 6)
    w, p = wilcoxon(np.array(sres["stochastic_opt"][0.95]["rec"]),
                    np.array(sres["crl_full"][0.95]["rec"]))
    stress["crl_vs_so_p"] = round(p, 6)

    results = {"n_eval": N_EVAL, "seeds": SEEDS, "n_train": N_TRAIN,
               "ablation": ablation, "ablation_tests": abl_tests,
               "threshold_sensitivity": threshold_sens,
               "ranking_preserved_90_95_98": bool(rank_preserved),
               "stress_test_compound_n300": stress}
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=1))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
