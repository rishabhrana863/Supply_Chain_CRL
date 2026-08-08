"""
Round-2 analyses:
  1. Censoring robustness: Wilcoxon on episodes with no censoring in any model/seed
  2. P4 inference: per-episode Causal Alignment Index with SE + paired Wilcoxon
  3. Stress-test anomaly investigation:
     a. no-action difficulty of ordinary-compound vs stress scenario sets
     b. rerun stress with random 3-of-4 types (incl. COVID) at severity 4-5
     c. SO action rates in both regimes
Outputs: results/round2_results.json
"""

import json
import csv
import numpy as np
from calibration import Calibration
from environment import SupplyChainEnv, N_ACTIONS, DISRUPTION_ONSET
from agents import PPOAgent, CRLAgent, StochasticOptAgent, fit_causal_model
from run_experiment import train, wilcoxon, cohens_d

out = {}
cal = Calibration(seed=42)
cm = fit_causal_model(cal, n_rollouts=400, seed=42)

# ── 1. censoring robustness from existing per-episode CSV ──
rows = list(csv.DictReader(open("results/per_episode.csv")))
def all_recovered(r):
    keys = ["stochastic_opt_recovered"] + [f"rl_s{i}_recovered" for i in range(5)] \
        + [f"crl_s{i}_recovered" for i in range(5)]
    return all(r[k] == "True" for k in keys if k in r)
sub = [r for r in rows if all_recovered(r)]
so = np.array([float(r["stochastic_opt_recovery_days"]) for r in sub])
rl = np.array([float(r["rl_only_recovery_days"]) for r in sub])
cr = np.array([float(r["crl_recovery_days"]) for r in sub])
w1, p1 = wilcoxon(so, cr); w2, p2 = wilcoxon(rl, cr)
out["censoring_robustness"] = {
    "n_uncensored_episodes": len(sub), "n_total": len(rows),
    "crl_vs_so": {"p": round(p1, 6), "d": round(cohens_d(so, cr), 3),
                  "means": [round(so.mean(), 2), round(cr.mean(), 2)]},
    "crl_vs_rl": {"p": round(p2, 6), "d": round(cohens_d(rl, cr), 3),
                  "means": [round(rl.mean(), 2), round(cr.mean(), 2)]},
}
print("censoring robustness:", json.dumps(out["censoring_robustness"]), flush=True)

# ── train agents for P4 + stress (5 seeds RL/CRL to match main design) ──
rls, crls = [], []
for sd in range(5):
    a = PPOAgent(state_dim=14, seed=42 + sd)
    train(a, cal, 1500, seed_base=43 + sd, label=f"rl-s{sd}")
    rls.append(a)
    b = CRLAgent(state_dim=14, causal_model=cm, seed=42 + sd)
    train(b, cal, 1500, seed_base=43 + sd, label=f"crl-s{sd}")
    crls.append(b)
so_agent = StochasticOptAgent(cal, seed=42)
print("agents trained", flush=True)

# ── 2. P4: per-episode alignment with inference ──
def ei_per_episode(agent, contexts, seed_base=777):
    vals = []
    for i, ctx in enumerate(contexts):
        env = SupplyChainEnv(ctx, cal, np.random.RandomState(seed_base + i))
        s = env.reset()
        aligned, total = 0, 0
        while True:
            a, _, _ = agent.act(s, ctx, greedy=True)
            if env.day > DISRUPTION_ONSET and env.severity_now > 0.5:
                total += 1
                st = cm.stratum(s)
                ates = np.array([0.0] + [cm.ate(st, x) for x in range(1, N_ACTIONS)])
                if ates[a] >= ates.max() - 0.01:
                    aligned += 1
            s, r, done, info = env.step(a)
            if done:
                break
        vals.append(aligned / max(total, 1))
    return np.array(vals)

rng = np.random.RandomState(777)
ei_contexts = cal.sample_episode_contexts(60, rng)
ei_rl = np.mean([ei_per_episode(a, ei_contexts) for a in rls], axis=0)
ei_cr = np.mean([ei_per_episode(a, ei_contexts) for a in crls], axis=0)
w, p = wilcoxon(ei_rl, ei_cr)  # lower tail: rl < crl
out["p4_inference"] = {
    "n_episodes": len(ei_contexts),
    "crl_mean": round(float(ei_cr.mean()), 4), "crl_se": round(float(ei_cr.std(ddof=1)/np.sqrt(len(ei_cr))), 4),
    "rl_mean": round(float(ei_rl.mean()), 4), "rl_se": round(float(ei_rl.std(ddof=1)/np.sqrt(len(ei_rl))), 4),
    "paired_wilcoxon_p": round(p, 6), "d": round(cohens_d(ei_cr, ei_rl), 3),
}
print("P4:", json.dumps(out["p4_inference"]), flush=True)

# ── 3. stress anomaly ──
def eval_set(agent_list, contexts, seed_base, is_so=False, record_actions=False):
    recs, act_counts = [], np.zeros(N_ACTIONS)
    for i, ctx in enumerate(contexts):
        per_seed = []
        for agent in (agent_list if not is_so else [agent_list]):
            env = SupplyChainEnv(ctx, cal, np.random.RandomState(seed_base + i))
            s = env.reset()
            if hasattr(agent, "reset"):
                agent.reset()
            while True:
                if is_so:
                    a = agent.act(s, env=env)
                else:
                    a, _, _ = agent.act(s, ctx, greedy=True)
                if record_actions:
                    act_counts[a] += 1
                s, r, done, info = env.step(a)
                if done:
                    break
            per_seed.append(env.episode_metrics()["recovery_days"])
        recs.append(float(np.mean(per_seed)))
    return np.array(recs), act_counts

class NoAction:
    def act(self, s, ctx=None, greedy=True):
        return 0, 0.0, 0.0

# ordinary compound subset from main eval distribution
rngA = np.random.RandomState(9000)
main_ctx = cal.sample_episode_contexts(200, rngA)
ordinary_compound = [c for c in main_ctx if c["scenario_type"] == "compound"]

# OLD stress design: fixed first-3 types (no COVID), severity 4-5
rngB = np.random.RandomState(1234)
old_stress = cal.sample_episode_contexts(150, rngB)
all_types = list(cal.disruption_type_probs.keys())
for c in old_stress:
    c["disruption_types"] = all_types[:3]
    c["severities"] = {t: int(rngB.choice([4, 5])) for t in c["disruption_types"]}
    c["total_severity"] = sum(c["severities"].values())
    c["scenario_type"] = "compound"

# NEW stress design: random 3 of 4 types (COVID included), severity 4-5
rngC = np.random.RandomState(4321)
new_stress = cal.sample_episode_contexts(300, rngC)
for c in new_stress:
    chosen = list(rngC.choice(all_types, size=3, replace=False))
    c["disruption_types"] = chosen
    c["severities"] = {t: int(rngC.choice([4, 5])) for t in chosen}
    c["total_severity"] = sum(c["severities"].values())
    c["scenario_type"] = "compound"

noact = NoAction()
diag = {}
for label, ctxs in [("ordinary_compound", ordinary_compound),
                    ("old_stress_no_covid", old_stress),
                    ("new_stress_random3of4", new_stress)]:
    na, _ = eval_set([noact], ctxs, seed_base=55000)
    covid_share = np.mean(["COVID_Lockdown" in c["disruption_types"] for c in ctxs])
    sev = np.mean([c["total_severity"] for c in ctxs])
    diag[label] = {"n": len(ctxs), "noaction_recovery": round(float(na.mean()), 2),
                   "covid_share": round(float(covid_share), 3),
                   "mean_total_severity": round(float(sev), 2)}
print("difficulty diagnostics:", json.dumps(diag), flush=True)
out["stress_diagnostics"] = diag

# SO action rates in ordinary vs stress
_, so_acts_ord = eval_set(so_agent, ordinary_compound, seed_base=56000, is_so=True, record_actions=True)
_, so_acts_str = eval_set(so_agent, new_stress[:100], seed_base=57000, is_so=True, record_actions=True)
out["so_action_rates"] = {
    "ordinary_nonzero_rate": round(float(so_acts_ord[1:].sum() / so_acts_ord.sum()), 4),
    "stress_nonzero_rate": round(float(so_acts_str[1:].sum() / so_acts_str.sum()), 4)}
print("SO action rates:", json.dumps(out["so_action_rates"]), flush=True)

# final NEW stress results, all three models
res_stress = {}
for name, agents, is_so in [("crl", crls, False), ("rl_only", rls, False),
                            ("stochastic_opt", so_agent, True)]:
    v, _ = eval_set(agents, new_stress, seed_base=77000, is_so=is_so)
    res_stress[name] = {"recovery_mean": round(float(v.mean()), 2),
                        "recovery_se": round(float(v.std(ddof=1)/np.sqrt(len(v))), 2)}
    res_stress[name]["_vals"] = v
w, p_rl = wilcoxon(res_stress["rl_only"]["_vals"], res_stress["crl"]["_vals"])
w, p_so = wilcoxon(res_stress["stochastic_opt"]["_vals"], res_stress["crl"]["_vals"])
for n in res_stress:
    del res_stress[n]["_vals"]
res_stress["crl_vs_rl_p"] = round(p_rl, 6)
res_stress["crl_vs_so_p"] = round(p_so, 6)
out["new_stress_results"] = res_stress
print("NEW stress:", json.dumps(res_stress), flush=True)

with open("results/round2_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved.")
