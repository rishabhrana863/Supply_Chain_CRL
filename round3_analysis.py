"""
Round-3 analyses:
  1. Survival robustness: log-rank tests treating never-recovered episodes as
     right-censored (pooled model-seed observations), + RMST note.
  2. LPI mapping-endpoint sensitivity: retrain under 'mild' and 'harsh'
     endpoint variants; check model ordering preserved.
Outputs: results/round3_results.json
"""

import json
import csv
import math
import numpy as np
import environment as envmod
from calibration import Calibration
from environment import SupplyChainEnv
from agents import PPOAgent, CRLAgent, StochasticOptAgent, fit_causal_model
from run_experiment import train

out = {}

# ── 1. log-rank tests from per-episode CSV ──
rows = list(csv.DictReader(open("results/per_episode.csv")))

def collect(model_prefixes):
    times, events = [], []
    for r in rows:
        for p in model_prefixes:
            t = float(r[f"{p}_recovery_days"])
            e = r[f"{p}_recovered"] == "True"
            times.append(t); events.append(1 if e else 0)
    return np.array(times), np.array(events)

def logrank(t1, e1, t2, e2):
    """Two-sample log-rank test (numpy)."""
    all_t = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
    O1 = E1 = V = 0.0
    for t in all_t:
        n1 = np.sum(t1 >= t); n2 = np.sum(t2 >= t)
        d1 = np.sum((t1 == t) & (e1 == 1)); d2 = np.sum((t2 == t) & (e2 == 1))
        n, d = n1 + n2, d1 + d2
        if n < 2 or d == 0:
            continue
        E1 += d * n1 / n
        O1 += d1
        V += d * (n1 / n) * (n2 / n) * (n - d) / (n - 1)
    if V <= 0:
        return 0.0, 1.0
    z = (O1 - E1) / math.sqrt(V)
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p

so_t, so_e = collect(["stochastic_opt"])
rl_t, rl_e = collect([f"rl_s{i}" for i in range(5)])
cr_t, cr_e = collect([f"crl_s{i}" for i in range(5)])
z1, p1 = logrank(cr_t, cr_e, so_t, so_e)
z2, p2 = logrank(cr_t, cr_e, rl_t, rl_e)
out["logrank"] = {
    "crl_vs_so": {"z": round(z1, 3), "p": round(p1, 8)},
    "crl_vs_rl": {"z": round(z2, 3), "p": round(p2, 8)},
    "note": "never-recovered episodes treated as right-censored at horizon; "
            "reported means are restricted-mean recovery times (censoring at tau=110)",
}
print("log-rank:", json.dumps(out["logrank"]), flush=True)

# ── 2. LPI mapping endpoint sensitivity ──
VARIANTS = {
    "mild":  {"customs_delay_at_lpi2": 4.0, "switch_setup_at_lpi2": 9.0,
              "alt_supplier_p_at_lpi2": 0.65, "transport_cv_at_lpi2": 0.25},
    "harsh": {"customs_delay_at_lpi2": 8.0, "switch_setup_at_lpi2": 15.0,
              "alt_supplier_p_at_lpi2": 0.45, "transport_cv_at_lpi2": 0.45},
}
BASE = {k: envmod.ASSUMPTIONS[k] for k in VARIANTS["mild"]}

cal = Calibration(seed=42)
sens = {}
for vname, changes in VARIANTS.items():
    envmod.ASSUMPTIONS.update(changes)
    cm = fit_causal_model(cal, n_rollouts=250, seed=42)
    rls, crls = [], []
    for sd in range(2):
        a = PPOAgent(state_dim=14, seed=42 + sd)
        train(a, cal, 800, seed_base=43 + sd, label=f"{vname}-rl{sd}")
        rls.append(a)
        b = CRLAgent(state_dim=14, causal_model=cm, seed=42 + sd)
        train(b, cal, 800, seed_base=43 + sd, label=f"{vname}-crl{sd}")
        crls.append(b)
    so = StochasticOptAgent(cal, seed=42)
    rng = np.random.RandomState(9000)
    ctxs = cal.sample_episode_contexts(100, rng)
    res = {}
    for name, agents, is_so in [("crl", crls, False), ("rl_only", rls, False),
                                ("stochastic_planner", so, True)]:
        vals = []
        for i, ctx in enumerate(ctxs):
            per = []
            for ag in (agents if not is_so else [agents]):
                env = SupplyChainEnv(ctx, cal, np.random.RandomState(63000 + i))
                s = env.reset()
                if hasattr(ag, "reset"):
                    ag.reset()
                while True:
                    if is_so:
                        act = ag.act(s, env=env)
                    else:
                        act, _, _ = ag.act(s, ctx, greedy=True)
                    s, r, done, info = env.step(act)
                    if done:
                        break
                per.append(env.episode_metrics()["recovery_days"])
            vals.append(float(np.mean(per)))
        res[name] = round(float(np.mean(vals)), 2)
    res["ordering_preserved"] = bool(res["crl"] < res["rl_only"] < res["stochastic_planner"])
    sens[vname] = res
    print(f"{vname}: {json.dumps(res)}", flush=True)
    envmod.ASSUMPTIONS.update(BASE)  # restore

out["lpi_mapping_sensitivity"] = sens
with open("results/round3_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved.")
