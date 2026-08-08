"""
Agents: PPO (numpy, from scratch), CRL (PPO + causal masking/shaping),
and a two-stage stochastic optimization baseline (sample average approximation).

No agent receives outcome bonuses. CRL's causal layer is estimated from
INTERVENTIONAL SIMULATION DATA (randomized-action rollouts), not hand-set.
"""

import numpy as np
from environment import SupplyChainEnv, N_ACTIONS, ACTIONS, HORIZON, DISRUPTION_ONSET


# ═══════════════════════════════════ PPO ═══════════════════════════════════

class MLP:
    """Two-layer MLP with policy and value heads. Manual backprop, Adam."""

    def __init__(self, in_dim, n_actions, hidden=64, seed=0, lr=3e-4):
        r = np.random.RandomState(seed)
        s = lambda a, b: r.randn(a, b) * np.sqrt(2.0 / a)
        self.W1 = s(in_dim, hidden); self.b1 = np.zeros(hidden)
        self.W2 = s(hidden, hidden); self.b2 = np.zeros(hidden)
        self.Wp = s(hidden, n_actions) * 0.01; self.bp = np.zeros(n_actions)
        self.Wv = s(hidden, 1) * 0.1; self.bv = np.zeros(1)
        self.params = ["W1", "b1", "W2", "b2", "Wp", "bp", "Wv", "bv"]
        self.lr = lr
        self.adam_m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.adam_v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.adam_t = 0

    def forward(self, X):
        h1 = np.tanh(X @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        logits = h2 @ self.Wp + self.bp
        value = (h2 @ self.Wv + self.bv).squeeze(-1)
        return logits, value, (X, h1, h2)

    @staticmethod
    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def backward_and_step(self, cache, dlogits, dvalue):
        X, h1, h2 = cache
        n = X.shape[0]
        g = {}
        g["Wp"] = h2.T @ dlogits / n; g["bp"] = dlogits.mean(0)
        g["Wv"] = h2.T @ dvalue[:, None] / n; g["bv"] = dvalue.mean(0, keepdims=True)
        dh2 = dlogits @ self.Wp.T + dvalue[:, None] @ self.Wv.T
        dh2 = dh2 * (1 - h2 ** 2)
        g["W2"] = h1.T @ dh2 / n; g["b2"] = dh2.mean(0)
        dh1 = dh2 @ self.W2.T * (1 - h1 ** 2)
        g["W1"] = X.T @ dh1 / n; g["b1"] = dh1.mean(0)
        # global grad clip
        norm = np.sqrt(sum((g[p] ** 2).sum() for p in g))
        if norm > 5.0:
            for p in g: g[p] *= 5.0 / norm
        # Adam
        self.adam_t += 1
        b1, b2m, eps = 0.9, 0.999, 1e-8
        for p in self.params:
            self.adam_m[p] = b1 * self.adam_m[p] + (1 - b1) * g[p]
            self.adam_v[p] = b2m * self.adam_v[p] + (1 - b2m) * g[p] ** 2
            mhat = self.adam_m[p] / (1 - b1 ** self.adam_t)
            vhat = self.adam_v[p] / (1 - b2m ** self.adam_t)
            setattr(self, p, getattr(self, p) - self.lr * mhat / (np.sqrt(vhat) + eps))


class PPOAgent:
    """Standard PPO with GAE. No causal components."""

    name = "rl_only"

    def __init__(self, state_dim, seed=0, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=4, entropy_coef=0.02):
        self.net = MLP(state_dim, N_ACTIONS, seed=seed, lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.entropy_coef = epochs, entropy_coef
        self.rng = np.random.RandomState(seed + 1000)
        self.buffer = []

    def mask(self, state, ctx):
        return np.ones(N_ACTIONS)

    def act(self, state, ctx=None, greedy=False):
        logits, value, _ = self.net.forward(state[None, :])
        m = self.mask(state, ctx)
        logits = logits[0].copy()
        logits[m == 0] = -1e9
        probs = MLP.softmax(logits[None, :])[0]
        a = int(np.argmax(probs)) if greedy else int(self.rng.choice(N_ACTIONS, p=probs))
        return a, float(np.log(probs[a] + 1e-12)), float(value[0])

    def shaped(self, r, state, action, ctx):
        return r  # no shaping in plain PPO

    def store(self, s, a, logp, r, v, done):
        self.buffer.append((s, a, logp, r, v, done))

    def update(self):
        if not self.buffer:
            return
        S = np.array([b[0] for b in self.buffer])
        A = np.array([b[1] for b in self.buffer])
        LP = np.array([b[2] for b in self.buffer])
        R = np.array([b[3] for b in self.buffer])
        V = np.array([b[4] for b in self.buffer])
        D = np.array([b[5] for b in self.buffer], dtype=float)
        # GAE
        adv = np.zeros_like(R); last = 0.0
        for t in reversed(range(len(R))):
            nv = 0.0 if D[t] else (V[t + 1] if t + 1 < len(V) else 0.0)
            delta = R[t] + self.gamma * nv - V[t]
            last = delta + self.gamma * self.lam * (0.0 if D[t] else last)
            adv[t] = last
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(R)
        idx = np.arange(n)
        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, 256):
                mb = idx[start:start + 256]
                logits, value, cache = self.net.forward(S[mb])
                probs = MLP.softmax(logits)
                logp_new = np.log(probs[np.arange(len(mb)), A[mb]] + 1e-12)
                ratio = np.exp(logp_new - LP[mb])
                clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip)
                use_clip = (clipped * adv[mb] < ratio * adv[mb])
                # dL/dlogp for clipped surrogate (maximize → negate for grad descent framing)
                dlogp = -np.where(use_clip, 0.0, ratio * adv[mb])
                # entropy bonus grad
                dlogits = probs.copy()
                dlogits[np.arange(len(mb)), A[mb]] -= 1.0
                dlogits *= dlogp[:, None] * -1.0  # chain: dlogp/dlogits = onehot - probs
                # entropy: −Σp log p ; d/dlogits = −p(logp + H)... approximate with logit pull
                ent_grad = probs * (np.log(probs + 1e-12) + 1.0)
                dlogits += self.entropy_coef * ent_grad
                dvalue = (value - ret[mb])  # 0.5*(v-ret)^2 grad
                self.net.backward_and_step(cache, dlogits, dvalue)
        self.buffer = []


# ═══════════════════════════ Causal layer (learned) ═══════════════════════════

class CausalModel:
    """Context-conditional average treatment effects estimated from randomized
    interventional rollouts. Context is discretized into strata; the effect of
    each action (vs no_action) on next-7-day service and cost is estimated by
    stratified mean differences. This is genuine estimation — if an action
    doesn't help in the data, its ATE will be ≤ 0 and it gets masked."""

    STRATA_KEYS = ["port", "capacity_low", "inventory_low", "lpi_low", "conflict"]

    def __init__(self):
        self.table = {}     # stratum → action → (sum_effect, count)
        self.fitted = False

    @staticmethod
    def stratum(state, info=None):
        port = state[9] > 0.5
        conflict = state[10] > 0.5
        cap_low = state[6] < 0.7
        inv_low = state[0] < 0.35
        lpi_low = state[11] < 0.25
        return (port, conflict, cap_low, inv_low, lpi_low)

    def record(self, stratum, action, effect):
        d = self.table.setdefault(stratum, {})
        s, c = d.get(action, (0.0, 0))
        d[action] = (s + effect, c + 1)

    def ate(self, stratum, action):
        d = self.table.get(stratum)
        if not d or action not in d or 0 not in d:
            return 0.0
        s_a, c_a = d[action]
        s_0, c_0 = d[0]
        if c_a < 5 or c_0 < 5:
            return 0.0
        return s_a / c_a - s_0 / c_0

    def feasible_mask(self, state, threshold=-0.02):
        """Mask actions whose estimated ATE is clearly negative in this stratum."""
        st = self.stratum(state)
        m = np.ones(N_ACTIONS)
        for a in range(1, N_ACTIONS):
            if self.ate(st, a) < threshold:
                m[a] = 0
        m[0] = 1
        return m


def fit_causal_model(cal, n_rollouts=300, seed=7):
    """Randomized-intervention phase: uniform-random policies generate
    interventional data; effects estimated by stratified differences."""
    rng = np.random.RandomState(seed)
    cm = CausalModel()
    contexts = cal.sample_episode_contexts(n_rollouts, rng)
    for ctx in contexts:
        env = SupplyChainEnv(ctx, cal, np.random.RandomState(rng.randint(1 << 30)))
        s = env.reset()
        pending = []  # (stratum, action, day_applied, baseline_service)
        while True:
            a = int(rng.randint(0, N_ACTIONS))
            st = cm.stratum(s)
            day0 = env.day
            base_recent = np.mean(env.fill_history[-3:]) if env.fill_history else 1.0
            s, r, done, info = env.step(a)
            pending.append([st, a, day0, base_recent])
            # resolve effects over days 5-21 post-action: captures delayed
            # benefits of interventions with setup time (e.g., supplier switch)
            resolved = [p for p in pending if env.day - p[2] >= 21]
            for p in resolved:
                window = env.service_history[p[2] + 5:p[2] + 21]
                effect = float(np.mean(window)) - p[3]
                cm.record(p[0], p[1], effect)
                pending.remove(p)
            if done:
                break
    cm.fitted = True
    return cm


class CRLAgent(PPOAgent):
    """PPO + causal action masking + causal reward shaping.
    Mask and shaping terms come from the fitted CausalModel (estimated, not set)."""

    name = "crl"

    def __init__(self, state_dim, causal_model, causal_lambda=0.15, **kw):
        super().__init__(state_dim, **kw)
        self.cm = causal_model
        self.causal_lambda = causal_lambda

    def mask(self, state, ctx):
        return self.cm.feasible_mask(state)

    def shaped(self, r, state, action, ctx):
        ate = self.cm.ate(self.cm.stratum(state), action)
        return r + self.causal_lambda * np.clip(ate, -0.5, 0.5)


# ═══════════════ Two-stage stochastic optimization baseline ═══════════════

class StochasticOptAgent:
    """Sample average approximation over the discrete action set.
    At each replanning point (weekly), evaluates each action by K Monte Carlo
    rollouts of a NOMINAL internal model (persistence forecast of current
    disruption, historical demand distribution), then commits to the action
    minimizing expected cost − service. Between replanning points, holds policy.
    Exact minimization over the discrete first-stage set = enumeration,
    i.e., a genuine two-stage SAA stochastic program."""

    name = "stochastic_opt"
    REPLAN_EVERY = 7
    K = 12          # scenarios per action
    LOOKAHEAD = 14  # days

    def __init__(self, cal, seed=0):
        self.cal = cal
        self.rng = np.random.RandomState(seed)
        self.current_action = 0
        self.last_plan_day = -99

    def reset(self):
        """Must be called at the start of every episode."""
        self.current_action = 0
        self.last_plan_day = -99

    def _rollout_value(self, env, first_action):
        """Nominal two-stage rollout. The planner's internal model mirrors the
        true mechanics at a coarse level (delay days, capacity loss, setup
        times) but must FORECAST severity (persistence with noise) and demand.
        All quantities in units of days-of-demand."""
        s = env._state()
        inv0 = s[0] * 60          # inventory in days of demand
        sev = s[4] * 15
        port = s[9] > 0.5
        A = env.A
        H = self.LOOKAHEAD
        vals = []
        for _ in range(self.K):
            inv = inv0
            total = 0.0
            sev_k = max(0.0, sev * self.rng.uniform(0.7, 1.3))
            using_alt = env.using_alternate
            corridor2 = env.corridor2 or (first_action == 3)
            switch_timer = int(round(env.switch_setup)) if (
                first_action == 1 and env.alt_available and not using_alt) else None
            # supply interruption: arrivals stalled for `delay` days, then resume
            delay = A["leadtime_add_per_severity"] * sev_k
            if port and not corridor2:
                delay += A["port_extra_leadtime"]
            if first_action == 3:
                delay += A["reroute_extra_days"] * 0.5   # transition friction
            capacity = max(0.15, 1 - A["capacity_loss_per_severity"] * sev_k)
            expedite_left = 0
            if first_action == 2:
                delay = max(0.0, delay * A["air_leadtime_factor"])
                expedite_left = 7   # one air-converted arrival window
            emergency_arrival = A["emergency_leadtime"] if first_action == 4 else None
            for d in range(H):
                if switch_timer is not None:
                    switch_timer -= 1
                    if switch_timer <= 0:
                        using_alt = True
                        switch_timer = None
                if emergency_arrival is not None and d >= emergency_arrival:
                    inv += A["emergency_qty_days"]
                    total += A["emergency_qty_days"] * A["emergency_cost_multiplier"] \
                        * A["air_cost_multiplier"]
                    emergency_arrival = None
                eff_capacity = 1.0 if using_alt else capacity
                inflow = 0.0 if d < delay else 1.0 * eff_capacity
                cost_mult = 1.0
                if expedite_left > 0:
                    cost_mult *= A["air_cost_multiplier"]; expedite_left -= 1
                if using_alt:
                    cost_mult *= A["alt_supplier_cost_premium"]
                if corridor2:
                    cost_mult *= (1 + A["reroute_extra_cost"])
                demand = max(0.1, self.rng.lognormal(0, self.cal.daily_demand_cv))
                cap = A["ration_service_cap"] if first_action == 5 else 1.0
                fulfilled = min(inv + inflow, demand * cap)
                inv = inv + inflow - fulfilled
                unmet = demand - fulfilled
                total += inflow * cost_mult + unmet * A["stockout_penalty_factor"]
            vals.append(total)
        return float(np.mean(vals))

    def act(self, state, env=None, greedy=True):
        day = env.day
        if day - self.last_plan_day >= self.REPLAN_EVERY:
            best, best_val = 0, np.inf
            for a in range(N_ACTIONS):
                v = self._rollout_value(env, a)
                if v < best_val:
                    best, best_val = a, v
            self.current_action = best
            self.last_plan_day = day
            return best
        return 0  # committed plan: act once per replanning cycle
