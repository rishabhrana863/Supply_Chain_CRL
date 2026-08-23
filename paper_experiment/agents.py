"""
Agents: PPO (numpy, from scratch), CRL (PPO + causal masking/shaping),
and a sample-average stochastic lookahead planner.

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
    each action (versus no_action) on mean service from days 5 through 20 is
    estimated by stratified mean differences. An action is masked when its
    estimated effect falls below the configured negative-effect threshold."""

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


# ═════════════════ Stochastic lookahead planning baseline ═════════════════

class StochasticOptAgent:
    """Sample-average lookahead over the discrete action set.

    At each replanning point, the planner evaluates every candidate action with
    K rollouts of a nominal internal model over a fixed lookahead horizon. The
    same K random-number streams are used for every action at a replanning
    point. This common-random-number design ensures that action comparisons are
    driven by modeled consequences rather than independent sampling noise.

    The model is a rolling-horizon lookahead planner, not a two-stage stochastic
    program. The weekly version acts once at each seven-day replanning point and
    takes no additional action between planning points. The daily version uses
    the identical model and replans every day.
    """

    name = "stochastic_opt"
    K = 12          # scenarios per action
    LOOKAHEAD = 14  # days

    def __init__(self, cal, seed=0, replan_every=7, lookahead=None,
                 scenario_count=None, allowed_actions=None, expedite_cooldown=0,
                 name=None):
        self.cal = cal
        self.rng = np.random.RandomState(seed)
        self.replan_every = int(replan_every)
        self.lookahead = int(self.LOOKAHEAD if lookahead is None else lookahead)
        self.scenario_count = int(self.K if scenario_count is None else scenario_count)
        self.allowed_actions = tuple(range(N_ACTIONS)) if allowed_actions is None \
            else tuple(int(a) for a in allowed_actions)
        self.expedite_cooldown = int(expedite_cooldown)
        if self.replan_every < 1:
            raise ValueError("replan_every must be at least one day")
        if self.lookahead < 1 or self.scenario_count < 1:
            raise ValueError("lookahead and scenario_count must be positive")
        if not self.allowed_actions or 0 not in self.allowed_actions:
            raise ValueError("allowed_actions must include no_action (0)")
        if name is not None:
            self.name = name
        self.current_action = 0
        self.last_plan_day = -99
        self.last_expedite_day = -999

    def reset(self):
        """Must be called at the start of every episode."""
        self.current_action = 0
        self.last_plan_day = -99
        self.last_expedite_day = -999

    def _rollout_value(self, env, first_action, severity_multipliers, demand_draws):
        """Return the mean nominal rollout cost for one candidate action.

        The planner's internal model mirrors the
        true mechanics at a coarse level (delay days, capacity loss, setup
        times) but must FORECAST severity (persistence with noise) and demand.
        All quantities are in units of days of demand. ``severity_multipliers``
        and ``demand_draws`` are shared across actions at the current replanning
        point. Scenario calculations are vectorized without changing the model.
        """
        s = env._state()
        inv0 = s[0] * 60          # inventory in days of demand
        sev = s[4] * 15
        port = s[9] > 0.5
        A = env.A
        H = self.lookahead
        sev_k = np.maximum(0.0, sev * severity_multipliers)
        scenario_count = len(sev_k)
        inv = np.full(scenario_count, inv0, dtype=float)
        total = np.zeros(scenario_count, dtype=float)
        using_alt = bool(env.using_alternate)
        corridor2 = bool(env.corridor2 or first_action == 3)
        switch_timer = int(round(env.switch_setup)) if (
            first_action == 1 and env.alt_available and not using_alt) else None

        delay = A["leadtime_add_per_severity"] * sev_k
        if port and not corridor2:
            delay += A["port_extra_leadtime"]
        if first_action == 3:
            delay += A["reroute_extra_days"] * 0.5
        capacity = np.maximum(
            0.15, 1 - A["capacity_loss_per_severity"] * sev_k
        )
        if first_action == 2:
            delay = np.maximum(0.0, delay * A["air_leadtime_factor"])
        expedite_left = 7 if first_action == 2 else 0
        emergency_arrival = A["emergency_leadtime"] if first_action == 4 else None

        for day_offset in range(H):
            if switch_timer is not None:
                switch_timer -= 1
                if switch_timer <= 0:
                    using_alt = True
                    switch_timer = None
            if emergency_arrival is not None and day_offset >= emergency_arrival:
                inv += A["emergency_qty_days"]
                total += A["emergency_qty_days"] * A["emergency_cost_multiplier"] \
                    * A["air_cost_multiplier"]
                emergency_arrival = None

            effective_capacity = 1.0 if using_alt else capacity
            inflow = np.where(day_offset < delay, 0.0, effective_capacity)
            cost_multiplier = 1.0
            if expedite_left > 0:
                cost_multiplier *= A["air_cost_multiplier"]
                expedite_left -= 1
            if using_alt:
                cost_multiplier *= A["alt_supplier_cost_premium"]
            if corridor2:
                cost_multiplier *= 1 + A["reroute_extra_cost"]

            demand = np.maximum(0.1, demand_draws[:, day_offset])
            service_cap = A["ration_service_cap"] if first_action == 5 else 1.0
            fulfilled = np.minimum(inv + inflow, demand * service_cap)
            inv = inv + inflow - fulfilled
            unmet = demand - fulfilled
            total += inflow * cost_multiplier + unmet * A["stockout_penalty_factor"]
        return float(np.mean(total))

    def act(self, state, env=None, greedy=True):
        day = env.day
        if day - self.last_plan_day >= self.replan_every:
            scenario_seeds = self.rng.randint(
                0, 2 ** 31 - 1, size=self.scenario_count
            )
            severity_multipliers = np.empty(self.scenario_count, dtype=float)
            demand_draws = np.empty((self.scenario_count, self.lookahead), dtype=float)
            for index, scenario_seed in enumerate(scenario_seeds):
                scenario_rng = np.random.RandomState(int(scenario_seed))
                severity_multipliers[index] = scenario_rng.uniform(0.7, 1.3)
                demand_draws[index] = scenario_rng.lognormal(
                    0, self.cal.daily_demand_cv, size=self.lookahead
                )
            candidate_actions = list(self.allowed_actions)
            if (2 in candidate_actions and self.expedite_cooldown > 0
                    and day - self.last_expedite_day < self.expedite_cooldown):
                candidate_actions.remove(2)
            best, best_val = 0, np.inf
            for a in candidate_actions:
                v = self._rollout_value(
                    env, a, severity_multipliers, demand_draws
                )
                if v < best_val:
                    best, best_val = a, v
            self.current_action = best
            self.last_plan_day = day
            if best == 2:
                self.last_expedite_day = day
            return best
        return 0  # no additional intervention between replanning points
