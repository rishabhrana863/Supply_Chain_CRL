"""
Four-tier healthcare supply chain simulation environment.

HONESTY CONTRACT:
  1. The environment NEVER knows which agent is acting. Identical dynamics for all.
  2. LPI affects ONLY environmental constraints:
       - customs clearance delay (low customs score → longer clearance)
       - alternate supplier availability & switching setup time
       - inland transport time variance
     LPI never modifies outcomes, rewards, or effectiveness directly.
  3. Action effects are mechanistic (capacity, lead time, inventory), never
     outcome bonuses. Whether an action helps depends on the situation.
  4. All marginal parameters come from the calibrated dataset; structural
     assumptions are listed in ASSUMPTIONS and cited in the paper.

Topology (fixed, documented):
  3 suppliers (1 primary + 2 alternates) → 2 freight corridors (3PL) →
  1 national distribution center → 4 service points (hospitals).
  Hospital inventories are aggregated at the DC level for tractability;
  the DC ships daily to hospitals with negligible domestic lead time
  relative to international replenishment.

Timestep = 1 day. Horizon = 120 days. Disruption onset at day 10.
"""

import numpy as np

# ── Documented structural assumptions (cited in paper's assumptions table) ──
ASSUMPTIONS = {
    # Disruption mechanics (per severity point 1-5, summed over concurrent types)
    "capacity_loss_per_severity": 0.12,   # primary supplier capacity −12%/point
    "leadtime_add_per_severity": 2.0,     # +2 days international lead time /point
    "port_extra_leadtime": 6.0,           # Port_Closure adds 6 days on ocean corridor 1
    "conflict_inland_risk": 0.15,         # Conflict: 15% chance/day inland shipment held
    "covid_demand_spike": 1.30,           # COVID_Lockdown raises demand 30%
    "flood_dc_loss": 0.08,                # Flood: one-time 8% DC inventory loss at onset
    "disruption_decay_days": 45.0,        # severity decays linearly to 0 over 45 days
    # LPI → environmental constraints (constraint-level only)
    "customs_delay_at_lpi2": 6.0,         # days customs clearance at customs score 2.0
    "customs_delay_at_lpi4": 1.0,         # days at score 4.0 (linear in between)
    "alt_supplier_p_at_lpi2": 0.55,       # P(alternate supplier available) at LPI 2
    "alt_supplier_p_at_lpi4": 0.95,       # at LPI 4
    "switch_setup_at_lpi2": 12.0,         # days to qualify/switch supplier at LPI 2
    "switch_setup_at_lpi4": 5.0,          # at LPI 4
    "transport_cv_at_lpi2": 0.35,         # inland transit time CV at LPI 2
    "transport_cv_at_lpi4": 0.10,
    # Action mechanics
    "air_leadtime_factor": 0.35,          # air cuts int'l lead time to 35%
    "air_cost_multiplier": 3.0,           # air costs 3x ocean (GHSC-PSM reports sea-freight savings)
    "land_cost_multiplier": 1.2,
    "reroute_extra_cost": 0.25,           # corridor 2 costs +25%
    "reroute_extra_days": 4.0,            # corridor 2 is 4 days longer nominally
    "emergency_cost_multiplier": 2.5,     # emergency procurement unit cost premium
    "emergency_qty_days": 10.0,           # emergency order = 10 days of demand
    "emergency_leadtime": 5.0,            # arrives in 5 days by air
    "alt_supplier_cost_premium": 1.15,    # alternate supplier +15% unit cost
    "alt_supplier_reliability_delta": -0.05,  # alternates slightly less proven
    "switch_requalification_cost_days": 3.0,  # one-time qualification cost (days of demand value)
    "ration_service_cap": 0.85,           # rationing caps fill rate at 85%
    "holding_cost_per_unit_day": 0.0004,  # fraction of unit cost per day
    "stockout_penalty_factor": 4.0,       # unmet demand penalized at 4x unit value (paper: service priority)
    # Reward normalization
    "reward_alpha": 0.40, "reward_beta": 0.30, "reward_gamma": 0.30,
}

ACTIONS = {
    0: "no_action",
    1: "switch_supplier",
    2: "expedite_air",
    3: "reroute_corridor",
    4: "emergency_procurement",
    5: "ration_allocate",
}
N_ACTIONS = len(ACTIONS)

DISRUPTION_ONSET = 10
HORIZON = 120
BASELINE_SERVICE_TARGET = 0.95  # recovery threshold (fraction of pre-disruption service)


def _lerp(lo_val, hi_val, x, lo=2.0, hi=4.0):
    """Linear interpolation of a parameter across LPI/customs score range."""
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return lo_val + t * (hi_val - lo_val)


class SupplyChainEnv:
    """State vector (14 dims, all in [0,1]-ish ranges):
       0 dc_inventory_days      (inventory / daily demand / 60)
       1 pipeline_days          (in-transit / daily demand / 60)
       2 service_level_7d       (rolling 7-day fill rate)
       3 effective_leadtime     (current int'l lead time / 100)
       4 active_severity        (current total severity / 15)
       5 days_since_onset       (/ 120)
       6 primary_capacity       (0-1)
       7 using_alternate        (0/1)
       8 corridor2_active       (0/1)
       9 port_closure_active    (0/1)
      10 conflict_active        (0/1)
      11 lpi_norm               ((lpi-2)/2.5)
      12 customs_norm           ((customs-2)/2.5)
      13 emergency_outstanding  (0/1)
    """
    STATE_DIM = 14

    def __init__(self, context: dict, cal, rng: np.random.RandomState):
        self.ctx = context
        self.cal = cal
        self.rng = rng
        self.A = ASSUMPTIONS

    def reset(self):
        c = self.ctx
        self.day = 0
        self.daily_demand_base = c["daily_demand"]
        self.unit_cost = c["base_freight"] / (self.daily_demand_base * 30)  # freight per ~monthly volume
        self.dc_inventory = self.daily_demand_base * 30.0     # 30 days in-country buffer (GHSC min-max ≈ 1-3 months)
        self.pipeline = []  # list of (arrival_day, qty, cost_multiplier)
        self.base_leadtime = c["base_lead_time"]
        self.customs_delay = _lerp(self.A["customs_delay_at_lpi2"], self.A["customs_delay_at_lpi4"], c["customs"])
        self.transport_cv = _lerp(self.A["transport_cv_at_lpi2"], self.A["transport_cv_at_lpi4"], c["lpi"])
        self.switch_setup = _lerp(self.A["switch_setup_at_lpi2"], self.A["switch_setup_at_lpi4"], c["lpi"])
        p_alt = _lerp(self.A["alt_supplier_p_at_lpi2"], self.A["alt_supplier_p_at_lpi4"], c["lpi"])
        self.alt_available = self.rng.random() < p_alt
        # steady-state replenishment: schedule regular arrivals
        for d in range(0, HORIZON + 60, 7):
            qty = self.daily_demand_base * 7
            self.pipeline.append([d + self.rng.normal(0, 2), qty, 1.0])
        # dynamic state
        self.primary_capacity = 1.0
        self.using_alternate = False
        self.switch_countdown = None
        self.corridor2 = False
        self.rationing = False
        self.emergency_outstanding = False
        self.severity_now = 0.0
        self.fill_history = []
        self.service_baseline = None   # measured pre-disruption
        self.total_cost = 0.0
        self.cost_history = []
        self.service_history = []
        self.recovered_streak = 0
        self.recovery_day = None
        self.pre_disruption_fills = []
        return self._state()

    # ── disruption profile ──
    def _severity_at(self, day):
        if day < DISRUPTION_ONSET:
            return 0.0, {}
        elapsed = day - DISRUPTION_ONSET
        decay = max(0.0, 1.0 - elapsed / self.A["disruption_decay_days"])
        active = {t: s * decay for t, s in self.ctx["severities"].items() if s * decay > 0.3}
        return sum(active.values()), active

    def step(self, action: int):
        A, c = self.A, self.ctx
        day = self.day
        sev_total, active = self._severity_at(day)
        self.severity_now = sev_total
        port = "Port_Closure" in active
        conflict = "Conflict" in active
        covid = "COVID_Lockdown" in active
        flood_onset = ("Flood" in self.ctx["severities"]) and day == DISRUPTION_ONSET

        # ── apply action mechanics ──
        day_cost = 0.0
        if action == 1 and self.alt_available and not self.using_alternate and self.switch_countdown is None:
            self.switch_countdown = int(round(self.switch_setup))
        elif action == 2:
            # expedite: next scheduled arrival converted to air
            future = [p for p in self.pipeline if p[0] > day + 2 and p[2] == 1.0]
            if future:
                nxt = min(future, key=lambda p: p[0])
                nxt[0] = day + max(2.0, (nxt[0] - day) * A["air_leadtime_factor"])
                nxt[2] = A["air_cost_multiplier"]
        elif action == 3 and not self.corridor2:
            self.corridor2 = True
            for p in self.pipeline:
                if p[0] > day + 1:
                    p[0] += A["reroute_extra_days"] * 0.5  # transition friction
        elif action == 4 and not self.emergency_outstanding:
            qty = self.daily_demand_base * A["emergency_qty_days"]
            self.pipeline.append([day + A["emergency_leadtime"], qty,
                                  A["emergency_cost_multiplier"] * A["air_cost_multiplier"]])
            self.emergency_outstanding = True
        self.rationing = (action == 5)

        # supplier switch completes
        if self.switch_countdown is not None:
            self.switch_countdown -= 1
            if self.switch_countdown <= 0:
                self.using_alternate = True
                self.switch_countdown = None
                day_cost += self.daily_demand_base * self.unit_cost \
                    * A["switch_requalification_cost_days"]  # one-time qualification cost

        # ── disruption mechanics (mechanistic, agent-blind) ──
        self.primary_capacity = max(0.15, 1.0 - A["capacity_loss_per_severity"] * sev_total)
        supply_capacity = 1.0 if self.using_alternate else self.primary_capacity

        if flood_onset:
            self.dc_inventory *= (1.0 - A["flood_dc_loss"] * self.ctx["severities"]["Flood"] / 3.0)

        # alternate supplier is genuinely less proven: stochastic shortfalls
        # (assumption alt_supplier_reliability_delta, applied mechanistically)
        if self.using_alternate:
            alt_rel = np.clip(self.ctx["reliability"] + A["alt_supplier_reliability_delta"], 0.5, 1.0)
            supply_capacity = 1.0 if self.rng.random() < alt_rel else 0.6

        # arrivals: shipments land if their (stochastic) arrival day has come
        leadtime_shift = A["leadtime_add_per_severity"] * sev_total
        if port and not self.corridor2:
            leadtime_shift += A["port_extra_leadtime"]
        arrived = []
        for p in self.pipeline:
            eff_arrival = p[0] + leadtime_shift + self.customs_delay * 0.3 \
                          + self.rng.normal(0, self.transport_cv * 3)
            if eff_arrival <= day:
                if conflict and self.rng.random() < A["conflict_inland_risk"]:
                    p[0] += 2.0  # held at checkpoint; retry later
                    continue
                qty = p[1] * supply_capacity
                unit_mult = p[2] * (A["alt_supplier_cost_premium"] if self.using_alternate else 1.0) \
                            * (1 + A["reroute_extra_cost"] if self.corridor2 else 1.0)
                self.dc_inventory += qty
                day_cost += qty * self.unit_cost * unit_mult
                arrived.append(p)
        for p in arrived:
            self.pipeline.remove(p)
            if p[2] >= A["emergency_cost_multiplier"]:
                self.emergency_outstanding = False

        # ── demand & fulfillment ──
        demand = self.daily_demand_base * (A["covid_demand_spike"] if covid else 1.0)
        demand *= max(0.1, self.rng.lognormal(0, self.cal.daily_demand_cv))
        cap = A["ration_service_cap"] if self.rationing else 1.0
        fulfilled = min(self.dc_inventory, demand * cap)
        self.dc_inventory -= fulfilled
        fill_rate = fulfilled / demand
        unmet = demand - fulfilled

        # ── costs ──
        day_cost += self.dc_inventory * self.unit_cost * A["holding_cost_per_unit_day"]
        day_cost += unmet * self.unit_cost * A["stockout_penalty_factor"]
        self.total_cost += day_cost

        # ── service tracking & recovery detection ──
        self.fill_history.append(fill_rate)
        self.service_history.append(fill_rate)
        self.cost_history.append(day_cost)
        if day < DISRUPTION_ONSET:
            self.pre_disruption_fills.append(fill_rate)
        if day == DISRUPTION_ONSET:
            self.service_baseline = float(np.mean(self.pre_disruption_fills)) if self.pre_disruption_fills else 1.0
        # ── normalized reward: Rt = .40*S~ − .30*C~ − .30*T~ ──
        # T~ = 1 while current 3-day service is below the recovery threshold
        # (recovery time itself is computed post-hoc in episode_metrics).
        S = fill_rate                                        # already [0,1]
        cost_scale = self.daily_demand_base * self.unit_cost * 3.0
        C = min(day_cost / cost_scale, 1.5) / 1.5            # [0,1]
        baseline = self.service_baseline if self.service_baseline else 1.0
        below = np.mean(self.fill_history[-3:]) < BASELINE_SERVICE_TARGET * baseline
        T = 1.0 if (day > DISRUPTION_ONSET and below) else 0.0
        reward = A["reward_alpha"] * S - A["reward_beta"] * C - A["reward_gamma"] * T

        self.day += 1
        done = self.day >= HORIZON
        return self._state(), reward, done, {
            "fill_rate": fill_rate, "day_cost": day_cost,
            "severity": sev_total, "active": list(active.keys()),
            "port": port, "conflict": conflict, "covid": covid,
        }

    def _state(self):
        recent = np.mean(self.fill_history[-7:]) if self.fill_history else 1.0
        sev, active = self._severity_at(self.day)
        eff_lt = self.base_leadtime + ASSUMPTIONS["leadtime_add_per_severity"] * sev
        pipeline_qty = sum(p[1] for p in self.pipeline if p[0] < self.day + 60)
        return np.array([
            np.clip(self.dc_inventory / self.daily_demand_base / 60.0, 0, 2),
            np.clip(pipeline_qty / self.daily_demand_base / 60.0, 0, 2),
            recent,
            eff_lt / 100.0,
            sev / 15.0,
            max(0, self.day - DISRUPTION_ONSET) / HORIZON,
            self.primary_capacity,
            1.0 if self.using_alternate else 0.0,
            1.0 if self.corridor2 else 0.0,
            1.0 if "Port_Closure" in active else 0.0,
            1.0 if "Conflict" in active else 0.0,
            (self.ctx["lpi"] - 2.0) / 2.5,
            (self.ctx["customs"] - 2.0) / 2.5,
            1.0 if self.emergency_outstanding else 0.0,
        ], dtype=np.float64)

    # ── episode summary metrics (manuscript definitions, computed post-hoc) ──
    def episode_metrics(self, threshold: float = BASELINE_SERVICE_TARGET):
        fills = np.array(self.service_history)
        baseline = self.service_baseline if self.service_baseline else 1.0
        thresh = threshold * baseline
        post = fills[DISRUPTION_ONSET:]
        # 3-day rolling mean of post-onset service
        roll = np.convolve(post, np.ones(3) / 3, mode="valid")
        below = roll < thresh
        if not below.any():
            rec = 0.0            # service never materially dipped
            recovered = True
        else:
            last_below = int(np.where(below)[0][-1])
            if last_below >= len(roll) - 1:
                rec = float(HORIZON - DISRUPTION_ONSET)   # censored: never recovered
                recovered = False
            else:
                rec = float(last_below + 3)  # day (post-onset) service is back & stays
                recovered = True
        # CoV of service during the recovery window (onset → recovery, min 7 days)
        window = post[:max(int(rec), 7)]
        cov = float(np.std(window) / np.mean(window)) if np.mean(window) > 0 else 1.0
        return {
            "recovery_days": rec,
            "recovered": recovered,
            "dipped": bool(below.any()),
            "total_cost": float(self.total_cost),
            "mean_service": float(np.mean(post)),
            "service_cov": cov,
            "min_service": float(np.min(post)),
        }
