"""
Calibration module: derives ALL environment parameters from the datasets.
Nothing in here knows about agents. No outcome differences are programmed.

Inputs:
  - GHSC_PSM_Synthetic_Resilience_Dataset_v2_consistent.csv (2,000 records)
  - International_LPI_from_2007_to_2023_0.xlsx (World Bank LPI, 2023 + 2018 sheets)
  - public_emdat_custom_request_*.xlsx (EM-DAT disaster records)

Outputs: a Calibration object with documented, data-derived parameters.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

UPLOADS = Path(__file__).resolve().parent.parent / "data"

GHSC_FILE = UPLOADS / "GHSC_PSM_Synthetic_Resilience_Dataset_v2_consistent.csv"
LPI_FILE = UPLOADS / "International_LPI_from_2007_to_2023.xlsx"
EMDAT_FILE = UPLOADS / "emdat_custom_request_2025-10-23.xlsx"

COUNTRIES = ["Ethiopia", "Ghana", "Kenya", "Malawi", "Mozambique", "Nigeria", "Uganda", "Zambia"]


def _ols_slope(x: np.ndarray, y: np.ndarray):
    """Simple OLS slope + intercept (numpy only)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0, ym
    slope = ((x - xm) * (y - ym)).sum() / denom
    return float(slope), float(ym - slope * xm)


class Calibration:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.ghsc = pd.read_csv(GHSC_FILE)
        self._calibrate_lpi()
        self._calibrate_lead_times()
        self._calibrate_disruptions()
        self._calibrate_costs()
        self._calibrate_demand()
        self._calibrate_reliability()

    # ── LPI: country-level infrastructure (2018 & 2023 editions only — no 2024) ──
    def _calibrate_lpi(self):
        xls = pd.ExcelFile(LPI_FILE)
        lpi23 = xls.parse("2023")
        # 2018 sheet has a different layout: header row at index 2,
        # columns: Country | Code | score | lower | upper | rank ...
        lpi18 = xls.parse("2018", header=2)
        lpi18 = lpi18.rename(columns={"Country": "Economy", "score": "LPI Score"})
        self.lpi = {}
        for c in COUNTRIES:
            row23 = lpi23[lpi23["Economy"].astype(str).str.strip() == c]
            row18 = lpi18[lpi18["Economy"].astype(str).str.strip() == c]
            score23 = float(row23["LPI Score"].iloc[0]) if len(row23) else np.nan
            score18 = float(row18["LPI Score"].iloc[0]) if len(row18) else np.nan
            infra23 = float(row23["Infrastructure Score"].iloc[0]) if len(row23) else np.nan
            customs23 = float(row23["Customs Score"].iloc[0]) if len(row23) else np.nan
            # Use 2023 as evaluation-period value; fall back to 2018 if missing
            self.lpi[c] = {
                "lpi": score23 if not np.isnan(score23) else score18,
                "lpi_2018": score18,
                "infrastructure": infra23,
                "customs": customs23,
            }
        # If any country entirely missing, use sample median (documented)
        med = np.nanmedian([v["lpi"] for v in self.lpi.values()])
        for c, v in self.lpi.items():
            for k in v:
                if v[k] is None or (isinstance(v[k], float) and np.isnan(v[k])):
                    v[k] = float(med)

    # ── Lead times: per country-commodity from GHSC records ──
    def _calibrate_lead_times(self):
        g = self.ghsc.groupby(["Country", "Commodity_Type"])["Lead_Time_Days"]
        self.lead_time_mean = g.mean().to_dict()
        self.lead_time_std = g.std().fillna(5.0).to_dict()
        self.lead_time_global_mean = float(self.ghsc["Lead_Time_Days"].mean())
        self.lead_time_global_std = float(self.ghsc["Lead_Time_Days"].std())

    # ── Disruptions: empirical frequencies + severity → delay regression ──
    def _calibrate_disruptions(self):
        df = self.ghsc
        disrupted = df[df["Disruption_Type"].notna()]
        # empirical type distribution
        counts = disrupted["Disruption_Type"].value_counts(normalize=True)
        self.disruption_type_probs = counts.to_dict()
        self.p_disruption = len(disrupted) / len(df)  # fraction of records disrupted
        # severity distribution per type
        self.severity_dist = {
            t: disrupted[disrupted["Disruption_Type"] == t]["Disruption_Severity"]
            .value_counts(normalize=True).sort_index().to_dict()
            for t in counts.index
        }
        # severity → delivery delay (data-derived dynamic)
        slope, intercept = _ols_slope(
            disrupted["Disruption_Severity"].values, disrupted["Delivery_Delay_Days"].values
        )
        self.delay_per_severity = slope        # extra delay days per severity point
        self.delay_intercept = intercept
        # severity → resupply time (recovery driver)
        slope_r, intercept_r = _ols_slope(
            disrupted["Disruption_Severity"].values, disrupted["Resupply_Time_Days"].values
        )
        self.resupply_per_severity = slope_r
        self.resupply_intercept = intercept_r
        # baseline (non-disrupted) delay stats
        base = df[df["Disruption_Type"].isna()]
        self.base_delay_mean = float(base["Delivery_Delay_Days"].mean())
        self.base_delay_std = float(base["Delivery_Delay_Days"].std())

    # ── Costs: freight by transport mode from GHSC records ──
    def _calibrate_costs(self):
        df = self.ghsc
        mode_means = df.groupby("Transport_Mode")["Freight_Cost_USD"].mean()
        ocean = mode_means.get("Ocean", df["Freight_Cost_USD"].mean())
        self.mode_cost_multiplier = {
            m: float(mode_means.get(m, ocean) / ocean) for m in ["Air", "Ocean", "Land"]
        }
        self.freight_cost_mean = float(df["Freight_Cost_USD"].mean())
        self.freight_cost_std = float(df["Freight_Cost_USD"].std())
        # per country-commodity base cost
        g = df.groupby(["Country", "Commodity_Type"])["Freight_Cost_USD"]
        self.freight_by_cc = g.mean().to_dict()

    # ── Demand: order volumes scaled to daily demand ──
    def _calibrate_demand(self):
        g = self.ghsc.groupby(["Country", "Commodity_Type"])["Order_Volume_Units"]
        # order volume ≈ quarterly replenishment → daily demand = volume / 90
        self.daily_demand_mean = {k: v / 90.0 for k, v in g.mean().to_dict().items()}
        self.daily_demand_cv = 0.25  # coefficient of variation for daily demand noise

    # ── Supplier reliability from data ──
    def _calibrate_reliability(self):
        g = self.ghsc.groupby("Country")["Supplier_Reliability_Score"]
        self.reliability_by_country = g.mean().to_dict()
        self.reliability_std = float(self.ghsc["Supplier_Reliability_Score"].std())
        self.otd_by_country = self.ghsc.groupby("Country")["On_Time_Delivery_%"].mean().to_dict()

    # ── Episode sampler: draws scenario contexts from real records ──
    def sample_episode_contexts(self, n_episodes: int, rng: np.random.RandomState):
        """Sample (country, commodity, disruption profile) contexts from the dataset.
        Concurrency (single/dual/compound) is assigned by sampling 1–3 co-occurring
        disruption types with empirical probabilities."""
        contexts = []
        types = list(self.disruption_type_probs.keys())
        type_p = np.array([self.disruption_type_probs[t] for t in types])
        type_p = type_p / type_p.sum()
        for i in range(n_episodes):
            row = self.ghsc.iloc[rng.randint(0, len(self.ghsc))]
            country = row["Country"]
            commodity = row["Commodity_Type"]
            # concurrency: 1, 2, or 3 simultaneous disruption types (balanced-ish design)
            n_types = 1 + (i % 3)  # deterministic rotation → balanced scenario counts
            chosen = list(rng.choice(types, size=min(n_types, len(types)), replace=False, p=type_p))
            severities = {}
            for t in chosen:
                sev_dist = self.severity_dist[t]
                sevs = np.array(sorted(sev_dist.keys()))
                probs = np.array([sev_dist[s] for s in sevs])
                severities[t] = int(rng.choice(sevs, p=probs / probs.sum()))
            scenario = {1: "single", 2: "dual", 3: "compound"}[len(chosen)]
            contexts.append({
                "episode_id": i,
                "country": country,
                "commodity": commodity,
                "lpi": self.lpi[country]["lpi"],
                "customs": self.lpi[country]["customs"],
                "infrastructure": self.lpi[country]["infrastructure"],
                "disruption_types": chosen,
                "severities": severities,
                "total_severity": sum(severities.values()),
                "scenario_type": scenario,
                "reliability": self.reliability_by_country[country],
                "base_lead_time": self.lead_time_mean.get(
                    (country, commodity), self.lead_time_global_mean),
                "base_freight": self.freight_by_cc.get(
                    (country, commodity), self.freight_cost_mean),
                "daily_demand": self.daily_demand_mean.get(
                    (country, commodity), 25000.0),
            })
        return contexts

    def summary(self):
        return {
            "n_records": len(self.ghsc),
            "countries": COUNTRIES,
            "lpi_2023": {c: self.lpi[c]["lpi"] for c in COUNTRIES},
            "p_disruption": round(self.p_disruption, 3),
            "disruption_type_probs": {k: round(v, 3) for k, v in self.disruption_type_probs.items()},
            "delay_per_severity_ols": round(self.delay_per_severity, 3),
            "resupply_per_severity_ols": round(self.resupply_per_severity, 3),
            "mode_cost_multiplier": {k: round(v, 3) for k, v in self.mode_cost_multiplier.items()},
            "lead_time_global_mean": round(self.lead_time_global_mean, 1),
        }


if __name__ == "__main__":
    cal = Calibration()
    print(json.dumps(cal.summary(), indent=2))
