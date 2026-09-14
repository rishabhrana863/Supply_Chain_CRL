# FDA validation of recovery-time distributions

**Decision:** FDA drug-shortage data do **not** validate the absolute recovery-time distributions in either the AOR paper or the available architecture-fit study. They instead reveal a large difference in timescale and construct. The AOR comparisons remain valid as within-simulator policy comparisons, but the recovery endpoint should be described as **operational service stabilization**, not national drug-shortage resolution.

## External benchmark

The live openFDA file is a rolling status file, so historical durations were reconstructed from 39 archived FDA CSV snapshots dated 2019-12-29 through 2023-12-21. Following the HHS/ASPE archival method, a package resolves at its first archived update marked *Resolved*; unresolved packages are right-censored at their last archived observation. Packages were collapsed to 9-digit product NDCs, and a product resolved only when all observed packages resolved. The cohort contains initial postings from 2018 through 2023.

FDA defines a resolved shortage at the national-market level: available supply is judged sufficient to cover market demand. That endpoint is not equivalent to restoration of one simulated service stream.

| FDA result | Estimate |
|---|---:|
| Product episodes | 2,042 |
| Observed resolutions / right-censored | 850 / 1,192 |
| Kaplan–Meier median | 992 days (2.72 years) |
| Recovery by day 110 | 1.95% (95% CI 1.43%–2.66%) |
| Favorable day-110 upper bound | 5.39% |
| Restricted mean unresolved time through day 110 | 109.23 days |
| Unresolved at one year | 87.68% |
| 2020–2023 day-110 sensitivity | 2.42% |

The reconstruction closely cross-checks the published HHS/ASPE result: 2,038 matched products and a 2.55-year median (95% CI 2.44–2.72).

## AOR paper

The comparison uses AOR stream-level records from the 30-seed replication and conditions on `dipped=True`, avoiding the structural zero assigned to streams that never experienced a service failure. It covers four multi-node transfer profiles and both agents.

| Distribution | Median | Recovery by day 110 | RMST through day 110 |
|---|---:|---:|---:|
| FDA product shortages | 992 days | 1.95% | 109.23 days unresolved |
| AOR transfer profiles, range across 8 profile-agent cells | 41–56 days | 92.85%–97.18% | 49.22–65.28 days to recovery |

The FDA-to-AOR median ratio is 17.7–24.2×, and the maximum cumulative-distribution gap over days 0–110 is 90.9–95.2 percentage points. This is substantive falsification of equal timescales, not a borderline fit question.

**Verdict:** retain claims about relative policy performance inside the simulator. Do not claim that FDA data validate the absolute `recovery_days` distribution. The construct is defensible as short-horizon operational service stabilization.

### Manuscript-ready AOR language

> We externally calibrated the recovery-time scale using 39 archived FDA drug-shortage snapshots. A product-level reconstruction of 2,042 shortage episodes yielded a Kaplan–Meier median of 992 days and 1.95% resolution by day 110. In contrast, simulated streams that experienced a service dip had median recovery times of 41–56 days and 92.85%–97.18% recovery by day 110 across the four transfer profiles. Because FDA resolution denotes restoration of adequate national market supply whereas our endpoint denotes stabilization of a local rolling service measure, these data do not establish distributional equivalence. We therefore interpret `recovery_days` as operational service stabilization and separate that calibration limitation from the within-simulator policy comparison.

## Architecture-fit paper

Only the one-replicate, 360-row **PRE-EVIDENCE pilot** is suitable for calculation from the preserved project files. Its rows average five evaluation streams, so conditional means and day-110 recovery were reconstructed by weighting the dipped fraction. The later v4 run cannot be used because CRL action masking was bypassed, and no completed v4.1 output was available.

| Pilot architecture | Dipped streams | Conditional mean | Recovery by day 110 | Gap vs FDA |
|---|---:|---:|---:|---:|
| Causal heuristic | 55 | 48.3 days | 100.0% | +98.0 pp |
| CRL | 281 | 92.9 days | 88.3% | +86.3 pp |
| PPO | 262 | 93.6 days | 90.8% | +88.9 pp |
| Rules | 170 | 88.3 days | 81.8% | +79.8 pp |

These pilot values are also much faster than national FDA shortage resolution and are not confirmatory results.

An exploratory FDA market-complexity proxy was directionally consistent with longer persistence: events with high manufacturer/product counts lasted longer than low-count events (log-rank p=0.0018). This is not confirmatory validation because the proxy is post hoc and partly mechanical—the market event cannot resolve until all included products clear. Shortage-reason completeness is not construct-equivalent to delayed, noisy, or missing operational sensor data and cannot validate the paper's signal-quality manipulation.

**Verdict:** the architecture-fit paper presently has no FDA-validated recovery-time distribution. Report the complexity result only as exploratory triangulation, omit any signal-quality validation claim, and rerun the repaired v4.1 design before confirmatory interpretation.

### Manuscript-ready architecture language

> FDA shortage data were used as an external calibration benchmark rather than as a causal validation dataset. National product shortages had a reconstructed 992-day median and 1.95% resolution by day 110, whereas the available one-replicate pilot produced 81.8%–100.0% conditional recovery by day 110 across architectures. The endpoints therefore operate on different scales. A post hoc manufacturer/product-count proxy was associated with longer FDA shortage persistence, but the measure is partly coupled to the all-products-clear resolution rule and is reported only as exploratory triangulation. FDA shortage-reason completeness was not treated as a valid proxy for operational signal delay, noise, or missingness. Confirmatory external-validity claims await the repaired v4.1 run.

## Recommended paper changes

1. Rename or explicitly define the simulated endpoint as **operational service stabilization**.
2. Separate internal policy efficacy from external timescale calibration.
3. State that FDA data reject absolute distributional equivalence over the 110-day horizon.
4. Treat the FDA complexity result as exploratory and proxy-coupled.
5. Do not use FDA shortage-reason missingness to validate signal quality.
6. For national-market claims, extend the simulator to multi-year capacity, manufacturer, and market-clearing dynamics.

## Sources and audit trail

- FDA Drug Shortages database and national resolution definition: https://www.accessdata.fda.gov/scripts/drugshortages/
- openFDA Drug Shortages API documentation: https://open.fda.gov/apis/drug/drugshortages/
- HHS/ASPE archival method and published duration estimates: https://www.ncbi.nlm.nih.gov/books/NBK611681/
- Archived FDA CSV source: https://www.accessdata.fda.gov/scripts/drugshortages/Drugshortages.cfm

The companion workbook contains the reconstructed product episodes, proxy events, snapshot dates, SHA-256 checksums, Kaplan–Meier curves, AOR cell-level comparisons, architecture pilot calculations, formulas, and sensitivity results.
