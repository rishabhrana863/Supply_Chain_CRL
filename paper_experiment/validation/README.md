# FDA recovery-time validation

External calibration of the simulated `recovery_days` distributions against FDA
drug-shortage data. The finding is negative and deliberately so: FDA data do not
validate either paper's absolute recovery-time distribution, and the comparison
falsifies distributional equivalence rather than confirming it. See
`FDA_Recovery_Time_Validation_Report.md` for the full argument and the
manuscript-ready language.

## Contents

| File | What it is |
|---|---|
| `FDA_Recovery_Time_Validation_Report.md` | The report: benchmark, both paper comparisons, verdicts, recommended changes |
| `FDA_Recovery_Time_Validation.xlsx` | Workbook: reconstructed episodes, Kaplan–Meier curves, per-cell comparisons, audit trail |
| `replication_30seed_per_stream.csv.gz` | Exact 30-seed AOR stream export used by the comparison (252,000 rows) |
| `verify_fda_benchmark.py` | Recomputes the headline FDA figures from the workbook's raw episode rows |
| `verify_manifest.py` | Checks file checksums and validates the stream export against the expected run |
| `MANIFEST.json` | SHA-256 and structural expectations for the reproducibility files |

## Reproducing the FDA benchmark

```
pip install openpyxl
python paper_experiment/validation/verify_fda_benchmark.py
```

This recomputes the Kaplan–Meier median, day-110 recovery, one-year survival and
the drug-clustered confidence interval directly from the `FDA Product Episodes`
sheet, so no headline number rests on a summary cell alone.

## Confidence intervals

The day-110 interval is **clustered by drug**. The 2,042 product episodes span
only 298 distinct generic names, and co-listed NDCs of one drug resolve together
by construction, so a binomial interval treating episodes as independent is too
narrow. The workbook records both:

- `FDA Benchmark!B10:B11` — the clustered bootstrap interval, 0.59%–3.73%. Quote this one.
- `FDA Benchmark!A38:C41` — the superseded log–log Greenwood interval, 1.43%–2.66%, retained for traceability against earlier drafts.
- `Drug Clusters` — the 298 resampling units, with episode, resolution and censoring counts.

The widening changes no conclusion: the FDA median remains 17.7–24.2× the AOR
range and the day-110 gap exceeds 90 percentage points.

## AOR comparison: which cells are primary

The primary range covers the **four multi-node transfer profiles**, eight
profile-agent cells, `AOR Validation` rows 8–15: medians 41–56 days. The
single-node `aggregate_baseline` profile is excluded and reported as sensitivity
only; including its two cells widens the range to 41–68 days without changing
the comparison. Each row is labelled in the `Validation set` column, and the two
ranges are stated side by side at `AOR Validation!A30:D33`. Executive Summary
formulas reference rows 8–15 and are therefore on the primary basis.

## AOR stream-level replication data

The exact stream-level export behind the workbook's `AOR Validation` sheet is
included as `replication_30seed_per_stream.csv.gz`. It contains 252,000 rows
covering seeds 52–81 and both raw agent keys, `crl` and `ppo`.

To verify the complete package from a clean clone:

```
python paper_experiment/validation/verify_manifest.py
```

`verify_manifest.py` does not merely checksum the file. It confirms the file is
the specific run the report describes by checking 252,000 total rows, the
30-seed range, and all ten profile-agent cells: dipped streams, recovery by day
110, censoring, and Kaplan–Meier medians. A file from a different run fails with
the discrepancies listed, which is the check that would have caught the earlier
frozen-versus-v12 mismatch.

## Limitations carried into the papers

- FDA resolution means adequate supply for national market demand; the simulator's
  endpoint is restoration of a local rolling service measure. Different constructs.
- The architecture-fit figures come from a one-replicate pilot. The v4 run was
  invalidated (CRL action masking bypassed) and v4.1 was not available.
- The market-complexity proxy is post hoc and partly coupled to the
  all-products-clear resolution rule. Exploratory only.
- FDA shortage-reason completeness is not a construct-valid proxy for delayed,
  noisy or missing operational sensor data.
