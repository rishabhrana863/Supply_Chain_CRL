# Post hoc guidance-dosage audit

All 30 prospective seeds (52–81) were reconstructed using the unchanged frozen source and inputs. This is a post hoc diagnostic, not part of the frozen confirmatory analysis. No treatment setting, random draw, learned-policy selection rule, primary result, or original file was changed.

## Evidence and validation

- Original archive manifest: all 59 entries match.
- All protocol fingerprints match, with the documented archive path mapping from `aor_experiment/results/` to `results/`.
- Raw stream SHA-256: `90cf8c762464ffb15087decf45c38b04f156cdc23b6e97a3747fa62b9b60a89c`. This matches the decompressed GitHub export.
- Causal model summary (strata, pairs, threshold and minimum): matched each seed's retained record.
- First/last 100 training-episode reward summaries: matched for every seed; maximum absolute error 0.
- All 18,000 guided-policy aggregate deployment streams matched on every retained numeric metric. Maximum absolute cost error: 6.98e-10, attributable to floating-point CSV representation. Cost tolerance was 1e-7 dollars; other metrics used 1e-10.
- The audit covers 45,000 training episodes, 5,400,000 training decisions, 18,000 deployment streams and 2,160,000 aggregate deployment decisions. It does not replay PPO training or multi-node deployment exposure.

## Realized mask dosage

| Quantity | Training | Aggregate deployment |
|---|---:|---:|
| Removed mechanically feasible slots | 53,434 | 31,611 |
| Mechanically feasible slots, including no action | 22,252,264 | 10,064,045 |
| Slot removal rate | 0.240128% | 0.314098% |
| Slot removal rate, excluding no action | 0.317073% | 0.399934% |
| Decisions with at least one removed slot | 0.932685% | 1.363657% |
| Seed-specific slot removal range | 0.000000–1.357266% | 0.000000–1.741123% |

The numerator is sum over visited states and actions of I(mechanically feasible AND causally blocked). The primary denominator is sum of mechanically feasible action slots, including no action. No-action slots are never blocked. The intervention-only denominator excludes action zero. Decision binding is distinct from the proportion of slots removed. These are trajectory-weighted pooled ratios; per-seed raw counts and ratios are retained in the CSV.

## Reward shaping during training

| Quantity | Value |
|---|---:|
| Steps with nonzero realized increment | 4.365111% |
| Signed mean increment per step | 0.000066099 |
| Mean absolute increment per step | 0.000073129 |
| Sum absolute increment / sum absolute base reward | 0.021190% |
| Observed minimum increment | -0.002866032 |
| Observed maximum increment | 0.043138795 |

ΔR is the shaped reward actually returned by the frozen agent minus its base reward. All training steps, including zero increments, enter the averages. The absolute-reward denominator avoids cancellation around zero. Shaping is applied only to training, and exported deployment rewards are unshaped; blank deployment shaping CSV fields mean not applicable. Small average increments can still change optimization trajectories. This is neither a masking/shaping ablation nor an estimate of efficacy at a stronger dose.

## Fitted effect tables

Across seeds, 40 of 2190 estimable cells were blocked. Estimable cells per seed ranged 60–85; blocked cells ranged 0–4. An estimable cell has at least five paired observations. Missing or sparse cells have effective effect zero and stay available. `reconstructed_causal_cells.csv` gives every recorded cell, its count, raw mean, effective mean and blocked flag. Stratum bits are, in order: active port closure; active conflict; primary capacity below 0.70; inventory below 0.35; normalized LPI below 0.25.

## Seed dispersion and precision

Cost differences have mean 22,403.896, SD 216,046.560, median −4,921.622, IQR [−47,794.740, 123,587.825], and observed range [−624,035.280, 558,147.989]. The observed paired range is 1,182,183.269 dollars; it is not an individual-policy cost range or a population prediction interval. The replicate seed also changes training contexts, causal-fitting samples, training environment randomness and policy/optimizer streams. Evaluation context and environment blocks are shared. These sources are not separately identified.

The review's approximate 346 and 208 seed counts use interval-width scaling, not a power calculation. Using unrounded frozen effects, n*=30×(95% CI half-width / |estimate|)^2 gives 345.689 and 210.452, rounded up to 346 and 211. It assumes the observed effect stays fixed, ignores Holm adjustment, assumes n^−1/2 contraction, and does not target a rejection probability. It also ignores a possible common-context uncertainty floor. It must not justify significance-driven extension. Future power planning needs a prespecified minimum relevant effect and a joint seed/context variance design.

## Reproduce

Install the original archive's `requirements_paper.txt`. Extract the original archive to a directory named `frozen_extracted`, then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python replay_dosage.py --root frozen_extracted --out outputs --seeds 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
python summarize_dosage.py
```

Disjoint seed sets can run concurrently into the same outputs directory. The instrumented subclass only observes masks and reward returns; it draws no random values and delegates each policy/learning operation to the unchanged frozen implementation. Per-seed JSON files retain raw counters, reconstructed cell means, training summaries and replay discrepancies; per-seed CSV files retain replayed aggregate stream outcomes. Summary generation requires exactly all 30 seeds and all replay checks passing. Runtime versions are recorded in `source_integrity.json`.
