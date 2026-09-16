"""Summarize all 30 validated post hoc replays, with explicit denominators."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
records=[json.loads(p.read_text()) for p in sorted((ROOT/'outputs').glob('seed_*.json'))]
assert sorted(r['model_seed'] for r in records)==list(range(52,82)), 'Require all 30 frozen seeds'
assert all(r['checks']['training_summary_match'] and r['checks']['aggregate_streams_match'] for r in records)
assert all(r['checks']['aggregate_streams_checked']==600 for r in records)

def combined(key):
    items=[r[key] for r in records]
    keys=('steps','bound_steps','removed','feasible','feasible_interventions','shaping_nonzero',
          'base_sum','base_abs_sum','base_sq_sum','delta_sum','delta_abs_sum','delta_sq_sum')
    out={k:sum(x[k] for x in items) for k in keys}
    out['delta_min']=min(x['delta_min'] for x in items)
    out['delta_max']=max(x['delta_max'] for x in items)
    out['removed_by_action']=[sum(x['removed_by_action'][i] for x in items) for i in range(6)]
    out['feasible_by_action']=[sum(x['feasible_by_action'][i] for x in items) for i in range(6)]
    out['slot_percent']=100*out['removed']/out['feasible']
    out['intervention_slot_percent']=100*out['removed']/out['feasible_interventions']
    out['binding_step_percent']=100*out['bound_steps']/out['steps']
    out['nonzero_shaping_percent']=100*out['shaping_nonzero']/out['steps']
    out['mean_shaping']=out['delta_sum']/out['steps']
    out['mean_abs_shaping']=out['delta_abs_sum']/out['steps']
    out['absolute_shaping_to_absolute_base_percent']=100*out['delta_abs_sum']/max(out['base_abs_sum'],1e-30)
    out['seed_slot_percent_min']=100*min(x['mask_slot_fraction'] for x in items)
    out['seed_slot_percent_max']=100*max(x['mask_slot_fraction'] for x in items)
    return out

t=combined('training_dosage');e=combined('aggregate_evaluation_mask_dosage')
estimable=[sum(c['n_pairs']>=5 for c in r['causal_cells']) for r in records]
blocked=[sum(c['blocked'] for c in r['causal_cells']) for r in records]
source=json.loads((ROOT/'source_integrity.json').read_text())
summary=dict(post_hoc=True,seeds=list(range(52,82)),original_source_untouched=True,
    training=t,aggregate_evaluation_mask=e,
    effect_table=dict(estimable_cells_total=sum(estimable),blocked_cells_total=sum(blocked),
        estimable_cells_per_seed_range=[min(estimable),max(estimable)],
        blocked_cells_per_seed_range=[min(blocked),max(blocked)],
        seeds_with_any_blocking=sum(b>0 for b in blocked)),
    checks=dict(all_training_summaries_match=True,aggregate_streams_checked=18000,
        all_aggregate_streams_match=True,original_manifest_files_verified=59,
        max_training_summary_error=max(abs(v) for r in records for v in r['checks']['training_summary_errors'].values()),
        max_cost_error=max(r['checks']['max_absolute_error_by_metric']['total_cost'] for r in records)))
summary['methods_paragraphs']=[
    'A post hoc manipulation check reconstructs all 30 causal tables and replays guided-policy training under the unchanged frozen source, inputs, contexts, and random seeds. Observers count mechanically feasible slots removed by the causal mask and the reward increment actually supplied to learning, ΔR = Rᵍ − R. The slot denominator sums mechanically feasible actions over visited states, including no action; a second denominator excludes no action. Binding-state frequency counts decisions with at least one removal. Mean absolute shaping is averaged over all training steps, including zeros, and its scale is compared with the sum of absolute base rewards. These are descriptive exposure measures, not treatment ablations.',
    'Replay verification compares retained first- and last-100-episode training reward summaries and every one of the 18,000 guided-policy aggregate deployment streams. This additional instrumentation was developed after inspecting the prospective results and does not alter the frozen primary inference. Deployment rewards in the export are unshaped; reward dosage therefore comes from replayed training transitions, not differences between exported episode rewards. Mask exposure in network-transfer deployments is not reconstructed in this check.'
]
summary['results_paragraphs']=[
    f'All 30 replays matched the retained training summaries and the 18,000 aggregate guided-policy streams within numerical precision. Across {t["steps"]:,} training decisions, causal screening removed {t["removed"]:,} of {t["feasible"]:,} mechanically feasible action slots ({t["slot_percent"]:.3f}%; {t["intervention_slot_percent"]:.3f}% when no action is excluded). At least one slot was removed at {t["binding_step_percent"]:.3f}% of training decisions. Seed-specific slot-removal rates ranged from {t["seed_slot_percent_min"]:.3f}% to {t["seed_slot_percent_max"]:.3f}%. Across the fitted tables, {sum(blocked)} of {sum(estimable)} estimable stratum–action cells met the blocking rule; realized exposure depends on whether policies visit those states and the action remains mechanically feasible.',
    f'The shaping increment was nonzero on {t["nonzero_shaping_percent"]:.2f}% of training steps. Its signed mean was {t["mean_shaping"]:.6f} and its mean absolute magnitude was {t["mean_abs_shaping"]:.6f} reward units per step; the sum of absolute increments was {t["absolute_shaping_to_absolute_base_percent"]:.3f}% of the sum of absolute base rewards. Across aggregate deployment states, the mask removed {e["removed"]:,} of {e["feasible"]:,} mechanically feasible slots ({e["slot_percent"]:.3f}%), binding on {e["binding_step_percent"]:.3f}% of decisions. Thus the treatment was present but sparse at the tested settings. These averages do not show that small or infrequent perturbations are behaviorally inconsequential, nor do they separate masking from shaping.'
]
summary['limitation_paragraph']='The treatment comparison concerns sparse realized guidance under the −0.02 threshold, five-pair minimum, and λ = 0.15 shaping coefficient. Its failure to establish an average advantage cannot distinguish weak exposure from ineffective guidance at stronger doses. The post hoc replay provides an auditable reconstruction of exposure, rather than contemporaneous telemetry, and does not identify separate masking and shaping effects. Dosage on transferred multi-node trajectories remains unmeasured.'
assert t['removed_by_action'][0:3]==[0,0,0] and t['removed_by_action'][5]==0
summary['results_paragraphs'][0]+=' All observed training removals affected corridor rerouting or emergency procurement; none removed an air-expediting slot.'
summary['abstract_sentence']=f'A post hoc replay found that screening removed {t["slot_percent"]:.3f}% of feasible training slots, with mean absolute shaping of {t["mean_abs_shaping"]:.6f} reward units per step. '
(ROOT/'dosage_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

rows=[];cells=[]
for r in records:
    for phase,key in [('training','training_dosage'),('aggregate_evaluation','aggregate_evaluation_mask_dosage')]:
        row=dict(model_seed=r['model_seed'],phase=phase,shaping_applied=phase=='training')
        row.update({k:v for k,v in r[key].items() if not isinstance(v,list)})
        if phase!='training':
            for k in list(row):
                if any(s in k for s in ['shaping','base_','delta_']) and k!='shaping_applied':row[k]=''
        rows.append(row)
    for c in r['causal_cells']:
        cells.append({'model_seed':r['model_seed'],**c,'stratum':''.join(map(str,c['stratum']))})
with (ROOT/'dosage_by_seed.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (ROOT/'reconstructed_causal_cells.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)

stats=json.loads((ROOT/'seed_statistics.json').read_text())
readme=f'''# Post hoc guidance-dosage audit

All 30 prospective seeds (52–81) were reconstructed using the unchanged frozen source and inputs. This is a post hoc diagnostic, not part of the frozen confirmatory analysis. No treatment setting, random draw, learned-policy selection rule, primary result, or original file was changed.

## Evidence and validation

- Original archive manifest: all 59 entries match.
- All protocol fingerprints match, with the documented archive path mapping from `aor_experiment/results/` to `results/`.
- Raw stream SHA-256: `{source['raw_stream_sha256']}`. This matches the decompressed GitHub export.
- Causal model summary (strata, pairs, threshold and minimum): matched each seed's retained record.
- First/last 100 training-episode reward summaries: matched for every seed; maximum absolute error {summary['checks']['max_training_summary_error']:.3g}.
- All 18,000 guided-policy aggregate deployment streams matched on every retained numeric metric. Maximum absolute cost error: {summary['checks']['max_cost_error']:.3g}, attributable to floating-point CSV representation. Cost tolerance was 1e-7 dollars; other metrics used 1e-10.
- The audit covers 45,000 training episodes, {t['steps']:,} training decisions, 18,000 deployment streams and {e['steps']:,} aggregate deployment decisions. It does not replay PPO training or multi-node deployment exposure.

## Realized mask dosage

| Quantity | Training | Aggregate deployment |
|---|---:|---:|
| Removed mechanically feasible slots | {t['removed']:,} | {e['removed']:,} |
| Mechanically feasible slots, including no action | {t['feasible']:,} | {e['feasible']:,} |
| Slot removal rate | {t['slot_percent']:.6f}% | {e['slot_percent']:.6f}% |
| Slot removal rate, excluding no action | {t['intervention_slot_percent']:.6f}% | {e['intervention_slot_percent']:.6f}% |
| Decisions with at least one removed slot | {t['binding_step_percent']:.6f}% | {e['binding_step_percent']:.6f}% |
| Seed-specific slot removal range | {t['seed_slot_percent_min']:.6f}–{t['seed_slot_percent_max']:.6f}% | {e['seed_slot_percent_min']:.6f}–{e['seed_slot_percent_max']:.6f}% |

The numerator is sum over visited states and actions of I(mechanically feasible AND causally blocked). The primary denominator is sum of mechanically feasible action slots, including no action. No-action slots are never blocked. The intervention-only denominator excludes action zero. Decision binding is distinct from the proportion of slots removed. These are trajectory-weighted pooled ratios; per-seed raw counts and ratios are retained in the CSV.

## Reward shaping during training

| Quantity | Value |
|---|---:|
| Steps with nonzero realized increment | {t['nonzero_shaping_percent']:.6f}% |
| Signed mean increment per step | {t['mean_shaping']:.9f} |
| Mean absolute increment per step | {t['mean_abs_shaping']:.9f} |
| Sum absolute increment / sum absolute base reward | {t['absolute_shaping_to_absolute_base_percent']:.6f}% |
| Observed minimum increment | {t['delta_min']:.9f} |
| Observed maximum increment | {t['delta_max']:.9f} |

ΔR is the shaped reward actually returned by the frozen agent minus its base reward. All training steps, including zero increments, enter the averages. The absolute-reward denominator avoids cancellation around zero. Shaping is applied only to training, and exported deployment rewards are unshaped; blank deployment shaping CSV fields mean not applicable. Small average increments can still change optimization trajectories. This is neither a masking/shaping ablation nor an estimate of efficacy at a stronger dose.

## Fitted effect tables

Across seeds, {sum(blocked)} of {sum(estimable)} estimable cells were blocked. Estimable cells per seed ranged {min(estimable)}–{max(estimable)}; blocked cells ranged {min(blocked)}–{max(blocked)}. An estimable cell has at least five paired observations. Missing or sparse cells have effective effect zero and stay available. `reconstructed_causal_cells.csv` gives every recorded cell, its count, raw mean, effective mean and blocked flag. Stratum bits are, in order: active port closure; active conflict; primary capacity below 0.70; inventory below 0.35; normalized LPI below 0.25.

## Seed dispersion and precision

Cost differences have mean 22,403.896, SD 216,046.560, median −4,921.622, IQR [−47,794.740, 123,587.825], and observed range [−624,035.280, 558,147.989]. The observed paired range is 1,182,183.269 dollars; it is not an individual-policy cost range or a population prediction interval. The replicate seed also changes training contexts, causal-fitting samples, training environment randomness and policy/optimizer streams. Evaluation context and environment blocks are shared. These sources are not separately identified.

The review's approximate 346 and 208 seed counts use interval-width scaling, not a power calculation. Using unrounded frozen effects, n*=30×(95% CI half-width / |estimate|)^2 gives {stats['total_cost']['ci_width_scaling_n']:.3f} and {stats['intervention_rate']['ci_width_scaling_n']:.3f}, rounded up to 346 and 211. It assumes the observed effect stays fixed, ignores Holm adjustment, assumes n^−1/2 contraction, and does not target a rejection probability. It also ignores a possible common-context uncertainty floor. It must not justify significance-driven extension. Future power planning needs a prespecified minimum relevant effect and a joint seed/context variance design.

## Reproduce

Install the original archive's `requirements_paper.txt`. Extract the original archive to a directory named `frozen_extracted`, then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python replay_dosage.py --root frozen_extracted --out outputs --seeds 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
python summarize_dosage.py
```

Disjoint seed sets can run concurrently into the same outputs directory. The instrumented subclass only observes masks and reward returns; it draws no random values and delegates each policy/learning operation to the unchanged frozen implementation. Per-seed JSON files retain raw counters, reconstructed cell means, training summaries and replay discrepancies; per-seed CSV files retain replayed aggregate stream outcomes. Summary generation requires exactly all 30 seeds and all replay checks passing. Runtime versions are recorded in `source_integrity.json`.
'''
(ROOT/'README.md').write_text(readme)
print(json.dumps({k:summary[k] for k in ['training','aggregate_evaluation_mask','effect_table','checks']},indent=2))
