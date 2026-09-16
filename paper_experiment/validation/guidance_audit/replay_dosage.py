"""Post hoc observational instrumentation of the unchanged frozen CRL learner.

Run with --root extracted_archive --seeds 52 ... --out output_directory.
No random draws or learning changes are introduced by the observer. Source
fingerprints are checked before imports; all training steps are counted.
Replay validity is checked against retained training summaries and every
aggregate-baseline CRL deployment stream. No new primary inference is done.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


class Counters:
    def __init__(self):
        self.steps = self.bound_steps = self.removed = self.feasible = 0
        self.feasible_interventions = self.shaping_nonzero = 0
        self.base_sum = self.base_abs_sum = self.delta_sum = self.delta_abs_sum = 0.
        self.delta_min = self.delta_max = 0.
        self.base_sq_sum = self.delta_sq_sum = 0.
        self.removed_by_action = np.zeros(6, dtype=int)
        self.feasible_by_action = np.zeros(6, dtype=int)

    def mask(self, mechanical, combined):
        removed = ((mechanical > 0) & (combined == 0)).astype(int)
        self.steps += 1
        self.bound_steps += int(removed.any())
        self.removed += int(removed.sum())
        self.feasible += int(mechanical.sum())
        self.feasible_interventions += int(mechanical[1:].sum())
        self.removed_by_action += removed
        self.feasible_by_action += mechanical.astype(int)

    def reward(self, base, shaped):
        delta = shaped - base
        self.shaping_nonzero += int(delta != 0)
        self.base_sum += base
        self.base_abs_sum += abs(base)
        self.base_sq_sum += base * base
        self.delta_sum += delta
        self.delta_abs_sum += abs(delta)
        self.delta_sq_sum += delta * delta
        self.delta_min = min(self.delta_min, delta)
        self.delta_max = max(self.delta_max, delta)

    def result(self):
        out = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in vars(self).items()}
        out.update(mask_slot_fraction=self.removed/self.feasible,
                   mask_intervention_slot_fraction=self.removed/self.feasible_interventions,
                   mask_bound_step_fraction=self.bound_steps/self.steps,
                   mean_shaping=self.delta_sum/self.steps,
                   mean_abs_shaping=self.delta_abs_sum/self.steps,
                   abs_shaping_to_abs_base=self.delta_abs_sum/max(self.base_abs_sum,1e-30),
                   shaping_nonzero_fraction=self.shaping_nonzero/self.steps)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    for line in (root/'MANIFEST_SHA256.txt').read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected, relative
    sys.path.insert(0, str(root))
    from aor_experiment import _paths
    from calibration import Calibration
    from aor_experiment.agents import CorrectedCRLAgent
    from aor_experiment.causal_fitting import fit_paired_causal_model
    from aor_experiment.feasibility import feasible_mask
    from aor_experiment.runner import train_agent, run_episode

    class ObservedCRL(CorrectedCRLAgent):
        def decision_mask(self, state, env):
            combined = super().decision_mask(state, env)
            self.counter.mask(feasible_mask(env), combined)
            return combined

        def shaped_reward(self, reward, state, action):
            shaped = super().shaped_reward(reward, state, action)
            self.counter.reward(reward, shaped)
            return shaped

    records = [json.loads(p.read_text()) for p in (root/'results').glob('replication_s*_protocol_records.json')]
    expected_train = {x['label']: x for d in records for x in d['training']}
    expected_causal = {x['model_seed']: x for d in records for x in d['causal_models']}
    frozen = pd.read_csv(root/'results/replication_30seed_per_stream.csv')
    frozen = frozen[(frozen.profile=='aggregate_baseline') & (frozen.agent=='crl')]
    cal = Calibration(seed=42)
    evaluation = cal.sample_episode_contexts(200, np.random.RandomState(9000))
    for seed in args.seeds:
        started=time.time()
        print(f'Seed {seed}: reconstructing causal table', flush=True)
        cm = fit_paired_causal_model(cal, n_contexts=400, seed=seed+5000)
        assert {'model_seed':seed, **cm.summary()} == expected_causal[seed]
        cells=[]
        for st, actions in cm.table.items():
            for action, values in actions.items():
                cells.append(dict(stratum=[int(x) for x in st], action=action,
                    n_pairs=len(values), raw_mean=float(np.mean(values)),
                    effective_mean=cm.effect(st,action),
                    blocked=cm.effect(st,action)<cm.negative_threshold))
        print(f'Seed {seed}: causal model matched, training', flush=True)
        crl=ObservedCRL(state_dim=14, causal_model=cm, seed=seed)
        crl.counter=Counters()
        contexts=cal.sample_episode_contexts(1500,np.random.RandomState(30000+seed))
        training=train_agent(crl,cal,contexts,seed_base=seed,label=f'crl-s{seed}',log_every=500)
        train_errors={k:training[k]-expected_train[f'crl-s{seed}'][k]
                      for k in ('reward_first_100','reward_last_100')}
        train_dose=crl.counter.result()
        train_match=all(abs(v)<=1e-10 for v in train_errors.values())
        print(f'Seed {seed}: training summary match={train_match}; evaluating 600 frozen streams',flush=True)
        crl.counter=Counters()
        expected=frozen[frozen.model_seed==seed].set_index(['episode_id','policy_stream'])
        metric_errors={}
        rows=[]
        for episode,ctx in enumerate(evaluation):
            for stream in range(3):
                metrics=run_episode(crl,ctx,cal,env_seed=63000+episode,
                    policy_seed=700000+seed*1000000+episode*100+stream)
                original=expected.loc[(episode,stream)]
                rows.append(dict(model_seed=seed,episode_id=episode,policy_stream=stream,**metrics))
                for key,value in metrics.items():
                    if key in original and isinstance(value,(int,float,np.integer,np.floating)):
                        error=abs(float(value)-float(original[key]))
                        metric_errors[key]=max(error,metric_errors.get(key,0.))
        # CSV round-tripping permits floating-point noise, never policy changes.
        evaluation_match=all(v <= (1e-7 if k=='total_cost' else 1e-10)
                             for k,v in metric_errors.items())
        result=dict(model_seed=seed,post_hoc=True,causal_summary=cm.summary(),
            causal_cells=cells,training=training,training_dosage=train_dose,
            aggregate_evaluation_mask_dosage=crl.counter.result(),
            checks=dict(training_summary_match=train_match,training_summary_errors=train_errors,
                aggregate_streams_checked=len(rows),aggregate_streams_match=evaluation_match,
                max_absolute_error_by_metric=metric_errors),elapsed_s=time.time()-started)
        (args.out/f'seed_{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
        pd.DataFrame(rows).to_csv(args.out/f'seed_{seed}_aggregate_replay.csv',index=False)
        print(f'Seed {seed}: complete in {time.time()-started:.1f}s, evaluation match={evaluation_match}',flush=True)
        if not (train_match and evaluation_match):
            raise RuntimeError('Replay differs from frozen evidence; do not label dosage as original-run measurement')


if __name__=='__main__':
    main()
