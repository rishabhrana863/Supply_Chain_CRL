"""Verify deposited source, behavioral results, and retained replay evidence.

This quick check does not retrain agents. To reconstruct training exposure,
run guidance_audit/replay_dosage.py as documented in REPRODUCE.md.
Run without Python's -O flag: the frozen checks use assertions.
"""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from frozen_archive import open_frozen_archive

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def main():
    require(__debug__, 'Run without -O: archived scripts use assertions')
    subprocess.run([sys.executable, str(HERE/'verify_manifest.py')], check=True)
    behavior = subprocess.run([sys.executable, str(HERE/'behavioral_divergence.py')],
                              check=True, capture_output=True, text=True)
    require(behavior.stdout == (HERE/'behavioral_divergence_output.txt').read_text(),
            'Behavioral output differs from recorded output')
    print('PASS: 29 behavioral checks; recorded output matches', flush=True)
    archive = HERE/'frozen/AOR_Reproducibility_Package_Prospective_Replication.zip'
    integrity = json.loads((HERE/'guidance_audit/source_integrity.json').read_text())
    with open_frozen_archive(archive) as stream, zipfile.ZipFile(stream) as z:
        for member in z.namelist():
            p = Path(member)
            require(not p.is_absolute() and '..' not in p.parts, 'Unsafe archive member')
        manifest = z.read('MANIFEST_SHA256.txt').decode().splitlines()
        for line in manifest:
            expected, name = line.split(maxsplit=1)
            name = Path(name).as_posix()
            with z.open(name) as f:
                require(digest_stream(f) == expected, f'Frozen checksum mismatch: {name}')
        for item in integrity['protocol_fingerprints']:
            name = item['path'].replace('aor_experiment/results/', 'results/')
            with z.open(name) as f:
                require(digest_stream(f) == item['expected'], f'Protocol mismatch: {name}')
        with z.open('results/replication_30seed_per_stream.csv') as f:
            raw_hash = digest_stream(f)
        with gzip.open(HERE/'replication_30seed_per_stream.csv.gz', 'rb') as f:
            require(digest_stream(f) == raw_hash == integrity['raw_stream_sha256'],
                    'Frozen and public stream exports differ')
        expected_training = {}
        for name in z.namelist():
            if name.startswith('results/replication_s') and name.endswith('_protocol_records.json'):
                for row in json.loads(z.read(name))['training']:
                    expected_training[row['label']] = row
        with tempfile.TemporaryDirectory(prefix='aor-frozen-check-') as temp:
            z.extractall(temp)
            subprocess.run([sys.executable, str(HERE/'verify_corrected_masks.py'),
                            '--root', temp], check=True)

    raw = pd.read_csv(HERE/'replication_30seed_per_stream.csv.gz')
    expected = raw[(raw.profile == 'aggregate_baseline') & (raw.agent == 'crl')]
    keys = ['model_seed', 'episode_id', 'policy_stream']
    require(not expected.duplicated(keys).any(), 'Duplicate frozen deployment keys')
    expected = expected.set_index(keys).sort_index()
    records = [json.loads(p.read_text()) for p in sorted((HERE/'guidance_audit/outputs').glob('seed_*.json'))]
    require(sorted(r['model_seed'] for r in records) == list(range(52, 82)), 'Need all 30 seeds exactly once')
    total_streams = 0
    max_cost_error = 0.
    for record in records:
        seed = record['model_seed']
        require(record['post_hoc'], 'Replay must be labelled post hoc')
        train = record['training']
        original = expected_training[f'crl-s{seed}']
        for name in ['n_train_episodes', 'reward_first_100', 'reward_last_100', 'entropy_schedule']:
            require(train[name] == original[name], f'Training summary mismatch: seed {seed}, {name}')
        replay = pd.read_csv(HERE/f'guidance_audit/outputs/seed_{seed}_aggregate_replay.csv')
        require(len(replay) == 600 and not replay.duplicated(keys).any(), f'Bad replay keys: {seed}')
        replay = replay.set_index(keys).sort_index()
        target = expected.loc[expected.index.get_level_values('model_seed') == seed]
        require(replay.index.equals(target.index), f'Deployment keys differ: {seed}')
        require(set(replay.columns) == set(record['checks']['max_absolute_error_by_metric']),
                f'Replay metric coverage differs: {seed}')
        for metric in replay.columns:
            actual = replay[metric].to_numpy(dtype=float)
            want = target[metric].to_numpy(dtype=float)
            error = np.abs(actual - want)
            tolerance = 1e-7 if metric == 'total_cost' else 1e-10
            require(np.isfinite(error).all() and (error <= tolerance).all(),
                    f'Deployment mismatch: seed {seed}, {metric}')
            if metric == 'total_cost':
                max_cost_error = max(max_cost_error, float(error.max()))
        total_streams += len(replay)

    totals = {key: sum(r['training_dosage'][key] for r in records)
              for key in ['steps', 'removed', 'feasible', 'bound_steps', 'shaping_nonzero',
                          'delta_abs_sum', 'base_abs_sum']}
    require(totals['steps'] == 5400000, 'Incorrect number of training decisions')
    exposure = {
        'removed_feasible_slots_pct': 100 * totals['removed'] / totals['feasible'],
        'binding_decisions_pct': 100 * totals['bound_steps'] / totals['steps'],
        'nonzero_shaping_pct': 100 * totals['shaping_nonzero'] / totals['steps'],
        'absolute_shaping_to_base_pct': 100 * totals['delta_abs_sum'] / totals['base_abs_sum'],
    }
    claims = [(0.240, 0.0005), (0.933, 0.0005), (4.37, 0.005), (0.0212, 0.00005)]
    for (name, actual), (printed, tolerance) in zip(exposure.items(), claims):
        require(abs(actual - printed) <= tolerance, f'Exposure differs from manuscript: {name}')
    summary = json.loads((HERE/'guidance_audit/dosage_summary.json').read_text())
    for name, actual in totals.items():
        require(actual == summary['training'][name], f'Exposure summary mismatch: {name}')
    print(json.dumps({
        'status': 'pass', 'new_training_runs_performed': 0,
        'frozen_manifest_files': len(manifest),
        'protocol_fingerprints': len(integrity['protocol_fingerprints']),
        'behavioral_checks': 29, 'replay_training_summaries_checked': len(records),
        'replay_aggregate_streams_checked': total_streams,
        'max_cost_error': max_cost_error, 'training_exposure': exposure,
        'scope': 'Retained guided-policy aggregate replay evidence; PPO retraining and network-transfer replay are not covered.'
    }, indent=2))


if __name__ == '__main__':
    main()
