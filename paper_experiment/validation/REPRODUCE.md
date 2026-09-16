# Reproduce the seed risk evidence

Run the commands below from this validation directory. The deposit includes the original frozen experiment archive unchanged, the post hoc exposure replay harness, all 30 per-seed replay records and deployment outputs, and the behavioral-divergence analysis. Cite the immutable repository commit containing this folder.

## Quick verification

Use Python 3.12 with NumPy, pandas, SciPy, and openpyxl installed. The recorded runtime is in `guidance_audit/source_integrity.json`. Do not run with Python's `-O` option because archived checks use assertions.

```bash
python verify_complete_deposit.py
```

This checks all manifest-listed files, all 59 internal frozen checksums, the 13 protocol fingerprints, equality of the public and frozen stream exports, all 29 behavioral checks, the mechanical/causal mask intersection and stored-mask update behavior, retained training summaries for all 30 guided seeds, all 18,000 recorded aggregate replay streams, and the four exposure quantities recalculated from retained training counters. It exits nonzero on a discrepancy. It does not rerun training; retained counter checks are distinct from independently regenerating those counters.

The original verifier remains unchanged:

```bash
python verify_manifest.py
python behavioral_divergence.py
python verify_fda_benchmark.py
```

The first now exits zero because the missing materials have been supplied and individually listed in `MANIFEST.json`. The second reproduces the recorded text output byte for byte. The third recomputes the external benchmark from its workbook.

## Reconstruct training exposure

Stage the original archive in a new directory. For GitHub API transfer, the ZIP is stored as ordered checksummed parts. The adapter automatically reconstructs it in a temporary file and verifies its original SHA-256 (`f818b878e86d16036d16f314f52442b2d0a8f2367123a138c27302cf1276db63`) before extraction. No source or result byte is changed. The adapter verifies its checksums and copies the top-level `results/` to the location expected by its original analysis modules. It does not modify source or the preserved archive.

```bash
python prepare_frozen_analysis.py --work frozen_work
cp -R guidance_audit replay_work
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python replay_work/replay_dosage.py --root frozen_work --out replay_work/outputs --seeds 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
python replay_work/summarize_dosage.py
```

Run the replay command successfully before summarizing; do not continue after a nonzero exit. Work in the separate copy so reruns do not overwrite the deposited evidence or invalidate its manifest. The harness verifies frozen source before imports and checks reconstructed training summaries and aggregate deployment outcomes against original records. It observes masks and reward increments without drawing additional random values or changing the learner. Thirty guided-policy training runs are required for this reconstruction. The replay covers neither unguided PPO retraining nor network-transfer exposure.

## Recompute the frozen statistical analysis

In a new directory, use the adapter's `--run` option:

```bash
python prepare_frozen_analysis.py --work analysis_work --run
```

This runs the original prospective analysis and postrun sensitivity modules. Original result files remain in the preserved ZIP. The Stage 1 and prospective p values are in `results/prospective_extension_results.json` inside that ZIP.

## Implementation map

Within `frozen/AOR_Reproducibility_Package_Prospective_Replication.zip`:

- `aor_experiment/run_aor_experiment.py` constructs the corrected agents.
- `aor_experiment/agents.py` defines `CorrectedPPOAgent` and `CorrectedCRLAgent`, combining masks at selection and reusing stored masks during updates.
- `aor_experiment/feasibility.py` supplies mechanical feasibility.
- `aor_experiment/causal_fitting.py` supplies the paired causal model.
- `aor_experiment/tests/test_protocol.py` contains the original protocol tests.
- `paper_experiment/agents.py` supplies the reused MLP. Its legacy PPO implementation is not the corrected runner's agent.
- `PROSPECTIVE_EXTENSION_PROTOCOL.md`, data, results, run logs, and the original internal manifest are preserved together.

The separate `verify_corrected_masks.py` is a post hoc semantic check. Its negative control removes mask reuse only from an in-memory copy and confirms the check rejects the defect. The frozen source is unchanged.

## Citation and scope

Cite the immutable repository commit containing this complete folder. The earlier `ce850d6` citation describes the smaller historical deposit and cannot attest to these additional files.

Training exposure is reconstructed post hoc. Zero `infeasible_selected` values alone do not establish which implementation ran or that mechanical feasibility was checked independently.
