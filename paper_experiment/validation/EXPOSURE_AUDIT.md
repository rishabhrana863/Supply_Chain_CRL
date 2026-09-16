# Training-exposure audit — what it reports and what it requires

Section 6.3 of the manuscript reports four exposure quantities, recovered by a
post hoc replay of all 30 guided-policy training runs from the frozen
configuration:

| Quantity | Reported | Denominator |
|---|---|---|
| Action slots removed by the causal mask | 0.240% | mechanically feasible action slots offered across training |
| Decisions with at least one action removed | 0.933% | training decisions |
| Steps with a nonzero shaping term | 4.37% | training environment steps |
| Summed absolute shaping increment | 0.0212% | summed absolute base reward over those same steps |

These are properties of the **training trajectories**, not of deployment. They
cannot be recomputed from `replication_30seed_per_stream.csv.gz`, which records
converged-policy deployment streams only. `behavioral_divergence.py` in this
folder reproduces everything in Section 6.3 that the export *can* support; the
exposure rows must come from the replay.

The frozen implementation, replay harness, and all per-seed replay records are now included in this package. `verify_manifest.py` retains its completeness check and exits zero when all listed files are present and correct. `verify_complete_deposit.py` additionally checks retained replay evidence and the four exposure quantities. These quick checks do not themselves regenerate training trajectories; the full reconstruction command is in `REPRODUCE.md`.

## What the replay script must satisfy to back the Section 6.3 text

1. **It replays, it does not re-derive.** The original runs did not log exposure.
   The audit re-runs the 30 guided-policy training runs from the frozen
   configuration with instrumentation added and no change to the learning path,
   and must demonstrate that it reproduced them — the manuscript claims the
   replay matched the retained training summaries and the 18,000 aggregate
   deployment streams. The script should assert that match and fail loudly if it
   breaks, in the style of `verify_manifest.py`.

2. **The two masks are counted separately.** The mask-removal numerator is
   actions removed by the *causal* mask; the denominator is the slots the
   *mechanical feasibility* mask left available. A script that conflates them
   reports a different quantity from the one in the table.

3. **The shaping denominator is stated.** 4.37% is over training environment
   steps, not optimizer update steps. The 0.0212% figure is a ratio of two sums
   taken over the same steps — summed |shaping increment| over summed |base
   reward| — not a per-step increment compared against an episode-level
   magnitude.

4. **Exposure is reported as observed, not assigned.** Mask exposure depends on
   which states the evolving policy visited and shaping exposure on which
   actions it selected, so the audit characterizes the treatment as administered
   along those trajectories. The script's output should say so; the manuscript
   does.

## Lineage: which code produced the reported runs

The code in this repository at commit `ce850d6` is **not** the implementation the
audit instruments. `paper_experiment/agents.py` there contains a causal mask only
— `CausalModel.feasible_mask` is the *causal* mask despite its name — and
`PPOAgent.mask` returns an all-ones vector, so there is no mechanical feasibility
layer to supply the denominator in row 1 of the table above. Its `update()` also
recomputes action likelihoods without reapplying the mask used at sampling time.
`src/healthcare_crl/` is a separate DQN with a different action space and
hardcoded heuristic masks. A replay script written against either module would
report quantities other than the ones reported in Section 6.3.

The frozen `aor_experiment/` implementation is preserved, unchanged, in `frozen/AOR_Reproducibility_Package_Prospective_Replication.zip`. Its fingerprints match the prospective protocol; the guided-policy replay reproduced the retained training summaries and 18,000 aggregate deployment streams. The harness and records are in `guidance_audit/`. `REPRODUCE.md` maps the corrected implementation's entry points and dependencies and distinguishes quick verification from retraining. This evidence does not cover retraining unguided PPO or replaying network-transfer exposure.

The feasible-slot denominator includes no action, which is never causally blocked. The rate excluding no action is 0.317073%; the manuscript's 0.240% uses the inclusive denominator. Mask combination and stored-mask likelihood checks are provided in the separate post hoc `verify_corrected_masks.py`.

These materials are supplied together; cite the immutable commit containing this complete folder. The historical `ce850d6` commit remains unchanged and does not include this expanded deposit.

Nothing in the deposited export establishes which module ran. In particular, the
`infeasible_selected` column is zero throughout, but that is consistent with a
mechanical feasibility layer, with an action space in which no selected action
was ever mechanically infeasible, and with the column never being populated. It
is not evidence either way, and should not be cited as such.
