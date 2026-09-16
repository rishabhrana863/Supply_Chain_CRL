"""Focused, post hoc semantic checks; does not edit the frozen implementation.

python verify_corrected_masks.py --root /path/to/extracted/frozen/archive
The update probe checks the exact logits supplied to softmax and confirms that
removing mask reapplication in an in-memory copy is detected by the assertion.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    root=args.root.resolve()
    manifest=(root/'MANIFEST_SHA256.txt').read_text().splitlines()
    for line in manifest:
        expected,relative=line.split(maxsplit=1)
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,relative
    sys.path.insert(0,str(root))
    from aor_experiment import agents as module
    from aor_experiment.causal_fitting import PairedCausalModel
    from aor_experiment.feasibility import feasible_mask, classify_action

    env=SimpleNamespace(alt_available=False,using_alternate=False,
        switch_countdown=None,corridor2=True,emergency_outstanding=True,
        pipeline=[(10,1.,1.)],day=0)
    state=np.zeros(14)
    mechanical=feasible_mask(env)
    assert mechanical.tolist()==[1,0,1,0,0,1]
    cm=PairedCausalModel(min_pairs=1)
    cm.record(cm.stratum(state),2,-.5)
    agent=module.CorrectedCRLAgent(14,causal_model=cm,seed=8)
    assert agent.decision_mask(state,env).tolist()==[1,0,0,0,0,1]

    def update_probe(update):
        a=module.CorrectedPPOAgent(14,seed=7,epochs=1,entropy_coef=0.)
        mask=np.array([1.,1.,0.,0.,0.,0.])
        a.net.forward=lambda states:(np.zeros((len(states),6)),np.zeros(len(states)),None)
        a.net.backward_and_step=lambda *args:None
        for action,reward in [(0,1.),(1,-1.)]:
            a.store(state,action,float(np.log(.5+1e-12)),reward,0.,True,mask)
        calls=[]
        original=module.MLP.softmax
        def checked(logits):
            assert np.all(logits[:,2:]==-1e9),'Update softmax received unmasked unavailable actions'
            result=original(logits)
            assert np.all(result[:,2:]==0.)
            assert np.allclose(result[:,:2],.5)
            calls.append(1)
            return result
        module.MLP.softmax=staticmethod(checked)
        try:
            update(a)
        finally:
            module.MLP.softmax=staticmethod(original)
        assert calls

    update_probe(module.CorrectedPPOAgent.update)
    source=textwrap.dedent(inspect.getsource(module.CorrectedPPOAgent.update))
    line='masked_logits[masks[mb] == 0] = -1e9'
    assert source.count(line)==1
    altered=source.replace(line,'pass  # intentionally disabled in-memory for negative control')
    ns=dict(module.__dict__)
    exec(compile(altered,'<in-memory-negative-control>','exec'),ns)
    rejected=False
    try:
        update_probe(ns['update'])
    except AssertionError as error:
        rejected='Update softmax received unmasked unavailable actions' in str(error)
    assert rejected,'Negative control must fail the mask check'
    # Shows why zero export counts do not independently establish feasibility.
    assert mechanical[1]==0
    assert classify_action(1,np.ones(6))=='executed'
    print(json.dumps(dict(frozen_manifest_entries_verified=len(manifest),
        mechanical_and_causal_intersection='pass',stored_mask_update_softmax='pass',
        removed_update_mask_negative_control='detected',
        all_ones_mask_can_label_mechanically_infeasible_action_executed=True,
        original_source_modified=False),indent=2))


if __name__=='__main__':main()
