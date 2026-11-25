#!/usr/bin/env python3
"""Check Fig. 3's 156 correlations from raw weighted counts, without permutations.

Uses weighted first/second energy moments instead of expanding shots as in the
historical generator. Permutation p-values remain archived with their seed/code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'revision_v7'))
from diagnostic_controls import faithful_reference
from audit_physical_data import light_cones


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=HERE/'generated')
    args=parser.parse_args()
    hashes={}
    def read(path):
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    read(ROOT/'data/frozen_rung1.json');read(ROOT/'results/exact51_plan.json')
    reference=read(ROOT/'results/exact51_cross_block.json')
    _,_,linear,quadratic,_,components=faithful_reference()
    blocks=sorted(components,key=lambda b:(-len(b),b))
    if blocks != [b['qubits'] for b in reference['blocks']]:
        raise AssertionError('Block ordering differs.')
    checked=[]
    for context,name in [('defaults','exact51_counts.json'),('matched','exact51_matched_counts.json')]:
        raw=read(ROOT/'results'/name)
        counts=raw['circuits']['faithful_exact51_p1']['counts']
        mult=np.asarray(list(counts.values()),dtype=float)
        bits=np.array([[int(x) for x in key.replace(' ','')[::-1]] for key in counts],dtype=float)
        weights=mult/mult.sum()
        energies=[]
        for block in blocks:
            local=bits[:,block]
            energies.append(local@linear[block]+np.einsum('bi,ij,bj->b',local,quadratic[np.ix_(block,block)],local))
        energy=np.column_stack(energies)
        centered=energy-weights@energy
        covariance=centered.T@(weights[:,None]*centered)
        sd=np.sqrt(np.diag(covariance))
        correlation=covariance/np.outer(sd,sd)
        ops=read(ROOT/'results/ibm_circuits'/('exact51_'+context+'.ops.json'))
        _,cones=light_cones(ops,blocks)
        expected=reference['runs'][context]
        if raw['job_id']!=expected['job_id'] or int(mult.sum())!=expected['shots']:
            raise AssertionError('Job or shot mismatch.')
        differences=[]
        for pair in expected['pairs']:
            i,j=pair['blocks'];r=float(correlation[i,j]);differences.append(abs(r-pair['r']))
            np.testing.assert_allclose(r,pair['r'],rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(r*np.sqrt(mult.sum()),pair['z'],rtol=1e-10,atol=1e-10)
            if bool(cones[i]&cones[j]) != (pair['cls']=='cone'):
                raise AssertionError('Backward-cone classification mismatch.')
        variance=float(covariance.sum())
        np.testing.assert_allclose(variance,expected['total_energy_variance'],rtol=1e-10)
        checked.append(dict(context=context,job_id=raw['job_id'],shots=int(mult.sum()),
                            pairs_checked=len(differences),max_absolute_r_difference=max(differences),
                            cone_classification_verified=True))
    args.out_dir.mkdir(parents=True,exist_ok=True)
    output=dict(status='PASS',method='Independent weighted energy moments from raw bitstring counts',
                scope='Pearson r, z, total variance and cone classification; permutation p-values not repeated.',
                sources_sha256=hashes,runs=checked)
    (args.out_dir/'cross_block_points.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(checked))


if __name__=='__main__':main()
