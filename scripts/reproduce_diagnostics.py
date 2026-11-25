#!/usr/bin/env python3
"""Verify released inputs and reproduce diagnostics offline, without the manuscript.

Default: raw-count point checks, exact controls, calibration audit and Figures 1-4.
--full also repeats all 14 block bootstraps and the three campaign bootstraps.
No IBM account, new hardware jobs or manuscript files are required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT/'analysis/revision_v7'
V8 = ROOT/'analysis/revision_v8'


def compare(actual, expected, pointer='root'):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError('Different keys at '+pointer)
        for k in expected: compare(actual[k], expected[k], pointer+'.'+k)
    elif isinstance(expected, list):
        if len(actual) != len(expected): raise AssertionError('Different lengths at '+pointer)
        for i, (a,b) in enumerate(zip(actual,expected)): compare(a,b,f'{pointer}[{i}]')
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10):
            raise AssertionError(f'Numerical mismatch at {pointer}: {actual} != {expected}')
    elif actual != expected:
        raise AssertionError('Mismatch at '+pointer)


def canonical_inputs(value):
    """Historical JSON hashes retain original paths; PATH_MAPPING documents relocation."""
    if isinstance(value, dict):
        return {k.replace('paper/','analysis/',1) if k.startswith('paper/') else k:
                canonical_inputs(v) for k,v in value.items()}
    if isinstance(value,list): return [canonical_inputs(v) for v in value]
    if isinstance(value,str) and value.startswith('paper/'):
        return value.replace('paper/','analysis/',1)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    report=dict(status='PASS',full_reanalysis=args.full,steps=[])
    manifest=json.loads((ROOT/'DATA_MANIFEST.json').read_text())
    for name,entry in manifest['files'].items():
        data=(ROOT/name).read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise AssertionError('Input checksum mismatch: '+name)
    report['verified_data_files']=len(manifest['files'])
    print('PASS: released input hashes',flush=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             MPLCONFIGDIR=str(out/'matplotlib'))

    for key in ('PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MPLCONFIGDIR'):
        os.environ[key]=env[key]

    def run(name,command):
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True)
        (out/(name+'.log')).write_text(result.stdout)
        if result.returncode: raise RuntimeError(name+' failed:\n'+result.stdout)
        report['steps'].append(dict(name=name,status='PASS'))
        print('PASS: '+name,flush=True)

    def read(path): return json.loads(path.read_text())

    # Scientific tests only: historical manuscript-layout tests are deliberately
    # outside this distribution. No tests are silently skipped by discovery.
    sys.path.insert(0,str(V7))
    import test_revision
    suite=unittest.TestSuite()
    classes=[]
    for name, cls in vars(test_revision).items():
        if isinstance(cls,type) and issubclass(cls,unittest.TestCase) and name!='ManuscriptTests':
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls));classes.append(name)
    with (out/'scientific_regression.log').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    if not result.wasSuccessful(): raise RuntimeError((out/'scientific_regression.log').read_text())
    report['steps'].append(dict(name='scientific_regression',status='PASS',
                               tests=result.testsRun,classes=classes,
                               excluded='ManuscriptTests: historical TeX/figure-layout tests require the separate manuscript.'))
    print(f'PASS: {result.testsRun} scientific tests',flush=True)
    for script,artifact in [('diagnostic_controls.py','diagnostic_controls.json'),
                            ('campaign_sensitivity.py','campaign_sensitivity.json'),
                            ('mixer_summary.py','mixer_summary.json'),
                            ('ensemble_analysis.py','ensemble_analysis.json')]:
        run(script[:-3],[sys.executable,str(V7/script),'--out-dir',str(out)])
        compare(read(out/artifact),read(V7/'generated'/artifact))
    run('physical_audit',[sys.executable,str(V8/'audit_physical_data.py'),'--out-dir',str(out)])
    for name in ['campaign_block_errors.csv','campaign_qubit_errors.csv']:
        if (out/name).read_text()!=(V8/'generated'/name).read_text():
            raise AssertionError('Physical summary differs: '+name)
    run('readout_audit',[sys.executable,str(V8/'readout_audit.py'),'--out-dir',str(out)])
    compare(read(out/'readout_audit.json'),read(V8/'generated/readout_audit.json'))
    run('cross_block_points',[sys.executable,str(V8/'cross_block_points.py'),'--out-dir',str(out)])
    compare(read(out/'cross_block_points.json'),read(V8/'generated/cross_block_points.json'))
    block_input=V7/'generated/block_covariance.json'
    if args.full:
        run('block_bootstrap',[sys.executable,str(V7/'block_covariance.py'),'--out-dir',str(out)])
        block_input=out/'block_covariance.json'
        compare(canonical_inputs(read(block_input)),canonical_inputs(read(V7/'generated/block_covariance.json')))
        outputs=[]
        for i,stem in enumerate(('e01_20261008','e02_20261014','e03_20261014'),1):
            output=out/f'e0{i}_analysis.json';outputs.append(output)
            base=ROOT/'analysis/ibm_variance_campaign/results'
            run(stem,[sys.executable,str(V7/'analyze_epoch.py'),'--plan',str(base/(stem+'_plan.json')),
                '--counts',str(base/(stem+'_counts.json')),'--out',str(output),'--bootstrap','4000'])
            compare(read(output),read(V7/'results'/output.name))
        run('combine_epochs',[sys.executable,str(V7/'combine_epochs.py'),*map(str,outputs),
                              '--out',str(out/'between_jobs.json')])
        compare(read(out/'between_jobs.json'),read(V7/'results/between_jobs.json'))
    run('campaign_figure',[sys.executable,str(V7/'campaign_artifacts.py'),
                          '--results',str(out if args.full else V7/'results'),'--out-dir',str(out)])
    run('block_figure',[sys.executable,str(V7/'block_covariance_artifacts.py'),
                       '--input',str(block_input),'--out-dir',str(out)])
    # Existing figure generators use a module-level destination; override it so
    # reproduction never overwrites released reference inputs.
    sys.path.insert(0,str(ROOT/'scripts'))
    import matplotlib
    matplotlib.use('Agg')
    for name in ('make_block_figure','make_exact51_layout_figure'):
        spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.FIGURES_DIR=out;module.main()
        report['steps'].append(dict(name=name,status='PASS'))
    report['figures']=['fig_blocks.pdf','fig_three_jobs.pdf','fig_exact51_routing.pdf','fig_block_covariance.pdf']
    report['elapsed_seconds']=round(time.monotonic()-started,2)
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
