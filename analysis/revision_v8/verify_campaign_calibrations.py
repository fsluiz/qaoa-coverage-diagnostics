#!/usr/bin/env python3
"""Independently verify archived calibration summaries with IBM SDK accessors.

This script is offline: it instantiates BackendProperties from the saved JSON,
not a Runtime service. Checks raw checksums, date consistency, coverage and all
18 campaign block summaries. Requires qiskit-ibm-runtime (no account required).
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
PROJECT = PAPER.parent


def dt(value):
    return datetime.fromisoformat(str(value).replace('Z','+00:00'))


def main():
    from qiskit_ibm_runtime.models import BackendProperties

    audit=json.loads((HERE/'generated/physical_data_audit.json').read_text())
    ops=json.loads((PROJECT/'results/ibm_circuits/exact51_matched.ops.json').read_text())
    native_cz=[(i,qs) for i,(name,qs,_) in enumerate(ops) if name=='cz']
    edges={tuple(sorted(qs)) for _,qs in native_cz}
    assert len(native_cz)==372 and len(edges)==40
    checked=[]
    signatures=[]
    for c in audit['contexts'][2:]:
        epoch={'E01':'e01_20261008','E02':'e02_20261014','E03':'e03_20261014'}[c['context']]
        path=HERE/'calibration_archives'/(epoch+'_properties.json')
        raw=json.loads(path.read_text())
        assert path.with_suffix('.sha256').read_text().split()[0]==hashlib.sha256(path.read_bytes()).hexdigest()
        assert raw['job_id']==c['job_id'] and raw['backend']=='ibm_kingston'
        assert dt(raw['calibration_date'])==dt(c['calibration_date'])
        props=BackendProperties.from_dict(raw['backend_properties'])
        assert props.backend_name=='ibm_kingston'
        start=dt(raw['execution_timestamps']['running'])
        assert dt(props.last_update_date)<=start
        gate_values={}
        gate_signatures={}
        for edge in edges:
            try:
                error,timestamp=props.gate_property('cz',edge,'gate_error')
            except Exception:
                error,timestamp=props.gate_property('cz',edge[::-1],'gate_error')
            assert dt(timestamp)<=start
            gate_values[edge]=float(error)
            gate_signatures[edge]=(float(error),str(timestamp))
        signatures.append(gate_signatures)
        measured={qs[0] for name,qs,_ in ops if name=='measure'}
        assert len(measured)==51
        for q in measured:
            for parameter in ('readout_error','prob_meas0_prep1','prob_meas1_prep0','T1','T2'):
                _,timestamp=props.qubit_property(q,parameter)
                assert dt(timestamp)<=start
        for b in c['blocks']:
            # Independent backward traversal followed by per-occurrence SDK values.
            wires=set(b['measured_physical_qubits'])
            gate_errors=[]
            for name,qs,_ in reversed(ops):
                if name in ('barrier','measure') or wires.isdisjoint(qs):
                    continue
                wires.update(qs)
                if name=='cz':
                    gate_errors.append(gate_values[tuple(sorted(qs))])
            readout=[props.readout_error(q) for q in b['measured_physical_qubits']]
            d=b['calibration_descriptors']
            assert len(gate_errors)==b['native_two_qubit_gates_in_backward_cone']
            for actual,expected in (
                (sum(gate_errors)/len(gate_errors),d['gate_count_weighted_mean_error']),
                (max(gate_errors),d['max_two_qubit_error']),
                (sum(readout)/len(readout),d['mean_readout_error']),
                (max(readout),d['max_readout_error'])):
                assert math.isclose(actual,expected,rel_tol=1e-13,abs_tol=1e-15)
        checked.append(dict(context=c['context'],job_id=c['job_id'],
                            measured_qubits=51,used_couplers=40,blocks_verified=6,
                            native_cz_gates=372,parameters_not_later_than_job_start=True))
    assert signatures[1]==signatures[2]
    report=dict(status='passed',method='Offline BackendProperties accessors and independent per-occurrence gate traversal.',
                campaign_jobs=checked,blocks_verified=18,
                E02_E03_used_cz_values_and_timestamps_identical=True,
                calibration_archives_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((HERE/'calibration_archives').glob('*_properties.json'))})
    (HERE/'generated/calibration_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified 18 block summaries, 51 measured qubits and 40 CZ couplers per job; checksums and timestamps match.')
    print('E02/E03 used CZ calibration values and parameter timestamps are identical.')


if __name__=='__main__':
    main()
