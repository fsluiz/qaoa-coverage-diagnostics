#!/usr/bin/env python3
"""Audit archived block-to-hardware provenance without IBM access or new shots.

Reconstruct conservative backward gate cones through the submitted native gates.
Barriers impose scheduling constraints, not quantum influence, and are skipped.
Historical error descriptors are resource summaries, not predictions of C_alpha.
"""
from __future__ import annotations

from collections import Counter
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
PROJECT = PAPER.parent
OUT = HERE / 'generated'
CONTEXTS = ('defaults', 'matched', 'E01', 'E02', 'E03')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def light_cones(ops, blocks):
    measured = {cs[0]: qs[0] for name, qs, cs in ops if name == 'measure'}
    if len(measured) != 51 or set(measured) != set(range(51)):
        raise ValueError('Expected a complete 51-bit measurement map.')
    cones = []
    for block in blocks:
        wires, cone = {measured[q] for q in block}, set()
        for index in range(len(ops)-1, -1, -1):
            name, qs, _ = ops[index]
            if name in ('measure', 'barrier') or not wires.intersection(qs):
                continue
            cone.add(index)
            wires.update(qs)
        cones.append(cone)
    return measured, cones


def normalized_job_errors(archive, ops):
    """Select calibration records by native gate name and physical operands."""
    raw = archive['backend_properties']
    gate_records = {(g['gate'], tuple(g['qubits'])):g for g in raw['gates']}
    counts = Counter(tuple(sorted(qs)) for name,qs,_ in ops
                     if len(qs)==2 and name not in ('barrier','measure'))
    native = {(name,tuple(qs)) for name,qs,_ in ops
              if len(qs)==2 and name not in ('barrier','measure')}
    if {name for name,_ in native} != {'cz'}:
        raise ValueError('Edge-only descriptor requires the recorded CZ-native circuit.')
    errors = {}
    for name,qubits in sorted(native):
        record = gate_records.get((name,qubits))
        if record is None:
            # CZ is symmetric; retain the original calibration operands in raw archive.
            record = gate_records.get((name,qubits[::-1]))
        if record is None:
            raise ValueError('No historical calibration for native gate '+str((name,qubits)))
        param = next(p for p in record['parameters'] if p['name']=='gate_error')
        value = float(param['value'])
        if not 0 <= value <= 1:
            raise ValueError('Gate error is not a probability.')
        key = tuple(sorted(qubits))
        if key in errors and errors[key] != value:
            raise ValueError('Directional CZ calibration differs; do not collapse orientations.')
        errors[key] = value
    readout = {}
    directional = {}
    for name,qubits,_ in ops:
        if name!='measure':
            continue
        q=qubits[0]
        params={p['name']:p for p in raw['qubits'][q]}
        readout[str(q)] = float(params['readout_error']['value'])
        if not 0 <= readout[str(q)] <= 1:
            raise ValueError('Readout error is not a probability.')
        directional[str(q)] = {name:params[name] for name in
                              ('prob_meas0_prep1','prob_meas1_prep0') if name in params}
    return dict(job_id=archive['job_id'], readout_errors=readout,
                directional_readout=directional,
                two_qubit_edges=[dict(edge=list(e),count=n,error=errors[e])
                                 for e,n in sorted(counts.items())])


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out_dir
    provenance = {}

    def load(path):
        provenance[str(path.relative_to(PROJECT))] = digest(path)
        return read(path)

    profiles = load(PAPER/'revision_v7/generated/block_covariance.json')
    blocks = profiles['reference']['blocks']
    estimates = {c['context']: c['statistics']['retention']['estimate']
                 for c in profiles['contexts']}
    layout = load(PROJECT/'results/exact51_physical_layout.json')
    registry = load(PROJECT/'results/ibm_circuits/index.json')
    resource = load(PROJECT/'results/resource_model_exact.json')
    sources = {c['job_id']: c for c in resource['circuits']}
    job_properties = {j['job_id']: j for j in resource['jobs']}
    contexts = []
    recovered_archives = {}
    for context in CONTEXTS:
        campaign = context.startswith('E')
        source = 'matched' if campaign else context
        stem = 'exact51_'+source
        entry = registry[stem]
        qpy = PROJECT/'results/ibm_circuits'/entry['file']
        qpy_hash = digest(qpy)
        provenance[str(qpy.relative_to(PROJECT))] = qpy_hash
        if qpy_hash != entry['sha256']:
            raise ValueError('QPY registry checksum mismatch: '+stem)
        ops_path = qpy.with_suffix('.ops.json')
        ops = load(ops_path)
        if digest(ops_path) != entry['ops_sha256']:
            raise ValueError('Instruction-list registry checksum mismatch: '+stem)
        measured, cones = light_cones(ops, blocks)
        if measured != {int(k):v for k,v in layout[source]['logical_to_physical'].items()}:
            raise ValueError('Measurement map differs from archived layout.')
        job = entry['job_id']
        calibration = job_properties[job]['calibration_date']
        shots = 100000
        props = sources[job]
        if campaign:
            epoch = {'E01':'e01_20261008','E02':'e02_20261014','E03':'e03_20261014'}[context]
            base = PAPER/'ibm_variance_campaign/results'
            plan = load(base/(epoch+'_plan.json'))
            archive = load(base/(epoch+'_counts.json'))
            if archive['job_id'] != plan['job_id']:
                raise ValueError('Plan/count job mismatch.')
            pubs = [p for p in plan['pubs'] if p['tag']=='faithful51']
            if len(pubs)!=4 or {p['round'] for p in pubs}!=set(range(4)):
                raise ValueError('Expected four faithful-circuit rounds.')
            shots = 0
            for pub in pubs:
                recorded = archive['pubs'][str(pub['position'])]
                if pub['qpy_sha256'] != qpy_hash or PROJECT/pub['qpy'] != qpy:
                    raise ValueError('Campaign does not use the same archived QPY.')
                if pub['n_2q']!=entry['two_qubit_gates'] or pub['depth']!=entry['depth']:
                    raise ValueError('Plan resource descriptors differ from registry.')
                if recorded['tag']!='faithful51' or recorded['round']!=pub['round']:
                    raise ValueError('Pub tag or round mismatch.')
                if sum(recorded['counts'].values())!=pub['shots']:
                    raise ValueError('Count total differs from plan.')
                if any(len(key)!=51 or set(key)-{'0','1'} for key in recorded['counts']):
                    raise ValueError('Expected 51-bit faithful counts.')
                shots += pub['shots']
            job, calibration = plan['job_id'], archive['calibration_date']
            # September properties cannot substitute for the campaign job properties.
            props = sources.get(job)
            recovered = HERE/'calibration_archives'/(epoch+'_properties.json')
            if recovered.exists():
                payload = load(recovered)
                iso = lambda x: datetime.fromisoformat(str(x).replace('Z','+00:00'))
                if payload['job_id']!=job or payload['backend']!=plan['backend']:
                    raise ValueError('Recovered properties belong to another job/backend.')
                if iso(payload['calibration_date'])!=iso(calibration):
                    raise ValueError('Recovered calibration date does not match the counts archive.')
                props = normalized_job_errors(payload,ops)
                recovered_archives[job] = str(recovered.relative_to(PROJECT))
        cone_multiplicity = Counter(i for cone in cones for i in cone)
        edge_count = Counter(tuple(sorted(qs)) for name,qs,_ in ops
                             if len(qs)==2 and name not in ('barrier','measure'))
        if sum(edge_count.values()) != entry['two_qubit_gates']:
            raise ValueError('Native two-qubit count differs from registry.')
        rows = []
        errors = ({tuple(sorted(e['edge'])):e['error'] for e in props['two_qubit_edges']}
                  if props else {})
        if props:
            expected_edges={tuple(sorted(e['edge'])):e['count'] for e in props['two_qubit_edges']}
            if dict(edge_count)!=expected_edges:
                raise ValueError('Historical calibration edge multiplicities differ from circuit.')
        for index,(block,cone) in enumerate(zip(blocks,cones)):
            physical = [measured[q] for q in block]
            gates = [i for i in cone if len(ops[i][1])==2]
            row = dict(logical_qubits=block, size=len(block), measured_physical_qubits=physical,
                       C_alpha=estimates[context][index],
                       native_two_qubit_gates_in_backward_cone=len(gates),
                       shared_two_qubit_gates=sum(cone_multiplicity[i]>1 for i in gates),
                       native_gate_couplers=sorted({tuple(sorted(ops[i][1])) for i in gates}))
            row['calibration_descriptors'] = None
            if props:
                readout=[props['readout_errors'][str(q)] for q in physical]
                gate_errors=[errors[tuple(sorted(ops[i][1]))] for i in gates]
                row['calibration_descriptors']=dict(
                    readout_errors=dict(zip(map(str,physical),readout)),
                    mean_readout_error=sum(readout)/len(readout), max_readout_error=max(readout),
                    gate_count_weighted_mean_error=sum(gate_errors)/len(gate_errors),
                    max_two_qubit_error=max(gate_errors))
                if 'directional_readout' in props:
                    row['calibration_descriptors']['directional_readout']={
                        str(q):props['directional_readout'][str(q)] for q in physical}
            rows.append(row)
        contexts.append(dict(context=context, job_id=job, calibration_date=calibration,
                             qpy_sha256=qpy_hash, source_circuit=entry['file'],
                             source_circuit_job_id=entry['job_id'], faithful_shots=shots,
                             calibration_error_values_available=props is not None, blocks=rows))
    # Check current local JSON archives for additional campaign calibration payloads.
    # Derived manuscripts and exported copies are not independent metadata sources.
    searched = [PROJECT/'results', PAPER/'ibm_variance_campaign/results', PAPER/'revision_v7/results']
    error_keys = {'readout_error', 'readout_errors', 'gate_error', 'gate_errors',
                  'backend_properties', 'two_qubit_edges'}
    search_results = {c['job_id']:dict(files=[], possible_calibration_payloads=[])
                      for c in contexts[2:]}

    def payload_paths(node, pointer=''):
        found = []
        if isinstance(node, dict):
            for key, value in node.items():
                child = pointer+'/'+key
                if key in error_keys and isinstance(value,(dict,list,int,float)):
                    found.append(child)
                found.extend(payload_paths(value, child))
        elif isinstance(node, list):
            for index,value in enumerate(node):
                found.extend(payload_paths(value,pointer+'/'+str(index)))
        return found

    scanned = 0
    for directory in searched:
        for path in sorted(directory.rglob('*.json')):
            scanned += 1
            raw = path.read_text()
            jobs = [job for job in search_results if job in raw]
            if not jobs:
                continue
            payloads = payload_paths(json.loads(raw))
            relative = str(path.relative_to(PROJECT))
            provenance[relative] = digest(path)
            for job in jobs:
                search_results[job]['files'].append(relative)
                search_results[job]['possible_calibration_payloads'].extend(
                    relative+'#'+pointer for pointer in payloads)
    if any(v['possible_calibration_payloads'] for v in search_results.values()):
        raise ValueError('Additional campaign calibration payload found; inspect before reporting missing values.')
    missing_calibrations=[c['context'] for c in contexts[2:]
                          if not c['calibration_error_values_available']]
    report=dict(scope='Offline provenance and descriptive resource audit; no causal attribution.',
                method='Backward native-gate cones; barriers skipped; shared gates may count in several cones.',
                sources_sha256=provenance, contexts=contexts,
                calibration_archive_search=dict(
                    directories=[str(d.relative_to(PROJECT)) for d in searched],
                    json_files_scanned=scanned, campaign_jobs=search_results,
                    separately_recovered_archives=recovered_archives),
                missing_inputs=([f'{c} job-associated per-qubit/coupler calibration values.'
                                 for c in missing_calibrations]+
                                ['Execution-time schedules/idle durations and noise characterization for mechanism attribution.']),
                interpretation='Reused layout is verified by QPY hashes; recovered campaign values are matched by job ID, backend and calibration date. Calibration descriptors do not determine covariance attenuation.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'physical_data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    table = [r'\begin{table*}[t]',r'\caption{Physical placement and output ratios for all six interacting blocks in E01--E03. Logical block labels refer to the sorted block sizes. Physical qubits are listed in logical-index order. Gate counts include native two-qubit instructions in a conservative backward cone, including shared instructions; counts do not partition the circuit. All three jobs use the same QPY checksum.}',
             r'\label{tab:sm_physical_campaign}',r'\small',r'\begin{ruledtabular}',
             r'\begin{tabular}{rp{0.40\textwidth}rrrr}',
             r'Size & Measured physical qubits & Cone gates & $C_{\rm E01}$ & $C_{\rm E02}$ & $C_{\rm E03}$ \\',r'\hline']
    for i,b in enumerate(blocks):
        row=contexts[2]['blocks'][i]
        table.append(f"{len(b)} & "+', '.join(map(str,row['measured_physical_qubits']))+' & '+
                     str(row['native_two_qubit_gates_in_backward_cone'])+' & '+
                     ' & '.join(f"{c['blocks'][i]['C_alpha']:.3f}" for c in contexts[2:])+r' \\')
    table += [r'\end{tabular}',r'\end{ruledtabular}',r'\end{table*}']
    if not missing_calibrations:
        table += [r'\begin{table*}[t]',r'\caption{Recovered job-associated calibration descriptors for all six interacting blocks in E01--E03. Percentages summarize the recorded native CZ errors (gate-count-weighted over the backward cone) and readout errors (mean and maximum over measured block qubits). Parameter timestamps and units are retained in the raw property archives. These descriptors are not fitted error rates for the covariance ratio.}',
                  r'\label{tab:sm_physical_campaign_errors}',r'\small',r'\begin{ruledtabular}',
                  r'\begin{tabular}{lrrrrr}',
                  r'Job & Size & $C_\alpha$ & Mean gate error (\%) & Mean readout error (\%) & Max readout error (\%) \\',r'\hline']
        for c in contexts[2:]:
            for row in c['blocks']:
                d=row['calibration_descriptors']
                table.append(f"{c['context']} & {row['size']} & {row['C_alpha']:.3f} & "+
                             f"{100*d['gate_count_weighted_mean_error']:.3f} & "+
                             f"{100*d['mean_readout_error']:.3f} & {100*d['max_readout_error']:.3f}"+r' \\')
        table += [r'\end{tabular}',r'\end{ruledtabular}',r'\end{table*}']
    table += [r'\begin{table*}[t]',r'\caption{Historical job-associated calibration descriptors for the two faithful executions. Gate error is averaged over native two-qubit gates in each backward cone, weighted by their multiplicities; readout error is the maximum among the block\textquoteright s measured qubits. Percentages are calibration summaries, not fitted covariance attenuation or process fidelity. Placements, compilation, date and mitigation differ between runs.}',
              r'\label{tab:sm_physical_historical}',r'\small',r'\begin{ruledtabular}',
              r'\begin{tabular}{lrrrrr}',
              r'Context & Size & $C_\alpha$ & Cone gates & Mean gate error (\%) & Max readout error (\%) \\',r'\hline']
    for context in contexts[:2]:
        for row in context['blocks']:
            desc=row['calibration_descriptors']
            table.append(f"{context['context']} & {row['size']} & {row['C_alpha']:.3f} & "+
                         f"{row['native_two_qubit_gates_in_backward_cone']} & "+
                         f"{100*desc['gate_count_weighted_mean_error']:.3f} & {100*desc['max_readout_error']:.3f}"+r' \\')
    table += [r'\end{tabular}',r'\end{ruledtabular}',r'\end{table*}']
    (OUT/'physical_block_table.tex').write_text('\n'.join(table)+'\n')
    if not missing_calibrations:
        fields = ['epoch','block_size','C_alpha','mean_cz_error_percent',
                  'mean_readout_error_percent','max_readout_error_percent',
                  'cone_cz_gates','measured_physical_qubits']
        with (OUT/'campaign_block_errors.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            for c in contexts[2:]:
                for row in c['blocks']:
                    d=row['calibration_descriptors']
                    writer.writerow(dict(epoch=c['context'],block_size=row['size'],
                        C_alpha=row['C_alpha'],mean_cz_error_percent=100*d['gate_count_weighted_mean_error'],
                        mean_readout_error_percent=100*d['mean_readout_error'],
                        max_readout_error_percent=100*d['max_readout_error'],
                        cone_cz_gates=row['native_two_qubit_gates_in_backward_cone'],
                        measured_physical_qubits=';'.join(map(str,row['measured_physical_qubits']))))
        # Export all 51 measured qubits, including the seven singleton blocks.
        qubit_fields=['epoch','logical_qubit','physical_qubit','block_size',
                      'readout_error_percent','prob_meas0_prep1_percent',
                      'prob_meas1_prep0_percent','readout_calibration_date']
        qubit_measurements={cs[0]:qs[0] for name,qs,cs in ops if name=='measure'}
        sizes={q:len(b) for b in blocks+profiles['reference']['singleton_blocks'] for q in b}
        with (OUT/'campaign_qubit_errors.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=qubit_fields)
            writer.writeheader()
            for c in contexts[2:]:
                raw=read(PROJECT/recovered_archives[c['job_id']])['backend_properties']
                for logical,q in sorted(qubit_measurements.items()):
                    params={v['name']:v for v in raw['qubits'][q]}
                    writer.writerow(dict(epoch=c['context'],logical_qubit=logical,
                        physical_qubit=q,block_size=sizes[logical],
                        readout_error_percent=100*params['readout_error']['value'],
                        prob_meas0_prep1_percent=100*params['prob_meas0_prep1']['value'],
                        prob_meas1_prep0_percent=100*params['prob_meas1_prep0']['value'],
                        readout_calibration_date=params['readout_error']['date']))
    for c in contexts:
        five=next(b for b in c['blocks'] if b['size']==5)
        print(c['context'], 'physical=',five['measured_physical_qubits'],
              'C=',round(five['C_alpha'],6), 'calibration=',c['calibration_error_values_available'])
    print('Verified QPY and instruction hashes, measurement maps, edge counts and four faithful pubs per campaign job.')


if __name__ == '__main__':
    main()
