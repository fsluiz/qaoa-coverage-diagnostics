#!/usr/bin/env python3
"""Independent asymmetric readout at archived rates on the exact block reference.

Offline model comparison; no fitted parameters or hardware submission. The
deficit fraction compares point predictions with 1-C_observed, not a causal
attribution or a confidence bound. Raw calibration uncertainties are unavailable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'revision_v7'))
from diagnostic_controls import faithful_reference, readout_moments

STEMS = ('e01_20261008', 'e02_20261014', 'e03_20261014')


def analyze():
    sources = {}

    def read(path):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    read(ROOT / 'data/frozen_rung1.json')
    plan = read(ROOT / 'results/exact51_plan.json')
    profiles = read(HERE.parent / 'revision_v7/generated/block_covariance.json')
    layout = read(ROOT / 'results/exact51_physical_layout.json')['matched']['logical_to_physical']
    mean, second, linear, quadratic, _, _ = faithful_reference()
    observed = {c['context']: c['statistics']['retention']['estimate'] for c in profiles['contexts']}
    rows, rates = [], []
    for index, stem in enumerate(STEMS, 1):
        epoch = f'E0{index}'
        path = HERE / 'calibration_archives' / (stem + '_properties.json')
        archive = read(path)
        checksum = path.with_suffix('.sha256').read_text().split()[0]
        if sources[str(path.relative_to(ROOT))] != checksum:
            raise ValueError('Calibration checksum mismatch: ' + stem)
        campaign_plan = read(HERE.parent / 'ibm_variance_campaign/results' / (stem + '_plan.json'))
        if archive['job_id'] != campaign_plan['job_id']:
            raise ValueError('Calibration/plan job mismatch.')
        # Columns are prepared bits; rows are reported bits. p01=P(1|0).
        p01, p10 = np.zeros(len(mean)), np.zeros(len(mean))
        for logical in range(len(mean)):
            physical = layout[str(logical)]
            params = {v['name']: v for v in archive['backend_properties']['qubits'][physical]}
            p01[logical] = params['prob_meas1_prep0']['value']
            p10[logical] = params['prob_meas0_prep1']['value']
            rates.append(dict(epoch=epoch, logical_qubit=logical, physical_qubit=physical,
                              p_report1_prepared0=float(p01[logical]),
                              p_report0_prepared1=float(p10[logical]),
                              timestamp_report1_prepared0=params['prob_meas1_prep0']['date'],
                              timestamp_report0_prepared1=params['prob_meas0_prep1']['date']))
        transformed_mean, transformed_second = readout_moments(mean, second, p01, p10)
        for bindex, block in enumerate(profiles['reference']['blocks']):
            ix = np.ix_(block, block)
            q = quadratic[ix]
            terms = q * (np.outer(mean[block], mean[block]) - second[ix])
            denominator = float(terms.sum())
            if not np.isclose(denominator, profiles['reference']['delta'][bindex], atol=1e-12):
                raise ValueError('Ideal denominator differs from archived reference.')
            attenuation = float(np.sum(q * (np.outer(transformed_mean[block], transformed_mean[block])
                                            - transformed_second[ix])) / denominator)
            point = observed[epoch][bindex]
            row = dict(epoch=epoch, job_id=archive['job_id'], block_size=len(block),
                       logical_qubits=block, physical_qubits=[layout[str(x)] for x in block],
                       C_observed=point, C_readout=attenuation,
                       modeled_deficit_fraction=(1-attenuation)/(1-point),
                       ideal_denominator=denominator,
                       ideal_positive_terms=float(terms[terms>0].sum()),
                       ideal_negative_terms=float(terms[terms<0].sum()))
            if len(block) == 5:
                # Independent check: explicitly propagate all 32 probabilities,
                # rather than using the moment-transformation identity.
                integers = np.arange(2**len(block))
                bits = ((integers[:, None] >> np.arange(len(block))) & 1).astype(float)
                energy = bits @ linear[block] + np.einsum('bi,ij,bj->b', bits, q, bits)
                state = np.exp(-1j*plan['gamma']*energy) / np.sqrt(len(bits))
                for bit in range(len(block)):
                    state = (np.cos(plan['beta'])*state
                             - 1j*np.sin(plan['beta'])*state[integers ^ (1 << bit)])
                probability = abs(state)**2
                transition = np.ones((len(bits), len(bits)))
                for local, logical in enumerate(block):
                    channel = np.array([[1-p01[logical], p10[logical]],
                                        [p01[logical], 1-p10[logical]]])
                    transition *= channel[bits[:, local].astype(int)[:, None],
                                          bits[:, local].astype(int)[None, :]]
                np.testing.assert_allclose(transition.sum(axis=0), 1, atol=1e-13)
                noisy = transition @ probability
                direct_mean = noisy @ bits
                direct_second = bits.T @ (noisy[:, None]*bits)
                np.testing.assert_allclose(direct_mean, transformed_mean[block], atol=1e-12)
                np.testing.assert_allclose(direct_second, transformed_second[ix], atol=1e-12)
                direct = float(np.sum(q*(np.outer(direct_mean, direct_mean)-direct_second))/denominator)
                np.testing.assert_allclose(direct, attenuation, atol=1e-12)
                row['full_32_state_channel_verified'] = True
            rows.append(row)
    return dict(scope='Independent readout-only point model at archived rates; no mechanism attribution.',
                formula='Cov(Y_i,Y_j)=(1-p01_i-p10_i)(1-p01_j-p10_j)Cov(X_i,X_j), i!=j',
                fraction_definition='(1-C_readout)/(1-C_observed); reference C=1',
                limitations=['No calibration-rate uncertainty or drift modeled.',
                             'No correlated readout, gate errors or idling modeled.',
                             'Fractions are conditional point comparisons, not causal fractions or upper bounds.',
                             'Independent measurement twirling symmetrizes p01,p10 but preserves their sum and these covariance predictions.'],
                sources_sha256=sources, blocks=rows, qubit_rates=rates)


def write(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/'readout_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    rows = [r for r in result['blocks'] if r['block_size']==5]
    fields = ['epoch','job_id','C_observed','C_readout','modeled_deficit_fraction','ideal_denominator']
    with (output/'readout_five_block.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    with (output/'directional_readout_rates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['qubit_rates'][0]))
        writer.writeheader(); writer.writerows(result['qubit_rates'])
    table = [r'\begin{table}[t]',
             r'\caption{Independent readout applied to the exact five-variable block distribution. The last column is $(1-C_\alpha^{\rm ro})/(1-C_\alpha^{\rm obs})$, using $C=1$ as ideal reference. Rates are fixed at their archived calibration values; these are model-conditional point comparisons, not causal fractions or confidence bounds.}',
             r'\label{tab:sm_readout_control}',r'\begin{ruledtabular}',r'\begin{tabular}{lrrr}',
             r'Job & $C_\alpha^{\rm obs}$ & $C_\alpha^{\rm ro}$ & Deficit fraction (\%) \\',r'\hline']
    for row in rows:
        table.append(f"{row['epoch']} & {row['C_observed']:.4f} & {row['C_readout']:.4f} & {100*row['modeled_deficit_fraction']:.2f}"+r' \\')
    table += [r'\end{tabular}',r'\end{ruledtabular}',r'\end{table}']
    (output/'readout_table.tex').write_text('\n'.join(table)+'\n')
    macros = [rf'\newcommand{{\ReadoutC{r["epoch"]}}}{{{r["C_readout"]:.4f}}}' for r in rows]
    # TeX control-sequence names cannot contain digits.
    macros = [line.replace('E01','One').replace('E02','Two').replace('E03','Three') for line in macros]
    fractions = [100*r['modeled_deficit_fraction'] for r in rows]
    macros += [rf'\newcommand{{\ReadoutFractionMin}}{{{min(fractions):.1f}}}',
               rf'\newcommand{{\ReadoutFractionMax}}{{{max(fractions):.1f}}}']
    (output/'readout_numbers.tex').write_text('\n'.join(macros)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=HERE/'generated')
    args = parser.parse_args()
    result = analyze(); write(result, args.out_dir)
    for row in result['blocks']:
        if row['block_size']==5:
            print(row['epoch'], row['C_observed'], row['C_readout'], row['modeled_deficit_fraction'])


if __name__ == '__main__':
    main()
