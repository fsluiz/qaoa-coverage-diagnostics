#!/usr/bin/env python3
"""Joint and separate execution of the Rung-1 circuit in one batch, on the same qubits.

The earlier separate execution of the 13 block circuits (job dan62g5r85ps73fdlp00) ran on a
different day and on different physical qubits from the joint 24-qubit run, so it bounded the
cost of executing all blocks at once without measuring it.  This job removes both confounders:

  pub 0      the joint p=1 circuit on 24 qubits (1e5 shots), transpiled once;
  pubs 1-13  each block circuit alone (1e4 shots each), cut out of the transpiled joint circuit:
             exactly the physical instructions that act on the block's qubits in pub 0, so
             the separate and joint executions apply the same gates on the same qubits.

All pubs run in one SamplerV2 job, hence one calibration and one session, with the XY4
dynamical decoupling and gate/measurement twirling of the other QAOA jobs.  Before submission
the script checks that every block circuit stays on its pinned qubits.

    python submit_joint_separate_batch.py --instance open-instance            # dry run
    python submit_joint_separate_batch.py --instance open-instance --submit
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _paths import DATA_DIR, RESULTS_DIR
from calc_block_factorisation import blocks_of

SHOTS_JOINT = 100_000
SHOTS_BLOCK = 10_000
SEED = 20260929


def qaoa_p1(qubits, h, J, gamma, beta):
    from qiskit import QuantumCircuit
    m = len(qubits)
    idx = {q: t for t, q in enumerate(qubits)}
    qc = QuantumCircuit(m)
    qc.h(range(m))
    for q in qubits:
        if h[q] != 0.0:
            qc.rz(2 * gamma * h[q], idx[q])
    for (i, j), v in sorted(J.items()):
        if i in idx and j in idx:
            qc.rzz(2 * gamma * v, idx[i], idx[j])
    for t in range(m):
        qc.rx(2 * beta, t)
    qc.measure_all()
    return qc


def used_physical(tqc):
    used = set()
    for inst in tqc.data:
        if inst.operation.name in ("barrier", "delay"):
            continue
        for q in inst.qubits:
            used.add(tqc.find_bit(q).index)
    return used


def two_qubit_edges(tqc):
    edges = {}
    for inst in tqc.data:
        if inst.operation.num_qubits == 2 and inst.operation.name != "barrier":
            e = tuple(sorted(tqc.find_bit(x).index for x in inst.qubits))
            edges[e] = edges.get(e, 0) + 1
    return [{"edge": list(e), "count": c} for e, c in sorted(edges.items())]


def measured_map(tqc):
    return {tqc.find_bit(i.clbits[0]).index: tqc.find_bit(i.qubits[0]).index
            for i in tqc.data if i.operation.name == "measure"}


def slice_block(joint, phys, clbit_of):
    """The instructions of the transpiled joint circuit that act only on the qubits `phys`."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(joint.num_qubits, len(clbit_of))
    for inst in joint.data:
        qs = [joint.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "barrier":
            continue
        if set(qs) <= phys:
            if inst.operation.name == "measure":
                c = clbit_of[joint.find_bit(inst.clbits[0]).index]
                qc.measure(qs[0], c)
            else:
                qc.append(inst.operation, qs)
        elif set(qs) & phys:
            raise RuntimeError(f"instruction {inst.operation.name} on {qs} straddles {phys}")
    return qc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None)
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    d = json.load(open(DATA_DIR / "frozen_rung1.json"))
    h = np.asarray(d["h"], float)
    J = {tuple(int(x) for x in k.split(",")): float(v) for k, v in d["J"].items()}
    gamma, beta = d["qaoa_angles"]["1"]["gamma"][0], d["qaoa_angles"]["1"]["beta"][0]
    n = d["n_b"]
    blocks = [sorted(int(q) for q in b) for b in blocks_of(n, J)]

    service = QiskitRuntimeService(instance=a.instance)
    backend = service.backend(a.backend)
    joint = transpile(qaoa_p1(list(range(n)), h, J, gamma, beta), backend=backend,
                      optimization_level=3, seed_transpiler=SEED)
    init = joint.layout.initial_index_layout(filter_ancillas=True)   # logical q -> physical
    final = measured_map(joint)
    circuits, meta = [joint], [dict(tag="joint", logical_qubits=list(range(n)),
                                    initial_physical=init, measured_physical=final,
                                    n_2q=sum(e["count"] for e in two_qubit_edges(joint)),
                                    depth=joint.depth(), two_qubit_edges=two_qubit_edges(joint),
                                    shots=SHOTS_JOINT)]
    joint_block_phys = {}
    for k, b in enumerate(blocks):
        phys_b = {final[q] for q in b} | {init[q] for q in b}
        joint_block_phys[k] = phys_b
        tq = slice_block(joint, phys_b, {q: t for t, q in enumerate(b)})
        used = used_physical(tq)
        if not used <= phys_b:
            raise RuntimeError(f"block {b} left its qubits: used {sorted(used)}")
        circuits.append(tq)
        meta.append(dict(tag=f"block_{k:02d}", logical_qubits=b,
                         initial_physical=[init[q] for q in b],
                         measured_physical={t: final[q] for t, q in enumerate(b)},
                         n_2q=sum(e["count"] for e in two_qubit_edges(tq)), depth=tq.depth(),
                         two_qubit_edges=two_qubit_edges(tq), shots=SHOTS_BLOCK))
        print(f"block {b}: physical {sorted(used)}, 2Q {meta[-1]['n_2q']}, depth {tq.depth()}")
    assert sum(m["n_2q"] for m in meta[1:]) == meta[0]["n_2q"], "block slices must hold every gate"
    # the joint circuit must itself keep every gate inside a block
    lab = {q: k for k, b in enumerate(blocks) for q in b}
    p2l_final = {v: k for k, v in final.items()}
    cross = [e for e in meta[0]["two_qubit_edges"]
             if lab[p2l_final.get(e["edge"][0], -1)] != lab[p2l_final.get(e["edge"][1], -1)]
             ] if all(q in p2l_final for e in meta[0]["two_qubit_edges"] for q in e["edge"]) else None
    print(f"joint: {meta[0]['n_2q']} two-qubit gates, depth {meta[0]['depth']}, "
          f"cross-block gates: {cross if cross is None else len(cross)}")
    plan = dict(description=__doc__.split("\n\n")[0], backend=a.backend, seed_transpiler=SEED,
                angles=dict(gamma=gamma, beta=beta), mitigation="XY4 DD + gate/measure twirling",
                pubs=meta, submitted=False, job_id=None)
    if a.submit:
        sampler = SamplerV2(mode=backend)
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        job = sampler.run([(c, None, m["shots"]) for c, m in zip(circuits, meta)])
        plan.update(submitted=True, job_id=job.job_id())
        print(f"submitted job {job.job_id()} ({len(circuits)} pubs)")
    else:
        print("DRY RUN -- add --submit to execute")
    json.dump(plan, open(RESULTS_DIR / "joint_separate_batch_plan.json", "w"), indent=1)
    print("-> results/joint_separate_batch_plan.json")


if __name__ == "__main__":
    main()
