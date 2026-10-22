#!/usr/bin/env python3
"""Interleaved, randomised replication of the joint/separate comparison on three placements.

Job dau04mahcrkc73dur170 compared the joint Rung-1 circuit with its block slices in one job,
but ran the joint pub first with 1e5 shots and the 13 slices after it with 1e4 shots each, and
its joint circuit was a new compilation of the principal run's circuit (job dakjch39k43c73aik9rg),
so order, shot budget, compilation and day were not separated.  This job fixes all four.

It resubmits, unchanged, three archived ISA circuits families (results/ibm_circuits/*.qpy):

  J_A, S_A   the batch joint circuit and its 13 slices (dau04mahcrkc73dur170)
  J_B, S_B   the principal-run joint circuit and its 13 slices, cut here by backward light cone
  C          the six multi-qubit block circuits of the first separate job (dan62g5r85ps73fdlp00),
             whose four-qubit block sits on qubits 122-124, 136 rather than 2, 3, 4, 16

Every pub has the same shots, the pub order is a seeded random permutation of R rounds of the
34 circuits, and twirling uses an explicit number of randomizations per pub.  One job, one
calibration.  No circuit is re-transpiled; the sampler adds only XY4 dynamical decoupling and
gate/measurement twirling, as in the earlier jobs.

    python submit_interleaved_batch.py --instance open-instance            # dry run
    python submit_interleaved_batch.py --instance open-instance --submit
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _paths import RESULTS_DIR

CIRC = RESULTS_DIR / "ibm_circuits"
ROUNDS = 4
SHOTS = 10_000
RANDOMIZATIONS = 20                       # 20 x 500 shots per pub
SEED = 20260930


def load(tag):
    from qiskit import qpy
    with open(CIRC / f"{tag}.qpy", "rb") as f:
        return qpy.load(f)[0]


def ops_of(qc):
    return [(i, [qc.find_bit(q).index for q in i.qubits]) for i in qc.data]


def measured(qc):
    return {qc.find_bit(i.clbits[0]).index: qc.find_bit(i.qubits[0]).index
            for i in qc.data if i.operation.name == "measure"}


def light_cone(qc, phys_out):
    """Physical wires in the backward light cone of the measured qubits `phys_out`."""
    S = set(phys_out)
    for inst, qs in reversed(ops_of(qc)):
        if inst.operation.name in ("measure", "barrier"):
            continue
        if set(qs) & S:
            S |= set(qs)
    return S


def slice_by_cone(qc, logical):
    """The instructions of `qc` inside the light cone of the logical qubits `logical`."""
    from qiskit import QuantumCircuit
    m = measured(qc)
    wires = light_cone(qc, {m[q] for q in logical})
    clbit_of = {q: t for t, q in enumerate(logical)}
    out = QuantumCircuit(qc.num_qubits, len(logical))
    for inst, qs in ops_of(qc):
        if inst.operation.name == "barrier":
            continue
        if set(qs) <= wires:
            if inst.operation.name == "measure":
                out.measure(qs[0], clbit_of[qc.find_bit(inst.clbits[0]).index])
            else:
                out.append(inst.operation, qs)
        elif set(qs) & wires:
            raise RuntimeError(f"{inst.operation.name} on {qs} straddles the cone {sorted(wires)}")
    return out, sorted(wires)


def n2q(qc):
    return sum(1 for i in qc.data if i.operation.num_qubits == 2 and i.operation.name != "barrier")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None)
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    batch = json.load(open(RESULTS_DIR / "joint_separate_batch_plan.json"))
    blocks = [p["logical_qubits"] for p in batch["pubs"] if p["tag"] != "joint"]
    old = json.load(open(RESULTS_DIR / "ibm_block_results" / "kingston_blocks_plan.json"))

    circuits = {}
    jA, jB = load("batch_joint"), load("rung1_original")
    circuits["J_A"] = (jA, list(range(24)), dict(source="dau04mahcrkc73dur170:joint"))
    circuits["J_B"] = (jB, list(range(24)), dict(source="dakjch39k43c73aik9rg:rung1_qaoa_p1"))
    for fam, joint in (("A", jA), ("B", jB)):
        cones = []
        for k, b in enumerate(blocks):
            if fam == "A":
                qc = load(f"batch_block_{k:02d}")
                wires = sorted({q for i, qs in ops_of(qc) for q in qs
                                if i.operation.name != "barrier"})
                src = f"dau04mahcrkc73dur170:block_{k:02d}"
            else:
                qc, wires = slice_by_cone(joint, b)
                src = "dakjch39k43c73aik9rg:rung1_qaoa_p1 light-cone slice"
            cones.append(set(wires))
            circuits[f"S_{fam}_{k:02d}"] = (qc, b, dict(source=src, physical=wires))
        assert all(not (cones[i] & cones[j]) for i in range(13) for j in range(i + 1, 13)), \
            f"family {fam}: block light cones overlap"
        assert sum(n2q(circuits[f"S_{fam}_{k:02d}"][0]) for k in range(13)) == \
            n2q(circuits[f"J_{fam}"][0]), f"family {fam}: slices must hold every gate"
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService(instance=a.instance)
    oldjob = service.job(old["job_id"])
    for k, (pub, blk) in enumerate(zip(oldjob.inputs["pubs"], old["blocks"])):
        qc = pub[0] if isinstance(pub, (tuple, list)) else pub.circuit
        if n2q(qc) == 0:
            continue
        kb = blocks.index(sorted(blk["original_qubits"]))
        assert blk["original_qubits"] == sorted(blk["original_qubits"])
        circuits[f"C_{kb:02d}"] = (qc, blocks[kb], dict(
            source=f"{old['job_id']}:{blk['tag']}", physical=blk["measured_physical_qubits"]))

    tags = sorted(circuits)
    rng = np.random.default_rng(SEED)
    order = [(r, t) for r in range(ROUNDS) for t in tags]
    order = [order[i] for i in rng.permutation(len(order))]
    meta = [dict(position=i, round=r, tag=t, logical_qubits=circuits[t][1],
                 n_2q=n2q(circuits[t][0]), depth=circuits[t][0].depth(),
                 measured_physical={str(k): v for k, v in measured(circuits[t][0]).items()},
                 shots=SHOTS, **circuits[t][2]) for i, (r, t) in enumerate(order)]
    for t in tags:
        print(f"{t:8} n2q={n2q(circuits[t][0]):3d}  {circuits[t][2]['source']}")
    print(f"{len(tags)} circuits x {ROUNDS} rounds = {len(order)} pubs, "
          f"{len(order) * SHOTS:,} shots")
    plan = dict(description=__doc__.split("\n\n")[0], backend="ibm_kingston", rounds=ROUNDS,
                shots_per_pub=SHOTS, num_randomizations=RANDOMIZATIONS, order_seed=SEED,
                mitigation="XY4 DD + gate/measure twirling", blocks=blocks, pubs=meta,
                submitted=False, job_id=None)
    if a.submit:
        backend = service.backend("ibm_kingston")
        sampler = SamplerV2(mode=backend)
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        sampler.options.twirling.num_randomizations = RANDOMIZATIONS
        sampler.options.twirling.shots_per_randomization = SHOTS // RANDOMIZATIONS
        job = sampler.run([(circuits[t][0], None, SHOTS) for _, t in order])
        plan.update(submitted=True, job_id=job.job_id())
        print(f"submitted job {job.job_id()}")
    else:
        print("DRY RUN -- add --submit to execute")
    json.dump(plan, open(RESULTS_DIR / "interleaved_batch_plan.json", "w"), indent=1)
    print("-> results/interleaved_batch_plan.json")


if __name__ == "__main__":
    main()
