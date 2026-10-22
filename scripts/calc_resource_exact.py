#!/usr/bin/env python3
"""Reconstruct the paper-v3 error proxy from the IBM job records.

For every hardware circuit used in paper v3, this script retrieves the submitted,
already-transpiled circuit and the calibration attached to its job.  It archives enough
information to recompute

    F_gate = product_g (1 - error_g)
    F_est  = F_gate * product_q (1 - readout_error_q)

without later IBM access: measured physical qubits, two-qubit edge multiplicities and the
corresponding job-time error values are all written to results/resource_model_exact.json.
The script is read-only with respect to IBM Quantum and never submits work.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

from _paths import RESULTS_DIR


JOBS = (
    dict(job_id="dakjch39k43c73aik9rg", plan="kingston_rerun_plan_rung1.json"),
    dict(job_id="dakjbtfi3e6s738rm2h0", plan="kingston_rerun_plan_arco.json"),
    dict(job_id="dakjjpfi3e6s738rmbng", plan="exact51_plan.json",
         tags=("faithful_exact51_p1",)),
    # Version 4: the same faithful circuit with the QAOA jobs' XY4 + twirling settings.
    dict(job_id="dat665lvr3kc73el167g", plan="exact51_matched_plan.json",
         tags=("faithful_exact51_p1_matched",)),
    # Paper v3 compares this transferred-angle Arco run with its optimised successor.
    dict(job_id="d9211s357qjs73b76k00",
         pub_indices=((3, "arco_qaoa_p1_transferred"),)),
)


def plan_tags(spec: dict) -> list[str]:
    plan = json.load(open(RESULTS_DIR / spec["plan"]))
    return list(spec.get("tags") or [row["tag"] for row in plan.get("jobs", [])])


def main() -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(instance=os.environ.get("IBM_INSTANCE"))
    output = {"model": "independent job-time gate and readout errors", "jobs": [],
              "circuits": []}

    print(f"{'circuit':31} {'job':10} {'nq':>3} {'n2Q':>5} "
          f"{'F_gate':>8} {'F_read':>8} {'F_est':>8}")
    for spec in JOBS:
        job = service.job(spec["job_id"])
        props = job.properties()
        pubs = job.inputs["pubs"]
        if "pub_indices" in spec:
            selected_pubs = [(tag, pubs[index]) for index, tag in spec["pub_indices"]]
        else:
            tags = plan_tags(spec)
            if len(tags) != len(pubs):
                raise RuntimeError(
                    f"{spec['job_id']}: {len(tags)} tags for {len(pubs)} circuits")
            selected = set(spec.get("select", tags))
            aliases = dict(zip(spec.get("select", ()), spec.get("aliases", ())))
            selected_pubs = [(aliases.get(tag, tag), pub) for tag, pub in zip(tags, pubs)
                             if tag in selected]

        edge_error = {}
        for gate in props.gates:
            if len(gate.qubits) != 2:
                continue
            values = [p.value for p in gate.parameters if p.name == "gate_error"]
            if values:
                edge_error[tuple(sorted(gate.qubits))] = float(values[0])
        readout_error = {q: float(props.readout_error(q))
                         for q in range(len(props.qubits))}

        backend_name = job.backend().name
        if callable(backend_name):
            backend_name = backend_name()
        output["jobs"].append(dict(
            job_id=spec["job_id"], backend=backend_name,
            creation_date=str(job.creation_date),
            calibration_date=str(props.last_update_date),
            quantum_seconds=job.usage_estimation.get("quantum_seconds"),
        ))

        for out_tag, pub in selected_pubs:
            circuit = pub[0]
            measured = sorted({circuit.find_bit(inst.qubits[0]).index
                               for inst in circuit.data if inst.operation.name == "measure"})
            pairs = [tuple(sorted(circuit.find_bit(q).index for q in inst.qubits))
                     for inst in circuit.data if inst.operation.num_qubits == 2]
            multiplicity = Counter(pairs)
            missing = sorted(edge for edge in multiplicity if edge not in edge_error)
            if missing:
                raise RuntimeError(f"{out_tag}: missing calibration for edges {missing}")

            f_gate = float(np.prod([(1.0 - edge_error[e]) ** count
                                    for e, count in multiplicity.items()]))
            f_read = float(np.prod([1.0 - readout_error[q] for q in measured]))
            edge_rows = [dict(edge=list(edge), count=count, error=edge_error[edge])
                         for edge, count in sorted(multiplicity.items())]
            ro_rows = {str(q): readout_error[q] for q in measured}
            row = dict(
                tag=out_tag, job_id=spec["job_id"], n_qubits=len(measured),
                depth=circuit.depth(), n_2Q=len(pairs), operations=dict(circuit.count_ops()),
                measured_qubits=measured, two_qubit_edges=edge_rows,
                readout_errors=ro_rows, F_gates=f_gate, F_readout=f_read,
                F_exact=f_gate * f_read,
            )
            output["circuits"].append(row)
            print(f"{out_tag:31} {spec['job_id'][:10]} {len(measured):3d} {len(pairs):5d} "
                  f"{f_gate:8.4f} {f_read:8.4f} {f_gate*f_read:8.4f}")

    path = RESULTS_DIR / "resource_model_exact.json"
    json.dump(output, open(path, "w"), indent=1, sort_keys=True)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
