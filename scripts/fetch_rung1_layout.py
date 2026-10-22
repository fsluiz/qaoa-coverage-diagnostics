#!/usr/bin/env python3
"""Archive where the Rung-1 circuit ran on the chip: logical -> physical qubits.

Reads the submitted, backend-transpiled p=1 circuit of job dakjch39k43c73aik9rg and records,
for every logical qubit (the clbit it is measured into), the physical Kingston qubit that
carried it, together with the device coupling map.  Needs IBM access once; the figure
script then works offline from results/rung1_physical_layout.json.

    python fetch_rung1_layout.py --instance open-instance
"""
from __future__ import annotations

import argparse
import json

from _paths import RESULTS_DIR

JOB = "dakjch39k43c73aik9rg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None)
    a = ap.parse_args()
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=a.instance)
    job = service.job(JOB)
    plan = json.load(open(RESULTS_DIR / "kingston_rerun_plan_rung1.json"))
    tags = [j["tag"] for j in plan["jobs"]]
    pubs = job.inputs["pubs"]
    out = {"job_id": JOB, "backend": job.backend().name, "circuits": {}}
    for tag, pub in zip(tags, pubs):
        qc = pub[0] if isinstance(pub, (tuple, list)) else pub.circuit
        logical_to_physical = {}
        for inst in qc.data:
            if inst.operation.name == "measure":
                q = qc.find_bit(inst.qubits[0]).index
                c = qc.find_bit(inst.clbits[0]).index
                logical_to_physical[c] = q
        edges = {}
        for inst in qc.data:
            if inst.operation.num_qubits == 2 and inst.operation.name != "barrier":
                e = tuple(sorted(qc.find_bit(x).index for x in inst.qubits))
                edges[e] = edges.get(e, 0) + 1
        out["circuits"][tag] = {
            "logical_to_physical": {str(k): v for k, v in sorted(logical_to_physical.items())},
            "two_qubit_edges": [{"edge": list(e), "count": n} for e, n in sorted(edges.items())],
        }
    backend = service.backend(out["backend"])
    out["coupling_map"] = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    json.dump(out, open(RESULTS_DIR / "rung1_physical_layout.json", "w"), indent=1)
    p1 = out["circuits"]["rung1_qaoa_p1"]
    print("p=1 logical -> physical:", p1["logical_to_physical"])
    print(f"{len(p1['two_qubit_edges'])} two-qubit edges; coupling map {len(out['coupling_map'])} edges")
    print("-> results/rung1_physical_layout.json")


if __name__ == "__main__":
    main()
