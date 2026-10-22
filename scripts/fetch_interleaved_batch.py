#!/usr/bin/env python3
"""Archive the counts and job-time calibration of the interleaved batch job.

Writes results/interleaved_batch_counts.json with, for each of the 136 pubs (keyed by position), the raw counts,
and for the physical qubits and couplers used, the readout and two-qubit error rates and T1/T2 from the
calibration attached to the job.  Needs IBM access once; calc_interleaved_batch.py then
works offline.

    python fetch_interleaved_batch.py --instance open-instance
"""
from __future__ import annotations

import argparse
import json

from _paths import RESULTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None)
    a = ap.parse_args()
    plan = json.load(open(RESULTS_DIR / "interleaved_batch_plan.json"))
    from qiskit_ibm_runtime import QiskitRuntimeService
    job = QiskitRuntimeService(instance=a.instance).job(plan["job_id"])
    result = job.result()
    props = job.properties()
    qubits = sorted({q for p in plan["pubs"] for q in p["measured_physical"].values()})
    readout = {str(q): float(props.readout_error(q)) for q in qubits}
    coherence = {str(q): dict(t1_us=float(props.t1(q)) * 1e6, t2_us=float(props.t2(q)) * 1e6)
                 for q in qubits}
    edges, circuit_edges = set(), {}
    for meta, pub in zip(plan["pubs"], job.inputs["pubs"]):
        qc = pub[0] if isinstance(pub, (tuple, list)) else pub.circuit
        count = {}
        for i in qc.data:
            if i.operation.num_qubits == 2 and i.operation.name != "barrier":
                e = tuple(sorted(qc.find_bit(x).index for x in i.qubits))
                count[e] = count.get(e, 0) + 1
        edges |= set(count)
        circuit_edges[meta["tag"]] = [{"edge": list(e), "count": c} for e, c in sorted(count.items())]
    gate_err = {}
    for e in sorted(edges):
        err = None
        for name in ("cz", "ecr", "cx"):
            try:
                err = float(props.gate_error(name, list(e)))
                break
            except Exception:
                continue
        gate_err[f"{e[0]},{e[1]}"] = err
    out = {"job_id": plan["job_id"], "backend": job.backend().name,
           "creation_date": str(job.creation_date),
           "calibration_date": str(props.last_update_date),
           "readout_errors": readout, "coherence": coherence, "two_qubit_errors": gate_err,
           "circuit_two_qubit_edges": circuit_edges,
           "usage_seconds": job.metrics().get("usage", {}).get("quantum_seconds"), "pubs": {}}
    for meta, pub in zip(plan["pubs"], result):
        names = list(pub.data.keys())          # joint: 'meas'; block slices: 'c'
        assert len(names) == 1, names
        counts = getattr(pub.data, names[0]).get_counts()
        out["pubs"][str(meta["position"])] = {"shots": sum(counts.values()), "counts": counts}
        print(f"{meta['position']:3d} {meta['tag']:8} shots={sum(counts.values())}")
    json.dump(out, open(RESULTS_DIR / "interleaved_batch_counts.json", "w"), indent=0,
              sort_keys=True)
    print("-> results/interleaved_batch_counts.json")


if __name__ == "__main__":
    main()
