#!/usr/bin/env python3
"""Archive the counts and job-time calibration of the joint/separate batch job.

Writes results/joint_separate_batch_counts.json with, for each of the 14 pubs, the raw counts,
and for the physical qubits and couplers used, the readout and two-qubit error rates from the
calibration attached to the job.  Needs IBM access once; calc_joint_separate_batch.py then
works offline.

    python fetch_joint_separate_batch.py --instance open-instance
"""
from __future__ import annotations

import argparse
import json

from _paths import RESULTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None)
    a = ap.parse_args()
    plan = json.load(open(RESULTS_DIR / "joint_separate_batch_plan.json"))
    from qiskit_ibm_runtime import QiskitRuntimeService
    job = QiskitRuntimeService(instance=a.instance).job(plan["job_id"])
    result = job.result()
    props = job.properties()
    qubits = sorted({q for p in plan["pubs"] for q in p["measured_physical"].values()})
    edges = sorted({tuple(e["edge"]) for p in plan["pubs"] for e in p["two_qubit_edges"]})
    readout = {str(q): float(props.readout_error(q)) for q in qubits}
    gate_err = {}
    for e in edges:
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
           "readout_errors": readout, "two_qubit_errors": gate_err,
           "usage_seconds": job.usage_estimation.get("quantum_seconds"), "pubs": {}}
    for meta, pub in zip(plan["pubs"], result):
        names = list(pub.data.keys())          # joint: 'meas'; block slices: 'c'
        assert len(names) == 1, names
        counts = getattr(pub.data, names[0]).get_counts()
        out["pubs"][meta["tag"]] = {"shots": sum(counts.values()), "counts": counts}
        print(f"{meta['tag']:9} shots={sum(counts.values())}")
    json.dump(out, open(RESULTS_DIR / "joint_separate_batch_counts.json", "w"), indent=0,
              sort_keys=True)
    print("-> results/joint_separate_batch_counts.json")


if __name__ == "__main__":
    main()
