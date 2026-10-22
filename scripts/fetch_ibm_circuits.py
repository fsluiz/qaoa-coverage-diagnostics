#!/usr/bin/env python3
"""Archive the backend-transpiled circuits of the executed Kingston jobs as QPY.

The layout files keep only the logical -> physical map at measurement and the aggregate list
of two-qubit edges.  Routing SWAPs move logical information during the circuit, so the causal
analysis (calc_exact51_cross_block.py) and the new interleaved experiment need the full instruction
sequence.  This script reads the submitted circuits of

  rung1_original   job dakjch39k43c73aik9rg, circuit rung1_qaoa_p1 (the principal 24-qubit run)
  batch_*          job dau04mahcrkc73dur170, the joint pub and its 13 block slices
  exact51_*        jobs dakjjpfi3e6s738rmbng (defaults) and dat665lvr3kc73el167g (matched)

and writes results/ibm_circuits/<tag>.qpy, a plain instruction list <tag>.ops.json (name, physical
qubits, clbits; readable without Qiskit, for the offline light-cone analysis) and an index
with the SHA-256 of each file.  Read only: no job is submitted.  Needs IBM access once.

    python fetch_ibm_circuits.py --instance open-instance
    python fetch_ibm_circuits.py --from-qpy        # rebuild the .ops.json files offline
"""
from __future__ import annotations

import argparse
import hashlib
import json

from _paths import RESULTS_DIR

OUT = RESULTS_DIR / "ibm_circuits"


def circuit_of(pub):
    return pub[0] if isinstance(pub, (tuple, list)) else pub.circuit


def ops_json(qc) -> list:
    return [[i.operation.name, [qc.find_bit(q).index for q in i.qubits],
             [qc.find_bit(c).index for c in i.clbits]]
            for i in qc.data if i.operation.name != "barrier"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None)
    ap.add_argument("--from-qpy", action="store_true")
    a = ap.parse_args()
    from qiskit import qpy
    OUT.mkdir(exist_ok=True)
    if a.from_qpy:
        index = json.load(open(OUT / "index.json"))
        for tag, row in index.items():
            with open(OUT / row["file"], "rb") as f:
                qc = qpy.load(f)[0]
            json.dump(ops_json(qc), open(OUT / f"{tag}.ops.json", "w"))
            row["ops_sha256"] = hashlib.sha256((OUT / f"{tag}.ops.json").read_bytes()).hexdigest()
        json.dump(index, open(OUT / "index.json", "w"), indent=1)
        print(f"-> {OUT}/*.ops.json")
        return
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=a.instance)
    wanted = []
    rung1_tags = [j["tag"] for j in
                  json.load(open(RESULTS_DIR / "kingston_rerun_plan_rung1.json"))["jobs"]]
    pubs = service.job("dakjch39k43c73aik9rg").inputs["pubs"]
    wanted.append(("rung1_original", "dakjch39k43c73aik9rg",
                   circuit_of(pubs[rung1_tags.index("rung1_qaoa_p1")])))
    plan = json.load(open(RESULTS_DIR / "joint_separate_batch_plan.json"))
    pubs = service.job(plan["job_id"]).inputs["pubs"]
    for meta, pub in zip(plan["pubs"], pubs):
        wanted.append((f"batch_{meta['tag']}", plan["job_id"], circuit_of(pub)))
    for tag, jid in (("exact51_defaults", "dakjjpfi3e6s738rmbng"),
                     ("exact51_matched", "dat665lvr3kc73el167g")):
        wanted.append((tag, jid, circuit_of(service.job(jid).inputs["pubs"][0])))
    index = {}
    for tag, jid, qc in wanted:
        path = OUT / f"{tag}.qpy"
        with open(path, "wb") as f:
            qpy.dump(qc, f)
        ops = OUT / f"{tag}.ops.json"
        json.dump(ops_json(qc), open(ops, "w"))
        n2q = sum(1 for i in qc.data if i.operation.num_qubits == 2
                  and i.operation.name != "barrier")
        index[tag] = dict(job_id=jid, file=path.name, num_qubits=qc.num_qubits,
                          num_clbits=qc.num_clbits, two_qubit_gates=n2q, depth=qc.depth(),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          ops_sha256=hashlib.sha256(ops.read_bytes()).hexdigest())
        print(f"{tag:22} {jid}  2Q={n2q:4d}  depth={qc.depth()}")
    json.dump(index, open(OUT / "index.json", "w"), indent=1)
    print(f"-> {OUT}/")


if __name__ == "__main__":
    main()
