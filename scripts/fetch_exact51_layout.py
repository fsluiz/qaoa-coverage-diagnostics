#!/usr/bin/env python3
"""Archive where the two faithful 51-qubit executions ran: logical -> physical qubits.

Reads the submitted, backend-transpiled circuit of each faithful-encoding job and records,
for every logical qubit (the clbit it is measured into), the physical Kingston qubit that
carried it at measurement, together with every two-qubit gate.  Needs IBM access once;
calc_exact51_cross_block.py and make_exact51_layout_figure.py then work offline.

    python fetch_exact51_layout.py --instance open-instance
"""
from __future__ import annotations

import argparse
import json

from _paths import RESULTS_DIR

JOBS = (("defaults", "dakjjpfi3e6s738rmbng"), ("matched", "dat665lvr3kc73el167g"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None)
    a = ap.parse_args()
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=a.instance)
    out = {}
    for tag, jid in JOBS:
        job = service.job(jid)
        pub = job.inputs["pubs"][0]
        qc = pub[0] if isinstance(pub, (tuple, list)) else pub.circuit
        l2p, edges = {}, {}
        for inst in qc.data:
            if inst.operation.name == "measure":
                l2p[qc.find_bit(inst.clbits[0]).index] = qc.find_bit(inst.qubits[0]).index
            elif inst.operation.num_qubits == 2 and inst.operation.name != "barrier":
                e = tuple(sorted(qc.find_bit(x).index for x in inst.qubits))
                edges[e] = edges.get(e, 0) + 1
        out[tag] = {"job_id": jid,
                    "logical_to_physical": {str(k): v for k, v in sorted(l2p.items())},
                    "two_qubit_edges": [{"edge": list(e), "count": c}
                                        for e, c in sorted(edges.items())]}
        print(f"{tag}: {len(l2p)} measured qubits, {sum(edges.values())} two-qubit gates")
    json.dump(out, open(RESULTS_DIR / "exact51_physical_layout.json", "w"), indent=1)
    print("-> results/exact51_physical_layout.json")


if __name__ == "__main__":
    main()
