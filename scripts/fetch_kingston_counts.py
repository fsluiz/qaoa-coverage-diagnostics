#!/usr/bin/env python3
"""Retrieve raw bitstring distributions for an IBM Kingston job and archive them.

The analysis scripts originally stored only aggregates (P_opt, P_feas, <k>, mean energy),
which made the reported probabilities impossible to recompute from the repository alone.
This utility pulls the full counts for all or selected circuits in a job.

  python fetch_kingston_counts.py JOB_ID --plan results/PLAN.json --out results/COUNTS.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id", help="IBM Runtime job identifier")
    ap.add_argument("--plan", type=Path,
                    help="submission-plan JSON defining circuit order and tags")
    ap.add_argument("--out", required=True, type=Path,
                    help="destination JSON for the archived counts")
    ap.add_argument("--tag", action="append", dest="tags",
                    help="output tag; repeat in the same order as --pub-index")
    ap.add_argument("--pub-index", action="append", type=int, dest="indices",
                    help="retrieve only this publication index; may be repeated")
    ap.add_argument("--instance", default=None, help="IBM instance name, e.g. open-instance")
    args = ap.parse_args()

    plan = json.load(open(args.plan)) if args.plan else {}
    plan_tags = [j["tag"] for j in plan.get("jobs", [])]

    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(instance=args.instance)
    job = svc.job(args.job_id)
    result = job.result()
    if args.indices:
        if args.tags and len(args.tags) != len(args.indices):
            raise ValueError("repeat --tag once for each --pub-index")
        tags = args.tags or [plan_tags[i] if i < len(plan_tags) else f"circuit_{i}"
                             for i in args.indices]
        selected = list(zip(args.indices, tags))
    else:
        tags = args.tags or plan_tags
        if not tags and len(result) == 1:
            tags = ["faithful_exact51_p1"]
        if len(result) != len(tags):
            raise RuntimeError(f"job has {len(result)} circuits but {len(tags)} tags were supplied")
        selected = list(enumerate(tags))

    backend = job.backend().name
    if callable(backend):
        backend = backend()
    out = {"job_id": args.job_id, "backend": backend,
           "creation_date": str(job.creation_date),
           "circuits": {}}
    total_unique = 0
    for index, tag in selected:
        pub = result[index]
        counts = pub.data.meas.get_counts()
        shots = sum(counts.values())
        out["circuits"][tag] = {"shots": shots, "n_unique": len(counts), "counts": counts}
        total_unique += len(counts)
        print(f"{tag:26} shots={shots:6}  unique bitstrings={len(counts):6}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    path = args.out
    json.dump(out, open(path, "w"), indent=0, sort_keys=True)
    print(f"\n{len(out['circuits'])} circuits, {total_unique} unique bitstrings total")
    print(f"-> {path} ({path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
