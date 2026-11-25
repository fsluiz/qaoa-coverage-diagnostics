#!/usr/bin/env python3
"""Fetch one submitted variance-campaign job without altering old evidence."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=0, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--account", default=None)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not plan.get("submitted") or not plan.get("job_id"):
        raise ValueError("The plan does not contain a submitted job.")

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = (
        QiskitRuntimeService(name=args.account)
        if args.account
        else QiskitRuntimeService()
    )
    job = service.job(plan["job_id"])
    print(f"{job.job_id()} status before result(): {job.status()}")
    result = job.result()
    properties = job.properties()
    metrics = job.metrics()
    usage = metrics.get("usage", {})
    usage_seconds = usage.get(
        "quantum_seconds", usage.get("qpu_charge_time_seconds")
    )
    usage_resource_units = (
        usage.get("value") if usage.get("unit") == "resource_units" else None
    )
    archive = {
        "schema_version": 1,
        "campaign": plan["campaign"],
        "epoch": plan["epoch"],
        "job_id": job.job_id(),
        "backend": job.backend().name,
        "creation_date": str(job.creation_date),
        "calibration_date": (
            str(properties.last_update_date) if properties is not None else None
        ),
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "usage_seconds": usage_seconds,
        "usage_resource_units": usage_resource_units,
        "execution_timestamps": metrics.get("timestamps"),
        "twirling_strata_archived": False,
        "pubs": {},
    }
    if len(result) != len(plan["pubs"]):
        raise RuntimeError(
            f"Expected {len(plan['pubs'])} pubs, received {len(result)}."
        )
    for metadata, pub_result in zip(plan["pubs"], result):
        register_names = list(pub_result.data.keys())
        if len(register_names) != 1:
            raise RuntimeError(
                f"Unexpected classical registers at position "
                f"{metadata['position']}: {register_names}"
            )
        counts = getattr(pub_result.data, register_names[0]).get_counts()
        shots = sum(counts.values())
        if shots != metadata["shots"]:
            raise RuntimeError(
                f"Position {metadata['position']} has {shots} shots; "
                f"expected {metadata['shots']}."
            )
        archive["pubs"][str(metadata["position"])] = {
            "round": metadata["round"],
            "tag": metadata["tag"],
            "shots": shots,
            "counts": counts,
        }
        print(
            f"{metadata['position']:3d} round={metadata['round']} "
            f"{metadata['tag']:10s} shots={shots}"
        )
    write_new(args.out, archive)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
