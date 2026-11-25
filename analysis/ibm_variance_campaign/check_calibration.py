#!/usr/bin/env python3
"""Report whether the backend is ready for a distinct campaign epoch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_kingston")
    parser.add_argument("--account", default=None)
    parser.add_argument("plans", nargs="*", type=Path)
    args = parser.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = (
        QiskitRuntimeService(name=args.account)
        if args.account
        else QiskitRuntimeService()
    )
    backend = service.backend(args.backend)
    properties = backend.properties()
    current = str(properties.last_update_date) if properties is not None else None
    represented = []
    for path in args.plans:
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("submitted"):
            represented.append(plan.get("submission_calibration_date"))
    status = backend.status()
    print(
        json.dumps(
            {
                "backend": args.backend,
                "operational": status.operational,
                "pending_jobs": status.pending_jobs,
                "current_calibration": current,
                "represented_calibrations": represented,
                "eligible_for_new_epoch": (
                    status.operational and current not in represented
                ),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
