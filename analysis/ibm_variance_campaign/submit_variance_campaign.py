#!/usr/bin/env python3
"""Submit one calibration epoch of the three-level IBM variance campaign.

The script deliberately reuses archived ISA circuits without retranspilation:

* one 24-qubit joint PEF circuit;
* its thirteen block slices on the same physical placement;
* one matched-mitigation faithful 51-qubit circuit.

Each round is a contiguous temporal block. Circuit order is randomized only
inside a round. The output is created exclusively and is never overwritten.
Without --submit this is an offline dry run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit
import qiskit_ibm_runtime
from qiskit import qpy


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
QPY_DIR = PROJECT / "results" / "ibm_circuits"
TAGS = ["joint", *[f"slice_{i:02d}" for i in range(13)], "faithful51"]
QPY_NAME = {
    "joint": "batch_joint.qpy",
    **{f"slice_{i:02d}": f"batch_block_{i:02d}.qpy" for i in range(13)},
    "faithful51": "exact51_matched.qpy",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_circuit(tag: str):
    path = QPY_DIR / QPY_NAME[tag]
    with path.open("rb") as stream:
        circuit = qpy.load(stream)[0]
    return path, circuit


def n_two_qubit(circuit) -> int:
    return sum(
        instruction.operation.num_qubits == 2
        and instruction.operation.name != "barrier"
        for instruction in circuit.data
    )


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=1, sort_keys=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_kingston")
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--randomizations", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--account", default=None)
    parser.add_argument(
        "--distinct-from",
        action="append",
        default=[],
        type=Path,
        help=(
            "Refuse submission if the current backend calibration matches "
            "the submitted plan in this file. Repeat for multiple epochs."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.rounds < 2:
        raise ValueError("At least two rounds are required for within-job variation.")
    if args.randomizations < 1 or args.shots % args.randomizations:
        raise ValueError("shots must be divisible by randomizations.")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")

    loaded = {tag: load_circuit(tag) for tag in TAGS}
    circuits = {tag: item[1] for tag, item in loaded.items()}
    rng = np.random.default_rng(args.seed)
    order: list[tuple[int, int, str]] = []
    for round_index in range(args.rounds):
        for within_round, index in enumerate(rng.permutation(len(TAGS))):
            order.append((round_index, within_round, TAGS[int(index)]))

    pubs = []
    for position, (round_index, within_round, tag) in enumerate(order):
        path, circuit = loaded[tag]
        pubs.append(
            {
                "position": position,
                "round": round_index,
                "within_round_position": within_round,
                "tag": tag,
                "role": (
                    "joint"
                    if tag == "joint"
                    else "faithful_encoding"
                    if tag == "faithful51"
                    else "isolated_slice"
                ),
                "qpy": str(path.relative_to(PROJECT)),
                "qpy_sha256": sha256(path),
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "n_2q": n_two_qubit(circuit),
                "shots": args.shots,
            }
        )

    now = datetime.now(timezone.utc).isoformat()
    plan = {
        "schema_version": 1,
        "campaign": "ibm_three_level_variance_v6",
        "epoch": args.epoch,
        "created_utc": now,
        "backend": args.backend,
        "rounds": args.rounds,
        "circuits_per_round": len(TAGS),
        "shots_per_pub": args.shots,
        "num_randomizations": args.randomizations,
        "shots_per_randomization": args.shots // args.randomizations,
        "order_seed": args.seed,
        "round_definition": "contiguous temporal block; random order within round",
        "mitigation": "XY4 DD + gate and measurement twirling",
        "software": {
            "python": platform.python_version(),
            "qiskit": qiskit.__version__,
            "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__,
            "numpy": np.__version__,
        },
        "status": "dry_run",
        "submitted": False,
        "job_id": None,
        "submission_calibration_date": None,
        "pubs": pubs,
    }

    print(
        f"{args.epoch}: {args.rounds} rounds x {len(TAGS)} circuits = "
        f"{len(order)} pubs, {len(order) * args.shots:,} shots"
    )
    for tag in TAGS:
        circuit = circuits[tag]
        print(
            f"{tag:10s} clbits={circuit.num_clbits:2d} "
            f"depth={circuit.depth():3d} n2q={n_two_qubit(circuit):3d}"
        )

    if args.submit:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = (
            QiskitRuntimeService(name=args.account)
            if args.account
            else QiskitRuntimeService()
        )
        backend = service.backend(args.backend)
        status = backend.status()
        if not status.operational:
            raise RuntimeError(f"{args.backend} is not operational: {status.status_msg}")
        properties = backend.properties()
        calibration_date = (
            str(properties.last_update_date) if properties is not None else None
        )
        previous_calibrations = []
        for previous_path in args.distinct_from:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if not previous.get("submitted"):
                raise ValueError(
                    f"Reference plan {previous_path} was not submitted."
                )
            previous_calibrations.append(
                previous.get("submission_calibration_date")
            )
        if calibration_date in previous_calibrations:
            raise RuntimeError(
                "Current calibration is already represented by a prior epoch: "
                f"{calibration_date}. No job was submitted."
            )

        sampler = SamplerV2(mode=backend)
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        sampler.options.twirling.num_randomizations = args.randomizations
        sampler.options.twirling.shots_per_randomization = (
            args.shots // args.randomizations
        )
        job = sampler.run(
            [(circuits[tag], None, args.shots) for _, _, tag in order]
        )
        plan.update(
            {
                "status": "submitted",
                "submitted": True,
                "job_id": job.job_id(),
                "submission_calibration_date": calibration_date,
                "backend_status_at_submission": {
                    "operational": status.operational,
                    "pending_jobs": status.pending_jobs,
                    "status_msg": status.status_msg,
                },
            }
        )
        print(
            f"submitted {job.job_id()} on calibration {calibration_date}; "
            f"pending jobs before submission: {status.pending_jobs}"
        )
    else:
        print("DRY RUN -- add --submit only after reviewing this plan.")

    write_new(args.out, plan)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
