#!/usr/bin/env python3
"""Compare the new standardized epoch with exact-circuit historical runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=1, allow_nan=False)
        stream.write("\n")


def comparison(old: float, new: float) -> dict:
    return {
        "historical": old,
        "new_epoch": new,
        "absolute_difference_new_minus_historical": new - old,
        "ratio_new_over_historical": new / old,
        "relative_change": new / old - 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-analysis", type=Path, required=True)
    parser.add_argument("--new-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")

    new = json.loads(args.new_analysis.read_text(encoding="utf-8"))
    plan = json.loads(args.new_plan.read_text(encoding="utf-8"))
    historical_interleaved = json.load(
        open(PROJECT / "results" / "interleaved_batch_v5.json")
    )
    historical_faithful = json.load(
        open(PROJECT / "results" / "hardware_decomposition.json")
    )

    old_gated = float(
        historical_interleaved["families"]["A"][
            "gated_round_geometric_mean"
        ]
    )
    new_gated = float(
        new["epoch_summary"]["joint_over_slice_gated"]["estimate"]
    )
    old_faithful = float(
        historical_faithful["faithful_matched"][
            "correlation_energy_retained_vs_ideal"
        ]
    )
    new_faithful = float(
        new["epoch_summary"][
            "faithful51_correlation_energy_retained"
        ]["estimate"]
    )
    qpy_hashes = {
        publication["tag"]: publication["qpy_sha256"]
        for publication in plan["pubs"]
        if publication["round"] == 0
    }
    output = {
        "schema_version": 1,
        "comparison_type": (
            "endpoint-specific descriptive between-calibration comparison"
        ),
        "new_epoch": {
            "job_id": new["job_id"],
            "calibration_date": new["calibration_date"],
        },
        "pef24_joint_over_slice_gated": {
            "historical_job_id": historical_interleaved["job_id"],
            "historical_calibration_date": historical_interleaved[
                "calibration_date"
            ],
            "historical_round_values": [
                value["gated"]
                for value in historical_interleaved["families"]["A"][
                    "per_round"
                ]
            ],
            "new_round_values": new["epoch_summary"]["round_values"][
                "joint_over_slice_gated"
            ],
            "qpy_sha256": {
                key: value
                for key, value in qpy_hashes.items()
                if key == "joint" or key.startswith("slice_")
            },
            **comparison(old_gated, new_gated),
        },
        "faithful51_correlation_energy_retained": {
            "historical_job_id": historical_faithful[
                "faithful_matched"
            ]["job_id"],
            "historical_calibration_date": "2026-09-28 09:25:27-03:00",
            "qpy_sha256": qpy_hashes["faithful51"],
            **comparison(old_faithful, new_faithful),
        },
        "limitations": [
            (
                "The historical interleaved rounds were globally shuffled "
                "rather than contiguous temporal blocks."
            ),
            (
                "The historical faithful circuit was one 100,000-shot pub, "
                "whereas the new epoch has four 10,000-shot pubs."
            ),
            (
                "Two calibration observations permit a descriptive contrast "
                "but not a stable between-calibration variance estimate."
            ),
        ],
        "status": (
            "All three levels are observable descriptively. Two additional "
            "standardized epochs remain planned for the confirmatory analysis."
        ),
    }
    write_new(args.out, output)
    print(
        "PEF joint/slice new/historical:",
        output["pef24_joint_over_slice_gated"]["ratio_new_over_historical"],
    )
    print(
        "faithful51 new/historical:",
        output["faithful51_correlation_energy_retained"][
            "ratio_new_over_historical"
        ],
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
