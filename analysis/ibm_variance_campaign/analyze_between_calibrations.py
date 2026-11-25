#!/usr/bin/env python3
"""Combine at least three independently calibrated campaign epochs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def summary(
    values: np.ndarray,
    epoch_estimation_variances: np.ndarray,
    log_scale: bool,
) -> dict:
    transformed = np.log(values) if log_scale else values
    n = len(values)
    mean = float(transformed.mean())
    observed_variance = float(transformed.var(ddof=1))
    mean_estimation_variance = float(epoch_estimation_variances.mean())
    excess_variance = max(0.0, observed_variance - mean_estimation_variance)
    half = float(
        student_t.ppf(0.975, n - 1)
        * transformed.std(ddof=1)
        / math.sqrt(n)
    )
    if log_scale:
        estimate = float(np.exp(mean))
        interval = [float(np.exp(mean - half)), float(np.exp(mean + half))]
    else:
        estimate = mean
        interval = [mean - half, mean + half]
    return {
        "estimate": estimate,
        "ci95": interval,
        "scale": "log" if log_scale else "identity",
        "df": n - 1,
        "epoch_values": values.tolist(),
        "observed_between_epoch_variance": observed_variance,
        "observed_between_epoch_sd": math.sqrt(observed_variance),
        "epoch_estimation_variances": epoch_estimation_variances.tolist(),
        "mean_epoch_estimation_variance": mean_estimation_variance,
        "excess_between_calibration_variance_method_of_moments": (
            excess_variance
        ),
        "excess_between_calibration_sd_method_of_moments": math.sqrt(
            excess_variance
        ),
        "t_interval_note": (
            "Interval from the dispersion of three epoch point estimates; "
            "it is descriptive and is not a total uncertainty interval."
        ),
    }


def epoch_mean_variance(epoch: dict, key: str) -> float:
    """Approximate estimation variance of one epoch mean on its model scale."""
    component = epoch["epoch_summary"]["variance_description"][key]
    per_round = (
        component["mean_conditional_shot_variance"]
        + component["excess_round_variance_method_of_moments"]
    )
    return float(per_round / len(epoch["rounds"]))


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=1, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if len(args.epochs) < 3:
        raise ValueError("At least three calibrated epochs are required.")

    epochs = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.epochs
    ]
    calibrations = [epoch["calibration_date"] for epoch in epochs]
    if len(set(calibrations)) != len(calibrations):
        raise ValueError(
            "Epochs must have distinct job-time calibration timestamps."
        )
    job_ids = [epoch["job_id"] for epoch in epochs]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("Duplicate IBM job IDs were supplied.")

    gated = np.asarray(
        [
            epoch["epoch_summary"]["joint_over_slice_gated"]["estimate"]
            for epoch in epochs
        ]
    )
    faithful = np.asarray(
        [
            epoch["epoch_summary"][
                "faithful51_correlation_energy_retained"
            ]["estimate"]
            for epoch in epochs
        ]
    )
    gated_epoch_variances = np.asarray(
        [
            epoch_mean_variance(
                epoch, "joint_over_slice_gated_log_scale"
            )
            for epoch in epochs
        ]
    )
    faithful_epoch_variances = np.asarray(
        [
            epoch_mean_variance(
                epoch,
                "faithful51_correlation_retention_identity_scale",
            )
            for epoch in epochs
        ]
    )
    output = {
        "schema_version": 1,
        "campaign": "ibm_three_level_variance_v6",
        "levels": {
            "shot": (
                "conditional bootstrap within each pub; see epoch files"
            ),
            "within_job": (
                "four contiguous round estimates within each IBM job"
            ),
            "between_calibration": (
                "epoch estimates from distinct calibration timestamps"
            ),
        },
        "epochs": [
            {
                "epoch": epoch["epoch"],
                "job_id": epoch["job_id"],
                "calibration_date": epoch["calibration_date"],
            }
            for epoch in epochs
        ],
        "between_calibration_summary": {
            "joint_over_slice_gated": summary(
                gated, gated_epoch_variances, log_scale=True
            ),
            "faithful51_correlation_energy_retained": summary(
                faithful, faithful_epoch_variances, log_scale=False
            ),
        },
        "interpretation": (
            "The method-of-moments excess variance subtracts the mean "
            "estimated variance of an epoch mean from the observed variance "
            "among epoch estimates. With three epochs it is descriptive, has "
            "no stable variance interval, and does not establish long-term "
            "backend stationarity. The t intervals use only three epoch point "
            "estimates and are not total uncertainty intervals."
        ),
    }
    write_new(args.out, output)
    print(json.dumps(output["between_calibration_summary"], indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
