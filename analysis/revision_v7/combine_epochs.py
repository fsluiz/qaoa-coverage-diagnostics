"""Descriptive variance subtraction for the three fixed-circuit jobs, without CIs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ENDPOINTS = {
    "joint_over_slice_gated": ("joint_over_slice_gated_log_scale", True),
    "faithful51_correlation_energy_retained": (
        "faithful51_correlation_retention_identity_scale", False
    ),
}


def combine(epochs: list[dict]) -> dict:
    if len(epochs) != 3 or len({e["job_id"] for e in epochs}) != 3:
        raise ValueError("Exactly three distinct jobs are required.")
    result = {}
    for key, (variance_key, log_scale) in ENDPOINTS.items():
        values = np.array([e["epoch_summary"][key]["estimate"] for e in epochs])
        model_values = np.log(values) if log_scale else values
        variances = np.array([
            e["epoch_summary"]["variance_description"][variance_key]
            ["working_variance_of_mean_method_of_moments"] for e in epochs
        ])
        observed, mean_variance = float(model_values.var(ddof=1)), float(variances.mean())
        excess = max(0.0, observed - mean_variance)
        result[key] = {
            "estimate": float(np.exp(model_values.mean())) if log_scale else float(values.mean()),
            "scale": "log" if log_scale else "identity",
            "epoch_values": values.tolist(),
            "observed_between_job_variance": observed,
            "epoch_estimation_variances": variances.tolist(),
            "mean_epoch_estimation_variance": mean_variance,
            "excess_between_job_variance_method_of_moments": excess,
            "excess_between_job_sd_method_of_moments": math.sqrt(excess),
            "observed_to_estimation_variance_ratio": observed / mean_variance,
        }
    return {
        "schema_version": 2,
        "analysis_version": "working_v7",
        "epochs": [{k: e[k] for k in ["epoch", "job_id", "calibration_date"]} for e in epochs],
        "between_job_summary": result,
        "interpretation": (
            "Three snapshots, two on one day, do not identify population variance, "
            "stationarity or a calibration cause. The subtraction uses a working "
            "independent-round variance approximation and is descriptive. No "
            "between-job confidence interval is reported. Distinct backend-properties "
            "timestamps do not establish independent physical replicates."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=Path, nargs=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = combine([json.loads(p.read_text()) for p in args.epochs])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["between_job_summary"], indent=2))


if __name__ == "__main__":
    main()
