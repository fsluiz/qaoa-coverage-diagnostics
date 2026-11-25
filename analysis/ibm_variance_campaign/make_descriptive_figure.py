#!/usr/bin/env python3
"""Plot the endpoint-specific calibration and round comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text(encoding="utf-8"))
    historical = json.load(
        open(PROJECT / "results" / "interleaved_batch_v5.json")
    )
    decomposition = json.load(
        open(PROJECT / "results" / "hardware_decomposition.json")
    )

    new_gated = np.asarray(
        [
            row["block"]["joint_over_slice_gated"]
            for row in data["rounds"]
        ]
    )
    new_gated_ci = np.asarray(
        [
            row["block"]["shot_bootstrap_ci95_conditional"][
                "joint_over_slice_gated"
            ]
            for row in data["rounds"]
        ]
    )
    old_gated = np.asarray(
        [
            row["gated"]
            for row in historical["families"]["A"]["per_round"]
        ]
    )
    new_faithful = np.asarray(
        [
            row["faithful51"]["correlation_energy_retained"]
            for row in data["rounds"]
        ]
    )
    new_faithful_ci = np.asarray(
        [
            row["faithful51"]["shot_bootstrap_ci95_conditional"][
                "correlation_energy_retained"
            ]
            for row in data["rounds"]
        ]
    )
    old_faithful = float(
        decomposition["faithful_matched"][
            "correlation_energy_retained_vs_ideal"
        ]
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    axis = axes[0]
    x_old = np.arange(4)
    x_new = np.arange(4) + 5
    axis.scatter(x_old, old_gated, color="#777777", marker="s", label="2026-09-30")
    lower = new_gated - new_gated_ci[:, 0]
    upper = new_gated_ci[:, 1] - new_gated
    axis.errorbar(
        x_new,
        new_gated,
        yerr=np.vstack([lower, upper]),
        color="#0066aa",
        marker="o",
        linestyle="none",
        capsize=2,
        label="2026-10-08",
    )
    axis.hlines(
        historical["families"]["A"]["gated_round_geometric_mean"],
        -0.35,
        3.35,
        color="#777777",
        linewidth=1,
    )
    axis.hlines(
        data["epoch_summary"]["joint_over_slice_gated"]["estimate"],
        4.65,
        8.35,
        color="#0066aa",
        linewidth=1,
    )
    axis.set_xticks([1.5, 6.5], ["historical", "epoch 1"])
    axis.set_ylabel("joint / isolated retention")
    axis.set_title("(a) PEF-24 gated blocks")
    axis.legend(frameon=False, loc="lower left")

    axis = axes[1]
    x_new = np.arange(4) + 1
    lower = new_faithful - new_faithful_ci[:, 0]
    upper = new_faithful_ci[:, 1] - new_faithful
    axis.scatter(
        [0],
        [old_faithful],
        color="#777777",
        marker="s",
        label="2026-09-28",
    )
    axis.errorbar(
        x_new,
        new_faithful,
        yerr=np.vstack([lower, upper]),
        color="#aa3377",
        marker="o",
        linestyle="none",
        capsize=2,
        label="2026-10-08",
    )
    axis.hlines(
        data["epoch_summary"][
            "faithful51_correlation_energy_retained"
        ]["estimate"],
        0.65,
        4.35,
        color="#aa3377",
        linewidth=1,
    )
    axis.set_xticks([0, 2.5], ["historical", "epoch 1"])
    axis.set_ylabel("ideal correlation energy retained")
    axis.set_title("(b) Faithful 51-qubit circuit")
    axis.legend(frameon=False, loc="lower left")

    figure.suptitle(
        "Fixed-circuit descriptive comparison across rounds and calibrations",
        y=1.01,
    )
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
