#!/usr/bin/env python3
"""Plot the two primary endpoints across three standardized IBM epochs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def endpoint(epoch: dict, key: str) -> tuple[float, np.ndarray]:
    estimate = float(epoch["epoch_summary"][key]["estimate"])
    interval = np.asarray(
        epoch["epoch_summary"][
            "hierarchical_round_and_shot_ci95_descriptive"
        ][key],
        dtype=float,
    )
    return estimate, interval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", nargs=3, type=Path)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    epochs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.epochs
    ]
    combined = json.loads(args.combined.read_text(encoding="utf-8"))
    calibrations = [epoch["calibration_date"] for epoch in epochs]
    if len(set(calibrations)) != 3:
        raise ValueError("The figure requires three distinct calibrations.")

    joint = [endpoint(epoch, "joint_over_slice_gated") for epoch in epochs]
    faithful = [
        endpoint(epoch, "faithful51_correlation_energy_retained")
        for epoch in epochs
    ]
    labels = ["E01\nOct 8", "E02\nOct 14", "E03\nOct 14"]
    colors = ["#4477AA", "#EE6677", "#228833"]
    x = np.arange(3)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for axis, values, key, ylabel, title in [
        (
            axes[0],
            joint,
            "joint_over_slice_gated",
            "joint / isolated retention",
            "(a) PEF-24 gated blocks",
        ),
        (
            axes[1],
            faithful,
            "faithful51_correlation_energy_retained",
            "ideal correlation energy retained",
            "(b) Faithful 51-qubit circuit",
        ),
    ]:
        estimates = np.asarray([value[0] for value in values])
        intervals = np.vstack([value[1] for value in values])
        errors = np.vstack(
            [estimates - intervals[:, 0], intervals[:, 1] - estimates]
        )
        for index in range(3):
            axis.errorbar(
                x[index],
                estimates[index],
                yerr=errors[:, index].reshape(2, 1),
                color=colors[index],
                marker="o",
                markersize=5,
                linestyle="none",
                capsize=3,
                linewidth=1.2,
            )
        across_epoch = combined["between_calibration_summary"][key][
            "estimate"
        ]
        axis.axhline(
            across_epoch,
            color="#555555",
            linestyle="--",
            linewidth=0.9,
            label="three-epoch mean",
        )
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)

    axes[0].set_ylim(0.74, 0.97)
    axes[1].set_ylim(0.44, 0.56)
    axes[0].legend(frameon=False, loc="lower right")
    figure.tight_layout(w_pad=2.2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
