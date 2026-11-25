#!/usr/bin/env python3
"""Reproduce and verify the working-v6 three-calibration campaign offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EPOCHS = ["e01_20261008", "e02_20261014", "e03_20261014"]
COMBINED = RESULTS / "e01_e02_e03_between_calibrations_v2.json"
MANIFEST = HERE / "working_manifest_v6.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, cwd=HERE.parent, env=env)


def verify_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "working_author_draft":
        raise RuntimeError("The v6 manifest is not marked as a working draft.")
    for relative, expected in manifest["sha256"].items():
        path = HERE.parent / relative
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Hash mismatch for {relative}: {observed} != {expected}"
            )


def assert_equivalent(expected, observed, path: str = "root") -> None:
    """Compare JSON data while tolerating last-bit floating-point variation."""
    if isinstance(expected, bool) or isinstance(observed, bool):
        if expected != observed:
            raise RuntimeError(f"Reproduction differs at {path}.")
    elif isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if not math.isclose(
            float(expected), float(observed), rel_tol=1e-10, abs_tol=1e-10
        ):
            raise RuntimeError(
                f"Reproduction differs at {path}: {expected} != {observed}."
            )
    elif isinstance(expected, dict) and isinstance(observed, dict):
        if expected.keys() != observed.keys():
            raise RuntimeError(f"Reproduction has different keys at {path}.")
        for key in expected:
            assert_equivalent(expected[key], observed[key], f"{path}.{key}")
    elif isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            raise RuntimeError(f"Reproduction has different length at {path}.")
        for index, (left, right) in enumerate(zip(expected, observed)):
            assert_equivalent(left, right, f"{path}[{index}]")
    elif expected != observed:
        raise RuntimeError(f"Reproduction differs at {path}.")


def compare_epoch(expected: dict, observed: dict) -> None:
    for key in [
        "campaign",
        "epoch",
        "job_id",
        "calibration_date",
        "bootstrap_replicates",
        "bootstrap_seed",
        "twirling_strata_archived",
        "ideal_correlation_energy",
        "rounds",
        "epoch_summary",
    ]:
        assert_equivalent(expected[key], observed[key], f"epoch.{key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Verify numerical outputs without regenerating the PDF figure.",
    )
    args = parser.parse_args()
    verify_manifest()

    with tempfile.TemporaryDirectory(prefix="qas-v6-") as temporary:
        temp = Path(temporary)
        reproduced_epochs = []
        for epoch in EPOCHS:
            plan = RESULTS / f"{epoch}_plan.json"
            counts = RESULTS / f"{epoch}_counts.json"
            expected_path = RESULTS / f"{epoch}_analysis_v2.json"
            observed_path = temp / f"{epoch}_analysis.json"
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            run(
                [
                    sys.executable,
                    str(HERE / "analyze_variance_epoch.py"),
                    "--plan",
                    str(plan),
                    "--counts",
                    str(counts),
                    "--out",
                    str(observed_path),
                    "--bootstrap",
                    str(expected["bootstrap_replicates"]),
                ]
            )
            observed = json.loads(observed_path.read_text(encoding="utf-8"))
            compare_epoch(expected, observed)
            reproduced_epochs.append(observed_path)

        reproduced_combined = temp / "between_calibrations.json"
        run(
            [
                sys.executable,
                str(HERE / "analyze_between_calibrations.py"),
                *[str(path) for path in reproduced_epochs],
                "--out",
                str(reproduced_combined),
            ]
        )
        assert_equivalent(
            json.loads(COMBINED.read_text(encoding="utf-8")),
            json.loads(reproduced_combined.read_text(encoding="utf-8")),
            "combined",
        )

        if not args.skip_figure:
            figure = temp / "fig_three_calibrations.pdf"
            environment = os.environ.copy()
            environment.setdefault("MPLCONFIGDIR", str(temp / "matplotlib"))
            run(
                [
                    sys.executable,
                    str(HERE / "make_three_calibration_figure.py"),
                    *[str(path) for path in reproduced_epochs],
                    "--combined",
                    str(reproduced_combined),
                    "--out",
                    str(figure),
                ],
                env=environment,
            )
            if not figure.is_file() or figure.stat().st_size == 0:
                raise RuntimeError("The three-calibration figure was not generated.")

    print("working-v6 three-calibration reproduction: PASS")


if __name__ == "__main__":
    main()
