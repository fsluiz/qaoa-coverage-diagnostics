"""Reproduce the v7 corrections offline in a temporary directory."""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from manifest import HERE, PROJECT, STEMS, MANIFEST, verify


def compare(a, b, path="root"):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f"Keys differ at {path}")
        for k in a:
            compare(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise AssertionError(f"Lengths differ at {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
            raise AssertionError(f"Numerical mismatch at {path}: {a} != {b}")
    elif a != b:
        raise AssertionError(f"Mismatch at {path}: {a} != {b}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Hashes, tests and small controls; skip campaign/ensemble and 14x10000 block bootstrap.")
    parser.add_argument("--report", type=Path, help="Write the completed verification report.")
    args = parser.parse_args()
    start = time.monotonic()
    checked = verify()
    report = dict(status="PASS", dependency_count=checked, full_reanalysis=not args.quick,
                  input_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  numerical_tolerance={"relative": 1e-10, "absolute": 1e-10},
                  steps=[])
    with tempfile.TemporaryDirectory(prefix="qas-v7-verify-") as temporary:
        work = Path(temporary)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", MPLCONFIGDIR=str(work/"matplotlib"))

        def run(command):
            result = subprocess.run(command, cwd=PROJECT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout)
            return result.stdout

        text = run([sys.executable, str(HERE/"test_revision.py")])
        report["steps"].append({"name": "v7 independent/regression tests", "output": text.strip()})
        print("PASS: v7 tests", flush=True)
        text = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    str(PROJECT/"tests/test_v5_corrections.py")])
        report["steps"].append({"name": "pre-existing v5 regression tests", "output": text.strip()})
        print("PASS: pre-existing regression tests", flush=True)
        # New deterministic diagnostics also run in quick mode; they do not retrain circuits.
        new_artifacts = work/"diagnostics"
        for script in ("diagnostic_controls.py", "campaign_sensitivity.py", "mixer_summary.py"):
            run([sys.executable, str(HERE/script), "--out-dir", str(new_artifacts)])
        for name in ("diagnostic_controls.json", "campaign_sensitivity.json", "mixer_summary.json"):
            compare(json.loads((HERE/"generated"/name).read_text()),
                    json.loads((new_artifacts/name).read_text()))
        for name in ("diagnostic_numbers.tex", "sensitivity_numbers.tex",
                     "sensitivity_table.tex", "topology_table.tex"):
            if (HERE/"generated"/name).read_text() != (new_artifacts/name).read_text():
                raise AssertionError(f"Generated diagnostic LaTeX differs: {name}")
        if run(["pdftotext", str(HERE/"generated/fig_diagnostic_controls.pdf"), "-"]) != run(
                ["pdftotext", str(new_artifacts/"fig_diagnostic_controls.pdf"), "-"]):
            raise AssertionError("Diagnostic figure text differs.")
        report["steps"].append({"name": "output controls, campaign sensitivity, mixer metadata/topology",
                                "status": "PASS"})
        print("PASS: new offline diagnostics and artifacts", flush=True)
        block_input = HERE/"generated/block_covariance.json"
        if not args.quick:
            run([sys.executable, str(HERE/"block_covariance.py"), "--out-dir", str(new_artifacts)])
            block_input = new_artifacts/"block_covariance.json"
            compare(json.loads((HERE/"generated/block_covariance.json").read_text()),
                    json.loads(block_input.read_text()))
            print("PASS: 14 joint block bootstraps, 10000 draws each", flush=True)
        run([sys.executable, str(HERE/"block_covariance_artifacts.py"), "--input", str(block_input),
             "--out-dir", str(new_artifacts)])
        for name in ("block_covariance_table.tex", "block_covariance_numbers.tex"):
            if (HERE/"generated"/name).read_text() != (new_artifacts/name).read_text():
                raise AssertionError(f"Block LaTeX differs: {name}")
        if run(["pdftotext", str(HERE/"generated/fig_block_covariance.pdf"), "-"]) != run(
                ["pdftotext", str(new_artifacts/"fig_block_covariance.pdf"), "-"]):
            raise AssertionError("Block figure text differs.")
        report["steps"].append({"name": "block profiles, simultaneous bands and artifacts",
                                "raw_count_reanalysis": not args.quick, "status": "PASS"})
        if not args.quick:
            outputs = [work/f"e0{i}_analysis.json" for i in range(1, 4)]
            archive = HERE.parent/"ibm_variance_campaign/results"
            def epoch(item):
                stem, output = item
                run([sys.executable, str(HERE/"analyze_epoch.py"),
                     "--plan", str(archive/f"{stem}_plan.json"),
                     "--counts", str(archive/f"{stem}_counts.json"),
                     "--out", str(output), "--bootstrap", "4000"])
                compare(json.loads((HERE/"results"/output.name).read_text()),
                        json.loads(output.read_text()))
                print(f"PASS: {stem} from raw counts", flush=True)
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(epoch, zip(STEMS, outputs)))
            run([sys.executable, str(HERE/"combine_epochs.py"), *map(str, outputs),
                 "--out", str(work/"between_jobs.json")])
            compare(json.loads((HERE/"results/between_jobs.json").read_text()),
                    json.loads((work/"between_jobs.json").read_text()))
            run([sys.executable, str(HERE/"campaign_artifacts.py"), "--results", str(work),
                 "--out-dir", str(work/"generated")])
            run([sys.executable, str(HERE/"ensemble_analysis.py"),
                 "--out-dir", str(work/"generated")])
            compare(json.loads((HERE/"generated/ensemble_analysis.json").read_text()),
                    json.loads((work/"generated/ensemble_analysis.json").read_text()))
            for name in ("campaign_table.tex", "campaign_numbers.tex", "ensemble_table.tex"):
                if (HERE/"generated"/name).read_text() != (work/"generated"/name).read_text():
                    raise AssertionError(f"Generated LaTeX differs: {name}")
            for name in ("fig_three_jobs.pdf", "fig_ensemble.pdf"):
                # PDF metadata timestamps can differ; compare visible labels.
                original = run(["pdftotext", str(HERE/"generated"/name), "-"])
                regenerated = run(["pdftotext", str(work/"generated"/name), "-"])
                if original != regenerated:
                    raise AssertionError(f"Figure text differs: {name}")
            report["steps"].append({"name": "3 epochs, combination, fixed-cell ensemble, tables and figures",
                                    "status": "PASS"})
        report["elapsed_seconds"] = round(time.monotonic()-start, 2)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k,v in report.items() if k != "steps"}, indent=2))


if __name__ == "__main__":
    main()
