#!/usr/bin/env python3
"""One-command offline reproduction of the audited version-3 manuscript.

Superseded by reproduce_v4.py; kept so that the version-3 snapshot remains auditable.
It uses the frozen version-3 claim verifiers (``*_v3.py``).

The default mode recomputes every deterministic/classical result, every
version-3 quantum-search audit, the four active figures, both PDFs, and both
claim verifiers.  ``--quick`` preserves the expensive archived search
endpoints but recomputes their interpretations and all other evidence.

IBM jobs are never contacted.  Their raw bitstrings, plans and job-time
calibration record are immutable inputs whose identifiers and SHA-256 hashes
are checked by ``verify_reproduction_manifest.py``.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
REPORT = RESULTS / "reproduction_report.json"


def py(script: str, *arguments: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script), *arguments]


CORE_ANALYSES = [
    py("verify_arco_provenance.py"),
    py("freeze_instances.py"),
    py("run_m_ladder.py"),
    py("calc_triviality.py"),
    py("calc_ilp_geographic.py", "--coordinate-type", "centroid"),
    py("calc_continental_blocks.py"),
    py("calc_ilp_symmetry.py"),
    py("calc_block_decomposition.py"),
    py("calc_block_factorisation.py"),
    py("calc_ladder_rungs.py"),
    py("calc_qwoa_simulability.py"),
    py("calc_dimension_sweep.py"),
    py("calc_coverage_preserving_2d.py"),
    py("calc_frozen_target_sweep.py"),
    py("calc_ensemble_squeeze.py"),
    py("calc_exact_encoding.py"),
    py("calc_protection_2d.py"),
    py("calc_graceful_degradation.py"),
    py("calc_tactical_replanning.py"),
    py("calc_persistent_surveillance.py"),
]

HARDWARE_REANALYSES = [
    py("analyze_kingston.py", "--counts"),
    py(
        "analyze_kingston.py",
        "--plan", "results/kingston_rerun_plan_arco.json",
        "--counts", "results/kingston_100k_arco_optangles_counts.json",
        "--out", "results/kingston_100k_arco_optangles.json",
    ),
    py(
        "analyze_kingston.py",
        "--plan", "results/kingston_transferred_arco_plan.json",
        "--counts", "results/kingston_transferred_arco_counts.json",
        "--out", "results/kingston_transferred_arco.json",
    ),
    py(
        "analyze_exact51.py",
        "--counts", "results/exact51_counts.json",
        "--out", "results/exact51_results.json",
    ),
]

SCIENTIFIC_AUDITS = [
    py("calc_statistical_audit.py"),
    py("calc_rydberg_encoding_audit.py"),
    py("calc_factorisation_scope.py"),
]

QUANTUM_SEARCHES = [
    py("run_qwoa_expanded.py", "--seeds", "0:24", "--no-resume"),
    py("check_qwoa_optimizer_stability.py"),
    py("run_qwoa_convergence.py", "--no-resume"),
    py("run_qwoa_p1_high_resolution.py"),
    py("analyze_qwoa_convergence.py"),
    py("check_qwoa_gradients.py"),
    py("run_same_sector_mixer_control.py", "--no-resume"),
    py("run_mixer_scale_sensitivity.py", "--no-resume"),
    py("analyze_same_sector_mixer_control.py"),
    py("check_same_sector_mixer_gradients.py"),
]

QUANTUM_POSTPROCESS = [
    py("analyze_qwoa_convergence.py"),
    py("check_qwoa_gradients.py"),
    py("analyze_same_sector_mixer_control.py"),
    py("check_same_sector_mixer_gradients.py"),
]

FIGURES = [
    py("make_pef_map.py"),
    py("make_ilp_figure.py"),
    py("make_nzz_figure.py"),
    py("make_squeeze_figure.py"),
]

VERIFIERS = [
    py("verify_paper_claims_v3.py"),
    py("verify_research_audit_v3.py"),
]


def display(command: list[str]) -> str:
    rendered = []
    for token in command:
        try:
            rendered.append(str(Path(token).relative_to(ROOT)))
        except (ValueError, TypeError):
            rendered.append(token)
    return " ".join(rendered)


def run(command: list[str], env: dict[str, str], records: list[dict], cwd=ROOT) -> None:
    label = display(command)
    print(f"\n>>> {label}", flush=True)
    started = time.time()
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    duration = time.time() - started
    records.append(
        {"command": label, "returncode": completed.returncode,
         "duration_s": round(duration, 3)}
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def check_latex_logs() -> None:
    forbidden = (
        "undefined references",
        "multiply defined",
        "rerun to get cross-references right",
        "overfull \\hbox",
        "overfull \\vbox",
    )
    failures = []
    for name in ("supplement_v3.log", "paper_v3.log"):
        text = (PAPER / name).read_text(encoding="utf-8", errors="replace").lower()
        for fragment in forbidden:
            if fragment in text:
                failures.append(f"{name}: {fragment}")
    if failures:
        raise RuntimeError("LaTeX quality gate failed: " + "; ".join(failures))


def build_pdfs(env: dict[str, str], records: list[dict]) -> dict[str, int]:
    if shutil.which("latexmk") is None or shutil.which("pdfinfo") is None:
        raise RuntimeError("latexmk and pdfinfo are required to build and inspect the PDFs")
    sequence = (
        ["latexmk", "-g", "-pdf", "-interaction=nonstopmode", "supplement_v3.tex"],
        ["latexmk", "-g", "-pdf", "-interaction=nonstopmode", "paper_v3.tex"],
        ["latexmk", "-g", "-pdf", "-interaction=nonstopmode", "supplement_v3.tex"],
        ["latexmk", "-g", "-pdf", "-interaction=nonstopmode", "paper_v3.tex"],
    )
    for command in sequence:
        run(command, env, records, cwd=PAPER)
    check_latex_logs()
    pages = {}
    for name in ("paper_v3.pdf", "supplement_v3.pdf"):
        output = subprocess.check_output(
            ["pdfinfo", name], cwd=PAPER, env=env, text=True
        )
        page_line = next(line for line in output.splitlines() if line.startswith("Pages:"))
        pages[name] = int(page_line.split(":", 1)[1].strip())
    for stem in ("paper_v3.tex", "supplement_v3.tex"):
        run(["latexmk", "-c", stem], env, records, cwd=PAPER)
    return pages


def environment_record() -> dict:
    packages = {}
    for name in (
        "numpy", "pandas", "scipy", "matplotlib", "openpyxl",
        "geopandas", "Shapely", "pyproj",
    ):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def write_report(mode: str, status: str, started: float, records: list[dict],
                 pages: dict[str, int] | None, error: str | None = None,
                 resume: dict | None = None) -> None:
    payload = {
        "schema_version": 1,
        "workflow": "audited version-3 offline reproduction",
        "mode": mode,
        "status": status,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - started, 3),
        "environment": environment_record(),
        "commands": records,
        "pdf_pages": pages,
        "error": error,
        "ibm_network_access": False,
        "resume": resume,
        "cumulative_command_duration_s": round(
            sum(record["duration_s"] for record in records), 3
        ),
    }
    REPORT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick", action="store_true",
        help="reuse expensive archived quantum-search endpoints",
    )
    mode.add_argument(
        "--checks-only", action="store_true",
        help="run manifest and claim checks without regenerating artifacts",
    )
    mode.add_argument(
        "--resume-full", action="store_true",
        help="resume the successful prefix recorded by a failed full run",
    )
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()
    selected_mode = "checks-only" if args.checks_only else ("quick" if args.quick else "full")
    started = time.time()
    records: list[dict] = []
    pages = None
    resume_context = None

    searches = QUANTUM_POSTPROCESS if args.quick else QUANTUM_SEARCHES
    planned = [py("verify_reproduction_manifest.py", "--manifest", "reproducibility/manifest_v3.json")]
    if not args.checks_only:
        for group in (CORE_ANALYSES, HARDWARE_REANALYSES, SCIENTIFIC_AUDITS):
            planned.extend(group)
        planned.extend(searches)
        planned.extend(FIGURES)
    planned.extend(VERIFIERS)

    start_at = 0
    if args.resume_full:
        if not REPORT.exists():
            parser.error("--resume-full requires an existing failed full report")
        previous = json.loads(REPORT.read_text(encoding="utf-8"))
        if previous.get("mode") != "full" or previous.get("status") != "failed":
            parser.error("--resume-full requires status=failed and mode=full")
        for index, record in enumerate(previous.get("commands", [])):
            if index >= len(planned) or record.get("returncode") != 0:
                break
            expected = display(planned[index])
            if record.get("command") != expected:
                parser.error(
                    "recorded command prefix differs from the current full workflow"
                )
            records.append(record)
            start_at += 1
        if start_at == 0:
            parser.error("failed report has no successful command prefix to resume")
        resume_context = {
            "used": True,
            "previous_started_utc": previous.get("started_utc"),
            "previous_finished_utc": previous.get("finished_utc"),
            "previous_duration_s": previous.get("duration_s"),
            "successful_prefix_commands": start_at,
            "previous_error": previous.get("error"),
        }

    with tempfile.TemporaryDirectory(prefix="qas-v3-") as temporary:
        env = dict(os.environ)
        env.update(
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MPLCONFIGDIR": str(Path(temporary) / "matplotlib"),
                "PYTHONHASHSEED": "0",
            }
        )
        try:
            if args.resume_full:
                preflight: list[dict] = []
                run(py("verify_reproduction_manifest.py", "--manifest", "reproducibility/manifest_v3.json"), env, preflight)
                preflight[0]["resume_preflight"] = True
                records.extend(preflight)
            for command in planned[start_at:]:
                run(command, env, records)
            if not args.skip_pdf and not args.checks_only:
                pages = build_pdfs(env, records)
            write_report(
                selected_mode, "passed", started, records, pages,
                resume=resume_context,
            )
        except Exception as exc:
            write_report(
                selected_mode, "failed", started, records, pages, str(exc),
                resume_context,
            )
            raise

    print(
        f"\nReproduction {selected_mode} passed in {time.time() - started:.1f}s; "
        f"report: {REPORT.relative_to(ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
