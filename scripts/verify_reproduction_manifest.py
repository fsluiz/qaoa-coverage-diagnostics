#!/usr/bin/env python3
"""Validate a versioned dependency, input, job and active-artifact manifest.

    python verify_reproduction_manifest.py                      # current (version 4)
    python verify_reproduction_manifest.py --manifest reproducibility/manifest_v3.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_parser = argparse.ArgumentParser()
_parser.add_argument("--manifest", default="reproducibility/manifest_v4.json")
MANIFEST_PATH = ROOT / _parser.parse_args().manifest
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
# Older manifests predate these two fields; they described the version-3 tree.
TEX_SOURCES = manifest.get("tex_sources", ["paper/paper_v3.tex", "paper/supplement_v3.tex"])
ORCHESTRATOR = manifest.get("orchestrator", "scripts/reproduce_v3.py")
checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    checks.append((name, bool(condition), detail))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


check(
    "Python version matches the audited lock",
    platform.python_version() == manifest["python"]["version"],
    f"found {platform.python_version()}",
)
for distribution, expected in manifest["packages"].items():
    try:
        observed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        observed = "missing"
    check(
        f"dependency {distribution} is pinned at {expected}",
        observed == expected,
        f"found {observed}",
    )

lock = (ROOT / manifest["python"]["lock_file"]).read_text(encoding="utf-8")
for distribution, version in manifest["packages"].items():
    check(
        f"lock file declares {distribution}=={version}",
        f"{distribution}=={version}" in lock,
    )

for relative, expected in manifest["immutable_inputs_sha256"].items():
    path = ROOT / relative
    observed = sha256(path) if path.is_file() else "missing"
    check(f"immutable hash: {relative}", observed == expected, observed)

for job_id, relative in manifest["ibm_jobs"].items():
    path = ROOT / relative
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    check(
        f"IBM archive {relative} carries immutable job {job_id}",
        payload.get("job_id") == job_id,
        str(payload.get("job_id")),
    )

tex = "\n".join(
    (ROOT / relative).read_text(encoding="utf-8")
    for relative in TEX_SOURCES
)
included = {
    f"figures/{match}"
    for match in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+\.pdf)\}", tex)
}
expected_figures = set(manifest["active_figures"])
present_figures = {
    str(path.relative_to(ROOT)) for path in (ROOT / "figures").glob("*.pdf")
}
check(f"TeX includes exactly the {len(expected_figures)} active figures",
      included == expected_figures,
      f"found {sorted(included)}")
check("figures/ contains no unused PDF", present_figures == expected_figures,
      f"found {sorted(present_figures)}")
check("legacy QWOA figures are absent from active TeX", all(
    name not in tex for name in ("fig_advantage_vs_m.pdf", "fig_qwoa_vs_qaoa.pdf")
))

orchestrator = (ROOT / ORCHESTRATOR).read_text(encoding="utf-8")
check("active workflow never invokes the legacy figure generator",
      "make_paper_figures.py" not in orchestrator)
legacy_only = (
    "data/benchmark_instance.json",
    "results/benchmark_qwoa_qaoa_results.json",
    "results/compare_strong_results.json",
    "results/path_b_results.json",
    "scripts/run_compare_strong.py",
    "scripts/run_path_b.py",
    "scripts/make_paper_figures.py",
)
check("legacy-only QWOA artifacts are absent from the active snapshot", all(
    not (ROOT / relative).exists() for relative in legacy_only
))
check("active workflow has no sibling-archive dependency",
      "quantum-amazon-surveillance-legacy" not in orchestrator)
check("adaptive p=1 audit has a dedicated generator",
      (ROOT / "scripts" / "run_qwoa_p1_high_resolution.py").is_file()
      and "run_qwoa_p1_high_resolution.py" in orchestrator)
check("claim matrix is present", (ROOT / "CLAIM_MATRIX.md").is_file())
check("latexmk is installed", shutil.which("latexmk") is not None)
check("pdfinfo is installed", shutil.which("pdfinfo") is not None)

for relative in manifest["active_pdfs"] + manifest["active_figures"]:
    check(f"active artifact exists: {relative}", (ROOT / relative).is_file())

for name, passed, detail in checks:
    suffix = f" ({detail})" if detail else ""
    print(f"{'PASS' if passed else 'FAIL'}  {name}{suffix}")
failed = [name for name, passed, _ in checks if not passed]
print(f"\n{len(checks) - len(failed)}/{len(checks)} manifest checks passed")
if failed:
    sys.exit(1)
