"""Mutable working-draft dependency manifest; not a publication/release record."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MANIFEST = HERE / "working_manifest.json"
STEMS = ("e01_20261008", "e02_20261014", "e03_20261014")
BASELINES = {
    "paper/paper_v6.tex": "d7c3c623bd8d6bdc19e46567aa06b10ba5d469205f2d21c97e78c2f0abd41d36",
    "paper/supplement_v6.tex": "240e2ad167f8d54ec5a25dcc0b1e087d40d8198aba96f93a03b727019749736f",
    "paper/refs.bib": "d3c3c022ca07c4ad29df3c28668d0196a76e290024900ea09655513977d1c0de",
    "analysis/ibm_variance_campaign/analyze_variance_epoch.py": "a5d36524763d1aa5f0afe03a67191c7d8ef8abf38dadf32f517c8ee503a1445d",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependencies():
    files = {PROJECT / name for name in BASELINES}
    for name in ("paper_v7.tex", "supplement_v7.tex", "refs_v7.bib",
                 "paper_v7.pdf", "supplement_v7.pdf", "CORRECOES_V7.md"):
        files.add(HERE.parent / name)
    files.update(HERE.glob("*.py"))
    files.add(HERE / "README.md")
    files.update((HERE / "results").glob("*.json"))
    files.update((HERE / "generated").iterdir())
    for name in (
        "data/frozen_rung1.json", "data/frozen_arco.json",
        "results/exact51_plan.json", "results/hardware_decomposition.json",
        "results/exact51_counts.json", "results/exact51_matched_counts.json",
        "results/exact51_matched_plan.json",
        "paper/BASELINE_V4.md", "paper/paper_v4.tex", "paper/supplement_v4.tex",
        "requirements-reproduce-v3.txt", "requirements-reproduce-v7.txt",
        "results/joint_separate_batch_plan.json", "results/block_decomposition.json",
        "results/ensemble_squeeze.json", "figures/fig_blocks.pdf",
        "figures/fig_exact51_routing.pdf", "tests/test_v5_corrections.py",
        "scripts/calc_ensemble_squeeze.py",
        "results/qwoa_20seeds.json", "data/pef_planilha.xlsx",
        "results/dimension_sweep.json", "results/coverage_preserving_2d.json",
        "scripts/calc_dimension_sweep.py", "scripts/calc_coverage_preserving_2d.py",
        "scripts/make_block_figure.py", "scripts/make_exact51_layout_figure.py",
        "data/legal_amazon_outline_2024.geojson", "data/ibge_amazonia_legal_2024.csv",
        "results/ilp_geographic_ibge_2024_centroids.json", "results/rung1_physical_layout.json",
        "results/block_product_null.json", "results/joint_separate_batch.json",
        "results/ibm_block_results/kingston_blocks_analysis.json",
        "results/exact51_physical_layout.json", "results/exact51_cross_block.json",
        "results/ibm_circuits/exact51_defaults.ops.json",
        "results/ibm_circuits/exact51_matched.ops.json",
    ):
        files.add(PROJECT / name)
    for stem in STEMS:
        archive = HERE.parent / "ibm_variance_campaign/results"
        for suffix in ("plan", "counts", "analysis_v2"):
            files.add(archive / f"{stem}_{suffix}.json")
        plan = json.loads((archive / f"{stem}_plan.json").read_text())
        for pub in plan["pubs"]:
            qpy = PROJECT / pub["qpy"]
            if digest(qpy) != pub["qpy_sha256"]:
                raise AssertionError(f"Submitted QPY hash mismatch: {qpy}")
            files.add(qpy)
    # Track the transitive closure of imported local scientific helpers.
    visited, pending = set(), [p for p in files if p.suffix == ".py"]
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                     else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                for base in (HERE, PROJECT / "scripts"):
                    candidate = base / (name.split(".")[0] + ".py")
                    if candidate.is_file() and candidate not in files:
                        files.add(candidate)
                        pending.append(candidate)
    return files


def verify():
    content = json.loads(MANIFEST.read_text())
    for name, expected in content["sha256"].items():
        if digest(PROJECT / name) != expected:
            raise AssertionError(f"Working dependency changed: {name}")
    for name, expected in BASELINES.items():
        if digest(PROJECT / name) != expected:
            raise AssertionError(f"Historical baseline changed: {name}")
    return len(content["sha256"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Refresh after intentional draft edits.")
    args = parser.parse_args()
    if args.write:
        for name, expected in BASELINES.items():
            if digest(PROJECT / name) != expected:
                raise AssertionError(f"Historical baseline changed: {name}")
        output = dict(
            status="mutable internal author draft; not an immutable release or DOI record",
            scope="v7 revisions and their direct/transitive helper dependencies; not all legacy analyses",
            python=platform.python_version(),
            packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib", "pandas", "openpyxl")},
            baseline_sha256=BASELINES,
            sha256={str(p.relative_to(PROJECT)): digest(p) for p in sorted(dependencies())},
        )
        MANIFEST.write_text(json.dumps(output, indent=2) + "\n")
    print(f"PASS: {verify()} working-draft dependencies and unchanged v6 sources.")


if __name__ == "__main__":
    main()
