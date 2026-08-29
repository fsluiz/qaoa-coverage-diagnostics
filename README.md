# Quantum Optimization for Amazon Frontier Surveillance

This repository is the curated computational companion to:

> F. S. Luiz and M. C. de Oliveira, “Quantum Optimization for Amazon Frontier
> Surveillance: A Geographic Difficulty Ladder from Set Cover to Dominating Set.”

It was assembled from the computational working directory associated with the
article. The manuscript, bibliography, and PDF are maintained collaboratively on
Overleaf and are intentionally excluded from this Git repository.

Important: read [AUDIT.md](AUDIT.md) and
[OVERLEAF_CORRECTIONS.md](OVERLEAF_CORRECTIONS.md) before updating the article.
The August 2026 audit replaces the synthetic geographic fallback with the exact
IBGE 2024 municipality set and replaces sampled Rydberg values with deterministic
state-vector probabilities.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Standalone ILP, QWOA/QAOA, Rydberg, and IBM hardware programs |
| `data/` | Small input datasets and frozen benchmark instances |
| `results/` | Archived JSON outputs used for tables and claims |
| `figures/` | Paper figures in vector PDF and preview PNG formats |
| `notebooks/` | Tactical-layer notebook retained because no standalone generator exists |

## Paper-to-code map

| Paper component | Main program | Input | Archived output |
|---|---|---|---|
| Range and redundancy ladder (Table I) | `scripts/run_m_ladder.py` | `data/pef_planilha.xlsx` | `results/m_ladder_results.json` |
| ILP symmetry windows (Fig. 1) | `scripts/calc_symmetry_breaking.py` | `data/ilp_ratio_benchmark.json` | `results/ilp_symmetry_breaking.json` |
| Legal Amazon data preparation | `scripts/prepare_ibge_amazonia_legal.py` | official IBGE 2024 shapefiles | `data/ibge_amazonia_legal_2024.csv` |
| Legal Amazon MDS check | `scripts/calc_ilp_geographic.py` | official municipality table | `results/ilp_geographic_ibge_2024_centroids.json` |
| IBM Kingston frozen instances | `scripts/freeze_instances.py` | PEF, Arco, and overshoot inputs | `data/frozen_*.json` |
| IBM circuit build/submission | `scripts/submit_kingston.py` | frozen instances | `results/kingston_rerun_plan*.json` |
| IBM result retrieval | `scripts/analyze_kingston.py` | IBM job and frozen instances | `results/kingston_rerun_results.json` |
| Cross-scale IBM observable | `scripts/calc_unified_observable.py` | archived Kingston results | `results/unified_observable.json` |
| Rydberg 3×3 MDS (deterministic) | `scripts/rydberg_exact_sim.py` | internal 3×3 instance | `results/rydberg_3x3_exact.json` |
| Corrected Rydberg figure | `scripts/make_rydberg_figure.py` | exact Rydberg result | `figures/fig4_rydberg_results.*` |
| Cross-scale graph figure | `scripts/make_nzz_figure.py` | official geographic result | `figures/fig3_nzz_fidelity.*` |
| No-jump Rydberg decay check | `scripts/lindblad_3x3_sim.py` | internal 3×3 instance | `results/rydberg_decay_scan.json` |
| QWOA–QAOA base benchmark | `scripts/benchmark_qwoa_qaoa.py` | PEF spreadsheet | `results/benchmark_qwoa_qaoa_results.json` |
| Strong-optimizer cardinality descent | `scripts/run_compare_strong.py` | `data/overshoot_instance.json` | `results/compare_strong_results.json` |
| Eight-instance, equal-CX comparison | `scripts/run_path_b.py` | generated seeded instances | `results/path_b_results.json` |
| QWOA paper figures | `scripts/make_paper_figures.py` | archived benchmark results | two figure pairs in `figures/` |
| Dynamic tactical context | `notebooks/v7_tatico.ipynb` | notebook state/data | `results/v7_tatico_results.json` |

## Quick start

Use Python 3.10 or newer. The core environment is enough for the ILP ladder,
QWOA/QAOA statevector programs, the consistency audit, and figure generation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/verify_paper_claims.py
python3 scripts/run_m_ladder.py
python3 scripts/make_paper_figures.py
python3 scripts/rydberg_exact_sim.py
python3 scripts/make_rydberg_figure.py
```

Optional environments:

```bash
python3 -m pip install -r requirements-geo.txt
python3 -m pip install -r requirements-rydberg.txt
python3 -m pip install -r requirements-hardware.txt
```

The long QWOA/QAOA jobs are intentionally not part of a quick test. The archived
metadata records approximately 3.1 hours for `run_compare_strong.py` and 3.4 hours
for `run_path_b.py` on the machine originally used.

## Safe hardware workflow

`submit_kingston.py` performs a dry run by default. It submits work to IBM only
when explicitly called with `--submit`. Authentication is delegated to the local
Qiskit Runtime configuration; no credential is stored in this repository.

```bash
python3 scripts/submit_kingston.py
python3 scripts/submit_kingston.py --submit --backend ibm_kingston --shots 8192
python3 scripts/analyze_kingston.py IBM_JOB_ID
```

Backend availability, calibration, and Qiskit APIs change over time, so archived
hardware JSON files are the evidence for the paper snapshot.

## Official geographic instance

The compact, versioned CSV contains 773 municipality codes, polygon centroids,
and municipal seats from the IBGE Amazônia Legal 2024 product. To redownload and
verify the raw archives, then reproduce both coordinate variants:

```bash
python3 scripts/prepare_ibge_amazonia_legal.py
python3 scripts/calc_ilp_geographic.py
python3 scripts/calc_ilp_geographic.py --coordinate-type seat
```

The 56 MB raw polygon archive is hash-verified but not committed.

## Public-release checklist

- Apply the numerical replacements in `OVERLEAF_CORRECTIONS.md` on Overleaf.
- Choose and add an explicit software/data license.
- Confirm that the PEF spreadsheet may be redistributed publicly.
- Add the final DOI, journal metadata, and author identifiers when available.
- Create the GitHub remote named in the paper and push the reviewed history.
