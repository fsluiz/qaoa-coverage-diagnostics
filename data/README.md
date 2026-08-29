# Input data

- `pef_planilha.xlsx`: 24 Special Border Platoon base records used by the frontier
  calculations. Confirm redistribution rights before public release.
- `arco_fogo_problem.json`: frozen 79-base Arco do Fogo set-cover instance.
- `overshoot_instance.json`: frozen 14-qubit, `m=3` instance used for cardinality
  descent.
- `benchmark_instance.json`: reduced PEF instance emitted by the base benchmark.
- `v7_strategic_results.json`: strategic selection consumed by the tactical notebook.
- `frozen_*.json`: QUBO/Ising instances used by the Kingston scripts.
- `ilp_ratio_benchmark.json`: archived regular-grid ILP sweep used by the
  symmetry-breaking analysis. Its original generator was not found.
- `ibge_amazonia_legal_2024.csv`: exact 773-municipality set derived from the
  official 2024 IBGE polygons and seats. It includes IBGE codes, names, state,
  polygon centroid, seat coordinates, and legal-Amazon area attributes.
- `ibge_amazonia_legal_2024_provenance.json`: source URLs, SHA-256 hashes, CRS,
  state counts, and centroid method for the derived municipality table.
