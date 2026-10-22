# Version-5 input data

- `pef_planilha.xlsx`: 24 Special Border Platoon records used by the frontier
  calculations and map.
- `arco_fogo_problem.json`: frozen 79-base Arco do Fogo set-cover input.
- `overshoot_instance.json`: frozen 14-qubit, m=3 cardinality-descent input.
- `frozen_rung1.json`, `frozen_arco.json`, and `frozen_overshoot.json`: complete
  QUBO/Ising instances consumed by the hardware and exact-analysis scripts.
- `ilp_ratio_benchmark.json`: regular-grid ILP timing input used by the corrected
  symmetry analysis.
- `ibge_amazonia_legal_2024.csv`: exact 773-municipality table derived from the
  official 2024 IBGE products. It contains municipality codes, names, state, polygon
  centroids, municipal-seat coordinates, and Legal Amazon attributes. Version 3 uses
  polygon centroids for the continental MDS calculation.
- `ibge_amazonia_legal_2024_provenance.json`: source URLs, SHA-256 hashes, CRS,
  state counts, and centroid construction method.

The frozen Arco and overshoot inputs are part of the version-5 specification. Their
outputs can be reproduced from these files even though the earlier exploratory
search that selected them is outside the reproducibility snapshot.

The superseded reduced QWOA benchmark is preserved in the sibling legacy
archive and is not an input to the current manuscript.
