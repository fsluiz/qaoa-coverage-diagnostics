# Reproducibility and provenance audit

Audit updated August 29, 2026. The manuscript itself is maintained on Overleaf
and is intentionally absent from this repository.

## Resolved critical findings

### Official Legal Amazon instance

The former geographic script silently generated 761 synthetic coordinates when
no shapefile was present. That fallback and its archived output have been removed.

The replacement pipeline downloads the official IBGE Amazônia Legal 2024 polygon
and seat products, verifies fixed SHA-256 hashes, enforces 773 unique municipality
codes and the official counts in all nine states, and writes a compact derived CSV.
Centroids are computed in EPSG:5880 before conversion to longitude/latitude.

At 200 km, the corrected centroid graph has `k_opt=47`, `n_ZZ=39,974`, and
average open degree 41.29. The seat-coordinate sensitivity graph has `k_opt=44`
and `n_ZZ=39,631`. See `OVERLEAF_CORRECTIONS.md` for manuscript changes.

### Deterministic Rydberg values

The former headline numbers were based on a finite, unseeded Bloqade sample. The
canonical result now comes from direct final-state probabilities produced by the
independent SciPy state-vector integrator in `scripts/rydberg_exact_sim.py`.

At `alpha=0.5` and `T=2.6 us`, the corrected values are `P(opt)=14.7998%`,
`k_mean=1.76464`, exact uniform baseline `1.953125%`, and concentration `7.5775x`.
The scanned alpha optimum remains 0.5, but the scanned duration optimum is 3.0 us
with `P(opt)=36.3326%`. The hardware specification is
`C6 = 2 pi x 862,690 MHz um^6`, equivalent to approximately 5,420,441.13
`rad us^-1 um^6`. The superseded simulation omitted the `2 pi` conversion when
using an angular-frequency Hamiltonian.

The decay result is explicitly labelled a no-jump non-Hermitian check. It must not
be presented as a full Lindblad or hardware prediction.

## Remaining limitations

1. The tactical sentence comparing single and double cover uses different range
   thresholds: `m=1` at 200 km and `m=2` at 500 km. State both radii inline.
2. Table I uses the canonical task seed 0 (`n_ZZ=17` at 200 km), while the
   Kingston hardware row uses a different 24-base instance (`n_ZZ=21`). The
   distinct instances and seeds need explicit labels.
3. Full Scale-2 Kingston counts were not retained. As documented in
   `results/unified_observable.json`, `P_hw` cannot be reconstructed for the
   79-qubit experiment; only full-coverage probability and normalized energy
   reduction are available.
4. Generator code for some early article figures was not identified. The Rydberg
   figure now has a deterministic generator, as do both QWOA figures.
5. The tactical JSON is traceable only to `notebooks/v7_tatico.ipynb`; converting
   the final calculation to a standalone, seeded script would improve provenance.
6. IBM job identifiers are present in archived JSON. No token, password, private
   key, or saved IBM credential was found in the selected files.
7. No software/data license has been selected. The authors must choose reuse terms
   and confirm that the PEF spreadsheet may be redistributed.

## Repository selection

Included files are those needed to inspect the computational claims, plus frozen
inputs and archived outputs. The manuscript/PDF, old drafts, LaTeX products, early
notebooks, duplicate directories, caches, animations, unrelated images, and
downloaded reference PDFs are excluded. Git history retains previous calculations,
but current documentation points only to corrected geographic and Rydberg results.
