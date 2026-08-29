# Corrections to apply on Overleaf

This repository does not track the manuscript. The following replacements are
based on the official IBGE 2024 data and deterministic Rydberg state-vector
calculations regenerated on August 29, 2026.

## Legal Amazon municipality instance

Replace every occurrence of **761 municipalities** with **773 municipalities**.
The official state counts are AC 22, AM 62, AP 16, MA 181, MT 142, PA 144,
RO 52, RR 15, and TO 139. The current total includes Boa Esperança do Norte
(IBGE code 5101837), created in Mato Grosso.

Also replace the abstract's **5.2 million km2** with the current official area
**5,014,107.875 km2** (about **5.014 million km2**). The rounded sum of the
municipality-level `AREA_INT` attributes in the downloaded file is
5,014,108.324 km2; the 0.449 km2 difference is caused by per-row rounding, so the
product-page total should be cited in prose.

The primary calculation uses the geometric centroids of the official municipality
polygons after projection to SIRGAS 2000 / Brazil Polyconic (EPSG:5880). At a
200 km range, replace the geographic-instance values with:

| Quantity | Superseded | Corrected (centroids) |
|---|---:|---:|
| municipalities / graph nodes | 761 | 773 |
| optimum dominating-set size | 47 | 47 |
| selected fraction | 6.2% | 6.08% (round to 6.1%) |
| QUBO `ZZ` pairs | 38,329 | 39,974 |
| average open-neighbour degree | not reported | 41.29 |

Thus, the larger official set raises `n_ZZ` by 4.29% but does not change the
optimal cardinality at 200 km. The conclusion that the full geographic instance
is beyond near-term gate-model hardware remains unchanged.

Update the Rung-3/Table-II/Figure-3 labels from `761q` to `773q`, and use
approximately `40,000` rather than `38,000` interactions when rounding. The old
statement that all geographic ILPs solved in at most 44.5 s refers to the
superseded instance. The corrected centroid sweep was optimal at every tested
radius with no timeout. Because wall times vary across reruns and machines,
report the solver, hardware, and a controlled benchmark if a time bound is retained.

As a sensitivity check, using the official municipal seats instead of polygon
centroids gives `k*=44`, `n_ZZ=39,631`, and average degree 43.13 at 200 km.
Therefore, the text must explicitly say **municipality polygon centroids**; the
value 47 is not invariant to the choice of representative point.

Suggested methods sentence:

> We use the 773 municipalities in the IBGE Amazônia Legal 2024 product. Each
> graph node is placed at the geometric centroid of the official municipal
> polygon, computed in SIRGAS 2000 / Brazil Polyconic (EPSG:5880), and edges join
> centroids within the stated great-circle range.

Official sources:

- https://www.ibge.gov.br/geociencias/organizacao-do-territorio/estrutura-territorial/15819-amazonia-legal.html
- https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/amazonia_legal/2024/

## Rydberg 3x3 instance

The old values came from a finite bitstring sample and should not be reported as
exact emulator probabilities. The corrected program evaluates observables
directly from the final 512-dimensional state vector. At `alpha=0.5` and
`T=2.6 us`, replace:

Use `C6 = 2 pi x 862,690 MHz um^6`, or equivalently
`C6 = 5,420,441.13 rad us^-1 um^6` in the angular-frequency Hamiltonian. The
former program used 862,690 directly in angular units and therefore omitted a
factor of `2 pi`.

| Quantity | Superseded | Corrected |
|---|---:|---:|
| `P(opt)` | 15.75% | 14.7998% (round to 14.80%) |
| mean excitation `k_mean` | 1.95 | 1.76464 (round to 1.765) |
| uniform random baseline | 1.88% | `10/512 = 1.953125%` |
| concentration over baseline | 8.4x | 7.57750x (round to 7.58x) |
| scan-optimal duration | 2.6 us | 3.0 us |
| `P(opt)` at scan-optimal duration | not consistently reported | 36.3326% |

Replace the complete sampled Table III with the deterministic alpha scan below
(`T=2.6 us`; all ratios use the exact 1.953125% baseline):

| `alpha` | `P(opt)` | mean excitation | `P/P_random` |
|---:|---:|---:|---:|
| 0.30 | 8.8164% | 1.75518 | 4.51398x |
| 0.40 | 1.7213% | 1.93649 | 0.88131x |
| 0.50 | 14.7998% | 1.76464 | 7.57750x |
| 0.60 | 6.1210% | 1.69498 | 3.13395x |
| 0.70 | 8.9037% | 1.94793 | 4.55868x |

The scanned optimal local-detuning scale remains `alpha*=0.5`. The qualitative
conclusion also remains: at the fixed 2.6 us operating point, the optimal-state
probability is 7.58 times the uniform random baseline. The magnitude is slightly
weaker than previously stated, but still substantial.

The discussion of duration must change materially. `T=2.6 us` may be retained as
a **fixed comparison point**, but it cannot be labelled `T*` or the time-scan
optimum. Of the tested durations, `T*=3.0 us` yields `P(opt)=36.33%`.

For `Gamma=0.01 us^-1`, the no-jump non-Hermitian check gives
`P(opt)=14.1336%`, a 4.50% relative reduction from the noiseless 2.6 us value.
This is only a no-jump approximation, not a complete Lindblad master-equation
prediction; qualify any hardware-noise claim accordingly.

Suggested discussion replacement:

> At the fixed operating point alpha=0.5 and T=2.6 us, direct evaluation of the
> final state vector gives P(opt)=14.80%, 7.58 times the exact uniform baseline
> of 10/512=1.953%. The alpha optimum is unchanged. A separate duration scan,
> however, peaks at T=3.0 us with P(opt)=36.33%, so 2.6 us is a reference point
> rather than the scan-optimal duration.

The regenerated figure is `figures/fig4_rydberg_results.pdf`.

Reference specifications:

- https://www.quera.com/aquila
- https://bloqade.quera.com/v0.32.0/reference/bloqade-analog/src/bloqade/analog/emulate/ir/state_vector/
