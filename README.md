# QAOA coverage diagnostics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24539594.svg)](https://doi.org/10.5281/zenodo.24539594)

Code accompanying **Classical simulability and hardware diagnostics for QAOA
on geographic coverage benchmarks**. Version 8.0.0.

Authors: F. S. Luiz, D. Abreu, A. J. G. Abelém, J. E. da Fonseca,
A. F. P. Santos, and M. C. de Oliveira.

This repository and its Zenodo software archive contain code only.
The article and supplemental manuscript are excluded from this software deposit.
The associated preprint is [arXiv:2611.27507](https://arxiv.org/abs/2611.27507).

## Contents

- `scripts/`: instance construction, exact references, optimization controls,
  hardware reanalysis and figure generators.
- `analysis/revision_v7/`: counting uncertainty, structural references,
  covariance decomposition, controls and scientific tests.
- `analysis/revision_v8/`: historical calibration retrieval, physical-block
  resource summaries and independent verification.
- `analysis/ibm_variance_campaign/`: fixed-circuit campaign scripts.
- `tests/`: historical scientific regression checks.
- `requirements*.txt`: analysis and optional hardware dependencies.

## Inputs and execution

Raw counts, frozen instances, QPY circuits, calibration responses, generated
tables and figures are not bundled in this code-only release. They must be
provided separately to run archived-input workflows. Those scripts use
repository-relative `data/` and `results/` paths and sibling campaign archive
paths under `analysis/`; historical regression/manifest checks also refer to
the original manuscript snapshot. This release does not claim self-contained
reproduction without those external inputs.

The code was adapted from the locally checked manuscript companion. Validation
of this distribution includes Python syntax and input-independent mathematical
tests; full archived-data reproduction is a separate workflow. IBM retrieval
scripts use the configured account. Hardware submission scripts must be invoked
explicitly; importing the code does not submit jobs.

## Publication and citation

Software repository: https://github.com/fsluiz/qaoa-coverage-diagnostics
Software DOI: https://doi.org/10.5281/zenodo.24539594
Article: [https://arxiv.org/abs/2611.27507](https://arxiv.org/abs/2611.27507) (preprint).

See `CITATION.cff` and `.zenodo.json` for the six-author metadata.

## License

MIT; see LICENSE.
