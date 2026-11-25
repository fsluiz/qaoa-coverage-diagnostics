# QAOA coverage diagnostics

Code and reproducibility data accompanying **Classical simulability and hardware
diagnostics for QAOA on geographic coverage benchmarks**, version 8.1.0.

Authors: F. S. Luiz, D. Abreu, A. J. G. Abelém, J. E. da Fonseca,
A. F. P. Santos and M. C. de Oliveira.

Article: https://arxiv.org/abs/2611.27507

## Reproduce the diagnostics

```sh
python3 -m pip install -r requirements-reproduce-v7.txt
python3 scripts/reproduce_diagnostics.py --output /tmp/qaoa-reproduced --full
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the figure/input map, commands,
numerical tolerances and the scope of reproduction. The completed release check
is recorded in [VERIFICATION.json](VERIFICATION.json).

- `data/`: frozen instances and attributed geographic inputs.
- `results/`: raw counts, plans, physical layouts, native circuits and supporting calculations.
- `analysis/ibm_variance_campaign/results/`: the three fixed-circuit jobs.
- `analysis/revision_v7/`: exact references, conditional counting models and block diagnostics.
- `analysis/revision_v8/`: physical provenance audit, complete job-associated calibration archives and independent readout control.
- `DATA_MANIFEST.json`: SHA-256 checksums, sizes, roles and original paths.
- `PATH_MAPPING.json`: mapping of historical analysis paths to this public layout.

## Citation and archive

Release: https://github.com/fsluiz/qaoa-coverage-diagnostics/releases/tag/v8.1.0
Version 8.1.0 adds the reproducibility data and is archived at
**https://doi.org/10.5281/zenodo.24540635**.
The Zenodo ZIP was verified byte for byte against release commit
`c44a7e2bd330c07867ab2b99fd316b55c4f491f7`: 277 files, including 148 data/reference files.
The earlier code-only version 8.0.0 remains available at
https://doi.org/10.5281/zenodo.24539594.
See `CITATION.cff` and `.zenodo.json` for author and release metadata.

## License

MIT; see LICENSE. Third-party geographic inputs retain their source attribution
and applicable terms, documented in `data/README.md` and source-provenance JSON.
The manuscript and its supplemental text are distributed separately on arXiv.
