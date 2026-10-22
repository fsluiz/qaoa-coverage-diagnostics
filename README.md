# Classical simulability and hardware diagnostics for QAOA

Computational companion to the version-7 draft by F. S. Luiz, J. E. da Fonseca,
A. F. P. Santos, and M. C. de Oliveira. This snapshot is prepared for coauthor
review and a subsequent upload to the project repository. It is not a final
publication or a claim that a public release already exists.

## Current manuscript

- [Article source](paper/paper_v7.tex) and [PDF](paper/paper_v7.pdf).
- [Supplement source](paper/supplement_v7.tex) and [PDF](paper/supplement_v7.pdf).
- [Bibliography](paper/refs_v7.bib).
- [Version-7 computational workflow](paper/revision_v7/README.md).
- [Citation metadata](CITATION.cff).

The underlying scientific inputs and outputs are the ones checked after the
third correction round, `v7-round3-20261021`. The availability statement now
describes the planned GitHub deposit after coauthor review; no public release
identifier has been invented. The new file hashes are recorded in
`PACKAGE_MANIFEST.json`, and the working dependency manifest has been refreshed
for this snapshot. Historical frozen-source checks remain unchanged.

## Reproduce the v7 analyses

Use Python 3.11 and the pinned environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-reproduce-v7.txt
python3 paper/revision_v7/reproduce.py --quick
python3 paper/revision_v7/reproduce.py --report /tmp/v7-reproduction.json
```

These commands run offline and do not need IBM credentials or submit jobs.
They require `pdftotext` as well as the Python dependencies. The quick mode runs
39 tests and checks hashes, controls and generated artifacts. The second command
also repeats the archived bootstraps, campaign combination and fixed-cell
ensemble analysis. It does not rerun every historical variational optimization
or all historical ILPs. The package validation uses the existing audited Python
environment; it is not a test of installing a fresh environment.

## Compile the article and supplement

Install a TeX Live environment with REVTeX 4.2, BibTeX and the standard packages
used in the sources. From `paper/`, run:

```sh
pdflatex paper_v7
pdflatex supplement_v7
bibtex paper_v7
bibtex supplement_v7
pdflatex supplement_v7
pdflatex paper_v7
pdflatex supplement_v7
pdflatex paper_v7
```

Both documents must be compiled because they import labels from each other.

## Files and provenance

| Directory/file | Purpose |
|---|---|
| `paper/paper_v7.*`, `paper/supplement_v7.*`, `paper/refs_v7.bib` | Current draft and compiled documents |
| `paper/revision_v7/` | Revised analyses, uncertainty, tests, generated figures/tables and manifests |
| `paper/ibm_variance_campaign/` | Campaign scripts, plans and archived bitstring distributions |
| `data/` | Frozen instances and geographic/source inputs |
| `results/` | Raw counts, submitted circuits, numerical results and historical evidence |
| `scripts/` | Scientific generators, reanalyses and figure generators |
| `figures/` | Vector figures, including the figures used by the manuscript |
| `tests/` | Regression tests required by the v7 workflow |
| `requirements*.txt` | Pinned offline environment and optional workflows |
| `paper/BASELINE_V4.md`, v4/v6 sources and `paper/refs.bib` | Historical dependencies retained for regression checks |

The optional hardware requirements are needed only to retrieve or submit new
IBM experiments. Submission scripts are included for provenance, not run by
the offline reproduction. The editorial history in `paper/CORRECOES_V7.md` is
retained as a checked dependency; its references to internal review documents
are not required to execute the scientific workflow. Internal simulated referee
reports and coauthor correspondence are outside this upload package.

After coauthor changes, update the source files and regenerate affected results,
then intentionally refresh `paper/revision_v7/working_manifest.json` and rerun
the checks. The package manifest describes this snapshot and should not be
treated as the hash list for later edited versions.
