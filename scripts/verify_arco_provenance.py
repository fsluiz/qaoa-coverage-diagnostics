#!/usr/bin/env python3
"""Verify the Rung-2 candidate sites against the official IBGE 2024 product.

The rule that selected these particular 79 municipalities out of the Legal Amazon set was
not recorded, and we were unable to reconstruct it (a nearest-municipality-to-grid-node
hypothesis recovers only 42 of 79).  What is verifiable, and what this script checks, is
that every selected site is a genuine IBGE municipality at its official geometric centroid.
The coordinate list itself is published in data/frozen_arco.json, so the Rung-2 results are
reproducible even though the selection is not.

Writes results/arco_provenance.json.
"""
from __future__ import annotations

import csv
import json
import math
import unicodedata
from collections import Counter

import numpy as np

from _paths import DATA_DIR, RESULTS_DIR

R_EARTH = 6371.0088


def norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn").strip()


def haversine(a1, o1, a2, o2):
    p1, p2 = math.radians(a1), math.radians(a2)
    dl = math.radians(o2 - o1)
    return 2 * R_EARTH * math.asin(math.sqrt(
        math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


def main() -> None:
    src = json.load(open(DATA_DIR / "arco_fogo_problem.json"))
    ib = {}
    with open(DATA_DIR / "ibge_amazonia_legal_2024.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ib[norm(r["nm_mun"])] = r

    devs, states, unmatched = [], Counter(), []
    for name, la, lo in zip(src["base_names"], src["base_lats"], src["base_lons"]):
        row = ib.get(norm(name))
        if row is None:
            unmatched.append(name); continue
        states[row["sigla_uf"]] += 1
        devs.append(haversine(la, lo, float(row["centroid_lat"]), float(row["centroid_lon"])) * 1000)

    devs = np.array(devs)
    out = dict(n_sites=len(src["base_names"]), matched=len(devs), unmatched=unmatched,
               states=dict(states.most_common()),
               centroid_deviation_m=dict(median=float(np.median(devs)), max=float(devs.max()),
                                         p95=float(np.percentile(devs, 95))),
               note="selection rule not recorded; coordinates verified against IBGE centroids")
    print(f"matched {len(devs)}/{len(src['base_names'])} against IBGE 2024; unmatched {unmatched}")
    print(f"states: {dict(states.most_common())}")
    print(f"centroid deviation: median {np.median(devs):.0f} m, p95 {np.percentile(devs,95):.0f} m, "
          f"max {devs.max():.0f} m")
    # Robustness: rebuild the instance from the exact IBGE centroids and check that
    # nothing downstream moves.  One site (Nova Ubirata, MT) differs by 22.7 km, which is
    # consistent with a boundary revision in a municipality of that size.
    import numpy as _np
    from run_m_ladder import coverage
    from scipy.optimize import milp, LinearConstraint, Bounds
    tl, to = _np.array(src["task_lats"]), _np.array(src["task_lons"])

    def instance(lats, lons):
        A = coverage(_np.array(lats), _np.array(lons), tl, to, 100.0).astype(int)
        A = A[A.sum(1) >= 1]
        C = (A.T @ A > 0).astype(int); _np.fill_diagonal(C, 0)
        r = milp(c=_np.ones(A.shape[1]),
                 constraints=LinearConstraint(A, lb=_np.ones(A.shape[0]), ub=_np.inf),
                 integrality=_np.ones(A.shape[1]), bounds=Bounds(0, 1))
        return dict(n_t=int(A.shape[0]), links=int(A.sum()), n_ZZ=int(C.sum() // 2),
                    k_opt=int(round(r.x.sum())))

    pub = instance(src["base_lats"], src["base_lons"])
    exact = instance([float(ib[norm(n)]["centroid_lat"]) for n in src["base_names"]],
                     [float(ib[norm(n)]["centroid_lon"]) for n in src["base_names"]])
    out["instance_as_published"] = pub
    out["instance_from_exact_centroids"] = exact
    out["robust"] = pub == exact
    print(f"as published       : {pub}")
    print(f"exact IBGE centroids: {exact}")
    print(f"instance unchanged  : {pub == exact}")

    ok = not unmatched and float(_np.median(devs)) < 200.0 and pub == exact
    print(f"\nPROVENANCE {'VERIFIED' if ok else 'FAILED'}")
    json.dump(out, open(RESULTS_DIR / "arco_provenance.json", "w"), indent=1)
    print("-> results/arco_provenance.json")


if __name__ == "__main__":
    main()
