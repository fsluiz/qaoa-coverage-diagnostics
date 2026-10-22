#!/usr/bin/env python3
"""Control experiment: sweep R with the targets held fixed.

In `calc_ensemble_squeeze.py` each target is drawn within 0.7R of its anchor base, so the
target distribution dilates as R grows and R is not an independent variable.  Table 3 of the
main text is the opposite: the targets are fixed and only the coverage predicate moves.  The
two experiments disagree about whether phi keeps falling (Table 3) or plateaus near 0.2
(ensemble), and the generator difference is a candidate explanation.

This script settles it: targets are drawn ONCE at the smallest radius and then held fixed
while R is swept, exactly as in Table 3, at several register sizes.

Writes results/frozen_target_sweep.json.
"""
from __future__ import annotations

import csv
import json
from math import comb

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from _paths import DATA_DIR, RESULTS_DIR
from calc_ensemble_squeeze import haversine, load_sites
from calc_triviality import n_zz_of, propagate

N_B_VALUES = (20, 40, 80, 160)
N_INSTANCES = 15
R_FRACTIONS = (0.30, 0.45, 0.60, 0.80, 1.00, 1.30)   # x3 gives R/delta 0.9..3.9
ILP_TIME_LIMIT = 30.0


def main() -> None:
    sites = load_sites()
    rows = []
    for n_b in N_B_VALUES:
        rng0 = np.random.default_rng(0)
        sub = sites[rng0.choice(len(sites), n_b, replace=False)]
        Ds = haversine(sub[:, 0], sub[:, 1], sub[:, 0], sub[:, 1])
        np.fill_diagonal(Ds, np.inf)
        delta = float(np.median(Ds.min(1)))
        R0 = R_FRACTIONS[0] * delta * 3.0                     # targets drawn at the SMALLEST R
        for s in range(N_INSTANCES):
            rng = np.random.default_rng(77_000 * n_b + s)
            idx = rng.choice(len(sites), n_b, replace=False)
            b = sites[idx]
            anchor = rng.integers(0, n_b, size=n_b)
            off = rng.uniform(0.0, 0.7 * R0, size=n_b)         # fixed offsets, fixed targets
            ang = rng.uniform(0.0, 2 * np.pi, size=n_b)
            tlat = b[anchor, 0] + off * np.cos(ang) / 111.0
            tlon = b[anchor, 1] + off * np.sin(ang) / (111.0 * np.cos(np.radians(b[anchor, 0])))
            D = haversine(tlat, tlon, b[:, 0], b[:, 1])
            for f in R_FRACTIONS:
                R = f * delta * 3.0
                A = (D <= R).astype(int)
                A = A[A.sum(1) >= 1]
                if A.shape[0] < 5:
                    continue
                r = milp(c=np.ones(A.shape[1]),
                         constraints=LinearConstraint(A, lb=np.ones(A.shape[0]), ub=np.inf),
                         integrality=np.ones(A.shape[1]), bounds=Bounds(0, 1),
                         options={"time_limit": ILP_TIME_LIMIT})
                if r.x is None or r.status != 0:
                    continue      # status 1 = time limit: an incumbent is not an optimum
                k = int(round(r.x.sum()))
                p = propagate(A, np.ones(A.shape[0], int))
                fb, nk = A.shape[1] - p["forced"], k - p["forced"]
                rows.append(dict(n_b=n_b, R_over_delta=round(R / delta, 3), seed=s,
                                 n_t=int(A.shape[0]), n_ZZ=n_zz_of(A), k_opt=k,
                                 forced=p["forced"], phi=round(p["forced"] / k, 4),
                                 residual=comb(fb, nk) if 0 <= nk <= fb else None))
        print(f"n_B={n_b}: {len([r for r in rows if r['n_b']==n_b])} rows", flush=True)

    print(f"\n{'n_B':>5} {'R/delta':>8} {'n_ZZ':>9} {'phi':>7} {'k_opt':>7}")
    for n_b in N_B_VALUES:
        for f in R_FRACTIONS:
            sel = [r for r in rows if r["n_b"] == n_b and abs(r["R_over_delta"] - 3*f) < 0.02]
            if sel:
                print(f"{n_b:>5} {3*f:>8.2f} {np.mean([r['n_ZZ'] for r in sel]):>9.1f} "
                      f"{np.mean([r['phi'] for r in sel]):>7.3f} "
                      f"{np.mean([r['k_opt'] for r in sel]):>7.1f}")
    json.dump(dict(meta=dict(n_instances=N_INSTANCES, note="targets drawn once at the smallest "
                             "radius and held fixed while R is swept"), rows=rows),
              open(RESULTS_DIR / "frozen_target_sweep.json", "w"), indent=1)
    print("-> results/frozen_target_sweep.json")


if __name__ == "__main__":
    main()
