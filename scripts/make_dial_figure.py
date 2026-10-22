#!/usr/bin/env python3
"""The range dial on the frozen PEF-24 instance: what it costs to run, to simulate, to solve.

For each radius the frozen bases and waypoints are re-linked (targets held fixed), and three
quantities are computed from scratch:

  n_ZZ          circuit cost: nonzero ZZ couplings of the set-cover QUBO
  sum_i 2^|V_i| circuit-simulation cost: combined block-state dimension of G_ZZ
  ILP time      instance cost: median wall time of 25 HiGHS solves (scipy.milp)

The first two rise by five orders of magnitude together; the third does not move.  Writes
results/dial_sweep.json and figures/fig_dial.pdf.
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from _paths import DATA_DIR, FIGURES_DIR, RESULTS_DIR

RADII = (200, 250, 300, 350, 400, 450, 500, 550, 600)
REPEATS = 25


def haversine(lat1, lon1, lat2, lon2, r=6371.0):
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def solve(A):
    n = A.shape[1]
    cons = LinearConstraint(A.astype(float), np.ones(A.shape[0]), np.inf)
    times, k = [], None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        res = milp(np.ones(n), constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1))
        times.append(time.perf_counter() - t0)
        k = int(round(res.fun))
    return k, 1e3 * float(np.median(times))


def main() -> None:
    d = json.load(open(DATA_DIR / "frozen_rung1.json"))
    D = haversine(np.asarray(d["task_lats"]), np.asarray(d["task_lons"]),
                  np.asarray(d["base_lats"]), np.asarray(d["base_lons"]))
    rows = []
    for R in RADII:
        A = (D <= R).astype(int)                      # (n_t, n_b)
        A = A[A.any(1)]
        C = (A.T @ A) > 0
        np.fill_diagonal(C, False)
        nzz = int(np.triu(C).sum())
        nc, lab = connected_components(csr_matrix(C), directed=False)
        sizes = sorted(np.bincount(lab).tolist(), reverse=True)
        k, ms = solve(A)
        rows.append(dict(R_km=R, n_ZZ=nzz, blocks=nc, largest=sizes[0], block_sizes=sizes,
                         state_dim=int(sum(2 ** s for s in sizes)), k_opt=k, ilp_ms=ms))
        print(f"R={R}: nZZ={nzz} blocks={nc} largest={sizes[0]} "
              f"dim={rows[-1]['state_dim']} k={k} ilp={ms:.2f} ms")
    json.dump(dict(instance="frozen_rung1.json", ilp_repeats=REPEATS, rows=rows),
              open(RESULTS_DIR / "dial_sweep.json", "w"), indent=1)

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.dpi": 600,
                         "savefig.bbox": "tight"})
    blue, red, grey = "#4477AA", "#CC6677", "#555555"
    R = np.array(RADII)
    fig, ax = plt.subplots(3, 1, figsize=(3.4, 4.0), sharex=True,
                           gridspec_kw=dict(hspace=0.55))
    ax[0].plot(R, [r["n_ZZ"] for r in rows], "o-", color=blue, ms=3.5, lw=1.2)
    ax[0].set_ylabel(r"$n_{ZZ}$")
    ax[0].set_title("(a) cost of running the circuit", loc="left", fontsize=8)
    ax[1].semilogy(R, [r["state_dim"] for r in rows], "s-", color=red, ms=3.5, lw=1.2)
    ax[1].axhline(2 ** 24, color=grey, ls=":", lw=0.8)
    ax[1].text(212, 2 ** 24 * 1.6, r"full register $2^{24}$", fontsize=6.5, color=grey)
    ax[1].set_ylim(10, 2 ** 24 * 12)
    ax[1].set_ylabel(r"$\sum_i 2^{|V_i|}$")
    ax[1].set_title("(b) cost of simulating it exactly", loc="left", fontsize=8)
    ax[2].plot(R, [r["ilp_ms"] for r in rows], "^-", color=grey, ms=3.5, lw=1.2)
    ax[2].set_ylim(0, 5 * max(1.0, float(np.ceil(max(r["ilp_ms"] for r in rows) / 5))) * 2)
    ax[2].set_ylabel("ILP (ms)")
    ax[2].set_title("(c) cost of solving the instance", loc="left", fontsize=8)
    ax[2].set_xlabel(r"coverage radius $R$ (km)")
    for a in ax:
        a.tick_params(labelsize=7)
    fig.align_ylabels(ax)
    fig.savefig(FIGURES_DIR / "fig_dial.pdf")
    print(f"-> {FIGURES_DIR / 'fig_dial.pdf'}")


if __name__ == "__main__":
    main()
