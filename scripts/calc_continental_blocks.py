#!/usr/bin/env python3
"""The continental instance: the far end of the trade, regenerated rather than asserted.

`results/continental_blocks.json` previously had no generating script anywhere in the
repository, which meant the paper's most extreme claim -- 773 municipalities, n_ZZ = 39,974,
an exact block cost of 2^770 -- rested on a hand-placed file that no audit could rerun.
This script derives every field of it from the IBGE centroids and the shared primitives.

Writes results/continental_blocks.json.
"""
from __future__ import annotations

import json

import numpy as np

from _paths import RESULTS_DIR
from calc_block_decomposition import decompose
from calc_ilp_geographic import haversine_matrix, load_municipalities, solve_mds_ilp

R_KM = 200.0
TIMEOUT = 600.0


def main() -> None:
    lat, lon, table = load_municipalities("centroid")
    D = haversine_matrix(lat, lon)
    n = len(lat)

    # Base protection: a centroid covers itself and every centroid within R.  A is
    # [target, base], and the targets are the centroids themselves, which makes it a domination
    # problem rather than a bipartite cover.
    A = (D <= R_KM).astype(np.int64)
    np.fill_diagonal(A, 1)

    d = decompose(A)
    ilp = solve_mds_ilp(D, R_KM, TIMEOUT)
    assert ilp["status"] == "optimal", f"continental ILP not proven optimal: {ilp['status']}"

    # Unit propagation: a target reachable by exactly one base forces that base.
    deg = A.sum(1)
    forced = sorted({int(np.where(A[t] == 1)[0][0]) for t in np.where(deg == 1)[0]})
    isolated = [f"{table.iloc[b]['nm_mun']} ({table.iloc[b]['sigla_uf']})" for b in forced]

    out = dict(n=n, R_km=R_KM, n_ZZ=d["n_ZZ"], k_opt=int(ilp["k_opt"]),
               n_components=d["n_components"], block_sizes=d["block_sizes"],
               largest_block=d["largest_block"],
               log2_exact_cost=round(d["log2_enumeration_cost"], 4),
               forced=len(forced), phi=round(len(forced) / int(ilp["k_opt"]), 4),
               deg1_targets=int((deg == 1).sum()), isolated=isolated,
               direct_sum=d["direct_sum"])
    print(f"{'field':>18}  value")
    for k, v in out.items():
        print(f"{k:>18}  {v}")
    json.dump(out, open(RESULTS_DIR / "continental_blocks.json", "w"), indent=1)
    print("\n-> results/continental_blocks.json")


if __name__ == "__main__":
    main()
