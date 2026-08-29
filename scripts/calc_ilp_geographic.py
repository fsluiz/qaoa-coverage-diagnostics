"""Solve MDS on the exact 2024 IBGE Legal Amazon municipality set.

The default uses polygon centroids, matching the manuscript methodology. The
official municipal seats can be selected as a geographic sensitivity check.
There is deliberately no synthetic-data fallback.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _paths import DATA_DIR, RESULTS_DIR


EARTH_R_KM = 6371.0088
INPUT_PATH = DATA_DIR / "ibge_amazonia_legal_2024.csv"
DEFAULT_R_VALUES = [50, 75, 100, 125, 150, 175, 200, 250, 300]
EXPECTED_N = 773


def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return the vectorised pairwise great-circle distance matrix in km."""
    phi = np.radians(lats)
    lam = np.radians(lons)
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi[:, None])
        * np.cos(phi[None, :])
        * np.sin(dlam / 2.0) ** 2
    )
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_municipalities(
    coordinate_type: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load and validate the versioned table derived from official IBGE data."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run scripts/prepare_ibge_amazonia_legal.py first."
        )
    table = pd.read_csv(INPUT_PATH, dtype={"cd_mun": str})
    if len(table) != EXPECTED_N or table["cd_mun"].nunique() != EXPECTED_N:
        raise ValueError(
            f"Expected {EXPECTED_N} unique IBGE municipality codes, found "
            f"{len(table)} rows and {table['cd_mun'].nunique()} codes"
        )
    prefix = "centroid" if coordinate_type == "centroid" else "seat"
    lats = table[f"{prefix}_lat"].to_numpy(dtype=np.float64)
    lons = table[f"{prefix}_lon"].to_numpy(dtype=np.float64)
    if not np.isfinite(lats).all() or not np.isfinite(lons).all():
        raise ValueError(f"Missing or non-finite {coordinate_type} coordinates")
    return lats, lons, table


def solve_mds_ilp(D: np.ndarray, radius_km: float, timeout: float) -> dict[str, Any]:
    """Solve the binary minimum dominating set ILP with scipy/HiGHS."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix

    n = D.shape[0]
    closed_adj = (D <= radius_km).astype(np.float64)
    constraint = LinearConstraint(
        csr_matrix(closed_adj), lb=np.ones(n), ub=np.full(n, np.inf)
    )

    start = time.perf_counter()
    try:
        result = milp(
            np.ones(n),
            constraints=constraint,
            integrality=np.ones(n, dtype=int),
            bounds=Bounds(np.zeros(n), np.ones(n)),
            options={"time_limit": timeout, "disp": False},
        )
    except Exception as exc:
        return {
            "status": "error",
            "solver_status_code": None,
            "solver_message": str(exc),
            "k_opt": None,
            "k_incumbent": None,
            "mip_gap": None,
            "t_ilp_s": round(time.perf_counter() - start, 4),
            "selected_indices": None,
        }

    elapsed = time.perf_counter() - start
    status_names = {
        0: "optimal",
        1: "time_or_iteration_limit",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }
    status = status_names.get(result.status, "unknown")
    has_incumbent = result.x is not None and result.fun is not None
    incumbent = int(round(float(result.fun))) if has_incumbent else None
    selected = (
        np.flatnonzero(np.asarray(result.x) > 0.5).astype(int).tolist()
        if has_incumbent
        else None
    )
    return {
        "status": status,
        "solver_status_code": int(result.status),
        "solver_message": str(result.message),
        "k_opt": incumbent if result.status == 0 else None,
        "k_incumbent": incumbent,
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "t_ilp_s": round(elapsed, 4),
        "selected_indices": selected,
    }


def graph_metrics(D: np.ndarray, radius_km: float) -> tuple[float, int]:
    """Return open-neighbour average degree and exact QUBO ZZ-pair count.

    A QUBO pair x_j x_k occurs when candidates j and k co-cover at least one
    target in a *closed* neighbourhood. Thus n_ZZ counts positive entries in
    the strict upper triangle of C.T @ C, where C includes the diagonal.
    """
    closed_adj = (D <= radius_km).astype(np.uint16)
    open_adj = closed_adj.copy()
    np.fill_diagonal(open_adj, 0)
    average_degree = float(open_adj.sum(axis=1).mean())
    co_cover = closed_adj.T @ closed_adj
    n_zz = int(np.count_nonzero(np.triu(co_cover, k=1)))
    return average_degree, n_zz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coordinate-type", choices=("centroid", "seat"), default="centroid"
    )
    parser.add_argument("--radii", type=float, nargs="+", default=DEFAULT_R_VALUES)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or RESULTS_DIR / (
        "ilp_geographic_ibge_2024_centroids.json"
        if args.coordinate_type == "centroid"
        else "ilp_geographic_ibge_2024_seats.json"
    )

    print(
        "=== MDS on 773 official IBGE Legal Amazon municipalities "
        f"({args.coordinate_type}s) ==="
    )
    lats, lons, table = load_municipalities(args.coordinate_type)
    start = time.perf_counter()
    distances = haversine_matrix(lats, lons)
    print(f"Distance matrix calculated in {time.perf_counter() - start:.2f}s")

    sweep: list[dict[str, Any]] = []
    for radius_km in args.radii:
        average_degree, n_zz = graph_metrics(distances, radius_km)
        solved = solve_mds_ilp(distances, radius_km, args.timeout)
        selected_indices = solved.pop("selected_indices")
        if selected_indices is not None:
            closed_adj = distances <= radius_km
            if not np.all(closed_adj[:, selected_indices].any(axis=1)):
                raise RuntimeError(
                    f"Solver incumbent failed domination check at R={radius_km} km"
                )
            if len(selected_indices) != solved["k_incumbent"]:
                raise RuntimeError(
                    f"Solver cardinality mismatch at R={radius_km} km"
                )
        selected_codes = (
            table.iloc[selected_indices]["cd_mun"].tolist()
            if selected_indices is not None
            else None
        )
        row = {
            "R_km": float(radius_km),
            "n_nodes": EXPECTED_N,
            "avg_deg": round(average_degree, 4),
            "n_ZZ": n_zz,
            **solved,
            "selected_cd_mun": selected_codes,
        }
        sweep.append(row)
        print(
            f"R={radius_km:6.1f} km avg_deg={average_degree:7.2f} "
            f"n_ZZ={n_zz:7d} k={row['k_opt']} status={row['status']} "
            f"time={row['t_ilp_s']:.2f}s"
        )

    payload = {
        "description": (
            "Minimum Dominating Set on the official IBGE Amazônia Legal 2024 "
            f"municipality {args.coordinate_type} coordinates"
        ),
        "data_file": "data/ibge_amazonia_legal_2024.csv",
        "provenance_file": "data/ibge_amazonia_legal_2024_provenance.json",
        "coordinate_type": args.coordinate_type,
        "n_nodes": EXPECTED_N,
        "earth_radius_km": EARTH_R_KM,
        "timeout_s": args.timeout,
        "n_ZZ_definition": (
            "Number of unordered candidate pairs that co-cover at least one "
            "target under closed-neighbourhood domination"
        ),
        "sweep": sweep,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
