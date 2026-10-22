#!/usr/bin/env python3
"""Test whether the current 3x3 Rydberg Hamiltonian encodes MDS.

The audit distinguishes three different objects that were conflated in paper v3:

1. the grid graph on which minimum dominating sets are defined;
2. the blockade graph, which forbids simultaneous neighbouring excitations and
   therefore supplies an independence-type constraint;
3. the full soft van-der-Waals Hamiltonian used in the time evolution.

The program enumerates all 512 computational states, compares the relevant state
sets, diagonalises the final classical Hamiltonian, and re-integrates the reported
operating point so its target probability can be partitioned into blockade-
compatible and blockade-incompatible MDS states.

Writes results/rydberg_encoding_audit.json.
"""
from __future__ import annotations

import json
import itertools
import math

import numpy as np
from scipy.integrate import solve_ivp
import scipy.sparse as sp

from _paths import RESULTS_DIR

N_Q = 9
LAMBDA_QUBO = 2.0
C6_RAD_PER_US_UM6 = 2.0 * math.pi * 862_690.0
OMEGA_MAX = 15.0
R_B = (C6_RAD_PER_US_UM6 / OMEGA_MAX) ** (1.0 / 6.0)
T_RAMP = 0.3
DELTA_0 = 10.0
DELTA_F = 1.0
RTOL = 1e-9
ATOL = 1e-11
CLOSED_DEGREES = np.asarray([3, 4, 3, 4, 5, 4, 3, 4, 3], dtype=float)
DELTA_LOC_CORRECT = LAMBDA_QUBO * CLOSED_DEGREES - 1.0


def basis_bits() -> np.ndarray:
    """All bitstrings, with q0 as the most-significant basis bit."""
    indices = np.arange(1 << N_Q, dtype=np.uint16)
    shifts = np.arange(N_Q - 1, -1, -1, dtype=np.uint16)
    return ((indices[:, None] >> shifts[None, :]) & 1).astype(np.float64)


def geometry(spacing: float) -> tuple[list[tuple[float, float]], np.ndarray]:
    positions = [(i * spacing, j * spacing) for i in range(3) for j in range(3)]
    points = np.asarray(positions, dtype=float)
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    return positions, distances


def adjacency(distances: np.ndarray, radius: float) -> np.ndarray:
    return ((distances <= radius) & (distances > 0.0)).astype(np.int8)


def is_dominating_set(bits: np.ndarray, graph: np.ndarray) -> bool:
    selected = bits.astype(bool)
    return bool(np.all(selected | ((graph @ selected.astype(np.int8)) > 0)))


def enumerate_mds(graph: np.ndarray, k_opt: int) -> list[int]:
    bits = basis_bits().astype(np.int8)
    return [
        index for index, row in enumerate(bits)
        if int(row.sum()) == k_opt and is_dominating_set(row, graph)
    ]


def gamma_of(graph: np.ndarray) -> tuple[int, list[int]]:
    for k in range(1, N_Q + 1):
        for chosen in itertools.combinations(range(N_Q), k):
            row = np.zeros(N_Q, dtype=np.int8)
            row[list(chosen)] = 1
            if is_dominating_set(row, graph):
                return k, list(chosen)
    raise RuntimeError("finite graph has no dominating set")


def operators(spacing: float, detuning: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    _, distances = geometry(spacing)
    bits = basis_bits()
    interaction_energy = np.zeros(1 << N_Q, dtype=float)
    for i in range(N_Q):
        for j in range(i + 1, N_Q):
            coupling = C6_RAD_PER_US_UM6 / distances[i, j] ** 6
            interaction_energy += coupling * bits[:, i] * bits[:, j]

    rows: list[int] = []
    cols: list[int] = []
    for index in range(1 << N_Q):
        for qubit in range(N_Q):
            rows.append(index)
            cols.append(index ^ (1 << (N_Q - 1 - qubit)))
    transverse = sp.csr_matrix(
        (np.full(len(rows), 0.5), (rows, cols)),
        shape=(1 << N_Q, 1 << N_Q),
        dtype=float,
    )
    return {
        "diagonal_static": interaction_energy - bits @ (alpha * detuning),
        "n_total": bits.sum(axis=1),
        "transverse": transverse,
    }


def protocol(duration: float):
    flat_duration = duration - 2.0 * T_RAMP
    if flat_duration < 0.2 - 1e-12:
        raise ValueError("duration must be at least 0.8 us")

    def omega(t: float) -> float:
        if t <= T_RAMP:
            return OMEGA_MAX * t / T_RAMP
        if t <= T_RAMP + flat_duration:
            return OMEGA_MAX
        return OMEGA_MAX * (duration - t) / T_RAMP

    def global_detuning(t: float) -> float:
        return -DELTA_0 + (DELTA_0 + DELTA_F) * t / duration

    return omega, global_detuning


def state_set(indices: np.ndarray | list[int], bits: np.ndarray) -> list[list[int]]:
    return [np.flatnonzero(bits[int(i)]).astype(int).tolist() for i in indices]


def independent_mask(bits: np.ndarray, graph: np.ndarray) -> np.ndarray:
    edges = np.argwhere(np.triu(graph, 1) > 0)
    if len(edges) == 0:
        return np.ones(len(bits), dtype=bool)
    return np.asarray([
        all(not (row[i] and row[j]) for i, j in edges)
        for row in bits
    ], dtype=bool)


def dominating_mask(bits: np.ndarray, graph: np.ndarray) -> np.ndarray:
    return np.asarray([is_dominating_set(row.astype(int), graph) for row in bits], dtype=bool)


def evolve_probabilities(ops: dict, duration: float) -> np.ndarray:
    omega, global_detuning = protocol(duration)
    static = ops["diagonal_static"]
    n_total = ops["n_total"]
    transverse = ops["transverse"]

    def rhs(t: float, psi: np.ndarray) -> np.ndarray:
        diagonal = static - global_detuning(t) * n_total
        return -1j * (diagonal * psi + omega(t) * (transverse @ psi))

    initial = np.zeros(1 << N_Q, dtype=np.complex128)
    initial[0] = 1.0
    solution = solve_ivp(
        rhs,
        (0.0, duration),
        initial,
        method="DOP853",
        t_eval=[duration],
        rtol=RTOL,
        atol=ATOL,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    probability = np.abs(solution.y[:, -1]) ** 2
    return probability / probability.sum()


def audit_spacing(spacing: float, duration: float = 2.6) -> dict:
    _, distances = geometry(spacing)
    grid = adjacency(distances, 1.01 * spacing)
    blockade = adjacency(distances, R_B)
    bits = basis_bits().astype(np.int8)
    cardinality = bits.sum(axis=1)

    grid_gamma, _ = gamma_of(grid)
    blockade_gamma, _ = gamma_of(blockade)
    mds_indices = np.asarray(enumerate_mds(grid, k_opt=grid_gamma), dtype=int)
    mds_mask = np.zeros(len(bits), dtype=bool)
    mds_mask[mds_indices] = True
    independent = independent_mask(bits, blockade)
    grid_dominating = dominating_mask(bits, grid)

    max_independent = int(cardinality[independent].max())
    max_independent_mask = independent & (cardinality == max_independent)

    ops = operators(spacing, DELTA_LOC_CORRECT, 1.0)
    final_diagonal = ops["diagonal_static"] - DELTA_F * ops["n_total"]
    minimum = float(final_diagonal.min())
    ground = np.isclose(final_diagonal, minimum, atol=1e-9, rtol=0.0)
    lowest_mds = float(final_diagonal[mds_mask].min())

    probability = evolve_probabilities(ops, duration)
    compatible_mds = mds_mask & independent
    incompatible_mds = mds_mask & ~independent

    return {
        "spacing_um": spacing,
        "blockade_radius_um": R_B,
        "blockade_reaches_grid_edges": bool(spacing < R_B),
        "blockade_reaches_grid_diagonals": bool(math.sqrt(2.0) * spacing < R_B),
        "grid": {
            "n_edges": int(grid.sum() // 2),
            "domination_number": int(grid_gamma),
            "n_minimum_dominating_sets": int(mds_mask.sum()),
            "minimum_dominating_sets": state_set(mds_indices, bits),
        },
        "blockade_graph": {
            "n_edges": int(blockade.sum() // 2),
            "domination_number_not_an_encoded_objective": int(blockade_gamma),
            "maximum_independent_set_size": max_independent,
            "n_maximum_independent_sets": int(max_independent_mask.sum()),
        },
        "mds_vs_blockade": {
            "n_blockade_compatible_mds": int(compatible_mds.sum()),
            "n_blockade_incompatible_mds": int(incompatible_mds.sum()),
            "compatible_sets": state_set(np.flatnonzero(compatible_mds), bits),
            "incompatible_sets": state_set(np.flatnonzero(incompatible_mds), bits),
            "all_mds_are_blockade_allowed": bool(np.all(independent[mds_mask])),
        },
        "final_classical_hamiltonian": {
            "omega_rad_per_us": 0.0,
            "global_detuning_rad_per_us": DELTA_F,
            "ground_energy": minimum,
            "ground_degeneracy": int(ground.sum()),
            "ground_cardinalities": sorted(set(cardinality[ground].astype(int).tolist())),
            "ground_bit_sets": state_set(np.flatnonzero(ground), bits),
            "n_ground_states_that_are_grid_mds": int((ground & mds_mask).sum()),
            "n_ground_states_that_dominate_grid": int((ground & grid_dominating).sum()),
            "lowest_grid_mds_energy": lowest_mds,
            "n_states_below_lowest_grid_mds": int(np.sum(final_diagonal < lowest_mds - 1e-9)),
            "ground_manifold_equals_grid_mds": bool(np.array_equal(ground, mds_mask)),
        },
        "dynamics_at_reported_duration": {
            "duration_us": duration,
            "P_all_grid_mds": float(probability[mds_mask].sum()),
            "P_blockade_compatible_grid_mds": float(probability[compatible_mds].sum()),
            "P_blockade_incompatible_grid_mds": float(probability[incompatible_mds].sum()),
            "P_final_hamiltonian_ground_manifold": float(probability[ground].sum()),
            "P_any_grid_dominating_set": float(probability[grid_dominating].sum()),
            "P_any_blockade_independent_set": float(probability[independent].sum()),
            "uniform_grid_mds_baseline": float(mds_mask.mean()),
        },
    }


def main() -> None:
    rows = [audit_spacing(5.0), audit_spacing(7.0)]
    out = {
        "description": "Exact enumeration and statevector audit of the 3x3 Rydberg interpretation",
        "cases": rows,
        "scientific_conclusion": (
            "The blockade supplies an independence-type exclusion, not domination. "
            "For neither audited spacing is the final Rydberg ground manifold the set "
            "of minimum dominating sets.  P_all_grid_mds is therefore a target-set "
            "overlap of a heuristic dynamics, not success probability of a faithful MDS encoding."
        ),
        "allowed_next_draft_language": (
            "Rydberg-Ising heuristic whose output is scored against externally enumerated "
            "minimum-dominating-set bitstrings"
        ),
        "prohibited_next_draft_language": (
            "native or faithful encoding of the domination constraint by blockade"
        ),
    }
    path = RESULTS_DIR / "rydberg_encoding_audit.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        relation = row["mds_vs_blockade"]
        final = row["final_classical_hamiltonian"]
        dyn = row["dynamics_at_reported_duration"]
        print(
            f"d={row['spacing_um']:.1f} um: MDS allowed {relation['n_blockade_compatible_mds']}/"
            f"{row['grid']['n_minimum_dominating_sets']}; ground∩MDS="
            f"{final['n_ground_states_that_are_grid_mds']}; P(MDS)={dyn['P_all_grid_mds']:.6f}"
        )
    print(f"-> {path}")


if __name__ == "__main__":
    main()
