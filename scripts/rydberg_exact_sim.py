"""Deterministic state-vector calculation for the 3x3 Rydberg MDS instance.

The former result was estimated from finite, unseeded Bloqade bitstring samples.
This program integrates the same 512-dimensional Hamiltonian and obtains the
probability and excitation mean directly from the final state vector.
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import scipy.integrate as integrate
import scipy.sparse as sp

from _paths import RESULTS_DIR
from bloqade_3x3_sim import (
    C6_BLOQADE,
    C6_PAPER_WRONG,
    C6_RATIO_2PI,
    DELTA_0,
    DELTA_F,
    DELTA_LOC_BASE,
    N_Q,
    OMEGA_MAX,
    POSITIONS,
    T_RAMP,
    build_adjacency,
    build_interaction_strengths_corrected,
    enumerate_mds,
)


ALPHA_VALUES = [0.30, 0.40, 0.50, 0.60, 0.70]
T_VALUES_US = [0.8, 1.2, 1.6, 2.0, 2.4, 2.6, 3.0, 3.5, 4.0]
PAPER_ALPHA = 0.50
PAPER_T_US = 2.6
GAMMA_CHECK_US_INV = 0.01
RTOL = 1e-9
ATOL = 1e-11


def basis_bits() -> np.ndarray:
    """Bits ordered as q0..q8, with q0 the most-significant basis bit."""
    indices = np.arange(2**N_Q, dtype=np.uint16)
    shifts = np.arange(N_Q - 1, -1, -1, dtype=np.uint16)
    return ((indices[:, None] >> shifts[None, :]) & 1).astype(np.float64)


def build_operators(alpha: float) -> dict[str, Any]:
    """Construct the Hamiltonian without an external quantum SDK."""
    bits = basis_bits()
    n_total = bits.sum(axis=1)
    interactions = build_interaction_strengths_corrected(POSITIONS, C6_BLOQADE)

    interaction_energy = np.zeros(2**N_Q, dtype=np.float64)
    for i in range(N_Q):
        for j in range(i + 1, N_Q):
            interaction_energy += interactions[i, j] * bits[:, i] * bits[:, j]
    local_energy = bits @ (alpha * DELTA_LOC_BASE)
    diagonal_static = interaction_energy - local_energy

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for index in range(2**N_Q):
        for qubit in range(N_Q):
            rows.append(index)
            cols.append(index ^ (1 << (N_Q - 1 - qubit)))
            values.append(0.5)
    transverse = sp.csr_matrix(
        (values, (rows, cols)), shape=(2**N_Q, 2**N_Q), dtype=np.float64
    )
    return {
        "diagonal_static": diagonal_static,
        "n_total": n_total,
        "transverse": transverse,
    }


def protocol(T_total: float):
    flat_duration = T_total - 2.0 * T_RAMP
    if flat_duration < 0.2 - 1e-12:
        raise ValueError("T_total must be at least 0.8 us for the fixed 0.3 us ramps")

    def omega(t: float) -> float:
        if t <= T_RAMP:
            return OMEGA_MAX * t / T_RAMP
        if t <= T_RAMP + flat_duration:
            return OMEGA_MAX
        return OMEGA_MAX * (T_total - t) / T_RAMP

    def global_detuning(t: float) -> float:
        return -DELTA_0 + (DELTA_0 + DELTA_F) * t / T_total

    return omega, global_detuning


def evolve(
    operators: dict[str, Any],
    T_total: float,
    mds_indices: list[int],
    gamma: float = 0.0,
) -> dict[str, Any]:
    """Integrate the state and return deterministic final-state observables."""
    omega, global_detuning = protocol(T_total)
    static = operators["diagonal_static"]
    n_total = operators["n_total"]
    transverse = operators["transverse"]

    def rhs(t: float, psi: np.ndarray) -> np.ndarray:
        diagonal = static - global_detuning(t) * n_total - 0.5j * gamma * n_total
        return -1j * (diagonal * psi + omega(t) * (transverse @ psi))

    initial = np.zeros(2**N_Q, dtype=np.complex128)
    initial[0] = 1.0
    start = time.perf_counter()
    solution = integrate.solve_ivp(
        rhs,
        (0.0, T_total),
        initial,
        method="DOP853",
        t_eval=[T_total],
        rtol=RTOL,
        atol=ATOL,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    final = solution.y[:, -1]
    probabilities = np.abs(final) ** 2
    norm = float(probabilities.sum())
    return {
        "P_opt": float(probabilities[mds_indices].sum()),
        "k_mean": float(probabilities @ n_total / norm),
        "state_norm": norm,
        "nfev": int(solution.nfev),
        "runtime_s": round(time.perf_counter() - start, 4),
    }


def main() -> None:
    adjacency = build_adjacency(POSITIONS)
    mds_indices = enumerate_mds(adjacency, k_opt=3)
    random_baseline = len(mds_indices) / (2**N_Q)
    print(
        f"MDS states={len(mds_indices)}, exact random baseline="
        f"{100 * random_baseline:.6f}%"
    )

    alpha_scan: list[dict[str, Any]] = []
    for alpha in ALPHA_VALUES:
        row = {"alpha": alpha, **evolve(build_operators(alpha), PAPER_T_US, mds_indices)}
        row["concentration_over_uniform"] = row["P_opt"] / random_baseline
        alpha_scan.append(row)
        print(
            f"alpha={alpha:.2f}, T={PAPER_T_US:.1f} us: "
            f"P(opt)={100 * row['P_opt']:.6f}%, k_mean={row['k_mean']:.6f}"
        )

    paper_point = next(row for row in alpha_scan if row["alpha"] == PAPER_ALPHA)
    paper_operators = build_operators(PAPER_ALPHA)
    time_scan: list[dict[str, Any]] = []
    for duration in T_VALUES_US:
        row = {"T_us": duration, **evolve(paper_operators, duration, mds_indices)}
        row["concentration_over_uniform"] = row["P_opt"] / random_baseline
        time_scan.append(row)
        print(f"alpha={PAPER_ALPHA:.2f}, T={duration:.1f} us: P(opt)={100 * row['P_opt']:.6f}%")

    best_alpha = max(alpha_scan, key=lambda row: row["P_opt"])
    best_time = max(time_scan, key=lambda row: row["P_opt"])
    decay_point = evolve(
        paper_operators, PAPER_T_US, mds_indices, gamma=GAMMA_CHECK_US_INV
    )
    relative_degradation = 1.0 - decay_point["P_opt"] / paper_point["P_opt"]

    output = {
        "description": "Deterministic state-vector calculation for 3x3 Rydberg MDS",
        "method": "scipy.solve_ivp DOP853; direct final-state probabilities; no sampling",
        "basis_dimension": 2**N_Q,
        "C6_rad_per_us_um6": C6_BLOQADE,
        "C6_specification_MHz_um6_times_2pi": C6_PAPER_WRONG,
        "incorrect_value_when_2pi_is_omitted_rad_per_us_um6": C6_PAPER_WRONG,
        "C6_correction_factor": C6_RATIO_2PI,
        "solver": {"rtol": RTOL, "atol": ATOL},
        "n_mds_states": len(mds_indices),
        "mds_basis_indices": mds_indices,
        "uniform_random_baseline": random_baseline,
        "alpha_scan_at_T_2p6_us": alpha_scan,
        "best_alpha_on_scan": best_alpha["alpha"],
        "paper_operating_point": paper_point,
        "time_scan_at_alpha_0p5": time_scan,
        "best_time_on_scan_us": best_time["T_us"],
        "best_time_on_scan_P_opt": best_time["P_opt"],
        "non_hermitian_decay_check": {
            "qualification": (
                "No-jump non-Hermitian approximation, not a full Lindblad result"
            ),
            "Gamma_us_inv": GAMMA_CHECK_US_INV,
            **decay_point,
            "relative_P_opt_degradation": relative_degradation,
        },
        "superseded_reported_values": {
            "P_opt": 0.1575,
            "k_mean": 1.95,
            "uniform_random_baseline": 0.0188,
            "concentration_over_uniform": 8.4,
            "T_opt_us": 2.6,
        },
    }
    output_path = RESULTS_DIR / "rydberg_3x3_exact.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")
    print(
        "Corrected paper point: "
        f"P(opt)={100 * paper_point['P_opt']:.6f}%, "
        f"k_mean={paper_point['k_mean']:.6f}, "
        f"concentration={paper_point['concentration_over_uniform']:.6f}x"
    )
    print(
        f"Best scanned T={best_time['T_us']:.1f} us, "
        f"P(opt)={100 * best_time['P_opt']:.6f}%"
    )


if __name__ == "__main__":
    main()
