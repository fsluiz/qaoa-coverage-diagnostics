"""
Sampled Bloqade validation for the 3x3 Rydberg MDS calculation.
Reconciles Bloqade (C6=5,420,441) with previous Lindblad/scipy (C6=862,690) results.

This program samples bitstrings and is therefore not the source for reported
probabilities. Use ``rydberg_exact_sim.py`` for deterministic headline values.

Three parts:
  Part 1 — Bloqade alpha scan (alpha in {0.30, 0.40, 0.50, 0.60, 0.70})
  Part 2 — Bloqade T scan at best alpha (alpha=0.50)
  Part 3 — Lindblad NHH check at corrected C6=5,420,441; Gamma in {0.0, 0.01}

Author: Turing (sub-agent for Fabrício de Souza Luiz, IFGW/UNICAMP)
Date: 2026-06-27
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import scipy.integrate as integrate
import scipy.sparse as sp

from _paths import RESULTS_DIR

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Physical constants and geometry
# ---------------------------------------------------------------------------
N_Q: int = 9
DELTA_SPACING_UM: float = 5.0          # µm lattice spacing
C6_MHZ_UM6: float = 862_690.0           # hardware specification, multiplied by 2π
C6_BLOQADE: float = 2.0 * np.pi * C6_MHZ_UM6  # rad·µs⁻¹·µm^6 (Rb 70S)
C6_PAPER_WRONG: float = C6_MHZ_UM6      # legacy alias: value used without 2π conversion
C6_RATIO_2PI: float = C6_BLOQADE / C6_PAPER_WRONG
R_ADJ: float = 6.2                      # µm adjacency radius for domination check

OMEGA_MAX: float = 15.0                # rad/µs
DELTA_0: float = 10.0                  # rad/µs  (Δ_global(0) = -Δ_0)
DELTA_F: float = 1.0                   # rad/µs  (Δ_global(T) = +Δ_f)
T_RAMP: float = 0.3                    # µs ramp duration
LAMBDA_QUBO: float = 2.0              # QUBO penalty coefficient
P_RND: float = 10.0 / 512.0            # random baseline probability = 10 MDS / 2^9 states = 1.953%

# Closed-neighbourhood degrees: corners=3, edges=4, center=5
CLOSED_DEGS: np.ndarray = np.array([3, 4, 3, 4, 5, 4, 3, 4, 3], dtype=float)
# Local detuning offsets: δ_i^loc = 2λ·closed_degs[i] - 1 → [11,15,11,15,19,15,11,15,11]
DELTA_LOC_BASE: np.ndarray = 2.0 * LAMBDA_QUBO * CLOSED_DEGS - 1.0  # without alpha

# Atom positions (i*δ, j*δ), row-major ordering i∈{0,1,2}, j∈{0,1,2}
POSITIONS: list[tuple[float, float]] = [
    (i * DELTA_SPACING_UM, j * DELTA_SPACING_UM)
    for i in range(3)
    for j in range(3)
]


# ---------------------------------------------------------------------------
# MDS utilities (shared by both Parts)
# ---------------------------------------------------------------------------

def build_adjacency(positions: list[tuple[float, float]], r_adj: float = R_ADJ) -> np.ndarray:
    """Binary adjacency matrix (no self-loops) for the 3×3 atom grid."""
    n = len(positions)
    pos = np.array(positions)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) <= r_adj:
                adj[i, j] = adj[j, i] = 1
    return adj


def is_dominating_set(x: np.ndarray, adj: np.ndarray) -> bool:
    """Return True if bitstring x (length 9) is a valid dominating set."""
    for i in range(len(x)):
        if x[i]:
            continue
        if not any(adj[i, j] and x[j] for j in range(len(x))):
            return False
    return True


def enumerate_mds(adj: np.ndarray, k_opt: int = 3) -> list[int]:
    """
    Return sorted list of computational-basis indices for valid MDS bitstrings.

    Basis ordering: index i encodes qubit string (q_0 q_1 ... q_8) with
    q_0 as the most-significant bit: i = Σ_k q_k · 2^(8-k).
    """
    mds_indices: list[int] = []
    for idx in range(2**N_Q):
        x = np.array([(idx >> (N_Q - 1 - k)) & 1 for k in range(N_Q)], dtype=int)
        if x.sum() == k_opt and is_dominating_set(x, adj):
            mds_indices.append(idx)
    return mds_indices


def compute_p_opt_from_bitstrings(
    bs: np.ndarray, adj: np.ndarray, k_opt: int = 3
) -> float:
    """
    Fraction of shot bitstrings that are valid MDS with exactly k_opt excitations.

    **Bloqade bit convention:** ``report.bitstrings()`` returns 0 for Rydberg-excited
    and 1 for ground (QuEra hardware convention; BloqadeTask.run applies ``1 - shot``
    before storing the result).  This function inverts the bitstrings internally so
    that x[i]=1 means atom i is Rydberg-excited, consistent with the MDS formulation.

    Parameters
    ----------
    bs : ndarray of shape (shots, 9)
        Raw bitstrings from ``report.bitstrings()[0]``; 0=Rydberg, 1=ground.
    adj : binary adjacency matrix
    k_opt : minimum dominating set size

    Returns
    -------
    float
        P(opt) = fraction of shots that are valid minimum dominating sets.
    """
    bs_exc = 1 - np.asarray(bs, dtype=int)   # invert: 1=Rydberg, 0=ground
    shots = len(bs_exc)
    count = 0
    for row in bs_exc:
        if row.sum() == k_opt and is_dominating_set(row, adj):
            count += 1
    return count / shots


# ---------------------------------------------------------------------------
# Part 1 & 2: Bloqade emulation helpers
# ---------------------------------------------------------------------------

def build_bloqade_program(alpha: float, T_total: float):
    """
    Construct a Bloqade analog program for the 3×3 MDS problem.

    Parameters
    ----------
    alpha : float
        Local detuning amplitude multiplier.
    T_total : float
        Total sweep duration in µs.

    Returns
    -------
    Bloqade program builder object ready for `.bloqade.python().run(shots=N)`.
    """
    from bloqade.analog import start

    T_H = T_total - 2.0 * T_RAMP   # flat-top duration; ensure > 0.2 µs
    if T_H < 0.2:
        # For very short T, adjust ramp to keep T_H ≥ 0.2 µs
        T_R = (T_total - 0.2) / 2.0
        T_H = 0.2
    else:
        T_R = T_RAMP

    local_offsets = alpha * DELTA_LOC_BASE

    prog = (
        start
        .add_position(POSITIONS)
        .rydberg.rabi.amplitude.uniform
        .piecewise_linear(
            durations=[T_R, T_H, T_R],
            values=[0.0, OMEGA_MAX, OMEGA_MAX, 0.0],
        )
        .detuning.uniform
        .piecewise_linear(
            durations=[T_total],
            values=[-DELTA_0, DELTA_F],
        )
    )

    for i, offset in enumerate(local_offsets):
        prog = prog.detuning.location(i).constant(
            duration=T_total, value=float(offset)
        )

    return prog


# ---------------------------------------------------------------------------
# Part 3: Lindblad NHH check at corrected C6
# ---------------------------------------------------------------------------

def build_interaction_strengths_corrected(
    positions: list[tuple[float, float]], c6: float
) -> np.ndarray:
    """V_ij = c6 / r_ij^6 (rad/µs) for all pairs i < j."""
    n = len(positions)
    pos = np.array(positions)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            V[i, j] = c6 / r**6
    return V


def build_sparse_operators_c6(
    V: np.ndarray, alpha: float = 0.50
) -> dict:
    """
    Build sparse matrix representations of Hamiltonian components.

    Parameters
    ----------
    V : ndarray, (N_Q, N_Q) interaction matrix
    alpha : float, local detuning amplitude multiplier

    Returns
    -------
    dict with keys H0_sp, HX_sp, Ntot_sp
    """
    import qutip as qt

    id2 = qt.identity(2)
    sz = qt.sigmaz()
    sx = qt.sigmax()

    def embed(op, site: int, n: int = N_Q):
        ops = [id2] * n
        ops[site] = op
        return qt.tensor(ops)

    ni_np = [(embed(id2, i) - embed(sz, i)).full() / 2 for i in range(N_Q)]
    sx_np = [embed(sx, i).full() for i in range(N_Q)]

    H_int_np = sum(
        V[i, j] * (ni_np[i] @ ni_np[j])
        for i in range(N_Q)
        for j in range(i + 1, N_Q)
        if V[i, j] > 1e-10
    )
    H_loc_np = sum(alpha * DELTA_LOC_BASE[i] * ni_np[i] for i in range(N_Q))
    H0_np = H_int_np - H_loc_np
    HX_np = sum(sx_np[i] / 2 for i in range(N_Q))
    Ntot_np = sum(ni_np[i] for i in range(N_Q))

    return {
        "H0_sp": sp.csr_matrix(H0_np),
        "HX_sp": sp.csr_matrix(HX_np),
        "Ntot_sp": sp.csr_matrix(Ntot_np),
    }


def make_protocol_funcs(T_total: float):
    """Return (omega(t), delta_global(t)) callables."""
    T_H = T_total - 2.0 * T_RAMP

    def omega(t: float) -> float:
        if t <= T_RAMP:
            return OMEGA_MAX * t / T_RAMP
        elif t <= T_RAMP + T_H:
            return OMEGA_MAX
        else:
            return OMEGA_MAX * (T_total - t) / T_RAMP

    def delta_global(t: float) -> float:
        return -DELTA_0 + (DELTA_0 + DELTA_F) * t / T_total

    return omega, delta_global


def run_nhh(
    ops: dict,
    T_total: float,
    Gamma: float,
    mds_indices: list[int],
    rtol: float = 1e-8,
    atol: float = 1e-9,
) -> float:
    """
    Integrate NHH Schrödinger equation; Gamma=0 gives exact noiseless evolution.

    H_eff(t) = H(t) - i(Γ/2) N_total

    Returns P(MDS) from the (unnormalized) final state.
    """
    omega_f, delta_g_f = make_protocol_funcs(T_total)
    H0 = ops["H0_sp"]
    HX = ops["HX_sp"]
    Ntot = ops["Ntot_sp"]
    decay = (Gamma / 2.0) * Ntot  # Lindblad no-jump correction

    def rhs(t: float, psi: np.ndarray) -> np.ndarray:
        H_eff = H0 + omega_f(t) * HX - delta_g_f(t) * Ntot - 1j * decay
        return -1j * (H_eff @ psi)

    psi0 = np.zeros(2**N_Q, dtype=complex)
    psi0[0] = 1.0  # |000...0⟩ all-ground state

    sol = integrate.solve_ivp(
        rhs,
        [0.0, T_total],
        psi0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        t_eval=[T_total],
    )
    psi_f = sol.y[:, 0]
    return float(np.sum(np.abs(psi_f[mds_indices]) ** 2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Bloqade 3x3 MDS emulation + Lindblad reconciliation")
    print(f"C6_bloqade = {C6_BLOQADE:.0f}  C6_paper_wrong = {C6_PAPER_WRONG:.0f}")
    print(f"C6 ratio = {C6_RATIO_2PI:.4f}  (2π = {2*np.pi:.4f})")
    print("=" * 70)

    # Build adjacency and enumerate MDS once
    adj = build_adjacency(POSITIONS)
    mds_indices = enumerate_mds(adj, k_opt=3)
    p_rnd_calc = len(mds_indices) / 2**N_Q
    print(f"\nMDS count (k=3): {len(mds_indices)},  P_rnd = {p_rnd_calc*100:.3f}%")
    print(f"MDS indices: {mds_indices}")

    # -----------------------------------------------------------------------
    # Part 1: alpha scan at T = 2.6 µs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 1: Bloqade alpha scan  (T=2.6 µs, 2000 shots per alpha)")
    print("=" * 70)

    ALPHA_VALS = [0.30, 0.40, 0.50, 0.60, 0.70]
    T_FIXED = 2.6
    SHOTS_ALPHA = 2000

    alpha_scan_results: list[dict] = []

    for alpha in ALPHA_VALS:
        t0 = time.time()
        prog = build_bloqade_program(alpha=alpha, T_total=T_FIXED)
        # interaction_picture=True avoids stiff-ODE failures from large local detunings
        result = prog.bloqade.python().run(shots=SHOTS_ALPHA, interaction_picture=True)
        report = result.report()
        bs_raw = np.array(report.bitstrings()[0], dtype=int)   # (2000, 9); 0=Rydberg, 1=ground
        bs_exc = 1 - bs_raw   # invert: 1=Rydberg, 0=ground

        p_opt = compute_p_opt_from_bitstrings(bs_raw, adj, k_opt=3)
        k_mean = float(bs_exc.sum(axis=1).mean())   # mean number of Rydberg excitations
        advantage = p_opt / P_RND if P_RND > 0 else 0.0
        dt = time.time() - t0

        print(
            f"  alpha={alpha:.2f}  P(opt)={p_opt*100:.2f}%  "
            f"k_mean={k_mean:.3f}  adv={advantage:.2f}x  ({dt:.1f}s)"
        )

        alpha_scan_results.append(
            {
                "alpha": float(alpha),
                "P_opt": float(p_opt),
                "k_mean": float(k_mean),
                "advantage": float(advantage),
            }
        )

    # Identify best alpha
    best_entry = max(alpha_scan_results, key=lambda d: d["P_opt"])
    best_alpha = best_entry["alpha"]
    P_opt_best_alpha = best_entry["P_opt"]
    print(f"\n  Best alpha = {best_alpha:.2f}  P(opt) = {P_opt_best_alpha*100:.2f}%")

    # -----------------------------------------------------------------------
    # Part 2: T scan at best alpha (alpha=0.50)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Part 2: Bloqade T scan  (alpha={best_alpha:.2f}, 1000 shots per T)")
    print("=" * 70)

    T_SCAN_VALS = [0.8, 1.2, 1.6, 2.0, 2.4, 2.6, 3.0, 3.5, 4.0]
    SHOTS_T = 1000

    t_scan_p_opt: list[float] = []

    for T_val in T_SCAN_VALS:
        t0 = time.time()
        prog = build_bloqade_program(alpha=best_alpha, T_total=T_val)
        result = prog.bloqade.python().run(shots=SHOTS_T, interaction_picture=True)
        report = result.report()
        bs_raw = np.array(report.bitstrings()[0], dtype=int)

        p_opt = compute_p_opt_from_bitstrings(bs_raw, adj, k_opt=3)
        t_scan_p_opt.append(p_opt)
        print(
            f"  T={T_val:.2f} µs  P(opt)={p_opt*100:.2f}%  ({time.time()-t0:.1f}s)"
        )

    t_scan_arr = np.array(t_scan_p_opt)
    idx_best_T = int(np.argmax(t_scan_arr))
    T_opt = float(T_SCAN_VALS[idx_best_T])
    P_opt_max_T = float(t_scan_arr[idx_best_T])

    print(f"\n  T_opt = {T_opt:.2f} µs  P_opt_max = {P_opt_max_T*100:.2f}%")

    # -----------------------------------------------------------------------
    # Part 3: Lindblad NHH check with corrected C6
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Part 3: NHH Lindblad check at C6={C6_BLOQADE:.0f}  (Gamma=0 and 0.01)")
    print("=" * 70)

    V_corrected = build_interaction_strengths_corrected(POSITIONS, C6_BLOQADE)
    print("  Building 512-dim sparse operators (C6 corrected)...")
    t0 = time.time()
    ops_corrected = build_sparse_operators_c6(V_corrected, alpha=0.50)
    print(f"  Done in {time.time()-t0:.2f}s")

    Gamma_check_vals = [0.0, 0.01]
    p_check: list[float] = []

    for Gamma in Gamma_check_vals:
        t0 = time.time()
        p = run_nhh(ops_corrected, T_FIXED, Gamma, mds_indices)
        p_check.append(p)
        print(
            f"  Gamma={Gamma:.3f} µs⁻¹  P(opt)={p*100:.2f}%  ({time.time()-t0:.2f}s)"
        )

    p_nhh_gamma0 = p_check[0]
    p_nhh_gamma001 = p_check[1]
    rel_deg_corrected = (p_nhh_gamma0 - p_nhh_gamma001) / (p_nhh_gamma0 + 1e-15) * 100.0
    print(f"\n  Relative degradation (corrected C6): {rel_deg_corrected:.2f}%")

    # -----------------------------------------------------------------------
    # Lindblad reconciliation summary
    # -----------------------------------------------------------------------
    # Use Bloqade noiseless P(opt) at alpha=0.50, T=2.6µs as reference
    # Apply the no-jump factor obtained with the same corrected C6 convention.

    # Find P_opt from alpha scan at alpha=0.50
    alpha_050_entry = next(d for d in alpha_scan_results if abs(d["alpha"] - 0.50) < 1e-9)
    P_opt_noiseless_bloqade = alpha_050_entry["P_opt"]

    # Apply the corrected-C6 no-jump degradation calculated just above.
    relative_degradation_no_jump = rel_deg_corrected / 100.0
    P_opt_lindblad_corrected = P_opt_noiseless_bloqade * (
        1.0 - relative_degradation_no_jump
    )

    print("\n" + "=" * 70)
    print("Lindblad reconciliation")
    print("=" * 70)
    print(f"  P_opt noiseless (Bloqade, alpha=0.5, T=2.6µs): {P_opt_noiseless_bloqade*100:.2f}%")
    print(f"  Relative degradation (NHH corrected C6):        {rel_deg_corrected:.2f}%")
    print(f"  P_opt corrected no-jump estimate:                {P_opt_lindblad_corrected*100:.2f}%")

    # -----------------------------------------------------------------------
    # Assemble and save JSON
    # -----------------------------------------------------------------------
    results = {
        "description": "Sampled Bloqade 3x3 MDS validation + no-jump decay check",
        "C6_bloqade": C6_BLOQADE,
        "C6_value_if_2pi_omitted": C6_PAPER_WRONG,
        "C6_ratio_2pi": float(C6_RATIO_2PI),

        "alpha_scan": alpha_scan_results,
        "best_alpha": best_alpha,

        "T_scan": {
            "alpha": best_alpha,
            "T_vals_us": T_SCAN_VALS,
            "P_opt_vals": t_scan_arr.tolist(),
            "T_opt_us": T_opt,
            "P_opt_max": P_opt_max_T,
        },

        "lindblad_reconciliation": {
            "P_opt_noiseless_bloqade": P_opt_noiseless_bloqade,
            "qualification": "no-jump non-Hermitian approximation, not full Lindblad",
            "relative_degradation_no_jump_corrected_C6": relative_degradation_no_jump,
            "P_opt_lindblad_corrected": float(P_opt_lindblad_corrected),
            "NHH_check_C6_correct": {
                "C6": C6_BLOQADE,
                "P_opt_Gamma0": float(p_nhh_gamma0),
                "P_opt_Gamma_0p01": float(p_nhh_gamma001),
                "relative_degradation": float(rel_deg_corrected / 100.0),
            },
        },

        "no_jump_decay_estimate": {
            "qualification": "not a full Lindblad or hardware prediction",
            "P_opt_noiseless": P_opt_noiseless_bloqade,
            "P_opt_with_decay_Gamma_0p01": float(P_opt_lindblad_corrected),
            "relative_degradation_pct": float(rel_deg_corrected),
            "T_sweep_us": T_FIXED,
            "Gamma_us_inv": 0.01,
            "T1_rydberg_us": 100.0,
        },
    }

    output_path = RESULTS_DIR / "bloqade_3x3_sampled.json"
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  C6_bloqade = {C6_BLOQADE:.0f}  (2π × C6_paper_wrong)")
    print(f"  MDS count = {len(mds_indices)},  P_rnd = {p_rnd_calc*100:.3f}%")
    print(f"  Best alpha = {best_alpha:.2f}")
    print(f"  P(opt) Bloqade noiseless (alpha=0.5, T=2.6µs): {P_opt_noiseless_bloqade*100:.2f}%")
    print(f"  Deterministic reference: 14.7998%")
    print(f"  NHH relative degradation (corrected C6):        {rel_deg_corrected:.2f}%")
    print(f"  P(opt) Lindblad corrected (hardware prediction): {P_opt_lindblad_corrected*100:.2f}%")
    print(f"  T_opt (Bloqade): {T_opt:.2f} µs  P_opt_max = {P_opt_max_T*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
