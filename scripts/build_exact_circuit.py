#!/usr/bin/env python3
"""Build and validate the faithful (inequality-respecting) 51-qubit encoding of Rung 1.

    OFFLINE ONLY.  This script never creates a Sampler, a Batch, or calls .run(),
    and it never submits a job.  The only optional network use is a read-only
    `QiskitRuntimeService().backend("ibm_kingston")` lookup, used solely to hand a
    real coupling-map `Target` to `transpile()` so the reported physical depth and
    2-qubit count are the honest, backend-aware ones.  If that lookup fails (no
    credentials, no network) the script falls back to a synthetic 156-qubit
    heavy-hex target and says so.  Nothing here executes on hardware.

The cost operator used for the archived hardware runs is the two-sided quadratic
surrogate H = sum_b x_b + lam sum_t (1 - sum_b A_tb x_b)^2, whose ground state is
NOT the minimum cover (Sec. IV of the manuscript; see also calc_exact_encoding.py).
This script builds the alternative that IS faithful to the inequality constraint
sum_b A_tb x_b >= 1: one binary slack register per target t,

    sum_b A_tb x_b - 1 - s_t = 0 ,   s_t in {0, ..., 2^{w_t} - 1} >= {0, ..., deg_t-1},
    w_t = ceil(log2(deg_t))  (0 if deg_t == 1: the constraint is already an equality).

Because slack registers are private to their own target and never shared, and
because a target's cover-set is a clique in the base-interaction graph, the
extended interaction graph has the same 13 connected components as the plain
24-qubit surrogate -- just larger ones (bases + slack).  That decomposition is
exploited below to verify the encoding's ground state EXACTLY, without ever
materialising a 2^51 statevector.

Writes results/exact_circuit_51q.json.
"""
from __future__ import annotations

import itertools
import json
from math import comb

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from _paths import DATA_DIR, RESULTS_DIR
from calc_exact_encoding import extended_graph, slack_widths


# ----------------------------------------------------------------------------
# 1. The faithful QUBO -> Ising, built directly from the penalty expansion.
# ----------------------------------------------------------------------------
def exact_qubo(A: np.ndarray, lam: float):
    """H(x, s) = sum_b x_b + lam * sum_t (sum_b A_tb x_b - 1 - s_t)^2, s_t binary-encoded.

    Returns (lin, quad, n, n_b, w, slack_offset, const): the linear/quadratic QUBO
    coefficients over the extended register y = (x_0..x_{n_b-1}, s_bits...), the
    extended register size n, the base count n_b, the per-target slack widths w,
    each target's slack starting offset (relative to n_b), and the constant term
    (lam * n_t, from the "+1" inside every squared bracket -- irrelevant to the
    argmin, kept for an honest absolute energy).
    """
    n_t, n_b = A.shape
    w = slack_widths(A)
    slack_offset = np.concatenate([[0], np.cumsum(w)])[:-1]
    n = n_b + int(sum(w))

    # C[t, :] is the linear form c_t such that L_t = c_t . y - 1.
    C = np.zeros((n_t, n))
    C[:, :n_b] = A
    for t in range(n_t):
        for j in range(w[t]):
            C[t, n_b + slack_offset[t] + j] = -float(2 ** j)

    lin = np.zeros(n)
    lin[:n_b] = 1.0                      # sum_b x_b: cardinality cost on bases only
    quad: dict[tuple[int, int], float] = {}
    for t in range(n_t):
        idx = np.nonzero(C[t])[0]
        for i in idx:
            lin[i] += lam * C[t, i] ** 2 - 2.0 * lam * C[t, i]     # (c.y)^2 diag - 2 c.y
        for a in range(len(idx)):
            for b_ in range(a + 1, len(idx)):
                i, j = int(idx[a]), int(idx[b_])
                c = 2.0 * lam * C[t, i] * C[t, j]
                if c != 0.0:
                    quad[(i, j)] = quad.get((i, j), 0.0) + c
    const = lam * n_t
    return lin, quad, n, n_b, w, slack_offset, const


def qubo_to_ising(lin: np.ndarray, quad: dict, n: int):
    """x_b = (1 - Z_b)/2, same convention as freeze_instances.quadratic_qubo_ising."""
    h = np.zeros(n)
    for i in range(n):
        h[i] = -lin[i] / 2.0
    J: dict[tuple[int, int], float] = {}
    for (i, j), c in quad.items():
        J[(i, j)] = c / 4.0
        h[i] += -c / 4.0
        h[j] += -c / 4.0
    return h, J


# ----------------------------------------------------------------------------
# 2. The p=1 QAOA circuit: product init, RZ+RZZ cost layer, RX mixer.
# ----------------------------------------------------------------------------
def qaoa_circuit_exact(n: int, h: np.ndarray, J: dict, gamma: float, beta: float,
                        measure: bool = True) -> QuantumCircuit:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, name="qaoa_exact51_p1")
    qc.h(range(n))
    for q in range(n):
        if h[q] != 0.0:
            qc.rz(2 * gamma * h[q], q)
    for (i, j), Jij in J.items():
        qc.rzz(2 * gamma * Jij, i, j)
    for q in range(n):
        qc.rx(2 * beta, q)
    if measure:
        qc.measure_all()
    return qc


# ----------------------------------------------------------------------------
# 3. Block decomposition (from the Ising J, which must match calc_exact_encoding).
# ----------------------------------------------------------------------------
def blocks_from_edges(n: int, edges) -> list[np.ndarray]:
    C = np.zeros((n, n), np.int8)
    for (i, j) in edges:
        C[i, j] = C[j, i] = 1
    nc, lab = connected_components(csr_matrix(C), directed=False)
    return [np.where(lab == c)[0] for c in range(nc)]


# ----------------------------------------------------------------------------
# 4. Ground-state verification, block by block (never a 2^51 statevector).
# ----------------------------------------------------------------------------
def block_local_minimizers(block: np.ndarray, lin: np.ndarray, quad: dict):
    """Exhaustive minimisation of the QUBO restricted to one block (<= ~13 qubits:
    2^13 = 8192 states).  Valid because the direct-sum property means no term in
    lin/quad connects two different blocks, so the block's local minimiser set,
    combined freely across blocks, gives every global minimiser."""
    m = len(block)
    pos = {int(q): t for t, q in enumerate(block)}
    local_quad = [(pos[i], pos[j], c) for (i, j), c in quad.items()
                  if i in pos and j in pos]
    local_lin = lin[block]
    bits = ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, :]) & 1).astype(float)
    energy = bits @ local_lin
    for i, j, c in local_quad:
        energy += c * bits[:, i] * bits[:, j]
    e_min = energy.min()
    winners = np.where(np.isclose(energy, e_min, atol=1e-9))[0]
    return float(e_min), bits[winners].astype(int)          # (min energy, (n_win, m) bit table)


def true_minimum_covers(A: np.ndarray, k_opt: int) -> set[tuple[int, ...]]:
    """Independent ground truth: enumerate every weight-k_opt combination directly
    from A (no slack, no QUBO) -- the same style as analyze_kingston.random_baselines."""
    n_t, n_b = A.shape
    covers = set()
    for ones in itertools.combinations(range(n_b), k_opt):
        x = np.zeros(n_b, int)
        x[list(ones)] = 1
        if (A @ x >= 1).all():
            covers.add(ones)
    return covers


# ----------------------------------------------------------------------------
def main() -> None:
    from qiskit import transpile
    from qiskit.quantum_info import Statevector

    d = json.load(open(DATA_DIR / "frozen_rung1.json"))
    A = np.asarray(d["A"], int)
    n_t, n_b = A.shape
    lam, k_opt = d["lam"], d["k_opt"]
    print(f"Rung 1: n_b={n_b} n_t={n_t} lam={lam} k_opt={k_opt}")

    # --- build the faithful QUBO/Ising -------------------------------------
    lin, quad, n, n_b_chk, w, slack_offset, const = exact_qubo(A, lam)
    assert n_b_chk == n_b
    h, J = qubo_to_ising(lin, quad, n)
    print(f"extended register: n={n} (n_b={n_b} + n_slack={n - n_b}), n_ZZ={len(quad)}")

    # --- (ii) cross-check against calc_exact_encoding.py's structural graph ----
    C_struct, w_struct = extended_graph(A)
    assert w_struct == w, "slack widths disagree with calc_exact_encoding.slack_widths"
    struct_edges = {(i, j) for i in range(n) for j in range(i + 1, n) if C_struct[i, j]}
    my_edges = set(quad.keys())
    assert my_edges == struct_edges, (
        f"QUBO coupling graph disagrees with calc_exact_encoding's structural graph: "
        f"{len(my_edges)} vs {len(struct_edges)} edges, "
        f"symmetric_difference={my_edges ^ struct_edges}")
    print(f"n_ZZ = {len(quad)} matches calc_exact_encoding.py (structural graph identical)")

    blocks = blocks_from_edges(n, quad.keys())
    # Singletons (isolated qubits, e.g. a deg-1 target's forced base) are components
    # of size 1 with no edges; connected_components still finds them correctly since
    # every node is present in the (n x n) graph regardless of degree.
    sizes = sorted((len(b) for b in blocks), reverse=True)
    print(f"blocks = {len(blocks)}, sizes = {sizes}")
    assert len(blocks) == 13 and sizes == [13, 9, 7, 6, 5, 4, 1, 1, 1, 1, 1, 1, 1], (
        "block count/sizes disagree with results/exact_encoding.json's rung1.exact block")

    # --- (i) ground state: block-wise exhaustive minimisation --------------
    block_mins = []
    for b in blocks:
        e_min, winners_bits = block_local_minimizers(b, lin, quad)
        block_mins.append((b, e_min, winners_bits))
    global_min_energy = float(sum(e for _, e, _ in block_mins)) + const
    print(f"global minimum energy (incl. constant {const}) = {global_min_energy}")

    # Assemble every global minimiser's x-projection (cartesian product across
    # blocks of each block's local winners), restricted to the base qubits (< n_b).
    per_block_x_choices = []
    for b, _, winners_bits in block_mins:
        base_pos_in_block = [t for t, q in enumerate(b) if q < n_b]
        base_qubits = [int(b[t]) for t in base_pos_in_block]
        choices = set()
        for row in winners_bits:
            x_partial = tuple(sorted(q for pos_, q in zip(base_pos_in_block, base_qubits)
                                     if row[pos_] == 1))
            choices.add(x_partial)
        per_block_x_choices.append((base_qubits, choices))

    encoded_covers = set()
    n_global_minimizers = 1
    for _, choices in per_block_x_choices:
        n_global_minimizers *= len(choices)
    # Cartesian product would be exact but might be large in principle; here it is
    # bounded by 288 (one slack pattern per feasible cover), so materialise it.
    combo_iter = itertools.product(*(choices for _, choices in per_block_x_choices))
    for combo in combo_iter:
        ones = tuple(sorted(idx for part in combo for idx in part))
        encoded_covers.add(ones)

    truth = true_minimum_covers(A, k_opt)
    print(f"encoded ground state -> {len(encoded_covers)} distinct covers "
          f"(cartesian product size {n_global_minimizers})")
    print(f"independent brute force (C({n_b},{k_opt})={comb(n_b, k_opt)}) "
          f"-> {len(truth)} minimum covers")
    assert n_global_minimizers == len(truth) == 288, (
        f"ground-state degeneracy {n_global_minimizers} != 288 minimum covers")
    assert encoded_covers == truth, (
        "FAITHFUL ENCODING FAILED: ground state of the implemented operator is not "
        "exactly the set of minimum covers")
    print("PASS: ground state of the 51-qubit faithful encoding IS exactly the 288 "
          "minimum covers (no infeasible states, no missing covers).")

    # --- (iii) the ideal p=1 state still factorises over blocks -------------
    # (a) exact, general proof at the CIRCUIT level: every 2-qubit gate must stay
    #     inside one block, for ANY n -- this is checked on the actual 51-qubit
    #     circuit, not on a tractable subset.
    gamma, beta = d["qaoa_angles"]["1"]["gamma"][0], d["qaoa_angles"]["1"]["beta"][0]
    qc51 = qaoa_circuit_exact(n, h, J, gamma, beta, measure=False)
    label_of = np.full(n, -1, int)
    for bi, b in enumerate(blocks):
        label_of[b] = bi
    cross_block_gates = [instr for instr in qc51.data if len(instr.qubits) == 2
                          and label_of[qc51.find_bit(instr.qubits[0]).index]
                          != label_of[qc51.find_bit(instr.qubits[1]).index]]
    assert not cross_block_gates, f"{len(cross_block_gates)} gates couple two blocks"
    print(f"PASS: all {sum(1 for i in qc51.data if len(i.qubits) == 2)} two-qubit gates "
          f"of the 51-qubit p=1 circuit stay within their block (proof holds for the "
          f"full 51 qubits: 2^51 is never materialised).")

    # (b) redundant numerical cross-check on a tractable union of blocks (the two
    #     largest, 13+9=22 qubits: 2^22 statevector, ~64 MB, easy) -- catches an
    #     indexing/coefficient bug that (a) alone would not.
    big_two = sorted(blocks, key=len, reverse=True)[:2]

    # Place block[0]'s qubits at the LOW circuit positions and block[1]'s at the HIGH
    # ones (both internally sorted ascending), then build the joint circuit using
    # that exact placement, and the "product" state as a kron chain in the SAME
    # convention Qiskit itself uses: qubit index = bit weight 2^index, so a HIGHER-
    # position block is the OUTER (more-significant) kron factor. Building the kron
    # chain as kron(new_block, running_product) while appending blocks in increasing
    # position order reproduces exactly that convention -- verified against the
    # 24-qubit closed-form check in calc_block_factorisation.py's own construction.
    order: list[int] = []
    psi_prod = np.ones(1, dtype=complex)
    for blk in big_two:
        blk_sorted = np.sort(blk)
        rmp = {int(q): t for t, q in enumerate(blk_sorted)}
        h_b = h[blk_sorted]
        J_b = {(rmp[i], rmp[j]): v for (i, j), v in J.items() if i in rmp and j in rmp}
        qc_b = qaoa_circuit_exact(len(blk_sorted), h_b, J_b, gamma, beta, measure=False)
        psi_b = Statevector(qc_b).data
        psi_prod = np.kron(psi_b, psi_prod)          # new block is OUTER (higher positions)
        order += list(blk_sorted)

    m_union = len(order)
    remap = {int(q): t for t, q in enumerate(order)}   # qc_joint's qubit t = global qubit order[t]
    h_u = h[np.array(order)]
    J_u = {(remap[i], remap[j]): v for (i, j), v in J.items()
           if i in remap and j in remap}
    qc_joint = qaoa_circuit_exact(m_union, h_u, J_u, gamma, beta, measure=False)
    psi_joint = Statevector(qc_joint).data

    fidelity = float(abs(np.vdot(psi_prod, psi_joint)) ** 2)
    print(f"numerical cross-check on the largest two blocks ({m_union} qubits): "
          f"|<product|joint>|^2 = {fidelity:.12f}")
    assert abs(fidelity - 1.0) < 1e-9, "block factorisation failed on the 22-qubit check"
    print("PASS: the ideal p=1 state factorises exactly over blocks (both the exact "
          "circuit-level proof and the 22-qubit numerical cross-check agree).")

    # --- transpile against ibm_kingston's target, WITHOUT running -----------
    qc51_meas = qaoa_circuit_exact(n, h, J, gamma, beta, measure=True)
    logical = transpile(qc51_meas, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
    print(f"logical (basis-gate) depth={logical.depth()} "
          f"2q={logical.count_ops().get('cx', 0)}")

    backend_name, backend_source = None, None
    physical = None
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend("ibm_kingston")            # READ-ONLY metadata lookup
        backend_name = backend.name
        backend_source = "live ibm_kingston target (read-only lookup, no job)"
        physical = transpile(qc51_meas, backend=backend, optimization_level=3)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  (offline: could not reach ibm_kingston target: {exc!r}; "
              f"falling back to a synthetic heavy-hex 156q target)")
        from qiskit.transpiler import CouplingMap
        cmap = CouplingMap.from_heavy_hex(5)                  # 156-ish qubit heavy-hex
        backend_name = "synthetic heavy-hex (offline fallback)"
        backend_source = "no credentials/network: NOT ibm_kingston's real calibration"
        physical = transpile(qc51_meas, coupling_map=cmap, basis_gates=["cx", "rz", "sx", "x"],
                             optimization_level=3)

    n2q = physical.count_ops().get("cx", 0) + physical.count_ops().get("cz", 0) + \
          physical.count_ops().get("ecr", 0)
    print(f"physical ({backend_name}): depth={physical.depth()} 2q={n2q}  [{backend_source}]")
    print("\nNO circuit was submitted. This script builds, verifies, and transpiles only.")

    out = dict(
        n_b=n_b, n_t=n_t, lam=lam, k_opt=k_opt,
        n_qubits=n, n_slack=n - n_b, n_ZZ=len(quad),
        blocks=len(blocks), block_sizes=sizes,
        ground_state_n_covers=len(encoded_covers), ground_state_matches_truth=True,
        global_min_energy=global_min_energy,
        angles_used=dict(gamma=gamma, beta=beta),
        logical=dict(depth=logical.depth(), n2q=logical.count_ops().get("cx", 0)),
        physical=dict(backend=backend_name, source=backend_source,
                      depth=physical.depth(), n2q=int(n2q)),
        block_factorisation_fidelity_22q=fidelity,
    )
    json.dump(out, open(RESULTS_DIR / "exact_circuit_51q.json", "w"), indent=1)
    print("-> results/exact_circuit_51q.json")


if __name__ == "__main__":
    main()
