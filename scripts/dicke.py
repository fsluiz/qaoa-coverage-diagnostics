#!/usr/bin/env python3
"""
Exact deterministic Dicke-state preparation |D^n_k> = C(n,k)^{-1/2} sum_{|x|=k}|x>,
Baertschi & Eidenbenz, "Deterministic Preparation of Dicke States" (FCT 2019,
arXiv:1904.07358).  O(k(n-k)) two-qubit gates, no ancillae.

Self-test (`python dicke.py`) compares the prepared statevector to the ideal Dicke
state for many (n,k) and reports max fidelity error.  Used by submit_kingston.py for
the QWOA initial state.
"""
import numpy as np
from itertools import combinations
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate


def _ccry(qc, theta, c1, c2, t):
    qc.append(RYGate(theta).control(2), [c1, c2, t])


def _scs(qc, l, k):
    """Split & Cyclic-Shift block SCS_{l,k} on active qubits 0..l-1."""
    # i = 1: two-qubit block on (l-1, l-2)
    qc.cx(l - 2, l - 1)
    qc.cry(2.0 * np.arccos(np.sqrt(1.0 / l)), l - 1, l - 2)
    qc.cx(l - 2, l - 1)
    # i = 2..k: three-qubit blocks on (l-1, l-i, l-1-i)
    for i in range(2, k + 1):
        qc.cx(l - 1 - i, l - 1)
        _ccry(qc, 2.0 * np.arccos(np.sqrt(i / l)), l - 1, l - i, l - 1 - i)
        qc.cx(l - 1 - i, l - 1)


def dicke_circuit(n, k):
    """QuantumCircuit preparing |D^n_k> from |0>^n.  Uses the complement symmetry
    |D^n_k> = X^{otimes n}|D^n_{n-k}> so the cost scales as O(min(k,n-k)(n-...)):
    high-cardinality sectors (k near n) are CHEAP to prepare."""
    if not (0 <= k <= n):
        raise ValueError("require 0 <= k <= n")
    qc = QuantumCircuit(n, name=f"Dicke_{n}_{k}")
    if k == 0:
        return qc
    if k == n:
        qc.x(range(n))
        return qc
    kk = k if k <= n // 2 else n - k    # prepare the smaller weight
    for i in range(n - kk, n):          # initialise |0^{n-kk} 1^kk>
        qc.x(i)
    for l in range(n, 1, -1):           # U_{n,kk} = prod_{l=n..2} SCS_{l, min(kk,l-1)}
        _scs(qc, l, min(kk, l - 1))
    if kk != k:                         # flip to the complement weight
        qc.x(range(n))
    return qc


def ideal_dicke(n, k):
    import numpy as np
    psi = np.zeros(2 ** n, dtype=complex)
    for ones in combinations(range(n), k):
        idx = sum(1 << b for b in ones)
        psi[idx] = 1.0
    return psi / np.linalg.norm(psi)


def self_test(verbose=True):
    from qiskit.quantum_info import Statevector
    worst = 0.0
    cases = [(n, k) for n in range(2, 9) for k in range(0, n + 1)]
    for n, k in cases:
        sv = Statevector(dicke_circuit(n, k)).data
        ideal = ideal_dicke(n, k)
        fid = abs(np.vdot(ideal, sv)) ** 2
        err = abs(1.0 - fid)
        worst = max(worst, err)
        if err > 1e-9 and verbose:
            print(f"  FAIL n={n} k={k}: fidelity={fid:.6f}")
    if verbose:
        print(f"self-test: {len(cases)} cases, worst 1-fidelity = {worst:.2e}  "
              f"-> {'PASS' if worst < 1e-9 else 'FAIL'}")
    return worst < 1e-9


if __name__ == "__main__":
    ok = self_test()
    # report CX cost for the instances we will run
    from qiskit import transpile
    for n, k in [(14, 9), (14, 10), (14, 11), (14, 12), (24, 10)]:
        qc = dicke_circuit(n, k)
        dec = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
        print(f"  Dicke({n},{k}): logical 2q (pre-routing) CX = "
              f"{dec.count_ops().get('cx', 0)}, depth = {dec.depth()}")
