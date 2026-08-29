"""Corrected no-jump decay-rate scan for the 3x3 Rydberg MDS instance.

Despite the historical filename, this is not a full Lindblad master-equation
solver. It uses H_eff = H - i Gamma N/2 and labels the result accordingly.
"""
from __future__ import annotations

import json

from _paths import RESULTS_DIR
from bloqade_3x3_sim import POSITIONS, build_adjacency, enumerate_mds
from rydberg_exact_sim import PAPER_ALPHA, PAPER_T_US, build_operators, evolve


GAMMA_VALUES_US_INV = [0.0, 0.005, 0.01, 0.02]


def main() -> None:
    mds_indices = enumerate_mds(build_adjacency(POSITIONS), k_opt=3)
    operators = build_operators(PAPER_ALPHA)
    scan = []
    for gamma in GAMMA_VALUES_US_INV:
        row = {
            "Gamma_us_inv": gamma,
            **evolve(operators, PAPER_T_US, mds_indices, gamma=gamma),
        }
        scan.append(row)
        print(
            f"Gamma={gamma:.3f} us^-1: P(opt)={100 * row['P_opt']:.6f}%, "
            f"conditional k_mean={row['k_mean']:.6f}, norm={row['state_norm']:.6f}"
        )

    p0 = scan[0]["P_opt"]
    for row in scan:
        row["relative_P_opt_degradation"] = 1.0 - row["P_opt"] / p0

    output = {
        "description": "Corrected no-jump non-Hermitian Rydberg decay scan",
        "qualification": (
            "Approximation omits Lindblad recycling terms and is not a full "
            "master-equation or hardware prediction"
        ),
        "alpha": PAPER_ALPHA,
        "T_us": PAPER_T_US,
        "scan": scan,
    }
    output_path = RESULTS_DIR / "rydberg_decay_scan.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
