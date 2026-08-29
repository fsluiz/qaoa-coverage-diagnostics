#!/usr/bin/env python3
"""Check paper-v2 numerical claims against the archived JSON evidence.

The default mode reports known provenance conflicts as warnings and exits zero.
Use ``--strict`` to make warnings fail, which is appropriate before publication.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from _paths import DATA_DIR, RESULTS_DIR


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def close(actual: float, expected: float, tolerance: float = 5e-4) -> bool:
    return math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero status when known provenance warnings remain",
    )
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []

    def check(condition: bool, label: str) -> None:
        (passes if condition else failures).append(label)

    ladder = load_json(RESULTS_DIR / "m_ladder_results.json")
    rows = {int(row["R_km"]): row for row in ladder["sweep"]}
    expected_ladder = {
        200: (1, 17, 10, None, None, 12),
        300: (1, 29, 8, None, None, 10),
        400: (1, 65, 6, None, None, 6),
        500: (2, 98, 4, 9, None, 5),
        600: (3, 121, 3, 7, 11, 5),
    }
    for radius, expected in expected_ladder.items():
        row = rows[radius]
        task = row["ladder"]["T_taskcover"]
        protected = row["ladder"]["TP_base_protected"]
        actual = (
            row["min_task_deg"],
            row["n_zz_taskcover"],
            task["m=1"]["k_opt"],
            task["m=2"]["k_opt"],
            task["m=3"]["k_opt"],
            protected["m=1"]["k_opt"],
        )
        check(actual == expected, f"Table I range-ladder row R={radius} km")

    resilience = ladder["resilience"]
    check(
        resilience["m=1_at_R200"]["worst_uncovered_after_1_failure"] == 5
        and resilience["m=1_at_R200"]["n_single_failures_that_open_a_hole"] == 10,
        "Tactical m=1 single-failure claim at R=200 km",
    )
    check(
        resilience["m=2_at_threshold"]["R_km"] == 500
        and resilience["m=2_at_threshold"]["k_opt"] == 9
        and resilience["m=2_at_threshold"]["worst_uncovered_after_1_failure"] == 0,
        "Tactical m=2 resilience claim at R=500 km",
    )

    path_b = load_json(RESULTS_DIR / "path_b_results.json")["summary"]
    check(
        close(path_b["noiseless_ratio"]["mean"], 11.5, 0.06)
        and close(path_b["noiseless_ratio"]["std"], 12.0, 0.06),
        "Table IV equal-depth QWOA/QAOA ratio",
    )
    check(
        close(path_b["eqcx_ratio"]["mean"], 12.7, 0.06)
        and close(path_b["eqcx_ratio"]["std"], 12.6, 0.06),
        "Table IV equal-CX QWOA/QAOA ratio",
    )
    check(
        close(path_b["FP_ratio"]["mean"], 2.8, 0.06)
        and close(path_b["FP_ratio"]["std"], 3.2, 0.06),
        "Table IV hardware-adjusted ratio",
    )

    strong = load_json(RESULTS_DIR / "compare_strong_results.json")["headline"]
    check(close(strong["qwoa_direct"], 0.139, 5e-4), "Fig. 5 QWOA-direct P(opt)")
    check(close(strong["qwoa_descent"], 0.118, 5e-4), "Fig. 5 QWOA-descent P(opt)")
    check(close(strong["qaoa"], 1.8e-4, 5e-6), "Fig. 5 best QAOA P(opt)")

    kingston = load_json(RESULTS_DIR / "kingston_qaoa_results.json")
    check(
        kingston["n_b"] == 24
        and kingston["n_zz"] == 21
        and kingston["depth_transpiled"] == 64
        and kingston["n2q_transpiled"] == 53,
        "Kingston Rung-1 circuit resources",
    )
    check(
        close(kingston["p_real_sc"], 1.46e-3, 5e-6)
        and close(kingston["E_reduction_pct"], 24.0, 0.02),
        "Kingston Rung-1 probability and energy reduction",
    )

    arco = load_json(RESULTS_DIR / "kingston_arco_fogo_results.json")["configs"]["E2a"]
    check(
        arco["n_ZZ"] == 9
        and close(arco["p_cov_real"], 0.0354, 5e-5)
        and close(arco["E_reduction_pct"], 35.9, 0.04),
        "Kingston Rung-2 probability and energy reduction",
    )

    rydberg = load_json(RESULTS_DIR / "rydberg_3x3_exact.json")
    rydberg_point = rydberg["paper_operating_point"]
    check(
        close(rydberg_point["P_opt"], 0.1479979992, 1e-7)
        and close(rydberg_point["k_mean"], 1.7646366247, 1e-7)
        and close(rydberg["uniform_random_baseline"], 10 / 512, 1e-12)
        and close(rydberg_point["concentration_over_uniform"], 7.5774975614, 1e-7),
        "Corrected deterministic Rydberg operating point",
    )
    check(
        rydberg["best_alpha_on_scan"] == 0.5
        and rydberg["best_time_on_scan_us"] == 3.0
        and close(rydberg["best_time_on_scan_P_opt"], 0.3633255495, 1e-7),
        "Corrected deterministic Rydberg scan optima",
    )
    check(
        close(rydberg["C6_rad_per_us_um6"], 2 * math.pi * 862_690, 1e-6)
        and close(rydberg["C6_correction_factor"], 2 * math.pi, 1e-4),
        "Corrected Rydberg C6 convention",
    )
    decay = load_json(RESULTS_DIR / "rydberg_decay_scan.json")
    decay_gamma_001 = next(
        row for row in decay["scan"] if row["Gamma_us_inv"] == 0.01
    )
    check(
        close(decay_gamma_001["P_opt"], 0.1413357210, 1e-7)
        and close(decay_gamma_001["relative_P_opt_degradation"], 0.0450160018, 1e-7),
        "Corrected no-jump Rydberg decay check",
    )

    provenance = load_json(DATA_DIR / "ibge_amazonia_legal_2024_provenance.json")
    check(
        provenance["n_municipalities"] == 773
        and sum(provenance["state_counts"].values()) == 773
        and provenance["state_counts"]["MT"] == 142,
        "Official IBGE 2024 Legal Amazon municipality set",
    )
    geographic = load_json(
        RESULTS_DIR / "ilp_geographic_ibge_2024_centroids.json"
    )
    geo_200 = next(row for row in geographic["sweep"] if row["R_km"] == 200)
    check(
        geographic["n_nodes"] == 773
        and geo_200["status"] == "optimal"
        and geo_200["k_opt"] == 47
        and geo_200["n_ZZ"] == 39_974,
        "Corrected IBGE centroid MDS result at R=200 km",
    )
    geographic_seats = load_json(
        RESULTS_DIR / "ilp_geographic_ibge_2024_seats.json"
    )
    seat_200 = next(
        row for row in geographic_seats["sweep"] if row["R_km"] == 200
    )
    check(
        geographic_seats["n_nodes"] == 773
        and seat_200["status"] == "optimal"
        and seat_200["k_opt"] == 44
        and seat_200["n_ZZ"] == 39_631,
        "IBGE municipal-seat sensitivity result at R=200 km",
    )

    for label in passes:
        print(f"PASS  {label}")
    for label in failures:
        print(f"FAIL  {label}")
    for label in warnings:
        print(f"WARN  {label}")

    print(
        f"\nSummary: {len(passes)} passed, {len(failures)} failed, "
        f"{len(warnings)} provenance warnings."
    )
    return 1 if failures or (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
