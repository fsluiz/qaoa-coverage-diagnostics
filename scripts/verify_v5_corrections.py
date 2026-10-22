#!/usr/bin/env python3
"""Focused scientific and editorial gate for the corrected version-5 manuscript."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    checks.append((name, bool(condition), detail))


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


main = (PAPER / "paper_v5.tex").read_text(encoding="utf-8")
supp = (PAPER / "supplement_v5.tex").read_text(encoding="utf-8")
tex = main + "\n" + supp
workflow = (ROOT / "scripts" / "reproduce_v5.py").read_text(encoding="utf-8")

# QWOA: analytical block-cut spectrum and one unambiguous global-mixer convention.
qwoa = load("qwoa_simulability_v5.json")
check("QWOA audit uses the analytical Dicke-state spectrum",
      qwoa["method"] == "analytical_hypergeometric_schmidt_spectrum")
check("QWOA mixer is identified as exact global continuous-time evolution",
      qwoa["mixer_evolution"] == "global_continuous_time_exact_sector_spectral"
      and qwoa["hamiltonian_prefactor"] == 0.5)
check("rank and entropy maxima are recorded separately",
      qwoa["maximum_block_cut_rank"] == qwoa["bond_dimension_bound"] == 11
      and math.isclose(qwoa["maximum_block_cut_entropy_bits"], 2.3496865381073353,
                       abs_tol=1e-13)
      and qwoa["maximum_entropy_cut_after_block"] == 6)
check("QWOA state-space proxy is 682 rather than the full 2^24 space",
      qwoa["mps_simulation_cost"] == 682 and qwoa["naive_cost"] == 2 ** 24)
check("manuscript reports 2.35 bits and excludes a digital-QWOA execution claim",
      "2.35" in tex and "no digital QWOA circuit was compiled or executed" in tex)

# Geometry: both published sweeps must be valid 1-Lipschitz coordinates.
ladder = load("ladder_rungs_v5.json")["rung3"]
strip = ladder["strip"]
check("signed great-circle projections preserve the published strip occupancies",
      strip["north_south"]["worst_strip"] == 228
      and strip["east_west"]["worst_strip"] == 259)
check("both strip projections pass the numerical nonexpansivity audit",
      max(row["maximum_nonexpansivity_violation_km"] for row in strip.values()) <= 1e-9)
check("uniform-outline control is deterministic after the projection correction",
      ladder["uniform_in_outline"]["seed"] == 20260930
      and math.isclose(ladder["uniform_in_outline"]["strip_bound_mean"], 148.3,
                       abs_tol=1e-12)
      and ladder["uniform_in_outline"]["strip_bound_max"] == 172)
check("treewidth bracket is not promoted to a runtime lower bound",
      ladder["treewidth_lb"] == 112 and ladder["treewidth_ub"] == 151
      and "complexity lower bound" in main)

# Interleaved job: rounds are the inferential units; shot intervals are conditional.
interleaved = load("interleaved_batch_v5.json")
expected = {
    "A": (0.8771293502636498, [0.8160200292734560, 0.9428149671509033],
          [0.8516124910719416, 0.9037000103658132],
          [0.6694881125719797, 1.3069959277556447]),
    "B": (0.6713235499160577, [0.6617159579263774, 0.6810706365374368],
          [0.6502938055335907, 0.6920737666915947],
          [0.6989979586155427, 1.5289730178605723]),
}
for family, (point, interval, shot_interval, cross_interval) in expected.items():
    row = interleaved["families"][family]
    check(f"family {family}: primary estimate and round-level t interval",
          math.isclose(row["gated_round_geometric_mean"], point, abs_tol=1e-13)
          and np.allclose(row["gated_round_t_ci95"], interval, atol=1e-13)
          and row["n_rounds"] == 4 and row["round_t_df"] == 3)
    check(f"family {family}: shot bootstrap is explicitly conditional",
          "conditional" in row["shot_bootstrap_assumption"]
          and "gated_shot_bootstrap_ci95_conditional" in row
          and not row["twirling_strata_archived"]
          and np.allclose(row["gated_shot_bootstrap_ci95_conditional"],
                          shot_interval, atol=1e-13)
          and np.allclose(row["cross_block_factor_shot_bootstrap_ci95_conditional"],
                          cross_interval, atol=1e-13))
check("paper uses the round-level intervals as primary",
      "[0.816,0.943]" in main and "[0.662,0.681]" in main
      and "conditional shot-bootstrap" in main)
check("manuscript conditional intervals match the v5 RNG streams",
      "[0.852,0.904]" in tex and "[0.650,0.692]" in tex
      and "[0.67,1.31]" in supp and "[0.70,1.53]" in supp
      and "[0.652,0.693]" not in tex and "[0.69,1.51]" not in tex)

# Encoding and mathematical scope.
check("base-protection equation uses the closed neighborhood once",
      "\\sum_{b'\\in N[b]}x_{b'}\\ge m" in main
      and "N[b]=N(b)\\cup\\{b\\}" in main)
check("sparsification law is restricted to the bare-register graph",
      "E_{ZZ}^{\\rm bare}" in main and "n_{ZZ}^{\\rm slack}" in main)
check("finite-penalty faithfulness is qualified",
      "Ground-state faithfulness also depends on the penalty weight" in main)
check("interaction graph is described as a subgraph of the proximity graph",
      "is only a subgraph of it" in main)
check("two-local no-go statement is constraint-local rather than universal",
      "standard constraint-local faithful penalty" in main
      and "does not rule out a dense, instance-specific quadratic Hamiltonian" in main)
encoding = load("exact_encoding.json")["instances"]
check("lambda=10 faithfulness is archived for both frozen instances",
      encoding["rung1"]["faithfulness_lambda_10"] == {
          **encoding["rung1"]["faithfulness_lambda_10"],
          "ground_energy": 10,
          "ground_degeneracy": 288,
          "all_ground_states_feasible": True,
      }
      and encoding["arco"]["faithfulness_lambda_10"] == {
          **encoding["arco"]["faithfulness_lambda_10"],
          "ground_energy": 31,
          "ground_degeneracy": 4,
          "all_ground_states_feasible": True,
      })

# Scope and workflow: remove the invalid temporal proxy only from the active chain.
check("persistent surveillance is explicitly outside scope",
      "none of its application-level conclusions extends to persistent surveillance" in tex
      and "Constructing and analyzing that model is outside the scope" in tex)
check("active v5 workflow excludes the persistent-surveillance proxy",
      'py("calc_persistent_surveillance.py")' not in workflow)
check("active v5 workflow uses only suffixed corrected analyses", all(
    name in workflow for name in (
        "calc_qwoa_simulability_v5.py",
        "calc_ladder_rungs_v5.py",
        "calc_interleaved_batch_v5.py",
    )
))
check("both broad claim verifiers run explicitly in version-5 mode",
      'py("verify_paper_claims.py", "--version", "5")' in workflow
      and 'py("verify_research_audit.py", "--version", "5")' in workflow
      and 'py("verify_v5_corrections.py")' in workflow)
check("site-loss section is explicitly a static sensitivity analysis",
      "Static sensitivity to candidate-site loss and link removal" in supp
      and "not operational recommendations" in supp
      and "Tactical layer:" not in supp)
check("IBM interpretations retain exploratory and noncausal qualifiers",
      "exploratory association" in main
      and "neither bounds nor attributes a joint-execution penalty" in supp)
for forbidden in (
    "collapses to 12, 7 and finally 2 components",
    "six of the 23 waypoints",
    "3/3003",
    "four orders of magnitude in $\\nzz$",
):
    check(f"retired claim absent: {forbidden}", forbidden not in tex)


width = max(len(name) for name, _, _ in checks)
failed = 0
for name, passed, detail in checks:
    print(f"[{'PASS' if passed else 'FAIL'}] {name:{width}}" +
          (f"  {detail}" if detail else ""))
    failed += not passed
print(f"\n{len(checks) - failed}/{len(checks)} version-5 checks passed")
sys.exit(1 if failed else 0)
