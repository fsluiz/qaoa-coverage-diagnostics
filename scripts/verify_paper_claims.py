#!/usr/bin/env python3
"""Check a versioned manuscript's numerical claims against archived evidence.

Every check below refers to a number or scientific qualification in the active
manuscript.  The default preserves the frozen version-4 audit. ``--version 5``
reads the corrected manuscript and corrected v5 outputs while retaining the
broad evidence checks.

  python verify_paper_claims.py     # exits non-zero if any check fails
"""
from __future__ import annotations

import json
import argparse
import re
import sys

import numpy as np

from _paths import DATA_DIR, RESULTS_DIR

_parser = argparse.ArgumentParser()
_parser.add_argument("--version", type=int, choices=(4, 5), default=4)
_args = _parser.parse_args()
V5 = _args.version == 5
MAIN_TEX = f"paper_v{_args.version}.tex"
SUPP_TEX = f"supplement_v{_args.version}.tex"

checks: list[tuple[str, bool, str]] = []


_missing: list = []


def chk(name, ok, detail=""):
    if not ok and _missing:
        detail = (detail + " " if detail else "") + "missing: " + " | ".join(_missing)
    _missing.clear()
    checks.append((name, bool(ok), detail))


def close(a, b, tol):
    return abs(a - b) <= tol


J = lambda f: json.load(open(RESULTS_DIR / f))
D = lambda f: json.load(open(DATA_DIR / f))

# ---- frozen instances -------------------------------------------------------
r1, ar, ov = D("frozen_rung1.json"), D("frozen_arco.json"), D("frozen_overshoot.json")
chk("Rung-1 instance: n_b=24, n_t=23, links=51, n_ZZ=17, k_opt=10",
    (r1["n_b"], r1["n_t"], r1["links"], r1["n_ZZ"], r1["k_opt"]) == (24, 23, 51, 17, 10))
chk("Rung-2 instance: n_b=79, n_t=49, links=59, n_ZZ=9, k_opt=31",
    (ar["n_b"], ar["n_t"], ar["links"], ar["n_ZZ"], ar["k_opt"]) == (79, 49, 59, 9, 31))
A = np.asarray(ar["A"], int)
deg = A.sum(1)
chk("Rung-2 degree structure is 39 x deg-1 + 10 x deg-2",
    sorted(np.bincount(deg)[1:].tolist()) == sorted([39, 10]))

# ---- the squeeze ------------------------------------------------------------
tv = J("triviality.json")
sweep = {r["R_km"]: r for r in tv["pef_R_sweep"]}
chk("phi falls 0.40 -> 0.00 as R: 200 -> 600 km",
    close(sweep[200]["frac_forced"], 0.40, 1e-3) and sweep[600]["frac_forced"] == 0.0,
    f"{sweep[200]['frac_forced']} -> {sweep[600]['frac_forced']}")
chk("n_ZZ rises 17 -> 121 over the same sweep",
    sweep[200]["n_ZZ"] == 17 and sweep[600]["n_ZZ"] == 121)
arc = [r for r in tv["instances"] if r["name"].startswith("Rung 2")][0]
chk("Rung 2: phi = 0.94, residual 3 targets x 4 free bases",
    close(arc["frac_forced"], 0.935, 0.01) and (arc["residual_t"], arc["residual_b"]) == (3, 4))

# ---- hardware: archived raw counts and coherent statistical estimands --------
kr = {c["tag"]: c for c in J("kingston_100k_rung1.json")["circuits"]}
ka = {c["tag"]: c for c in J("kingston_100k_arco_optangles.json")["circuits"]}
kt = J("kingston_transferred_arco.json")["circuits"][0]
ex51 = J("exact51_results.json")
p1 = kr["rung1_qaoa_p1"]
sa = J("statistical_audit.json")
sa_r1 = {c["tag"]: c for c in sa["rung1"]["circuits"]}
sa_p1 = sa_r1["rung1_qaoa_p1"]
sa_51 = sa["faithful_51q"]
chk("Rung-1 p=1: 79/100000 optima -> P_hw = 7.90e-4",
    close(p1["p_opt"], 79 / 100000, 1e-12))
chk("Rung-1 p=1: coherent all-register eta = 46.02",
    close(sa_p1["eta_all_register"], 46.0208, 1e-3)
    and np.allclose(sa_p1["eta_all_register_wilson95"], [36.9326, 57.3433], atol=1e-3))
chk("Rung-1 p=1: coherent conditional-k eta = 35.05",
    sa_p1["n_in_optimal_sector"] == 15351
    and close(sa_p1["eta_conditional_k_opt"], 35.0455, 1e-3))
chk("Rung-1 depth series uses one full-register sample space",
    all(close(sa_r1[f"rung1_qaoa_p{p}"]["eta_all_register"], v, 0.01)
        for p, v in ((1, 46.02), (2, 27.96), (3, 15.15))))
chk("ideal-null deficit uses the exact binomial null, not sqrt(observed)",
    close(sa_p1["ideal_null_test"]["one_sided_exact_binomial_p"], 4.3628387e-18, 1e-24)
    and close(sa_p1["ideal_null_test"]["one_sided_normal_equivalent_z"], 8.58962, 1e-4))
chk("energy reduction 31.9 / 8.0 / 3.9 % at p = 1 / 2 / 3",
    all(close(kr[f"rung1_qaoa_p{p}"]["E_reduction_pct"], v, 0.06)
        for p, v in ((1, 31.9), (2, 8.0), (3, 3.9))))
chk("Rung 2 optimum remains unobserved with transferred and own angles",
    kt["p_opt"] == 0.0 and all(ka[f"arco_qaoa_p{p}"]["p_opt"] == 0.0
                               for p in (1, 2, 3)))
chk("Rung 2 energy reduction changes 27.2% -> 32.0% with its own angles",
    close(kt["E_reduction_pct"], 27.2, 0.06)
    and close(ka["arco_qaoa_p1"]["E_reduction_pct"], 32.0, 0.06))
chk("51q exact encoding: 39.7% measured vs 48.6% ideal energy reduction",
    close(ex51["reduction_pct"], 39.7, 0.06)
    and close(ex51["ideal_reduction_pct"], 48.6, 0.06))
chk("51q ideal optimum probability is calculated and archived",
    close(ex51["p_opt_ideal_max"], 8.610245012e-9, 1e-15)
    and ex51["p_opt_ideal_at_energy_angles"] < 1e-18)
chk("51q cover diagnostic uses a cardinality-matched null",
    close(sa_51["post_bits_form_cover"]["probability"], 0.53569, 1e-12)
    and close(sa_51["post_bits_form_cover"]["cardinality_matched_uniform_within_sector_baseline"],
              0.4470232, 1e-7)
    and close(sa_51["post_bits_form_cover"]["cardinality_matched_enhancement"], 1.1983494, 1e-7))
chk("51q: no measured string satisfies all encoded equalities",
    sa_51["all_measured_slack_equalities_satisfied"]["count"] == 0
    and close(sa_51["all_measured_slack_equalities_satisfied"]["wilson95"][1],
              3.8413113e-5, 1e-11))

# ---- Rydberg: encoding audit, not a positive optimization result -------------
rya = J("rydberg_encoding_audit.json")
ry_cases = {c["spacing_um"]: c for c in rya["cases"]}
r5, r7 = ry_cases[5.0], ry_cases[7.0]
chk("Rydberg blockade excludes valid grid-MDS states",
    r5["mds_vs_blockade"]["n_blockade_compatible_mds"] == 4
    and r7["mds_vs_blockade"]["n_blockade_compatible_mds"] == 8
    and not r7["mds_vs_blockade"]["all_mds_are_blockade_allowed"])
chk("Rydberg final ground manifolds contain no MDS and no dominating state",
    all(c["final_classical_hamiltonian"]["n_ground_states_that_are_grid_mds"] == 0
        and c["final_classical_hamiltonian"]["n_ground_states_that_dominate_grid"] == 0
        for c in (r5, r7)))
chk("reported d=7 overlap is reproduced but is not ground-state success",
    close(r7["dynamics_at_reported_duration"]["P_all_grid_mds"], 0.31273704, 1e-8)
    and close(r7["dynamics_at_reported_duration"]["P_final_hamiltonian_ground_manifold"],
              0.00387024, 1e-8)
    and r7["final_classical_hamiltonian"]["ground_bit_sets"] == [[0, 2, 6, 8]])

# ---- ILP --------------------------------------------------------------------
il = J("ilp_symmetry_corrected.json")
chk("invalid equal-degree cut returns 532 vs the true gamma = 141",
    il["mode_B_invalid"]["k_returned"] == 532
    and il["window_R_over_delta_1"]["closed_form_gamma"] == 141)
chk("Aut(P_n box P_n) = D4, order 8", il["aut_group"]["order"] == 8)

# ---- exact per-coupler / per-qubit resource model ----------------------------
rx = {c["tag"]: c for c in J("resource_model_exact.json")["circuits"]}
chk("exact model: new Rung 1 F = 0.817 (24q, 49 gates)",
    rx["rung1_qaoa_p1"]["n_2Q"] == 49
    and close(rx["rung1_qaoa_p1"]["F_exact"], 0.8167, 0.001))
chk("exact model: own-angle Rung 2 F = 0.573 (79q, 18 gates)",
    rx["arco_qaoa_p1"]["n_2Q"] == 18
    and close(rx["arco_qaoa_p1"]["F_exact"], 0.5731, 0.001))
chk("exact model: gate factor favours Rung 2, readout reverses it",
    rx["arco_qaoa_p1"]["F_gates"] > rx["rung1_qaoa_p1"]["F_gates"]
    and rx["arco_qaoa_p1"]["F_exact"] < rx["rung1_qaoa_p1"]["F_exact"])
chk("order statistic: the 79-qubit register accepts worse readout than the 24-qubit one",
    np.mean(list(rx["arco_qaoa_p1"]["readout_errors"].values()))
    > np.mean(list(rx["rung1_qaoa_p1"]["readout_errors"].values())))

# ---- ensemble ---------------------------------------------------------------
en = J("ensemble_squeeze.json")
chk("ensemble: 600 instances, no selection filter", len(en["rows"]) == 600)
chk("pooled coupling persists to n_B=160: -0.71 -> -0.95",
    all(close(en["corr_by_n_b"][str(n)]["spearman"], v, 0.02)
        for n, v in ((20, -0.714), (40, -0.761), (80, -0.879), (160, -0.950))))

# ---- the implemented Hamiltonian is not minimised by the target --------------
r1i = D("frozen_rung1.json")
Ai = np.asarray(r1i["A"], int)
ni = Ai.shape[1]
tg = np.minimum(r1i["m"], Ai.sum(1))
Xi = ((np.arange(1 << ni)[:, None] >> np.arange(ni)) & 1).astype(np.int8)
Ei = Xi.sum(1) + r1i["lam"] * ((tg[None, :] - Xi @ Ai.T) ** 2).sum(1)
optmask = (Xi @ Ai.T >= tg[None, :]).all(1) & (Xi.sum(1) == r1i["k_opt"])
chk("Rung-1: exactly 288 optimal covers", int(optmask.sum()) == 288)
chk("implemented QUBO ground state is NOT the target (E=19 vs 20, 144 states below)",
    Ei.min() == 19.0 and int((Ei == 19.0).sum()) == 144 and Ei[optmask].min() == 20.0)

# ---- what the hardware put on the implemented Hamiltonian's own ground state ---
cts = J("kingston_100k_rung1_counts.json")["circuits"]["rung1_qaoa_p1"]
gsi = set(np.where(Ei == Ei.min())[0].tolist())
p_gs = sum(c for b, c in cts["counts"].items() if int(b, 2) in gsi) / cts["shots"]
chk("hardware concentrates 83.9x on the implemented ground manifold (72/100000)",
    close(p_gs, 72 / 100000, 1e-12)
    and close(p_gs / (len(gsi) / 2 ** ni), 83.9, 0.1))

# ---- ensemble: what generalises and what does not -----------------------------
big = [r for r in en["rows"] if r["n_b"] == 160 and r["residual"]]
lo = np.median([r["residual"] for r in big if abs(r["R_over_delta"] - 0.9) < 1e-6])
hi = np.median([r["residual"] for r in big if abs(r["R_over_delta"] - 2.4) < 1e-6])
chk("ensemble residual GROWS with R at n_B=160 (opposite to PEF-24)", hi > lo * 1e6)
chk("all 600 ensemble instances PROVEN optimal (HiGHS status 0, gap 0)",
    en.get("n_proven_optimal") == 600
    and all(r["ilp_status"] == 0 for r in en["rows"])
    and max(r["mip_gap"] for r in en["rows"]) == 0.0)

# ---- the order statistic is the minority mechanism ---------------------------
ra, rb = rx["rung1_qaoa_p1"], rx["arco_qaoa_p1"]
per_qubit_survival = ra["F_readout"] ** (1 / ra["n_qubits"])
counterfactual = rb["F_gates"] * per_qubit_survival ** rb["n_qubits"]
register_share = (ra["F_exact"] - counterfactual) / (ra["F_exact"] - rb["F_exact"])
chk("register size explains ~60% of the new-job reversal",
    0.58 < register_share < 0.62 and counterfactual < ra["F_exact"])

# ---- the within-cell correlation is ~0 (the claim the paper now makes) --------
import scipy.stats as _st
wc = []
for nb in (20, 40, 80, 160):
    for rd in (0.9, 1.35, 1.8, 2.4, 3.0, 3.9):
        sel = [r for r in en["rows"] if r["n_b"] == nb and abs(r["R_over_delta"] - rd) < 0.02]
        if len(sel) > 5:
            c = _st.spearmanr(np.log10([max(r["n_ZZ"], 1) for r in sel]),
                              [r["phi"] for r in sel])[0]
            if not np.isnan(c):
                wc.append(c)
chk("within-cell Spearman is consistent with zero (paper says -0.04)",
    abs(float(np.mean(wc))) < 0.10, f"mean {np.mean(wc):+.3f} over {len(wc)} cells")

# ---- frozen-target control ---------------------------------------------------
fz = J("frozen_target_sweep.json")["rows"]
def mean_phi(rows_, nb, rd):
    v = [r["phi"] for r in rows_ if r["n_b"] == nb and abs(r["R_over_delta"] - rd) < 0.02]
    return float(np.mean(v)) if v else None
worst = max(abs(mean_phi(fz, nb, rd) - mean_phi(en["rows"], nb, rd))
            for nb in (20, 40, 80, 160) for rd in (0.9, 1.35, 1.8, 2.4, 3.0, 3.9)
            if mean_phi(fz, nb, rd) is not None and mean_phi(en["rows"], nb, rd) is not None)
chk("frozen-target control agrees with the anchored ensemble at n_B=160 (dphi <= 0.04)",
    max(abs(mean_phi(fz, 160, rd) - mean_phi(en["rows"], 160, rd))
        for rd in (0.9, 1.35, 1.8, 2.4, 3.0, 3.9)) <= 0.04)
chk("the phi floor is register size, not the generator: n_B=20 reaches 0.09 with frozen targets",
    close(mean_phi(fz, 20, 3.9), 0.086, 0.01))

# ---- submission metadata lives in the current source --------------------------
PAPER = DATA_DIR.parent / "paper"
src = open(PAPER / MAIN_TEX, encoding="utf-8").read()
import re as _re
md = dict(_re.findall(r"^%%([A-Z]+):[ \t]*(.*)$", src, _re.M))
chk(f"all seven submission metadata fields are declared in {MAIN_TEX}",
    set(md) >= {"EMAIL", "KEYWORDS", "ARTICLETYPE", "ROLES", "FUNDING", "DATA", "SUPPDATA"})
chk("the data statement names the archived hardware bitstring distributions",
    "measured bitstring distributions of every hardware circuit" in md["DATA"])
# ---- the frozen-target control, in units of one forced base -------------------
units = []
for nb in (20, 40, 80, 160):
    per = []
    for rd in (0.9, 1.35, 1.8, 2.4, 3.0, 3.9):
        a, b = mean_phi(en["rows"], nb, rd), mean_phi(fz, nb, rd)
        k = [r["k_opt"] for r in fz if r["n_b"] == nb and abs(r["R_over_delta"] - rd) < 0.02]
        if a is not None and b is not None and k:
            per.append(abs(a - b) * float(np.mean(k)))
    units.append((nb, float(np.mean(per)), max(per)))
chk("control tightest at small registers once phi's 1/k_opt quantum is divided out",
    close(units[0][1], 0.36, 0.03) and close(units[-1][1], 1.02, 0.05)
    and max(u[2] for u in units) < 2.4,
    " ".join(f"n_B={n}:{m:.2f}" for n, m, _ in units))

# ---- the block decomposition: the paper's central result ---------------------
bd = J("block_decomposition.json")
bi = bd["instances"]
chk("Rung 1 decomposes into 13 blocks of at most 4 bases",
    (bi["rung1"]["n_components"], bi["rung1"]["largest_block"]) == (13, 4))
chk("Rung 1 combined block-state dimension is 62, not 2^24",
    bi["rung1"]["enumeration_cost"] == 62 and bi["rung1"]["naive_cost"] == 2.0 ** 24)
chk("Rung 2 decomposes into 70 blocks of at most 3 bases, cost 160",
    (bi["arco"]["n_components"], bi["arco"]["largest_block"],
     bi["arco"]["enumeration_cost"]) == (70, 3, 160))
chk("the m=3 instance does NOT decompose (one block, full 2^14)",
    bi["overshoot"]["n_components"] == 1
    and bi["overshoot"]["enumeration_cost"] == 2 ** 14)
chk("cost operator is an exact direct sum on every frozen instance",
    all(v["direct_sum"] for v in bi.values()))
chk("direct-sum property holds across all 600 ensemble instances",
    bd["ensemble_all_direct_sum"] and len(bd["ensemble"]) == 600)
# NB: "n_components >= 1" is a tautology and was doing no work here; the bound is now
# tested against every ensemble row, which is what "everywhere" was claiming all along.
chk("n_components respects the n_B - n_ZZ bound on all 600 ensemble rows, not just the three",
    all(r["n_components"] >= r["lower_bound"] for r in bd["ensemble"])
    and all(v["n_components"] >= v["lower_bound"] for v in bi.values()),
    f"tested on {len(bd['ensemble'])} ensemble rows + {len(bi)} frozen instances")

# the two costs rise together, and the block cost keeps a signal at fixed geometry
bl = {(r["n_b"], r["R_over_delta"], r["seed"]): r for r in bd["ensemble"]}
ph = {(r["n_b"], r["R_over_delta"], r["seed"]): r for r in en["rows"]}
ks = sorted(set(bl) & set(ph))
pooled = _st.spearmanr(np.log10([max(bl[k]["n_ZZ"], 1) for k in ks]),
                       [bl[k]["log2_enum"] for k in ks])[0]
chk("pooled Spearman(log n_ZZ, log2 exact cost) = +0.97", close(pooled, 0.967, 0.01))
wb = []
for nb in (20, 40, 80, 160):
    for rd in (0.9, 1.35, 1.8, 2.4, 3.0, 3.9):
        sel = [k for k in ks if k[0] == nb and abs(k[1] - rd) < 0.02]
        if len(sel) > 5:
            c = _st.spearmanr(np.log10([max(bl[k]["n_ZZ"], 1) for k in sel]),
                              [bl[k]["log2_enum"] for k in sel])[0]
            if not np.isnan(c):
                wb.append(c)
chk("block cost keeps an instance-level signal where phi does not (+0.47 vs -0.04)",
    float(np.mean(wb)) > 0.3, f"blocks {np.mean(wb):+.3f} vs phi {np.mean(wc):+.3f}")

# ---- the PEF sweep numbers printed in the squeeze table ----------------------
from run_m_ladder import build_instance, coverage as _cov
from calc_block_decomposition import decompose as _dec
_bl, _bo, _tl, _to, _sd, _ = build_instance()
sweep_expected = {200: (13, 4, 62), 300: (9, 7, 174), 400: (3, 17, 131138),
                  600: (1, 24, 16777216)}
ok_sweep = True
for R, exp in sweep_expected.items():
    r = _dec(_cov(_bl, _bo, _tl, _to, float(R)).astype(int))
    ok_sweep &= (r["n_components"], r["largest_block"], r["enumeration_cost"]) == exp
chk("every block value printed in the squeeze table is reproduced from the instance",
    ok_sweep)

# ---- the continental instance: the far end of the trade ----------------------
cb = J("continental_blocks.json")
chk("continental instance: n_ZZ=39,974 and 4 blocks (770 + three isolated)",
    cb["n_ZZ"] == 39974 and cb["n_components"] == 4 and cb["largest_block"] == 770)
chk("its exact block cost is 2^770, i.e. no decomposition saving",
    close(cb["log2_exact_cost"], 770.0, 0.5))
chk("phi there is 0.064 (the three upper-Rio-Negro municipalities)",
    close(cb["phi"], 0.0638, 0.001) and cb["forced"] == 3 and len(cb["isolated"]) == 3)

# ---- refinement at fixed coverage: does the decomposition survive? -----------
ds = J("dimension_sweep.json")["regimes"]
fr, gr, po = ds["frontier"], ds["grid2d"], ds["poisson2d"]

chk("frontier and grid2d refine at complete coverage; poisson2d does not",
    all(r["covered"] == 1.0 for r in fr + gr) and all(r["covered"] < 1.0 for r in po),
    f"poisson2d coverage {min(r['covered'] for r in po):.3f}-{max(r['covered'] for r in po):.3f}")
chk("the block decomposition dies under coverage-preserving refinement",
    all(r["n_components"] == 1 for r in fr + gr))
chk("the quasi-1D frontier keeps a small treewidth bound while n_B grows 5.6x",
    max(r["treewidth_ub"] for r in fr) <= 10 and fr[-1]["treewidth_ub"] <= fr[0]["treewidth_ub"]
    and fr[-1]["n_b"] / fr[0]["n_b"] > 5,
    f"tw<= {fr[0]['treewidth_ub']} at n_B={fr[0]['n_b']} -> {fr[-1]['treewidth_ub']} at n_B={fr[-1]['n_b']}")
chk("only the regular lattice pushes the treewidth bound up (min-fill is an UPPER bound)",
    ds["grid2d_tw_exponent"] > 0.4 and ds["frontier_tw_exponent"] < 0,
    f"exponents: grid2d {ds['grid2d_tw_exponent']:+.2f}, frontier {ds['frontier_tw_exponent']:+.2f}")
chk("ceil(LP) equals the ILP optimum in all 16 swept instances",
    all(int(np.ceil(r["lp"] - 1e-9)) == r["k_opt"] for r in fr + gr + po))
chk("every swept instance is solved to proven optimality, comfortably inside a second",
    all(r["ilp_status"] == 0 for r in fr + gr + po)
    and max(r["seconds"] for r in fr + gr + po) < 1.0,
    "a bound, not a pinned value: wall times drift between runs and machines")

# rebuild each swept instance from the geometry and recompute, rather than trusting the JSON
from calc_block_decomposition import decompose as _dec
from calc_dimension_sweep import build as _build, frontier_polyline as _poly
from calc_ilp_geographic import load_municipalities as _lm

_la, _lo, _ = _lm("centroid")
_bbox = (_la.min(), _la.max(), _lo.min(), _lo.max())
_p = _poly()
ok_dim = True
for kind, rows in (("frontier", fr), ("grid2d", gr), ("poisson2d", po)):
    _rng = np.random.default_rng(J("dimension_sweep.json")["seed"])
    for r in rows:
        Ak, cov = _build(kind, float(r["R_km"]), _bbox, _p, _rng)
        d = _dec(Ak.astype(np.int64))          # int64: the int8 product would wrap silently
        ok_dim &= (d["n_ZZ"] == r["n_ZZ"] and d["n_components"] == r["n_components"]
                   and int(Ak.shape[1]) == r["n_b"] and close(cov, r["covered"], 5e-4))
chk("every row of the refinement sweep is reproduced from the raw geometry", ok_dim)

# ---- irregular 2D at genuinely complete coverage -----------------------------
cp = J("coverage_preserving_2d.json")
ex = cp["experiments"]
allrows = ex["full"] + ex["seeds"] + ex["scale"]
chk("raising the post intensity does reach 100% coverage at every radius",
    all(r["covered"] == 1.0 for r in allrows) and all(r["direct_sum"] for r in allrows))
chk("blocks collapse in irregular 2D once coverage is complete",
    all(r["n_components"] <= 7 for r in allrows)
    and max(r["largest_block"] / r["n_b"] for r in allrows) > 0.95,
    f"largest block holds >= {min(r['largest_block'] / r['n_b'] for r in allrows):.0%} of the posts")
# This check used to look only at the lattice arm and therefore passed while the paper
# claimed integrality was a lattice artifact -- which the frontier and Poisson arms refute.
# It now counts exactly-integral relaxations in EVERY regime, so the old claim would fail it.
_nonlattice = fr + po
_int_nonlat = sum(1 for r in _nonlattice if r["lp_gap"] == 0.0)
chk("exact LP integrality is generic here, not a property of the lattice",
    _int_nonlat == 9 and len(_nonlattice) == 11
    and all(int(np.ceil(r["lp"] - 1e-9)) == r["k_opt"] for r in fr + gr + po)
    and cp["seeds_with_loose_lp_bound"] == 3 and cp["seeds_tested"] == 9,
    f"relaxation exactly integral in {_int_nonlat}/{len(_nonlattice)} non-lattice instances "
    f"and {sum(1 for r in gr if r['lp_gap'] == 0.0)}/{len(gr)} lattice ones; the bound only "
    f"goes loose under irregular placement at complete coverage "
    f"({cp['seeds_with_loose_lp_bound']}/{cp['seeds_tested']})")
sc = ex["scale"]
chk("the LP bound loosens and the solve slows as the radius shrinks, but stays tractable",
    sc[-1]["bound_slack"] > sc[0]["bound_slack"] and sc[-1]["seconds"] > 20 * sc[0]["seconds"]
    and max(r["seconds"] for r in sc) < 60,
    f"slack {sc[0]['bound_slack']}->{sc[-1]['bound_slack']}, "
    f"{sc[0]['seconds']:.3f}s->{sc[-1]['seconds']:.3f}s over n_B {sc[0]['n_b']}->{sc[-1]['n_b']}")

# ---- the protection variant --------------------------------------------------
pr = J("protection_2d.json")["rows"]
gp = [r for r in pr if r["kind"] == "grid2d"]
pp = [r for r in pr if r["kind"] == "poisson2d"]

# The old second clause tested that the generator's own if/else was taken consistently with
# itself -- it could not fail.  What carries the claim is that ordinary domination returns a
# solution on every row, including the rows where TOTAL domination is infeasible.
_iso = [r for r in pr if r["isolated_posts"] > 0]
chk("ordinary domination is always feasible; only total domination can be infeasible",
    all(r["k_dominated"] is not None for r in pr)
    and len(_iso) > 0 and all(r["k_dominated"] is not None for r in _iso)
    and all(r["total_dom_status"] == -1 for r in _iso),
    f"{len(_iso)} rows carry isolated outposts; ordinary domination still solves all of them, "
    "so an isolated outpost forces x_b = 1 rather than making the instance infeasible")
chk("protection binds hard at short range and stops binding at long range",
    pp[0]["k_dominated"] / pp[0]["k_cover"] > 2.0
    and pp[-1]["k_dominated"] == pp[-1]["k_cover"],
    f"irregular: {pp[0]['k_dominated'] / pp[0]['k_cover']:.2f}x of the cover at R_p=0.9d, "
    f"1.00x at 3.0d")
chk("an optimal cover already dominates the lattice only once R_p >= 1.5 spacings",
    [r["optimal_cover_already_dominates"] for r in gp] == [False, False, True, True, True])
chk("protection kills the block decomposition, but only at long protection range",
    pp[0]["blocks_protected"] == pp[0]["blocks_cover"] and pp[-1]["blocks_protected"] == 1,
    f"irregular blocks {' -> '.join(str(r['blocks_protected']) for r in pp)} "
    f"as R_p/d goes {pp[0]['R_p_over_delta']} -> {pp[-1]['R_p_over_delta']}")
chk("the only instances HiGHS cannot close are total domination on the regular lattice",
    [r["total_dom_status"] for r in gp].count(1) == 2
    and all(r["total_dom_status"] != 1 for r in pp)
    and max(r["seconds_dominated"] for r in pr) < 1.0,
    "perfect lattice symmetry, not geography, is what defeats the solver")

# Penrose isolated-vertex threshold: unprotectable outposts = n_B exp(-pi rho R_p^2)
_A = 8.17e6
ok_pen = all(
    abs(r["isolated_posts"] - r["n_b"] * np.exp(-np.pi * (r["n_b"] / _A) * r["R_p_km"] ** 2))
    <= 0.2 * r["n_b"] * np.exp(-np.pi * (r["n_b"] / _A) * r["R_p_km"] ** 2) + 1.0
    for r in pp)
chk("unprotectable outposts follow the Poisson isolated-vertex law to within 20%",
    ok_pen and pp[-1]["isolated_posts"] == 0,
    "observed " + "/".join(str(r["isolated_posts"]) for r in pp) + " vs predicted "
    + "/".join(f"{r['n_b'] * np.exp(-np.pi * (r['n_b'] / _A) * r['R_p_km'] ** 2):.0f}" for r in pp))
chk("the Poisson law does not apply to the lattice, and visibly fails there",
    gp[0]["isolated_posts"] == gp[0]["n_b"],
    "at 0.9 spacings every lattice post is isolated at once; the Poisson law predicts ~23")

# ---- tactical re-planning after a loss ----------------------------------------
tr = J("tactical_replanning.json")["instances"]
r1t, art = tr["rung1"], tr["arco"]

_react = [row for v in r1t["reactive"].values() for row in v]
chk("re-planning after a loss costs milliseconds, not seconds",
    max(row["seconds"] for row in _react) < 0.05 and r1t["seconds_initial"] < 0.05,
    f"slowest re-solve {1e3 * max(row['seconds'] for row in _react):.1f} ms on {r1t['n_b']} posts "
    f"(a bound; wall times are machine-dependent)")
chk("a re-plan moves two posts: retire the casualty, commission one replacement",
    all(row["churn"] == 2 for row in _react if row["churn"] is not None))
chk("the deployment runs out of slack long before the solver runs out of time",
    any(not row["feasible"] for row in _react),
    "random losses survive 5, targeted losses survive 2")
chk("both real deployments admit no genuine redundancy at all",
    r1t["fragility"]["max_m_possible"] == 1 and art["fragility"]["max_m_possible"] == 1,
    f"singly-reachable targets: PEF-24 {r1t['fragility']['singly_reachable']}/{r1t['n_t']}, "
    f"Arco {art['fragility']['singly_reachable']}/{art['n_t']}")
chk("an adversary with a single kill makes the cover unrepairable, not merely dearer",
    all(x.get("unrepairable") for x in r1t["robust"] if x["gamma"] >= 1))
_r2 = [x for x in r1t["retrofit"] if x["m"] == 2][0]
_a2 = [x for x in art["retrofit"] if x["m"] == 2][0]
_a3 = [x for x in art["retrofit"] if x["m"] == 3][0]
chk("fault tolerance is purchasable on the frontier and partly unpurchasable on the arc",
    _r2["feasible"] and _r2["new_posts"] == 4 and _a2["feasible"] and _a2["new_posts"] == 25
    and not _a3["feasible"],
    f"m=2 costs +{_r2['new_posts']} posts (PEF-24) and +{_a2['new_posts']} (Arco); "
    f"m=3 unreachable on the Arco from all 773 candidate sites")

# ---- graceful degradation: maximum coverage once full coverage is gone --------
gd = J("graceful_degradation.json")["instances"]
g1, ga = gd["rung1"], gd["arco"]
_all = g1["single_loss"] + ga["single_loss"]

chk("re-optimising after a loss covers strictly more than keeping the survivors",
    max(r["gain"] for r in g1["single_loss"]) == 5
    and all(r["gain"] >= 0 for r in _all),
    f"PEF-24 recovers up to +5 targets (mean {np.mean([r['gain'] for r in g1['single_loss']]):.1f}); "
    f"Arco only up to +2 (mean {np.mean([r['gain'] for r in ga['single_loss']]):.1f})")
chk("the redundancy a deployment has is what re-optimisation can exploit",
    np.mean([r["gain"] for r in g1["single_loss"]])
    > 5 * np.mean([r["gain"] for r in ga["single_loss"]]),
    "26% singly-reachable on PEF-24 against 80% on the Arco")
chk("switching to maximum coverage opens no classical gap: greedy ties the ILP everywhere",
    all(r["greedy"] == r["active"] for r in _all), f"{len(_all)}/{len(_all)} losses")
chk("nor does the LP relaxation of maximum coverage fray (no 1-1/e gap in this geometry)",
    all(r["lp_relax"] == float(r["active"]) for r in _all))
chk("maximum coverage costs milliseconds, and greedy is cheaper still",
    max(r["ilp_seconds"] for r in _all) < 0.01
    and max(r["greedy_seconds"] for r in _all) < max(r["ilp_seconds"] for r in _all),
    f"ILP <= {1e3 * max(r['ilp_seconds'] for r in _all):.1f} ms, "
    f"greedy <= {1e3 * max(r['greedy_seconds'] for r in _all):.2f} ms")
chk("extra budget cannot buy back a target whose only post is gone",
    len({r["covered"] for r in g1["budget_curve"]["rows"]}) == 1
    and len({r["covered"] for r in ga["budget_curve"]["rows"]}) == 1,
    f"PEF-24 flat at {g1['budget_curve']['rows'][0]['covered']}/{g1['n_t']} and Arco at "
    f"{ga['budget_curve']['rows'][0]['covered']}/{ga['n_t']} across five extra posts")
chk("area jamming fragments coverage without costing a single post",
    not g1["jam"][1]["full_cover_possible"] and g1["jam"][1]["links_cut"] == 15
    and not ga["jam"][0]["full_cover_possible"] and ga["jam"][0]["links_cut"] == 6,
    f"PEF-24 needs 15 links cut to break; the Arco breaks on {ga['jam'][0]['links_cut']}")

# ---- the rewritten sections: every literal in the prose must match the evidence ----
from _paths import PAPER_DIR

# The prose checks search the manuscript and its supplement, so migrating a passage from
# one to the other is a layout decision rather than a silent check failure.
_SOURCES = (MAIN_TEX, SUPP_TEX)
_tex = re.sub(r"\s+", " ", " ".join(
    (PAPER_DIR / f).read_text(encoding="utf-8") for f in _SOURCES))


def says(*fragments):
    """Every fragment must appear verbatim in the manuscript."""
    miss = [f for f in fragments if f not in _tex]
    _missing.extend(repr(f[:90]) for f in miss)
    return not miss


ds2 = J("dimension_sweep.json")["regimes"]
_fr, _gr, _po = ds2["frontier"], ds2["grid2d"], ds2["poisson2d"]
cp2 = J("coverage_preserving_2d.json")["experiments"]
_allcp = cp2["full"] + cp2["seeds"] + cp2["scale"]
pr2 = [r for r in J("protection_2d.json")["rows"] if r["kind"] == "poisson2d"]
tr2 = J("tactical_replanning.json")["instances"]
gd2 = J("graceful_degradation.json")["instances"]

chk("prose: the refinement ranges quoted in Sec. II match the sweep",
    says(r"$\nB:35\to197$", r"$\nB:78\to879$")
    and (_fr[0]["n_b"], _fr[-1]["n_b"]) == (35, 197)
    and (_gr[0]["n_b"], _gr[-1]["n_b"]) == (78, 879))
_k = [2 * r["n_ZZ"] / r["n_b"] for r in _po]
chk("prose: the sub-critical mean degree 4.98-5.60 is the measured one",
    says("4.98$--$5.60$", "We do not compare these degrees with a percolation threshold")
    and close(min(_k), 4.98, 0.005) and close(max(_k), 5.60, 0.005))
_kc = [2 * r["n_ZZ"] / r["n_b"] for r in _allcp]
chk("prose: the super-critical mean degree 8.0-21.5 is the measured one",
    says("$8.0$--$21.5$") and close(min(_kc), 8.0, 0.05) and close(max(_kc), 21.5, 0.05))
chk("prose: the strip bound table is reproduced digit for digit",
    says(r"$12.5,\,16.7,\,25.0,\,33.4,\,50.0$", r"$14,\,15,\,24,\,33,\,48$")
    and [r["treewidth_ub"] for r in _gr] == [14, 15, 24, 33, 48]
    and all(close(2 * 2502.0 / r["R_km"], b, 0.06)
            for r, b in zip(_gr, (12.5, 16.7, 25.0, 33.4, 50.0))))
chk("prose: the frontier treewidth bound really falls from 10 to 6 over 5.6x in n_B",
    says("from $10$ to $6$", "factor of $5.6$")
    and (_fr[0]["treewidth_ub"], _fr[-1]["treewidth_ub"]) == (10, 6)
    and close(_fr[-1]["n_b"] / _fr[0]["n_b"], 5.6, 0.05))
chk("prose: the 879-post row is 48 wide and the quoted time is an honest upper bound",
    says("the width bound is $48$", "under $50$ ms")
    and _gr[-1]["treewidth_ub"] == 48 and _gr[-1]["seconds"] < 0.050)
chk("prose: 8234 posts in seconds, and the LP bound loose in three of nine",
    says("$\\nB=8234$", "a few seconds", "three of nine")
    and cp2["scale"][-1]["n_b"] == 8234 and 0.5 < cp2["scale"][-1]["seconds"] < 10.0
    and sum(r["bound_slack"] > 0 for r in cp2["seeds"]) == 3)
chk("prose: the protection component curve 15 / 13 / 1 is the measured one",
    says("$15$ components at", "still $13$ at", "a single component only at")
    and [r["blocks_protected"] for r in pr2] == [15, 14, 13, 7, 1])
chk("prose: protection costs 2.31x the unprotected cover at the shortest range",
    says("$2.31$ times") and close(pr2[0]["k_dominated"] / pr2[0]["k_cover"], 2.31, 0.005))
chk("prose: the Penrose prediction and the observed outpost counts are both quoted right",
    says("$181,111,59,15,0.3$", "$195,128,71,16,0$", "341-base")
    and [r["isolated_posts"] for r in pr2] == [195, 128, 71, 16, 0]
    and pr2[0]["n_b"] == 341)
_f1, _fa = tr2["rung1"]["fragility"], tr2["arco"]["fragility"]
chk("prose: the fragility counts in Sec. V are the measured degree structure",
    says("Six of the 23", "four of the 24", "39 of 49", "29 of 79")
    and (_f1["singly_reachable"], _f1["irreplaceable_posts"]) == (6, 4)
    and (_fa["singly_reachable"], _fa["irreplaceable_posts"]) == (39, 29))
_r2 = [x for x in tr2["rung1"]["retrofit"] if x["m"] == 2][0]
_r3 = [x for x in tr2["rung1"]["retrofit"] if x["m"] == 3][0]
_a2 = [x for x in tr2["arco"]["retrofit"] if x["m"] == 2][0]
_a3 = [x for x in tr2["arco"]["retrofit"] if x["m"] == 3][0]
chk("prose: the retrofit prices are the ILP's, not an estimate",
    (says("four additions", "24 in all", "triple coverage requires eleven",
          "double coverage requires 25 additions", "six targets remain short") if V5 else
     says("four new bases", "24 in all", "eleven for triple",
          "costs 25 new bases", "six targets remain short"))
    and (_r2["new_posts"], _r2["posts_used_total"], _r3["new_posts"]) == (4, 24, 11)
    and _a2["new_posts"] == 25 and not _a3["feasible"] and _a3["unreachable"] == 6)
_g1, _ga = gd2["rung1"]["single_loss"], gd2["arco"]["single_loss"]
chk("prose: the re-optimisation gains and their means are the measured ones",
    says("up to five waypoints (mean $1.5$)", "at most two (mean $0.2$)")
    and max(r["gain"] for r in _g1) == 5 and close(np.mean([r["gain"] for r in _g1]), 1.5, 0.05)
    and max(r["gain"] for r in _ga) == 2 and close(np.mean([r["gain"] for r in _ga]), 0.2, 0.02))
chk("prose: the flat budget curves and the solver costs are quoted right",
    (says("$21/23$", "$46/49$", "all 41 single-deletion instances", "a few milliseconds") if V5 else
     says("$21/23$", "$46/49$", "all 41 single-loss", "a few milliseconds"))
    and gd2["rung1"]["budget_curve"]["rows"][0]["covered"] == 21
    and gd2["arco"]["budget_curve"]["rows"][0]["covered"] == 46
    and len(_g1) + len(_ga) == 41
    and max(r["ilp_seconds"] for r in _g1 + _ga) < 0.020
    and max(r["greedy_seconds"] for r in _g1 + _ga)
    < max(r["ilp_seconds"] for r in _g1 + _ga))
chk("prose: the jamming figures match the simulated link cuts",
    says("removes 15 of the 51", "$18/23$", "$15/23$", "six cut links", "$32/49$")
    and gd2["rung1"]["jam"][1]["links_cut"] == 15
    and gd2["rung1"]["jam"][1]["covered"] == 18
    and gd2["rung1"]["jam"][3]["covered"] == 15
    and gd2["arco"]["jam"][0]["links_cut"] == 6
    and gd2["arco"]["jam"][3]["covered"] == 32)

# ---- the circuit factorises: the reframed central claim ----------------------
bf = J("block_factorisation.json")
chk("the ideal 24-qubit QAOA state is exactly a product over the 13 blocks",
    abs(bf["product_state_fidelity"] - 1.0) < 1e-12
    and abs(bf["entanglement_entropy_bits"]) < 1e-12,
    f"fidelity {bf['product_state_fidelity']:.15f}, "
    f"entropy {bf['entanglement_entropy_bits']:.1e} bits across the largest block")
chk("the 288-fold optimum is itself a free product over the blocks",
    bf["optima_factorise"] and bf["optima_product"] == 288 == bf["n_optimal_covers"],
    " x ".join(str(x) for x in bf["optima_per_block"] if x > 1) + " = 288")
chk("the ideal logical unitary factorises into 13 blocks of at most four qubits",
    bf["n_blocks"] == 13 and bf["largest_block_qubits"] == 4
    and bf["classical_simulation_cost"] == 62)
chk("the noiseless reference is computable and hardware reaches 43.3% of its probability",
    close(sa_p1["ideal_p_opt"], 0.0018253778, 1e-10)
    and close(sa_p1["ideal_eta_all_register"], 106.33596, 1e-4)
    and close(sa_p1["hardware_to_ideal_absolute_probability_ratio"], 0.432787, 1e-5))
chk("prose: the reframed abstract and body quote the factorisation evidence",
    (says("$|1-F|<10^{-12}$", "$|S|<10^{-12}$ bits",
          "double-precision round-off", "13 circuits of at most four qubits",
          "factorizes into 13 block circuits of at most four qubits") if V5 else
     says("$|1-F|<10^{-12}$", "$|S|<10^{-12}$ bits",
          "double-precision round-off", "13 independent circuits of at most four qubits",
          "factorizes into 13 block circuits of at most four qubits")))
chk("prose: the tensor-network consequence is stated with its citation",
    (says("markov2008simulating", "width at most $(p+1)(w+1)-1$",
          "$N_g^{O(1)}2^{O(p(w+1))}$", "$N_g^{O(1)}\\exp[O(p\\rho R H)]$") if V5 else
     says("markov2008simulating", "the width drops to $O(p(w+1))$",
          "$N_g^{O(1)}\\exp\\!\\big(O(p\\,\\rho R H)\\big)$")))

# ---- the ladder: three rungs, two mechanisms, one place both fail ------------
lr = J("ladder_rungs_v5.json" if V5 else "ladder_rungs.json")
r2, r3 = lr["rung2"], lr["rung3"]

_ang = lr["rung2_angles"]
chk("Rung 2's null survives angle optimisation, so it is the ansatz and not the angles",
    _ang["gain_from_optimising"] > 1e9
    and _ang["shots_for_one_hit_optimised"] > 1e8
    and _ang["ansatz_limited"] and r2["n_optimal_covers"] == 4,
    f"optimising the angles gains {_ang['gain_from_optimising']:.1e}x, and still needs "
    f"{_ang['shots_for_one_hit_optimised']:.1e} shots against the 100000 taken")
chk("Rung 2 concentrates well and it does not help: 217x onto 4 strings in 8.8e21",
    close(r2["eta_ideal"], 217.09, 0.5) and r2["n_blocks"] == 70
    and r2["largest_block"] == 3)
chk("Rung 3's unsimulability rests on a LOWER bound, as it must",
    r3["n_b"] == 773 and r3["n_ZZ"] == 39974
    and r3["treewidth_lb"] >= 112 and r3["treewidth_ub"] == 151
    and r3["treewidth_lb"] < r3["treewidth_ub"] <= r3["strip_bound_max_density"],
    f"clique of {r3['max_clique_found']} gives tw >= {r3['treewidth_lb']}; "
    f"bracketed [{r3['treewidth_lb']}, {r3['treewidth_ub']}], contraction >= 2^{r3['treewidth_lb']}")
chk("the strip bound needs maximum density: the mean understates it on real geography",
    r3["clustering_factor"] > 1.5
    and r3["strip"]["north_south"]["worst_strip"] == 228,
    f"worst 2R strip holds {r3['strip']['north_south']['worst_strip']} centroids against "
    f"{r3['strip_bound_mean_density']:.0f} for a uniform deployment "
    f"({r3['clustering_factor']}x)")
chk("prose: the ladder's new numbers are quoted as computed",
    (says("$P_{\\mathrm{ideal}}=9.9\\times10^{-20}$", "the own-angle run took $10^5$ shots",
          "$P_{\\mathrm{ideal}}^{\\max}=2.0\\times10^{-9}$", "$5\\times10^{8}$ shots",
          "$\\mathrm{tw}\\le151$", "$\\mathrm{tw}\\ge112$", "bracketed in $[112,151]$",
          "upper bounds establish simulability, never its absence",
          "228 centroids in the worst $2R$ strip", "by a factor of $1.5$",
          "The 8192-shot transferred-angle Rung-2 job uses that same pair",
          "block factorization makes the ideal target-set probability exactly computable") if V5 else
     says("$P_{\\mathrm{ideal}}=9.9\\times10^{-20}$", "the own-angle run took $10^5$ shots",
          "$P_{\\mathrm{ideal}}^{\\max}=2.0\\times10^{-9}$", "$5\\times10^{8}$ shots",
          "$\\mathrm{tw}\\le151$", "$\\mathrm{tw}\\ge112$", "bracketed in $[112,151]$",
          "upper bounds establish simulability, never its absence",
          "228 centroids in the worst $2R$ strip", "by a factor of $1.5$",
          "The 8192-shot transferred-angle Rung-2 job uses those angles",
          "the decomposed 79-qubit ideal distribution is optimized directly")))

# ---- the cardinality-preserving ansatz does not escape the argument ----------
qs = J("qwoa_simulability_v5.json" if V5 else "qwoa_simulability.json")
if V5:
    qs["worst_block_cut_rank"] = qs["maximum_block_cut_rank"]
    qs["worst_block_cut_entropy_bits"] = qs["maximum_block_cut_entropy_bits"]
chk("QWOA is genuinely entangled across blocks, unlike the ideal logical QAOA state",
    qs["worst_block_cut_rank"] > 1 and qs["worst_block_cut_entropy_bits"] > 1.0
    and abs(bf["entanglement_entropy_bits"]) < 1e-12,
    f"QWOA carries {qs['worst_block_cut_entropy_bits']:.2f} bits across the worst block cut; "
    f"QAOA carries {abs(bf['entanglement_entropy_bits']):.1e}")
chk("its Schmidt rank saturates the k+1 bond-dimension bound",
    qs["bound_holds"] and qs["worst_block_cut_rank"] == qs["bond_dimension_bound"] == 11)
chk("QWOA state-space proxy is eleven times the QAOA proxy, not a different regime",
    qs["mps_simulation_cost"] == 682
    and qs["mps_simulation_cost"] == (qs["k_opt"] + 1) * bf["classical_simulation_cost"]
    and qs["naive_cost"] == 2 ** 24
    and says("the proxy is $682$ against $2^{24}$"),
    f"{qs['mps_simulation_cost']} state-space proxy against 2^24, "
    f"and {qs['mps_simulation_cost'] // bf['classical_simulation_cost']}x the 62 of the QAOA circuit")
chk("prose: the QWOA argument is stated with its verified numbers",
    (says("saturating the bound $k+1=11$", "the proxy is $682$ against $2^{24}$",
          "maximum block-cut entropy is $2.35$ bits",
          "exact global continuous-time evolution") if V5 else
     says("saturating the bound $k+1=11$", "the proxy is $682$ against $2^{24}$",
          "the worst block cut carries $2.33$ bits",
          "those are \\emph{exactly} the $ZZ$ couplings")))

# ---- controlled same-sector mixer comparison --------------------------------
mc = J("same_sector_mixer_audit.json")
mc_raw = mc["summaries"]["problem_weighted_raw_minus_support_unweighted"]
mc_norm = mc["summaries"][
    "problem_weighted_rms_normalized_minus_support_unweighted"
]
mc_complete = mc["summaries"][
    "problem_weighted_rms_normalized_minus_complete_unweighted"
]
chk("same-sector weighted and unit-support energy wins split 3--3",
    mc_raw["weighted_lower_energy"] == mc_raw["control_lower_energy"] == 3
    and mc_norm["weighted_lower_energy"] == mc_norm["control_lower_energy"] == 3)
chk("raw same-sector P(opt) ratio is small and heterogeneous",
    mc_raw["weighted_larger_p_opt"] == 4
    and close(mc_raw["p_opt_ratio"]["median"], 1.1007921791, 1e-8)
    and np.allclose(
        [mc_raw["p_opt_ratio"]["minimum"], mc_raw["p_opt_ratio"]["maximum"]],
        [0.4831550166, 1.9831039485], atol=1e-8))
chk("RMS-normalized same-sector P(opt) wins split 3--3",
    mc_norm["weighted_larger_p_opt"] == mc_norm["control_larger_p_opt"] == 3
    and close(mc_norm["p_opt_ratio"]["median"], 0.9753102729, 1e-8)
    and np.allclose(
        [mc_norm["p_opt_ratio"]["minimum"], mc_norm["p_opt_ratio"]["maximum"]],
        [0.3002974840, 1.9831039512], atol=1e-8))
chk("complete unit-weight mixer wins four of six normalized energy comparisons",
    mc_complete["weighted_lower_energy"] == 2
    and mc_complete["control_lower_energy"] == 4
    and mc_complete["weighted_larger_p_opt"] == mc_complete["control_larger_p_opt"] == 3)
chk("same-sector table contains the six audited numerical rows",
    len(mc["rows"]) == 6
    and all(row["all_endpoints_converged"] for row in mc["rows"])
    and [(row["seed"], row["m"], row["k_opt"]) for row in mc["rows"]]
        == [(12, 1, 2), (12, 2, 5), (12, 3, 8),
            (17, 1, 2), (17, 2, 5), (17, 3, 8)])

# ---- temporal scope ----------------------------------------------------------
if V5:
    chk("v5 retires the invalid persistent-surveillance proxy and states the scope",
        says("none of its application-level conclusions extends to persistent surveillance",
             "Constructing and analyzing that model is outside the scope")
        and "calc_persistent_surveillance.py" not in (
            DATA_DIR.parent / "scripts" / "reproduce_v5.py"
        ).read_text(encoding="utf-8"))
else:
    ps = J("persistent_surveillance.json")
    _by_reach = {}
    for r in ps["rows"]:
        if r["endurance"] >= 99:
            _by_reach.setdefault(r["reach_km"], r["n_components"])
    chk("mobility collapses the blocks, so the direct sum does not carry to persistent coverage",
        ps["static"]["n_components"] == 13
        and [_by_reach[k] for k in (0.0, 150.0, 300.0, 600.0)] == [13, 12, 7, 2])
    chk("the time-expanded instances nonetheless stay easy",
        all(r["lp_gap"] == 0.0 for r in ps["rows"] if r["status"] == 0)
        and max(r["seconds"] for r in ps["rows"]) < 1.0
        and max(r["treewidth_ub"] for r in ps["rows"] if r["treewidth_ub"]) <= 16)
    chk("persistent surveillance of PEF-24 is infeasible under a binding endurance",
        any(r["status"] != 0 for r in ps["rows"] if r["endurance"] == 1))
    chk("and the infeasibility is caused by the singly-reachable waypoints",
        ps["attribution"]["caused_by_singly_reachable"]
        and ps["attribution"]["singly_reachable_waypoints"] == 6)
    chk("prose: the limitations section states the caveat with its measured numbers",
        says("collapses to 12, 7 and finally 2 components",
             "six of the 23 waypoints are reachable by a single base",
             "Removing exactly those six waypoints restores feasibility",
             "rising only from 4 to 16"))

chk("prose: the introduction poses the question and declares the static restriction",
    (says("The benchmark question is: given candidate service sites",
          "We study a deliberately static benchmark",
          "it is not an operational model of persistent surveillance",
          "restrict every application-level conclusion to the static benchmark family") if V5 else
     says("does protecting the frontier, and eventually the Legal Amazon, need a quantum computer?",
          "the formulation most favorable to the quantum side",
          "the temporal formulation is a distinct problem")))

# ---- the faithful inequality encoding ----------------------------------------
ee = J("exact_encoding.json")["instances"]
chk("the faithful slack encoding preserves the block count on both instances",
    all(v["blocks_preserved"] for v in ee.values())
    and ee["rung1"]["exact"]["blocks"] == 13 and ee["arco"]["exact"]["blocks"] == 70)
chk("it costs register and couplings, not decomposability",
    (ee["rung1"]["exact"]["n_qubits"], ee["rung1"]["exact"]["n_ZZ"]) == (51, 103)
    and (ee["arco"]["exact"]["n_qubits"], ee["arco"]["exact"]["n_ZZ"]) == (89, 29)
    and ee["rung1"]["exact"]["enumeration_cost"] == 8958
    and ee["arco"]["exact"]["enumeration_cost"] == 220,
    "Rung 1: 24 -> 51 qubits, n_ZZ 17 -> 103, cost 62 -> 8,958")
chk("prose: the surrogate is declared where the operator is defined, not fifteen pages later",
    says("Equation~(\\ref{eq:qubo_msc}) is a \\emph{two-sided surrogate}",
         "\\emph{the block count is unchanged}",
         "51 qubits against 24", "combined block-state dimension $8958$ against $2^{51}$",
         "89 qubits against 79"))

chk("prose: the algebraic theorem states its initial-state and layer hypotheses",
    says("Factorization theorem", "If the initial state is",
         "every cost and mixer layer is a tensor product",
         "H_C=\\sum_i H_C^{(i)}\\otimes\\mathbb 1_{\\bar V_i}")
    and "bigoplus_i H_C" not in _tex)
chk("prose: geometry is a sufficient corollary, not the theorem",
    (says("geometry is a sufficient way to produce its hypotheses, not part of the result",
          "E(G_{ZZ})\\subseteq E(G_B(2R))", "$G_{ZZ}$ is only a subgraph of it",
          "a nearby pair may fail to share a target", "The edge count is a weak diagnostic") if V5 else
     says("geometry is a sufficient way to produce its hypotheses, not part of the result",
          "E(G_{ZZ})\\subseteq E(G_B(2R))",
          "The converse is false", "The edge count is a weak diagnostic")))
chk("prose: the width bound is a max-occupancy statement, and width is separated from cost",
    (says("W_\\xi=\\max_s|\\mathcal B_s|", "On real geography $W$ must be measured",
          "the calculation does not identify the geographic cause of the clustering",
          "$\\exp[O(p\\sqrt{\\nB})]$", "this is an exponent-scale bound, not a literal state-vector cost") if V5 else
     says("$W$ is the largest number of bases any strip of width $2R$ contains",
          "clustering along rivers and roads", "On the plane it grows as $\\exp[O(p\\sqrt{\\nB})]$",
          "not a literal statement that the contraction uses no more resources than a 48-qubit state vector",
          "$\\mathrm{tw}(G)=\\max_i\\mathrm{tw}(G[V_i])\\le\\max_i|V_i|-1$")))
chk("prose: IBM statistics use same-space baselines; the binomial test is qualified in the SM",
    ((says("$\\eta_{\\rm all}=46.02", "$\\eta_{k=10}=35.05$",
           "An unconditional probability must not be divided by this conditional null",
           "the bootstrap covers shot noise but not drift") if V5 else
      says("$\\eta_{\\rm all}=46.02", "$\\eta_{k=10}=35.05$",
           "$4.3628\\times10^{-18}$", "cannot diagnose temporal drift or overdispersion",
           "one-sided exact binomial test"))
    and "coherent primary baseline" not in _tex))
chk("prose: 51q distinguishes marginal post coverage from encoded feasibility",
    (says("no sampled string satisfying all base--slack equalities",
          "base-register cover fractions", "marginal diagnostics rather than feasibility probabilities",
          "$1.198$ and $1.155$") if V5 else
     says("matched enhancement of $1.198$", "none of the $10^5$ full 51-bit strings",
          "marginal base-register cover rate")))
chk("prose: Rydberg is retained only as a negative encoding audit",
    says("blockade interaction suppresses simultaneous excitations",
         "implements an independence-type exclusion",
         "the four corners as its unique ground configuration",
         "transient overlap with externally enumerated target strings",
         "no claim of optimization gain"))
chk("prose: no native-domination claim survives the audit",
    "carry the domination constraint" not in _tex
    and "native encoding of the domination constraint" not in _tex
    and says("zero minimum dominating sets and zero dominating sets"))
chk("prose: the error model is named a success proxy, not a fidelity",
    says("first-order \\emph{success proxy}", "rather than a process fidelity"))

# ---- second-round referee corrections ----------------------------------------
chk("prose: the Rung-3 claim rules out the low-width route, not all classical simulation",
    (says("does not turn the $2^{O(pw)}$ contraction upper bound into a complexity lower bound",
          "does not exclude algorithms exploiting other structure",
          "This is a resource conclusion, not a lower bound against all classical simulation methods") if V5 else
     says("This is no lower bound on classical simulation in general, only on tree-decomposition contraction for exact sampling"))
    and "costs at least $2^{112}$" not in _tex)
chk("prose: the 2-local obstruction is scoped to the bare register",
    (says("the standard constraint-local faithful penalty is not 2-local on the bare register",
          "does not rule out a dense, instance-specific quadratic Hamiltonian") if V5 else
     says("no exact 2-local cost on the bare register", "on the $\\nB$ base variables alone")))
chk("prose: the Dicke-preparation claim no longer says a global rotation suffices",
    says("A global rotation does not prepare")
    and "prepared by a global rotation" not in _tex)
chk("prose: sparsity is not asserted to imply disconnection",
    says("a path on 24 variables has 23 interactions and one component")
    and "sparse interaction graph is a disconnected one" not in _tex)
chk("prose: no absolute 'no device can run' claim survives",
    "no device can run" not in _tex
    and (says("beyond current superconducting hardware before routing") if V5 else
         says("lies far beyond current superconducting hardware")))
chk("prose: resource proxy is not promoted to an output-probability model",
    says("not a calibrated stochastic output model", "do not convert $\\Fest=0.817$"))
chk("prose: the advantage claim is scoped to the simulations actually identified",
    (says("general impossibility result for quantum optimization",
          "does not claim that all range settings or all geographic coverage families behave likewise",
          "Nothing below is a claim of quantum advantage") if V5 else
     says("nonlocal correlation structure needed to evade the factorization and low-width"))
    and "entanglement a quantum advantage would have to come from" not in _tex)
chk("prose: the two-seed pilot is still reported in the SM with its 3--3 splits",
    says("median probability ratio is 0.975", "only two independent task seeds",
         "the complete control has lower energy in four of six cases"))
chk("prose: abstract and conclusion lead with the controlled negative mixer result",
    (says("A same-sector control over twenty task seeds found no detectable benefit",
          "The large uncontrolled QWOA/QAOA probability ratio is primarily a cardinality prior",
          "shared-task weights yield no detectable effect over twenty seeds") if V5 else
     says("because it is handed the optimal cardinality",
          "in a same-sector control over twenty task seeds, its problem-weighted mixer shows no detectable gain",
          "identifies the large uncontrolled QWOA/QAOA ratio as a cardinality prior",
          "the geographic weights produce no detectable gain; the cardinality information produces almost all of it")))
chk("editorial: superseded target-assisted figures are absent from the manuscript",
    "fig_advantage_vs_m.pdf" not in _tex
    and "fig_qwoa_vs_qaoa.pdf" not in _tex
    and "Legacy target-assisted scans" not in _tex
    and "Cardinality descent: sector sizes" not in _tex)
chk("editorial: the constrained mixer is audited rather than advocated",
    "ansatz we advocate" not in _tex
    and "cardinality-preserving mixer of" not in _tex
    and says("tests for and does not find a benefit from the problem-informed mixer",
             "constrained-mixer audit is also deliberately narrow"))

chk("prose: the width inequality is stated as elementary, separate from the scaling",
    (says("Retaining endpoints at vertex coordinates gives a finite path decomposition",
          "The density hypothesis is local and deterministic",
          "In a bounded-aspect planar region") if V5 else
     says("The inequality is elementary; its content lies in how $W$ grows",
          "the planar one also a region of bounded aspect ratio; the inequality above does not")))
chk("prose: the two width bounds are called a consistency check, not a validation",
    says("their\nproximity is a consistency check rather than a validation".replace("\n", " "),
         "their proximity is a consistency check rather than a validation"))
chk("prose: the expander counterexample bounds the thesis, in abstract and discussion",
    (says("A family of 3-regular expanders on $n$ vertices has $\\nzz=1.5n$",
          "treewidth $\\Theta(n)$", "sparse expanders need not have such an embedding",
          "the expander example concerns exact output sampling, not energy estimation") if V5 else
     says("A family of 3-regular expanders on $n$ vertices has $\\nzz=1.5n$",
          "treewidth $\\Theta(n)$", "any\nproblem class whose QUBO is an expander would escape every mechanism reported here".replace("\n", " "),
          "sparse expanders escape the argument")))

# =============================================================================
# ---- version 4: third-round referee revision ----------------------------------
# =============================================================================
hd = J("hardware_decomposition.json")
sd = hd["rung1_sector_decomposition"]

# 2.1: where the Rung-1 deficit sits
chk("v4: the 43.3% factorises into sector mass 0.879 x within-sector 0.492",
    close(sd["sector_mass_ratio"], 0.8789, 5e-4) and close(sd["within_sector_ratio"], 0.4924, 5e-4)
    and close(sd["total_ratio"], sa_p1["hardware_to_ideal_absolute_probability_ratio"], 1e-9),
    f"{sd['sector_mass_ratio']:.3f} x {sd['within_sector_ratio']:.3f} = {sd['total_ratio']:.4f}")
chk("v4: ideal P(k=10) = 0.1747 against measured 0.1535 and uniform 0.1169",
    close(sd["p_k10_ideal"], 0.17466, 5e-5) and close(sd["p_k10_hw"], 0.15351, 1e-9)
    and close(sd["p_k10_uniform"], 0.11690, 5e-5)
    and close(sa_p1["ideal_p_k_opt"], sd["p_k10_ideal"], 1e-9))
chk("v4: ideal in-sector concentration is 71.2, not ~100",
    close(sd["eta_k10_ideal"], 71.17, 0.01) and close(sd["eta_k10_hw"], 35.05, 0.01))
chk("v4: 85% of the logarithmic deficit is intra-sector; depolarising reading predicts ~87",
    close(sd["log_share_within_sector"], 0.846, 0.005)
    and close(sd["global_depolarising_eta_all"], 87.0, 0.1))
chk("prose: the sector decomposition is stated with its numbers",
    says("\\label{eq:sector_split}", "=0.879\\times0.492", "$\\eta_{k=10}^{\\rm ideal}=71.2$",
         "$85\\%$ of the logarithmic deficit is intra-sector",
         "$\\eta_{\\rm all}\\approx0.817\\times106.3+0.183\\approx87$"))

# the p=2,3 depth series is not a decoherence series
_id = {p: sa_r1[f"rung1_qaoa_p{p}"] for p in (1, 2, 3)}
chk("v4: ideal tiled-angle concentration already falls 106.3 -> 27.3 -> 9.9",
    all(close(_id[p]["ideal_eta_all_register"], v, 0.05)
        for p, v in ((1, 106.34), (2, 27.28), (3, 9.92))))
chk("v4: at p=2,3 the device matches or exceeds the ideal (ratios 1.03, 1.53)",
    close(_id[2]["hardware_to_ideal_absolute_probability_ratio"], 1.025, 0.005)
    and close(_id[3]["hardware_to_ideal_absolute_probability_ratio"], 1.527, 0.005))
chk("prose: the depth series is no longer called a decoherence control",
    says("the ideal tiled-angle states themselves fall to $\\eta_{\\rm all}^{\\rm ideal}=106.3,\\,27.3,\\,9.9$",
         "hardware-to-ideal ratios $1.03$ and $1.53$",
         "rather than depth-induced decoherence")
    and "qualitative depth-degradation control" not in _tex)

# 2.8: matched-mitigation 51-qubit run and energy nulls
ex51m = J("exact51_matched_results.json")
sa_51m = sa["faithful_51q_matched_mitigation"]
chk("v4: matched 51q job is dat665lvr3kc73el167g with XY4 + twirling, same 372 gates",
    ex51m["job_id"] == "dat665lvr3kc73el167g" and ex51m["mitigation"].startswith("XY4")
    and ex51m["physical_2q"] == 372 and ex51m["physical_depth"] == 251)
chk("v4: matched run reduces energy 36.6% (75% of ideal) against 39.7% (82%)",
    close(ex51m["reduction_pct"], 36.59, 0.01)
    and close(hd["faithful_matched"]["gap_closed_vs_ideal"], 0.753, 0.001)
    and close(hd["faithful_default"]["gap_closed_vs_ideal"], 0.816, 0.001))
chk("v4: matched run post-cover 0.4326 vs card.-matched null 0.3745; no encoded equality",
    close(sa_51m["post_bits_form_cover"]["probability"], 0.4326, 1e-9)
    and close(sa_51m["post_bits_form_cover"]["cardinality_matched_uniform_within_sector_baseline"],
              0.37447, 1e-5)
    and sa_51m["all_measured_slack_equalities_satisfied"]["count"] == 0)
chk("v4: matched job-time proxy F_est = 0.306 (readout 0.652)",
    close(rx["faithful_exact51_p1_matched"]["F_exact"], 0.3060, 5e-4)
    and close(rx["faithful_exact51_p1_matched"]["F_readout"], 0.6521, 5e-4)
    and rx["faithful_exact51_p1_matched"]["n_2Q"] == 372)
chk("v4: a product-marginal sampler reproduces ~90% of both measured reductions (85% ideally)",
    close(hd["faithful_default"]["product_null_share_of_hw_reduction"], 0.896, 0.005)
    and close(hd["faithful_matched"]["product_null_share_of_hw_reduction"], 0.901, 0.005)
    and close(hd["faithful_ideal"]["product_null_share_of_ideal_reduction"], 0.849, 0.005))
chk("v4: correlation energy 29.8 ideal, 16.7 / 14.8 measured -> 56% / 50% retained",
    close(hd["faithful_ideal"]["correlation_energy_ideal"], 29.80, 0.01)
    and close(hd["faithful_default"]["correlation_energy_hw"], 16.71, 0.01)
    and close(hd["faithful_matched"]["correlation_energy_hw"], 14.76, 0.01)
    and close(hd["faithful_default"]["correlation_energy_retained_vs_ideal"], 0.561, 0.005)
    and close(hd["faithful_matched"]["correlation_energy_retained_vs_ideal"], 0.495, 0.005))
chk("v4: for the 24q circuit the product null reaches only 43% of the reduction",
    close(hd["rung1_energy_null"]["product_null_share_of_hw_reduction"], 0.433, 0.005))
chk("prose: the 51q hardware figure of merit is the retained correlation energy",
    ((says("dat665lvr3kc73el167g", "product distribution with the measured marginals reproduces about $90\\%$",
           "the two runs retain $56\\%$ and $50\\%$", "correlation energy $E_{\\rm prod}-E$") if V5 else
      says("dat665lvr3kc73el167g", "$36.6\\%$ ($75\\%$ of the ideal reduction)",
           "already reaches $90\\%$ of the measured reduction",
           "the hardware retains $56\\%$ of it", "$75$--$82\\%$ of the ideal energy gap",
           "retained correlation energy"))
    and "closes $82\\%$ of the energy gap closed by its ideal circuit" not in _tex))

# 2.3: the seed-17 rows differ because beta*_raw * RMS leaves the beta window
_ctl = {(r["seed"], r["m"]): r for r in J("same_sector_mixer_control.json")["rows"]}
_scl = {(r["seed"], r["m"]): r for r in J("mixer_scale_sensitivity.json")["rows"]}
_bxr = {k: _ctl[k]["controls"]["problem_weighted"]["parameters"][1] * _scl[k]["raw_weight_rms"]
        for k in _ctl}
_out = sorted(k for k, v in _bxr.items() if v > np.pi)
_same = all(close(_ctl[k]["controls"]["problem_weighted"]["metrics"]["expected_energy"],
                  _scl[k]["rms_normalized_weighted"]["metrics"]["expected_energy"], 1e-6)
            for k in _ctl if k not in _out)
chk("v4: pilot endpoints coincide exactly iff beta*_raw * RMS <= pi (fails only seed 17, m=1,2)",
    _out == [(17, 1), (17, 2)] and _same
    and close(_bxr[(17, 1)], 29.46, 0.05) and close(_bxr[(17, 2)], 21.59, 0.05),
    ", ".join(f"{_bxr[k]:.1f}" for k in _out))
chk("prose: the scale-equivalence criterion is stated in the pilot table",
    says("\\beta^\\star_{\\rm raw}\\,\\mathrm{RMS}\\le\\pi", "=29.5$ and $21.6$", "same?"))

# 2.5: twenty seeds
q20 = J("qwoa_20seeds.json")
qs20, rows20 = q20["summary"], q20["rows"]
_pilot = {(r["seed"], r["m"]): r for r in J("qwoa_convergence_audit.json")["p1"]["rows"]}
chk("v4: 60 rows from 20 prespecified seeds that contain the pilot seeds",
    len(rows20) == 60 and qs20["n_independent_seeds"] == 20
    and {12, 17} <= set(q20["meta"]["seeds"])
    and q20["meta"]["seeds"]
    == np.random.default_rng(20260915).choice(24, 20, replace=False).tolist())
chk("v4: the six pilot endpoints are reproduced exactly",
    all(close(r["searches"]["weighted"]["expected_energy"],
              _pilot[(r["seed"], r["m"])]["qwoa_expected_energy"], 1e-6)
        and close(r["searches"]["qaoa"]["expected_energy"],
                  _pilot[(r["seed"], r["m"])]["qaoa_expected_energy"], 1e-6)
        for r in rows20 if (r["seed"], r["m"]) in _pilot)
    and sum((r["seed"], r["m"]) in _pilot for r in rows20) == 6)
chk("v4: convergence 60/60 for QAOA, W_RMS, U, K and 51/60 for raw W",
    qs20["jointly_converged"] == {"qaoa": 60, "weighted": 51, "weighted_rms": 60,
                                  "support": 60, "complete": 60})
_qq = qs20["qwoa_vs_qaoa"]
chk("v4: the uncontrolled ratio is the inverse sector mass (median 7.1e4 vs P_Q(k*) 1.68e-4)",
    _qq["qaoa_lower_energy"] == 60 and close(_qq["median_unconditional_ratio"], 7.13e4, 0.01e4)
    and close(_qq["median_qaoa_p_k_opt"], 1.677e-4, 0.005e-4)
    and close(_qq["median_conditional_ratio"], 2.82, 0.01) and _qq["qwoa_conditional_wins"] == 43)
_prim = qs20["weighted_rms_vs_support"]
chk("v4: primary contrast 39/60 energy, 36/60 P(opt), median ratio 1.02 (IQR 0.91-1.11)",
    (_prim["a_lower_energy"], _prim["a_larger_p_opt"]) == (39, 36)
    and close(_prim["median_p_opt_ratio"], 1.021, 0.001)
    and np.allclose(_prim["p_opt_ratio_iqr"], [0.911, 1.109], atol=0.002))
_sc = qs20["seed_clustered"]
_minp = min(min(v["wilcoxon_energy_two_sided"], v["wilcoxon_p_opt_two_sided"],
                v["sign_test_energy_two_sided"], v["sign_test_p_opt_two_sided"])
            for v in _sc.values())
chk("v4: no seed-level contrast reaches p < 0.15",
    _minp >= 0.15
    and (_sc["weighted_rms_vs_support"]["a_lower_energy_seeds"],
         _sc["weighted_rms_vs_support"]["a_larger_p_opt_seeds"]) == (12, 12),
    f"min seed-level p = {_minp:.3f}")
chk("v4: the row-level suggestions exist and are reported as such (p=0.027, p=0.005)",
    close(qs20["weighted_rms_vs_support"]["sign_test_energy_two_sided"], 0.0273, 5e-4)
    and close(qs20["weighted_vs_complete"]["sign_test_p_opt_two_sided"], 0.0046, 5e-4))
chk("v4: scale-equivalence holds in 49 of 60 rows", qs20["scale_equivalent_rows"] == 49)
chk("prose: the twenty-seed result is stated with its numbers",
    says("36 of 60 rows, with median ratio $1.02$ (interquartile range $0.91$--$1.11$)",
         "wins 12 of 20 in both energy and probability",
         "(Wilcoxon $p=0.16$ for energy and $0.26$ for $P(\\mathrm{opt})$)",
         "median ratio $7.1\\times10^4$", "$1.68\\times10^{-4}$",
         "sign test $p=0.027$", "36 of 51 ($p=0.005$)", "holds in 49 of 60 rows",
         "\\label{tab:sm_qwoa20}"))

# 2.6, 2.7, 2.9, 2.10
chk("prose: the QWOA/QAOA energy ordering is explained by the sector, not claimed forced",
    says("largely explained by the sector restriction rather than by the mixers", "$2+13=15$",
         "The ordering is not forced") and "structurally forced" not in _tex)
chk("prose: the Rydberg conclusion is scoped to the direct encoding and cites the gadget route",
    says("\\emph{direct} vertex-to-atom encoding", "not of the platform", "nguyen2023quantum",
         "& none identified \\\\")
    and "& Neutral atoms \\\\" not in _tex)
chk("prose: the expander bounds sampling, not energy; the task is named per regime",
    (says("the expander example concerns exact output sampling, not energy estimation",
          "Exact sampling by sequential conditional marginals",
          "For disconnected blocks, exact factorization is stronger than the width bound") if V5 else
     says("what the expander defeats is exact \\emph{sampling}",
          "Under exact factorization, sampling the output is trivial",
          "one conditional marginal per variable")))
_bib = (PAPER_DIR / "refs.bib").read_text(encoding="utf-8")
_newrefs = ("nguyen2023quantum", "barak2022classical", "hastings2019classical",
            "bravyi2020obstacles", "dupont2022calibrating", "medvidovic2021classical",
            "lykov2023sampling")
chk("v4: the added literature is cited and present in refs.bib",
    all(k in _tex and f"{{{k}," in _bib for k in _newrefs))

# Sec. 3 editorial items
chk("editorial: the v3 migration bugs and self-references are gone",
    "rather than here" not in _tex
    and says("Sec.~\\ref{sec:rung2} shows it is not sufficient on its own",
             "&&\\forall b, \\end{aligned}",
             "whose exponential part is independent of register size",
             "Eqs.~(\\ref*{main:eq:unitary_factor}) and (\\ref*{main:eq:factor})",
             "device-median model of Eq.~(\\ref*{main:eq:resource}) at $p=1$",
             "Block-locality itself is an identity")
    and "coherent all-register" not in _tex and "coherent primary" not in _tex)

# promoted figures and the range dial
dsw = {r["R_km"]: r for r in J("dial_sweep.json")["rows"]}
chk("v4: the dial figure's structure matches the squeeze table (62 -> 2^24)",
    [(dsw[R]["n_ZZ"], dsw[R]["blocks"], dsw[R]["largest"], dsw[R]["state_dim"])
     for R in (200, 350, 600)]
    == [(17, 13, 4, 62), (49, 4, 12, 4194), (121, 1, 24, 2 ** 24)])
chk("v4: the instance stays a millisecond-scale solve at every radius",
    max(r["ilp_ms"] for r in dsw.values()) < 1.0,
    f"max median {max(r['ilp_ms'] for r in dsw.values()):.2f} ms over {len(dsw)} radii "
    "(a bound; wall times are machine-dependent)")
_main = (PAPER_DIR / MAIN_TEX).read_text(encoding="utf-8")
chk("editorial: two supplementary figures are promoted to the main text",
    "{fig_dial.pdf}" in _main and "{fig5_squeeze.pdf}" in _main
    and "fig5_squeeze.pdf" not in (PAPER_DIR / SUPP_TEX).read_text(encoding="utf-8"))
_abs = _main[_main.index("\\begin{abstract}") + 16:_main.index("\\end{abstract}")]
_nw = len(re.sub(r"\$[^$]*\$", "X", _abs).split())
chk("editorial: the abstract is shortened (<= 330 words) and drops the Wilson/82% detail",
    _nw <= 330 and "Wilson" not in _abs and "82" not in _abs, f"{_nw} words")

# =============================================================================
# ---- version 4, fourth-round editorial pass -----------------------------------
# =============================================================================
chk("editorial r4: the Fig. 1 caption is no longer truncated",
    says("Panels (a) and (b) rise together, (b) by five orders of magnitude; (c) does not move.")
    and "(a) and (b) rise together by five orders of magnitude in (b)" not in _tex)
chk("editorial r4: phi is justified once (main text), not twice",
    "It is kept out of the main text" not in _tex
    and says("it carries no instance-level signal once register size and radius are controlled"))
_smbib = _bib[_bib.index("@misc{SM,"):]
_smbib = _smbib[:_smbib.index("\n}")]
chk("editorial r4: the Supplemental Material reference describes the v4 contents",
    "prefactor" not in _smbib and "twenty-seed" in _smbib and "sector decomposition" in _smbib
    and "prefactor" not in md["SUPPDATA"] and "twenty-seed" in md["SUPPDATA"])
chk("editorial r4: Table S8 explains n = 60 versus 51",
    says("The row count $n$ is 60 for contrasts with $W_{\\rm RMS}$ and 51"))
chk("editorial r4: the depth series is cited as the audit's own example (Sec. VIII A)",
    says("The check also pays after the run.", "$\\eta_{\\rm all}=46\\to28\\to15$",
         "the decline belongs to the ansatz and not to the device"))
chk("r4: the two faithful runs differ by cardinality shift, not structure",
    ((says("enhancements nearly coincide", "$1.198$ against $1.155$",
           "the difference is dominated by the shift in base cardinality") if V5 else
      says("the matched enhancement barely moves ($1.155$ against $1.198$)",
           "the difference is dominated by the shift in base cardinality"))
    and close(sa_51["post_bits_form_cover"]["cardinality_matched_enhancement"], 1.198, 5e-4)
    and close(sa_51m["post_bits_form_cover"]["cardinality_matched_enhancement"], 1.155, 5e-4)
    and close(sa_51["mean_post_cardinality"], 20.07, 0.005)
    and close(sa_51m["mean_post_cardinality"], 19.38, 0.005)))

bn = {c["circuit"]: c for c in J("block_product_null.json")["circuits"]}
b1 = bn["rung1_qaoa_p1"]
chk("r4: the per-block product reproduces the ideal P(opt) exactly (the ideal state is a block product)",
    all(close(bn[f"rung1_qaoa_p{p}"]["ideal_p_opt_product"], sa_r1[f"rung1_qaoa_p{p}"]["ideal_p_opt"], 1e-12)
        for p in (1, 2, 3)))
chk("r4: block-product null predicts 96.7 +/- 9.8 hits against 79 at p=1 (permutation p=0.07)",
    b1["observed_opt_hits"] == 79 and close(b1["block_product_null_hits"], 96.73, 0.01)
    and close(b1["permutation"]["mean_hits"], 96.7, 0.3) and close(b1["permutation"]["sd_hits"], 9.8, 0.2)
    and 0.05 < b1["permutation"]["p_two_sided"] < 0.10,
    f"p = {b1['permutation']['p_two_sided']:.4f}")
chk("r4: at p=2,3 observation and block-product null agree",
    all(bn[f"rung1_qaoa_p{p}"]["permutation"]["p_two_sided"] > 0.25 for p in (2, 3)))
chk("r4: no cross-block correlation beyond |phi| = 0.01; independent blocks give the k=10 mass",
    all(c["cross_block"]["max_abs_phi"] < 0.01 for c in bn.values())
    and close(b1["cross_block"]["p_k10_independent_blocks"], 0.1524, 5e-4)
    and close(b1["cross_block"]["p_k10_measured"], 0.1535, 5e-4))
_rb = b1["retention_by_block_size"]
chk("r4: 0.433 = 0.530 within-block x 0.817 cross-block; loss tracks entangling content",
    close(b1["product_of_block_ratios"], 0.530, 0.001) and close(b1["cross_block_factor"], 0.817, 0.001)
    and close(b1["product_of_block_ratios"] * b1["cross_block_factor"],
              sa_p1["hardware_to_ideal_absolute_probability_ratio"], 1e-9)
    and _rb["1"]["n_blocks"] == 7 and _rb["1"]["min"] > 0.975 and _rb["1"]["max"] < 1.015
    and close(_rb["4"]["product"], 0.701, 0.001)
    and close(np.log(_rb["4"]["product"]) / np.log(b1["product_of_block_ratios"]), 0.56, 0.005))
chk("prose: the block-local mechanism is stated with its numbers (main, abstract, SM)",
    ((says("It predicts 96.7 optimum hits against the 79 observed",
           "no pair of block indicators has a correlation coefficient above $0.01$ in magnitude",
           "retains $0.70$ and alone accounts for $56\\%$ of the within-block logarithmic loss",
           "The residual cross-block factor is $79/96.7=0.82$, with bootstrap 95\\% interval $[0.65,1.00]$",
           "For the six blocks with two-qubit gates", "$0.433$ ratio is therefore $0.879\\times0.492$",
           "\\label{tab:sm_blocknull}") if V5 else
      says("It predicts 96.7 optimum hits against the 79 observed",
           "no pair of block indicators has a correlation coefficient above $0.01$ in magnitude",
           "retains $0.70$ and alone accounts for $56\\%$ of the within-block logarithmic loss",
           "The residual cross-block factor is $79/96.7=0.82$, with bootstrap 95\\% interval $[0.65,1.00]$",
           "inside the blocks that carry two-qubit gates",
           "$0.433=0.530\\times0.817$", "\\label{tab:sm_blocknull}"))
    and "consistent with noise acting mainly on the within-block correlations" not in _tex))

# =============================================================================
# ---- Fig. 3: block structure from the map to the chip -------------------------
# =============================================================================
import itertools as _it
import networkx as _nx
from calc_block_factorisation import blocks_of as _blocks_of, load as _load_r1

_d1, _A1, _h1, _J1 = _load_r1()
_blocks1 = [sorted(int(q) for q in b) for b in _blocks_of(24, _J1)]
_lab1 = {q: k for k, b in enumerate(_blocks1) for q in b}
_lay = J("rung1_physical_layout.json")
_l2p = {int(k): v for k, v in _lay["circuits"]["rung1_qaoa_p1"]["logical_to_physical"].items()}
_gates = [tuple(g["edge"]) for g in _lay["circuits"]["rung1_qaoa_p1"]["two_qubit_edges"]]
_p2l = {v: k for k, v in _l2p.items()}
chk("fig3: the archived layout is the executed p=1 circuit (same 24 qubits, 49 gates)",
    _lay["job_id"] == "dakjch39k43c73aik9rg"
    and sorted(_l2p.values()) == sorted(rx["rung1_qaoa_p1"]["measured_qubits"])
    and sum(g["count"] for g in _lay["circuits"]["rung1_qaoa_p1"]["two_qubit_edges"]) == 49)
chk("fig3: every physical two-qubit gate stays inside one logical block",
    all(a in _p2l and b in _p2l and _lab1[_p2l[a]] == _lab1[_p2l[b]] for a, b in _gates))
_cm = [tuple(e) for e in _lay["coupling_map"]]
_cross = [(a, b) for a, b in _cm if a in _p2l and b in _p2l and _lab1[_p2l[a]] != _lab1[_p2l[b]]]
_Hc = _nx.Graph(_cm)
_mind = min(_nx.shortest_path_length(_Hc, a, b) for a, b in _it.combinations(_p2l, 2)
            if _lab1[_p2l[a]] != _lab1[_p2l[b]])
chk("fig3: no coupler joins qubits of different blocks (closest are two couplers apart)",
    not _cross and _mind == 2 and len(_cm) == 176)
_la, _lo = np.radians(_d1["base_lats"]), np.radians(_d1["base_lons"])
_dist = lambda i, j: 2 * 6371 * np.arcsin(np.sqrt(
    np.sin((_la[j] - _la[i]) / 2) ** 2 + np.cos(_la[i]) * np.cos(_la[j]) * np.sin((_lo[j] - _lo[i]) / 2) ** 2))
_inter = [_dist(i, j) for i, j in _it.combinations(range(24), 2) if _lab1[i] != _lab1[j]]
chk("fig3: 31 inter-block base pairs lie within 2R, the closest 35 km apart",
    sum(x <= 400 for x in _inter) == 31 and close(min(_inter), 35.2, 0.1))
_sep = J("ibm_block_results/kingston_blocks_analysis.json")
_sepb = {tuple(sorted(b["original_qubits"])): b for b in _sep["blocks"]}
_sep_prod = float(np.prod([b["p_opt"] / b["p_opt_ideal"] for b in _sep["blocks"]]))
chk("fig3: separate execution retains 0.784 of the ideal product (4-qubit block 0.932)",
    _sep["job_id"] == "dan62g5r85ps73fdlp00" and len(_sep["blocks"]) == 13
    and set(_sepb) == {tuple(b) for b in _blocks1}
    and close(_sep_prod, 0.784, 0.001)
    and close(_sepb[(2, 3, 4, 5)]["p_opt"] / _sepb[(2, 3, 4, 5)]["p_opt_ideal"], 0.932, 0.001)
    and all(b["shots"] == 10000 for b in _sep["blocks"]))
_r51 = rx["faithful_exact51_p1"]
_G51 = _nx.Graph(); _G51.add_nodes_from(_r51["measured_qubits"])
_G51.add_edges_from(e["edge"] for e in _r51["two_qubit_edges"])
chk("51q: routing merges logical blocks 13+9+5 into one 27-qubit physical component",
    sorted((len(c) for c in _nx.connected_components(_G51)), reverse=True)
    == [27, 7, 6, 4, 1, 1, 1, 1, 1, 1, 1])
chk("prose: Fig. 3 replaces the PEF map and the placement/separate-run text is present",
    "{fig_blocks.pdf}" in _main and "fig1_pef_map.pdf" not in _tex
    and says("the transpiler placed the 13 blocks on disjoint qubit sets with no coupler between them",
             "Sep 19 & $C$ & --- & 0.784",
             "the closest $35$ km apart", "dan62g5r85ps73fdlp00",
             "overlap on 27 physical qubits", "Hardware data come from eight"))

import geopandas as _gpd
from shapely.geometry import Point as _Pt
_outline = _gpd.read_file(DATA_DIR / "legal_amazon_outline_2024.geojson")
_poly = _outline.geometry.iloc[0]
_tout = [(_poly.boundary.distance(_Pt(lo, la)) * 111.2)
         for la, lo in zip(_d1["task_lats"], _d1["task_lons"]) if not _poly.contains(_Pt(lo, la))]
_bout = [(_poly.boundary.distance(_Pt(lo, la)) * 111.2)
         for la, lo in zip(_d1["base_lats"], _d1["base_lons"]) if not _poly.contains(_Pt(lo, la))]
chk("fig3: the outline is the hash-pinned IBGE 2024 product",
    _outline["source_sha256"].iloc[0]
    == "1af12c71fd1eb33f3eaa3265a26f767ef189efe40272104fd0968cd54385dea4"
    and int(_outline["n_municipalities"].iloc[0]) == 773)
chk("fig3: 11 of 23 synthetic waypoints lie outside the IBGE boundary, at most ~92 km",
    len(_tout) == 11 and 90 < max(_tout) < 94,
    f"{len(_tout)} outside, max {max(_tout):.1f} km; bases outside: "
    + ", ".join(f"{x:.1f}" for x in sorted(_bout)) + " km")
chk("prose: the unrestricted waypoint draw is disclosed in Sec. IV A, Methods and the caption",
    says("11 of the 23 waypoints fall outside the IBGE boundary, up to 92 km across the border",
         "waypoints outside it are discussed in Sec.~\\ref{ssec:pef}"))

# the separate execution is not predicted by better calibration
_rx1 = rx["rung1_qaoa_p1"]
_ro1 = {int(k): v for k, v in _rx1["readout_errors"].items()}
_ge1 = {tuple(sorted(e["edge"])): (e["error"], e["count"]) for e in _rx1["two_qubit_edges"]}
_hwsep = {b["tag"]: b for b in _sep["hardware_job"]["blocks"]}
_ratio = {}
for _b in _blocks1:
    _ph = [_l2p[q] for q in _b]
    _fj = float(np.prod([(1 - e) ** c for (a, b2), (e, c) in _ge1.items() if a in _ph and b2 in _ph])
                * np.prod([1 - _ro1[p] for p in _ph]))
    _fs = _hwsep[_sepb[tuple(_b)]["tag"]]["F_exact"]
    _ratio[tuple(_b)] = (_fj, _fs)
_Fj = float(np.prod([v[0] for v in _ratio.values()]))
_Fs = float(np.prod([v[1] for v in _ratio.values()]))
chk("fig3: calibration proxy predicts joint and separate placements as equivalent (0.817 vs 0.822)",
    close(_Fj, _rx1["F_exact"], 1e-6) and close(_Fj, 0.817, 0.0005) and close(_Fs, 0.822, 0.0005)
    and max(abs(fs / fj - 1) for fj, fs in _ratio.values()) < 0.006
    and close(_ratio[(2, 3, 4, 5)][0], 0.944, 0.0005) and close(_ratio[(2, 3, 4, 5)][1], 0.945, 0.0005),
    f"max per-block proxy difference {100 * max(abs(fs / fj - 1) for fj, fs in _ratio.values()):.2f}%")
chk("prose: the calibration-proxy comparison is stated in the main text and SM",
    (says("ranks the three implementations in the opposite order ($0.949$, $0.948$ and $0.931$)",
          "reported two-qubit and readout errors do not explain the contrast") if V5 else
     says("ranks the two placements the wrong way round: $0.949$ and $0.948$ against $0.931$",
          "so the reported gate and readout errors do not predict the gain")))

# ---- the 51-qubit circuit: light cones and cross-block correlation (Sec. IV C, Fig. 4) ----
x51 = J("exact51_cross_block.json")
_runs51 = x51["runs"]
chk("51q: in both runs exactly three block pairs (13-9, 13-5, 9-5) have overlapping light cones",
    all(r["cone_pairs"] == [[0, 1], [0, 4], [1, 4]] and r["cone_wires"] == 27 for r in _runs51.values())
    and [b["size"] for b in x51["blocks"]][:6] == [13, 9, 7, 6, 5, 4]
    and x51["rung1_reference"]["overlapping_cone_pairs"] == 0)
_shared = {t: {tuple(p["blocks"]): p["shared_two_qubit_gates"] for p in r["pairs"] if p["cls"] == "cone"}
           for t, r in _runs51.items()}
chk("51q: the overlapping cones share 135 two-qubit gates (72 in all three)",
    all(v == {(0, 1): 72, (0, 4): 72, (1, 4): 135} for v in _shared.values()), str(_shared))
_rz = {t: sorted(round(p["z"], 1) for p in r["pairs"] if p["cls"] == "cone") for t, r in _runs51.items()}
chk("51q: the two most correlated of 78 pairs lie among the cone pairs (z 4.0, 3.7; 3.3, 6.2)",
    all(r["top_two_in_cone_pairs"] and close(r["p_top_two_in_cone_pairs_if_exchangeable"], 1e-3, 1e-6)
        for r in _runs51.values())
    and _rz == {"defaults": [0.5, 3.7, 4.0], "matched": [0.6, 3.3, 6.2]}
    and all(p["permutation_p_two_sided"] <= 9e-4 for r in _runs51.values()
            for p in r["pairs"] if p["cls"] == "cone" and p["blocks"] != [0, 4])
    and all(p["permutation_p_two_sided"] >= 0.57 for r in _runs51.values()
            for p in r["pairs"] if p["blocks"] == [0, 4]),
    str(_rz))
chk("51q: disjoint-cone pairs stay below |z| = 2.6, the adjacent one below 0.3; 3 of 4 cone z pass Bonferroni",
    all(r["summary"]["distant"]["max_abs_z"] < 2.6 and r["summary"]["adjacent"]["max_abs_z"] < 0.3
        and r["summary"]["distant"]["n_above_bonferroni"] == 0 for r in _runs51.values())
    and close(x51["z_bonferroni_5pct"], 3.41, 0.01)
    and sum(r["summary"]["cone"]["n_above_bonferroni"] for r in _runs51.values()) == 3
    and close(x51["rung1_reference"]["max_abs_z"], 3.2, 0.05))
chk("51q: the effect is weak (r <= 0.02; cross-block covariance ~0.5% of energy variance)",
    all(abs(p["r"]) <= 0.02 for r in _runs51.values() for p in r["pairs"] if p["cls"] == "cone")
    and all(0.004 < r["sum_cross_block_energy_covariance"] / r["total_energy_variance"] < 0.006
            for r in _runs51.values()))
chk("51q: the measurement-time classification is kept for comparison (36 gates, misses 13-5)",
    all(r["cross_block_two_qubit_gates"] == 36 and r["routed_pairs"] == [[0, 1], [1, 4]]
        for r in _runs51.values()))
chk("prose: the light-cone reading is stated in Sec. IV C and the SM, without the crosstalk overclaim",
    ((says("three of the 78 block pairs have overlapping cones",
           "Two of those pairs carry the largest observed correlations",
           "the third overlapping pair is not correlated",
           "which gives no evidence of readout crosstalk on that coupler",
           "\\label{fig:exact51}", "following every routing SWAP") if V5 else
      says("exactly three of the 78 pairs overlap", "which gives no evidence of readout crosstalk on that coupler",
           "\\label{fig:exact51}", "following every routing SWAP"))
    and "excludes readout crosstalk" not in _tex and "{fig_exact51_routing.pdf}" in _main))

# ---- fifth round: joint and separate execution in one batch, same qubits --------
bt = J("joint_separate_batch.json")
_bb = {tuple(r["qubits"]): r for r in bt["blocks"]}
chk("batch: one job, joint circuit and 13 slices with identical gates on identical qubits",
    bt["job_id"] == "dau04mahcrkc73dur170" and len(bt["blocks"]) == 13
    and sum(r["n_2q"] for r in bt["blocks"]) == 47 and bt["joint_opt_hits"] == 100
    and close(bt["joint_over_ideal"], 0.548, 0.001))
chk("batch: joint execution costs ~9% of within-block retention, 0.906 [0.848, 0.965]",
    close(bt["within_block_retention_joint"], 0.542, 0.001)
    and close(bt["within_block_retention_separate"], 0.599, 0.001)
    and close(bt["joint_over_separate"], 0.906, 0.001)
    and bt["joint_over_separate_ci95"][1] < 1.0 and close(bt["joint_over_separate_ci95"][0], 0.848, 0.01),
    str([round(x, 3) for x in bt["joint_over_separate_ci95"]]))
chk("batch: the cost sits in blocks with two-qubit gates; singletons unchanged",
    all(0.97 < r["joint_over_separate"] < 0.99 for r in bt["blocks"] if r["size"] > 1)
    and all(r["joint_over_separate_ci95"][0] < 1 < r["joint_over_separate_ci95"][1]
            for r in bt["blocks"] if r["size"] == 1))
chk("batch: no cross-block loss (factor 1.01 [0.82, 1.20])",
    close(bt["cross_block_factor"], 1.01, 0.005)
    and bt["cross_block_factor_ci95"][0] < 1 < bt["cross_block_factor_ci95"][1])
_b4 = _bb[(2, 3, 4, 5)]
chk("batch: the four-qubit block on qubits 2,3,4,16 retains 0.70 jointly and 0.71 alone",
    _b4["physical"] == [2, 3, 4, 16] and sorted(_l2p[q] for q in (2, 3, 4, 5)) == [2, 3, 4, 16]
    and close(_b4["retention_joint"], 0.699, 0.001) and close(_b4["retention_separate"], 0.709, 0.001)
    and close(_b4["F_proxy"], 0.946, 0.0005))
chk("batch: the placement effect is invisible to the calibration proxy (0.946 vs 0.945; 0.70 vs 0.93)",
    close(_hwsep[_sepb[(2, 3, 4, 5)]["tag"]]["F_exact"], 0.945, 0.0005)
    and _sepb[(2, 3, 4, 5)]["p_opt"] / _sepb[(2, 3, 4, 5)]["p_opt_ideal"] > 0.93
    and close(bt["F_proxy_product"], 0.819, 0.0005))
chk("prose: the batch result is stated in Sec. IV C and the SM with its confounders",
    says("a ratio of $0.906$ $[0.848,0.965]$", "dau04mahcrkc73dur170",
         "This design leaves four confounders", "it has 47 rather than 49",
         "\\label{tab:sm_batch}"))
_bn1 = bn["rung1_qaoa_p1"]
chk("r5: cross-block factor reported with its bootstrap interval and the 0.817 coincidence flagged",
    close(_bn1["cross_block_factor_ci95"][0], 0.65, 0.02) and close(_bn1["cross_block_factor_ci95"][1], 1.00, 0.01)
    and says("Its coincidence with the success proxy $\\Fest=0.817$ of this circuit is fortuitous",
             "are two independent decompositions of the same\nratio, not successive steps".replace("\n", " ")),
    str(_bn1.get("cross_block_factor_ci95")))
_ref = x51["rung1_reference"]
chk("fig4: the 24-qubit contrast group has no pair above Bonferroni (max |z| 3.2)",
    _ref["n_pairs"] == 78 and _ref["n_above_bonferroni"] == 0 and close(_ref["max_abs_z"], 3.16, 0.02))
chk("r5: nine how-it-was-done passages moved to the SM section, pointers left in the main text",
    "\\label{sm:details}" in (PAPER_DIR / SUPP_TEX).read_text(encoding="utf-8")
    and _main.count("\\ref*{sm:sm:details}") >= (6 if V5 else 7)
    and "median of $98$ m" not in _main and "$\\langle k\\rangle=2\\nzz/\\nB=4.98$" not in _main)

# ---- sixth round ----------------------------------------------------------------
chk("r6: joint cost is 11% [8,15]% in blocks with two-qubit gates; singletons multiply to 1.022",
    close(bt["joint_over_separate_two_qubit_blocks"], 0.886, 0.001)
    and np.allclose(bt["joint_over_separate_two_qubit_blocks_ci95"], [0.849, 0.925], atol=0.005)
    and close(bt["joint_over_separate_single_qubit_blocks"], 1.022, 0.001))
_sg = {tuple(r["qubits"]): r for r in bt["blocks"] if r["size"] == 1}
from calc_block_product_null import optimal_block_sets as _obs
_os1, _ = _obs(_A1, _blocks1, 24, 10)
_bit = {tuple(b): list(_os1[i])[0][0] for i, b in enumerate(_blocks1) if len(b) == 1}
chk("r6: singleton deviations follow the optimal bit (bit 1 below one; 3 of 4 with bit 0 above)",
    all(_sg[k]["retention_joint"] < 1 for k in _sg if _bit[k] == 1)
    and sum(_sg[k]["retention_joint"] > 1 for k in _sg if _bit[k] == 0) == 3
    and sum(1 for k in _sg if _bit[k] == 1) == 3)
chk("r6: the second joint run gives 100 hits, eta_all 58.3, 0.548 of ideal; within-block 0.542 vs 0.530",
    bt["joint_opt_hits"] == 100 and close(bt["joint_eta_all"], 58.25, 0.01)
    and close(bt["within_block_retention_joint"], 0.542, 0.001) and close(b1["product_of_block_ratios"], 0.530, 0.001))
chk("r6: placement effect stated as a 25% relative loss (0.93 -> 0.70); no '30%' remains",
    close(1 - 0.699 / (_sepb[(2, 3, 4, 5)]["p_opt"] / _sepb[(2, 3, 4, 5)]["p_opt_ideal"]), 0.25, 0.01)
    and "$30\\%$" not in _tex
    and ((says("separately transpiled placement of one four-qubit block differed by $25\\%$",
               "the $25\\%$ contrast bundles placement, routing, scheduling and compilation") if V5 else
          says("in the same job, a relative loss of $25\\%$", "moving one block to other qubits changes it by $25\\%$"))))
chk("prose r7: the two joint compilations are not called the same circuit",
    says("$A$ is a later transpilation of the same logical circuit: 47 two-qubit gates against 49",
         "The same compiled circuit varies between days")
    and "measured again on the same circuit" not in _tex
    and "two joint executions of the same circuit" not in _tex
    and "same transpiled circuit" not in _tex)
chk("prose r7: light-cone reading is exploratory and limited to two transpilations",
    (says("This exploratory association recurs across only two transpilations",
          "show an exploratory association between small cross-block correlations and overlapping routed light cones") if V5 else
     says("correlations appear only between logical blocks whose light cones routing makes overlap")))
chk("prose r6: S15 no longer claims separation by more than 2R; conclusion lists four points",
    "fragmentation into geographic sectors separated by more than $2R$" not in _tex
    and ((says("Blocks are separated by the absence of a shared waypoint, not by distance",
               "three checks before hardware execution") if V5 else
          says("The fragmentation is not a matter of distance",
               "The four methodological points of Sec.~\\ref{sec:intro} generalize beyond this application")))
    and "Two methodological points" not in _tex)
chk("prose r6: limitations cover placement, run-to-run variation and the synthetic waypoints",
    says("The placement dependence was measured for a single four-qubit block on two placements",
         "so it is not a crosstalk rate", "fifteen days apart",
         "The Rung-2 instance is reproducible but its selection is not"))
chk("editorial r6: Table II points to the block-wise executions; Sec. IV C has two subtitles; Tables S5, S6 cited",
    says("The three block-wise executions are reported in Fig.~\\ref{fig:blocks} and Table~\\ref{tab:executions}")
    and _main.count("\\subsubsection{") >= 2
    and "\\ref*{sm:tab:sm_batch}" in _main and "\\ref*{sm:tab:sm_blocknull}" in _main
    and (not V5 or "\\ref*{sm:tab:sm_interleaved}" in _main))


# ---- seventh round ----------------------------------------------------------------
il = J("interleaved_batch_v5.json" if V5 else "interleaved_batch.json")
_fa, _fb = il["families"]["A"], il["families"]["B"]
if V5:
    for _family in (_fa, _fb):
        _family["joint_over_alone_gated"] = _family["gated_pooled_ratio"]
        _family["joint_over_alone_gated_ci95"] = _family[
            "gated_shot_bootstrap_ci95_conditional"
        ]
        _family["joint_over_alone"] = _family["all_pooled_ratio"]
chk("r7: interleaved job, 136 pubs of 1e4 shots in seeded random order, 4 rounds",
    il["job_id"] == "daul9oqhcrkc73dvsqj0" and il["rounds"] == 4 and il["shots_per_pub"] == 10000
    and _fa["joint_shots"] == _fb["joint_shots"] == 40000)
chk("r7: joint/alone gated-block point estimates and conditional shot intervals match",
    close(_fa["joint_over_alone_gated"], 0.877, 0.001) and close(_fb["joint_over_alone_gated"], 0.671, 0.001)
    and np.allclose(_fa["joint_over_alone_gated_ci95"], [0.852, 0.904], atol=0.003)
    and np.allclose(_fb["joint_over_alone_gated_ci95"],
                    [0.650, 0.692] if V5 else [0.652, 0.693], atol=0.003)
    and close(_fa["joint_over_alone"], 0.869, 0.001) and close(_fb["joint_over_alone"], 0.691, 0.001),
    str([_fa["joint_over_alone_gated_ci95"], _fb["joint_over_alone_gated_ci95"]]))
_b01 = {f: next(x for x in il["families"][f]["per_block"] if x["qubits"] == [0, 1]) for f in "AB"}
chk("r7: in B the cost sits in block {0,1} (0.898 alone, 0.640 jointly); qubit 10 T2 = 24 us",
    close(_b01["B"]["retention_alone"], 0.898, 0.001) and close(_b01["B"]["retention_joint"], 0.640, 0.001)
    and close(J("interleaved_batch_counts.json")["coherence"]["10"]["t2_us"], 24.5, 0.5))
_f4 = il["four_qubit_block"]
chk("r7: four-qubit block 0.69 (A), 0.65 (B), 0.93 (C) in one job; proxy 0.949/0.948/0.931",
    close(_f4["A_joint"], 0.691, 0.001) and close(_f4["B_joint"], 0.652, 0.001)
    and close(_f4["C_alone"], 0.929, 0.001) and _f4["C_physical"] == [122, 123, 124, 136]
    and close(_f4["F_proxy_A"], 0.949, 0.0005) and close(_f4["F_proxy_B"], 0.948, 0.0005)
    and close(_f4["F_proxy_C"], 0.931, 0.0005)
    and close(1 - _f4["A_joint"] / _f4["C_alone"], 0.25, 0.01))
_rp = il["replication_of_dau04"]
chk("r7: compilation A reproduces across days; B fell from 0.530 to 0.340; joint P/P_ideal 0.37-0.55",
    close(_rp["now_retention_joint"], 0.519, 0.001) and close(_fb["within_block_retention_joint"], 0.340, 0.001)
    and close(_fa["joint_over_ideal"], 0.507, 0.001) and close(_fb["joint_over_ideal"], 0.370, 0.001)
    and _fa["joint_opt_hits"] == 37 and _fb["joint_opt_hits"] == 27
    and close(il["compilation_contrast"]["z"], 1.25, 0.01))
chk("prose r7: the interleaved replication is stated in Sec. IV C, Table III and the SM",
    (says("\\label{tab:executions}", "\\label{tab:sm_interleaved}", "daul9oqhcrkc73dvsqj0",
          "not a crosstalk rate", "$[0.852,0.904]$", "$[0.650,0.692]$") if V5 else
     says("\\label{tab:executions}", "\\label{tab:sm_interleaved}", "daul9oqhcrkc73dvsqj0",
          "rather than a crosstalk rate", "$0.877$ $[0.852,0.904]$", "$0.671$ $[0.652,0.693]$")))
chk("r7: n_ZZ counts nonzero couplings; 76 of 103 faithful couplings are negative",
    says("\\tilde J_{bb'}\\neq0", "(76 of the 103 Rung-1 couplings)", "^{m_e}"))
_uni = r3["uniform_in_outline"]
chk("r7: clustering factor 1.5 against uniform points in the official outline (149, 136-165, max 173)",
    close(r3["clustering_factor"], 1.53, 0.005)
    and close(_uni["strip_bound_mean"] + 1, 149.3 if V5 else 149.2, 0.05)
    and close(_uni["strip_bound_ci95"][0] + 1, 136, 1) and close(_uni["strip_bound_ci95"][1] + 1, 165, 1)
    and _uni["strip_bound_max"] + 1 == 173 and close(_uni["polygon_area_km2"], 5.06e6, 1e4)
    and says("$149$ on average", "none above $173$ in 200 draws"))
from math import comb as _comb
chk("r7: block-weight ceiling of QWOA on Rung 1 is 432/C(24,10) = 2.2e-4",
    close(432 / _comb(24, 10), 2.2e-4, 5e-6)
    and says("432/1{,}961{,}256=2.2\\times10^{-4}"))
_q20 = J("qwoa_20seeds.json")["summary"]
_ex = _q20["excluding_pilot_seeds"]
chk("r7: without the pilot seeds 11/18, Wilcoxon p=0.18, median ratio 1.02 [0.95,1.17]; Holm >= 0.15",
    _ex["weighted_rms_vs_support"]["n_seeds"] == 18
    and _ex["weighted_rms_vs_support"]["a_larger_p_opt_seeds"] == 11
    and close(_ex["weighted_rms_vs_support"]["wilcoxon_p_opt_two_sided"], 0.181, 0.001)
    and np.allclose(_ex["median_ratio_ci95"], [0.951, 1.167], atol=0.002)
    and min(_q20["by_m_holm"]["holm"].values()) >= 0.149
    and says("Holm", "absence of evidence at the stated precision"))
chk("r7: structural wording tightened (no 'structurally forced', no 'the only RZZ', no 'single-threaded')",
    "structurally forced" not in _tex and "The only RZZ-implementable" not in _tex
    and "single-threaded" not in _tex and "an earlier draft" not in _tex
    and "order-statistic analysis" not in _tex and "at most a factor $8$" not in _tex)

# ---- report -----------------------------------------------------------------
w = max(len(n) for n, _, _ in checks)
bad = 0
for n, ok, d in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {n:{w}}" + (f"   {d}" if d else ""))
    bad += not ok
print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
sys.exit(1 if bad else 0)
