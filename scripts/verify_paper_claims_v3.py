#!/usr/bin/env python3
"""Check the version-3 manuscript's numerical claims against the archived evidence (frozen copy).

Every check below refers to a number or scientific qualification in the active text.

  python verify_paper_claims.py     # exits non-zero if any check fails
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

from _paths import DATA_DIR, RESULTS_DIR

checks: list[tuple[str, bool, str]] = []


def chk(name, ok, detail=""):
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
src = open(PAPER / "paper_v3.tex", encoding="utf-8").read()
import re as _re
md = dict(_re.findall(r"^%%([A-Z]+):[ \t]*(.*)$", src, _re.M))
chk("all seven submission metadata fields are declared in paper_v3.tex",
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
_SOURCES = ("paper_v3.tex", "supplement_v3.tex")
_tex = re.sub(r"\s+", " ", " ".join(
    (PAPER_DIR / f).read_text(encoding="utf-8") for f in _SOURCES))


def says(*fragments):
    """Every fragment must appear verbatim in the manuscript."""
    return all(f in _tex for f in fragments)


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
    says("4.98$--$5.60$", "$\\eta_c=4.512$")
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
    says("$181,111,59,15,0.3$", "$195,128,71,16,0$", "341-post")
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
    says("\\textbf{four} new posts", "24 in all", "eleven for triple",
         "costs 25 new posts", "six targets remain short")
    and (_r2["new_posts"], _r2["posts_used_total"], _r3["new_posts"]) == (4, 24, 11)
    and _a2["new_posts"] == 25 and not _a3["feasible"] and _a3["unreachable"] == 6)
_g1, _ga = gd2["rung1"]["single_loss"], gd2["arco"]["single_loss"]
chk("prose: the re-optimisation gains and their means are the measured ones",
    says("up to five waypoints (mean $1.5$)", "at most two (mean $0.2$)")
    and max(r["gain"] for r in _g1) == 5 and close(np.mean([r["gain"] for r in _g1]), 1.5, 0.05)
    and max(r["gain"] for r in _ga) == 2 and close(np.mean([r["gain"] for r in _ga]), 0.2, 0.02))
chk("prose: the flat budget curves and the solver costs are quoted right",
    says("$21/23$", "$46/49$", "all 41 single-loss", "a few milliseconds")
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
    says("$|1-F|<10^{-12}$", "$|S|<10^{-12}$ bits",
         "double-precision round-off",
         "$1.825\\times10^{-3}$", "ideal logical unitary submitted as a 24-qubit circuit",
         "factorises into 13 block circuits of at most four qubits"))
chk("prose: the tensor-network consequence is stated with its citation",
    says("markov2008simulating", "underlying circuit graph has width $O(p(w+1))$",
         "$T^{O(1)}\\exp\\!\\big(O(p\\,\\rho R H)\\big)$"))

# ---- the ladder: three rungs, two mechanisms, one place both fail ------------
lr = J("ladder_rungs.json")
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
    says("$P_{\\mathrm{ideal}}=9.9\\times10^{-20}$", "the own-angle run took $10^5$ shots",
         "$P_{\\mathrm{ideal}}^{\\max}=2.0\\times10^{-9}$", "$5\\times10^{8}$ shots",
         "$\\mathrm{tw}\\le151$", "$\\mathrm{tw}\\ge112$", "bracketed in $[112,151]$",
         "upper bounds establish simulability, never its absence",
         "228 centroids in the worst $2R$ strip", "factor of $1.8$",
         "The 8192-shot Rung-2 predecessor transfers those angles",
         "the decomposed 79-qubit ideal distribution is optimized directly"))

# ---- the cardinality-preserving ansatz does not escape the argument ----------
qs = J("qwoa_simulability.json")
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
    and says("$(k+1)\\sum_i2^{|V_i|}=682$"),
    f"{qs['mps_simulation_cost']} state-space proxy against 2^24, "
    f"and {qs['mps_simulation_cost'] // bf['classical_simulation_cost']}x the 62 of the QAOA circuit")
chk("prose: the QWOA argument is stated with its verified numbers",
    says("saturating the bound $k+1=11$", "the proxy is $682$ against $2^{24}$",
         "the worst block cut carries $2.33$ bits",
         "those are \\emph{exactly} the $ZZ$ couplings"))

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

# ---- the static formulation's scope: persistent surveillance --------------------
ps = J("persistent_surveillance.json")
_by_reach = {}
for r in ps["rows"]:
    if r["endurance"] >= 99:
        _by_reach.setdefault(r["reach_km"], r["n_components"])
chk("mobility collapses the blocks, so the direct sum does not carry to persistent coverage",
    ps["static"]["n_components"] == 13
    and [_by_reach[k] for k in (0.0, 150.0, 300.0, 600.0)] == [13, 12, 7, 2],
    "13 blocks static; 13 / 12 / 7 / 2 as the per-slice reach grows to 600 km")
chk("the time-expanded instances nonetheless stay easy",
    all(r["lp_gap"] == 0.0 for r in ps["rows"] if r["status"] == 0)
    and max(r["seconds"] for r in ps["rows"]) < 1.0
    and max(r["treewidth_ub"] for r in ps["rows"] if r["treewidth_ub"]) <= 16)
chk("persistent surveillance of PEF-24 is infeasible under a binding endurance",
    any(r["status"] != 0 for r in ps["rows"] if r["endurance"] == 1))
chk("and the infeasibility is caused by the singly-reachable waypoints",
    ps["attribution"]["caused_by_singly_reachable"]
    and ps["attribution"]["singly_reachable_waypoints"] == 6,
    "removing those six restores feasibility at every horizon tested")
chk("prose: the limitations section states the caveat with its measured numbers",
    says("collapses to 12, 7 and finally 2 components",
         "six of the 23 waypoints are reachable by a single base",
         "Removing exactly those six waypoints restores feasibility",
         "rising only from 4 to 16"))

chk("prose: the introduction poses the question and declares the static restriction",
    says("does protecting the frontier, and eventually the Legal Amazon, need a quantum computer?",
         "the formulation most favourable to the quantum side",
         "the temporal formulation is a distinct problem"))

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
         "\\textbf{the block count is unchanged}",
         "51 qubits against 24", "combined block-state dimension $8{,}958$ against $2^{51}$",
         "89 qubits against 79"))

chk("prose: the algebraic theorem states its initial-state and layer hypotheses",
    says("Factorisation theorem", "If the initial state is",
         "every cost and driver layer is a tensor product",
         "H_C=\\sum_i H_C^{(i)}\\otimes\\mathbb 1_{\\bar V_i}")
    and "bigoplus_i H_C" not in _tex)
chk("prose: geometry is a sufficient corollary, not the theorem",
    says("Geometry is not required for the theorem",
         "E(G_{ZZ})\\subseteq E(G_B(2R))",
         "The converse is false", "sparsity alone does not imply disconnection"))
chk("prose: the width bound is a max-occupancy statement, and width is separated from cost",
    says("$W$ is the largest number of posts any strip of width $2R$ contains",
         "clustering along rivers and roads",
         "on the plane it grows as $\\exp[O(p\\sqrt{\\nB})]$",
         "not a literal statement that the contraction uses no more resources than a 48-qubit state vector",
         "$\\mathrm{tw}(G)=\\max_i\\mathrm{tw}(G[V_i])\\le\\max_i|V_i|-1$"))
chk("prose: IBM statistics use coherent baselines and an exact qualified test",
    says("$\\eta_{\\rm all}=46.02", "$\\eta_{k=10}=35.05$",
         "exact one-sided binomial probability $4.36\\times10^{-18}$",
         "cannot test drift or overdispersion"))
chk("prose: 51q distinguishes marginal post coverage from encoded feasibility",
    says("matched enhancement of $1.198$", "none of the $10^5$ full 51-bit strings",
         "marginal post-register cover rate"))
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
    says("We claim no lower bound on classical simulation in general",
         "constrains tree-decomposition contraction, not every conceivable method")
    and "costs at least $2^{112}$" not in _tex)
chk("prose: the 2-local obstruction is scoped to the bare register",
    says("no exact 2-local cost on the bare register", "on the $\\nB$ post variables alone"))
chk("prose: the Dicke-preparation claim no longer says a global rotation suffices",
    says("prepares a spin-coherent superposition across excitation sectors")
    and "prepared by a global rotation" not in _tex)
chk("prose: sparsity is not asserted to imply disconnection",
    says("sparsity alone does not imply disconnection: a path on 24 variables")
    and "sparse interaction graph is a disconnected one" not in _tex)
chk("prose: no absolute 'no device can run' claim survives",
    "no device can run" not in _tex
    and says("lies far beyond current superconducting hardware"))
chk("prose: resource proxy is not promoted to an output-probability model",
    says("not a calibrated stochastic output model", "do not convert $\\Fest=0.817$"))
chk("prose: the advantage claim is scoped to the simulations actually identified",
    says("nonlocal correlation structure needed to evade the factorisation and low-width")
    and "entanglement a quantum advantage would have to come from" not in _tex)
chk("prose: same-sector control rejects a robust p=1 mixer advantage",
    says("perform a same-sector control",
         "Energy then still splits three--three",
         "median ratio $0.975$", "only two independent task seeds",
         "complete-graph unit-weight control has lower energy in four of six"))
chk("prose: abstract and conclusion lead with the controlled negative mixer result",
    says("A same-sector $p=1$ control fixes the Dicke state",
         "a cardinality prior, not an identified mixer advantage",
         "Cardinality information is therefore a prior supplied to the ansatz"))
chk("editorial: superseded target-assisted figures are absent from the manuscript",
    "fig_advantage_vs_m.pdf" not in _tex
    and "fig_qwoa_vs_qaoa.pdf" not in _tex
    and "Legacy target-assisted scans" not in _tex
    and "Cardinality descent: sector sizes" not in _tex)
chk("editorial: the constrained mixer is audited rather than advocated",
    "ansatz we advocate" not in _tex
    and "cardinality-preserving mixer of" not in _tex
    and says("controlled comparison below tests, and does not find",
             "constrained-mixer audit is also deliberately narrow"))

chk("prose: the width inequality is stated as elementary, separate from the scaling",
    says("The content is not in the inequality, which\nis elementary, but in how $W$ grows".replace("\n", " "),
         "written with a density\n$\\rho=W/(2RH)$ the bound reads $2\\rho RH-1$".replace("\n", " "),
         "both need the density hypothesis; the\ninequality above does not".replace("\n", " ")))
chk("prose: the two width bounds are called a consistency check, not a validation",
    says("their\nproximity is a consistency check rather than a validation".replace("\n", " "),
         "at $R=400$ km the strip bound is\nthe tighter of the two".replace("\n", " ")))
chk("prose: the expander counterexample bounds the thesis, in abstract and discussion",
    says("A family of 3-regular expanders on $n$ vertices has $\\nzz=1.5n$",
         "treewidth $\\Theta(n)$", "any\nproblem class whose QUBO is an expander would escape every mechanism reported here".replace("\n", " "),
         "sparse expanders violate the width premise"))

# ---- report -----------------------------------------------------------------
w = max(len(n) for n, _, _ in checks)
bad = 0
for n, ok, d in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {n:{w}}" + (f"   {d}" if d else ""))
    bad += not ok
print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
sys.exit(1 if bad else 0)
