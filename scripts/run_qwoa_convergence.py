#!/usr/bin/env python3
"""Convergence-first QWOA/QAOA comparison on a prespecified pilot sample.

This script addresses the main limitation of ``qwoa_expanded_p1.json``.  It
optimises the expectation value of the same exact one-sided coverage objective
for both ansatzes and uses the optimum probability only as an out-of-sample
diagnostic of the resulting angles.  It does *not* attach QUBO CX counts to the
exact threshold oracle: physical compilation is a separate experiment.

Protocol
--------
* Two task seeds are selected without looking at performance by
  ``default_rng(20260915).choice(24, 2, replace=False)``: seeds 12 and 17.
* The strict instances have minimum task degree three and are paired across
  m=1,2,3 and p=1,2,3.
* The penalty is fixed a priori at lambda=n+1=13, which makes every feasible
  bit string cheaper than every infeasible bit string.
* At p=1 a dense two-dimensional grid is combined with two-resolution
  simplicial-homology global optimisation and exact-gradient local refinement.
* At p=2,3 three independent differential-evolution searches use the same
  objective-call budget for QWOA and QAOA.  The same L-BFGS-B limits and exact
  adjoint gradients are then used for local refinement.
* QWOA is run in the k=k* sector.  This is an oracle-assisted modelling choice,
  not a free algorithmic advantage; QAOA's mass and optimum probability
  conditional on k=k* are therefore also reported.

The QWOA cost phase has fundamental gamma period 2*pi/lambda because cardinality
is constant within its sector.  QAOA retains period 2*pi because its energy also
contains the varying integer cardinality.  Both beta searches use the declared
compact time window [0, pi]; this is a protocol window, not a proof that a
weighted XY mixer has no better solution at longer evolution times.

Writes ``results/qwoa_convergence_pilot.json``.  Pure NumPy/SciPy.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize, shgo
from scipy.stats import qmc

import benchmark_qwoa_qaoa as bm
from _paths import RESULTS_DIR


SELECTION_SEED = 20260915
PILOT_SEEDS = np.random.default_rng(SELECTION_SEED).choice(
    24, size=2, replace=False
).tolist()
M_VALUES = (1, 2, 3)
P_VALUES = (1, 2, 3)
GRID_SIZE = 41
P1_SHGO_LEVELS = ((1024, 1), (2048, 2))
P1_SHGO_ADAPTIVE_LEVEL = (4096, 3)
DE_REPEATS = 3
DE_POPULATION = 32
DE_CALLS = {2: 1024, 3: 2048}
MUTATION = 0.72
CROSSOVER = 0.88
OBJECTIVE_ABS_TOL = 1e-3
P_OPT_REL_TOL = 0.02
LOCAL_MAXITER = 400
LOCAL_MAXFUN = 1200


@dataclass
class EvalCounter:
    """Count calls while retaining the best point seen."""

    fn: Callable[[np.ndarray], float]
    calls: int = 0
    best_value: float = math.inf
    best_x: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> float:
        value = float(self.fn(np.asarray(x, dtype=float)))
        if not np.isfinite(value):
            raise FloatingPointError("non-finite variational objective")
        self.calls += 1
        if value < self.best_value:
            self.best_value = value
            self.best_x = np.asarray(x, dtype=float).copy()
        return value


def parse_int_spec(spec: str) -> list[int]:
    if ":" in spec:
        values = [int(v) if v else None for v in spec.split(":")]
        if len(values) == 2:
            start, stop = values
            step = 1
        elif len(values) == 3:
            start, stop, step = values
        else:
            raise ValueError(spec)
        return list(range(0 if start is None else start, stop, 1 if step is None else step))
    return [int(v) for v in spec.split(",") if v.strip()]


def parameter_bounds(ansatz: str, p: int, penalty: int) -> np.ndarray:
    gamma_hi = 2.0 * np.pi / penalty if ansatz == "qwoa" else 2.0 * np.pi
    return np.asarray([(0.0, gamma_hi)] * p + [(0.0, np.pi)] * p, dtype=float)


def from_unit(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return bounds[:, 0] + np.asarray(x, dtype=float) * (bounds[:, 1] - bounds[:, 0])


def qaoa_state_value_grad(pc: dict, n: int, p: int, params: np.ndarray):
    """Return state, exact expectation, and adjoint gradient for QAOA."""
    params = np.asarray(params, dtype=float)
    gammas, betas = params[:p], params[p:]
    energy = pc["energy"]
    state = np.full(1 << n, 1.0 / math.sqrt(1 << n), dtype=complex)
    cost_states = []
    output_states = []
    phases = []
    for gamma, beta in zip(gammas, betas):
        phase = np.exp(-1j * gamma * energy)
        after_cost = phase * state
        # apply_x_mixer works through a reshape view and therefore mutates its
        # input; retain the pre-mixer state required by the adjoint derivative.
        cost_states.append(after_cost.copy())
        state = bm.apply_x_mixer(after_cost.copy(), beta, n)
        phases.append(phase)
        output_states.append(state)

    value = float(np.real(np.vdot(state, energy * state)))
    adjoint = energy * state
    gradient = np.empty(2 * p, dtype=float)
    indices = np.arange(1 << n)
    for layer in range(p - 1, -1, -1):
        mixer_generator_state = np.zeros_like(state)
        for bit in range(n):
            mixer_generator_state += output_states[layer][indices ^ (1 << bit)]
        gradient[p + layer] = 2.0 * np.real(
            -1j * np.vdot(adjoint, mixer_generator_state)
        )
        adjoint_cost = bm.apply_x_mixer(adjoint, -betas[layer], n)
        gradient[layer] = 2.0 * np.real(
            -1j * np.vdot(adjoint_cost, energy * cost_states[layer])
        )
        adjoint = np.conjugate(phases[layer]) * adjoint_cost
    return state, value, gradient


def qwoa_state_value_grad(sector: bm.QWOASector, p: int, params: np.ndarray):
    """Return sector state, exact expectation, and adjoint gradient for QWOA."""
    params = np.asarray(params, dtype=float)
    gammas, betas = params[:p], params[p:]
    state = np.full(sector.dim, 1.0 / math.sqrt(sector.dim), dtype=complex)
    cost_states = []
    output_states = []
    phases = []

    def mix(vector: np.ndarray, beta: float) -> np.ndarray:
        spectral = sector.evecs.conj().T @ vector
        return sector.evecs @ (np.exp(-1j * beta * sector.evals) * spectral)

    def mixer_generator(vector: np.ndarray) -> np.ndarray:
        spectral = sector.evecs.conj().T @ vector
        return sector.evecs @ (sector.evals * spectral)

    for gamma, beta in zip(gammas, betas):
        phase = np.exp(-1j * gamma * sector.Esec)
        after_cost = phase * state
        state = mix(after_cost, beta)
        phases.append(phase)
        cost_states.append(after_cost)
        output_states.append(state)

    value = float(np.real(np.vdot(state, sector.Esec * state)))
    adjoint = sector.Esec * state
    gradient = np.empty(2 * p, dtype=float)
    for layer in range(p - 1, -1, -1):
        gradient[p + layer] = 2.0 * np.real(
            -1j * np.vdot(adjoint, mixer_generator(output_states[layer]))
        )
        adjoint_cost = mix(adjoint, -betas[layer])
        gradient[layer] = 2.0 * np.real(
            -1j * np.vdot(adjoint_cost, sector.Esec * cost_states[layer])
        )
        adjoint = np.conjugate(phases[layer]) * adjoint_cost
    return state, value, gradient


def gradient_polish(
    value_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    bounds: np.ndarray,
) -> dict:
    """Bounded L-BFGS-B refinement with identical caps for both ansatzes."""
    calls = 0
    best_value = math.inf
    best_x = np.asarray(x0, dtype=float).copy()
    best_gradient = None

    def evaluate(x):
        nonlocal calls, best_value, best_x, best_gradient
        value, gradient = value_grad(np.asarray(x, dtype=float))
        value = float(value)
        gradient = np.asarray(gradient, dtype=float)
        calls += 1
        if value < best_value:
            best_value = value
            best_x = np.asarray(x, dtype=float).copy()
            best_gradient = gradient.copy()
        return value, gradient

    result = minimize(
        evaluate,
        best_x,
        jac=True,
        method="L-BFGS-B",
        bounds=[tuple(v) for v in bounds],
        options={
            "maxiter": LOCAL_MAXITER,
            "maxfun": LOCAL_MAXFUN,
            "ftol": 1e-13,
            "gtol": 1e-9,
            "maxls": 30,
        },
    )
    return {
        "parameters": best_x.tolist(),
        "final_expected_energy": best_value,
        "local_value_gradient_calls": calls,
        "gradient_infinity_norm": (
            float(np.max(np.abs(best_gradient))) if best_gradient is not None else None
        ),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
    }


def p1_landscape_search(
    objective: Callable[[np.ndarray], float],
    value_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    bounds: np.ndarray,
    include_adaptive: bool = False,
) -> dict:
    """Dense p=1 landscape plus deterministic SHGO replication searches.

    The prespecified pilot stores its two primary resolutions here and writes
    the adaptive third resolution through ``run_qwoa_p1_high_resolution.py``.
    Same-sector controls set ``include_adaptive=True`` because they are single
    self-contained experiments rather than inputs to that separate audit.
    """
    counter = EvalCounter(objective)
    gamma_grid = np.linspace(0.0, 1.0, GRID_SIZE, endpoint=False)
    beta_grid = np.linspace(0.0, 1.0, GRID_SIZE)
    landscape = []
    for ug in gamma_grid:
        for ub in beta_grid:
            unit = np.asarray([ug, ub], dtype=float)
            landscape.append((counter(from_unit(unit, bounds)), unit))
    landscape.sort(key=lambda item: item[0])

    refinements = []
    local_minima = []
    scipy_bounds = [tuple(v) for v in bounds]
    levels = list(P1_SHGO_LEVELS)
    for level_index, (samples, iterations) in enumerate(levels):
        result = shgo(
            objective,
            scipy_bounds,
            n=samples,
            iters=iterations,
            sampling_method="sobol",
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "jac": lambda x: value_grad(np.asarray(x, dtype=float))[1],
                "bounds": scipy_bounds,
                "options": {
                    "maxiter": LOCAL_MAXITER,
                    "maxfun": LOCAL_MAXFUN,
                    "ftol": 1e-13,
                    "gtol": 1e-9,
                    "maxls": 30,
                },
            },
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"p=1 SHGO failed: {result.message}")
        refinements.append({
            "repeat": f"shgo_n{samples}_iters{iterations}",
            "convergence_endpoint": True,
            "parameters": np.asarray(result.x, dtype=float).tolist(),
            "final_expected_energy": float(result.fun),
            "objective_calls": int(result.nfev),
            "number_local_minima": int(len(result.funl)),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
        })
        order = np.argsort(result.funl)[:20]
        local_minima.append({
            "samples": samples,
            "iterations": iterations,
            "top_local_minima": [
                {
                    "expected_energy": float(result.funl[i]),
                    "parameters": np.asarray(result.xl[i], dtype=float).tolist(),
                }
                for i in order
            ],
        })
        if (
            include_adaptive
            and
            level_index == len(P1_SHGO_LEVELS) - 1
            and abs(refinements[0]["final_expected_energy"] - refinements[1]["final_expected_energy"])
            > OBJECTIVE_ABS_TOL
        ):
            levels.append(P1_SHGO_ADAPTIVE_LEVEL)
    refinements.sort(key=lambda row: row["final_expected_energy"])
    return {
        "parameters": refinements[0]["parameters"],
        "expected_energy": refinements[0]["final_expected_energy"],
        "objective_calls_grid": counter.calls,
        "objective_calls_shgo": int(sum(row["objective_calls"] for row in refinements)),
        "grid_size": GRID_SIZE,
        "grid_calls": GRID_SIZE * GRID_SIZE,
        "shgo_levels": [list(level) for level in levels],
        "best_grid_expected_energy": float(landscape[0][0]),
        "local_minima": local_minima,
        "refinements": refinements,
    }


def differential_evolution_unit(
    counter: EvalCounter,
    bounds: np.ndarray,
    rng: np.random.Generator,
    call_budget: int,
) -> tuple[np.ndarray, float, list[dict]]:
    """Small transparent DE/rand/1/bin implementation with a fixed call count."""
    dim = len(bounds)
    if call_budget < DE_POPULATION or call_budget % DE_POPULATION:
        raise ValueError("DE budget must be a multiple of the population")
    sampler = qmc.LatinHypercube(d=dim, seed=rng)
    population = sampler.random(DE_POPULATION)
    values = np.asarray([counter(from_unit(x, bounds)) for x in population])
    trace = [{"calls": counter.calls, "best_expected_energy": float(values.min())}]
    generations = call_budget // DE_POPULATION - 1
    indices = np.arange(DE_POPULATION)
    for _ in range(generations):
        for i in range(DE_POPULATION):
            pool = indices[indices != i]
            a, b, c = rng.choice(pool, size=3, replace=False)
            mutant = population[a] + MUTATION * (population[b] - population[c])
            mutant[: dim // 2] %= 1.0
            mutant[dim // 2 :] = np.clip(mutant[dim // 2 :], 0.0, 1.0)
            mask = rng.random(dim) < CROSSOVER
            mask[int(rng.integers(dim))] = True
            trial = np.where(mask, mutant, population[i])
            trial_value = counter(from_unit(trial, bounds))
            if trial_value <= values[i]:
                population[i] = trial
                values[i] = trial_value
        trace.append({"calls": counter.calls, "best_expected_energy": float(values.min())})
    best = int(np.argmin(values))
    return population[best].copy(), float(values[best]), trace


def multistart_search(
    objective: Callable[[np.ndarray], float],
    value_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    bounds: np.ndarray,
    p: int,
    random_seed: int,
    warm_starts: list[np.ndarray] | None = None,
) -> dict:
    runs = []
    for repeat in range(DE_REPEATS):
        counter = EvalCounter(objective)
        rng = np.random.default_rng(random_seed + 1009 * repeat)
        x, de_value, trace = differential_evolution_unit(
            counter, bounds, rng, DE_CALLS[p]
        )
        polished = gradient_polish(value_grad, from_unit(x, bounds), bounds)
        runs.append({
            "repeat": repeat,
            "convergence_endpoint": True,
            "de_expected_energy": de_value,
            "global_objective_calls": counter.calls,
            "de_trace": trace,
            **polished,
        })
    for label, start in enumerate(warm_starts or []):
        polished = gradient_polish(value_grad, np.asarray(start, dtype=float), bounds)
        runs.append({
            "repeat": f"warm_start_{label}",
            "convergence_endpoint": False,
            "global_objective_calls": 0,
            **polished,
        })
    runs.sort(key=lambda row: row["final_expected_energy"])
    return {
        "parameters": runs[0]["parameters"],
        "expected_energy": runs[0]["final_expected_energy"],
        "global_objective_calls_per_repeat": DE_CALLS[p],
        "local_maxiter": LOCAL_MAXITER,
        "local_maxfun": LOCAL_MAXFUN,
        "repeats": runs,
    }


def qaoa_metrics(probs: np.ndarray, pc: dict, opt_states: np.ndarray, k_opt: int) -> dict:
    opt_mask = np.zeros(len(probs), dtype=bool)
    opt_mask[opt_states] = True
    sector_mask = pc["card"] == k_opt
    p_sector = float(probs[sector_mask].sum())
    p_opt = float(probs[opt_mask].sum())
    p_feas = float(probs[pc["feasible"]].sum())
    return {
        "p_opt": p_opt,
        "p_feasible": p_feas,
        "p_k_opt": p_sector,
        "p_opt_given_k_opt": p_opt / p_sector if p_sector > 0.0 else None,
        "expected_energy": float(probs @ pc["energy"]),
        "expected_cardinality": float(probs @ pc["card"]),
        "expected_violated_tasks": float(probs @ pc["n_violated"]),
    }


def qwoa_metrics(probs: np.ndarray, sector: bm.QWOASector) -> dict:
    p_feas = float(probs[sector.feas_sec].sum())
    return {
        "p_opt": p_feas,
        "p_feasible": p_feas,
        "p_k_opt": 1.0,
        "p_opt_given_k_opt": p_feas,
        "expected_energy": float(probs @ sector.Esec),
        "expected_cardinality": float(sector.k),
        "expected_violated_tasks": float(probs @ ((sector.Esec - sector.k) / bm.LAM)),
    }


def attach_run_metrics(
    search: dict,
    probability_fn: Callable[[np.ndarray], np.ndarray],
    metric_fn: Callable[[np.ndarray], dict],
) -> dict:
    """Evaluate reported metrics for every independent endpoint and the best run."""
    endpoint_rows = search.get("repeats", search.get("refinements", []))
    for endpoint in endpoint_rows:
        endpoint["metrics"] = metric_fn(
            probability_fn(np.asarray(endpoint["parameters"], dtype=float))
        )
    best_probs = probability_fn(np.asarray(search["parameters"], dtype=float))
    search["metrics"] = metric_fn(best_probs)
    convergence_rows = [
        row for row in endpoint_rows if row.get("convergence_endpoint", True)
    ]
    energies = np.asarray(
        [row["metrics"]["expected_energy"] for row in convergence_rows], dtype=float
    )
    p_opts = np.asarray(
        [row["metrics"]["p_opt"] for row in convergence_rows], dtype=float
    )
    energy_range = float(np.ptp(energies)) if len(energies) else 0.0
    best_energy = float(energies.min()) if len(energies) else math.inf
    near = energies <= best_energy + OBJECTIVE_ABS_TOL
    near_p_opts = p_opts[near]
    success_fraction = float(near.mean()) if len(near) else 0.0
    p_opt_rel_range = (
        float(np.ptp(near_p_opts) / max(float(np.mean(near_p_opts)), 1e-15))
        if len(near_p_opts)
        else math.inf
    )
    objective_replicated = bool(np.sum(near) >= 2 and success_fraction >= 2.0 / 3.0)
    search["stability"] = {
        "endpoint_expected_energy_range": energy_range,
        "convergence_endpoints": len(convergence_rows),
        "best_expected_energy": best_energy,
        "endpoints_within_objective_tolerance": int(np.sum(near)),
        "endpoint_success_fraction": success_fraction,
        "near_optimal_endpoint_p_opt_relative_range": p_opt_rel_range,
        "objective_converged": objective_replicated,
        "p_opt_stable": bool(p_opt_rel_range <= P_OPT_REL_TOL),
        "joint_criterion_passed": bool(
            objective_replicated and p_opt_rel_range <= P_OPT_REL_TOL
        ),
    }
    return search


def run_case(seed: int, m: int, p_values: list[int]) -> dict:
    instance = bm.build_instance(min_task_degree=3, rng_seed=seed)
    A = np.asarray(instance["A_q"], dtype=int)
    n = A.shape[0]
    penalty = n + 1
    bm.LAM = float(penalty)
    pc = bm.precompute(A, m)
    k_opt, opt_states_raw = bm.classical_optimum(pc)
    if k_opt is None:
        raise RuntimeError(f"seed={seed}, m={m}: strict instance is infeasible")
    k_ilp, _, ilp_status = bm.ilp_mcover(A, m)
    if ilp_status != 0 or k_ilp != k_opt:
        raise AssertionError("brute-force and MILP optimum disagree")
    opt_states = np.asarray(opt_states_raw, dtype=int)
    edges = bm.cooperation_edges(A)
    sector = bm.QWOASector(A, edges, k_opt, pc)
    qwoa_baseline = len(opt_states) / math.comb(n, k_opt)
    qaoa_baseline = len(opt_states) / (1 << n)

    row = {
        "seed": seed,
        "m": m,
        "n": n,
        "n_tasks": int(A.shape[1]),
        "range_km": float(instance["R_chosen"]),
        "minimum_task_degree": int(A.sum(axis=0).min()),
        "mean_task_degree": float(A.sum(axis=0).mean()),
        "penalty": penalty,
        "k_opt": int(k_opt),
        "n_opt": int(len(opt_states)),
        "qwoa_sector_dimension": int(sector.dim),
        "uniform_qwoa_sector_p_opt": qwoa_baseline,
        "uniform_qaoa_all_register_p_opt": qaoa_baseline,
        "depths": {},
    }
    previous_best: dict[str, np.ndarray] = {}
    for p in p_values:
        qwoa_probability = lambda params, p=p: sector.evolve(p, params)
        qwoa_objective = lambda params, p=p: float(
            qwoa_probability(params, p) @ sector.Esec
        )
        qaoa_probability = lambda params, p=p: bm.qaoa_run(pc, n, p, params)
        qaoa_objective = lambda params, p=p: float(
            qaoa_probability(params, p) @ pc["energy"]
        )
        qwoa_value_grad = lambda params, p=p: qwoa_state_value_grad(
            sector, p, params
        )[1:]
        qaoa_value_grad = lambda params, p=p: qaoa_state_value_grad(
            pc, n, p, params
        )[1:]
        searches = {}
        for ansatz, objective, value_grad, probability_fn, metric_fn in (
            (
                "qwoa",
                qwoa_objective,
                qwoa_value_grad,
                qwoa_probability,
                lambda probs: qwoa_metrics(probs, sector),
            ),
            (
                "qaoa",
                qaoa_objective,
                qaoa_value_grad,
                qaoa_probability,
                lambda probs: qaoa_metrics(probs, pc, opt_states, k_opt),
            ),
        ):
            bounds = parameter_bounds(ansatz, p, penalty)
            if p == 1:
                search = p1_landscape_search(
                    objective,
                    value_grad,
                    bounds,
                )
                search["method"] = (
                    "dense_grid_plus_two_resolution_simplicial_homology_global_optimization"
                )
            else:
                alg_offset = 0 if ansatz == "qwoa" else 0  # paired random stream
                prev = previous_best[ansatz]
                old_p = p - 1
                old_gammas, old_betas = prev[:old_p], prev[old_p:]
                warm_starts = [
                    np.concatenate([old_gammas, [0.0], old_betas, [0.0]]),
                    np.concatenate(
                        [old_gammas, old_gammas[-1:], old_betas, old_betas[-1:]]
                    ),
                ]
                search = multistart_search(
                    objective,
                    value_grad,
                    bounds,
                    p,
                    random_seed=SELECTION_SEED + 100000 * seed + 1000 * m + 10 * p + alg_offset,
                    warm_starts=warm_starts,
                )
                search["method"] = (
                    "three_fixed_budget_DE_restarts_plus_two_depth_warm_starts_"
                    "and_exact_gradient_polish"
                )
            search["parameter_bounds"] = bounds.tolist()
            searches[ansatz] = attach_run_metrics(search, probability_fn, metric_fn)
            previous_best[ansatz] = np.asarray(
                searches[ansatz]["parameters"], dtype=float
            )

        qwoa_p = searches["qwoa"]["metrics"]["p_opt"]
        qaoa_p = searches["qaoa"]["metrics"]["p_opt"]
        qaoa_cond = searches["qaoa"]["metrics"]["p_opt_given_k_opt"]
        row["depths"][str(p)] = {
            **searches,
            "p_opt_ratio_qwoa_over_qaoa_unconditional": qwoa_p / max(qaoa_p, 1e-300),
            "p_opt_ratio_qwoa_over_qaoa_conditional_k_opt": (
                qwoa_p / max(qaoa_cond, 1e-300) if qaoa_cond is not None else None
            ),
        }
    return row


def summarise(rows: list[dict]) -> dict:
    summary = {}
    for p in P_VALUES:
        selected = [row["depths"][str(p)] for row in rows if str(p) in row["depths"]]
        if not selected:
            continue
        jointly_converged = [
            item
            for item in selected
            if item["qwoa"]["stability"]["joint_criterion_passed"]
            and item["qaoa"]["stability"]["joint_criterion_passed"]
        ]

        def ratio_summary(items):
            if not items:
                return None
            raw = np.asarray(
                [item["p_opt_ratio_qwoa_over_qaoa_unconditional"] for item in items]
            )
            conditional = np.asarray(
                [item["p_opt_ratio_qwoa_over_qaoa_conditional_k_opt"] for item in items]
            )
            return {
                "median_unconditional_p_opt_ratio": float(np.median(raw)),
                "range_unconditional_p_opt_ratio": [float(raw.min()), float(raw.max())],
                "median_ratio_after_qaoa_k_opt_conditioning": float(
                    np.median(conditional)
                ),
                "range_ratio_after_qaoa_k_opt_conditioning": [
                    float(conditional.min()),
                    float(conditional.max()),
                ],
            }

        summary[str(p)] = {
            "n_cases": len(selected),
            "n_jointly_converged": len(jointly_converged),
            "qwoa_joint_convergence_passes": int(
                sum(item["qwoa"]["stability"]["joint_criterion_passed"] for item in selected)
            ),
            "qaoa_joint_convergence_passes": int(
                sum(item["qaoa"]["stability"]["joint_criterion_passed"] for item in selected)
            ),
            "jointly_converged_only": ratio_summary(jointly_converged),
            "all_best_known_not_for_inference": ratio_summary(selected),
            "qaoa_lower_expected_energy_among_jointly_converged": int(
                sum(
                    item["qaoa"]["metrics"]["expected_energy"]
                    < item["qwoa"]["metrics"]["expected_energy"]
                    for item in jointly_converged
                )
            ),
        }
    return summary


def checkpoint(
    path: Path,
    rows: list[dict],
    started: float,
    requested: dict,
    prior_runtime_s: float = 0.0,
) -> None:
    payload = {
        "meta": {
            "description": "Prespecified convergence-first QWOA/QAOA pilot",
            "scientific_status": "pilot; scale-up is conditional on convergence",
            "selection_rule": (
                "default_rng(20260915).choice(24, size=2, replace=False), "
                "evaluated before inspecting pilot performance"
            ),
            "selection_seed": SELECTION_SEED,
            "prespecified_pilot_seeds": PILOT_SEEDS,
            "requested": requested,
            "objective": "expectation of cardinality + (n+1) * number of violated tasks",
            "p_opt_role": "evaluation metric only; never the optimized objective",
            "qwoa_cardinality_assumption": "oracle-assisted k=k_opt sector",
            "beta_window": [0.0, math.pi],
            "p1_grid_size": GRID_SIZE,
            "p1_shgo_levels": [list(level) for level in P1_SHGO_LEVELS],
            "p1_shgo_adaptive_level_on_disagreement": list(P1_SHGO_ADAPTIVE_LEVEL),
            "de_repeats": DE_REPEATS,
            "de_population": DE_POPULATION,
            "de_calls": DE_CALLS,
            "local_optimizer": "L-BFGS-B with exact adjoint gradients",
            "local_maxiter": LOCAL_MAXITER,
            "local_maxfun": LOCAL_MAXFUN,
            "mutation": MUTATION,
            "crossover": CROSSOVER,
            "convergence_criteria": {
                "minimum_endpoints_within_energy_tolerance": 2,
                "minimum_endpoint_success_fraction": 2.0 / 3.0,
                "expected_energy_tolerance": OBJECTIVE_ABS_TOL,
                "near_optimal_endpoint_p_opt_relative_range_at_most": P_OPT_REL_TOL,
            },
            "resource_accounting": (
                "intentionally absent: the exact threshold oracle has not been compiled, "
                "so QUBO CX proxies would not describe these statevectors"
            ),
            "runtime_s": round(prior_runtime_s + time.time() - started, 3),
        },
        "rows": sorted(rows, key=lambda row: (row["seed"], row["m"])),
        "summary": summarise(rows),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(v) for v in PILOT_SEEDS))
    parser.add_argument("--m-values", default="1,2,3")
    parser.add_argument("--p-values", default="1,2,3")
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR / "qwoa_convergence_pilot.json"
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    seeds = parse_int_spec(args.seeds)
    m_values = parse_int_spec(args.m_values)
    p_values = parse_int_spec(args.p_values)
    for p in p_values:
        if p not in P_VALUES:
            raise ValueError(f"unsupported p={p}; choose from {P_VALUES}")
    if p_values != list(range(1, max(p_values) + 1)):
        raise ValueError("p-values must be the contiguous sequence 1,...,p_max")
    requested = {"seeds": seeds, "m_values": m_values, "p_values": p_values}
    started = time.time()
    rows = []
    prior_runtime_s = 0.0
    if args.out.exists() and not args.no_resume:
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        rows = previous.get("rows", [])
        prior_runtime_s = float(previous.get("meta", {}).get("runtime_s", 0.0))
        print(f"resume: loaded {len(rows)} complete cases from {args.out}", flush=True)
    completed = {(int(row["seed"]), int(row["m"])) for row in rows}
    for seed, m in itertools.product(seeds, m_values):
        if (seed, m) in completed:
            print(f"seed={seed}, m={m}: already complete", flush=True)
            continue
        before = time.time()
        print(f"seed={seed}, m={m}: starting p={p_values}", flush=True)
        row = run_case(seed, m, p_values)
        rows.append(row)
        checkpoint(args.out, rows, started, requested, prior_runtime_s)
        status = [
            (
                p,
                row["depths"][str(p)]["qwoa"]["stability"]["joint_criterion_passed"],
                row["depths"][str(p)]["qaoa"]["stability"]["joint_criterion_passed"],
                row["depths"][str(p)]["p_opt_ratio_qwoa_over_qaoa_unconditional"],
            )
            for p in p_values
        ]
        print(
            f"seed={seed}, m={m}: done in {time.time() - before:.1f}s; "
            f"(p,conv_QWOA,conv_QAOA,P-ratio)={status}",
            flush=True,
        )
    checkpoint(args.out, rows, started, requested, prior_runtime_s)
    print(f"-> {args.out} ({len(rows)} cases, {time.time() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
