#!/usr/bin/env python3
"""V7: conditional counting uncertainty and separate descriptive round variation.

No nested round-plus-shot bootstrap and no round-level t interval are reported.
Raw IBM archives and the v6 analysis remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from uncertainty import conditional_mean_draws, variance_components


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_exact_circuit import exact_qubo  # noqa: E402
from calc_block_factorisation import blocks_of, load  # noqa: E402
from calc_block_product_null import block_state, optimal_block_sets  # noqa: E402


LAM = 10.0


def decode(counts: dict[str, int], width: int) -> tuple[np.ndarray, np.ndarray]:
    keys = list(counts)
    if not keys or not all(
        len(key.replace(" ", "")) == width
        and set(key.replace(" ", "")) <= {"0", "1"}
        for key in keys
    ):
        raise ValueError(f"Expected nonempty binary count keys of width {width}.")
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in counts.values()):
        raise ValueError("Count multiplicities must be positive integers.")
    bits = np.array(
        [
            [int(key.replace(" ", "")[::-1][qubit]) for qubit in range(width)]
            for key in keys
        ],
        dtype=np.int8,
    )
    multiplicity = np.array([counts[key] for key in keys], dtype=np.int64)
    return bits, multiplicity


def interval(values: list[float]) -> list[float]:
    return [
        float(np.percentile(values, 2.5, method="linear")),
        float(np.percentile(values, 97.5, method="linear")),
    ]


def point_summary(values: np.ndarray, log_scale: bool) -> dict:
    transformed = np.log(values) if log_scale else values
    mean = float(transformed.mean())
    return {
        "estimate": float(np.exp(mean)) if log_scale else mean,
        "scale": "log" if log_scale else "identity",
        "n_rounds": len(values),
    }


def qubo_energy(bits, linear, quadratic, constant):
    energy = bits @ linear + constant
    for (i, j), value in quadratic.items():
        energy = energy + value * bits[:, i] * bits[:, j]
    return energy


def product_energy(marginals, linear, quadratic, constant) -> float:
    return float(
        constant
        + marginals @ linear
        + sum(
            value * marginals[i] * marginals[j]
            for (i, j), value in quadratic.items()
        )
    )


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=1, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        raise ValueError("At least two bootstrap draws are required.")

    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    archive = json.loads(args.counts.read_text(encoding="utf-8"))
    if archive["job_id"] != plan["job_id"]:
        raise ValueError("Plan and count archive refer to different jobs.")
    submission_calibration = plan.get("submission_calibration_date")
    effective_calibration = archive.get("calibration_date")
    if not submission_calibration or not effective_calibration:
        raise ValueError("Both submission and job-time calibrations are required.")

    d, A, h, J = load()
    n_base, k_opt = d["n_b"], d["k_opt"]
    batch = json.load(open(PROJECT / "results" / "joint_separate_batch_plan.json"))
    blocks = [
        publication["logical_qubits"]
        for publication in batch["pubs"]
        if publication["tag"] != "joint"
    ]
    expected_blocks = [tuple(sorted(int(q) for q in block)) for block in blocks_of(n_base, J)]
    if sorted(map(tuple, blocks)) != sorted(expected_blocks):
        raise ValueError("Archived block order is inconsistent with the support graph.")
    optimum_sets, _ = optimal_block_sets(A, blocks, n_base, k_opt)
    gamma = d["qaoa_angles"]["1"]["gamma"][0]
    beta = d["qaoa_angles"]["1"]["beta"][0]
    ideal_block_probability = []
    for index, block in enumerate(blocks):
        m = len(block)
        probability = block_state(block, h, J, [gamma], [beta])
        local = (
            (
                np.arange(2**m)[:, None]
                >> np.arange(m)[None, ::-1]
            )
            & 1
        )
        ideal_block_probability.append(
            float(
                sum(
                    probability[state]
                    for state in range(2**m)
                    if tuple(local[state]) in optimum_sets[index]
                )
            )
        )
    ideal_block_probability = np.asarray(ideal_block_probability)
    gated = np.asarray([len(block) > 1 for block in blocks])

    frozen = json.load(open(PROJECT / "data" / "frozen_rung1.json"))
    incidence = np.asarray(frozen["A"], dtype=int)
    linear, quadratic, n_exact, n_posts, _, _, constant = exact_qubo(
        incidence, LAM
    )
    ideal_decomposition = json.load(
        open(PROJECT / "results" / "hardware_decomposition.json")
    )
    ideal_correlation = float(
        ideal_decomposition["faithful_ideal"]["correlation_energy_ideal"]
    )

    by_round: dict[int, dict[str, dict]] = {}
    metadata_by_position = {
        int(publication["position"]): publication for publication in plan["pubs"]
    }
    if len(metadata_by_position) != len(plan["pubs"]):
        raise ValueError("Duplicate publication position in plan.")
    if set(metadata_by_position) != {int(p) for p in archive["pubs"]}:
        raise ValueError("Plan and archive publication positions differ.")
    for position_string, record in archive["pubs"].items():
        position = int(position_string)
        metadata = metadata_by_position[position]
        if metadata["tag"] != record["tag"] or metadata["round"] != record["round"]:
            raise ValueError(f"Metadata mismatch at publication {position}.")
        round_records = by_round.setdefault(metadata["round"], {})
        if metadata["tag"] in round_records:
            raise ValueError("Repeated circuit tag within a round.")
        round_records[metadata["tag"]] = record["counts"]

    if sorted(by_round) != list(range(plan["rounds"])):
        raise ValueError("The count archive does not contain every round.")

    rng = np.random.default_rng(int(plan["order_seed"]) + 991)
    round_results = []
    primary_bootstrap = {
        "joint_over_slice_gated": [],
        "faithful51_correlation_energy_retained": [],
    }
    for round_index in range(plan["rounds"]):
        records = by_round[round_index]
        if set(records) != set(["joint", "faithful51", *[f"slice_{i:02d}" for i in range(13)]]):
            raise ValueError(f"Round {round_index} is incomplete.")

        joint_bits, joint_mult = decode(records["joint"], n_base)
        joint_shots = int(joint_mult.sum())
        indicator = np.array(
            [
                [
                    tuple(bitstring[block]) in optimum_sets[index]
                    for bitstring in joint_bits
                ]
                for index, block in enumerate(blocks)
            ],
            dtype=bool,
        )
        all_indicator = indicator.all(axis=0)
        joint_probability = (indicator * joint_mult).sum(axis=1) / joint_shots
        joint_optimum_probability = float(
            (all_indicator * joint_mult).sum() / joint_shots
        )
        slice_probability = np.zeros(len(blocks))
        slice_shots = np.zeros(len(blocks), dtype=int)
        for index, block in enumerate(blocks):
            bits, multiplicity = decode(records[f"slice_{index:02d}"], len(block))
            success = np.array(
                [tuple(bitstring) in optimum_sets[index] for bitstring in bits],
                dtype=bool,
            )
            slice_shots[index] = int(multiplicity.sum())
            slice_probability[index] = float(
                (success * multiplicity).sum() / slice_shots[index]
            )
        ratio = joint_probability / slice_probability
        block_metrics = {
            "joint_over_slice_all": float(np.prod(ratio)),
            "joint_over_slice_gated": float(np.prod(ratio[gated])),
            "joint_retention": float(
                np.prod(joint_probability / ideal_block_probability)
            ),
            "slice_retention": float(
                np.prod(slice_probability / ideal_block_probability)
            ),
            "cross_block_factor": float(
                joint_optimum_probability / np.prod(joint_probability)
            ),
            "joint_p_opt": joint_optimum_probability,
        }

        block_bootstrap = {key: [] for key in block_metrics}
        for _ in range(args.bootstrap):
            resampled_joint = rng.multinomial(
                joint_shots, joint_mult / joint_shots
            )
            bootstrap_joint = (
                (indicator * resampled_joint).sum(axis=1) / joint_shots
            )
            bootstrap_all = float(
                (all_indicator * resampled_joint).sum() / joint_shots
            )
            bootstrap_slice = np.array(
                [
                    rng.binomial(slice_shots[i], slice_probability[i])
                    / slice_shots[i]
                    for i in range(len(blocks))
                ]
            )
            bootstrap_ratio = bootstrap_joint / np.maximum(
                bootstrap_slice, 1e-12
            )
            values = {
                "joint_over_slice_all": float(np.prod(bootstrap_ratio)),
                "joint_over_slice_gated": float(
                    np.prod(bootstrap_ratio[gated])
                ),
                "joint_retention": float(
                    np.prod(
                        bootstrap_joint
                        / ideal_block_probability
                    )
                ),
                "slice_retention": float(
                    np.prod(
                        bootstrap_slice
                        / ideal_block_probability
                    )
                ),
                "cross_block_factor": float(
                    bootstrap_all
                    / max(float(np.prod(bootstrap_joint)), 1e-15)
                ),
                "joint_p_opt": bootstrap_all,
            }
            for key, value in values.items():
                block_bootstrap[key].append(value)

        faithful_bits, faithful_mult = decode(records["faithful51"], n_exact)
        faithful_shots = int(faithful_mult.sum())
        faithful_float = faithful_bits.astype(float)
        energy_vector = qubo_energy(
            faithful_float, linear, quadratic, constant
        )
        mean_energy = float(energy_vector @ faithful_mult / faithful_shots)
        marginals = (
            faithful_float.T @ faithful_mult / faithful_shots
        )
        product_null = product_energy(
            marginals, linear, quadratic, constant
        )
        posts = faithful_bits[:, :n_posts]
        covered = (incidence @ posts.T >= 1).all(axis=0)
        p_feasible = float((covered * faithful_mult).sum() / faithful_shots)
        mean_posts = float(
            (posts.sum(axis=1) * faithful_mult).sum() / faithful_shots
        )
        faithful_metrics = {
            "mean_energy": mean_energy,
            "product_null_energy": product_null,
            "correlation_energy": product_null - mean_energy,
            "correlation_energy_retained": (
                product_null - mean_energy
            )
            / ideal_correlation,
            "p_feasible": p_feasible,
            "mean_posts": mean_posts,
        }
        faithful_bootstrap = {key: [] for key in faithful_metrics}
        for _ in range(args.bootstrap):
            resampled = rng.multinomial(
                faithful_shots, faithful_mult / faithful_shots
            )
            bootstrap_energy = float(
                energy_vector @ resampled / faithful_shots
            )
            bootstrap_marginals = (
                faithful_float.T @ resampled / faithful_shots
            )
            bootstrap_product = product_energy(
                bootstrap_marginals, linear, quadratic, constant
            )
            values = {
                "mean_energy": bootstrap_energy,
                "product_null_energy": bootstrap_product,
                "correlation_energy": bootstrap_product - bootstrap_energy,
                "correlation_energy_retained": (
                    bootstrap_product - bootstrap_energy
                )
                / ideal_correlation,
                "p_feasible": float(
                    (covered * resampled).sum() / faithful_shots
                ),
                "mean_posts": float(
                    (posts.sum(axis=1) * resampled).sum()
                    / faithful_shots
                ),
            }
            for key, value in values.items():
                faithful_bootstrap[key].append(value)

        primary_bootstrap["joint_over_slice_gated"].append(
            np.asarray(
                block_bootstrap["joint_over_slice_gated"], dtype=float
            )
        )
        primary_bootstrap[
            "faithful51_correlation_energy_retained"
        ].append(
            np.asarray(
                faithful_bootstrap["correlation_energy_retained"],
                dtype=float,
            )
        )
        round_results.append(
            {
                "round": round_index,
                "block": {
                    **block_metrics,
                    "shot_bootstrap_ci95_conditional": {
                        key: interval(values)
                        for key, values in block_bootstrap.items()
                    },
                    "shot_bootstrap_sd_conditional": {
                        key: float(np.std(values, ddof=1))
                        for key, values in block_bootstrap.items()
                    },
                },
                "faithful51": {
                    **faithful_metrics,
                    "shot_bootstrap_ci95_conditional": {
                        key: interval(values)
                        for key, values in faithful_bootstrap.items()
                    },
                    "shot_bootstrap_sd_conditional": {
                        key: float(np.std(values, ddof=1))
                        for key, values in faithful_bootstrap.items()
                    },
                },
            }
        )

    gated_rounds = np.asarray(
        [
            result["block"]["joint_over_slice_gated"]
            for result in round_results
        ]
    )
    faithful_rounds = np.asarray(
        [
            result["faithful51"]["correlation_energy_retained"]
            for result in round_results
        ]
    )
    gated_bootstrap = np.vstack(
        primary_bootstrap["joint_over_slice_gated"]
    )
    faithful_bootstrap = np.vstack(
        primary_bootstrap[
            "faithful51_correlation_energy_retained"
        ]
    )
    gated_epoch_conditional = conditional_mean_draws(gated_bootstrap, log_scale=True)
    faithful_epoch_conditional = conditional_mean_draws(faithful_bootstrap)
    gated_components = variance_components(
        np.log(gated_rounds), np.log(gated_bootstrap).var(axis=1, ddof=1)
    )
    faithful_components = variance_components(
        faithful_rounds, faithful_bootstrap.var(axis=1, ddof=1)
    )
    output = {
        "schema_version": 2,
        "analysis_version": "working_v7",
        "campaign": plan["campaign"],
        "epoch": plan["epoch"],
        "job_id": plan["job_id"],
        "calibration_date": effective_calibration,
        "submission_calibration_date": submission_calibration,
        "calibration_changed_after_submission": (
            effective_calibration != submission_calibration
        ),
        "levels_available": [
            "conditional_shot_noise",
            "within_job_round_variation",
        ],
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": int(plan["order_seed"]) + 991,
        "twirling_strata_archived": False,
        "uncertainty_scope": (
            "Counting-model percentile intervals conditional on four fixed observed "
            "rounds and aggregate pub counts; no resampling of rounds. Unarchived "
            "twirling strata, drift and between-job variation are not quantified "
            "by these intervals. Variance subtraction is descriptive only."
        ),
        "ideal_correlation_energy": ideal_correlation,
        "rounds": round_results,
        "epoch_summary": {
            "joint_over_slice_gated": point_summary(
                gated_rounds, log_scale=True
            ),
            "faithful51_correlation_energy_retained": point_summary(
                faithful_rounds, log_scale=False
            ),
            "conditional_shot_epoch_ci95": {
                "joint_over_slice_gated": interval(
                    gated_epoch_conditional.tolist()
                ),
                "faithful51_correlation_energy_retained": interval(
                    faithful_epoch_conditional.tolist()
                ),
            },
            "variance_description": {
                "joint_over_slice_gated_log_scale": gated_components,
                "faithful51_correlation_retention_identity_scale": faithful_components,
            },
            "round_values": {
                "joint_over_slice_gated": gated_rounds.tolist(),
                "faithful51_correlation_energy_retained": faithful_rounds.tolist(),
            },
        },
    }
    write_new(args.out, output)
    print(
        "joint/slice gated:",
        output["epoch_summary"]["joint_over_slice_gated"],
    )
    print(
        "faithful51 correlation retention:",
        output["epoch_summary"][
            "faithful51_correlation_energy_retained"
        ],
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
