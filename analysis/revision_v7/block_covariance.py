"""Exploratory block covariance profiles with joint, fixed-pub counting bootstrap."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from diagnostic_controls import faithful_reference

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT_SEED = 2026102102
BOOTSTRAP = 10000
CONTEXTS = ("defaults", "matched", "E01", "E02", "E03")
STEMS = ("e01_20261008", "e02_20261014", "e03_20261014")


def reference():
    mean, second, linear, quadratic, constant, all_blocks = faithful_reference()
    blocks = sorted((b for b in all_blocks if len(b) > 1), key=lambda b: (len(b), b))
    pairs = []
    delta = []
    for block in blocks:
        ii, jj = np.nonzero(quadratic[np.ix_(block, block)])
        i, j = np.asarray(block)[ii], np.asarray(block)[jj]
        q = quadratic[i, j]
        pairs.append((i, j, q))
        delta.append(float(np.sum(q * (mean[i]*mean[j] - second[i, j]))))
    delta = np.asarray(delta)
    if np.any(abs(delta) < 1e-10):
        raise ValueError("A block reference denominator is too small for normalization.")
    if abs(delta.sum()) < 1e-10:
        raise ValueError("The global reference denominator is too small.")
    return dict(blocks=blocks, singleton_blocks=[b for b in all_blocks if len(b) == 1],
                pairs=pairs, delta=delta, weights=delta/delta.sum(),
                n=len(mean), linear=linear, constant=constant,
                mean=mean, second=second,
                energy_random=float(constant + .5*linear.sum() + .25*quadratic.sum()))


def features(bits, pairs):
    """First bit moments and one quadratic-energy statistic per block suffice."""
    bits = np.asarray(bits, dtype=float)
    quadratic = np.column_stack([
        (bits[:, i]*bits[:, j]) @ q for i, j, q in pairs
    ])
    return np.column_stack((bits, quadratic))


def statistics(moment, ref):
    moment = np.atleast_2d(moment)
    mu = moment[:, :ref["n"]]
    product_pairs = np.column_stack([
        (mu[:, i]*mu[:, j]) @ q for i, j, q in ref["pairs"]
    ])
    delta = product_pairs - moment[:, ref["n"]:]
    ratio = delta/ref["delta"]
    total = delta.sum(axis=1)
    global_ratio = total/ref["delta"].sum()
    energy = ref["constant"] + mu @ ref["linear"] + moment[:, ref["n"]:].sum(axis=1)
    product_energy = energy + total
    output = dict(delta=delta, retention=ratio, correlation_energy=total,
                  global_retention=global_ratio, energy=energy, product_energy=product_energy,
                  reduction=1-energy/ref["energy_random"],
                  product_reduction=1-product_energy/ref["energy_random"],
                  contrast=ratio-global_ratio[:, None])
    return add_share(output)


def add_share(values):
    if np.any(abs(values["reduction"]) < 1e-10):
        raise ValueError("Marginal share is undefined near zero energy reduction.")
    values["marginal_share"] = values["product_reduction"]/values["reduction"]
    return values


def average_rounds(values):
    # Average nonlinear statistics of each fixed round, not moments of pooled counts.
    output = {key: np.mean([v[key] for v in values], axis=0)
              for key in values[0] if key != "marginal_share"}
    return add_share(output)


def bootstrap_moments(feature, multiplicity, rng, replicates, batch_size=32):
    shots = int(multiplicity.sum())
    probability = multiplicity/shots
    out = np.empty((replicates, feature.shape[1]))
    uniform = bool(np.all(multiplicity == multiplicity[0]))
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates-start)
        if uniform:
            # Category occupancies of uniform index draws are exactly multinomial.
            weights = np.stack([np.bincount(rng.integers(len(feature), size=shots),
                                           minlength=len(feature)) for _ in range(size)])
        else:
            weights = rng.multinomial(shots, probability, size=size)
        out[start:start+size] = (weights @ feature)/shots
    return out


def decode(counts, n):
    keys = list(counts)
    if not keys or any(len(k) != n or set(k)-{"0", "1"} for k in keys):
        raise ValueError("Invalid binary count keys.")
    multiplicity = np.asarray([counts[k] for k in keys])
    if np.any(multiplicity <= 0) or not np.issubdtype(multiplicity.dtype, np.integer):
        raise ValueError("Counts must be positive integers.")
    integers = np.asarray([int(k, 2) for k in keys], dtype=np.uint64)
    bits = ((integers[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
    return bits, multiplicity


def sources():
    output = []
    for context in CONTEXTS[:2]:
        counts_path = PROJECT/"results"/(
            "exact51_counts.json" if context == "defaults" else "exact51_matched_counts.json")
        plan_path = PROJECT/"results"/(
            "exact51_plan.json" if context == "defaults" else "exact51_matched_plan.json")
        archive, plan = [json.loads(p.read_text()) for p in (counts_path, plan_path)]
        if archive["job_id"] != plan["job_id"]:
            raise ValueError("Historical plan/count mismatch.")
        output.append(dict(context=context, round=0, counts_path=counts_path,
                           plan_path=plan_path, job_id=archive["job_id"],
                           counts=archive["circuits"]["faithful_exact51_p1"]["counts"]))
    archive_dir = PROJECT/"analysis/ibm_variance_campaign/results"
    for context, stem in zip(CONTEXTS[2:], STEMS):
        counts_path, plan_path = [archive_dir/f"{stem}_{suffix}.json" for suffix in ("counts", "plan")]
        archive, plan = [json.loads(p.read_text()) for p in (counts_path, plan_path)]
        if archive["job_id"] != plan["job_id"]:
            raise ValueError("Campaign plan/count mismatch.")
        pubs = sorted((p for p in plan["pubs"] if p["tag"] == "faithful51"), key=lambda p: p["round"])
        if [p["round"] for p in pubs] != list(range(4)):
            raise ValueError("Expected four fixed faithful-circuit rounds.")
        for p in pubs:
            record = archive["pubs"][str(p["position"])]
            if record["tag"] != p["tag"] or record["round"] != p["round"]:
                raise ValueError("Campaign pub metadata mismatch.")
            output.append(dict(context=context, round=p["round"], counts_path=counts_path,
                               plan_path=plan_path, job_id=archive["job_id"], counts=record["counts"]))
    return output


def summarize(point, draws):
    return {key: dict(estimate=np.asarray(value[0]).tolist(),
                      ci95=np.percentile(draws[key], [2.5, 97.5], axis=0).tolist(),
                      conditional_se=np.std(draws[key], axis=0, ddof=1).tolist())
            for key, value in point.items()}


def simultaneous(point, draws):
    center = np.concatenate([v["contrast"][0] for v in point])
    samples = np.concatenate([v["contrast"] for v in draws], axis=1)
    se = samples.std(axis=0, ddof=1)
    active = se > 1e-14
    if np.any(abs(samples[:, ~active]-center[~active]) > 1e-10):
        raise ValueError("Degenerate contrast variance with nonconstant draws.")
    maximum = np.max(abs((samples[:, active]-center[active])/se[active]), axis=1) if active.any() else np.zeros(len(samples))
    q = float(np.percentile(maximum, 95))
    bands = np.column_stack((center-q*se, center+q*se))
    resolved = (bands[:, 0] > 0) | (bands[:, 1] < 0)
    half = len(maximum)//2
    half_q = [float(np.percentile(x, 95)) for x in (maximum[:half], maximum[half:])]
    half_flags = [(abs(center) > x*se).tolist() for x in half_q]
    return dict(family_size=len(center), quantile95=q, half_sample_quantiles95=half_q,
                half_sample_classification_stable=half_flags[0] == half_flags[1],
                zero_variance_contrasts=int((~active).sum()),
                bands95=bands.reshape(len(point), -1, 2).tolist(),
                resolved=resolved.reshape(len(point), -1).tolist())


def analyze(replicates=BOOTSTRAP, progress=False):
    if replicates < 100:
        raise ValueError("At least 100 bootstrap draws required.")
    ref = reference()
    records = sources()
    streams = np.random.SeedSequence(ROOT_SEED).spawn(len(records))
    pubs = []
    context_points, context_draws = [], []
    hashes = {}
    for context in CONTEXTS:
        points, draws = [], []
        for index, row in enumerate(records):
            if row["context"] != context:
                continue
            bits, multiplicity = decode(row["counts"], ref["n"])
            feature = features(bits, ref["pairs"])
            del bits
            point = statistics((multiplicity @ feature)/multiplicity.sum(), ref)
            boot = bootstrap_moments(feature, multiplicity, np.random.default_rng(streams[index]), replicates)
            sample = statistics(boot, ref)
            points.append(point)
            draws.append(sample)
            for path in (row["counts_path"], row["plan_path"]):
                hashes[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            pubs.append(dict(context=context, round=row["round"], job_id=row["job_id"],
                             source_counts=str(row["counts_path"].relative_to(PROJECT)),
                             shots=int(multiplicity.sum()), distinct_strings=len(multiplicity),
                             stream_index=index, statistics=summarize(point, sample)))
            if progress:
                print(f"PASS: {context} round {row['round']}, {replicates} joint draws", flush=True)
        context_points.append(average_rounds(points))
        context_draws.append(average_rounds(draws))
    bands = simultaneous(context_points, context_draws)
    summaries = []
    for index, (context, point, draws) in enumerate(zip(CONTEXTS, context_points, context_draws)):
        if not np.allclose(point["retention"] @ ref["weights"], point["global_retention"], atol=1e-11):
            raise AssertionError("Point reconstruction failed.")
        if not np.allclose(draws["retention"] @ ref["weights"], draws["global_retention"], atol=1e-11):
            raise AssertionError("Bootstrap reconstruction failed.")
        summaries.append(dict(context=context, statistics=summarize(point, draws),
                              contrast_simultaneous_ci95=bands["bands95"][index],
                              contrast_resolved=bands["resolved"][index]))
    return dict(
        scope="Exploratory, conditional counting-model block profiles; not causal or population inference.",
        protocol=dict(replicates=replicates, root_seed=ROOT_SEED, pub_streams="SeedSequence.spawn in archived source order",
                      resampling="Joint full-bitstring multinomial; independent pub draws; four observed rounds fixed",
                      intervals="Pointwise percentile 95%; simultaneous 95% standardized-max bands for 30 contrasts",
                      contrast="C_alpha - C_global", family="Six blocks in five observed contexts",
                      marginal_share="Ratio of round-mean product reduction to round-mean energy reduction"),
        reference=dict(n_qubits=ref["n"], blocks=[list(map(int, b)) for b in ref["blocks"]],
                       block_sizes=[len(b) for b in ref["blocks"]],
                       singleton_blocks=[list(map(int, b)) for b in ref["singleton_blocks"]],
                       delta=ref["delta"].tolist(), weights=ref["weights"].tolist(),
                       global_delta=float(ref["delta"].sum()), energy_random=ref["energy_random"]),
        input_sha256=hashes, pubs=pubs, contexts=summaries, simultaneous=bands)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=HERE/"generated")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args()
    started = time.monotonic()
    result = analyze(args.bootstrap, progress=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/"block_covariance.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    performance = dict(elapsed_seconds=round(time.monotonic()-started, 2),
                       numpy_version=np.__version__, bootstrap=args.bootstrap,
                       note="Timing metadata are not compared numerically by reproduction.")
    (args.out_dir/"block_covariance_performance.json").write_text(json.dumps(performance, indent=2)+"\n")
    print(json.dumps(performance), flush=True)


if __name__ == "__main__":
    main()
