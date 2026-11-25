"""Read-only referee check, independent of project numerical helper functions."""
import json
import math
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def components(adjacency):
    unseen = set(range(len(adjacency)))
    result = []
    while unseen:
        todo = [min(unseen)]
        unseen.remove(todo[0])
        block = []
        while todo:
            q = todo.pop()
            block.append(q)
            neighbors = set(np.flatnonzero(adjacency[q])) & unseen
            unseen -= neighbors
            todo.extend(neighbors)
        result.append(sorted(block))
    return result


def evaluate(filename):
    data = json.loads((PROJECT / 'data' / filename).read_text())
    incidence = np.asarray(data['A'], dtype=np.int64)
    nt, nb = incidence.shape
    penalty = data['lam']
    overlap = incidence.T @ incidence
    np.fill_diagonal(overlap, 0)
    blocks = components(overlap != 0)
    k_total, degeneracy, ground, ground_degeneracy = 0, 1, 0, 1
    ideal_opt = 1.0
    ideal_sector = np.array([1.0])
    for block in blocks:
        m = len(block)
        integers = np.arange(2**m)
        bits = ((integers[:, None] >> np.arange(m)) & 1)
        local = incidence[:, block]
        local = local[np.any(local, axis=1)]
        covers = bits @ local.T
        feasible = np.all(covers >= 1, axis=1)
        weights = bits.sum(axis=1)
        k = int(weights[feasible].min())
        optimal = feasible & (weights == k)
        k_total += k
        degeneracy *= int(optimal.sum())
        energy = weights + penalty * ((1 - covers)**2).sum(axis=1)
        ground += int(energy.min())
        ground_degeneracy *= int((energy == energy.min()).sum())
        gamma, beta = data['qaoa_angles']['1']['gamma'][0], data['qaoa_angles']['1']['beta'][0]
        psi = np.exp(-1j * gamma * energy) / math.sqrt(2**m)
        for q in range(m):
            psi = math.cos(beta)*psi - 1j*math.sin(beta)*psi[integers ^ (1 << q)]
        probs = abs(psi)**2
        ideal_opt *= float(probs[optimal].sum())
        ideal_sector = np.convolve(ideal_sector, np.bincount(weights, weights=probs, minlength=m+1))
    degrees = incidence.sum(axis=1)
    slack_counts = [int(math.ceil(math.log2(int(d)))) for d in degrees]
    total = nb + sum(slack_counts)
    coefficients = np.zeros((nt, total), dtype=np.int64)
    coefficients[:, :nb] = incidence
    cursor = nb
    for t, ell in enumerate(slack_counts):
        coefficients[t, cursor:cursor+ell] = -(2 ** np.arange(ell))
        cursor += ell
    qmat = coefficients.T @ coefficients
    np.fill_diagonal(qmat, 0)
    exact_blocks = components(qmat != 0)
    exact_ground, exact_deg = 0, 1
    all_ground_faithful = True
    actual = json.loads((PROJECT / 'results' / 'exact51_plan.json').read_text()) if nb == 24 else None
    exact_ideal_energy, exact_ideal_product, exact_ideal_optimum = 0.0, 0.0, 1.0
    for block in exact_blocks:
        m = len(block)
        integers = np.arange(2**m)
        bits = ((integers[:, None] >> np.arange(m)) & 1)
        local = coefficients[:, block]
        local = local[np.any(local, axis=1)]
        residual = bits @ local.T - 1
        base_weight = bits[:, np.asarray(block) < nb].sum(axis=1)
        energies = base_weight + penalty*(residual**2).sum(axis=1)
        minimizers = energies == energies.min()
        exact_ground += int(energies.min())
        exact_deg *= int(minimizers.sum())
        all_ground_faithful &= bool(np.all(residual[minimizers] == 0))
        if actual is not None:
            g, b = actual['gamma'], actual['beta']
            psi = np.exp(-1j*g*energies) / math.sqrt(2**m)
            for q in range(m):
                psi = math.cos(b)*psi - 1j*math.sin(b)*psi[integers ^ (1 << q)]
            probability = abs(psi)**2
            marginals = probability @ bits
            exact_ideal_energy += float(probability @ energies)
            exact_ideal_optimum *= float(probability[minimizers].sum())
            # E[(sum c_i X_i - 1)^2] for independent Bernoulli marginals.
            mean_residual = local @ marginals - 1
            variance_residual = (local**2) @ (marginals * (1 - marginals))
            exact_ideal_product += float(marginals[np.asarray(block) < nb].sum() + penalty*(mean_residual**2 + variance_residual).sum())
    out = dict(instance=filename, n_base=nb, n_targets=nt, k_opt=k_total,
               n_opt=degeneracy, bare_nzz=int(np.count_nonzero(overlap)//2),
               bare_blocks=len(blocks), bare_sizes=sorted(map(len,blocks),reverse=True),
               bare_block_dimension=sum(2**len(b) for b in blocks),
               bare_ground_energy=ground, bare_ground_degeneracy=ground_degeneracy,
               bare_ideal_p_opt=ideal_opt, bare_ideal_p_k=ideal_sector[k_total],
               exact_n=total, exact_nzz=int(np.count_nonzero(qmat)//2),
               exact_negative_couplings=int((qmat < 0).sum()//2),
               raw_bare_clique_bound=sum(math.comb(int(d),2) for d in degrees),
               raw_slack_clique_bound=sum(math.comb(int(d)+ell,2) for d,ell in zip(degrees,slack_counts)),
               exact_blocks=len(exact_blocks), exact_largest=max(map(len,exact_blocks)),
               exact_block_dimension=sum(2**len(b) for b in exact_blocks),
               exact_ground_energy=exact_ground, exact_ground_degeneracy=exact_deg,
               exact_all_ground_faithful=all_ground_faithful)

    if actual is not None:
        out['executed_faithful'] = dict(energy=exact_ideal_energy, product_energy=exact_ideal_product, correlation_energy=exact_ideal_product-exact_ideal_energy, p_opt=exact_ideal_optimum)
    return out
