"""Offline regression and independent checks for the v7 corrections."""
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import unittest

import numpy as np
from scipy.stats import spearmanr

from uncertainty import conditional_mean_draws, variance_components
from combine_epochs import combine, ENDPOINTS
from ensemble_analysis import analyze, spearman_rows, paired_bootstrap
from independent_encoding import evaluate
from analyze_epoch import decode
from diagnostic_controls import moments, readout_moments, dependence_control, analyze as control_analysis
import block_covariance as bc

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ARCHIVE = HERE.parent / "ibm_variance_campaign/results"
STEMS = ("e01_20261008", "e02_20261014", "e03_20261014")


def read(path):
    return json.loads(path.read_text())


class CountingTests(unittest.TestCase):
    def test_fixed_round_mean_does_not_resample_round_effects(self):
        rng = np.random.default_rng(1701)
        draws = np.arange(4)[:, None] * 100 + rng.normal(size=(4, 100000))
        mean_draws = conditional_mean_draws(draws)
        self.assertAlmostEqual(mean_draws.var(), 1/4, delta=.004)
        self.assertAlmostEqual(mean_draws.mean(), 150, delta=.006)
        np.testing.assert_allclose(
            np.log(conditional_mean_draws(np.exp(draws / 100), True)),
            draws.mean(axis=0) / 100,
        )

    def test_variance_subtraction_not_addition(self):
        result = variance_components(np.array([0., 1., 2., 3.]), np.ones(4))
        self.assertAlmostEqual(result["observed_round_variance"], 5/3)
        self.assertAlmostEqual(result["excess_round_variance_method_of_moments"], 2/3)
        self.assertAlmostEqual(result["working_variance_of_mean_method_of_moments"], 5/12)
        low = variance_components(np.ones(4), np.full(4, 2.))
        self.assertEqual(low["excess_round_variance_method_of_moments"], 0.)
        self.assertEqual(low["working_variance_of_mean_method_of_moments"], .5)

    def test_invalid_inputs_are_rejected(self):
        for bad in [np.ones(4), np.array([[np.nan, 1], [2, 3]])]:
            with self.assertRaises(ValueError):
                conditional_mean_draws(bad)
        with self.assertRaises(ValueError):
            conditional_mean_draws(np.zeros((4, 4)), True)
        with self.assertRaises(ValueError):
            variance_components(np.ones(4), -np.ones(4))
        for counts in [{}, {"101": -1}, {"201": 2}, {"101": 1.5}]:
            with self.assertRaises(ValueError):
                decode(counts, 3)

    def test_bit_order(self):
        bits, multiplicity = decode({"10 01": 2, "0110": 3}, 4)
        np.testing.assert_array_equal(bits, [[1, 0, 0, 1], [0, 1, 1, 0]])
        np.testing.assert_array_equal(multiplicity, [2, 3])

    def test_shot_variance_is_already_in_round_dispersion(self):
        rng = np.random.default_rng(912)
        observed = rng.normal(0, 2, (40000, 4)) + rng.normal(0, 1, (40000, 4))
        self.assertAlmostEqual(observed.var(axis=1, ddof=1).mean(), 5., delta=.06)
        self.assertAlmostEqual(observed.mean(axis=1).var(), 5/4, delta=.03)
        # Validates the working model, not independence of the archived jobs.


class EncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pef = evaluate("frozen_rung1.json")
        cls.arco = evaluate("frozen_arco.json")

    def test_distinct_and_raw_interaction_counts(self):
        for row, expected in [(self.pef, (40, 17, 126, 103)),
                              (self.arco, (10, 9, 30, 29))]:
            self.assertEqual(tuple(row[k] for k in (
                "raw_bare_clique_bound", "bare_nzz",
                "raw_slack_clique_bound", "exact_nzz")), expected)

    def test_ground_states_and_dimensions(self):
        for row, optimum, degeneracy, bare_dim, exact_dim in [
            (self.pef, 10, 288, 62, 8958), (self.arco, 31, 4, 160, 220)
        ]:
            self.assertEqual(row["k_opt"], optimum)
            self.assertEqual(row["n_opt"], degeneracy)
            self.assertEqual(row["exact_ground_energy"], optimum)
            self.assertEqual(row["exact_ground_degeneracy"], degeneracy)
            self.assertTrue(row["exact_all_ground_faithful"])
            self.assertEqual(row["bare_block_dimension"], bare_dim)
            self.assertEqual(row["exact_block_dimension"], exact_dim)
        self.assertEqual(self.pef["bare_ground_energy"], 19)
        self.assertEqual(self.pef["bare_ground_degeneracy"], 144)

    def test_exact_references_at_executed_angles(self):
        self.assertAlmostEqual(self.pef["bare_ideal_p_opt"], .0018253777517512502, places=14)
        reference = self.pef["executed_faithful"]
        self.assertAlmostEqual(reference["correlation_energy"], 29.800203679101685, places=10)
        self.assertAlmostEqual(reference["energy"], 209.18341225192935, places=10)
        self.assertAlmostEqual(reference["p_opt"]/5.473905486707746e-20, 1., places=9)
        archived = read(PROJECT / "results/hardware_decomposition.json")
        self.assertAlmostEqual(reference["correlation_energy"],
                               archived["faithful_ideal"]["correlation_energy_ideal"], places=10)

    def test_three_job_points_directly_from_count_archives(self):
        # No project decoder, QUBO builder or optimum-set helper is used below.
        incidence = np.asarray(read(PROJECT/"data/frozen_rung1.json")["A"], dtype=int)
        blocks = [p["logical_qubits"] for p in
                  read(PROJECT/"results/joint_separate_batch_plan.json")["pubs"]
                  if p["tag"] != "joint"]
        optimal = []
        for block in blocks:
            configs = np.asarray(list(itertools.product([0, 1], repeat=len(block))))
            local = incidence[:, block]
            local = local[np.any(local, axis=1)]
            good = (configs @ local.T >= 1).all(axis=1)
            weight = configs.sum(axis=1)
            optimal.append({tuple(x) for x in configs[good & (weight == weight[good].min())]})
        coeff = np.zeros((len(incidence), 51), dtype=int)
        coeff[:, :24] = incidence
        cursor = 24
        for t, degree in enumerate(incidence.sum(axis=1)):
            width = math.ceil(math.log2(int(degree)))
            coeff[t, cursor:cursor+width] = -(2**np.arange(width))
            cursor += width
        for number, stem in enumerate(STEMS, 1):
            archive = read(ARCHIVE/f"{stem}_counts.json")
            analysis = read(HERE/f"results/e0{number}_analysis.json")
            for round_result in analysis["rounds"]:
                records = {p["tag"]: p["counts"] for p in archive["pubs"].values()
                           if p["round"] == round_result["round"]}
                def unpack(tag):
                    counts = records[tag]
                    bits = np.array([[int(b) for b in k.replace(" ", "")[::-1]] for k in counts])
                    return bits, np.array(list(counts.values()), dtype=float)
                joint, counts = unpack("joint")
                product = 1.
                for i, block in enumerate(blocks):
                    if len(block) == 1:
                        continue
                    success = [tuple(x[block]) in optimal[i] for x in joint]
                    pj = counts @ success / counts.sum()
                    sliced, scounts = unpack(f"slice_{i:02d}")
                    ps = scounts @ [tuple(x) in optimal[i] for x in sliced] / scounts.sum()
                    product *= pj / ps
                self.assertAlmostEqual(product, round_result["block"]["joint_over_slice_gated"], places=13)
                bits, counts = unpack("faithful51")
                probabilities = counts/counts.sum()
                residual = bits @ coeff.T - 1
                energy = bits[:, :24].sum(axis=1) + 10*(residual**2).sum(axis=1)
                mu = probabilities @ bits
                null = mu[:24].sum() + 10*(((coeff @ mu - 1)**2) +
                            (coeff**2) @ (mu*(1-mu))).sum()
                retained = (null - probabilities @ energy)/self.pef["executed_faithful"]["correlation_energy"]
                self.assertAlmostEqual(retained, round_result["faithful51"]["correlation_energy_retained"], places=11)


class StructuralTests(unittest.TestCase):
    def test_doubled_path_sum_matches_statevector_marginal(self):
        n, p = 2, 2
        states = np.array(list(itertools.product([0, 1], repeat=n)))
        energy = states @ np.array([.3, -.5]) + .7*states[:, 0]*states[:, 1]
        gamma, beta = (.2, -.7), (.4, .15)
        one_bit = [np.array([[math.cos(b), -1j*math.sin(b)],
                             [-1j*math.sin(b), math.cos(b)]]) for b in beta]
        mixer = [np.kron(m, m) for m in one_bit]
        psi = np.full(4, .5, dtype=complex)
        for layer in range(p):
            psi = mixer[layer] @ (np.exp(-1j*gamma[layer]*energy)*psi)
        marginal = sum(abs(psi[s])**2 for s in range(4) if states[s, 0] == 1)
        paths = {}
        for end in range(4):
            paths[end] = [.5 * np.exp(-1j*(gamma[0]*energy[a]+gamma[1]*energy[b]))
                          * mixer[0][b,a]*mixer[1][end,b]
                          for a,b in itertools.product(range(4), repeat=2)]
        doubled = sum(x*y.conjugate() for end, terms in paths.items()
                      if states[end,0] == 1 for x,y in itertools.product(terms, repeat=2))
        self.assertAlmostEqual(doubled.imag, 0., places=13)
        self.assertAlmostEqual(doubled.real, marginal, places=13)

    def test_dicke_allocation_and_schmidt_formula(self):
        sizes, k = (2, 3, 2), 3
        total = math.comb(sum(sizes), k)
        weights = [math.prod(math.comb(n, j) for n, j in zip(sizes, js))/total
                   for js in itertools.product(*(range(n+1) for n in sizes)) if sum(js) == k]
        self.assertAlmostEqual(sum(weights), 1.)
        # Schmidt values directly from the bipartite coefficient matrix.
        a, n = 2, sum(sizes)
        matrix = np.zeros((2**a, 2**(n-a)))
        for i, j in itertools.product(range(2**a), range(2**(n-a))):
            if i.bit_count()+j.bit_count() == k:
                matrix[i, j] = 1/math.sqrt(total)
        spectrum = np.linalg.svd(matrix, compute_uv=False)**2
        analytic = [math.comb(a,j)*math.comb(n-a,k-j)/total for j in range(a+1)]
        np.testing.assert_allclose(sorted(spectrum[spectrum>1e-12]), sorted(analytic))

    def test_seed_draws_are_without_replacement(self):
        pilot = np.random.default_rng(20260915).choice(24, 2, replace=False)
        extension = np.random.default_rng(20260915).choice(24, 20, replace=False)
        self.assertEqual(set(pilot), {12, 17})
        self.assertEqual(len(set(extension)), 20)
        self.assertTrue(set(pilot) <= set(extension))


class EnsembleTests(unittest.TestCase):
    def test_vectorized_spearman_matches_scipy(self):
        x = np.array([[1, 1, 2, 3], [5, 2, 2, 1]], float)
        y = np.array([[8, 6, 2, 2], [0, 1, 2, 3]], float)
        np.testing.assert_allclose(spearman_rows(x,y), [spearmanr(a,b).statistic for a,b in zip(x,y)])

    def test_bootstrap_preserves_pairing(self):
        x = np.arange(25.)
        draws, rejected = paired_bootstrap(x, [x, -x], np.random.default_rng(4), 100)
        np.testing.assert_allclose(draws, np.tile([1, -1], (100,1)))
        self.assertEqual(rejected, 0)

    def test_fixed_cell_ensemble_reproduces(self):
        recomputed, _ = analyze()
        self.assertEqual(recomputed, read(HERE/"generated/ensemble_analysis.json"))
        self.assertEqual(len(recomputed["cells"]), 24)
        self.assertTrue(all(c["n"] == 25 for c in recomputed["cells"]))
        for size in recomputed["by_register_size"]:
            self.assertGreater(size["dimension"]["ci95"][0], 0)
            self.assertLess(size["phi"]["ci95"][0], 0)
            self.assertGreater(size["phi"]["ci95"][1], 0)


class CampaignTests(unittest.TestCase):
    def test_central_values_unchanged_and_no_nested_intervals(self):
        epochs = [read(HERE/f"results/e0{i}_analysis.json") for i in range(1,4)]
        for epoch, stem in zip(epochs, STEMS):
            old = read(ARCHIVE/f"{stem}_analysis_v2.json")
            for key in ENDPOINTS:
                self.assertAlmostEqual(epoch["epoch_summary"][key]["estimate"],
                                       old["epoch_summary"][key]["estimate"], places=13)
            self.assertEqual(epoch["bootstrap_replicates"], 4000)
            self.assertNotIn("hierarchical", json.dumps(epoch))
        self.assertEqual(combine(epochs), read(HERE/"results/between_jobs.json"))
        with self.assertRaises(ValueError):
            combine([epochs[0]]*3)


class DiagnosticRevisionTests(unittest.TestCase):
    def test_output_channel_against_full_stochastic_distribution(self):
        bits = np.array(list(itertools.product((0, 1), repeat=3)), dtype=float)
        probability = np.arange(1., 9.)/36
        p01, p10 = np.array([.03, .15, .1]), np.array([.2, .01, .4])
        transformed = np.zeros(8)
        for i, source in enumerate(bits):
            one_probability = np.where(source == 0, p01, 1-p10)
            for j, target in enumerate(bits):
                transition = np.prod(np.where(target == 1, one_probability, 1-one_probability))
                transformed[j] += probability[i]*transition
        actual = moments(transformed, bits)
        predicted = readout_moments(*moments(probability, bits), p01, p10)
        for x, y in zip(actual, predicted):
            np.testing.assert_allclose(x, y, atol=1e-14)

    def test_dependence_control_keeps_local_diagnostics_but_changes_global_mass(self):
        independent, coupled = dependence_control()
        for key in ("energy", "product_energy", "correlation_retention", "block_optimum_probabilities"):
            self.assertEqual(independent[key], coupled[key])
        self.assertEqual(independent["cross_block_factor"], 1.)
        self.assertEqual(coupled["cross_block_factor"], 2.)

    def test_new_reference_matches_independently_archived_simulation(self):
        actual = control_analysis()["reference"]
        archived = read(PROJECT/"results/hardware_decomposition.json")["faithful_ideal"]
        self.assertAlmostEqual(actual["energy_ideal"], archived["energy_ideal"], places=9)
        self.assertAlmostEqual(actual["ideal_correlation_energy"], archived["correlation_energy_ideal"], places=9)

    def test_unit_forcing_and_penalty_repair_on_small_covering_systems(self):
        from calc_triviality import propagate
        for n in range(1, 5):
            rows = [x for x in itertools.product((0, 1), repeat=n) if any(x)]
            configurations = np.array(list(itertools.product((0, 1), repeat=n)))
            for row1, row2 in itertools.product(rows, repeat=2):
                incidence = np.array([row1, row2])
                for m in range(1, int(incidence.sum(axis=1).min())+1):
                    forced = incidence[incidence.sum(axis=1) == m].any(axis=0)
                    self.assertEqual(propagate(incidence, np.full(2, m))["forced"], int(forced.sum()))
                    deficit = np.maximum(0, m-configurations @ incidence.T)
                    energy = configurations.sum(axis=1) + 1.01*(deficit**2).sum(axis=1)
                    self.assertTrue(np.all(deficit[np.isclose(energy, energy.min(), rtol=0, atol=1e-12)] == 0))
        one_base = np.array([0., 1.])
        np.testing.assert_array_equal(one_base + (1-one_base)**2, [1., 1.])


class BlockCovarianceTests(unittest.TestCase):
    @staticmethod
    def toy():
        bits = np.array(list(itertools.product((0., 1.), repeat=4)))
        pairs = [(np.array([0]), np.array([1]), np.array([-1.])),
                 (np.array([2]), np.array([3]), np.array([-1.]))]
        ref = dict(n=4, pairs=pairs, linear=np.zeros(4), constant=5.,
                   delta=np.array([.25, .25]), weights=np.array([.5, .5]), energy_random=4.5)
        return bits, ref

    def test_decomposition_against_enumerated_distribution(self):
        bits, ref = self.toy()
        probability = np.arange(1., 17.)/136
        mu = probability @ bits
        direct_energy = probability @ (5-bits[:, 0]*bits[:, 1]-bits[:, 2]*bits[:, 3])
        product = 5-mu[0]*mu[1]-mu[2]*mu[3]
        stats = bc.statistics(probability @ bc.features(bits, ref['pairs']), ref)
        self.assertAlmostEqual(stats['energy'][0], direct_energy)
        self.assertAlmostEqual(stats['correlation_energy'][0], product-direct_energy)
        np.testing.assert_allclose(stats['retention'] @ ref['weights'], stats['global_retention'])

    def test_joint_bootstrap_preserves_between_block_covariance(self):
        bits, ref = self.toy()
        selected = bits[[3, 12]]
        boot = bc.bootstrap_moments(bc.features(selected, ref['pairs']), np.array([50, 50]),
                                    np.random.default_rng(78), 1000)
        self.assertLess(np.cov(boot[:, -2:].T)[0, 1], 0)
        self.assertGreater(boot[:, 0].var(), 0)
        np.testing.assert_allclose(bc.statistics(boot, ref)['energy'], 4.)

    def test_fixed_round_statistics_are_not_pooled_moments(self):
        bits, ref = self.toy()
        feature = bc.features(bits, ref['pairs'])
        observed = bc.average_rounds([bc.statistics(feature[[0]], ref), bc.statistics(feature[[-1]], ref)])
        pooled = bc.statistics((feature[[0]]+feature[[-1]])/2, ref)
        self.assertEqual(observed['global_retention'][0], 0.)
        self.assertEqual(pooled['global_retention'][0], 1.)

    def test_homogeneous_output_channel_and_zero_variance_contrasts(self):
        ref = bc.reference()
        mu, pair = readout_moments(ref['mean'], ref['second'], .1, .1)
        moment = np.r_[mu, [np.sum(q*pair[i,j]) for i,j,q in ref['pairs']]]
        output = bc.statistics(moment, ref)
        np.testing.assert_allclose(output['retention'], .64, atol=1e-12)
        np.testing.assert_allclose(output['contrast'], 0, atol=1e-12)
        point = {'contrast': np.zeros((1,6))}
        draws = {'contrast': np.zeros((100,6))}
        band = bc.simultaneous([point], [draws])
        self.assertEqual(band['zero_variance_contrasts'], 6)
        self.assertFalse(any(band['resolved'][0]))

    def test_archived_global_values_and_reconstruction(self):
        result = read(HERE/'generated/block_covariance.json')
        self.assertEqual(result['protocol']['replicates'], 10000)
        self.assertEqual(sum(p['shots'] for p in result['pubs']), 320000)
        self.assertEqual(result['simultaneous']['family_size'], 30)
        weights = np.array(result['reference']['weights'])
        for context in result['contexts']:
            stats = context['statistics']
            self.assertAlmostEqual(weights @ stats['retention']['estimate'],
                                   stats['global_retention']['estimate'], places=12)
            if context['context'].startswith('E'):
                epoch = read(HERE/f"results/{context['context'].lower()}_analysis.json")
                expected = epoch['epoch_summary']['faithful51_correlation_energy_retained']['estimate']
                self.assertAlmostEqual(stats['global_retention']['estimate'], expected, places=11)
            else:
                name = 'exact51_results.json' if context['context'] == 'defaults' else 'exact51_matched_results.json'
                self.assertAlmostEqual(stats['energy']['estimate'], read(PROJECT/'results'/name)['mean_energy'], places=9)


class ManuscriptTests(unittest.TestCase):
    def test_active_structural_derivations_and_cross_references(self):
        main = (HERE.parent/"paper_v7.tex").read_text()
        supplement = (HERE.parent/"supplement_v7.tex").read_text()
        self.assertNotIn(r"\begin{comment}", supplement)
        for label in ("sm:dicke", "sm:contraction", "eq:sm_convolution", "eq:sm_dicke_schmidt"):
            self.assertIn(r"\label{"+label+"}", supplement)
        generated = "\n".join(p.read_text() for p in (HERE/"generated").glob("*.tex"))
        labels = lambda text: set(re.findall(r"\\label\{([^}]+)\}", text))
        available = labels(main) | labels(supplement+generated)
        available |= {"main:"+x for x in labels(main)}
        available |= {"sm:"+x for x in labels(supplement+generated)}
        references = set(re.findall(r"\\(?:eq)?ref\*?\{([^}]+)\}", main+supplement))
        self.assertEqual(references - available, set())

    def test_base_qubit_notation_in_historical_table(self):
        supplement = (HERE.parent/"supplement_v7.tex").read_text()
        at = supplement.index(r"\label{tab:kingston}")
        table = supplement[at:supplement.index(r"\end{table*}", at)]
        self.assertIn(r"$\nq$", table)
        self.assertNotIn(r"$\nB$", table)
        self.assertIn("Faithful encoding (51q)", table)
        self.assertIn(r"24 bases and 27 slack bits", supplement[:at])

    def test_visible_figure_labels_match_estimands(self):
        def visible(name):
            return subprocess.check_output(["pdftotext", str(HERE/"generated"/name), "-"],
                                           text=True).lower()
        ensemble = visible("fig_ensemble.pdf")
        self.assertIn("representation size", ensemble)
        self.assertIn("within each cell", ensemble)
        self.assertNotIn("exact classical cost", ensemble)
        campaign = visible("fig_three_jobs.pdf")
        self.assertIn("conditional counting interval", campaign)
        self.assertIn("individual observed round", campaign)
        self.assertNotIn("hierarchical", campaign)


if __name__ == "__main__":
    unittest.main(verbosity=2)
