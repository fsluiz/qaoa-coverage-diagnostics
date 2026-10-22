"""Regression tests for the version-5 scientific corrections."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_qwoa_qaoa as benchmark  # noqa: E402
from calc_ilp_geographic import haversine_matrix, load_municipalities  # noqa: E402
from calc_ladder_rungs_v5 import signed_meridian_distance_km  # noqa: E402
from calc_qwoa_simulability_v5 import (  # noqa: E402
    entropy_bits,
    hypergeometric_schmidt_spectrum,
)


def result(name: str):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


class QWOATests(unittest.TestCase):
    def test_hypergeometric_spectrum_and_archived_values(self):
        for a in range(25):
            spectrum = hypergeometric_schmidt_spectrum(24, 10, a)
            self.assertAlmostEqual(sum(spectrum), 1.0, places=13)
            self.assertLessEqual(len(spectrum), 11)
        self.assertAlmostEqual(
            entropy_bits(hypergeometric_schmidt_spectrum(24, 10, 4)),
            1.9132494738253447,
            places=13,
        )
        data = result("qwoa_simulability_v5.json")
        self.assertEqual(data["maximum_block_cut_rank"], 11)
        self.assertAlmostEqual(data["maximum_block_cut_entropy_bits"], 2.3496865381073353)
        self.assertEqual(data["maximum_entropy_cut_after_block"], 6)
        self.assertEqual(data["mps_simulation_cost"], 682)

    def test_global_sector_mixer_matches_matrix_exponential(self):
        n, k = 3, 1
        edges = {(0, 1): 1.3, (1, 2): 0.7}
        cardinality = np.array([int(i).bit_count() for i in range(1 << n)])
        problem = {
            "energy": np.zeros(1 << n),
            "feasible": np.ones(1 << n, dtype=bool),
            "card": cardinality,
        }
        sector = benchmark.QWOASector(np.zeros((n, n)), edges, k, problem)
        expected_h = np.zeros((sector.dim, sector.dim))
        positions = {state: i for i, state in enumerate(sector.states)}
        for (i, j), weight in edges.items():
            for row, state in enumerate(sector.states):
                if ((state >> i) & 1) != ((state >> j) & 1):
                    swapped = state ^ (1 << i) ^ (1 << j)
                    expected_h[row, positions[swapped]] += weight
        observed_h = sector.evecs @ np.diag(sector.evals) @ sector.evecs.conj().T
        np.testing.assert_allclose(observed_h, expected_h, atol=1e-12)

        beta = 0.37
        global_u = sector.evecs @ np.diag(np.exp(-1j * beta * sector.evals)) @ sector.evecs.conj().T
        np.testing.assert_allclose(global_u, expm(-1j * beta * expected_h), atol=1e-12)

        edge_hamiltonians = []
        for edge, weight in edges.items():
            h_edge = np.zeros_like(expected_h)
            i, j = edge
            for row, state in enumerate(sector.states):
                if ((state >> i) & 1) != ((state >> j) & 1):
                    swapped = state ^ (1 << i) ^ (1 << j)
                    h_edge[row, positions[swapped]] += weight
            edge_hamiltonians.append(h_edge)
        trotter_u = expm(-1j * beta * edge_hamiltonians[1]) @ expm(
            -1j * beta * edge_hamiltonians[0]
        )
        self.assertGreater(np.linalg.norm(global_u - trotter_u), 1e-3)

    def test_single_edge_gate_angle_identity(self):
        x = np.array([[0, 1], [1, 0]], complex)
        y = np.array([[0, -1j], [1j, 0]], complex)
        beta, weight = 0.37, 1.3
        exact = expm(-1j * beta * weight * (np.kron(x, x) + np.kron(y, y)) / 2)
        rxx = expm(-1j * beta * weight * np.kron(x, x) / 2)
        ryy = expm(-1j * beta * weight * np.kron(y, y) / 2)
        np.testing.assert_allclose(exact, rxx @ ryy, atol=1e-12)

    def test_legacy_digital_qwoa_entry_point_is_disabled(self):
        source = (SCRIPTS / "submit_kingston.py").read_text(encoding="utf-8")
        self.assertIn("def qwoa_first_order_trotter_circuit", source)
        self.assertIn("qwoa_circuit was a legacy unweighted ordered-edge product", source)
        self.assertIn("--include-qwoa-trotter-prototype", source)
        self.assertNotIn("qc.rxx(2 * b[L]", source)


class GeometryTests(unittest.TestCase):
    def test_signed_meridian_projection_is_nonexpansive(self):
        lat, lon, _ = load_municipalities("centroid")
        projection = signed_meridian_distance_km(lat, lon, float(np.min(lon)))
        distances = haversine_matrix(lat, lon)
        violation = np.abs(projection[:, None] - projection[None, :]) - distances
        self.assertLessEqual(float(violation.max()), 1e-9)

    def test_strip_regression(self):
        rung = result("ladder_rungs_v5.json")["rung3"]
        self.assertEqual(rung["strip"]["north_south"]["worst_strip"], 228)
        self.assertEqual(rung["strip"]["east_west"]["worst_strip"], 259)
        self.assertAlmostEqual(rung["uniform_in_outline"]["strip_bound_mean"], 148.3)
        self.assertEqual(rung["uniform_in_outline"]["strip_bound_max"], 172)

    def test_projection_wraps_longitude(self):
        p1 = signed_meridian_distance_km([10.0], [179.5], -179.0)[0]
        p2 = signed_meridian_distance_km([10.0], [-180.5], -179.0)[0]
        self.assertAlmostEqual(float(p1), float(p2), places=10)


class StatisticalAndWorkflowTests(unittest.TestCase):
    def test_interleaved_round_level_primary_interval(self):
        data = result("interleaved_batch_v5.json")
        expected = {
            "A": (0.8771293502636498, [0.8160200292734560, 0.9428149671509033],
                  [0.8516124910719416, 0.9037000103658132],
                  [0.6694881125719797, 1.3069959277556447]),
            "B": (0.6713235499160577, [0.6617159579263774, 0.6810706365374368],
                  [0.6502938055335907, 0.6920737666915947],
                  [0.6989979586155427, 1.5289730178605723]),
        }
        for family, (point, interval, shot_interval, cross_interval) in expected.items():
            row = data["families"][family]
            self.assertAlmostEqual(row["gated_round_geometric_mean"], point, places=13)
            np.testing.assert_allclose(row["gated_round_t_ci95"], interval, atol=1e-13)
            np.testing.assert_allclose(
                row["gated_shot_bootstrap_ci95_conditional"], shot_interval, atol=1e-13
            )
            np.testing.assert_allclose(
                row["cross_block_factor_shot_bootstrap_ci95_conditional"],
                cross_interval,
                atol=1e-13,
            )
            self.assertEqual(row["uncertainty_unit"], "round_within_one_job")
            self.assertFalse(row["twirling_strata_archived"])
            self.assertIn("conditional", row["shot_bootstrap_assumption"])

    def test_v5_workflow_excludes_temporal_proxy(self):
        workflow = (SCRIPTS / "reproduce_v5.py").read_text(encoding="utf-8")
        self.assertNotIn('py("calc_persistent_surveillance.py")', workflow)
        for name in (
            "calc_qwoa_simulability_v5.py",
            "calc_ladder_rungs_v5.py",
            "calc_interleaved_batch_v5.py",
        ):
            self.assertIn(name, workflow)
        self.assertIn('MAIN, SUPP = "paper_v5", "supplement_v5"', workflow)
        self.assertIn('py("verify_paper_claims.py", "--version", "5")', workflow)
        self.assertIn('py("verify_research_audit.py", "--version", "5")', workflow)

    def test_lambda_10_faithfulness_is_archived(self):
        instances = result("exact_encoding.json")["instances"]
        expected = {"rung1": (10, 288), "arco": (31, 4)}
        for name, (energy, degeneracy) in expected.items():
            audit = instances[name]["faithfulness_lambda_10"]
            self.assertEqual(audit["penalty_lambda"], 10)
            self.assertEqual(audit["ground_energy"], energy)
            self.assertEqual(audit["ground_degeneracy"], degeneracy)
            self.assertTrue(audit["all_ground_states_feasible"])

    def test_v4_sources_remain_frozen_against_recorded_baseline(self):
        baseline = (ROOT / "paper" / "BASELINE_V4.md").read_text(encoding="utf-8")
        import hashlib
        for filename in ("paper_v4.tex", "supplement_v4.tex"):
            digest = hashlib.sha256((ROOT / "paper" / filename).read_bytes()).hexdigest()
            self.assertIn(digest, baseline)


if __name__ == "__main__":
    unittest.main()
