import unittest

import numpy as np

from scripts.algo2_simrank.core import (
    _point_flip_weight,
    _symmetric_positive_radius,
    analyze_pooled_scores,
    analyze_scores,
    gaussian_max_t,
    rank_sets_from_pair_intervals,
)
from scripts.algo2_simrank.validate_decisions import method_decisions, scenario_abilities, truth_verdicts


class SimultaneousRankingTests(unittest.TestCase):
    def test_rank_sets_from_resolved_and_unresolved_pairs(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        rows = rank_sets_from_pair_intervals(
            3,
            pairs,
            lower=np.array([0.2, 0.3, -0.1]),
            upper=np.array([0.4, 0.5, 0.1]),
            point_scores=np.array([0.8, 0.6, 0.5]),
        )
        self.assertEqual((rows[0]["rank_lower"], rows[0]["rank_upper"]), (1, 1))
        self.assertEqual((rows[1]["rank_lower"], rows[1]["rank_upper"]), (2, 3))
        self.assertEqual((rows[2]["rank_lower"], rows[2]["rank_upper"]), (2, 3))

    def test_weight_flip_and_confidence_radius(self):
        self.assertAlmostEqual(_point_flip_weight(0.2, -0.1), 1.0 / 3.0)
        self.assertIsNone(_point_flip_weight(0.2, 0.1))
        self.assertAlmostEqual(_symmetric_positive_radius(0.2, -0.1, 0.5), 1.0 / 6.0)
        self.assertEqual(_symmetric_positive_radius(-0.2, -0.1, 0.5), 0.0)

    def test_gaussian_calibration_is_reproducible(self):
        rng = np.random.default_rng(11)
        centered = rng.normal(size=(6, 200))
        centered -= centered.mean(axis=1, keepdims=True)
        a = gaussian_max_t(centered, 0.05, 500, 19)
        b = gaussian_max_t(centered, 0.05, 500, 19)
        self.assertEqual(a, b)
        self.assertGreater(a.critical_value, 1.96)

    def test_protocol_reversal_is_detected(self):
        rng = np.random.default_rng(7)
        n = 1200
        latent = rng.random((2, n))
        scores = np.empty((2, 2, n), dtype=float)
        scores[0, 0] = latent[0] < 0.75
        scores[1, 0] = latent[0] < 0.45
        scores[0, 1] = latent[1] < 0.35
        scores[1, 1] = latent[1] < 0.70
        result = analyze_scores(scores, ["A", "B"], bootstrap_repetitions=1000, seed=9)
        self.assertEqual(result["pairs"][0]["verdict"], "protocol-dependent")
        self.assertIsNotNone(result["pairs"][0]["point_flip_weight"])
        self.assertEqual(result["pairs"][0]["reference_winner"], 0)

    def test_decision_scenarios_have_expected_truth(self):
        scenarios = scenario_abilities(4)
        self.assertEqual(set(truth_verdicts(scenarios["global-null"])), {"unresolved"})
        self.assertTrue(all(value.startswith("stable-") for value in truth_verdicts(scenarios["stable"])))
        self.assertEqual(set(truth_verdicts(scenarios["protocol-reversal"])), {"protocol-dependent"})

    def test_pooled_method_cannot_report_protocol_dependence(self):
        abilities = scenario_abilities(4)["protocol-reversal"]
        probabilities = 1.0 / (1.0 + np.exp(-abilities[:, :, None]))
        rng = np.random.default_rng(91)
        scores = rng.random((4, 2, 2000)) < np.broadcast_to(probabilities, (4, 2, 2000))
        decisions = method_decisions(
            scores.astype(float),
            [f"model-{index}" for index in range(4)],
            alpha=0.05,
            bootstrap_repetitions=200,
            seed=92,
        )
        self.assertNotIn("protocol-dependent", decisions["pooled-max-t"][0])

    def test_pooled_max_t_is_reproducible(self):
        rng = np.random.default_rng(93)
        scores = rng.integers(0, 2, size=(4, 2, 400)).astype(float)
        a = analyze_pooled_scores(scores, ["A", "B", "C", "D"], bootstrap_repetitions=300, seed=94)
        b = analyze_pooled_scores(scores, ["A", "B", "C", "D"], bootstrap_repetitions=300, seed=94)
        self.assertEqual(a["critical_value"], b["critical_value"])
        self.assertTrue(all(row["verdict"] != "protocol-dependent" for row in a["pairs"]))


if __name__ == "__main__":
    unittest.main()
