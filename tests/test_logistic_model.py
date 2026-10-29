"""Checks for the modeling code used by the notebook."""

import unittest

import numpy as np

from logistic_model import (
    SurveyFeatures,
    best_f1_threshold,
    blend_logistic_probabilities,
    classification_metrics,
    fit_elastic_logistic,
    fit_logistic,
    predict_probabilities,
    sigmoid,
    stratified_folds,
    stratified_split,
)


class LogisticModelTests(unittest.TestCase):
    def test_blending_matches_a_weighted_logistic_score(self):
        scores = np.array([[-2.0, 0.2, 3.0], [1.0, -0.8, -1.0]])
        actual = blend_logistic_probabilities(sigmoid(scores), [0.35, 0.65])
        expected = sigmoid(0.35 * scores[0] + 0.65 * scores[1])
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        np.testing.assert_allclose(
            blend_logistic_probabilities(sigmoid(scores), [1.0, 0.0]),
            sigmoid(scores[0]),
        )

    def test_blending_handles_endpoints_and_rejects_invalid_inputs(self):
        actual = blend_logistic_probabilities([[0.0, 1.0], [1.0, 0.0]], [1, 1])
        np.testing.assert_allclose(actual, 0.5)
        for probabilities, weights in [
            ([[0.2], [0.8]], [0, 0]),
            ([[0.2], [0.8]], [1, -1]),
            ([[0.2], [0.8]], [1]),
            ([[np.nan], [0.8]], [1, 1]),
            ([[1.2], [0.8]], [1, 1]),
        ]:
            with self.assertRaises(ValueError):
                blend_logistic_probabilities(probabilities, weights)

    def test_elastic_net_satisfies_optimality_conditions(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(3000, 15)).astype(np.float32)
        truth = np.r_[1.0, -0.8, 0.4, np.zeros(12)]
        y = (rng.random(len(x)) < sigmoid(x @ truth - 0.7)).astype(float)
        l1, l2 = 0.02, 0.001
        w, info = fit_elastic_logistic(x, y, l1=l1, lambda_=l2)
        residual = predict_probabilities(x, w) - y
        gradient = x.T @ residual / len(y) + 2 * l2 * w[:-1]
        nonzero = w[:-1] != 0
        self.assertTrue((~nonzero).any())
        self.assertLess(
            np.max(np.abs(gradient[nonzero] + l1 * np.sign(w[:-1][nonzero]))), 3e-5
        )
        self.assertLessEqual(np.max(np.abs(gradient[~nonzero])), l1 + 3e-5)
        self.assertLess(abs(residual.mean()), 3e-5)

    def test_threshold_matches_exhaustive_search_with_ties(self):
        rng = np.random.default_rng(12)
        y = rng.integers(0, 2, 300)
        probabilities = np.round(rng.random(300), 1)
        threshold, score = best_f1_threshold(y, probabilities)
        expected = max(
            classification_metrics(y, probabilities >= t)["f1"]
            for t in np.unique(probabilities)
        )
        self.assertAlmostEqual(score, expected)
        self.assertAlmostEqual(
            score, classification_metrics(y, probabilities >= threshold)["f1"]
        )
        self.assertEqual(best_f1_threshold(np.zeros(3), np.ones(3))[1], 0.0)

    def test_optimizer_reaches_a_stationary_point(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(4000, 8)).astype(np.float32)
        y = (rng.random(len(x)) < sigmoid(x @ rng.normal(size=8) - 1)).astype(float)
        w, info = fit_logistic(x, y, lambda_=1e-3, tolerance=2e-6)
        self.assertEqual(info["status"], "converged")
        residual = predict_probabilities(x, w) - y
        gradient = np.r_[x.T @ residual / len(y) + 2e-3 * w[:-1], residual.mean()]
        self.assertLess(np.max(np.abs(gradient)), 3e-6)

    def test_intercept_is_unpenalized_and_class_weight_is_respected(self):
        x = np.zeros((1000, 2), dtype=np.float32)
        y = np.r_[np.ones(100), np.zeros(900)]
        for positive_weight in [1.0, 4.0]:
            w, _ = fit_logistic(x, y, lambda_=100.0, pos_weight=positive_weight)
            expected = 100 * positive_weight / (900 + 100 * positive_weight)
            np.testing.assert_allclose(predict_probabilities(x, w), expected, atol=1e-6)
            np.testing.assert_array_equal(w[:-1], 0.0)

    def test_preprocessing_does_not_learn_from_validation_rows(self):
        rng = np.random.default_rng(3)
        x = np.column_stack(
            (
                rng.choice([1.0, 2.0, 7.0, 9.0], 1000),
                rng.normal(size=1000),
                np.full(1000, np.nan),
            )
        )
        x[::10, 1] = np.nan
        encoder = SurveyFeatures("bins")
        training = encoder.fit_transform(x, ["EMPLOY1", "measure", "empty"])
        expected = encoder.transform(x[:10]).copy()
        extra = np.array([[123.0, 1e30, np.nan], [np.nan, np.nan, np.nan]])
        actual = encoder.transform(np.vstack((x[:10], extra)))
        np.testing.assert_array_equal(actual[:10], expected)
        self.assertTrue(np.isfinite(training).all())
        self.assertTrue(np.isfinite(actual).all())
        self.assertIn("EMPLOY1=7", encoder.feature_names)
        self.assertIn("EMPLOY1=9", encoder.feature_names)
        self.assertFalse(
            any(name.startswith("empty") for name in encoder.feature_names)
        )

    def test_explicit_categories_override_the_cardinality_heuristic(self):
        x = np.repeat(np.arange(1, 34, dtype=float), 30)[:, None]
        encoder = SurveyFeatures(categorical_columns=["PHYSHLTH"])
        transformed = encoder.fit_transform(x, ["PHYSHLTH"])
        self.assertEqual(transformed.shape[1], 33)
        self.assertIn("PHYSHLTH=7", encoder.feature_names)
        self.assertNotIn("PHYSHLTH", encoder.feature_names)

    def test_rank_encoding_is_monotone_and_uses_training_quantiles(self):
        x = np.geomspace(1, 1e8, 1000)[:, None]
        x[::20, 0] = np.nan
        encoder = SurveyFeatures(mode="rank")
        training = encoder.fit_transform(x, ["measure"])
        points = np.array([[0.0], [10.0], [1e4], [1e8], [1e20], [np.nan]])
        expected = encoder.transform(points)
        extended = encoder.transform(np.vstack([points, [[-1e30], [1e30]]]))
        np.testing.assert_array_equal(extended[: len(points)], expected)
        self.assertTrue(np.isfinite(training).all())
        self.assertTrue(np.isfinite(expected).all())
        self.assertTrue((np.diff(expected[:5, 0]) >= 0).all())
        self.assertEqual(expected[3, 0], expected[4, 0])

    def test_health_day_codes_and_structural_skips_are_decoded(self):
        rng = np.random.default_rng(19)
        x = rng.integers(1, 31, (1200, 3)).astype(float)
        x[:600] = [88, 88, np.nan]
        x[600:620] = [77, 99, 77]
        encoder = SurveyFeatures(clean_health_days=True)
        encoded = encoder.fit_transform(x, ["PHYSHLTH", "MENTHLTH", "POORHLTH"])
        probes = np.array(
            [[88, 88, np.nan], [88, 88, 88], [77, 99, 77], [np.nan, np.nan, np.nan]]
        )
        transformed = encoder.transform(probes)
        np.testing.assert_array_equal(transformed[0], transformed[1])
        np.testing.assert_array_equal(transformed[2], transformed[3])
        self.assertTrue(np.isfinite(encoded).all())
        self.assertTrue(
            all(kind in {"linear", "missing"} for _, kind, _ in encoder.specs)
        )
        np.testing.assert_array_equal(x[0], [88, 88, np.nan])

    def test_splines_are_restricted_and_can_interact_with_categories(self):
        rng = np.random.default_rng(8)
        x = np.column_stack(
            [rng.uniform(18, 80, 1000), rng.integers(1, 3, 1000), rng.normal(size=1000)]
        )
        encoder = SurveyFeatures(
            mode="spline",
            interactions=True,
            nonlinear_columns=["_AGE80"],
            n_knots=5,
            nonlinear_interactions=True,
        )
        encoded = encoder.fit_transform(x, ["_AGE80", "SEX", "unrelated"])
        self.assertEqual(
            sum(name.startswith("_AGE80>") for name in encoder.feature_names), 5
        )
        self.assertFalse(
            any(name.startswith("unrelated>") for name in encoder.feature_names)
        )
        self.assertTrue(
            any(
                encoder.specs[i][1] == "hinge" or encoder.specs[j][1] == "hinge"
                for i, j in encoder.pairs
            )
        )
        expected = encoder.transform(x[:10])
        actual = encoder.transform(np.vstack([x[:10], [1e8, 9, -1e8]]))
        np.testing.assert_array_equal(actual[:10], expected)
        self.assertTrue(np.isfinite(encoded).all())
        self.assertTrue(np.isfinite(actual).all())

    def test_validation_partitions_are_disjoint_and_reproducible(self):
        y = np.r_[np.zeros(90), np.ones(10)]
        train, validation = stratified_split(y)
        self.assertEqual(len(np.intersect1d(train, validation)), 0)
        self.assertEqual(len(train) + len(validation), len(y))
        np.testing.assert_array_equal(train, stratified_split(y)[0])
        folds = stratified_folds(y)
        np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(len(y)))
        self.assertTrue(all(y[fold].sum() == 2 for fold in folds))


if __name__ == "__main__":
    unittest.main()
