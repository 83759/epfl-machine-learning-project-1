"""Behavioral checks for the NumPy histogram boosting implementation."""

import unittest

import numpy as np

from boosted_trees import (
    HistogramFeatures,
    fit_boosted_trees,
    predict_boosted_probabilities,
)


class BoostedTreeTests(unittest.TestCase):
    def test_missing_values_can_define_a_split(self):
        x = np.r_[np.zeros(400), np.ones(400), np.full(200, np.nan)][:, None]
        y = np.isnan(x[:, 0]).astype(float)
        encoder = HistogramFeatures()
        encoded = encoder.fit_transform(x, ["measure"])
        model, _ = fit_boosted_trees(
            encoded, y, max_depth=1, n_estimators=20, min_leaf=20, l2=1
        )
        probabilities = predict_boosted_probabilities(encoded, model)
        np.testing.assert_array_equal(probabilities >= 0.5, y == 1)
        self.assertTrue(np.isfinite(probabilities).all())

    def test_validation_selects_a_real_checkpoint_and_transform_is_fixed(self):
        rng = np.random.default_rng(15)
        x = rng.normal(size=(1000, 3))
        y = ((x[:, 0] > 0) & (x[:, 1] > 0)).astype(float)
        encoder = HistogramFeatures()
        fitting = encoder.fit_transform(x[:700], ["a", "b", "c"])
        validation = encoder.transform(x[700:])
        before = validation.copy()
        encoder.transform(np.array([[1e10, np.nan, -1e10]]))
        np.testing.assert_array_equal(encoder.transform(x[700:]), before)
        model, info = fit_boosted_trees(
            fitting,
            y[:700],
            max_depth=2,
            n_estimators=50,
            min_leaf=10,
            l2=1,
            feature_fraction=1,
            validation_data=(validation, y[700:]),
        )
        best = max(info["history"], key=lambda row: row["f1"])
        self.assertEqual(len(model["trees"]), best["iterations"])
        probabilities = predict_boosted_probabilities(validation, model)
        predictions = probabilities >= info["threshold"]
        f1 = (
            2
            * np.sum(predictions & (y[700:] == 1))
            / (predictions.sum() + y[700:].sum())
        )
        self.assertAlmostEqual(f1, info["validation_f1"])
        self.assertGreater(f1, 0.9)
        replay, _ = fit_boosted_trees(
            fitting,
            y[:700],
            max_depth=2,
            n_estimators=info["iterations"],
            min_leaf=10,
            l2=1,
            feature_fraction=1,
        )
        np.testing.assert_array_equal(
            predict_boosted_probabilities(validation, replay), probabilities
        )


if __name__ == "__main__":
    unittest.main()
