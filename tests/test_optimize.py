"""Tests for leakage-safe Optuna configuration helpers."""
import unittest

import pandas as pd
import optuna

from src.optimize import suggest_xgb_params, validate_tuning_config


class OptimizeTests(unittest.TestCase):
    def test_parameter_suggestions_are_within_expected_ranges(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "n_estimators": 250,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_child_weight": 2,
                "subsample": 0.8,
                "colsample_bytree": 0.75,
                "gamma": 1.0,
                "reg_alpha": 0.01,
                "reg_lambda": 1.0,
                "scale_pos_weight": 2.0,
            }
        )
        params = suggest_xgb_params(trial)
        self.assertEqual(params["n_estimators"], 250)
        self.assertEqual(params["max_depth"], 4)
        self.assertEqual(params["n_jobs"], 1)
        self.assertEqual(params["eval_metric"], "logloss")
        self.assertGreaterEqual(params["scale_pos_weight"], 1.0)

    def test_rejects_invalid_trial_count(self) -> None:
        y = pd.Series([0, 0, 0, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "n_trials must be at least 1"):
            validate_tuning_config(y, n_trials=0, n_splits=3)

    def test_rejects_fold_count_larger_than_smallest_class(self) -> None:
        y = pd.Series([0, 0, 0, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "cannot exceed the smallest class count"):
            validate_tuning_config(y, n_trials=2, n_splits=4)

    def test_rejects_single_class_target(self) -> None:
        y = pd.Series([1, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            validate_tuning_config(y, n_trials=2, n_splits=2)


if __name__ == "__main__":
    unittest.main()
