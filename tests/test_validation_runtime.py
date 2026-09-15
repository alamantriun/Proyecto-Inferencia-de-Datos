import time
import unittest

from src.evaluation.validate_models import (
    CLASSIFICATION_CANDIDATES,
    REGRESSION_CANDIDATES,
    TARGET,
    high_yield_threshold,
    load_data,
    predict_classifier,
    predict_regressor,
)


class ValidationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_data()
        cls.threshold = high_yield_threshold(cls.df)

    def test_all_catboost_variants_fit_one_fold_within_45_seconds(self):
        classification_eligible = self.df[
            self.df[TARGET].notna() & self.df["rendimiento_lag_1"].notna()
        ]
        class_train = classification_eligible[classification_eligible["anio"] < 2019]
        class_test = classification_eligible[classification_eligible["anio"] == 2019]
        regression_eligible = self.df[
            self.df[TARGET].notna()
            & self.df["rendimiento_lag_1"].notna()
            & self.df["media_rendimiento_3y"].notna()
        ]
        reg_train = regression_eligible[regression_eligible["anio"] < 2019]
        reg_test = regression_eligible[regression_eligible["anio"] == 2019]

        started = time.perf_counter()
        for candidate in CLASSIFICATION_CANDIDATES:
            if candidate.family == "catboost":
                predictions = predict_classifier(
                    candidate, class_train, class_test, self.threshold
                )
                self.assertEqual(len(predictions), len(class_test))
        for candidate in REGRESSION_CANDIDATES:
            if candidate.family == "catboost":
                predictions = predict_regressor(candidate, reg_train, reg_test)
                self.assertEqual(len(predictions), len(reg_test))
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 45.0)


if __name__ == "__main__":
    unittest.main()
