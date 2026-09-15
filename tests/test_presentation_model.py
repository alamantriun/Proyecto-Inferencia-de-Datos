import unittest
import importlib
import importlib.util

import numpy as np
import pandas as pd


def make_temporal_frame() -> pd.DataFrame:
    rows = []
    for year in range(2016, 2025):
        for index in range(16):
            signal = (index % 4) / 3
            rows.append(
                {
                    "departamento": "A" if index % 2 == 0 else "B",
                    "municipio": f"M{index % 5}",
                    "cultivo": "CAFE",
                    "anio": year,
                    "lags_incompletos": 0,
                    "rendimiento_lag_1": 0.6 + signal,
                    "rendimiento_lag_2": 0.5 + signal,
                    "rendimiento_lag_3": 0.4 + signal,
                    "produccion_lag_1": 20 + index,
                    "area_cosechada_lag_1": 10 + index / 10,
                    "area_sembrada_lag_1": 11 + index / 10,
                    "media_rendimiento_3y": 0.5 + signal,
                    "variabilidad_rendimiento_3y": 0.05 + index / 100,
                    "tendencia_rendimiento_3y": -0.1 + signal / 5,
                    "rendimiento_t_ha": 0.7 + signal,
                    "rendimiento_alto": int(index % 4 >= 2),
                }
            )
    return pd.DataFrame(rows)


class PresentationModelTests(unittest.TestCase):
    def setUp(self):
        module_name = "src.evaluation.presentation_model"
        self.assertIsNotNone(
            importlib.util.find_spec(module_name),
            "Falta el modulo de modelo pequeno para presentacion.",
        )
        self.subject = importlib.import_module(module_name)

    def test_weka_report_matches_hand_calculated_binary_example(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        probability = np.array([0.1, 0.6, 0.7, 0.8])

        report = self.subject.weka_report(y_true, y_pred, probability)
        details = report["detailed_accuracy"].set_index("Class")

        self.assertAlmostEqual(details.loc["Rendimiento bajo", "Precision"], 1.0)
        self.assertAlmostEqual(
            details.loc["Rendimiento alto", "Precision"], 2 / 3
        )
        self.assertAlmostEqual(details.loc["Weighted Avg.", "Precision"], 5 / 6)
        self.assertAlmostEqual(report["summary"]["accuracy"], 0.75)
        self.assertAlmostEqual(report["summary"]["mcc"], 1 / np.sqrt(3))
        self.assertEqual(report["summary"]["confusion_matrix"], [[1, 1], [0, 2]])

    def test_raw_year_evaluation_never_trains_on_the_evaluated_year(self):
        frame = make_temporal_frame()

        predictions, _ = self.subject.evaluate_year_raw(
            frame, "logistic_plain", year=2024
        )

        self.assertEqual(set(predictions["anio"]), {2024})
        self.assertEqual(set(predictions["train_max_year"]), {2023})
        self.assertEqual(len(predictions), 16)
        self.assertTrue((predictions["decision_threshold"] == 0.5).all())

    def test_batch_prediction_preserves_cases_and_returns_explainable_labels(self):
        frame = make_temporal_frame()
        _, model = self.subject.evaluate_year_raw(
            frame, "logistic_plain", year=2024
        )
        cases = frame.loc[frame["anio"] == 2024].head(5)

        result = self.subject.predict_cases(model, cases)

        self.assertEqual(len(result), 5)
        self.assertEqual(list(result["municipio"]), list(cases["municipio"]))
        self.assertTrue(result["probabilidad_alto"].between(0, 1).all())
        self.assertTrue(
            set(result["prediccion"]).issubset(
                {"Rendimiento bajo", "Rendimiento alto"}
            )
        )

    def test_batch_prediction_rejects_incomplete_cases(self):
        frame = make_temporal_frame()
        _, model = self.subject.evaluate_year_raw(
            frame, "logistic_plain", year=2024
        )

        with self.assertRaisesRegex(ValueError, "columnas obligatorias"):
            self.subject.predict_cases(model, frame[["municipio"]].head(2))


if __name__ == "__main__":
    unittest.main()
