import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib import MatplotlibDeprecationWarning
import numpy as np
import pandas as pd

from src.evaluation.production_pipeline import (
    REQUIRED_PLOTS,
    add_high_yield_target,
    audit_dataset,
    build_classifier,
    build_manifest,
    choose_threshold,
    classification_backtest,
    classification_metric_tables,
    classification_metrics,
    create_all_plots,
    evaluate_classification_holdout,
    evaluate_regression_holdout,
    export_artifacts,
    fit_calibrator,
    fit_predict_classifier,
    fit_predict_regressor,
    manifest_status,
    permutation_feature_importance,
    production_gate,
    regression_backtest,
    regression_metric_tables,
    regression_metrics,
    run_production_pipeline,
    select_regressor,
    select_classifier,
    smotenc_class_counts,
    temporal_splits,
)


def make_classification_frame(n: int, last_year: int) -> pd.DataFrame:
    index = np.arange(n)
    return pd.DataFrame(
        {
            "departamento": np.where(index % 2 == 0, "A", "B"),
            "municipio": np.where(index % 3 == 0, "M1", "M2"),
            "cultivo": np.full(n, "Café"),
            "anio": np.full(n, last_year),
            "lags_incompletos": np.zeros(n),
            "rendimiento_lag_1": 0.7 + (index % 4) * 0.2,
            "rendimiento_lag_2": 0.6 + (index % 4) * 0.2,
            "rendimiento_lag_3": 0.5 + (index % 4) * 0.2,
            "produccion_lag_1": 10.0 + index,
            "area_cosechada_lag_1": 5.0 + index / 10,
            "area_sembrada_lag_1": 6.0 + index / 10,
            "media_rendimiento_3y": 0.6 + (index % 4) * 0.2,
            "variabilidad_rendimiento_3y": 0.1 + (index % 3) * 0.05,
            "tendencia_rendimiento_3y": -0.1 + (index % 5) * 0.05,
            "rendimiento_alto": (index % 4 == 0).astype(int),
            "rendimiento_t_ha": 0.8 + (index % 4) * 0.2,
        }
    )


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "departamento": ["A", "A", "A", "A"],
                "municipio": ["M", "M", "M", "M"],
                "cultivo": ["Café", "Café", "Café", "Café"],
                "anio": [2017, 2018, 2019, 2024],
                "rendimiento_t_ha": [1.0, 3.0, 2.5, 4.0],
                "rendimiento_lag_1": [np.nan, 1.0, 3.0, 2.5],
                "rendimiento_lag_2": [np.nan, np.nan, 1.0, 3.0],
                "rendimiento_lag_3": [np.nan, np.nan, np.nan, 1.0],
                "produccion_lag_1": [np.nan, 10.0, 11.0, 12.0],
                "area_cosechada_lag_1": [np.nan, 4.0, 4.0, 4.0],
                "area_sembrada_lag_1": [np.nan, 5.0, 5.0, 5.0],
                "media_rendimiento_3y": [np.nan, 1.0, 2.0, 2.5],
                "variabilidad_rendimiento_3y": [np.nan, 0.0, 1.0, 0.5],
                "tendencia_rendimiento_3y": [np.nan, 0.0, 2.0, 0.5],
                "lags_incompletos": [1, 1, 0, 0],
                "fuente_eva": ["old", "old", "new", "new"],
            }
        )

    def test_threshold_uses_only_pre_2019_rows(self):
        enriched, threshold = add_high_yield_target(self.df)

        self.assertEqual(threshold, 2.0)
        self.assertEqual(
            enriched.loc[enriched.anio == 2024, "rendimiento_alto"].item(), 1
        )

    def test_duplicate_analytic_keys_are_rejected(self):
        duplicate = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "llaves duplicadas"):
            audit_dataset(duplicate)

    def test_holdout_is_disjoint_from_selection(self):
        splits = temporal_splits(self.df)

        self.assertEqual(set(splits["holdout"].anio), {2024})
        self.assertTrue((splits["selection"].anio < 2024).all())


class SmoteIsolationTests(unittest.TestCase):
    def test_smotenc_pipeline_contains_sampler_only_for_smote_candidates(self):
        with_smote = build_classifier("logistic_smotenc")
        without_smote = build_classifier("logistic_plain")

        self.assertIn("sampler", with_smote.named_steps)
        self.assertNotIn("sampler", without_smote.named_steps)

    def test_sampler_learns_counts_from_train_and_never_resamples_test(self):
        train = make_classification_frame(24, last_year=2018)
        test = make_classification_frame(7, last_year=2019)

        probabilities, fitted = fit_predict_classifier(
            "logistic_smotenc", train, test
        )

        sampler = fitted.named_steps["sampler"]
        self.assertEqual(sampler.sampling_strategy_, {1: 12})
        self.assertEqual(len(probabilities), len(test))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())


class ClassificationEvaluationTests(unittest.TestCase):
    def test_mcc_precision_and_matrix_are_reproducible(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        probability = np.array([0.1, 0.7, 0.8, 0.9])

        metrics = classification_metrics(y_true, y_pred, probability)

        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["mcc"], 1 / np.sqrt(3))
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tp"], 2)

    def test_selection_uses_challenger_mcc_before_balanced_accuracy(self):
        summary = pd.DataFrame(
            [
                {
                    "model": "A",
                    "is_challenger": True,
                    "mean_mcc": 0.30,
                    "mean_balanced_accuracy": 0.80,
                },
                {
                    "model": "B",
                    "is_challenger": True,
                    "mean_mcc": 0.31,
                    "mean_balanced_accuracy": 0.70,
                },
                {
                    "model": "lag1",
                    "is_challenger": False,
                    "mean_mcc": 0.90,
                    "mean_balanced_accuracy": 0.95,
                },
            ]
        )

        self.assertEqual(select_classifier(summary), "B")

    def test_every_challenger_fits_and_returns_probabilities(self):
        train = make_classification_frame(40, last_year=2018)
        test = make_classification_frame(9, last_year=2019)

        for name in [
            "logistic_plain",
            "logistic_smotenc",
            "random_forest_plain",
            "random_forest_smotenc",
            "catboost_weighted",
        ]:
            with self.subTest(model=name):
                probabilities, _ = fit_predict_classifier(name, train, test)
                self.assertEqual(len(probabilities), 9)
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_calibration_rejects_holdout_and_threshold_is_frozen_grid_value(self):
        safe_oof = pd.DataFrame(
            {
                "anio": [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022],
                "y_true": [0, 1, 0, 1, 0, 1, 0, 1],
                "probability_raw": [0.1, 0.9, 0.15, 0.85, 0.2, 0.8, 0.25, 0.75],
            }
        )
        calibrator = fit_calibrator(safe_oof)
        threshold = choose_threshold(safe_oof, calibrator)

        self.assertAlmostEqual(threshold * 100, round(threshold * 100))
        self.assertGreaterEqual(threshold, 0.05)
        self.assertLessEqual(threshold, 0.95)

        contaminated = pd.concat(
            [safe_oof, safe_oof.iloc[[0]].assign(anio=2024)], ignore_index=True
        )
        with self.assertRaisesRegex(ValueError, "2024"):
            fit_calibrator(contaminated)

    def test_backtest_uses_only_prior_years_and_equal_prediction_counts(self):
        train = make_classification_frame(40, last_year=2018)
        test = make_classification_frame(9, last_year=2019)
        data = pd.concat([train, test], ignore_index=True)

        predictions = classification_backtest(
            data,
            high_yield_threshold=1.0,
            years=[2019],
            model_names=["logistic_plain"],
        )

        self.assertEqual(
            set(predictions["model"]), {"majority", "lag1", "logistic_plain"}
        )
        self.assertEqual(set(predictions["train_max_year"]), {2018})
        self.assertEqual(set(predictions.groupby("model").size()), {len(test)})

        annual, summary = classification_metric_tables(predictions)
        self.assertEqual(set(annual["anio"]), {2019})
        self.assertEqual(len(summary), 3)

    def test_holdout_predictions_are_only_2024(self):
        history = make_classification_frame(40, last_year=2018)
        selection = make_classification_frame(12, last_year=2019)
        holdout = make_classification_frame(10, last_year=2024)
        data = pd.concat([history, selection, holdout], ignore_index=True)
        oof = classification_backtest(
            data,
            high_yield_threshold=1.0,
            years=[2019],
            model_names=["logistic_plain"],
        )
        selected = oof[oof["model"] == "logistic_plain"]
        calibrator = fit_calibrator(selected)
        threshold = choose_threshold(selected, calibrator)

        predictions, _ = evaluate_classification_holdout(
            data, "logistic_plain", calibrator, threshold
        )

        self.assertEqual(set(predictions["anio"]), {2024})
        self.assertEqual(set(predictions["train_max_year"]), {2019})
        self.assertEqual(len(predictions), len(holdout))


class RegressionAndGateTests(unittest.TestCase):
    def test_regression_metrics_are_hand_reproducible(self):
        metrics = regression_metrics(
            np.array([1.0, 2.0]), np.array([1.0, 3.0])
        )

        self.assertEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.5))
        self.assertEqual(metrics["r2"], -1.0)

    def test_gate_lists_every_failed_condition(self):
        gate = production_gate(
            integrity_ok=True,
            challenger_mean_mcc=0.31,
            lag1_mean_mcc=0.30,
            challenger_annual_mcc=[0.2, -0.01, 0.3],
            holdout_mcc=0.49,
            holdout_balanced_accuracy=0.69,
            regression_mae=0.96,
            lag1_regression_mae=1.0,
            calibrated_brier=0.21,
            prevalence_brier=0.20,
        )

        self.assertEqual(gate["status"], "experimental_not_approved")
        self.assertEqual(len(gate["failed_checks"]), 6)

    def test_regressors_fit_and_challenger_selection_ignores_baseline(self):
        train = make_classification_frame(40, last_year=2018)
        test = make_classification_frame(9, last_year=2019)
        for name in [
            "elastic_net",
            "random_forest_regressor",
            "catboost_regressor",
        ]:
            with self.subTest(model=name):
                prediction, _ = fit_predict_regressor(name, train, test)
                self.assertEqual(len(prediction), len(test))
                self.assertTrue(np.isfinite(prediction).all())

        summary = pd.DataFrame(
            [
                {"model": "elastic_net", "is_challenger": True, "mean_mae": 0.3},
                {"model": "random_forest_regressor", "is_challenger": True, "mean_mae": 0.2},
                {"model": "lag1", "is_challenger": False, "mean_mae": 0.1},
            ]
        )
        self.assertEqual(select_regressor(summary), "random_forest_regressor")

    def test_regression_backtest_and_holdout_keep_temporal_boundary(self):
        history = make_classification_frame(40, last_year=2018)
        selection = make_classification_frame(12, last_year=2019)
        selection.loc[selection.index[0], "media_rendimiento_3y"] = np.nan
        holdout = make_classification_frame(10, last_year=2024)
        data = pd.concat([history, selection, holdout], ignore_index=True)

        oof = regression_backtest(
            data,
            years=[2019],
            model_names=["elastic_net"],
        )
        self.assertEqual(set(oof["model"]), {"lag1", "mean3", "elastic_net"})
        self.assertEqual(set(oof["train_max_year"]), {2018})
        self.assertEqual(set(oof.groupby("model").size()), {len(selection) - 1})

        annual, summary = regression_metric_tables(oof)
        self.assertEqual(set(annual["anio"]), {2019})
        self.assertEqual(len(summary), 3)

        final, _ = evaluate_regression_holdout(data, "elastic_net")
        self.assertEqual(set(final["anio"]), {2024})
        self.assertEqual(set(final["train_max_year"]), {2019})
        self.assertEqual(len(final), len(holdout))


class ArtifactContractTests(unittest.TestCase):
    def test_manifest_status_depends_on_failed_checks(self):
        self.assertEqual(manifest_status([]), "production_candidate")
        self.assertEqual(
            manifest_status(["MCC holdout inferior a 0.50"]),
            "experimental_not_approved",
        )

    def test_required_plot_inventory_is_complete(self):
        self.assertEqual(
            set(REQUIRED_PLOTS),
            {
                "01_distribucion_rendimiento.png",
                "02_boxplot_rendimiento_anio.png",
                "03_faltantes_variables.png",
                "04_balance_clase_anio.png",
                "05_smotenc_antes_despues.png",
                "06_modelos_mcc_balanced_accuracy.png",
                "07_mcc_anual.png",
                "08_matriz_confusion_2024.png",
                "09_curvas_roc_pr_2024.png",
                "10_calibracion_2024.png",
                "11_mae_modelo_anio.png",
                "12_real_vs_predicho_2024.png",
                "13_importancia_variables.png",
            },
        )

    def test_smotenc_snapshot_balances_train_without_changing_input(self):
        train = make_classification_frame(24, last_year=2018)
        original_rows = len(train)

        counts = smotenc_class_counts(train)

        before = counts.set_index("class_label")["before"].to_dict()
        after = counts.set_index("class_label")["after"].to_dict()
        self.assertEqual(before, {0: 18, 1: 6})
        self.assertEqual(after, {0: 18, 1: 18})
        self.assertEqual(len(train), original_rows)

    def test_manifest_preserves_gate_and_confusion_matrix_contract(self):
        gate = {
            "status": "experimental_not_approved",
            "checks": [],
            "failed_checks": ["El MCC 2024 es inferior a 0.50."],
        }
        manifest = build_manifest(
            audit={"rows": 100, "year_min": 2007, "year_max": 2024},
            dataset_sha256="abc123",
            high_yield_threshold=0.9,
            decision_threshold=0.55,
            classification_winner="logistic_smotenc",
            classification_selection={"mean_mcc": 0.4},
            classification_holdout={
                "mcc": 0.49,
                "precision": 0.8,
                "tn": 10,
                "fp": 2,
                "fn": 3,
                "tp": 15,
            },
            regression_winner="elastic_net",
            regression_selection={"mean_mae": 0.2},
            regression_holdout={"mae": 0.3, "rmse": 0.4, "r2": 0.1},
            gate=gate,
        )

        self.assertEqual(manifest["status"], "experimental_not_approved")
        self.assertEqual(
            manifest["classification"]["holdout_2024"]["confusion_matrix"],
            [[10, 2], [3, 15]],
        )
        self.assertEqual(manifest["training_windows"]["evaluation_max_year"], 2023)
        self.assertEqual(manifest["training_windows"]["operational_max_year"], 2024)

    def test_plot_bundle_creates_every_nonempty_png(self):
        data = make_classification_frame(40, last_year=2019)
        data["anio"] = np.tile([2017, 2018, 2019, 2020], 10)
        class_holdout = pd.DataFrame(
            {
                "y_true": [0, 0, 1, 1, 0, 1],
                "y_pred": [0, 1, 1, 1, 0, 0],
                "probability_calibrated": [0.1, 0.6, 0.9, 0.8, 0.2, 0.4],
            }
        )
        result = {
            "data": data,
            "audit": {
                "missing_percent": {
                    "rendimiento_lag_1": 2.0,
                    "rendimiento_lag_2": 5.0,
                }
            },
            "smote_counts": pd.DataFrame(
                {
                    "class_label": [0, 1],
                    "class_name": ["Rendimiento bajo", "Rendimiento alto"],
                    "before": [30, 10],
                    "after": [30, 30],
                }
            ),
            "classification_summary": pd.DataFrame(
                {
                    "model": ["lag1", "logistic_smotenc"],
                    "mean_mcc": [0.30, 0.40],
                    "mean_balanced_accuracy": [0.65, 0.72],
                }
            ),
            "classification_annual": pd.DataFrame(
                {
                    "model": ["lag1", "lag1", "logistic_smotenc", "logistic_smotenc"],
                    "anio": [2019, 2020, 2019, 2020],
                    "mcc": [0.2, 0.4, 0.3, 0.5],
                }
            ),
            "classification_holdout": class_holdout,
            "regression_annual": pd.DataFrame(
                {
                    "model": ["lag1", "lag1", "elastic_net", "elastic_net"],
                    "anio": [2019, 2020, 2019, 2020],
                    "mae": [0.3, 0.4, 0.2, 0.3],
                }
            ),
            "regression_holdout": pd.DataFrame(
                {"y_true": [0.8, 1.0, 1.2], "y_pred": [0.9, 0.9, 1.1]}
            ),
            "feature_importance": pd.DataFrame(
                {"feature": ["rendimiento_lag_1", "anio"], "importance": [0.2, 0.1]}
            ),
        }

        with TemporaryDirectory() as directory:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                paths = create_all_plots(result, Path(directory))

            self.assertEqual({path.name for path in paths}, set(REQUIRED_PLOTS))
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            deprecations = [
                warning
                for warning in caught
                if issubclass(warning.category, MatplotlibDeprecationWarning)
            ]
            self.assertEqual(deprecations, [])

    def test_permutation_importance_is_model_agnostic_and_feature_aligned(self):
        train = make_classification_frame(40, last_year=2018)
        test = make_classification_frame(12, last_year=2019)
        _, model = fit_predict_classifier("logistic_plain", train, test)

        importance = permutation_feature_importance(
            model,
            test,
            test["rendimiento_alto"].to_numpy(),
            repeats=2,
        )

        self.assertEqual(
            set(importance["feature"]),
            {
                "departamento",
                "municipio",
                "anio",
                "lags_incompletos",
                "rendimiento_lag_1",
                "rendimiento_lag_2",
                "rendimiento_lag_3",
                "produccion_lag_1",
                "area_cosechada_lag_1",
                "area_sembrada_lag_1",
                "media_rendimiento_3y",
                "variabilidad_rendimiento_3y",
                "tendencia_rendimiento_3y",
            },
        )
        self.assertTrue(np.isfinite(importance["importance"]).all())

    def test_export_writes_models_tables_manifest_and_model_card(self):
        frame = pd.DataFrame({"model": ["example"], "value": [1.0]})
        result = {
            "manifest": {
                "status": "experimental_not_approved",
                "classification": {"winner": "logistic_plain"},
                "regression": {"winner": "elastic_net"},
                "gate": {"failed_checks": ["evidencia insuficiente"]},
            },
            "audit": {"rows": 1},
            "balance": frame,
            "smote_counts": frame,
            "classification_oof": frame,
            "classification_summary": frame,
            "classification_annual": frame,
            "classification_holdout": frame,
            "regression_oof": frame,
            "regression_summary": frame,
            "regression_annual": frame,
            "regression_holdout": frame,
            "classification_evaluation_model": {"kind": "classification-eval"},
            "classification_operational_model": {"kind": "classification-op"},
            "calibrator": {"kind": "calibrator"},
            "regression_evaluation_model": {"kind": "regression-eval"},
            "regression_operational_model": {"kind": "regression-op"},
        }

        with TemporaryDirectory() as directory:
            paths = export_artifacts(result, Path(directory))
            names = {path.name for path in paths.values()}

            self.assertIn("model_manifest.json", names)
            self.assertIn("MODEL_CARD.md", names)
            self.assertIn("classifier_pipeline.joblib", names)
            self.assertIn("clasificacion_holdout_2024.csv", names)
            card = (Path(directory) / "MODEL_CARD.md").read_text(encoding="utf-8")
            self.assertIn("experimental_not_approved", card)
            self.assertIn("No usar", card)

    def test_orchestrator_stops_before_outputs_when_data_contract_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_data = root / "invalid.csv"
            output = root / "outputs"
            pd.DataFrame({"anio": [2024]}).to_csv(invalid_data, index=False)

            with self.assertRaisesRegex(ValueError, "columnas obligatorias"):
                run_production_pipeline(invalid_data, output)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
