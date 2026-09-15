from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.evaluation.annual_artifacts import (
    create_annual_figures,
    export_annual_artifacts,
)
from src.evaluation.annual_classifier import ModelRecipe, annual_metrics


def _complete_result(failed_years: list[int] | None = None) -> dict[str, object]:
    failed_years = failed_years or []
    prediction_rows: list[dict[str, object]] = []
    for year in range(2019, 2026):
        actual = [0, 0, 1, 1]
        predicted = [0, int(year in failed_years), 1, 1]
        probabilities = [0.10, 0.70 if year in failed_years else 0.20, 0.75, 0.90]
        for index, (truth, estimate, probability) in enumerate(
            zip(actual, predicted, probabilities)
        ):
            prediction_rows.append(
                {
                    "codigo_dane_municipio": f"05{index + 1:03d}",
                    "anio": year,
                    "y_true": truth,
                    "y_pred": estimate,
                    "probability": probability,
                    "model": "demo",
                    "clase_lag_1": truth,
                    "cambio_clase_vs_lag_1": int(index == 1),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    metrics, weka = annual_metrics(predictions)
    metrics["passes"] = (metrics["mcc"] >= 0.85) & (metrics["kappa"] >= 0.85)

    mart_rows: list[dict[str, object]] = []
    for year in range(2007, 2026):
        for index in range(4):
            mart_rows.append(
                {
                    "codigo_dane_municipio": f"05{index + 1:03d}",
                    "anio": year,
                    "rendimiento_t_ha": 0.6 + 0.25 * index + 0.01 * (year - 2007),
                    "rendimiento_alto": int(index >= 2),
                    "fuente_eva": "EVA_HISTORICA" if year <= 2018 else "EVA_RECIENTE",
                    "climate_available": int(year >= 2007),
                    "temperatura_media_c_lag_1": np.nan if index == 0 else 20 + index,
                }
            )
    mart = pd.DataFrame(mart_rows)
    estimator = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    gate = {
        "status": "target_not_met" if failed_years else "production_candidate",
        "threshold": 0.85,
        "failed_years": failed_years,
        "minimum_mcc": float(metrics["mcc"].min()),
        "minimum_kappa": float(metrics["kappa"].min()),
        "worst_year_improvement_over_persistence": 0.03,
        "failed_checks": [] if not failed_years else ["Hay años bajo el umbral."],
    }
    return {
        "mart": mart,
        "predictions": predictions,
        "winner_predictions": predictions,
        "annual_metrics": metrics,
        "winner_metrics": metrics,
        "weka_details": weka,
        "winner_weka_details": weka,
        "model_summary": pd.DataFrame(
            {
                "model": ["demo", "lag1"],
                "worst_joint_score": [0.86, 0.70],
                "mean_joint_score": [0.90, 0.75],
            }
        ),
        "smote_balance": pd.DataFrame(
            {
                "etapa": ["Antes", "Antes", "Después", "Después"],
                "clase": ["Bajo", "Alto", "Bajo", "Alto"],
                "casos": [20, 80, 80, 80],
            }
        ),
        "feature_importance": pd.DataFrame(
            {
                "feature": ["rendimiento_lag_1", "precipitacion_total_mm_lag_1"],
                "importance": [0.20, 0.05],
            }
        ),
        "winner": "demo",
        "recipe": ModelRecipe("demo", 0.50),
        "gate": gate,
        "source_manifest": {"generated_at_utc": "2026-09-15T00:00:00Z"},
        "estimator": estimator,
        "target_threshold": 0.9259,
    }


def test_figure_bundle_contains_all_annual_confusion_matrices(tmp_path: Path):
    paths = create_annual_figures(_complete_result(), tmp_path)
    names = {path.name for path in paths}

    assert "08_matrices_confusion_2019_2025.png" in names
    assert "09_matriz_confusion_consolidada.png" in names
    assert len(paths) >= 13
    assert all(path.stat().st_size > 0 for path in paths)


def test_export_manifest_keeps_real_failed_years(tmp_path: Path):
    output_dir = tmp_path / "bundle"
    paths = export_annual_artifacts(_complete_result([2019, 2020]), output_dir)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert manifest["status"] == "target_not_met"
    assert manifest["gate"]["failed_years"] == [2019, 2020]
    assert paths["model_card"].exists()
    assert paths["model"].exists()
    assert paths["winner_predictions"].exists()


def test_export_replaces_previous_bundle_only_after_complete_write(tmp_path: Path):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    stale = output_dir / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    paths = export_annual_artifacts(_complete_result(), output_dir)

    assert not stale.exists()
    assert paths["manifest"].exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))
