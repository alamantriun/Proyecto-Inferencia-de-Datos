from __future__ import annotations

import json
import hashlib
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score, confusion_matrix, matthews_corrcoef, precision_score


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "pipeline_interactivo_cafe.ipynb"
OUTPUT = ROOT / "reports" / "modelo_mcc_kappa_cafe"


def _notebook_output_text(notebook) -> str:
    """Collect values actually produced by executed code cells."""
    rendered = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                rendered.append(output.text)
            elif output.output_type in {"display_data", "execute_result"}:
                rendered.append(str(output.get("data", {}).get("text/plain", "")))
    return "\n".join(rendered)


def test_notebook_has_twenty_ordered_sections_and_strict_language():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    headings = [
        cell.source.splitlines()[0]
        for cell in notebook.cells
        if cell.cell_type == "markdown" and cell.source.startswith("## ")
    ]
    section_numbers = [int(value.split(".")[0].removeprefix("## ")) for value in headings]
    source = "\n".join(cell.source for cell in notebook.cells)

    assert section_numbers == list(range(1, 21))
    assert "SMOTENC" in source
    assert "MCC" in source
    assert "Kappa" in source
    assert "target_not_met" in source
    assert "RETRAIN_MODELS" in source
    assert "walk_forward_backtest" in source
    assert "evaluate_final_holdout" in source
    assert "train_test_split" not in source


def test_executed_notebook_foregrounds_and_proves_the_2025_result():
    """Catch a notebook that reports 2025 without auditable data lineage or math."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    headings = [
        cell.source.splitlines()[0]
        for cell in notebook.cells
        if cell.cell_type == "markdown" and cell.source.startswith("## ")
    ]
    rendered = _notebook_output_text(notebook)

    assert headings[0] == "## 1. Resultado central de 2025"
    assert "10,270 casos históricos elegibles (2008-2024)" in rendered
    assert "45 variables predictoras: 2 categóricas + 43 numéricas" in rendered
    assert "Balance antes de SMOTENC: Bajo=4,471; Alto=5,799" in rendered
    assert "Balance después de SMOTENC: Bajo=5,799; Alto=5,799" in rendered
    assert "2025: 640 filas iniciales - 4 excluidas = 636 casos evaluados" in rendered
    assert "05045" in rendered
    assert "23555" in rendered
    assert "23807" in rendered
    assert "68132" in rendered
    assert "Aciertos=567; errores=69; accuracy=0.891509" in rendered
    assert "MCC=59462/81353.673377=0.730907" in rendered
    assert "Kappa=(0.891509-0.597504)/(1-0.597504)=0.730456" in rendered
    assert "Precisión WEKA ponderada=0.893607 (89.36%)" in rendered
    assert "0.73 no es 73% de accuracy" in rendered


def test_bundle_has_all_figures_and_seven_confusion_matrices():
    figures = sorted(OUTPUT.glob("*.png"))
    matrices = pd.read_csv(OUTPUT / "matrices_confusion_2019_2025.csv")

    assert len(figures) == 14
    assert set(matrices["anio"]) == set(range(2019, 2026))
    assert (matrices[["tn", "fp", "fn", "tp"]].sum(axis=1) == matrices["instances"]).all()


def test_saved_metrics_recompute_exactly_from_annual_predictions():
    predictions = pd.read_csv(OUTPUT / "predicciones_modelo_seleccionado_2019_2025.csv")
    saved = pd.read_csv(OUTPUT / "metricas_anuales_modelo_seleccionado.csv").set_index("anio")

    for year, group in predictions.groupby("anio"):
        row = saved.loc[year]
        assert matthews_corrcoef(group.y_true, group.y_pred) == pytest.approx(row.mcc)
        assert cohen_kappa_score(group.y_true, group.y_pred) == pytest.approx(row.kappa)
        assert precision_score(group.y_true, group.y_pred, average="weighted", zero_division=0) == pytest.approx(row.weighted_precision)
        tn, fp, fn, tp = confusion_matrix(group.y_true, group.y_pred, labels=[0, 1]).ravel()
        assert [tn, fp, fn, tp] == [row.tn, row.fp, row.fn, row.tp]


def test_development_predictions_are_strictly_temporal_and_exclude_2025():
    predictions = pd.read_csv(OUTPUT / "predicciones_todos_modelos_2019_2024.csv")

    assert set(predictions["anio"]) == set(range(2019, 2025))
    assert (predictions["train_max_year"] < predictions["anio"]).all()
    assert (predictions["threshold_source_max_year"] < predictions["anio"]).all()


def test_manifest_refuses_production_when_any_annual_metric_is_below_target():
    manifest = json.loads((OUTPUT / "model_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUTPUT / "metricas_anuales_modelo_seleccionado.csv")
    failed = metrics.loc[(metrics.mcc < 0.85) | (metrics.kappa < 0.85), "anio"].tolist()

    assert manifest["status"] == "target_not_met"
    assert manifest["gate"]["failed_years"] == failed
    assert failed == list(range(2019, 2026))
    assert manifest["winner"] == "logistic_smotenc"


def test_serialized_model_reloads_with_identical_2025_probabilities():
    from src.evaluation.annual_classifier import MODEL_FEATURES

    manifest = json.loads((OUTPUT / "model_manifest.json").read_text(encoding="utf-8"))
    model = joblib.load(OUTPUT / "modelo_evaluado_2025.joblib")
    mart = pd.read_csv(
        ROOT / "data" / "processed" / "cafe_annual_model_mart.csv",
        dtype={"codigo_dane_municipio": str, "codigo_dane_departamento": str},
    )
    holdout = mart.query("anio == 2025 and rendimiento_lag_1.notna()")
    saved = pd.read_csv(OUTPUT / "predicciones_modelo_seleccionado_2019_2025.csv").query("anio == 2025")
    probabilities = model.predict_proba(holdout[MODEL_FEATURES])[:, 1]
    estimates = (probabilities >= manifest["recipe"]["decision_threshold"]).astype(int)

    assert np.allclose(probabilities, saved.probability.to_numpy(), rtol=0, atol=1e-12)
    assert np.array_equal(estimates, saved.y_pred.to_numpy())


def test_manifest_hashes_match_every_published_artifact():
    manifest = json.loads((OUTPUT / "model_manifest.json").read_text(encoding="utf-8"))

    for name, expected in manifest["artifact_hashes_sha256"].items():
        actual = hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
        assert actual == expected
