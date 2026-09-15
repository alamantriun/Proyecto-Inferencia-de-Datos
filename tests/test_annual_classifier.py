from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import time

from src.evaluation.annual_classifier import (
    MODEL_FEATURES,
    ModelRecipe,
    annual_metrics,
    build_estimator,
    evaluate_final_holdout,
    select_threshold_from_prior_years,
    strict_production_gate,
    walk_forward_backtest,
)


def _temporal_frame(start: int = 2012, end: int = 2020) -> pd.DataFrame:
    rows = []
    for year in range(start, end + 1):
        for index in range(12):
            high = int(index >= 6)
            row = {
                "codigo_dane_municipio": f"05{index + 1:03d}",
                "codigo_dane_departamento": "05",
                "departamento": "ANTIOQUIA",
                "municipio": f"M{index}",
                "anio": year,
                "rendimiento_t_ha": 0.7 + 0.7 * high,
                "rendimiento_alto": high,
                "clase_lag_1": high,
                "cambio_clase_vs_lag_1": 0,
            }
            for feature in MODEL_FEATURES:
                if feature in row:
                    continue
                if feature in {"codigo_dane_municipio", "codigo_dane_departamento"}:
                    continue
                row[feature] = high + (year - start) / 100
            rows.append(row)
    return pd.DataFrame(rows)


def test_annual_metrics_matches_hand_calculated_confusion_matrix_and_kappa():
    predictions = pd.DataFrame(
        {
            "anio": [2024] * 4,
            "y_true": [0, 0, 1, 1],
            "y_pred": [0, 1, 1, 1],
            "probability": [0.1, 0.6, 0.7, 0.8],
            "model": ["demo"] * 4,
        }
    )

    annual, detailed = annual_metrics(predictions)
    row = annual.iloc[0]

    assert [row.tn, row.fp, row.fn, row.tp] == [1, 1, 0, 2]
    assert row.mcc == pytest.approx(1 / np.sqrt(3))
    assert row.kappa == pytest.approx(0.5)
    assert row.weighted_precision == pytest.approx(5 / 6)
    assert "Weighted Avg." in set(detailed["Class"])


def test_threshold_selection_rejects_evaluated_or_future_labels():
    predictions = pd.DataFrame(
        {
            "anio": [2022, 2023, 2024],
            "y_true": [0, 1, 1],
            "probability": [0.2, 0.8, 0.7],
        }
    )

    with pytest.raises(ValueError, match="posteriores"):
        select_threshold_from_prior_years(predictions, evaluation_year=2024)


def test_threshold_selection_uses_worst_annual_mcc_and_kappa():
    predictions = pd.DataFrame(
        {
            "anio": [2021] * 4 + [2022] * 4,
            "y_true": [0, 0, 1, 1] * 2,
            "probability": [0.1, 0.4, 0.6, 0.9, 0.2, 0.45, 0.55, 0.8],
        }
    )

    threshold = select_threshold_from_prior_years(predictions, evaluation_year=2023)

    assert 0.45 < threshold <= 0.55


def test_smotenc_exists_only_in_smotenc_estimator():
    with_smote = build_estimator("logistic_smotenc")
    without_smote = build_estimator("logistic_plain")

    assert "sampler" in with_smote.named_steps
    assert "sampler" not in without_smote.named_steps


def test_logistic_preprocessing_stays_sparse_with_many_municipalities():
    frame = _temporal_frame(start=2012, end=2018)
    estimator = build_estimator("logistic_plain")

    transformed = estimator.named_steps["preprocess"].fit_transform(
        frame[MODEL_FEATURES]
    )

    assert sparse.issparse(transformed)


def test_logistic_candidate_fits_medium_panel_within_five_seconds():
    base = _temporal_frame(start=2012, end=2019)
    copies = []
    for block in range(80):
        copy = base.copy()
        copy["codigo_dane_municipio"] = copy["codigo_dane_municipio"].map(
            lambda code: f"{int(code) + block * 20:05d}"
        )
        copies.append(copy)
    frame = pd.concat(copies, ignore_index=True)
    rng = np.random.default_rng(42)
    for feature in MODEL_FEATURES:
        if feature not in {"codigo_dane_municipio", "codigo_dane_departamento"}:
            frame[feature] = rng.normal(size=len(frame))
    frame["rendimiento_alto"] = rng.integers(0, 2, size=len(frame))
    estimator = build_estimator("logistic_plain")

    started = time.perf_counter()
    estimator.fit(frame[MODEL_FEATURES], frame["rendimiento_alto"])
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0


def test_walk_forward_never_trains_or_selects_threshold_on_evaluated_year():
    frame = _temporal_frame()

    result = walk_forward_backtest(
        frame,
        model_names=["logistic_plain"],
        years=[2019],
        threshold_history_years=[2016, 2017, 2018],
    )
    predictions = result["predictions"]

    assert set(predictions["model"]) == {"majority", "lag1", "logistic_plain"}
    assert set(predictions["anio"]) == {2019}
    assert (predictions["train_max_year"] < predictions["anio"]).all()
    assert (predictions["threshold_source_max_year"] < predictions["anio"]).all()
    assert set(predictions.groupby("model").size()) == {12}


def test_final_holdout_uses_frozen_recipe_and_only_prior_rows():
    frame = _temporal_frame()
    recipe = ModelRecipe(name="logistic_plain", decision_threshold=0.5)

    predictions, model = evaluate_final_holdout(frame, recipe, year=2020)

    assert model is not None
    assert set(predictions["anio"]) == {2020}
    assert set(predictions["train_max_year"]) == {2019}
    assert set(predictions["decision_threshold"]) == {0.5}


def test_gate_fails_when_one_year_is_below_target():
    metrics = pd.DataFrame(
        {
            "anio": [2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "mcc": [.91, .90, .90, .89, .88, .87, .84],
            "kappa": [.90, .89, .90, .88, .87, .86, .88],
        }
    )
    baseline = metrics.assign(mcc=metrics.mcc - 0.05, kappa=metrics.kappa - 0.05)

    gate = strict_production_gate(metrics, baseline, threshold=0.85)

    assert gate["status"] == "target_not_met"
    assert gate["failed_years"] == [2025]
    assert gate["minimum_mcc"] == pytest.approx(0.84)


def test_gate_accepts_only_when_every_year_and_baseline_gain_pass():
    metrics = pd.DataFrame(
        {
            "anio": list(range(2019, 2026)),
            "mcc": [.90] * 7,
            "kappa": [.89] * 7,
        }
    )
    baseline = metrics.assign(mcc=.80, kappa=.80)

    gate = strict_production_gate(metrics, baseline, threshold=0.85)

    assert gate["status"] == "production_candidate"
    assert gate["failed_years"] == []


def test_gate_rejects_an_incomplete_annual_evaluation():
    metrics = pd.DataFrame(
        {"anio": list(range(2019, 2025)), "mcc": [.90] * 6, "kappa": [.90] * 6}
    )
    baseline = metrics.assign(mcc=.70, kappa=.70)

    gate = strict_production_gate(metrics, baseline, threshold=0.85)

    assert gate["status"] == "target_not_met"
    assert gate["missing_years"] == [2025]
    assert any("Faltan años" in failure for failure in gate["failed_checks"])


def test_gate_aligns_baseline_by_year_instead_of_row_position():
    metrics = pd.DataFrame(
        {"anio": list(range(2019, 2026)), "mcc": [.90] * 7, "kappa": [.89] * 7}
    )
    baseline = pd.DataFrame(
        {
            "anio": list(reversed(range(2019, 2026))),
            "mcc": [.80, .70, .70, .70, .70, .70, .70],
            "kappa": [.80, .70, .70, .70, .70, .70, .70],
        }
    )

    gate = strict_production_gate(metrics, baseline, threshold=0.85)

    assert gate["status"] == "production_candidate"
    assert gate["worst_year_improvement_over_persistence"] == pytest.approx(0.19)
