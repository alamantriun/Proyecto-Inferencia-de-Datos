"""Small, explainable classification helpers for the presentation notebook.

The predictive model is logistic regression.  SMOTENC is an optional training
step, never an operation applied to validation or future cases.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.production_pipeline import (
    CATEGORICAL_FEATURES,
    CLASS_TARGET,
    KEY_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    fit_predict_classifier,
)


PRESENTATION_MODELS = ("logistic_plain", "logistic_smotenc")
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
CLASS_NAMES = {0: "Rendimiento bajo", 1: "Rendimiento alto"}


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def weka_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Return the binary metrics shown in WEKA's detailed accuracy table."""
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    positive_probability = np.asarray(probability, dtype=float)
    if not (len(actual) == len(predicted) == len(positive_probability)):
        raise ValueError("y_true, y_pred y probability deben tener igual longitud.")
    if len(actual) == 0 or set(np.unique(actual)) != {0, 1}:
        raise ValueError("El reporte WEKA requiere observaciones de ambas clases.")
    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("Las predicciones deben usar solamente las clases 0 y 1.")
    if ((positive_probability < 0) | (positive_probability > 1)).any():
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")

    rows: list[dict[str, float | int | str]] = []
    for class_value in (0, 1):
        actual_class = (actual == class_value).astype(int)
        predicted_class = (predicted == class_value).astype(int)
        class_probability = (
            positive_probability if class_value == 1 else 1 - positive_probability
        )
        tn, fp, fn, tp = confusion_matrix(
            actual_class, predicted_class, labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "TP Rate": _safe_divide(tp, tp + fn),
                "FP Rate": _safe_divide(fp, fp + tn),
                "Precision": float(
                    precision_score(
                        actual_class, predicted_class, zero_division=0
                    )
                ),
                "Recall": float(
                    recall_score(actual_class, predicted_class, zero_division=0)
                ),
                "F-Measure": float(
                    f1_score(actual_class, predicted_class, zero_division=0)
                ),
                "MCC": float(matthews_corrcoef(actual_class, predicted_class)),
                "ROC Area": float(roc_auc_score(actual_class, class_probability)),
                "PRC Area": float(
                    average_precision_score(actual_class, class_probability)
                ),
                "Support": int(actual_class.sum()),
                "Class": CLASS_NAMES[class_value],
            }
        )

    detailed = pd.DataFrame(rows)
    weights = detailed["Support"].to_numpy(dtype=float)
    metric_columns = [
        "TP Rate",
        "FP Rate",
        "Precision",
        "Recall",
        "F-Measure",
        "MCC",
        "ROC Area",
        "PRC Area",
    ]
    weighted = {
        column: float(np.average(detailed[column], weights=weights))
        for column in metric_columns
    }
    weighted.update({"Support": int(weights.sum()), "Class": "Weighted Avg."})
    detailed = pd.concat([detailed, pd.DataFrame([weighted])], ignore_index=True)

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    summary = {
        "instances": int(len(actual)),
        "correctly_classified": int(np.trace(matrix)),
        "incorrectly_classified": int(len(actual) - np.trace(matrix)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "weighted_precision": float(weighted["Precision"]),
        "precision_high": float(
            detailed.loc[detailed["Class"] == CLASS_NAMES[1], "Precision"].iloc[0]
        ),
        "recall_high": float(
            detailed.loc[detailed["Class"] == CLASS_NAMES[1], "Recall"].iloc[0]
        ),
        "mcc": float(matthews_corrcoef(actual, predicted)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    return {"summary": summary, "detailed_accuracy": detailed}


def evaluate_year_raw(
    df: pd.DataFrame,
    model_name: str,
    year: int,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, BaseEstimator]:
    """Fit only on years before ``year`` and score every eligible case in it."""
    if model_name not in PRESENTATION_MODELS:
        raise ValueError(f"Modelo de presentacion desconocido: {model_name}")
    if not 0 <= threshold <= 1:
        raise ValueError("El umbral de decision debe estar entre 0 y 1.")
    required = set(FEATURES + [TARGET, CLASS_TARGET, "anio"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    eligible = df[df[TARGET].notna() & df["rendimiento_lag_1"].notna()].copy()
    train = eligible[eligible["anio"] < int(year)].copy()
    test = eligible[eligible["anio"] == int(year)].copy()
    if train.empty or test.empty:
        raise ValueError(f"El ano {year} no tiene entrenamiento previo o casos de prueba.")
    train_max_year = int(train["anio"].max())
    if train_max_year >= int(year):
        raise AssertionError("La evaluacion temporal incluyo datos futuros.")

    probability, model = fit_predict_classifier(model_name, train, test)
    prediction = (probability >= float(threshold)).astype(int)
    identity = [column for column in KEY_COLUMNS if column in test.columns]
    result = test[identity].reset_index(drop=False).rename(columns={"index": "row_id"})
    result["y_true"] = test[CLASS_TARGET].to_numpy(dtype=int)
    result["probability_raw"] = probability
    result["y_pred"] = prediction
    result["prediccion"] = pd.Series(prediction).map(CLASS_NAMES)
    result["model"] = model_name
    result["train_max_year"] = train_max_year
    result["decision_threshold"] = float(threshold)
    return result, model


def predict_cases(
    model: BaseEstimator, cases: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """Score several cases at once and return readable labels and probabilities."""
    missing = sorted(set(FEATURES).difference(cases.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias para predecir: {missing}")
    if not 0 <= threshold <= 1:
        raise ValueError("El umbral de decision debe estar entre 0 y 1.")
    probability = np.asarray(
        model.predict_proba(cases[FEATURES])[:, 1], dtype=float
    )
    prediction = (probability >= float(threshold)).astype(int)
    identity = [column for column in KEY_COLUMNS if column in cases.columns]
    result = cases[identity].reset_index(drop=True).copy()
    result["probabilidad_alto"] = probability
    result["prediccion_codigo"] = prediction
    result["prediccion"] = pd.Series(prediction).map(CLASS_NAMES)
    return result
