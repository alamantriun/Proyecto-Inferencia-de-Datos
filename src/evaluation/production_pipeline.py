"""Pipeline temporal riguroso para preparar modelos de rendimiento de café."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_curve,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "reports" / "tablas_entrenamiento" / "dataset_cafe_ml_ready.csv"
OUTPUT_DIR = ROOT / "reports" / "modelo_produccion_cafe"
TARGET = "rendimiento_t_ha"
CLASS_TARGET = "rendimiento_alto"
KEY_COLUMNS = ["departamento", "municipio", "cultivo", "anio"]
CATEGORICAL_FEATURES = ["departamento", "municipio"]
NUMERIC_FEATURES = [
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
]
LEAKAGE_COLUMNS = ["score_confiabilidad", "dato_copiado"]
SELECTION_YEARS = [2019, 2020, 2021, 2022, 2023]
HOLDOUT_YEAR = 2024
RANDOM_STATE = 42
CLASSIFIER_NAMES = [
    "logistic_plain",
    "logistic_smotenc",
    "random_forest_plain",
    "random_forest_smotenc",
    "catboost_weighted",
]
REGRESSOR_NAMES = [
    "elastic_net",
    "random_forest_regressor",
    "catboost_regressor",
]
REQUIRED_PLOTS = [
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
]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the modeling table without silently changing its schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el dataset requerido: {path}")
    return pd.read_csv(path)


def audit_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate the analytic grain, target, years, and one-year lag."""
    required = KEY_COLUMNS + [TARGET, "fuente_eva"] + NUMERIC_FEATURES
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    duplicate_count = int(df.duplicated(KEY_COLUMNS).sum())
    if duplicate_count:
        raise ValueError(
            f"Se encontraron {duplicate_count} llaves duplicadas en {KEY_COLUMNS}."
        )

    target_nulls = int(df[TARGET].isna().sum())
    if target_nulls:
        raise ValueError(f"El target contiene {target_nulls} valores nulos.")

    observed_years = set(df["anio"].astype(int).unique())
    missing_years = sorted(set(range(2007, HOLDOUT_YEAR + 1)).difference(observed_years))
    if missing_years:
        raise ValueError(f"Faltan años esperados en el dataset: {missing_years}")

    group_columns = ["departamento", "municipio", "cultivo"]
    previous = df[group_columns + ["anio", TARGET]].copy()
    previous["anio"] = previous["anio"] + 1
    previous = previous.rename(columns={TARGET: "target_previo_verificado"})
    checked = df.merge(previous, on=KEY_COLUMNS, how="left", validate="one_to_one")
    comparable = checked["target_previo_verificado"].notna() & checked[
        "rendimiento_lag_1"
    ].notna()
    matches = np.isclose(
        checked.loc[comparable, "rendimiento_lag_1"],
        checked.loc[comparable, "target_previo_verificado"],
        rtol=1e-9,
        atol=1e-12,
    )
    mismatch_count = int((~matches).sum())
    if mismatch_count:
        raise ValueError(
            f"rendimiento_lag_1 no coincide con el año anterior en {mismatch_count} filas."
        )

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "year_min": int(df["anio"].min()),
        "year_max": int(df["anio"].max()),
        "duplicate_keys": duplicate_count,
        "target_nulls": target_nulls,
        "lag_comparisons": int(comparable.sum()),
        "lag_matches": int(matches.sum()),
        "missing_percent": {
            column: float(value)
            for column, value in (df.isna().mean().mul(100)).items()
        },
        "source_counts": {
            str(source): int(count)
            for source, count in df["fuente_eva"].value_counts(dropna=False).items()
        },
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
    }


def add_high_yield_target(
    df: pd.DataFrame, cutoff_year: int = 2018
) -> tuple[pd.DataFrame, float]:
    """Create a binary target using a threshold frozen before selection years."""
    reference = df.loc[df["anio"] <= cutoff_year, TARGET].dropna()
    if reference.empty:
        raise ValueError("No existen observaciones hasta 2018 para calcular el umbral.")
    threshold = float(reference.median())
    result = df.copy()
    result[CLASS_TARGET] = (result[TARGET] > threshold).astype("int8")
    return result, threshold


def temporal_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return disjoint modeling periods with comparable lag-one coverage."""
    eligible = df[df[TARGET].notna() & df["rendimiento_lag_1"].notna()].copy()
    return {
        "history": eligible[eligible["anio"] <= 2018].copy(),
        "selection": eligible[eligible["anio"].isin(SELECTION_YEARS)].copy(),
        "holdout": eligible[eligible["anio"] == HOLDOUT_YEAR].copy(),
    }


def smotenc_class_counts(train: pd.DataFrame) -> pd.DataFrame:
    """Return class counts before and after resampling a training snapshot."""
    if train.empty or train[CLASS_TARGET].nunique() < 2:
        raise ValueError("SMOTENC requiere un entrenamiento con ambas clases.")
    y_train = train[CLASS_TARGET].astype(int).to_numpy()
    minority_count = int(pd.Series(y_train).value_counts().min())
    if minority_count < 2:
        raise ValueError("SMOTENC requiere al menos dos filas en la clase minoritaria.")
    transformed = _ordinal_preprocessor().fit_transform(
        train[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    )
    sampler = SMOTENC(
        categorical_features=[0, 1],
        random_state=RANDOM_STATE,
        k_neighbors=min(5, minority_count - 1),
    )
    _, resampled_y = sampler.fit_resample(transformed, y_train)
    before = pd.Series(y_train).value_counts().reindex([0, 1], fill_value=0)
    after = pd.Series(resampled_y).value_counts().reindex([0, 1], fill_value=0)
    return pd.DataFrame(
        {
            "class_label": [0, 1],
            "class_name": ["Rendimiento bajo", "Rendimiento alto"],
            "before": before.to_numpy(dtype=int),
            "after": after.to_numpy(dtype=int),
        }
    )


def _ordinal_preprocessor() -> ColumnTransformer:
    categorical = SkPipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    numeric = SkPipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
        ]
    )


def _one_hot_after_ordinal() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                [0, 1],
            ),
            (
                "numeric",
                "passthrough",
                list(range(2, 2 + len(NUMERIC_FEATURES))),
            ),
        ]
    )


class CatBoostClassifierPipeline(ClassifierMixin, BaseEstimator):
    """Small serializable adapter that learns all imputations from train."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostClassifierPipeline":
        frame = X[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
        self.numeric_medians_ = frame[NUMERIC_FEATURES].median().fillna(0.0)
        prepared = self._prepare(frame)
        counts = pd.Series(y).astype(int).value_counts()
        if set(counts.index) != {0, 1}:
            raise ValueError("CatBoost requiere ambas clases en entrenamiento.")
        total = float(counts.sum())
        weights = [total / (2.0 * counts[0]), total / (2.0 * counts[1])]
        self.model_ = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            class_weights=weights,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        self.model_.fit(prepared, pd.Series(y).astype(int), cat_features=CATEGORICAL_FEATURES)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = prepared.shape[1]
        return self

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "numeric_medians_")
        prepared = X[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
        for column in CATEGORICAL_FEATURES:
            prepared[column] = (
                prepared[column].astype("string").fillna("__MISSING__").astype(str)
            )
        prepared[NUMERIC_FEATURES] = prepared[NUMERIC_FEATURES].apply(
            pd.to_numeric, errors="coerce"
        )
        prepared[NUMERIC_FEATURES] = prepared[NUMERIC_FEATURES].fillna(
            self.numeric_medians_
        )
        return prepared

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict_proba(self._prepare(X)), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_classifier(name: str) -> BaseEstimator:
    """Build a fresh classifier under one of the documented candidate names."""
    if name == "catboost_weighted":
        return CatBoostClassifierPipeline()
    if name not in {
        "logistic_plain",
        "logistic_smotenc",
        "random_forest_plain",
        "random_forest_smotenc",
    }:
        raise ValueError(f"Clasificador desconocido: {name}")

    steps: list[tuple[str, Any]] = [("preprocessor", _ordinal_preprocessor())]
    if name.endswith("_smotenc"):
        steps.append(
            (
                "sampler",
                SMOTENC(categorical_features=[0, 1], random_state=RANDOM_STATE),
            )
        )
    if name.startswith("logistic"):
        steps.extend(
            [
                ("one_hot", _one_hot_after_ordinal()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        )
    else:
        steps.append(
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        )
    return ImbPipeline(steps)


def fit_predict_classifier(
    name: str, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, BaseEstimator]:
    """Fit on train and return positive-class probabilities for untouched test."""
    if train.empty or test.empty:
        raise ValueError("Train y test deben contener filas.")
    if train[CLASS_TARGET].nunique() < 2:
        raise ValueError("El entrenamiento contiene una sola clase.")
    model = build_classifier(name)
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    model.fit(train[features], train[CLASS_TARGET].astype(int))
    probabilities = model.predict_proba(test[features])[:, 1]
    if len(probabilities) != len(test):
        raise AssertionError("La predicción alteró el número de filas de test.")
    return np.asarray(probabilities, dtype=float), model


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    """Return weighted absolute calibration error across fixed probability bins."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(y_true[mask].mean()) - float(probability[mask].mean())
            )
    return float(error)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray
) -> dict[str, float | int]:
    """Compute imbalance-aware metrics and a fixed-label confusion matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if not (len(y_true) == len(y_pred) == len(probability)):
        raise ValueError("y_true, y_pred y probability deben tener igual longitud.")
    if np.unique(y_true).size < 2:
        raise ValueError("Las métricas requieren ambas clases en y_true.")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": expected_calibration_error(y_true, probability, bins=10),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_classifier(summary: pd.DataFrame) -> str:
    """Select the best learned challenger without allowing a baseline to win."""
    required = {"model", "is_challenger", "mean_mcc", "mean_balanced_accuracy"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Faltan columnas para seleccionar clasificador: {sorted(missing)}")
    challengers = summary[summary["is_challenger"].astype(bool)].copy()
    if challengers.empty:
        raise ValueError("No existen clasificadores challenger elegibles.")
    ranked = challengers.sort_values(
        ["mean_mcc", "mean_balanced_accuracy", "model"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return str(ranked.iloc[0]["model"])


def _calibration_features(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_calibrator(oof: pd.DataFrame) -> LogisticRegression:
    """Fit Platt calibration from selection-period out-of-fold probabilities."""
    required = {"anio", "y_true", "probability_raw"}
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"Faltan columnas para calibración: {sorted(missing)}")
    if (oof["anio"] >= HOLDOUT_YEAR).any():
        raise ValueError("La calibración no puede observar el holdout 2024.")
    if oof["y_true"].nunique() < 2:
        raise ValueError("La calibración requiere ambas clases.")
    calibrator = LogisticRegression(random_state=RANDOM_STATE)
    calibrator.fit(
        _calibration_features(oof["probability_raw"].to_numpy()),
        oof["y_true"].astype(int),
    )
    return calibrator


def calibrated_probabilities(
    calibrator: LogisticRegression, probability: np.ndarray
) -> np.ndarray:
    """Apply a fitted Platt calibrator to raw probabilities."""
    return calibrator.predict_proba(_calibration_features(probability))[:, 1]


def choose_threshold(oof: pd.DataFrame, calibrator: LogisticRegression) -> float:
    """Choose a frozen threshold by mean annual MCC, recall, then closeness to 0.5."""
    probabilities = calibrated_probabilities(
        calibrator, oof["probability_raw"].to_numpy()
    )
    y_true = oof["y_true"].to_numpy(dtype=int)
    years = oof["anio"].to_numpy(dtype=int)
    ranked: list[tuple[float, float, float, float]] = []
    for threshold in np.linspace(0.05, 0.95, 91):
        prediction = (probabilities >= threshold).astype(int)
        annual_mcc: list[float] = []
        annual_recall: list[float] = []
        for year in sorted(np.unique(years)):
            mask = years == year
            annual_mcc.append(float(matthews_corrcoef(y_true[mask], prediction[mask])))
            annual_recall.append(
                float(recall_score(y_true[mask], prediction[mask], zero_division=0))
            )
        ranked.append(
            (
                float(np.mean(annual_mcc)),
                float(np.mean(annual_recall)),
                -abs(float(threshold) - 0.5),
                float(threshold),
            )
        )
    return max(ranked)[3]


def _classification_prediction_frame(
    test: pd.DataFrame,
    model: str,
    probability: np.ndarray,
    prediction: np.ndarray,
    train_max_year: int,
    is_challenger: bool,
) -> pd.DataFrame:
    columns = [column for column in KEY_COLUMNS if column in test.columns]
    result = test[columns].reset_index(drop=False).rename(columns={"index": "row_id"})
    result["y_true"] = test[CLASS_TARGET].to_numpy(dtype=int)
    result["probability_raw"] = np.asarray(probability, dtype=float)
    result["y_pred"] = np.asarray(prediction, dtype=int)
    result["model"] = model
    result["train_max_year"] = int(train_max_year)
    result["is_challenger"] = bool(is_challenger)
    return result


def classification_backtest(
    df: pd.DataFrame,
    high_yield_threshold: float,
    years: list[int] | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Create rolling-origin classification predictions for model selection."""
    years = SELECTION_YEARS if years is None else list(years)
    model_names = CLASSIFIER_NAMES if model_names is None else list(model_names)
    unknown = sorted(set(model_names).difference(CLASSIFIER_NAMES))
    if unknown:
        raise ValueError(f"Clasificadores no registrados: {unknown}")
    eligible = df[df[TARGET].notna() & df["rendimiento_lag_1"].notna()].copy()
    frames: list[pd.DataFrame] = []
    for year in years:
        train = eligible[eligible["anio"] < year].copy()
        test = eligible[eligible["anio"] == year].copy()
        if train.empty or test.empty:
            raise ValueError(f"Fold {year} no tiene train o validación.")
        if train[CLASS_TARGET].nunique() < 2 or test[CLASS_TARGET].nunique() < 2:
            raise ValueError(f"Fold {year} contiene una sola clase.")
        train_max_year = int(train["anio"].max())
        if train_max_year >= year:
            raise AssertionError(f"Leakage temporal detectado en fold {year}.")

        prevalence = float(train[CLASS_TARGET].mean())
        majority_prediction = np.full(len(test), int(prevalence >= 0.5), dtype=int)
        majority_probability = np.full(len(test), prevalence, dtype=float)
        frames.append(
            _classification_prediction_frame(
                test,
                "majority",
                majority_probability,
                majority_prediction,
                train_max_year,
                False,
            )
        )

        lag_prediction = (
            test["rendimiento_lag_1"].to_numpy(dtype=float) > high_yield_threshold
        ).astype(int)
        frames.append(
            _classification_prediction_frame(
                test,
                "lag1",
                lag_prediction.astype(float),
                lag_prediction,
                train_max_year,
                False,
            )
        )

        for name in model_names:
            probability, _ = fit_predict_classifier(name, train, test)
            prediction = (probability >= 0.5).astype(int)
            frames.append(
                _classification_prediction_frame(
                    test,
                    name,
                    probability,
                    prediction,
                    train_max_year,
                    True,
                )
            )
    return pd.concat(frames, ignore_index=True)


def classification_metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize each model by year, then average years with equal weight."""
    annual_rows: list[dict[str, Any]] = []
    for (model, year), group in predictions.groupby(["model", "anio"], sort=True):
        metrics = classification_metrics(
            group["y_true"].to_numpy(),
            group["y_pred"].to_numpy(),
            group["probability_raw"].to_numpy(),
        )
        annual_rows.append(
            {
                "model": str(model),
                "anio": int(year),
                "is_challenger": bool(group["is_challenger"].iloc[0]),
                "n": int(len(group)),
                **metrics,
            }
        )
    annual = pd.DataFrame(annual_rows)
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier",
        "ece",
    ]
    summary = (
        annual.groupby(["model", "is_challenger"], as_index=False)[metric_columns]
        .mean()
        .rename(columns={column: f"mean_{column}" for column in metric_columns})
    )
    return annual, summary


def evaluate_classification_holdout(
    df: pd.DataFrame,
    winner: str,
    calibrator: LogisticRegression,
    decision_threshold: float,
) -> tuple[pd.DataFrame, BaseEstimator]:
    """Fit through 2023 and evaluate the frozen classifier once on 2024."""
    eligible = df[df[TARGET].notna() & df["rendimiento_lag_1"].notna()].copy()
    train = eligible[eligible["anio"] < HOLDOUT_YEAR].copy()
    test = eligible[eligible["anio"] == HOLDOUT_YEAR].copy()
    if train.empty or test.empty:
        raise ValueError("El holdout requiere train histórico y filas de 2024.")
    train_max_year = int(train["anio"].max())
    if train_max_year >= HOLDOUT_YEAR:
        raise AssertionError("El entrenamiento del holdout contiene 2024.")
    raw_probability, model = fit_predict_classifier(winner, train, test)
    probability = calibrated_probabilities(calibrator, raw_probability)
    prediction = (probability >= float(decision_threshold)).astype(int)
    result = _classification_prediction_frame(
        test,
        winner,
        raw_probability,
        prediction,
        train_max_year,
        True,
    )
    result["probability_calibrated"] = probability
    result["decision_threshold"] = float(decision_threshold)
    if len(result) != len(test) or set(result["anio"]) != {HOLDOUT_YEAR}:
        raise AssertionError("La evaluación final no conserva exactamente el holdout 2024.")
    return result, model


class CatBoostRegressorPipeline(RegressorMixin, BaseEstimator):
    """Serializable CatBoost regressor with train-only imputations."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostRegressorPipeline":
        frame = X[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
        self.numeric_medians_ = frame[NUMERIC_FEATURES].median().fillna(0.0)
        prepared = self._prepare(frame)
        self.model_ = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="MAE",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        self.model_.fit(prepared, pd.Series(y).astype(float), cat_features=CATEGORICAL_FEATURES)
        self.n_features_in_ = prepared.shape[1]
        return self

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "numeric_medians_")
        prepared = X[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
        for column in CATEGORICAL_FEATURES:
            prepared[column] = (
                prepared[column].astype("string").fillna("__MISSING__").astype(str)
            )
        prepared[NUMERIC_FEATURES] = prepared[NUMERIC_FEATURES].apply(
            pd.to_numeric, errors="coerce"
        )
        prepared[NUMERIC_FEATURES] = prepared[NUMERIC_FEATURES].fillna(
            self.numeric_medians_
        )
        return prepared

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(self._prepare(X)), dtype=float)


def build_regressor(name: str) -> BaseEstimator:
    """Build a fresh regression challenger."""
    if name == "catboost_regressor":
        return CatBoostRegressorPipeline()
    if name == "elastic_net":
        estimator: BaseEstimator = ElasticNet(
            alpha=0.01,
            l1_ratio=0.2,
            max_iter=10000,
            random_state=RANDOM_STATE,
        )
    elif name == "random_forest_regressor":
        estimator = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Regresor desconocido: {name}")
    return SkPipeline(
        [
            ("preprocessor", _ordinal_preprocessor()),
            ("one_hot", _one_hot_after_ordinal()),
            ("model", estimator),
        ]
    )


def fit_predict_regressor(
    name: str, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, BaseEstimator]:
    """Fit a regression challenger on train and predict untouched test."""
    if train.empty or test.empty:
        raise ValueError("Train y test deben contener filas.")
    model = build_regressor(name)
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    model.fit(train[features], train[TARGET].astype(float))
    prediction = np.asarray(model.predict(test[features]), dtype=float)
    if len(prediction) != len(test):
        raise AssertionError("La regresión alteró el número de filas de test.")
    return prediction, model


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute regression error in target units and explained variance."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener igual longitud.")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def select_regressor(summary: pd.DataFrame) -> str:
    """Select the learned regressor with lowest mean annual MAE."""
    required = {"model", "is_challenger", "mean_mae"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Faltan columnas para seleccionar regresor: {sorted(missing)}")
    challengers = summary[summary["is_challenger"].astype(bool)].copy()
    if challengers.empty:
        raise ValueError("No existen regresores challenger elegibles.")
    ranked = challengers.sort_values(
        ["mean_mae", "model"], ascending=[True, True], kind="mergesort"
    )
    return str(ranked.iloc[0]["model"])


def manifest_status(failed_checks: list[str]) -> str:
    """Map the gate outcome to the only two allowed artifact states."""
    return "production_candidate" if not failed_checks else "experimental_not_approved"


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_manifest(
    audit: dict[str, Any],
    dataset_sha256: str,
    high_yield_threshold: float,
    decision_threshold: float,
    classification_winner: str,
    classification_selection: dict[str, Any],
    classification_holdout: dict[str, Any],
    regression_winner: str,
    regression_selection: dict[str, Any],
    regression_holdout: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Build the complete machine-readable model governance record."""
    failed_checks = list(gate.get("failed_checks", []))
    normalized_gate = dict(gate)
    normalized_gate["status"] = manifest_status(failed_checks)
    class_metrics = dict(classification_holdout)
    class_metrics["confusion_matrix"] = [
        [int(class_metrics["tn"]), int(class_metrics["fp"])],
        [int(class_metrics["fn"]), int(class_metrics["tp"])],
    ]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": normalized_gate["status"],
        "purpose": "Priorización de rendimiento y riesgo productivo de café.",
        "prohibited_uses": [
            "Afirmar rentabilidad, ROI o éxito empresarial.",
            "Aprobar automáticamente crédito, inversión o subsidios.",
        ],
        "data": {
            "path": str(DATA_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": dataset_sha256,
            "audit": audit,
        },
        "targets": {
            "regression": TARGET,
            "classification": CLASS_TARGET,
            "high_yield_threshold_t_ha": float(high_yield_threshold),
        },
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "excluded_leakage": LEAKAGE_COLUMNS,
            "excluded_external": True,
        },
        "training_windows": {
            "threshold_max_year": 2018,
            "selection_years": SELECTION_YEARS,
            "holdout_year": HOLDOUT_YEAR,
            "evaluation_max_year": 2023,
            "operational_max_year": 2024,
        },
        "classification": {
            "winner": classification_winner,
            "selection_2019_2023": classification_selection,
            "decision_threshold": float(decision_threshold),
            "holdout_2024": class_metrics,
        },
        "regression": {
            "winner": regression_winner,
            "selection_2019_2023": regression_selection,
            "holdout_2024": regression_holdout,
        },
        "gate": normalized_gate,
        "software_versions": {
            package: version(package)
            for package in [
                "numpy",
                "pandas",
                "scikit-learn",
                "imbalanced-learn",
                "catboost",
                "joblib",
            ]
        },
        "random_state": RANDOM_STATE,
    }
    return _json_value(manifest)


def production_gate(
    integrity_ok: bool,
    challenger_mean_mcc: float,
    lag1_mean_mcc: float,
    challenger_annual_mcc: list[float],
    holdout_mcc: float,
    holdout_balanced_accuracy: float,
    regression_mae: float,
    lag1_regression_mae: float,
    calibrated_brier: float,
    prevalence_brier: float,
) -> dict[str, Any]:
    """Apply the seven frozen acceptance checks and explain every failure."""
    regression_improvement = (
        (lag1_regression_mae - regression_mae) / lag1_regression_mae
        if lag1_regression_mae > 0
        else float("-inf")
    )
    checks = [
        ("integridad_datos", bool(integrity_ok), "Falló la integridad de datos."),
        (
            "mejora_mcc_vs_lag1",
            challenger_mean_mcc - lag1_mean_mcc >= 0.02,
            "El MCC promedio no supera al baseline t-1 por al menos 0.02.",
        ),
        (
            "mcc_anual_no_negativo",
            bool(challenger_annual_mcc) and min(challenger_annual_mcc) >= 0.0,
            "Existe al menos un año de selección con MCC negativo.",
        ),
        (
            "mcc_holdout",
            holdout_mcc >= 0.50,
            "El MCC 2024 es inferior a 0.50.",
        ),
        (
            "balanced_accuracy_holdout",
            holdout_balanced_accuracy >= 0.70,
            "La balanced accuracy 2024 es inferior a 0.70.",
        ),
        (
            "mejora_mae_regresion",
            regression_improvement >= 0.05,
            "La regresión no mejora al baseline t-1 al menos 5% en MAE 2024.",
        ),
        (
            "brier_vs_prevalencia",
            calibrated_brier < prevalence_brier,
            "El Brier calibrado no mejora la probabilidad constante por prevalencia.",
        ),
    ]
    rendered = [
        {"name": name, "passed": bool(passed), "failure_reason": reason}
        for name, passed, reason in checks
    ]
    failures = [item["failure_reason"] for item in rendered if not item["passed"]]
    return {
        "status": manifest_status(failures),
        "checks": rendered,
        "failed_checks": failures,
        "regression_mae_improvement": float(regression_improvement),
    }


def _regression_prediction_frame(
    test: pd.DataFrame,
    model: str,
    prediction: np.ndarray,
    train_max_year: int,
    is_challenger: bool,
) -> pd.DataFrame:
    columns = [column for column in KEY_COLUMNS if column in test.columns]
    result = test[columns].reset_index(drop=False).rename(columns={"index": "row_id"})
    result["y_true"] = test[TARGET].to_numpy(dtype=float)
    result["y_pred"] = np.asarray(prediction, dtype=float)
    result["model"] = model
    result["train_max_year"] = int(train_max_year)
    result["is_challenger"] = bool(is_challenger)
    return result


def _regression_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Use one common row set so every regressor is compared fairly."""
    return df[
        df[TARGET].notna()
        & df["rendimiento_lag_1"].notna()
        & df["media_rendimiento_3y"].notna()
    ].copy()


def regression_backtest(
    df: pd.DataFrame,
    years: list[int] | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Create rolling-origin regression predictions for model selection."""
    years = SELECTION_YEARS if years is None else list(years)
    model_names = REGRESSOR_NAMES if model_names is None else list(model_names)
    unknown = sorted(set(model_names).difference(REGRESSOR_NAMES))
    if unknown:
        raise ValueError(f"Regresores no registrados: {unknown}")
    eligible = _regression_eligible(df)
    frames: list[pd.DataFrame] = []
    for year in years:
        train = eligible[eligible["anio"] < year].copy()
        test = eligible[eligible["anio"] == year].copy()
        if train.empty or test.empty:
            raise ValueError(f"Fold {year} no tiene train o validación.")
        train_max_year = int(train["anio"].max())
        if train_max_year >= year:
            raise AssertionError(f"Leakage temporal detectado en fold {year}.")

        frames.append(
            _regression_prediction_frame(
                test,
                "lag1",
                test["rendimiento_lag_1"].to_numpy(dtype=float),
                train_max_year,
                False,
            )
        )
        mean_three = test["media_rendimiento_3y"].to_numpy(dtype=float)
        frames.append(
            _regression_prediction_frame(
                test,
                "mean3",
                mean_three,
                train_max_year,
                False,
            )
        )
        for name in model_names:
            prediction, _ = fit_predict_regressor(name, train, test)
            frames.append(
                _regression_prediction_frame(
                    test,
                    name,
                    prediction,
                    train_max_year,
                    True,
                )
            )
    return pd.concat(frames, ignore_index=True)


def regression_metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize regression errors by model/year and equal-weight annual means."""
    annual_rows: list[dict[str, Any]] = []
    for (model, year), group in predictions.groupby(["model", "anio"], sort=True):
        metrics = regression_metrics(
            group["y_true"].to_numpy(), group["y_pred"].to_numpy()
        )
        annual_rows.append(
            {
                "model": str(model),
                "anio": int(year),
                "is_challenger": bool(group["is_challenger"].iloc[0]),
                "n": int(len(group)),
                **metrics,
            }
        )
    annual = pd.DataFrame(annual_rows)
    summary = (
        annual.groupby(["model", "is_challenger"], as_index=False)[
            ["mae", "rmse", "r2"]
        ]
        .mean()
        .rename(
            columns={"mae": "mean_mae", "rmse": "mean_rmse", "r2": "mean_r2"}
        )
    )
    return annual, summary


def evaluate_regression_holdout(
    df: pd.DataFrame, winner: str
) -> tuple[pd.DataFrame, BaseEstimator]:
    """Fit the selected regressor through 2023 and predict only 2024."""
    eligible = _regression_eligible(df)
    train = eligible[eligible["anio"] < HOLDOUT_YEAR].copy()
    test = eligible[eligible["anio"] == HOLDOUT_YEAR].copy()
    if train.empty or test.empty:
        raise ValueError("El holdout requiere train histórico y filas de 2024.")
    train_max_year = int(train["anio"].max())
    if train_max_year >= HOLDOUT_YEAR:
        raise AssertionError("El entrenamiento del regresor contiene 2024.")
    prediction, model = fit_predict_regressor(winner, train, test)
    result = _regression_prediction_frame(
        test, winner, prediction, train_max_year, True
    )
    if len(result) != len(test) or set(result["anio"]) != {HOLDOUT_YEAR}:
        raise AssertionError("La regresión final no conserva exactamente el holdout 2024.")
    return result, model


def _save_figure(figure: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def create_all_plots(result: dict[str, Any], output_dir: Path) -> list[Path]:
    """Create the complete initial-data and final-results visual evidence bundle."""
    output_dir = Path(output_dir)
    sns.set_theme(style="whitegrid")
    data = result["data"]
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(data=data, x=TARGET, bins=35, kde=True, ax=axis, color="#2f6f4e")
    axis.set(
        title="Distribución inicial del rendimiento de café (2007-2024)",
        xlabel="Rendimiento (t/ha)",
        ylabel="Observaciones",
    )
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[0]))

    figure, axis = plt.subplots(figsize=(13, 5))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="vert: bool was deprecated.*",
            category=MatplotlibDeprecationWarning,
        )
        sns.boxplot(
            data=data,
            x="anio",
            y=TARGET,
            ax=axis,
            color="#8eb69b",
            fliersize=1,
        )
    axis.set(
        title="Distribución anual del rendimiento de café (2007-2024)",
        xlabel="Año",
        ylabel="Rendimiento (t/ha)",
    )
    axis.tick_params(axis="x", rotation=45)
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[1]))

    missing = pd.Series(result["audit"]["missing_percent"], dtype=float).sort_values()
    missing = missing[missing > 0]
    if missing.empty:
        missing = pd.Series({"Sin faltantes": 0.0})
    figure, axis = plt.subplots(figsize=(10, max(6, len(missing) * 0.34)))
    axis.barh(missing.index, missing.values, color="#b9785c", edgecolor="#6f4938")
    for index, value in enumerate(missing.values):
        axis.text(value + 0.6, index, f"{value:.1f}%", va="center", fontsize=8)
    axis.set(
        title="Perfil de faltantes por variable (2007-2024)",
        xlabel="Observaciones faltantes (%)",
        ylabel="Variable",
        xlim=(0, max(100.0, float(missing.max()) + 8.0)),
    )
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[2]))

    balance = data.groupby("anio", as_index=False)[CLASS_TARGET].mean()
    balance["proporcion_pct"] = balance[CLASS_TARGET] * 100
    figure, axis = plt.subplots(figsize=(11, 5))
    sns.barplot(data=balance, x="anio", y="proporcion_pct", ax=axis, color="#4d9078")
    axis.axhline(50, color="#a44a3f", linestyle="--", linewidth=1)
    axis.set(
        title="Proporción anual de rendimiento alto (umbral pre-2019)",
        xlabel="Año",
        ylabel="Clase alta (%)",
        ylim=(0, 100),
    )
    axis.tick_params(axis="x", rotation=45)
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[3]))

    smote = result["smote_counts"].melt(
        id_vars=["class_label", "class_name"],
        value_vars=["before", "after"],
        var_name="stage",
        value_name="rows",
    )
    smote["stage"] = smote["stage"].map(
        {"before": "Antes (train real)", "after": "Después (train SMOTENC)"}
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=smote, x="class_name", y="rows", hue="stage", ax=axis)
    axis.set(
        title="Balance del entrenamiento 2007-2018 antes y después de SMOTENC",
        xlabel="Clase",
        ylabel="Filas de entrenamiento",
    )
    axis.legend(title="Momento")
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[4]))

    comparison = result["classification_summary"].melt(
        id_vars="model",
        value_vars=["mean_mcc", "mean_balanced_accuracy"],
        var_name="metric",
        value_name="score",
    )
    comparison["metric"] = comparison["metric"].map(
        {"mean_mcc": "MCC", "mean_balanced_accuracy": "Balanced accuracy"}
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=comparison, y="model", x="score", hue="metric", ax=axis)
    axis.set(
        title="Comparación temporal de clasificadores (promedio anual 2019-2023)",
        xlabel="Puntuación",
        ylabel="Modelo",
    )
    axis.legend(title="Métrica")
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[5]))

    annual_class = result["classification_annual"]
    mcc_matrix = annual_class.pivot(index="model", columns="anio", values="mcc")
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        mcc_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=min(0.0, float(mcc_matrix.min().min())),
        vmax=max(0.6, float(mcc_matrix.max().max())),
        cbar_kws={"label": "MCC"},
        ax=axis,
    )
    axis.set(
        title="Estabilidad anual del MCC por modelo (2019-2023)",
        xlabel="Año",
        ylabel="Modelo",
    )
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[6]))

    class_holdout = result["classification_holdout"]
    matrix = confusion_matrix(class_holdout["y_true"], class_holdout["y_pred"], labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicho bajo", "Predicho alto"],
        yticklabels=["Real bajo", "Real alto"],
        ax=axis,
    )
    axis.set(title="Matriz de confusión del holdout 2024", xlabel="Predicción", ylabel="Real")
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[7]))

    y_true = class_holdout["y_true"].to_numpy(dtype=int)
    probability = class_holdout["probability_calibrated"].to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(y_true, probability)
    precision, recall, _ = precision_recall_curve(y_true, probability)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#2f6f4e")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="ROC 2024", xlabel="Tasa de falsos positivos", ylabel="Recall")
    axes[1].plot(recall, precision, color="#a44a3f")
    axes[1].axhline(y_true.mean(), linestyle="--", color="gray")
    axes[1].set(title="Precision-Recall 2024", xlabel="Recall", ylabel="Precisión")
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[8]))

    observed, predicted = calibration_curve(y_true, probability, n_bins=10, strategy="quantile")
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(predicted, observed, marker="o", color="#2f6f4e", label="Modelo")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")
    axis.set(
        title="Calibración de probabilidades en 2024",
        xlabel="Probabilidad predicha",
        ylabel="Frecuencia observada",
    )
    axis.legend()
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[9]))

    annual_regression = result["regression_annual"]
    mae_matrix = annual_regression.pivot(index="model", columns="anio", values="mae")
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        mae_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrBr",
        cbar_kws={"label": "MAE (t/ha)"},
        ax=axis,
    )
    axis.set(
        title="MAE anual de regresión (2019-2023)",
        xlabel="Año",
        ylabel="Modelo",
    )
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[10]))

    regression_holdout = result["regression_holdout"]
    low = float(min(regression_holdout["y_true"].min(), regression_holdout["y_pred"].min()))
    high = float(max(regression_holdout["y_true"].max(), regression_holdout["y_pred"].max()))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        regression_holdout["y_true"],
        regression_holdout["y_pred"],
        alpha=0.45,
        color="#2f6f4e",
        edgecolors="none",
    )
    axis.plot([low, high], [low, high], "--", color="gray")
    axis.set(
        title="Rendimiento real frente a predicho en 2024",
        xlabel="Real (t/ha)",
        ylabel="Predicho (t/ha)",
    )
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[11]))

    importance = result["feature_importance"].sort_values("importance").tail(15)
    figure, axis = plt.subplots(figsize=(9, 6))
    colors = ["#cad4d0" if value < 0 else "#4d806d" for value in importance["importance"]]
    axis.barh(importance["feature"], importance["importance"], color=colors)
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set(
        title="Importancia por permutación del clasificador en 2024",
        xlabel="Caída media del MCC al permutar",
        ylabel="Variable",
    )
    paths.append(_save_figure(figure, output_dir, REQUIRED_PLOTS[12]))

    if {path.name for path in paths} != set(REQUIRED_PLOTS):
        raise AssertionError("El inventario de gráficas quedó incompleto.")
    return paths


def permutation_feature_importance(
    model: BaseEstimator,
    test: pd.DataFrame,
    y_true: np.ndarray,
    repeats: int = 5,
) -> pd.DataFrame:
    """Measure holdout MCC loss after permuting each raw input feature."""
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    measured = permutation_importance(
        model,
        test[features],
        np.asarray(y_true, dtype=int),
        scoring=make_scorer(matthews_corrcoef),
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": features,
            "importance": measured.importances_mean.astype(float),
            "importance_std": measured.importances_std.astype(float),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


def _model_card(manifest: dict[str, Any]) -> str:
    failures = manifest.get("gate", {}).get("failed_checks", [])
    failure_lines = "\n".join(f"- {reason}" for reason in failures) or "- Ninguna."
    return f"""# Model Card — Rendimiento de café

## Estado

`{manifest['status']}`

## Uso permitido

- Priorizar revisión técnica de rendimiento y riesgo productivo.
- Comparar señales históricas municipales antes de una evaluación agronómica humana.

## No usar

- No interpretar la clase alta como rentabilidad, ROI o éxito empresarial.
- No aprobar de forma automática créditos, inversiones, subsidios o intervenciones.
- No usar fuera de Colombia o para cultivos distintos de café sin una nueva validación.

## Modelos

- Clasificación: `{manifest['classification']['winner']}`.
- Regresión: `{manifest['regression']['winner']}`.
- Evaluación congelada hasta 2023 y holdout final 2024.
- Artefacto operativo reentrenado con datos disponibles hasta 2024.

## Fallos de la compuerta

{failure_lines}

## Limitaciones conocidas

- Existe drift anual y un cambio de fuente EVA a partir de 2019.
- Las variables externas de suelo y crédito fueron excluidas por trazabilidad temporal o geográfica insuficiente.
- `score_confiabilidad` y `dato_copiado` fueron excluidas porque dependen del target.
- Las probabilidades describen rendimiento alto respecto al umbral histórico, no éxito financiero.
"""


def export_artifacts(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write the machine-readable evidence, model card, and serialized estimators."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    json_payloads = {
        "manifest": ("model_manifest.json", result["manifest"]),
        "audit": ("calidad_datos.json", result["audit"]),
    }
    for key, (filename, payload) in json_payloads.items():
        path = output_dir / filename
        path.write_text(
            json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths[key] = path

    tables = {
        "balance": ("balance_clases.csv", result["balance"]),
        "smote": ("smotenc_antes_despues.csv", result["smote_counts"]),
        "classification_oof": (
            "clasificacion_oof_2019_2023.csv",
            result["classification_oof"],
        ),
        "classification_summary": (
            "clasificacion_resumen_modelos.csv",
            result["classification_summary"],
        ),
        "classification_annual": (
            "clasificacion_metricas_anuales.csv",
            result["classification_annual"],
        ),
        "classification_holdout": (
            "clasificacion_holdout_2024.csv",
            result["classification_holdout"],
        ),
        "regression_oof": ("regresion_oof_2019_2023.csv", result["regression_oof"]),
        "regression_summary": (
            "regresion_resumen_modelos.csv",
            result["regression_summary"],
        ),
        "regression_annual": (
            "regresion_metricas_anuales.csv",
            result["regression_annual"],
        ),
        "regression_holdout": (
            "regresion_holdout_2024.csv",
            result["regression_holdout"],
        ),
    }
    for key, (filename, table) in tables.items():
        path = output_dir / filename
        table.to_csv(path, index=False)
        paths[key] = path

    models = {
        "classification_evaluation_model": (
            "classifier_evaluation.joblib",
            result["classification_evaluation_model"],
        ),
        "classification_operational_model": (
            "classifier_pipeline.joblib",
            result["classification_operational_model"],
        ),
        "calibrator": ("probability_calibrator.joblib", result["calibrator"]),
        "regression_evaluation_model": (
            "regressor_evaluation.joblib",
            result["regression_evaluation_model"],
        ),
        "regression_operational_model": (
            "regressor_pipeline.joblib",
            result["regression_operational_model"],
        ),
    }
    for key, (filename, model) in models.items():
        path = output_dir / filename
        joblib.dump(model, path)
        paths[key] = path

    card_path = output_dir / "MODEL_CARD.md"
    card_path.write_text(_model_card(result["manifest"]), encoding="utf-8")
    paths["model_card"] = card_path
    return paths


def _fit_operational_classifier(df: pd.DataFrame, winner: str) -> BaseEstimator:
    eligible = df[df[TARGET].notna() & df["rendimiento_lag_1"].notna()].copy()
    model = build_classifier(winner)
    model.fit(
        eligible[CATEGORICAL_FEATURES + NUMERIC_FEATURES],
        eligible[CLASS_TARGET].astype(int),
    )
    return model


def _fit_operational_regressor(df: pd.DataFrame, winner: str) -> BaseEstimator:
    eligible = _regression_eligible(df)
    model = build_regressor(winner)
    model.fit(
        eligible[CATEGORICAL_FEATURES + NUMERIC_FEATURES],
        eligible[TARGET].astype(float),
    )
    return model


def _balance_table(df: pd.DataFrame) -> pd.DataFrame:
    balance = (
        df.groupby(["anio", CLASS_TARGET], as_index=False)
        .size()
        .rename(columns={"size": "rows", CLASS_TARGET: "class_label"})
    )
    totals = balance.groupby("anio")["rows"].transform("sum")
    balance["proportion"] = balance["rows"] / totals
    balance["class_name"] = balance["class_label"].map(
        {0: "Rendimiento bajo", 1: "Rendimiento alto"}
    )
    return balance


def _publish_completed_files(temp_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in temp_dir.iterdir():
        destination = output_dir / source.name
        os.replace(source, destination)


def run_production_pipeline(
    data_path: Path = DATA_PATH, output_dir: Path = OUTPUT_DIR
) -> dict[str, Any]:
    """Run the frozen temporal evaluation and publish artifacts after success."""
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    raw_data = load_dataset(data_path)
    audit = audit_dataset(raw_data)
    data, high_yield_threshold = add_high_yield_target(raw_data)
    balance = _balance_table(data)

    history = temporal_splits(data)["history"]
    smote_counts = smotenc_class_counts(history)

    classification_oof = classification_backtest(data, high_yield_threshold)
    classification_annual, classification_summary = classification_metric_tables(
        classification_oof
    )
    classification_winner = select_classifier(classification_summary)
    winner_oof = classification_oof[
        classification_oof["model"] == classification_winner
    ].copy()
    calibrator = fit_calibrator(winner_oof)
    decision_threshold = choose_threshold(winner_oof, calibrator)
    classification_holdout, classification_evaluation_model = (
        evaluate_classification_holdout(
            data, classification_winner, calibrator, decision_threshold
        )
    )
    class_holdout_metrics = classification_metrics(
        classification_holdout["y_true"].to_numpy(),
        classification_holdout["y_pred"].to_numpy(),
        classification_holdout["probability_calibrated"].to_numpy(),
    )

    training_prevalence = float(
        data.loc[
            (data["anio"] < HOLDOUT_YEAR) & data["rendimiento_lag_1"].notna(),
            CLASS_TARGET,
        ].mean()
    )
    prevalence_brier = float(
        brier_score_loss(
            classification_holdout["y_true"],
            np.full(len(classification_holdout), training_prevalence),
        )
    )
    class_holdout_metrics["prevalence_brier"] = prevalence_brier
    class_holdout_metrics["training_prevalence"] = training_prevalence

    regression_oof = regression_backtest(data)
    regression_annual, regression_summary = regression_metric_tables(regression_oof)
    regression_winner = select_regressor(regression_summary)
    regression_holdout, regression_evaluation_model = evaluate_regression_holdout(
        data, regression_winner
    )
    regression_holdout_metrics = regression_metrics(
        regression_holdout["y_true"].to_numpy(),
        regression_holdout["y_pred"].to_numpy(),
    )
    holdout_rows = data.loc[regression_holdout["row_id"].to_numpy()]
    lag1_holdout_metrics = regression_metrics(
        regression_holdout["y_true"].to_numpy(),
        holdout_rows["rendimiento_lag_1"].to_numpy(),
    )
    regression_holdout_metrics["lag1_mae"] = lag1_holdout_metrics["mae"]
    regression_holdout_metrics["mae_improvement_vs_lag1"] = float(
        (lag1_holdout_metrics["mae"] - regression_holdout_metrics["mae"])
        / lag1_holdout_metrics["mae"]
    )

    class_selection_row = classification_summary.loc[
        classification_summary["model"] == classification_winner
    ].iloc[0]
    lag1_class_row = classification_summary.loc[
        classification_summary["model"] == "lag1"
    ].iloc[0]
    regression_selection_row = regression_summary.loc[
        regression_summary["model"] == regression_winner
    ].iloc[0]
    winner_annual_mcc = classification_annual.loc[
        classification_annual["model"] == classification_winner, "mcc"
    ].tolist()
    gate = production_gate(
        integrity_ok=True,
        challenger_mean_mcc=float(class_selection_row["mean_mcc"]),
        lag1_mean_mcc=float(lag1_class_row["mean_mcc"]),
        challenger_annual_mcc=[float(value) for value in winner_annual_mcc],
        holdout_mcc=float(class_holdout_metrics["mcc"]),
        holdout_balanced_accuracy=float(
            class_holdout_metrics["balanced_accuracy"]
        ),
        regression_mae=float(regression_holdout_metrics["mae"]),
        lag1_regression_mae=float(lag1_holdout_metrics["mae"]),
        calibrated_brier=float(class_holdout_metrics["brier"]),
        prevalence_brier=prevalence_brier,
    )

    holdout_feature_rows = data.loc[classification_holdout["row_id"].to_numpy()]
    feature_importance = permutation_feature_importance(
        classification_evaluation_model,
        holdout_feature_rows,
        classification_holdout["y_true"].to_numpy(),
    )
    classification_operational_model = _fit_operational_classifier(
        data, classification_winner
    )
    regression_operational_model = _fit_operational_regressor(data, regression_winner)

    dataset_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest = build_manifest(
        audit=audit,
        dataset_sha256=dataset_sha256,
        high_yield_threshold=high_yield_threshold,
        decision_threshold=decision_threshold,
        classification_winner=classification_winner,
        classification_selection=class_selection_row.to_dict(),
        classification_holdout=class_holdout_metrics,
        regression_winner=regression_winner,
        regression_selection=regression_selection_row.to_dict(),
        regression_holdout=regression_holdout_metrics,
        gate=gate,
    )

    result: dict[str, Any] = {
        "data": data,
        "audit": audit,
        "balance": balance,
        "smote_counts": smote_counts,
        "classification_oof": classification_oof,
        "classification_summary": classification_summary,
        "classification_annual": classification_annual,
        "classification_holdout": classification_holdout,
        "classification_evaluation_model": classification_evaluation_model,
        "classification_operational_model": classification_operational_model,
        "calibrator": calibrator,
        "regression_oof": regression_oof,
        "regression_summary": regression_summary,
        "regression_annual": regression_annual,
        "regression_holdout": regression_holdout,
        "regression_evaluation_model": regression_evaluation_model,
        "regression_operational_model": regression_operational_model,
        "feature_importance": feature_importance,
        "manifest": manifest,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=".modelo_produccion_cafe-", dir=output_dir.parent)
    )
    try:
        plot_paths = create_all_plots(result, temp_dir)
        artifact_paths = export_artifacts(result, temp_dir)
        feature_path = temp_dir / "importancia_variables.csv"
        feature_importance.to_csv(feature_path, index=False)
        artifact_paths["feature_importance"] = feature_path
        _publish_completed_files(temp_dir, output_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    result["plot_paths"] = [output_dir / path.name for path in plot_paths]
    result["artifact_paths"] = {
        key: output_dir / path.name for key, path in artifact_paths.items()
    }
    return result
