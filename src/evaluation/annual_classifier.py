"""Leakage-safe annual classification, WEKA metrics, and strict acceptance gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from src.evaluation.presentation_model import weka_report
from src.features.cafe_annual_features import CLASS_TARGET, TARGET


RANDOM_STATE = 42
CATEGORICAL_FEATURES = ["codigo_dane_departamento", "codigo_dane_municipio"]
NUMERIC_FEATURES = [
    "anio",
    "source_break_2019",
    "history_years",
    "rendimiento_lag_1",
    "rendimiento_lag_2",
    "rendimiento_lag_3",
    "produccion_t_lag_1",
    "produccion_t_lag_2",
    "produccion_t_lag_3",
    "area_cosechada_ha_lag_1",
    "area_cosechada_ha_lag_2",
    "area_cosechada_ha_lag_3",
    "area_sembrada_ha_lag_1",
    "area_sembrada_ha_lag_2",
    "area_sembrada_ha_lag_3",
    "rendimiento_municipio_historico",
    "proporcion_alto_historica",
    "rendimiento_3y_conteo",
    "rendimiento_media_3y",
    "rendimiento_mediana_3y",
    "rendimiento_std_3y",
    "rendimiento_min_3y",
    "rendimiento_max_3y",
    "rendimiento_pendiente_3y",
    "rendimiento_5y_conteo",
    "rendimiento_media_5y",
    "rendimiento_mediana_5y",
    "rendimiento_std_5y",
    "rendimiento_min_5y",
    "rendimiento_max_5y",
    "rendimiento_pendiente_5y",
    "rendimiento_departamento_historico",
    "rendimiento_nacional_historico",
    "latitud",
    "longitud",
    "precipitacion_total_mm_lag_1",
    "temperatura_media_c_lag_1",
    "temperatura_min_c_lag_1",
    "temperatura_max_c_lag_1",
    "precipitacion_anomalia_lag_1",
    "temperatura_anomalia_lag_1",
    "distancia_grilla_grados",
    "climate_available",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
DEFAULT_MODELS = (
    "logistic_plain",
    "logistic_weighted",
    "logistic_smotenc",
    "hist_plain",
    "hist_smotenc",
    "random_forest",
    "catboost_plain",
    "catboost_weighted",
)
MODEL_COMPLEXITY = {
    "majority": 0,
    "lag1": 1,
    "logistic_plain": 2,
    "logistic_weighted": 3,
    "logistic_smotenc": 4,
    "hist_plain": 5,
    "hist_smotenc": 6,
    "random_forest": 7,
    "catboost_plain": 8,
    "catboost_weighted": 9,
}


@dataclass(frozen=True)
class ModelRecipe:
    name: str
    decision_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnnualCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """Small CatBoost wrapper that normalizes categorical values consistently."""

    def __init__(self, weighted: bool = False):
        self.weighted = weighted

    @staticmethod
    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        for column in CATEGORICAL_FEATURES:
            data[column] = data[column].fillna("DESCONOCIDO").astype(str)
        for column in NUMERIC_FEATURES:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        return data

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "AnnualCatBoostClassifier":
        y = np.asarray(target, dtype=int)
        parameters: dict[str, Any] = {
            "iterations": 220,
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5,
            "loss_function": "Logloss",
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "thread_count": -1,
            "allow_writing_files": False,
        }
        if self.weighted:
            counts = np.bincount(y, minlength=2)
            parameters["class_weights"] = [
                len(y) / (2 * count) if count else 1.0 for count in counts
            ]
        self.model_ = CatBoostClassifier(**parameters)
        self.model_.fit(
            self._prepare(features),
            y,
            cat_features=CATEGORICAL_FEATURES,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._prepare(features)), dtype=float)


class LagOneClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> "LagOneClassifier":
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probability = pd.to_numeric(features["clase_lag_1"], errors="raise").to_numpy(float)
        return np.column_stack([1 - probability, probability])


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "MajorityClassifier":
        self.probability_ = float(np.asarray(target, dtype=int).mean())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probability = np.full(len(features), self.probability_, dtype=float)
        return np.column_stack([1 - probability, probability])


def _plain_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                SklearnPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                SklearnPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", RobustScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def _ordinal_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                SklearnPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                SklearnPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", RobustScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def _post_smote_onehot() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                [0, 1],
            )
        ],
        remainder="passthrough",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def _logistic(class_weight: str | None = None) -> LogisticRegression:
    return LogisticRegression(
        solver="liblinear",
        C=0.5,
        class_weight=class_weight,
        max_iter=300,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )


def build_estimator(name: str) -> BaseEstimator:
    """Build one of the bounded candidate classifiers."""
    if name == "logistic_plain":
        return Pipeline([("preprocess", _plain_preprocessor()), ("model", _logistic())])
    if name == "logistic_weighted":
        return Pipeline(
            [("preprocess", _plain_preprocessor()), ("model", _logistic("balanced"))]
        )
    if name == "logistic_smotenc":
        return Pipeline(
            [
                ("preprocess", _ordinal_preprocessor()),
                (
                    "sampler",
                    SMOTENC(
                        categorical_features=[0, 1],
                        random_state=RANDOM_STATE,
                        k_neighbors=3,
                    ),
                ),
                ("onehot", _post_smote_onehot()),
                ("model", _logistic()),
            ]
        )
    if name in {"hist_plain", "hist_smotenc"}:
        steps: list[tuple[str, Any]] = [("preprocess", _ordinal_preprocessor())]
        if name == "hist_smotenc":
            steps.append(
                (
                    "sampler",
                    SMOTENC(
                        categorical_features=[0, 1],
                        random_state=RANDOM_STATE,
                        k_neighbors=3,
                    ),
                )
            )
        steps.append(
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=180,
                    max_leaf_nodes=15,
                    min_samples_leaf=25,
                    l2_regularization=2.0,
                    random_state=RANDOM_STATE,
                ),
            )
        )
        return Pipeline(steps)
    if name == "random_forest":
        return Pipeline(
            [
                ("preprocess", _ordinal_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=10,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if name == "catboost_plain":
        return AnnualCatBoostClassifier(weighted=False)
    if name == "catboost_weighted":
        return AnnualCatBoostClassifier(weighted=True)
    raise ValueError(f"Modelo anual desconocido: {name}")


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*MODEL_FEATURES, CLASS_TARGET, TARGET, "clase_lag_1", "anio"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Faltan columnas para clasificación anual: {missing}")
    return frame[
        frame[TARGET].notna() & frame["rendimiento_lag_1"].notna()
    ].copy()


def _fit_predict(
    name: str, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, BaseEstimator]:
    y_train = train[CLASS_TARGET].astype(int)
    if set(y_train.unique()) != {0, 1}:
        raise ValueError(f"El entrenamiento de {name} no contiene ambas clases")
    if name == "majority":
        estimator: BaseEstimator = MajorityClassifier().fit(train, y_train)
        probability = estimator.predict_proba(test)[:, 1]
    elif name == "lag1":
        estimator = LagOneClassifier().fit(train, y_train)
        probability = estimator.predict_proba(test)[:, 1]
    else:
        estimator = build_estimator(name)
        estimator.fit(train[MODEL_FEATURES], y_train)
        probability = estimator.predict_proba(test[MODEL_FEATURES])[:, 1]
    return np.asarray(probability, dtype=float), estimator


def _safe_metric(value: float) -> float:
    return float(value) if np.isfinite(value) else -1.0


def select_threshold_from_prior_years(
    predictions: pd.DataFrame, evaluation_year: int
) -> float:
    """Choose a threshold by worst annual MCC/Kappa using only prior labels."""
    required = {"anio", "y_true", "probability"}
    if missing := sorted(required.difference(predictions.columns)):
        raise ValueError(f"Faltan columnas para elegir umbral: {missing}")
    if (predictions["anio"] >= int(evaluation_year)).any():
        raise ValueError("El umbral contiene etiquetas del año evaluado o posteriores")
    if predictions.empty:
        return 0.5
    best_threshold = 0.5
    best_score: tuple[float, float, float, float] | None = None
    for threshold in np.linspace(0.05, 0.95, 91):
        annual_joint: list[float] = []
        predicted_all = (predictions["probability"].to_numpy(float) >= threshold).astype(int)
        for _, group in predictions.assign(_pred=predicted_all).groupby("anio"):
            actual = group["y_true"].to_numpy(int)
            predicted = group["_pred"].to_numpy(int)
            mcc = _safe_metric(matthews_corrcoef(actual, predicted))
            kappa = _safe_metric(cohen_kappa_score(actual, predicted))
            annual_joint.append(min(mcc, kappa))
        weighted_precision = precision_score(
            predictions["y_true"].to_numpy(int),
            predicted_all,
            average="weighted",
            zero_division=0,
        )
        score = (
            min(annual_joint),
            float(np.mean(annual_joint)),
            float(weighted_precision),
            -abs(float(threshold) - 0.5),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _prediction_frame(
    test: pd.DataFrame,
    name: str,
    probability: np.ndarray,
    decision_threshold: float,
    train_max_year: int,
    threshold_source_max_year: int,
) -> pd.DataFrame:
    identity = [
        column
        for column in (
            "codigo_dane_municipio",
            "codigo_dane_departamento",
            "departamento",
            "municipio",
            "anio",
            "clase_lag_1",
            "cambio_clase_vs_lag_1",
        )
        if column in test.columns
    ]
    result = test[identity].reset_index(drop=False).rename(columns={"index": "row_id"})
    result["y_true"] = test[CLASS_TARGET].to_numpy(int)
    result["probability"] = probability
    result["y_pred"] = (probability >= decision_threshold).astype(int)
    result["model"] = name
    result["decision_threshold"] = float(decision_threshold)
    result["train_max_year"] = int(train_max_year)
    result["threshold_source_max_year"] = int(threshold_source_max_year)
    return result


def annual_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute annual summary, confusion counts, Kappa, and WEKA details."""
    required = {"model", "anio", "y_true", "y_pred", "probability"}
    if missing := sorted(required.difference(predictions.columns)):
        raise ValueError(f"Faltan columnas de predicción: {missing}")
    summaries: list[dict[str, Any]] = []
    detailed_frames: list[pd.DataFrame] = []
    for (model, year), group in predictions.groupby(["model", "anio"], sort=True):
        actual = group["y_true"].to_numpy(int)
        predicted = group["y_pred"].to_numpy(int)
        probability = group["probability"].to_numpy(float)
        matrix = confusion_matrix(actual, predicted, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        report = weka_report(actual, predicted, probability)
        detail = report["detailed_accuracy"].copy()
        detail.insert(0, "anio", int(year))
        detail.insert(0, "model", model)
        detailed_frames.append(detail)
        summaries.append(
            {
                "model": model,
                "anio": int(year),
                "instances": int(len(group)),
                "support_low": int((actual == 0).sum()),
                "support_high": int((actual == 1).sum()),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "accuracy": float(accuracy_score(actual, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
                "weighted_precision": float(report["summary"]["weighted_precision"]),
                "mcc": float(matthews_corrcoef(actual, predicted)),
                "kappa": float(cohen_kappa_score(actual, predicted)),
            }
        )
    return pd.DataFrame(summaries), pd.concat(detailed_frames, ignore_index=True)


def _model_summary(annual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in annual.groupby("model"):
        joint = group[["mcc", "kappa"]].min(axis=1)
        rows.append(
            {
                "model": model,
                "worst_joint_score": float(joint.min()),
                "mean_joint_score": float(joint.mean()),
                "mean_mcc": float(group["mcc"].mean()),
                "mean_kappa": float(group["kappa"].mean()),
                "mean_weighted_precision": float(group["weighted_precision"].mean()),
                "complexity": MODEL_COMPLEXITY.get(model, 99),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "worst_joint_score",
            "mean_joint_score",
            "mean_weighted_precision",
            "complexity",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def walk_forward_backtest(
    frame: pd.DataFrame,
    model_names: Sequence[str] = DEFAULT_MODELS,
    years: Sequence[int] = tuple(range(2019, 2025)),
    threshold_history_years: Sequence[int] = (2016, 2017, 2018),
) -> dict[str, Any]:
    """Evaluate baselines and candidates in strict chronological order."""
    data = _eligible(frame)
    years = sorted(int(year) for year in years)
    names = ["majority", "lag1", *model_names]
    all_predictions: list[pd.DataFrame] = []
    histories: dict[str, pd.DataFrame] = {}
    for name in names:
        if name in {"majority", "lag1"}:
            histories[name] = pd.DataFrame()
            continue
        pieces: list[pd.DataFrame] = []
        for year in sorted(int(value) for value in threshold_history_years):
            train = data[data["anio"] < year]
            test = data[data["anio"] == year]
            if train.empty or test.empty:
                continue
            probability, _ = _fit_predict(name, train, test)
            pieces.append(
                pd.DataFrame(
                    {
                        "anio": year,
                        "y_true": test[CLASS_TARGET].to_numpy(int),
                        "probability": probability,
                    }
                )
            )
        histories[name] = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    for year in years:
        train = data[data["anio"] < year]
        test = data[data["anio"] == year]
        if train.empty or test.empty:
            raise ValueError(f"El año {year} no tiene entrenamiento previo o evaluación")
        train_max_year = int(train["anio"].max())
        if train_max_year >= year:
            raise AssertionError("El backtest incluyó el año evaluado en entrenamiento")
        for name in names:
            history = histories[name]
            if name in {"majority", "lag1"}:
                decision_threshold = 0.5
                threshold_source_max = train_max_year
            else:
                decision_threshold = select_threshold_from_prior_years(history, year)
                threshold_source_max = int(history["anio"].max()) if not history.empty else train_max_year
            probability, _ = _fit_predict(name, train, test)
            prediction = _prediction_frame(
                test,
                name,
                probability,
                decision_threshold,
                train_max_year,
                threshold_source_max,
            )
            all_predictions.append(prediction)
            if name not in {"majority", "lag1"}:
                histories[name] = pd.concat(
                    [
                        history,
                        pd.DataFrame(
                            {
                                "anio": year,
                                "y_true": test[CLASS_TARGET].to_numpy(int),
                                "probability": probability,
                            }
                        ),
                    ],
                    ignore_index=True,
                )
    predictions = pd.concat(all_predictions, ignore_index=True)
    annual, detailed = annual_metrics(predictions)
    summary = _model_summary(annual)
    winner = str(summary.iloc[0]["model"])
    if winner in {"majority", "lag1"}:
        final_threshold = 0.5
    else:
        final_threshold = select_threshold_from_prior_years(histories[winner], max(years) + 1)
    return {
        "predictions": predictions,
        "annual_metrics": annual,
        "weka_details": detailed,
        "model_summary": summary,
        "winner": winner,
        "recipe": ModelRecipe(winner, final_threshold),
        "threshold_histories": histories,
    }


def evaluate_final_holdout(
    frame: pd.DataFrame, recipe: ModelRecipe, year: int = 2025
) -> tuple[pd.DataFrame, BaseEstimator]:
    """Fit the frozen recipe on prior years and evaluate one final year."""
    data = _eligible(frame)
    train = data[data["anio"] < int(year)]
    test = data[data["anio"] == int(year)]
    if train.empty or test.empty:
        raise ValueError(f"El holdout {year} no tiene entrenamiento o casos")
    train_max_year = int(train["anio"].max())
    if train_max_year >= year:
        raise AssertionError("El holdout fue incluido en entrenamiento")
    probability, estimator = _fit_predict(recipe.name, train, test)
    prediction = _prediction_frame(
        test,
        recipe.name,
        probability,
        recipe.decision_threshold,
        train_max_year,
        train_max_year,
    )
    return prediction, estimator


def strict_production_gate(
    metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    threshold: float = 0.85,
    required_years: Sequence[int] = tuple(range(2019, 2026)),
) -> dict[str, Any]:
    """Require both priority metrics in every year plus baseline improvement."""
    required = {"anio", "mcc", "kappa"}
    if missing := sorted(required.difference(metrics.columns)):
        raise ValueError(f"Faltan métricas del modelo: {missing}")
    if missing := sorted(required.difference(baseline_metrics.columns)):
        raise ValueError(f"Faltan métricas del baseline: {missing}")
    required_years = sorted({int(year) for year in required_years})
    evaluated = metrics[metrics["anio"].isin(required_years)].sort_values("anio").copy()
    baseline = baseline_metrics[baseline_metrics["anio"].isin(required_years)].sort_values("anio").copy()
    present_years = set(evaluated["anio"].astype(int))
    baseline_years = set(baseline["anio"].astype(int))
    missing_years = sorted(set(required_years).difference(present_years))
    missing_baseline_years = sorted(set(required_years).difference(baseline_years))
    duplicate_years = sorted(
        evaluated.loc[evaluated["anio"].duplicated(keep=False), "anio"].astype(int).unique()
    )
    evaluated["passes"] = (
        (evaluated["mcc"] >= threshold) & (evaluated["kappa"] >= threshold)
    )
    failed_years = sorted(
        set(evaluated.loc[~evaluated["passes"], "anio"].astype(int)).union(missing_years)
    )
    model_joint = evaluated[["mcc", "kappa"]].min(axis=1)
    baseline_joint = baseline[["mcc", "kappa"]].min(axis=1)
    improvement = (
        float(model_joint.min() - baseline_joint.min())
        if not model_joint.empty and not baseline_joint.empty
        else float("nan")
    )
    failed_checks: list[str] = []
    if missing_years:
        failed_checks.append(f"Faltan años requeridos del modelo: {missing_years}.")
    if missing_baseline_years:
        failed_checks.append(f"Faltan años requeridos del baseline: {missing_baseline_years}.")
    if duplicate_years:
        failed_checks.append(f"Hay más de una fila de métricas en años: {duplicate_years}.")
    if failed_years:
        failed_checks.append(
            f"MCC o Kappa inferior a {threshold:.2f} en años: {failed_years}."
        )
    if not np.isfinite(improvement) or improvement < 0.02:
        failed_checks.append(
            f"La mejora del peor año frente a persistencia es {improvement:.4f}, inferior a 0.02."
        )
    if {"support_low", "support_high"}.issubset(evaluated.columns):
        missing_class = evaluated[
            (evaluated["support_low"] == 0) | (evaluated["support_high"] == 0)
        ]["anio"].astype(int).tolist()
        if missing_class:
            failed_checks.append(f"Falta una clase real en años: {missing_class}.")
    return {
        "status": "production_candidate" if not failed_checks else "target_not_met",
        "threshold": float(threshold),
        "failed_years": failed_years,
        "missing_years": missing_years,
        "missing_baseline_years": missing_baseline_years,
        "minimum_mcc": float(evaluated["mcc"].min()) if not evaluated.empty else float("nan"),
        "minimum_kappa": float(evaluated["kappa"].min()) if not evaluated.empty else float("nan"),
        "worst_year_improvement_over_persistence": improvement,
        "failed_checks": failed_checks,
    }
