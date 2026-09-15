"""Validación temporal reproducible de los modelos de café.

El objetivo original del proyecto es regresión (rendimiento_t_ha). Para poder
calcular matriz de confusión, precisión y MCC se añade una vista secundaria de
clasificación: rendimiento alto = rendimiento superior a la mediana histórica
2007-2018. El umbral se fija antes de observar 2019-2024.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "reports" / "tablas_entrenamiento" / "dataset_cafe_ml_ready.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "validacion_modelo"
SELECTION_YEARS = [2019, 2020, 2021, 2022, 2023]
HOLDOUT_YEAR = 2024
TARGET = "rendimiento_t_ha"
GROUP_KEYS = ["departamento", "municipio", "cultivo"]
CATEGORICAL_FEATURES = ["departamento", "municipio"]

STRICT_NUMERIC_FEATURES = [
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

EXTERNAL_NUMERIC_FEATURES = [
    "suelo_disponible",
    "tiene_credito",
    "credito_total",
    "colocacion_total",
    "num_operaciones_credito",
    "credito_promedio_operacion",
    "log_credito_total",
    "ph_media",
    "materia_organica_pct_media",
    "fosforo_ppm_media",
    "calcio_meq_media",
    "magnesio_meq_media",
    "potasio_meq_media",
    "salinidad_ds_m_media",
    "num_muestras_suelo",
    "ph_variabilidad",
    "materia_organica_pct_variabilidad",
    "fosforo_ppm_variabilidad",
    "calcio_meq_variabilidad",
    "magnesio_meq_variabilidad",
    "potasio_meq_variabilidad",
    "salinidad_ds_m_variabilidad",
    "tendencia_ph",
    "tendencia_materia_organica_pct",
    "tendencia_fosforo_ppm",
    "tendencia_calcio_meq",
    "tendencia_magnesio_meq",
    "tendencia_potasio_meq",
    "tendencia_salinidad_ds_m",
]

LEAKAGE_COLUMNS = [
    "score_confiabilidad",  # usa variabilidad/outliers del target de toda la serie
    "dato_copiado",  # compara directamente target(t) con target(t-1)
]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    feature_set: str = "strict"


CLASSIFICATION_CANDIDATES = [
    Candidate("Mayoría del entrenamiento", "majority"),
    Candidate("Regla rendimiento t-1", "lag1"),
    Candidate("Regresión logística balanceada", "logistic"),
    Candidate("Random Forest balanceado", "random_forest"),
    Candidate("CatBoost balanceado", "catboost"),
    Candidate("CatBoost + externas (exploratorio)", "catboost", "extended"),
]

REGRESSION_CANDIDATES = [
    Candidate("Baseline rendimiento t-1", "lag1"),
    Candidate("Baseline media móvil 3 años", "mean3"),
    Candidate("Elastic Net", "elastic_net"),
    Candidate("Random Forest Regressor", "random_forest"),
    Candidate("CatBoost Regressor", "catboost"),
    Candidate("CatBoost Regressor + externas (exploratorio)", "catboost", "extended"),
]


def _existing(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in df.columns]


def load_data(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {*GROUP_KEYS, "anio", TARGET, "rendimiento_lag_1", "media_rendimiento_3y"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    df["anio"] = pd.to_numeric(df["anio"], errors="raise").astype(int)
    return df.sort_values([*GROUP_KEYS, "anio"]).reset_index(drop=True)


def high_yield_threshold(df: pd.DataFrame) -> float:
    historical = df.loc[df["anio"] <= 2018, TARGET].dropna()
    if historical.empty:
        raise ValueError("No hay datos históricos hasta 2018 para fijar el umbral")
    return float(historical.median())


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def _sklearn_preprocessor(
    df: pd.DataFrame, numeric_features: list[str], scale: bool
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ]
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), _existing(numeric_features, df)),
            ("categorical", Pipeline(categorical_steps), _existing(CATEGORICAL_FEATURES, df)),
        ],
        remainder="drop",
    )


def _catboost_frame(df: pd.DataFrame, numeric_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    categorical = _existing(CATEGORICAL_FEATURES, df)
    numeric = _existing(numeric_features, df)
    frame = df[[*categorical, *numeric]].copy()
    for column in categorical:
        frame[column] = frame[column].fillna("__MISSING__").astype(str)
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, categorical


def _numeric_features(candidate: Candidate, df: pd.DataFrame) -> list[str]:
    features = STRICT_NUMERIC_FEATURES.copy()
    if candidate.feature_set == "extended":
        features.extend(EXTERNAL_NUMERIC_FEATURES)
    return _existing(features, df)


def predict_classifier(
    candidate: Candidate,
    train: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float,
) -> np.ndarray:
    y_train = (train[TARGET].to_numpy() > threshold).astype(int)

    if candidate.family == "majority":
        counts = np.bincount(y_train, minlength=2)
        majority = int(np.argmax(counts))
        return np.full(len(test), majority, dtype=int)
    if candidate.family == "lag1":
        return (test["rendimiento_lag_1"].to_numpy() > threshold).astype(int)

    numeric = _numeric_features(candidate, train)
    if candidate.family == "logistic":
        model = Pipeline(
            [
                ("preprocess", _sklearn_preprocessor(train, numeric, scale=True)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(train, y_train)
        return model.predict(test).astype(int)
    if candidate.family == "random_forest":
        model = Pipeline(
            [
                ("preprocess", _sklearn_preprocessor(train, numeric, scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=14,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(train, y_train)
        return model.predict(test).astype(int)
    if candidate.family == "catboost":
        x_train, categorical = _catboost_frame(train, numeric)
        x_test, _ = _catboost_frame(test, numeric)
        model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(x_train, y_train, cat_features=categorical)
        return model.predict(x_test).astype(int).reshape(-1)
    raise ValueError(f"Familia de clasificación desconocida: {candidate.family}")


def predict_regressor(
    candidate: Candidate, train: pd.DataFrame, test: pd.DataFrame
) -> np.ndarray:
    if candidate.family == "lag1":
        return test["rendimiento_lag_1"].to_numpy(dtype=float)
    if candidate.family == "mean3":
        return test["media_rendimiento_3y"].to_numpy(dtype=float)

    numeric = _numeric_features(candidate, train)
    y_train = train[TARGET].to_numpy(dtype=float)
    if candidate.family == "elastic_net":
        model = Pipeline(
            [
                ("preprocess", _sklearn_preprocessor(train, numeric, scale=True)),
                (
                    "model",
                    ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=10000, random_state=42),
                ),
            ]
        )
        model.fit(train, y_train)
        return np.clip(model.predict(test), 0, None)
    if candidate.family == "random_forest":
        model = Pipeline(
            [
                ("preprocess", _sklearn_preprocessor(train, numeric, scale=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=14,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(train, y_train)
        return np.clip(model.predict(test), 0, None)
    if candidate.family == "catboost":
        x_train, categorical = _catboost_frame(train, numeric)
        x_test, _ = _catboost_frame(test, numeric)
        model = CatBoostRegressor(
            iterations=150,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=5,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(x_train, y_train, cat_features=categorical)
        return np.clip(model.predict(x_test), 0, None)
    raise ValueError(f"Familia de regresión desconocida: {candidate.family}")


def _classification_backtest(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = df[df[TARGET].notna() & df["rendimiento_lag_1"].notna()].copy()
    for year in [*SELECTION_YEARS, HOLDOUT_YEAR]:
        train = eligible[eligible["anio"] < year]
        test = eligible[eligible["anio"] == year]
        y_true = (test[TARGET].to_numpy() > threshold).astype(int)
        for candidate in CLASSIFICATION_CANDIDATES:
            y_pred = predict_classifier(candidate, train, test, threshold)
            for source_index, actual, predicted in zip(test.index, y_true, y_pred):
                rows.append(
                    {
                        "source_index": int(source_index),
                        "anio": year,
                        "modelo": candidate.name,
                        "elegible_como_ganador": candidate.feature_set == "strict",
                        "y_true": int(actual),
                        "y_pred": int(predicted),
                    }
                )
    return pd.DataFrame(rows)


def _regression_backtest(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = df[
        df[TARGET].notna()
        & df["rendimiento_lag_1"].notna()
        & df["media_rendimiento_3y"].notna()
    ].copy()
    for year in [*SELECTION_YEARS, HOLDOUT_YEAR]:
        train = eligible[eligible["anio"] < year]
        test = eligible[eligible["anio"] == year]
        y_true = test[TARGET].to_numpy(dtype=float)
        for candidate in REGRESSION_CANDIDATES:
            y_pred = predict_regressor(candidate, train, test)
            for source_index, actual, predicted in zip(test.index, y_true, y_pred):
                rows.append(
                    {
                        "source_index": int(source_index),
                        "anio": year,
                        "modelo": candidate.name,
                        "elegible_como_ganador": candidate.feature_set == "strict",
                        "y_true": float(actual),
                        "y_pred": float(predicted),
                    }
                )
    return pd.DataFrame(rows)


def summarize_classification(predictions: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    data = predictions[predictions["anio"].isin(years)]
    rows = []
    for model_name, group in data.groupby("modelo", sort=False):
        pooled = classification_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        yearly_mcc = group.groupby("anio").apply(
            lambda part: matthews_corrcoef(part["y_true"], part["y_pred"]),
            include_groups=False,
        )
        rows.append(
            {
                "modelo": model_name,
                "elegible_como_ganador": bool(group["elegible_como_ganador"].iloc[0]),
                "n": len(group),
                "mcc_promedio_anual": yearly_mcc.mean(),
                "mcc_minimo_anual": yearly_mcc.min(),
                **pooled,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["elegible_como_ganador", "mcc_promedio_anual", "balanced_accuracy"],
        ascending=[False, False, False],
    )


def summarize_regression(predictions: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    data = predictions[predictions["anio"].isin(years)]
    rows = []
    for model_name, group in data.groupby("modelo", sort=False):
        pooled = regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        yearly_mae = group.groupby("anio").apply(
            lambda part: mean_absolute_error(part["y_true"], part["y_pred"]),
            include_groups=False,
        )
        rows.append(
            {
                "modelo": model_name,
                "elegible_como_ganador": bool(group["elegible_como_ganador"].iloc[0]),
                "n": len(group),
                "mae_promedio_anual": yearly_mae.mean(),
                "mae_maximo_anual": yearly_mae.max(),
                **pooled,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["elegible_como_ganador", "mae_promedio_anual", "rmse"],
        ascending=[False, True, True],
    )


def build_quality_tables(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_columns = [*GROUP_KEYS, "anio"]
    sorted_df = df.sort_values(key_columns).copy()
    previous_target = sorted_df.groupby(GROUP_KEYS)[TARGET].shift(1)
    previous_year = sorted_df.groupby(GROUP_KEYS)["anio"].shift(1)
    comparable = (
        ((sorted_df["anio"] - previous_year) == 1)
        & sorted_df["rendimiento_lag_1"].notna()
        & previous_target.notna()
    )
    mismatch = (
        sorted_df.loc[comparable, "rendimiento_lag_1"] - previous_target.loc[comparable]
    ).abs() > 1e-12
    municipality_departments = df.groupby("municipio")["departamento"].nunique()
    homonyms = set(municipality_departments[municipality_departments > 1].index)

    quality = pd.DataFrame(
        [
            ("filas", len(df), "informativo"),
            ("columnas", df.shape[1], "informativo"),
            ("duplicados_clave", int(df.duplicated(key_columns).sum()), "alto si > 0"),
            ("target_nulos", int(df[TARGET].isna().sum()), "alto si > 0"),
            ("lag1_comparaciones", int(comparable.sum()), "informativo"),
            ("lag1_inconsistencias", int(mismatch.sum()), "alto si > 0"),
            ("municipios_compuestos", int(df[GROUP_KEYS].drop_duplicates().shape[0]), "informativo"),
            ("nombres_municipio_homonimos", len(homonyms), "riesgo de join"),
            ("filas_afectadas_homonimos", int(df["municipio"].isin(homonyms).sum()), "riesgo de join"),
            ("score_confiabilidad_target_derivado", int("score_confiabilidad" in df), "fuga si se usa"),
            ("dato_copiado_target_derivado", int("dato_copiado" in df), "fuga si se usa"),
        ],
        columns=["chequeo", "valor", "interpretacion"],
    )

    periods: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("entrenamiento_2007_2018", lambda frame: frame["anio"] <= 2018),
        ("seleccion_2019_2023", lambda frame: frame["anio"].isin(SELECTION_YEARS)),
        ("holdout_2024", lambda frame: frame["anio"] == HOLDOUT_YEAR),
    ]
    balance_rows = []
    for name, selector in periods:
        part = df.loc[selector(df) & df["rendimiento_lag_1"].notna()]
        labels = (part[TARGET] > threshold).astype(int)
        counts = labels.value_counts().reindex([0, 1], fill_value=0)
        balance_rows.append(
            {
                "periodo": name,
                "n": len(part),
                "bajo_0": int(counts[0]),
                "alto_1": int(counts[1]),
                "alto_pct": float(counts[1] / len(part)) if len(part) else np.nan,
                "ratio_mayoritaria_minoritaria": float(counts.max() / counts.min())
                if counts.min() > 0
                else np.inf,
            }
        )
    for year, part in df[df["anio"] >= 2019].groupby("anio"):
        part = part[part["rendimiento_lag_1"].notna()]
        labels = (part[TARGET] > threshold).astype(int)
        counts = labels.value_counts().reindex([0, 1], fill_value=0)
        balance_rows.append(
            {
                "periodo": str(year),
                "n": len(part),
                "bajo_0": int(counts[0]),
                "alto_1": int(counts[1]),
                "alto_pct": float(counts[1] / len(part)),
                "ratio_mayoritaria_minoritaria": float(counts.max() / counts.min())
                if counts.min() > 0
                else np.inf,
            }
        )
    return quality, pd.DataFrame(balance_rows)


def save_plots(
    output_dir: Path,
    balance: pd.DataFrame,
    classification_selection: pd.DataFrame,
    confusion: np.ndarray,
    winner: str,
) -> None:
    sns.set_theme(style="whitegrid")
    yearly = balance[balance["periodo"].str.fullmatch(r"\d{4}")].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(yearly["periodo"], yearly["alto_pct"] * 100, color="#377eb8")
    ax.axhline(50, color="#d62728", linestyle="--", linewidth=1.5, label="50%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Clase rendimiento alto (%)")
    ax.set_xlabel("Año")
    ax.set_title("Balance de clases fuera de tiempo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "balance_clases.png", dpi=180)
    plt.close(fig)

    eligible = classification_selection[classification_selection["elegible_como_ganador"]].copy()
    eligible = eligible.sort_values("mcc_promedio_anual", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(eligible["modelo"], eligible["mcc_promedio_anual"], color="#4daf4a")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("MCC promedio anual (2019-2023)")
    ax.set_title("Comparación temporal de modelos de clasificación")
    fig.tight_layout()
    fig.savefig(output_dir / "comparacion_modelos_mcc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicho bajo", "Predicho alto"],
        yticklabels=["Real bajo", "Real alto"],
        ax=ax,
    )
    ax.set_title(f"Matriz de confusión 2024\n{winner}")
    fig.tight_layout()
    fig.savefig(output_dir / "matriz_confusion_mejor_modelo.png", dpi=200)
    plt.close(fig)


def write_report(
    output_dir: Path,
    threshold: float,
    quality: pd.DataFrame,
    balance: pd.DataFrame,
    class_selection: pd.DataFrame,
    class_holdout: pd.DataFrame,
    regression_selection: pd.DataFrame,
    regression_holdout: pd.DataFrame,
    class_winner: str,
    regression_winner: str,
    confusion: np.ndarray,
) -> None:
    class_final = class_holdout.set_index("modelo").loc[class_winner]
    regression_final = regression_holdout.set_index("modelo").loc[regression_winner]
    class_ranking = class_selection[class_selection["elegible_como_ganador"]].reset_index(drop=True)
    class_runner_up = class_ranking.iloc[1]
    class_margin = float(
        class_ranking.iloc[0]["mcc_promedio_anual"]
        - class_runner_up["mcc_promedio_anual"]
    )
    class_holdout_best = class_holdout[class_holdout["elegible_como_ganador"]].iloc[0]
    regression_holdout_best = regression_holdout[
        regression_holdout["elegible_como_ganador"]
    ].iloc[0]
    train_balance = balance.set_index("periodo").loc["entrenamiento_2007_2018"]
    selection_balance = balance.set_index("periodo").loc["seleccion_2019_2023"]
    holdout_balance = balance.set_index("periodo").loc["holdout_2024"]
    report = f"""# Validación temporal del modelo de café

## Resumen ejecutivo

- El objetivo nativo es regresión (`rendimiento_t_ha`). La matriz de confusión se calcula para `rendimiento alto`, definido como rendimiento > **{threshold:.6f} t/ha**, la mediana histórica 2007-2018 fijada sin mirar 2019-2024.
- Mejor clasificador elegible en 2019-2023: **{class_winner}**. En el holdout 2024 obtuvo exactitud **{class_final['accuracy']:.3f}**, precisión **{class_final['precision']:.3f}**, balanced accuracy **{class_final['balanced_accuracy']:.3f}** y MCC **{class_final['mcc']:.3f}**.
- La ventaja de selección sobre **{class_runner_up['modelo']}** fue solo **{class_margin:.3f} MCC promedio anual**: es un empate práctico, no evidencia de superioridad concluyente. En 2024, el MCC observado más alto fue el de **{class_holdout_best['modelo']}** ({class_holdout_best['mcc']:.3f}), pero ese año no se usó para cambiar al ganador.
- Su matriz 2024 (filas reales, columnas predichas; orden bajo/alto) fue: `[[{confusion[0,0]}, {confusion[0,1]}], [{confusion[1,0]}, {confusion[1,1]}]]`.
- Mejor regresor elegible en 2019-2023: **{regression_winner}**. En 2024 obtuvo MAE **{regression_final['mae']:.3f} t/ha**, RMSE **{regression_final['rmse']:.3f} t/ha** y R² **{regression_final['r2']:.3f}**.
- En el holdout 2024, el menor MAE observado fue el de **{regression_holdout_best['modelo']}** ({regression_holdout_best['mae']:.3f} t/ha), otra señal de que el ranking de modelos no es estable entre años.

## Balance de clases

- Entrenamiento histórico utilizable: {int(train_balance['bajo_0'])} bajos y {int(train_balance['alto_1'])} altos ({train_balance['alto_pct']:.1%} altos).
- Selección 2019-2023: {int(selection_balance['bajo_0'])} bajos y {int(selection_balance['alto_1'])} altos ({selection_balance['alto_pct']:.1%} altos).
- Holdout 2024: {int(holdout_balance['bajo_0'])} bajos y {int(holdout_balance['alto_1'])} altos ({holdout_balance['alto_pct']:.1%} altos).
- No se sobremuestreó la validación. Logistic Regression, Random Forest y CatBoost usaron pesos de clase calculados solo en entrenamiento. MCC fue la métrica primaria.

## Riesgos metodológicos encontrados

1. `score_confiabilidad` se calcula con la variabilidad y los outliers del target de toda la serie; usarlo como predictor o peso filtra información futura.
2. `dato_copiado` compara directamente el target del año con `lag_1`; tampoco puede usarse como feature ex ante.
3. Suelos y crédito se unen solo por `municipio`, aunque existen nombres homónimos entre departamentos. Las variantes con variables externas se reportan como exploratorias y no pueden ganar la selección.
4. La procedencia temporal de las muestras de suelo no está verificada. Por eso el ganador recomendado usa únicamente geografía conocida y variables históricas rezagadas.
5. La fuente EVA cambia de `historica` a `reciente` en 2019 y la proporción de clase alta cambia mucho por año; esto es drift y explica parte de la inestabilidad.
6. El script original elige el alpha de blending mirando 2019-2024 y después informa rendimiento sobre los mismos años; esa cifra es optimista para selección. Aquí 2024 quedó reservado.

## Interpretación

La clase significa **rendimiento alto**, no rentabilidad ni prosperidad financiera. Para llamar al resultado “negocio próspero” faltan costos, margen neto y un umbral económico validado.

## Estado

**Compartible con cautelas.** La evaluación temporal y las métricas son reproducibles, pero el drift de fuente y la trazabilidad incompleta de variables externas impiden afirmar que el modelo esté listo para decisiones de inversión.
"""
    (output_dir / "INFORME_VALIDACION.md").write_text(report, encoding="utf-8")


def run_validation(
    data_path: Path = DEFAULT_DATA,
    output_dir: Path = DEFAULT_OUTPUT,
    force: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "resumen_validacion.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    df = load_data(data_path)
    threshold = high_yield_threshold(df)
    quality, balance = build_quality_tables(df, threshold)
    classification_predictions = _classification_backtest(df, threshold)
    regression_predictions = _regression_backtest(df)

    class_selection = summarize_classification(classification_predictions, SELECTION_YEARS)
    class_holdout = summarize_classification(classification_predictions, [HOLDOUT_YEAR])
    regression_selection = summarize_regression(regression_predictions, SELECTION_YEARS)
    regression_holdout = summarize_regression(regression_predictions, [HOLDOUT_YEAR])

    eligible_class = class_selection[class_selection["elegible_como_ganador"]]
    class_winner = str(eligible_class.iloc[0]["modelo"])
    eligible_regression = regression_selection[regression_selection["elegible_como_ganador"]]
    regression_winner = str(eligible_regression.iloc[0]["modelo"])

    winner_2024 = classification_predictions[
        (classification_predictions["modelo"] == class_winner)
        & (classification_predictions["anio"] == HOLDOUT_YEAR)
    ]
    matrix = confusion_matrix(winner_2024["y_true"], winner_2024["y_pred"], labels=[0, 1])

    quality.to_csv(output_dir / "calidad_datos.csv", index=False)
    balance.to_csv(output_dir / "balance_clases.csv", index=False)
    classification_predictions.to_csv(output_dir / "predicciones_clasificacion.csv", index=False)
    regression_predictions.to_csv(output_dir / "predicciones_regresion.csv", index=False)
    class_selection.to_csv(output_dir / "comparacion_clasificacion_seleccion.csv", index=False)
    class_holdout.to_csv(output_dir / "comparacion_clasificacion_holdout_2024.csv", index=False)
    regression_selection.to_csv(output_dir / "comparacion_regresion_seleccion.csv", index=False)
    regression_holdout.to_csv(output_dir / "comparacion_regresion_holdout_2024.csv", index=False)
    pd.DataFrame(matrix, index=["real_bajo", "real_alto"], columns=["pred_bajo", "pred_alto"]).to_csv(
        output_dir / "matriz_confusion_mejor_modelo.csv"
    )

    save_plots(output_dir, balance, class_selection, matrix, class_winner)
    write_report(
        output_dir,
        threshold,
        quality,
        balance,
        class_selection,
        class_holdout,
        regression_selection,
        regression_holdout,
        class_winner,
        regression_winner,
        matrix,
    )

    class_final = class_holdout.set_index("modelo").loc[class_winner]
    regression_final = regression_holdout.set_index("modelo").loc[regression_winner]
    summary = {
        "data_path": str(data_path.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "threshold_high_yield_t_ha": threshold,
        "selection_years": SELECTION_YEARS,
        "holdout_year": HOLDOUT_YEAR,
        "classification_winner": class_winner,
        "classification_holdout_2024": {
            key: float(class_final[key])
            for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc"]
        },
        "confusion_matrix_2024": matrix.tolist(),
        "regression_winner": regression_winner,
        "regression_holdout_2024": {
            key: float(regression_final[key]) for key in ["mae", "rmse", "r2"]
        },
        "status": "Compartible con cautelas",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_validation()
    print(json.dumps(result, indent=2, ensure_ascii=False))
