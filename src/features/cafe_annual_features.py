"""Point-in-time feature engineering for annual municipal coffee yield."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.cafe_annual_sources import normalize_divipola_code


TARGET = "rendimiento_t_ha"
CLASS_TARGET = "rendimiento_alto"
KEY_COLUMNS = ["codigo_dane_municipio", "anio"]
HISTORY_VALUE_COLUMNS = [
    "rendimiento_t_ha",
    "produccion_t",
    "area_cosechada_ha",
    "area_sembrada_ha",
]
CLIMATE_FEATURE_COLUMNS = [
    "precipitacion_total_mm_lag_1",
    "temperatura_media_c_lag_1",
    "temperatura_min_c_lag_1",
    "temperatura_max_c_lag_1",
    "precipitacion_anomalia_lag_1",
    "temperatura_anomalia_lag_1",
    "distancia_grilla_grados",
]


def fixed_high_yield_threshold(frame: pd.DataFrame) -> float:
    """Return the fixed median yield observed through 2018."""
    if TARGET not in frame or "anio" not in frame:
        raise ValueError(f"Se requieren las columnas anio y {TARGET}")
    historical = pd.to_numeric(
        frame.loc[pd.to_numeric(frame["anio"], errors="coerce") <= 2018, TARGET],
        errors="coerce",
    ).dropna()
    if historical.empty:
        raise ValueError("No hay rendimiento histórico hasta 2018 para fijar el umbral")
    return float(historical.median())


def _exact_calendar_lags(base: pd.DataFrame) -> pd.DataFrame:
    result = base.copy()
    for lag in (1, 2, 3):
        previous = base[KEY_COLUMNS + HISTORY_VALUE_COLUMNS].copy()
        previous["anio"] = previous["anio"] + lag
        previous = previous.rename(
            columns={column: f"{column.replace('_t_ha', '')}_lag_{lag}" for column in HISTORY_VALUE_COLUMNS}
        )
        result = result.merge(previous, on=KEY_COLUMNS, how="left", validate="one_to_one")
    return result


def _slope(years: pd.Series, values: pd.Series) -> float:
    valid = pd.DataFrame({"year": years, "value": values}).dropna()
    if len(valid) < 2 or valid["year"].nunique() < 2:
        return np.nan
    return float(np.polyfit(valid["year"].to_numpy(float), valid["value"].to_numpy(float), 1)[0])


def _municipality_history_features(base: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for code, group in base.groupby("codigo_dane_municipio", sort=False):
        ordered = group.sort_values("anio")
        for index, current in ordered.iterrows():
            year = int(current["anio"])
            history = ordered[ordered["anio"] < year]
            record: dict[str, Any] = {
                "_row_index": index,
                "history_years": int(len(history)),
                "max_eva_feature_year": (
                    int(history["anio"].max()) if not history.empty else np.nan
                ),
                "rendimiento_municipio_historico": (
                    float(history[TARGET].mean()) if not history.empty else np.nan
                ),
                "proporcion_alto_historica": (
                    float((history[TARGET] > threshold).mean()) if not history.empty else np.nan
                ),
            }
            for window in (3, 5):
                recent = history[history["anio"] >= year - window]
                values = recent[TARGET].dropna()
                prefix = f"rendimiento_{window}y"
                record[f"{prefix}_conteo"] = int(len(values))
                record[f"rendimiento_media_{window}y"] = (
                    float(values.mean()) if len(values) else np.nan
                )
                record[f"rendimiento_mediana_{window}y"] = (
                    float(values.median()) if len(values) else np.nan
                )
                record[f"rendimiento_std_{window}y"] = (
                    float(values.std(ddof=0)) if len(values) else np.nan
                )
                record[f"rendimiento_min_{window}y"] = (
                    float(values.min()) if len(values) else np.nan
                )
                record[f"rendimiento_max_{window}y"] = (
                    float(values.max()) if len(values) else np.nan
                )
                record[f"rendimiento_pendiente_{window}y"] = _slope(
                    recent["anio"], recent[TARGET]
                )
            rows.append(record)
    history_frame = pd.DataFrame(rows).set_index("_row_index")
    return history_frame.reindex(base.index)


def _geographic_priors(base: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=base.index)
    result["rendimiento_departamento_historico"] = np.nan
    result["rendimiento_nacional_historico"] = np.nan
    for year in sorted(base["anio"].unique()):
        past = base[base["anio"] < year]
        current_index = base.index[base["anio"] == year]
        if past.empty:
            continue
        department_means = past.groupby("codigo_dane_departamento")[TARGET].mean()
        result.loc[current_index, "rendimiento_departamento_historico"] = (
            base.loc[current_index, "codigo_dane_departamento"].map(department_means).to_numpy()
        )
        result.loc[current_index, "rendimiento_nacional_historico"] = float(
            past[TARGET].mean()
        )
    return result


def _prepare_climate(climate: pd.DataFrame) -> pd.DataFrame:
    data = climate.copy()
    required = {
        "codigo_dane_municipio",
        "anio_clima",
        "precipitacion_total_mm",
        "temperatura_media_c",
        "temperatura_min_c",
        "temperatura_max_c",
    }
    if missing := sorted(required.difference(data.columns)):
        raise ValueError(f"Faltan columnas de clima: {missing}")
    data["codigo_dane_municipio"] = data["codigo_dane_municipio"].map(
        normalize_divipola_code
    )
    data = data.sort_values(["codigo_dane_municipio", "anio_clima"])
    data["precipitacion_normal_previa"] = data.groupby(
        "codigo_dane_municipio", sort=False
    )["precipitacion_total_mm"].transform(lambda series: series.expanding().mean().shift())
    data["temperatura_normal_previa"] = data.groupby(
        "codigo_dane_municipio", sort=False
    )["temperatura_media_c"].transform(lambda series: series.expanding().mean().shift())
    data["precipitacion_anomalia_lag_1"] = (
        data["precipitacion_total_mm"] - data["precipitacion_normal_previa"]
    )
    data["temperatura_anomalia_lag_1"] = (
        data["temperatura_media_c"] - data["temperatura_normal_previa"]
    )
    data = data.rename(
        columns={
            "precipitacion_total_mm": "precipitacion_total_mm_lag_1",
            "temperatura_media_c": "temperatura_media_c_lag_1",
            "temperatura_min_c": "temperatura_min_c_lag_1",
            "temperatura_max_c": "temperatura_max_c_lag_1",
            "anio_clima": "max_climate_feature_year",
        }
    )
    data["anio"] = data["max_climate_feature_year"] + 1
    keep = [
        "codigo_dane_municipio",
        "anio",
        "max_climate_feature_year",
        *CLIMATE_FEATURE_COLUMNS,
    ]
    return data[keep].drop_duplicates(KEY_COLUMNS)


def build_point_in_time_features(
    eva: pd.DataFrame,
    municipalities: pd.DataFrame,
    climate: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a row per observed municipality-year using prior information only."""
    required = {
        "codigo_dane_municipio",
        "codigo_dane_departamento",
        "departamento",
        "municipio",
        "cultivo",
        "anio",
        "area_sembrada_ha",
        "area_cosechada_ha",
        "produccion_t",
        TARGET,
        "fuente_eva",
    }
    if missing := sorted(required.difference(eva.columns)):
        raise ValueError(f"Faltan columnas EVA: {missing}")
    base = eva.copy().reset_index(drop=True)
    base["codigo_dane_municipio"] = base["codigo_dane_municipio"].map(
        normalize_divipola_code
    )
    base["codigo_dane_departamento"] = base["codigo_dane_departamento"].astype(str).str.zfill(2)
    base["anio"] = pd.to_numeric(base["anio"], errors="raise").astype(int)
    if base[KEY_COLUMNS].duplicated().any():
        raise ValueError("EVA contiene llaves municipio-año duplicadas")
    base = base.sort_values(KEY_COLUMNS).reset_index(drop=True)
    threshold = fixed_high_yield_threshold(base)

    result = _exact_calendar_lags(base)
    history = _municipality_history_features(base, threshold)
    result = pd.concat([result, history.reset_index(drop=True)], axis=1)
    priors = _geographic_priors(base)
    result = pd.concat([result, priors.reset_index(drop=True)], axis=1)

    municipality_columns = [
        "codigo_dane_municipio",
        *[column for column in ("latitud", "longitud") if column in municipalities],
    ]
    municipality_reference = municipalities[municipality_columns].copy()
    municipality_reference["codigo_dane_municipio"] = municipality_reference[
        "codigo_dane_municipio"
    ].map(normalize_divipola_code)
    if municipality_reference["codigo_dane_municipio"].duplicated().any():
        raise ValueError("DIVIPOLA contiene códigos municipales duplicados")
    result = result.merge(
        municipality_reference,
        on="codigo_dane_municipio",
        how="left",
        validate="many_to_one",
    )

    if climate is not None:
        climate_features = _prepare_climate(climate)
        result = result.merge(
            climate_features,
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )
    else:
        result["max_climate_feature_year"] = np.nan
        for column in CLIMATE_FEATURE_COLUMNS:
            result[column] = np.nan

    result["feature_cutoff_year"] = result["anio"] - 1
    result["source_break_2019"] = (result["anio"] >= 2019).astype(int)
    result["climate_available"] = result["max_climate_feature_year"].notna().astype(int)
    result[CLASS_TARGET] = (result[TARGET] > threshold).astype(int)
    lag_class = pd.Series(pd.NA, index=result.index, dtype="Int64")
    has_lag = result["rendimiento_lag_1"].notna()
    lag_class.loc[has_lag] = (
        result.loc[has_lag, "rendimiento_lag_1"] > threshold
    ).astype(int)
    result["clase_lag_1"] = lag_class
    result["cambio_clase_vs_lag_1"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[has_lag, "cambio_clase_vs_lag_1"] = (
        result.loc[has_lag, CLASS_TARGET].astype(int)
        != result.loc[has_lag, "clase_lag_1"].astype(int)
    ).astype(int)
    result.attrs["high_yield_threshold_t_ha"] = threshold
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def audit_point_in_time(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate uniqueness, geography, target, and feature provenance."""
    required = {
        *KEY_COLUMNS,
        TARGET,
        CLASS_TARGET,
        "feature_cutoff_year",
        "max_eva_feature_year",
        "max_climate_feature_year",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Faltan columnas point-in-time: {missing}")
    duplicates = int(frame[KEY_COLUMNS].duplicated().sum())
    if duplicates:
        raise ValueError(f"El mart contiene {duplicates} llaves duplicadas")
    valid_codes = frame["codigo_dane_municipio"].astype(str).str.fullmatch(r"\d{5}")
    if not valid_codes.all():
        raise ValueError("El mart contiene códigos DIVIPOLA inválidos")
    if frame[TARGET].isna().any() or (frame[TARGET] < 0).any():
        raise ValueError("El mart contiene targets vacíos o imposibles")
    future_eva = frame["max_eva_feature_year"].notna() & (
        frame["max_eva_feature_year"] >= frame["anio"]
    )
    future_climate = frame["max_climate_feature_year"].notna() & (
        frame["max_climate_feature_year"] >= frame["anio"]
    )
    if future_eva.any() or future_climate.any():
        raise ValueError("Se detectó una variable proveniente del año evaluado o del futuro")
    return {
        "rows": int(len(frame)),
        "municipalities": int(frame["codigo_dane_municipio"].nunique()),
        "year_min": int(frame["anio"].min()),
        "year_max": int(frame["anio"].max()),
        "duplicate_keys": duplicates,
        "climate_coverage": float(frame["max_climate_feature_year"].notna().mean()),
        "missing_percent": {
            column: float(value * 100)
            for column, value in frame.isna().mean().sort_values(ascending=False).items()
        },
    }
