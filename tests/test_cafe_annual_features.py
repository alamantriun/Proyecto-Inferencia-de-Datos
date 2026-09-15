from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.cafe_annual_features import (
    audit_point_in_time,
    build_point_in_time_features,
    fixed_high_yield_threshold,
)


def _municipalities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "codigo_dane_municipio": ["05001", "17001"],
            "codigo_dane_departamento": ["05", "17"],
            "latitud": [6.25, 5.07],
            "longitud": [-75.56, -75.52],
        }
    )


def _eva_panel() -> pd.DataFrame:
    rows = []
    values = {
        "05001": {2017: 0.8, 2018: 1.0, 2019: 1.1, 2020: 1.2, 2024: 1.4, 2025: 1.6},
        "17001": {2017: 0.6, 2018: 0.7, 2019: 0.9, 2020: 1.0, 2024: 1.1, 2025: 1.2},
    }
    for code, years in values.items():
        department = code[:2]
        for year, yield_value in years.items():
            area = 10.0
            rows.append(
                {
                    "codigo_dane_municipio": code,
                    "codigo_dane_departamento": department,
                    "departamento": "ANTIOQUIA" if department == "05" else "CALDAS",
                    "municipio": f"M-{code}",
                    "cultivo": "CAFE",
                    "anio": year,
                    "area_sembrada_ha": area + 1,
                    "area_cosechada_ha": area,
                    "produccion_t": area * yield_value,
                    "rendimiento_t_ha": yield_value,
                    "estado_fisico": "PERGAMINO O SECO DE TRILLA",
                    "fuente_eva": "historica" if year <= 2018 else "reciente",
                }
            )
    return pd.DataFrame(rows)


def _climate() -> pd.DataFrame:
    rows = []
    for code in ("05001", "17001"):
        for year in range(2016, 2025):
            rows.append(
                {
                    "codigo_dane_municipio": code,
                    "anio_clima": year,
                    "precipitacion_total_mm": 1000 + year,
                    "temperatura_media_c": 20 + (year - 2016) / 10,
                    "temperatura_min_c": 15.0,
                    "temperatura_max_c": 25.0,
                    "distancia_grilla_grados": 0.1,
                    "fuente_clima": "NASA_POWER_MERRA2",
                }
            )
    return pd.DataFrame(rows)


def test_fixed_threshold_uses_only_2007_2018_values():
    frame = _eva_panel()
    expected = np.median([0.8, 1.0, 0.6, 0.7])

    assert fixed_high_yield_threshold(frame) == pytest.approx(expected)


def test_features_for_2025_never_use_2025_values():
    result = build_point_in_time_features(_eva_panel(), _municipalities(), _climate())
    row = result.query("codigo_dane_municipio == '05001' and anio == 2025").iloc[0]

    assert row["rendimiento_lag_1"] == pytest.approx(1.4)
    assert row["max_eva_feature_year"] == 2024
    assert row["max_climate_feature_year"] == 2024
    assert row["feature_cutoff_year"] == 2024
    assert row["precipitacion_total_mm_lag_1"] == pytest.approx(3024.0)


def test_lag_requires_consecutive_calendar_year():
    result = build_point_in_time_features(_eva_panel(), _municipalities())
    row = result.query("codigo_dane_municipio == '05001' and anio == 2024").iloc[0]

    assert pd.isna(row["rendimiento_lag_1"])
    assert pd.isna(row["rendimiento_lag_3"])
    assert row["rendimiento_media_5y"] == pytest.approx(1.15)
    assert row["history_years"] == 4


def test_expanding_geographic_priors_exclude_current_year_target():
    frame = _eva_panel()
    result = build_point_in_time_features(frame, _municipalities())
    row = result.query("codigo_dane_municipio == '05001' and anio == 2019").iloc[0]

    assert row["rendimiento_municipio_historico"] == pytest.approx(0.9)
    assert row["rendimiento_departamento_historico"] == pytest.approx(0.9)
    assert row["rendimiento_nacional_historico"] == pytest.approx(0.775)


def test_audit_rejects_future_provenance_and_accepts_valid_mart():
    mart = build_point_in_time_features(_eva_panel(), _municipalities(), _climate())
    audit = audit_point_in_time(mart)

    assert audit["rows"] == len(mart)
    assert audit["duplicate_keys"] == 0
    assert audit["year_max"] == 2025

    invalid = mart.copy()
    invalid.loc[invalid.index[0], "max_eva_feature_year"] = invalid.loc[invalid.index[0], "anio"]
    with pytest.raises(ValueError, match="futuro"):
        audit_point_in_time(invalid)
