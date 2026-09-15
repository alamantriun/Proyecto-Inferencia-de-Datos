from __future__ import annotations

import pandas as pd
import pytest

from src.data.cafe_annual_sources import (
    download_official_sources,
    harmonize_eva,
    normalize_divipola_code,
    parse_divipola_features,
)


def _historical_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "c_d_dep": [5],
            "c_d_mun": [1],
            "departamento": ["Antioquia"],
            "municipio": ["Medellín"],
            "cultivo": ["Café"],
            "a_o": [2018],
            "rea_sembrada_ha": [12.0],
            "rea_cosechada_ha": [10.0],
            "producci_n_t": [8.0],
            "rendimiento_t_ha": [0.8],
            "estado_fisico_produccion": ["Pergamino o seco de trilla"],
        }
    )


def _recent_duplicate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "c_digo_dane_departamento": ["05", "05"],
            "c_digo_dane_municipio": ["05001", "05001"],
            "departamento": ["Antioquia", "Antioquia"],
            "municipio": ["Medellín", "Medellín"],
            "cultivo": ["Café", "Café"],
            "a_o": [2025, 2025],
            "rea_sembrada": [8.0, 14.0],
            "rea_cosechada": [5.0, 15.0],
            "producci_n": [5.0, 25.0],
            "rendimiento": [1.0, 25.0 / 15.0],
            "estado_f_sico_del_cultivo": [
                "Pergamino o seco de trilla",
                "Pergamino o seco de trilla",
            ],
            "c_digo_del_cultivo": ["2030300", "2030300"],
        }
    )


def test_normalize_divipola_preserves_leading_zeroes():
    assert normalize_divipola_code(5001) == "05001"
    assert normalize_divipola_code("05-001") == "05001"


def test_normalize_divipola_rejects_missing_and_wrong_length_codes():
    for value in (None, "", "nan", "123456"):
        with pytest.raises(ValueError, match="DIVIPOLA"):
            normalize_divipola_code(value)


def test_harmonize_eva_recomputes_weighted_yield_and_unique_keys():
    result = harmonize_eva(_historical_rows(), _recent_duplicate_rows())

    row = result.query("codigo_dane_municipio == '05001' and anio == 2025").iloc[0]
    assert row["produccion_t"] == pytest.approx(30.0)
    assert row["area_cosechada_ha"] == pytest.approx(20.0)
    assert row["rendimiento_t_ha"] == pytest.approx(1.5)
    assert result[["codigo_dane_municipio", "anio"]].duplicated().sum() == 0
    assert set(result["codigo_dane_municipio"].str.len()) == {5}
    assert set(result["fuente_eva"]) == {"historica", "reciente"}


def test_harmonize_eva_filters_non_coffee_recent_rows():
    recent = _recent_duplicate_rows()
    non_coffee = recent.iloc[[0]].assign(
        cultivo="Cacao", c_digo_del_cultivo="2040100"
    )

    result = harmonize_eva(_historical_rows(), pd.concat([recent, non_coffee]))

    assert set(result["cultivo"]) == {"CAFE"}
    assert len(result.query("anio == 2025")) == 1


def test_parse_divipola_uses_centroid_and_rejects_duplicate_codes():
    payload = {
        "features": [
            {
                "attributes": {
                    "MPIO_CDPMP": "05001",
                    "DPTO_CNMBRE": "ANTIOQUIA",
                    "MPIO_CNMBRE": "MEDELLÍN",
                },
                "centroid": {"x": -75.56, "y": 6.25},
            }
        ]
    }
    result = parse_divipola_features(payload)
    assert result.loc[0, "codigo_dane_municipio"] == "05001"
    assert result.loc[0, "latitud"] == pytest.approx(6.25)
    assert result.loc[0, "longitud"] == pytest.approx(-75.56)

    payload["features"].append(payload["features"][0])
    with pytest.raises(ValueError, match="duplicados"):
        parse_divipola_features(payload)


def test_download_official_sources_writes_traceable_cache(tmp_path):
    divipola = {
        "features": [
            {
                "attributes": {
                    "MPIO_CDPMP": "05001",
                    "DPTO_CNMBRE": "ANTIOQUIA",
                    "MPIO_CNMBRE": "MEDELLÍN",
                },
                "centroid": {"x": -75.56, "y": 6.25},
            }
        ]
    }

    def fake_fetch(url, params):
        if "2pnw-mmge" in url:
            return _historical_rows().to_dict(orient="records")
        if "uejq-wxrr" in url:
            return _recent_duplicate_rows().to_dict(orient="records")
        return divipola

    paths = download_official_sources(
        tmp_path, refresh=True, fetch_json=fake_fetch
    )

    assert set(paths) == {"eva", "divipola", "manifest"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    manifest = paths["manifest"].read_text(encoding="utf-8")
    assert "sha256" in manifest
    assert "uejq-wxrr" in manifest
    cached = pd.read_csv(paths["eva"], dtype={"codigo_dane_municipio": str})
    assert set(cached["anio"]) == {2018, 2025}
