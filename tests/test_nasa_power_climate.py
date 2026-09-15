from __future__ import annotations

import pandas as pd
import pytest

from src.data.nasa_power_climate import (
    aggregate_power_yearly,
    download_power_climate,
    nearest_grid_climate,
    parse_power_regional,
)


def _power_payload(parameter: str, value: float = 2.0) -> dict:
    values = {f"2020{month:02d}": value for month in range(1, 13)}
    values["202013"] = value
    return {
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-75.5, 6.0, 1000]},
                "properties": {"parameter": {parameter: values}},
            }
        ]
    }


def test_parse_power_regional_keeps_months_and_drops_annual_key_and_fill_value():
    payload = _power_payload("T2M", value=20.0)
    payload["features"][0]["properties"]["parameter"]["T2M"]["202006"] = -999

    result = parse_power_regional(payload, "T2M")

    assert set(result["mes"]) == set(range(1, 13)) - {6}
    assert len(result) == 11
    assert set(result["parametro"]) == {"T2M"}


def test_annual_precipitation_converts_monthly_mm_per_day_to_leap_year_total():
    parts = [
        parse_power_regional(_power_payload(parameter, value), parameter)
        for parameter, value in (
            ("PRECTOTCORR", 2.0),
            ("T2M", 20.0),
            ("T2M_MIN", 15.0),
            ("T2M_MAX", 25.0),
        )
    ]

    annual = aggregate_power_yearly(pd.concat(parts, ignore_index=True))

    assert annual.loc[0, "precipitacion_total_mm"] == pytest.approx(732.0)
    assert annual.loc[0, "temperatura_media_c"] == pytest.approx(20.0)
    assert annual.loc[0, "temperatura_min_c"] == pytest.approx(15.0)
    assert annual.loc[0, "temperatura_max_c"] == pytest.approx(25.0)


def test_nearest_grid_assignment_keeps_one_row_per_municipality_year():
    municipalities = pd.DataFrame(
        {
            "codigo_dane_municipio": ["05001", "17001"],
            "latitud": [6.05, 5.05],
            "longitud": [-75.45, -75.05],
        }
    )
    climate = pd.DataFrame(
        {
            "latitud_grilla": [6.0, 5.0],
            "longitud_grilla": [-75.5, -75.0],
            "anio_clima": [2020, 2020],
            "precipitacion_total_mm": [700.0, 800.0],
            "temperatura_media_c": [20.0, 21.0],
            "temperatura_min_c": [15.0, 16.0],
            "temperatura_max_c": [25.0, 26.0],
        }
    )

    result = nearest_grid_climate(municipalities, climate)

    assert result[["codigo_dane_municipio", "anio_clima"]].duplicated().sum() == 0
    assert result.set_index("codigo_dane_municipio").loc["05001", "precipitacion_total_mm"] == 700.0
    assert set(result["fuente_clima"]) == {"NASA_POWER_MERRA2"}


def test_download_power_climate_writes_cache_and_reuses_it(tmp_path):
    municipalities_path = tmp_path / "municipalities.csv"
    pd.DataFrame(
        {
            "codigo_dane_municipio": ["05001"],
            "latitud": [6.0],
            "longitud": [-75.5],
        }
    ).to_csv(municipalities_path, index=False)
    calls = []

    def fake_fetch(url, params):
        calls.append((url, params.copy()))
        return _power_payload(params["parameters"], value=2.0)

    output = download_power_climate(
        tmp_path,
        municipalities_path=municipalities_path,
        start_year=2020,
        end_year=2020,
        refresh=True,
        fetch_json=fake_fetch,
    )
    first_call_count = len(calls)
    reused = download_power_climate(
        tmp_path,
        municipalities_path=municipalities_path,
        start_year=2020,
        end_year=2020,
        refresh=False,
        fetch_json=fake_fetch,
    )

    assert output == reused
    assert output.exists() and output.stat().st_size > 0
    assert first_call_count == 16
    assert len(calls) == first_call_count
    result = pd.read_csv(output, dtype={"codigo_dane_municipio": str})
    assert result.loc[0, "codigo_dane_municipio"] == "05001"
    assert result.loc[0, "anio_clima"] == 2020
