"""Download bounded NASA POWER regional climate and map it to municipalities."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/regional"
POWER_PARAMETERS = ("PRECTOTCORR", "T2M", "T2M_MIN", "T2M_MAX")
COLOMBIA_TILES = (
    (-5.0, 5.0, -80.0, -70.0),
    (-5.0, 5.0, -70.0, -66.0),
    (5.0, 14.0, -80.0, -70.0),
    (5.0, 14.0, -70.0, -66.0),
)
FetchJson = Callable[[str, dict[str, Any]], Any]


def parse_power_regional(payload: dict[str, Any], parameter: str) -> pd.DataFrame:
    """Parse monthly values from a POWER regional GeoJSON response."""
    rows: list[dict[str, float | int | str]] = []
    for feature in payload.get("features", []):
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        if len(coordinates) < 2:
            continue
        values = (
            feature.get("properties", {})
            .get("parameter", {})
            .get(parameter, {})
        )
        for raw_period, raw_value in values.items():
            period = str(raw_period)
            if len(period) != 6 or not period.isdigit():
                continue
            year, month = int(period[:4]), int(period[4:])
            if month not in range(1, 13):
                continue
            value = pd.to_numeric(raw_value, errors="coerce")
            if pd.isna(value) or float(value) <= -900:
                continue
            rows.append(
                {
                    "longitud_grilla": float(coordinates[0]),
                    "latitud_grilla": float(coordinates[1]),
                    "anio": year,
                    "mes": month,
                    "parametro": parameter,
                    "valor": float(value),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"NASA POWER no devolvió meses válidos para {parameter}")
    return result.drop_duplicates(
        ["longitud_grilla", "latitud_grilla", "anio", "mes", "parametro"]
    ).reset_index(drop=True)


def aggregate_power_yearly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Convert POWER monthly daily means into annual climate summaries."""
    required = {
        "longitud_grilla",
        "latitud_grilla",
        "anio",
        "mes",
        "parametro",
        "valor",
    }
    missing = sorted(required.difference(monthly.columns))
    if missing:
        raise ValueError(f"Faltan columnas mensuales POWER: {missing}")
    data = monthly.copy()
    data["dias_mes"] = [
        calendar.monthrange(int(year), int(month))[1]
        for year, month in zip(data["anio"], data["mes"], strict=True)
    ]
    keys = ["longitud_grilla", "latitud_grilla", "anio"]
    rain = data[data["parametro"] == "PRECTOTCORR"].copy()
    rain["aporte_mm"] = rain["valor"] * rain["dias_mes"]
    rain_year = rain.groupby(keys, as_index=False)["aporte_mm"].sum().rename(
        columns={"aporte_mm": "precipitacion_total_mm"}
    )

    names = {
        "T2M": "temperatura_media_c",
        "T2M_MIN": "temperatura_min_c",
        "T2M_MAX": "temperatura_max_c",
    }
    result = rain_year
    for parameter, column in names.items():
        part = data[data["parametro"] == parameter].copy()
        part["ponderado"] = part["valor"] * part["dias_mes"]
        grouped = part.groupby(keys, as_index=False).agg(
            ponderado=("ponderado", "sum"), dias=("dias_mes", "sum")
        )
        grouped[column] = grouped["ponderado"] / grouped["dias"]
        result = result.merge(grouped[keys + [column]], on=keys, how="outer")
    result = result.rename(columns={"anio": "anio_clima"})
    return result.sort_values(["anio_clima", "latitud_grilla", "longitud_grilla"]).reset_index(
        drop=True
    )


def nearest_grid_climate(
    municipalities: pd.DataFrame, climate: pd.DataFrame
) -> pd.DataFrame:
    """Assign each municipality centroid to its nearest POWER grid point."""
    municipality_required = {"codigo_dane_municipio", "latitud", "longitud"}
    climate_required = {"latitud_grilla", "longitud_grilla", "anio_clima"}
    if missing := sorted(municipality_required.difference(municipalities.columns)):
        raise ValueError(f"Faltan columnas DIVIPOLA: {missing}")
    if missing := sorted(climate_required.difference(climate.columns)):
        raise ValueError(f"Faltan columnas climáticas: {missing}")
    points = climate[["latitud_grilla", "longitud_grilla"]].drop_duplicates().reset_index(
        drop=True
    )
    if points.empty:
        raise ValueError("No hay puntos de grilla climática")
    municipality_coordinates = municipalities[["latitud", "longitud"]].to_numpy(float)
    grid_coordinates = points[["latitud_grilla", "longitud_grilla"]].to_numpy(float)
    distances = ((municipality_coordinates[:, None, :] - grid_coordinates[None, :, :]) ** 2).sum(
        axis=2
    )
    nearest = distances.argmin(axis=1)
    assignments = municipalities[["codigo_dane_municipio"]].copy()
    assignments["latitud_grilla"] = points.iloc[nearest]["latitud_grilla"].to_numpy()
    assignments["longitud_grilla"] = points.iloc[nearest]["longitud_grilla"].to_numpy()
    assignments["distancia_grilla_grados"] = np.sqrt(distances[np.arange(len(nearest)), nearest])
    result = assignments.merge(
        climate,
        on=["latitud_grilla", "longitud_grilla"],
        how="left",
        validate="many_to_many",
    )
    result["fuente_clima"] = "NASA_POWER_MERRA2"
    if result[["codigo_dane_municipio", "anio_clima"]].duplicated().any():
        raise ValueError("La asignación climática produjo llaves duplicadas")
    return result.sort_values(["codigo_dane_municipio", "anio_clima"]).reset_index(drop=True)


def _request_json(url: str, params: dict[str, Any]) -> Any:
    for attempt in range(5):
        response = requests.get(url, params=params, timeout=180)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 4:
                response.raise_for_status()
            time.sleep(min(30, 2**attempt))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("NASA POWER no respondió después de cinco intentos")


def _write_json_atomic(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=destination.parent, encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False)
    temporary.replace(destination)


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, dir=destination.parent, encoding="utf-8", newline=""
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    temporary.replace(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_power_climate(
    cache_dir: Path,
    municipalities_path: Path | None = None,
    start_year: int = 2006,
    end_year: int = 2024,
    refresh: bool = False,
    fetch_json: FetchJson | None = None,
) -> Path:
    """Download a bounded regional POWER grid and publish municipality-year climate."""
    if start_year > end_year:
        raise ValueError("El año inicial del clima no puede superar el final")
    cache_dir = Path(cache_dir)
    output = cache_dir / f"nasa_power_municipio_anio_{start_year}_{end_year}.csv"
    if output.exists() and not refresh:
        return output
    municipalities_path = municipalities_path or cache_dir / "divipola_municipios_2025.csv"
    municipalities = pd.read_csv(
        municipalities_path, dtype={"codigo_dane_municipio": str}
    )
    fetch = fetch_json or _request_json
    frames: list[pd.DataFrame] = []
    raw_dir = cache_dir / "power_raw"
    for parameter in POWER_PARAMETERS:
        for tile_index, tile in enumerate(COLOMBIA_TILES, start=1):
            params = {
                "latitude-min": tile[0],
                "latitude-max": tile[1],
                "longitude-min": tile[2],
                "longitude-max": tile[3],
                "parameters": parameter,
                "community": "AG",
                "start": start_year,
                "end": end_year,
                "format": "JSON",
            }
            payload = fetch(POWER_URL, params)
            raw_path = raw_dir / f"{parameter}_{start_year}_{end_year}_tile_{tile_index}.json"
            _write_json_atomic(payload, raw_path)
            frames.append(parse_power_regional(payload, parameter))
    monthly = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["longitud_grilla", "latitud_grilla", "anio", "mes", "parametro"]
    )
    annual_grid = aggregate_power_yearly(monthly)
    result = nearest_grid_climate(municipalities, annual_grid)
    _write_csv_atomic(result, output)
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": POWER_URL,
        "source_model": "NASA POWER MERRA-2",
        "start_year": int(start_year),
        "end_year": int(end_year),
        "parameters": list(POWER_PARAMETERS),
        "tiles": [list(tile) for tile in COLOMBIA_TILES],
        "rows": int(len(result)),
        "sha256": _sha256(output),
    }
    _write_json_atomic(manifest, cache_dir / "nasa_power_manifest.json")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external/cafe_annual"))
    parser.add_argument("--municipalities", type=Path)
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    output = download_power_climate(
        args.cache_dir,
        municipalities_path=args.municipalities,
        start_year=args.start_year,
        end_year=args.end_year,
        refresh=args.refresh,
    )
    print(output)


if __name__ == "__main__":
    main()
