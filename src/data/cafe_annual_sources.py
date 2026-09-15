"""Official, traceable source loading for the annual coffee classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import time
import unicodedata
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


EVA_HISTORICAL_ID = "2pnw-mmge"
EVA_RECENT_ID = "uejq-wxrr"
SOCRATA_BASE_URL = "https://www.datos.gov.co/resource"
DIVIPOLA_URL = (
    "https://geoportal.dane.gov.co/mparcgis/rest/services/Divipola/"
    "Serv_DIVIPOLA_MGN_2025/FeatureServer/317/query"
)
COFFEE_CROP_CODE = "2030300"

FetchJson = Callable[[str, dict[str, Any]], Any]


def normalize_text(value: object) -> str:
    """Return a stable upper-case ASCII label for geographic comparisons."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip())
    return " ".join(text.encode("ascii", "ignore").decode("ascii").upper().split())


def normalize_divipola_code(value: object) -> str:
    """Normalize a municipality identifier to the canonical five digits."""
    if value is None or pd.isna(value):
        raise ValueError("Código DIVIPOLA vacío")
    raw = str(value).strip()
    if raw.lower() in {"", "nan", "none", "null"}:
        raise ValueError("Código DIVIPOLA vacío")
    digits = re.sub(r"\D", "", raw)
    if not digits or len(digits) > 5:
        raise ValueError(f"Código DIVIPOLA inválido: {value!r}")
    return digits.zfill(5)


def _department_code(value: object) -> str:
    if value is None or pd.isna(value):
        raise ValueError("Código DIVIPOLA de departamento vacío")
    digits = re.sub(r"\D", "", str(value))
    if not digits or len(digits) > 2:
        raise ValueError(f"Código DIVIPOLA de departamento inválido: {value!r}")
    return digits.zfill(2)


def _historical_municipality_code(department: object, municipality: object) -> str:
    municipality_digits = re.sub(r"\D", "", str(municipality))
    if len(municipality_digits) <= 3:
        return _department_code(department) + municipality_digits.zfill(3)
    return normalize_divipola_code(municipality)


def _prepare_historical(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "c_d_dep",
        "c_d_mun",
        "departamento",
        "municipio",
        "cultivo",
        "a_o",
        "rea_sembrada_ha",
        "rea_cosechada_ha",
        "producci_n_t",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas EVA histórica: {missing}")
    data = frame.copy()
    data["codigo_dane_departamento"] = data["c_d_dep"].map(_department_code)
    data["codigo_dane_municipio"] = [
        _historical_municipality_code(dep, mun)
        for dep, mun in zip(data["c_d_dep"], data["c_d_mun"], strict=True)
    ]
    data = data.rename(
        columns={
            "a_o": "anio",
            "rea_sembrada_ha": "area_sembrada_ha",
            "rea_cosechada_ha": "area_cosechada_ha",
            "producci_n_t": "produccion_t",
            "estado_fisico_produccion": "estado_fisico",
        }
    )
    data["fuente_eva"] = "historica"
    return data


def _prepare_recent(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "c_digo_dane_departamento",
        "c_digo_dane_municipio",
        "departamento",
        "municipio",
        "cultivo",
        "a_o",
        "rea_sembrada",
        "rea_cosechada",
        "producci_n",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas EVA reciente: {missing}")
    data = frame.copy()
    if "c_digo_del_cultivo" in data:
        data = data[data["c_digo_del_cultivo"].astype(str) == COFFEE_CROP_CODE].copy()
    data["codigo_dane_departamento"] = data["c_digo_dane_departamento"].map(
        _department_code
    )
    data["codigo_dane_municipio"] = data["c_digo_dane_municipio"].map(
        normalize_divipola_code
    )
    data = data.rename(
        columns={
            "a_o": "anio",
            "rea_sembrada": "area_sembrada_ha",
            "rea_cosechada": "area_cosechada_ha",
            "producci_n": "produccion_t",
            "estado_f_sico_del_cultivo": "estado_fisico",
        }
    )
    data["fuente_eva"] = "reciente"
    return data


def harmonize_eva(historical: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    """Harmonize both EVA eras and aggregate coffee with area-weighted yield."""
    prepared = [_prepare_historical(historical), _prepare_recent(recent)]
    data = pd.concat(prepared, ignore_index=True, sort=False)
    for column in ("departamento", "municipio", "cultivo", "estado_fisico"):
        if column not in data:
            data[column] = ""
        data[column] = data[column].map(normalize_text)
    data = data[data["cultivo"].str.contains("CAFE", na=False)].copy()
    for column in (
        "anio",
        "area_sembrada_ha",
        "area_cosechada_ha",
        "produccion_t",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[
        data["anio"].notna()
        & data["area_cosechada_ha"].gt(0)
        & data["produccion_t"].ge(0)
    ].copy()
    data["anio"] = data["anio"].astype(int)

    keys = ["codigo_dane_municipio", "anio"]
    result = data.groupby(keys, as_index=False, sort=True).agg(
        codigo_dane_departamento=("codigo_dane_departamento", "first"),
        departamento=("departamento", "first"),
        municipio=("municipio", "first"),
        cultivo=("cultivo", "first"),
        area_sembrada_ha=("area_sembrada_ha", "sum"),
        area_cosechada_ha=("area_cosechada_ha", "sum"),
        produccion_t=("produccion_t", "sum"),
        estado_fisico=("estado_fisico", "first"),
        fuente_eva=("fuente_eva", "first"),
    )
    result["rendimiento_t_ha"] = (
        result["produccion_t"] / result["area_cosechada_ha"]
    )
    if result[keys].duplicated().any():
        raise ValueError("La EVA armonizada contiene llaves duplicadas")
    columns = [
        "codigo_dane_municipio",
        "codigo_dane_departamento",
        "departamento",
        "municipio",
        "cultivo",
        "anio",
        "area_sembrada_ha",
        "area_cosechada_ha",
        "produccion_t",
        "rendimiento_t_ha",
        "estado_fisico",
        "fuente_eva",
    ]
    return result[columns].sort_values(keys).reset_index(drop=True)


def parse_divipola_features(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse municipality attributes and WGS84 centroids from DANE ArcGIS."""
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        attributes = feature.get("attributes", {})
        centroid = feature.get("centroid") or {}
        rows.append(
            {
                "codigo_dane_municipio": normalize_divipola_code(
                    attributes.get("MPIO_CDPMP")
                ),
                "codigo_dane_departamento": _department_code(
                    attributes.get("MPIO_CDPMP", "")[:2]
                ),
                "departamento_divipola": normalize_text(
                    attributes.get("DPTO_CNMBRE")
                ),
                "municipio_divipola": normalize_text(
                    attributes.get("MPIO_CNMBRE")
                ),
                "longitud": float(centroid.get("x")),
                "latitud": float(centroid.get("y")),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("La respuesta DIVIPOLA no contiene municipios")
    if result["codigo_dane_municipio"].duplicated().any():
        duplicated = result.loc[
            result["codigo_dane_municipio"].duplicated(False),
            "codigo_dane_municipio",
        ].tolist()
        raise ValueError(f"La respuesta DIVIPOLA contiene códigos duplicados: {duplicated}")
    if not result[["latitud", "longitud"]].notna().all().all():
        raise ValueError("DIVIPOLA contiene centroides vacíos")
    return result.sort_values("codigo_dane_municipio").reset_index(drop=True)


def _request_json(url: str, params: dict[str, Any]) -> Any:
    for attempt in range(4):
        response = requests.get(url, params=params, timeout=120)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 3:
                response.raise_for_status()
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"No fue posible descargar {url}")


def _download_socrata(
    dataset_id: str,
    select: str,
    where: str,
    fetch_json: FetchJson,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offset = 0
    batch_size = 50_000
    url = f"{SOCRATA_BASE_URL}/{dataset_id}.json"
    while True:
        batch = fetch_json(
            url,
            {
                "$select": select,
                "$where": where,
                "$order": "a_o,c_digo_dane_municipio"
                if dataset_id == EVA_RECENT_ID
                else "a_o,c_d_dep,c_d_mun",
                "$limit": batch_size,
                "$offset": offset,
            },
        )
        if not isinstance(batch, list):
            raise ValueError(f"Respuesta SODA inválida para {dataset_id}")
        rows.extend(batch)
        if len(batch) < batch_size:
            break
        offset += len(batch)
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, dir=destination.parent, encoding="utf-8", newline=""
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    temporary.replace(destination)


def download_official_sources(
    cache_dir: Path,
    refresh: bool = False,
    fetch_json: FetchJson | None = None,
) -> dict[str, Path]:
    """Download, harmonize, and cache official EVA and DANE sources."""
    cache_dir = Path(cache_dir)
    paths = {
        "eva": cache_dir / "eva_cafe_2007_2025.csv",
        "divipola": cache_dir / "divipola_municipios_2025.csv",
        "manifest": cache_dir / "source_manifest.json",
    }
    if not refresh and all(path.exists() for path in paths.values()):
        return paths

    fetch = fetch_json or _request_json
    historical = _download_socrata(
        EVA_HISTORICAL_ID,
        (
            "c_d_dep,c_d_mun,departamento,municipio,cultivo,a_o,"
            "rea_sembrada_ha,rea_cosechada_ha,producci_n_t,rendimiento_t_ha,"
            "estado_fisico_produccion"
        ),
        "upper(cultivo) like '%CAF%'",
        fetch,
    )
    recent = _download_socrata(
        EVA_RECENT_ID,
        (
            "c_digo_dane_departamento,c_digo_dane_municipio,departamento,"
            "municipio,cultivo,a_o,rea_sembrada,rea_cosechada,producci_n,"
            "rendimiento,estado_f_sico_del_cultivo,c_digo_del_cultivo"
        ),
        f"c_digo_del_cultivo='{COFFEE_CROP_CODE}'",
        fetch,
    )
    eva = harmonize_eva(historical, recent)
    divipola_payload = fetch(
        DIVIPOLA_URL,
        {
            "where": "1=1",
            "outFields": "MPIO_CDPMP,DPTO_CNMBRE,MPIO_CNMBRE",
            "returnGeometry": "false",
            "returnCentroid": "true",
            "outSR": 4326,
            "f": "json",
        },
    )
    divipola = parse_divipola_features(divipola_payload)
    _atomic_csv(eva, paths["eva"])
    _atomic_csv(divipola, paths["divipola"])

    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "eva_historical": {
                "dataset_id": EVA_HISTORICAL_ID,
                "url": f"{SOCRATA_BASE_URL}/{EVA_HISTORICAL_ID}.json",
                "rows_raw": int(len(historical)),
            },
            "eva_recent": {
                "dataset_id": EVA_RECENT_ID,
                "url": f"{SOCRATA_BASE_URL}/{EVA_RECENT_ID}.json",
                "rows_raw": int(len(recent)),
            },
            "divipola": {
                "dataset_id": "Serv_DIVIPOLA_MGN_2025/317",
                "url": DIVIPOLA_URL,
                "rows_raw": int(len(divipola)),
            },
        },
        "outputs": {
            "eva": {"rows": int(len(eva)), "sha256": _sha256(paths["eva"])},
            "divipola": {
                "rows": int(len(divipola)),
                "sha256": _sha256(paths["divipola"]),
            },
        },
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=cache_dir, encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temporary.replace(paths["manifest"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external/cafe_annual"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    paths = download_official_sources(args.cache_dir, refresh=args.refresh)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
