"""
Solar-Rank — Fase 1.1: Exploración de la API SUI Facturación
=============================================================
Fuente: Superservicios - Facturación a Usuarios Energía
URL: https://www.datos.gov.co/resource/gw2d-7n7y.json
API: Socrata Open Data API (SODA)

Este script:
1. Conecta a la API SODA sin autenticación (público)
2. Descubre las columnas reales del dataset
3. Cuenta el volumen total de registros
4. Identifica el rango de fechas disponible
5. Filtra y muestra una muestra de usuarios residenciales
6. Genera un informe de exploración

Ejecutar: python src/data/explore_sui.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────
BASE_URL = "https://www.datos.gov.co/resource/gw2d-7n7y.json"
REPORT_DIR = Path("reports/tables")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def soda_query(query_params: dict, limit: int = 1000) -> list:
    """Ejecuta una query contra la SODA API y retorna los resultados."""
    params = {"$limit": limit}
    params.update(query_params)
    response = requests.get(BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def get_total_count() -> int:
    """Obtiene el conteo total de registros en el dataset."""
    result = soda_query({"$select": "count(*) as total"}, limit=1)
    return int(result[0]["total"]) if result else 0


def get_columns_sample() -> pd.DataFrame:
    """Obtiene una muestra para descubrir las columnas reales."""
    data = soda_query({}, limit=5)
    if not data:
        print("⚠ No se obtuvieron datos de la API.")
        sys.exit(1)
    return pd.DataFrame(data)


def get_date_range() -> dict:
    """Identifica las fechas mínima y máxima del dataset."""
    # Intentar con columnas comunes de fecha en SUI
    date_candidates = [
        "fecha", "fecha_facturacion", "periodo", "anio", "mes",
        "a_o", "periodo_facturacion", "fecha_inicio", "fecha_fin"
    ]

    # Primero descubrir qué columnas existen
    sample = soda_query({}, limit=1)
    if not sample:
        return {"error": "No hay datos"}

    available_cols = list(sample[0].keys())
    date_cols_found = [c for c in date_candidates if c in available_cols]

    result = {"columnas_fecha_encontradas": date_cols_found}

    for col in date_cols_found:
        try:
            min_result = soda_query(
                {"$select": f"min({col}) as min_val"}, limit=1
            )
            max_result = soda_query(
                {"$select": f"max({col}) as max_val"}, limit=1
            )
            result[col] = {
                "min": min_result[0].get("min_val", "N/A"),
                "max": max_result[0].get("max_val", "N/A"),
            }
        except Exception as e:
            result[col] = {"error": str(e)}

    return result


def get_categorical_distribution(column: str, top_n: int = 20) -> pd.DataFrame:
    """Distribución de valores de una columna categórica."""
    try:
        data = soda_query(
            {
                "$select": f"{column}, count(*) as conteo",
                "$group": column,
                "$order": "conteo DESC",
            },
            limit=top_n,
        )
        return pd.DataFrame(data)
    except Exception as e:
        print(f"  ⚠ No se pudo consultar '{column}': {e}")
        return pd.DataFrame()


def get_residential_sample(limit: int = 100) -> pd.DataFrame:
    """Obtiene una muestra de usuarios residenciales."""
    # Intentar filtros comunes
    filters = [
        "tipo_usuario='Residencial'",
        "tipo_usuario='RESIDENCIAL'",
        "tipo_usuario='1'",
        "clase_servicio='Residencial'",
    ]

    for f in filters:
        try:
            data = soda_query({"$where": f}, limit=limit)
            if data:
                print(f"  ✓ Filtro exitoso: {f} → {len(data)} registros")
                return pd.DataFrame(data)
        except Exception:
            continue

    print("  ⚠ No se encontró filtro residencial. Retornando muestra general.")
    data = soda_query({}, limit=limit)
    return pd.DataFrame(data)


def main():
    """Exploración completa de la fuente SUI."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = {
        "fuente": "SUI Formato 1743 - Facturación a Usuarios Energía",
        "url": BASE_URL,
        "fecha_exploracion": timestamp,
    }

    print("=" * 70)
    print("Solar-Rank — Exploración API SUI Facturación")
    print("=" * 70)

    # ── 1. Volumen total ──
    print("\n[1/6] Contando registros totales...")
    total = get_total_count()
    report["total_registros"] = total
    print(f"  → Total: {total:,} registros")

    # ── 2. Columnas reales ──
    print("\n[2/6] Descubriendo columnas del dataset...")
    sample_df = get_columns_sample()
    columns = list(sample_df.columns)
    report["columnas"] = columns
    report["num_columnas"] = len(columns)
    print(f"  → {len(columns)} columnas encontradas:")
    for col in columns:
        dtype = sample_df[col].dtype
        sample_val = sample_df[col].iloc[0] if not sample_df[col].isna().all() else "N/A"
        print(f"    • {col} ({dtype}): ej. {str(sample_val)[:60]}")

    # ── 3. Rango de fechas ──
    print("\n[3/6] Buscando rango de fechas...")
    date_info = get_date_range()
    report["fechas"] = date_info
    for key, val in date_info.items():
        if isinstance(val, dict) and "min" in val:
            print(f"  → {key}: {val['min']} → {val['max']}")
        elif key == "columnas_fecha_encontradas":
            print(f"  → Columnas de fecha detectadas: {val}")

    # ── 4. Distribuciones categóricas ──
    print("\n[4/6] Analizando distribuciones categóricas...")
    categorical_candidates = [
        "tipo_usuario", "tipo_tarifa", "departamento",
        "clase_servicio", "mercado", "empresa"
    ]
    report["distribuciones"] = {}
    for col in categorical_candidates:
        if col in columns:
            print(f"\n  Distribución de '{col}':")
            dist = get_categorical_distribution(col)
            if not dist.empty:
                report["distribuciones"][col] = dist.to_dict("records")
                for _, row in dist.head(10).iterrows():
                    print(f"    {row.get(col, '?'):40s} → {int(row.get('conteo', 0)):>12,}")

    # ── 5. Muestra residencial ──
    print("\n[5/6] Obteniendo muestra de usuarios residenciales...")
    residential = get_residential_sample()
    report["muestra_residencial"] = {
        "registros": len(residential),
        "columnas": list(residential.columns),
    }

    # ── 6. Departamentos/Municipios ──
    print("\n[6/6] Cobertura geográfica...")
    for geo_col in ["departamento", "municipio"]:
        if geo_col in columns:
            dist = get_categorical_distribution(geo_col, top_n=50)
            if not dist.empty:
                report[f"cobertura_{geo_col}"] = len(dist)
                print(f"  → {geo_col}: {len(dist)} valores únicos (top 50)")

    # ── Guardar informe ──
    report_path = REPORT_DIR / "01_sui_exploration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ Informe guardado en: {report_path}")

    # ── Guardar muestra como CSV ──
    sample_path = REPORT_DIR / "01_sui_sample.csv"
    sample_df_full = get_residential_sample(limit=500)
    sample_df_full.to_csv(sample_path, index=False)
    print(f"✅ Muestra guardada en: {sample_path}")

    print("\n" + "=" * 70)
    print("Exploración completada.")
    print("=" * 70)

    # ── Resumen ──
    print(f"\nResumen:")
    print(f"  • Registros totales: {total:,}")
    print(f"  • Columnas: {len(columns)}")
    print(f"  • Muestra residencial: {len(residential)} registros")
    print(f"\nSiguiente paso: Revisar el informe y proceder con las demás fuentes.")


if __name__ == "__main__":
    main()
