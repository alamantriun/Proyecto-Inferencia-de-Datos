"""
Paso 08: Extracción Precio Internacional del Cacao (FRED/ICE)
=============================================================
Extrae el precio global del Cacao (USD/Ton Métrica) desde la Reserva
Federal de EE.UU. (FRED), serie PCOCOUSDM.

Esta es una variable MACRO (aplica a todos los municipios por igual).
Regla Leakage (L02): El precio promedio del año t se usa como feature
para el rendimiento del año t+1. El precio t refleja las condiciones de
mercado que motivaron (o desmotivaron) la inversión del productor.

Features derivadas:
- precio_cacao_usd: Promedio anual del precio internacional.
- cambio_precio_pct: Variación porcentual interanual del precio.
  Si subió → el agricultor invierte más → sube rendimiento futuro.
  Si bajó → el agricultor abandona → baja rendimiento futuro.
"""

import pandas as pd
import numpy as np
import csv
import io
import yaml
from pathlib import Path


def extract_international_prices(fred_id: str, cultivo: str) -> pd.DataFrame:
    """Extrae precios mensuales de FRED y agrega a promedios anuales."""
    print(f"\n=== Precio Internacional para {cultivo.upper()} (FRED: {fred_id}) ===")

    fred_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}&cosd=2005-01-01&coed=2025-12-01"

    try:
        import urllib.request
        req = urllib.request.Request(fred_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        print(f"⚠ Error descargando de FRED: {e}")
        print("  Intentando cargar caché local o generando fallback seguro...")
        cache = Path(f"data/raw/fred_{cultivo}_monthly.csv")
        if cache.exists():
            text = cache.read_text()
        else:
            # Fix Error #6: Fallback parametrizado por cultivo
            # Cada cultivo tiene su propio nivel de precio histórico aproximado
            print("  ⚠ Fallback: Generando datos sintéticos de precios internacionales...")
            anios = list(range(2007, 2026))
            
            # Precios base aproximados (USD/Ton) por cultivo
            precios_base = {
                "cacao": {"base": 2500, 2023: 3200, 2024: 7500, 2025: 6000},
                "cafe":  {"base": 3000, 2023: 3500, 2024: 4200, 2025: 4000},
                "arroz": {"base": 400,  2023: 500,  2024: 450,  2025: 430},
                "platano": {"base": 800, 2023: 900, 2024: 950, 2025: 920},
            }
            cultivo_key = cultivo.lower().replace(' ', '_')
            precios_cult = precios_base.get(cultivo_key, {"base": 1000, 2023: 1200, 2024: 1300, 2025: 1250})
            
            rows = []
            for a in anios:
                p = precios_cult.get(a, precios_cult["base"])
                rows.append({"date": f"{a}-01-01", "price": p})
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df["anio"] = df["date"].dt.year
            
            # Feature derivada
            annual = df.groupby("anio")["price"].agg(
                precio_internacional_usd="mean",
                precio_internacional_max="max",
                precio_internacional_min="min",
                volatilidad_precio="std"
            ).reset_index()
            annual["cambio_precio_pct"] = annual["precio_internacional_usd"].pct_change() * 100
            annual["rango_precio_ratio"] = 1.0
            annual = annual.fillna(0)
            return annual

    # Parsear CSV
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        date_str = row.get("observation_date", row.get("DATE", ""))
        price_str = row.get(fred_id, row.get("VALUE", ""))
        if date_str and price_str and price_str != ".":
            rows.append({"date": date_str, "price": float(price_str)})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["anio"] = df["date"].dt.year

    # Guardar caché de datos mensuales
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / f"fred_{cultivo}_monthly.csv", index=False)
    print(f"  ✓ Caché mensual: {len(df)} registros ({df['anio'].min()}-{df['anio'].max()})")

    # Agregar a promedio anual
    annual = df.groupby("anio")["price"].agg(
        precio_internacional_usd="mean",
        precio_internacional_max="max",
        precio_internacional_min="min",
        volatilidad_precio="std"
    ).reset_index()

    # Feature derivada: Cambio porcentual interanual
    annual = annual.sort_values("anio")
    annual["cambio_precio_pct"] = annual["precio_internacional_usd"].pct_change() * 100

    # Ratio max/min como proxy de inestabilidad del mercado
    annual["rango_precio_ratio"] = annual["precio_internacional_max"] / annual["precio_internacional_min"].replace(0, np.nan)

    print(f"\n  Precios Anuales Promedio (USD/Ton):")
    for _, r in annual.iterrows():
        cambio = f"({r['cambio_precio_pct']:+.1f}%)" if pd.notna(r["cambio_precio_pct"]) else ""
        print(f"    {int(r['anio'])}: ${r['precio_internacional_usd']:,.0f} {cambio}")

    return annual


def main():
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    fred_id = config["fuentes"].get("fred_precios", {}).get(cultivo)

    if not fred_id:
        print(f"⚠ No hay un FRED ID configurado para el cultivo: {cultivo}. Saltando extracción.")
        return

    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    annual = extract_international_prices(fred_id, cultivo)
    if annual.empty:
        return

    file_name = f"precio_internacional_{cultivo}.csv"
    annual.to_csv(proc_dir / file_name, index=False)
    print(f"\n✓ Guardado: {proc_dir / file_name}")

if __name__ == "__main__":
    main()
