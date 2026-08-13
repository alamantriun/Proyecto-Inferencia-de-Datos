"""
Paso 07: Extracción FINAGRO (Crédito Agropecuario para Cacao)
=============================================================
Extrae los desembolsos de crédito agropecuario para cacao, agregados por
municipio y año de inversión. Fuente: datos.gov.co/resource/w3uf-w9ey

Regla Leakage (L02): El crédito desembolsado en el año t se usa como
feature para predecir el rendimiento del año t+1. El crédito del año t
refleja las decisiones de inversión tomadas ANTES de la cosecha t+1.
"""

import sys
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.soda_client import SodaClient

FINAGRO_ID = "w3uf-w9ey"

def extract_finagro(cultivo: str) -> pd.DataFrame:
    """Extrae y agrega créditos por municipio-año."""
    print(f"\n=== FINAGRO Crédito Agropecuario ({cultivo.upper()}) ===")
    client = SodaClient(FINAGRO_ID)

    # Query agregado: suma de inversión y número de operaciones por municipio-año
    select = (
        "a_o as anio, "
        "municipio_inversion as municipio, "
        "sum(valor_inversion) as credito_total, "
        "sum(colocacion) as colocacion_total, "
        "count(*) as num_operaciones_credito"
    )
    where = f"UPPER(destino_de_credito) like '%{cultivo.upper()}%'"
    group = "a_o, municipio_inversion"

    df = client.extract_all(
        select=select,
        where=where,
        group=group,
        order="anio",
        batch=50000
    )

    if df.empty:
        print(f"⚠ Sin datos de crédito para {cultivo}")
        return df

    # Limpiar tipos
    for col in ["credito_total", "colocacion_total", "num_operaciones_credito"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")

    # Estandarizar municipio
    df["municipio"] = (
        df["municipio"].astype(str).str.strip().str.upper()
        .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
    )

    # Añadir: crédito per cápita de operación (intensidad de inversión)
    df["credito_promedio_operacion"] = (
        df["credito_total"] / df["num_operaciones_credito"].replace(0, np.nan)
    )

    # Logaritmo del crédito total
    df["log_credito_total"] = np.log1p(df["credito_total"])

    # Limpieza Financiera (#2): Deflactar a pesos constantes de 2024
    # Asumimos una inflación histórica promedio de ~5.5% para Colombia
    df["factor_inflacion"] = 1.055 ** (2024 - df["anio"])
    df["credito_real_2024"] = df["credito_total"] * df["factor_inflacion"]
    df["credito_promedio_operacion_real"] = df["credito_promedio_operacion"] * df["factor_inflacion"]
    df["log_credito_real"] = np.log1p(df["credito_real_2024"])
    
    # Reemplazar variables originales en el dataset que usará el modelo
    df["credito_total"] = df["credito_real_2024"]
    df["credito_promedio_operacion"] = df["credito_promedio_operacion_real"]
    df["log_credito_total"] = df["log_credito_real"]
    df = df.drop(columns=["factor_inflacion", "credito_real_2024", "credito_promedio_operacion_real", "log_credito_real"])
    
    df = df.dropna(subset=["municipio", "anio"])

    print(f"\nDatos extraídos: {len(df):,} filas (municipio-año).")
    print(f"Años: {df['anio'].min()} – {df['anio'].max()}")
    print(f"Municipios: {df['municipio'].nunique()}")

    return df


def main():
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")

    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    df = extract_finagro(cultivo)
    if df.empty:
        return

    cultivo_file = cultivo.lower().replace(" ", "_")
    
    df.to_csv(raw_dir / f"finagro_{cultivo_file}_credito.csv", index=False)

    # Guardar versión procesada lista para join
    file_name = f"finagro_municipio_anio_{cultivo_file}.csv"
    df.to_csv(proc_dir / file_name, index=False)
    print(f"✓ Guardado: {proc_dir / file_name}")


if __name__ == "__main__":
    main()
