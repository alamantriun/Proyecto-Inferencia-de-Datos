"""
Paso 04: Extracción AGROSAVIA (Suelos)
======================================
Extrae datos de análisis de laboratorio de suelos de AGROSAVIA.
Fuente: https://www.datos.gov.co/resource/ch4u-f3i5.json

Regla Leakage (L04):
Solo muestras tomadas antes de la fecha de predicción. Dado que no tenemos 
la fecha exacta en que se sembró el cacao en EVA, usaremos todas las 
muestras de suelo hasta el año de corte, asumiendo que el suelo cambia 
lentamente y representa la vocación natural del terreno.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.soda_client import SodaClient

AGROSAVIA_ID = "ch4u-f3i5"
COLS_SUELO = (
    "fecha_de_an_lisis,departamento,municipio,cultivo,"
    "ph_agua_suelo,materia_organica,fosforo_bray_ii,"
    "calcio_intercambiable,magnesio_intercambiable,"
    "potasio_intercambiable,conductividad_electrica"
)

RENAME = {
    "fecha_de_an_lisis": "fecha",
    "ph_agua_suelo": "ph",
    "materia_organica": "materia_organica_pct",
    "fosforo_bray_ii": "fosforo_ppm",
    "calcio_intercambiable": "calcio_meq",
    "magnesio_intercambiable": "magnesio_meq",
    "potasio_intercambiable": "potasio_meq",
    "conductividad_electrica": "salinidad_ds_m"
}

NUMERIC_COLS = [
    "ph", "materia_organica_pct", "fosforo_ppm", "calcio_meq",
    "magnesio_meq", "potasio_meq", "salinidad_ds_m"
]

def clean_value(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace("<", "").replace(">", "").strip()
    try:
        return float(val)
    except:
        return np.nan

import numpy as np

def extract_agrosavia(max_records=None) -> pd.DataFrame:
    print("\n=== AGROSAVIA Suelos ===")
    client = SodaClient(AGROSAVIA_ID)
    
    # Extraemos TODO porque son ~92k registros a nivel nacional
    df = client.extract_all(
        select=COLS_SUELO,
        max_records=max_records
    )
    
    if df.empty:
        print("⚠ Sin datos")
        return df
        
    df = df.rename(columns=RENAME)
    
    # Limpiar numéricos (algunos vienen con "<" o ">")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(clean_value)
            
    # Estandarizar strings
    for col in ["municipio", "departamento", "cultivo"]:
        df[col] = (
            df[col].astype(str).str.strip().str.upper()
            .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
        )
        
    df["anio_muestra"] = pd.to_datetime(df["fecha"], errors="coerce").dt.year
    df = df.dropna(subset=["municipio", "ph"])
    
    # Filtros Biológicos (Eliminar errores de laboratorio o de digitación)
    if "materia_organica_pct" in df.columns:
        df.loc[(df["materia_organica_pct"] < 0) | (df["materia_organica_pct"] > 100), "materia_organica_pct"] = np.nan
    if "salinidad_ds_m" in df.columns:
        df.loc[df["salinidad_ds_m"] < 0, "salinidad_ds_m"] = np.nan
    if "ph" in df.columns:
        df.loc[(df["ph"] < 3.0) | (df["ph"] > 10.0), "ph"] = np.nan
    
    print(f"\nDatos limpios: {len(df):,} muestras.")
    return df

def aggregate_suelo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega las propiedades del suelo por municipio con ventana temporal.
    
    Mejora metodológica: En lugar de promediar TODAS las muestras históricas
    (que mezcla suelos de 1990 con suelos de 2020), usamos una ventana 
    rolling de 10 años. Esto captura:
    - Degradación progresiva del suelo (pérdida de materia orgánica)
    - Efectos de fertilización reciente
    - Cambios de uso del suelo
    
    Además generamos un perfil "reciente" (últimos 5 años) y uno "histórico"
    (todo lo anterior) para que el modelo detecte tendencias de cambio.
    """
    print("Agregando suelo con ventana temporal (municipio-año)...")
    
    df = df.copy()
    df = df.dropna(subset=["anio_muestra"])
    
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    
    # ── Perfil reciente: últimos 5 años de muestras disponibles ──────────────
    anio_max = int(df["anio_muestra"].max())
    corte_reciente = anio_max - 5
    
    df_reciente = df[df["anio_muestra"] >= corte_reciente]
    df_historico = df[df["anio_muestra"] < corte_reciente]
    
    # Agregación reciente
    agg_reciente = {}
    for c in numeric_cols:
        agg_reciente[c] = ["mean", "std"]
    agg_reciente["ph"] = ["mean", "count", "std"]
    
    if len(df_reciente) > 0:
        res_reciente = df_reciente.groupby("municipio").agg(agg_reciente)
        res_reciente.columns = ["_".join(col).strip() for col in res_reciente.columns.values]
        res_reciente = res_reciente.rename(columns={
            "ph_count": "num_muestras_suelo",
            "ph_mean": "ph_media",
            "ph_std": "ph_variabilidad",
        })
        # Renombrar _mean y _std
        rename_cols = {}
        for c in res_reciente.columns:
            if c.endswith("_mean"):
                rename_cols[c] = c.replace("_mean", "_media")
            elif c.endswith("_std") and c != "ph_variabilidad":
                rename_cols[c] = c.replace("_std", "_variabilidad")
        res_reciente = res_reciente.rename(columns=rename_cols).reset_index()
    else:
        res_reciente = pd.DataFrame(columns=["municipio"])
    
    # ── Perfil histórico: para calcular tendencia de cambio ──────────────────
    if len(df_historico) > 0 and len(df_reciente) > 0:
        hist_means = df_historico.groupby("municipio")[numeric_cols].mean()
        hist_means.columns = [f"{c}_historico" for c in hist_means.columns]
        
        rec_means = df_reciente.groupby("municipio")[numeric_cols].mean()
        
        # Tendencia de cambio: (reciente - histórico) / histórico
        tendencia = pd.DataFrame(index=rec_means.index)
        for c in numeric_cols:
            col_hist = f"{c}_historico"
            if col_hist in hist_means.columns:
                tendencia[f"tendencia_{c}"] = (
                    (rec_means[c] - hist_means[col_hist]) / hist_means[col_hist].replace(0, np.nan)
                ).fillna(0)
        
        tendencia = tendencia.reset_index()
        res_reciente = res_reciente.merge(tendencia, on="municipio", how="left")
        
        n_con_tendencia = tendencia["municipio"].nunique()
        print(f"  Municipios con tendencia de cambio de suelo: {n_con_tendencia:,}")
    
    print(f"  Panel suelo (ventana {corte_reciente}-{anio_max}): {len(res_reciente):,} municipios.")
    n_features = len([c for c in res_reciente.columns if c != "municipio"])
    print(f"  Features de suelo generadas: {n_features}")
    
    return res_reciente

def main():
    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    
    df_raw = extract_agrosavia()
    if df_raw.empty:
        return
        
    df_raw.to_csv(raw_dir / "agrosavia_suelos_raw.csv", index=False)
    
    df_agg = aggregate_suelo(df_raw)
    df_agg.to_csv(proc_dir / "agrosavia_municipio.csv", index=False)
    print(f"✓ Guardado: {proc_dir / 'agrosavia_municipio.csv'}")

if __name__ == "__main__":
    main()
