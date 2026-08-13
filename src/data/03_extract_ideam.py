"""
Paso 03: Extracción IDEAM (Precipitación)
=========================================
Extrae datos meteorológicos de la API del IDEAM.
Fuente: https://www.datos.gov.co/resource/s54a-sgyg.json (Precipitación)

Para evitar desbordar memoria o timeouts, extraemos un lote representativo 
o iteramos por años específicos, agregando localmente a nivel municipio-año.

Regla Leakage (L02): El clima debe agregarse solo con fecha_observacion <= fecha_corte.
Para el MVP anual, asumimos que el clima del año t-1 explica la cosecha en el año t.
Por lo tanto, extraeremos clima, lo agruparemos por año, y luego lo cruzaremos
aplicando un shift (o haciendo join con el año t-1 del panel).
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.soda_client import SodaClient

# ── Configuración ──────────────────────────────────────────────────────────────

IDEAM_PRECIP_ID = "s54a-sgyg"
COLS_IDEAM = "municipio,departamento,fechaobservacion,valorobservado"

def extract_ideam_precip(max_records=500000) -> pd.DataFrame:
    """Extrae un volumen manejable de precipitación para el MVP."""
    print("\n=== IDEAM Precipitación ===")
    client = SodaClient(IDEAM_PRECIP_ID)
    
    # Filtramos años relevantes para nuestro backtest (2018-2023)
    # Fecha en formato ISO8601: "2018-01-01T00:00:00.000"
    where = "fechaobservacion >= '2018-01-01T00:00:00.000'"
    
    df = client.extract_all(
        select=COLS_IDEAM,
        where=where,
        order="fechaobservacion DESC",
        max_records=max_records
    )
    
    if df.empty:
        print("⚠ Sin datos")
        return df
        
    # Limpieza básica
    df["valorobservado"] = pd.to_numeric(df["valorobservado"], errors="coerce")
    df["fechaobservacion"] = pd.to_datetime(df["fechaobservacion"], errors="coerce")
    
    # Estandarizar strings
    for col in ["municipio", "departamento"]:
        df[col] = (
            df[col].astype(str).str.strip().str.upper()
            .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
        )
    
    df["anio"] = df["fechaobservacion"].dt.year
    df = df.dropna(subset=["municipio", "anio", "valorobservado"])
    
    print(f"\nDatos extraídos: {len(df):,} observaciones diarias.")
    return df

def aggregate_climate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega precipitaciones a nivel municipio-año con resolución estacional.
    
    Mejora metodológica: En lugar de solo sumar la precipitación anual 
    (que oculta estrés hídrico temporal), generamos:
    - Precipitación por trimestre (Q1-Q4) para capturar fenología
    - Coeficiente de variación mensual (concentración de lluvia)
    - Máximos días secos consecutivos (proxy de sequía)
    - Intensidad máxima diaria (proxy de eventos extremos/inundación)
    """
    print("Agregando clima con resolución estacional (municipio-año)...")
    
    df = df.copy()
    df["mes"] = df["fechaobservacion"].dt.month
    df["trimestre"] = df["fechaobservacion"].dt.quarter
    
    # ── 1. Agregación anual base ──────────────────────────────────────────────
    agg_anual = df.groupby(["municipio", "anio"]).agg(
        precipitacion_acumulada_mm=("valorobservado", "sum"),
        dias_lluvia=("valorobservado", lambda x: (x > 0).sum()),
        intensidad_max_diaria_mm=("valorobservado", "max"),
        precipitacion_mediana_diaria=("valorobservado", "median"),
    ).reset_index()
    
    # ── 2. Precipitación por trimestre (captura fenología) ────────────────────
    # Q1(Ene-Mar), Q2(Abr-Jun), Q3(Jul-Sep), Q4(Oct-Dic)
    trim_agg = df.groupby(["municipio", "anio", "trimestre"])["valorobservado"].sum().unstack(fill_value=0)
    trim_agg.columns = [f"precip_Q{int(q)}_mm" for q in trim_agg.columns]
    trim_agg = trim_agg.reset_index()
    
    # ── 3. Variabilidad intra-anual (coeficiente de variación mensual) ───────
    # Un CV alto = lluvia concentrada en pocos meses (mala distribución)
    # Un CV bajo = lluvia bien repartida (mejor para cultivos permanentes)
    mensual = df.groupby(["municipio", "anio", "mes"])["valorobservado"].sum().reset_index()
    cv_mensual = mensual.groupby(["municipio", "anio"])["valorobservado"].agg(
        lambda x: x.std() / x.mean() if x.mean() > 0 else 0
    ).reset_index()
    cv_mensual.columns = ["municipio", "anio", "cv_precipitacion_mensual"]
    
    # ── 4. Máximos días secos consecutivos (proxy de sequía) ─────────────────
    def max_dias_secos(group):
        """Calcula el máximo de días consecutivos sin lluvia significativa."""
        sorted_g = group.sort_values("fechaobservacion")
        es_seco = (sorted_g["valorobservado"] <= 1.0).astype(int)  # <1mm = día seco
        # Contar rachas consecutivas
        if len(es_seco) == 0:
            return 0
        cambios = es_seco.diff().ne(0).cumsum()
        rachas = es_seco.groupby(cambios).agg(["sum", "count"])
        rachas_secas = rachas[rachas["sum"] == rachas["count"]]["count"]
        return int(rachas_secas.max()) if len(rachas_secas) > 0 else 0
    
    print("  Calculando máx. días secos consecutivos (puede tardar)...")
    dias_secos = df.groupby(["municipio", "anio"]).apply(
        max_dias_secos, include_groups=False
    ).reset_index()
    dias_secos.columns = ["municipio", "anio", "max_dias_secos_consecutivos"]
    
    # ── 5. Ratio de concentración de lluvia ──────────────────────────────────
    # ¿Qué fracción de la lluvia anual cayó en el trimestre más lluvioso?
    if len(trim_agg.columns) > 2:  # tiene al menos un trimestre
        precip_cols = [c for c in trim_agg.columns if c.startswith("precip_Q")]
        trim_agg["precip_max_trimestre"] = trim_agg[precip_cols].max(axis=1)
        trim_agg["precip_total_trim"] = trim_agg[precip_cols].sum(axis=1)
        trim_agg["ratio_concentracion_lluvia"] = (
            trim_agg["precip_max_trimestre"] / trim_agg["precip_total_trim"].replace(0, 1)
        )
        trim_agg = trim_agg.drop(columns=["precip_max_trimestre", "precip_total_trim"])
    
    # ── Unir todo ─────────────────────────────────────────────────────────────
    res = agg_anual
    res = res.merge(trim_agg, on=["municipio", "anio"], how="left")
    res = res.merge(cv_mensual, on=["municipio", "anio"], how="left")
    res = res.merge(dias_secos, on=["municipio", "anio"], how="left")
    
    res["dias_lluvia"] = res["dias_lluvia"].fillna(0).astype(int)
    
    # Filtro de completitud (Limpieza #1): 
    # Requerimos al menos 30 observaciones diarias por municipio-año.
    # Nota: El IDEAM puede tener múltiples estaciones por municipio, por lo que
    # el conteo real de "días" puede ser mayor que 365. Un umbral de 30 asegura
    # que tengamos al menos ~1 mes de datos reales sin eliminar toda la cobertura.
    obs_count = df.groupby(["municipio", "anio"]).size().reset_index(name="n_observaciones")
    res = res.merge(obs_count, on=["municipio", "anio"])
    
    n_antes = len(res)
    res = res[res["n_observaciones"] >= 30].copy()
    n_eliminados = n_antes - len(res)
    if n_eliminados > 0:
        print(f"  Filtro completitud: {n_eliminados} filas eliminadas (< 30 obs)")
    res = res.drop(columns=["n_observaciones"])
    
    n_features_clima = len([c for c in res.columns if c not in ["municipio", "anio"]])
    print(f"  Panel clima resultante: {len(res):,} filas × {n_features_clima} features climáticas.")
    return res

def main():
    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    
    # 1. Extraer un lote manejable para evitar el timeout del servidor SODA
    df_raw = extract_ideam_precip(max_records=150000)
    if df_raw.empty:
        return
        
    df_raw.to_csv(raw_dir / "ideam_precip_raw.csv", index=False)
    
    # 2. Agregar
    df_agg = aggregate_climate(df_raw)
    df_agg.to_csv(proc_dir / "ideam_municipio_anio.csv", index=False)
    print(f"✓ Guardado: {proc_dir / 'ideam_municipio_anio.csv'}")

if __name__ == "__main__":
    main()
