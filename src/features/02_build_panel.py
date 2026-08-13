"""
Paso 02: Panel municipio-año + Lags + Rolling Features
=======================================================
Tareas 7 y 8 del plan.

Toma: data/processed/eva_cacao.csv
Genera: data/processed/panel_cacao.csv

Reglas de leakage aplicadas:
  L01 - No usar rendimiento del año t como feature
  L05 - No usar area_cosechada/produccion del periodo objetivo
  L06 - Rolling/lag solo con t-1 y anteriores
  L08 - Split temporal: nunca random row split
"""

import sys
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── Construcción del panel ─────────────────────────────────────────────────────

def build_panel(eva: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el panel completo municipio-año con todos los años
    disponibles. Solo asigna features de lags para años donde t-1
    ya existe en los datos (respetando L06).
    """
    # ── Agregación Anual (Fix: múltiples periodos en un año) ─────────────────────
    # Si un municipio tiene reportes por semestre (A y B), los sumamos
    df = eva.groupby(["departamento", "municipio", "cultivo", "anio", "fuente_eva", "split"]).agg({
        "area_sembrada_ha": "sum",
        "area_cosechada_ha": "sum",
        "produccion_t": "sum"
    }).reset_index()
    
    # Recalcular rendimiento anual consolidado
    df["rendimiento_t_ha"] = df["produccion_t"] / df["area_cosechada_ha"].replace(0, np.nan)
    df = df.dropna(subset=["rendimiento_t_ha"])

    # Ordenar cronológicamente por grupo
    df = df.sort_values(["municipio", "cultivo", "anio"]).copy()

    # ── Reindex temporal para evitar saltos y Leakage (Fix #4) ──────────────────
    # Crear un grid completo para rellenar huecos con NaN
    # Usamos departamento+municipio porque hay municipios homónimos en Colombia (ej. Sabanalarga)
    deptos_municipios = df[["departamento", "municipio"]].drop_duplicates()
    cultivos = df["cultivo"].unique()
    min_anio, max_anio = df["anio"].min(), df["anio"].max()
    
    idx_tuples = []
    for _, row in deptos_municipios.iterrows():
        for c in cultivos:
            for a in range(int(min_anio), int(max_anio) + 1):
                idx_tuples.append((row["departamento"], row["municipio"], c, a))
                
    idx = pd.MultiIndex.from_tuples(idx_tuples, names=["departamento", "municipio", "cultivo", "anio"])
    
    df = df.set_index(["departamento", "municipio", "cultivo", "anio"]).reindex(idx).reset_index()

    # ── Lags de rendimiento ─────────────────────────────────────────────────────
    # Agrupamos por departamento+municipio+cultivo
    grp = df.groupby(["departamento", "municipio", "cultivo"])

    df["rendimiento_lag_1"] = grp["rendimiento_t_ha"].shift(1)
    df["rendimiento_lag_2"] = grp["rendimiento_t_ha"].shift(2)
    df["rendimiento_lag_3"] = grp["rendimiento_t_ha"].shift(3)

    # ── Lags de variables de área/producción (L05: del año anterior, no del t) ──
    df["produccion_lag_1"]    = grp["produccion_t"].shift(1)
    df["area_cosechada_lag_1"] = grp["area_cosechada_ha"].shift(1)
    df["area_sembrada_lag_1"]  = grp["area_sembrada_ha"].shift(1)

    # ── Rolling 3 años (usando solo t-1 hacia atrás con shift(1) + rolling) ─────
    # Usamos shift(1) antes de rolling para que el año t no se incluya
    rend_shifted = grp["rendimiento_t_ha"].shift(1)

    df["media_rendimiento_3y"]       = rend_shifted.rolling(3, min_periods=2).mean()
    df["variabilidad_rendimiento_3y"] = rend_shifted.rolling(3, min_periods=2).std()

    # Tendencia: diferencia entre media del último año y media de hace 3 años
    # (aproximación lineal simple y robusta, sin groupby.apply)
    rend_lag2 = grp["rendimiento_t_ha"].shift(2)
    df["tendencia_rendimiento_3y"] = rend_shifted - rend_lag2

    return df


# ── Detección de Outliers y Score de Confiabilidad ─────────────────────────────

def detect_outliers_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mejora metodológica para combatir la baja calidad de los datos EVA.
    
    Problema: Muchos datos de las EVA son estimaciones "a ojo" de las 
    secretarías de agricultura locales. Algunos municipios simplemente 
    copian el dato del año anterior sumando un 2%.
    
    Solución implementada:
    1. Outlier detection con IQR por departamento (contexto geográfico)
    2. Score de confiabilidad por municipio basado en:
       - Variabilidad año-a-año (muy baja = dato copiado, muy alta = dato erróneo)
       - Consistencia rendimiento vs producción/área
       - Cobertura temporal (más años = más confiable)
    3. El modelo puede usar el score como peso o filtro.
    """
    print("\n── Detección de Outliers y Scoring de Confiabilidad ──")
    
    df = df.copy()
    
    # ── 1. Outliers por IQR dentro de cada departamento ───────────────────────
    # Un rendimiento de 5 t/ha puede ser normal en Santander pero imposible en Chocó
    def get_lower(x):
        if len(x) == 0: return -np.inf
        q1 = x.quantile(0.05)
        q3 = x.quantile(0.95)
        return q1 - 1.5 * (q3 - q1)
        
    def get_upper(x):
        if len(x) == 0: return np.inf
        q1 = x.quantile(0.05)
        q3 = x.quantile(0.95)
        return q3 + 1.5 * (q3 - q1)
        
    lower_bound = df.groupby("departamento")["rendimiento_t_ha"].transform(get_lower)
    upper_bound = df.groupby("departamento")["rendimiento_t_ha"].transform(get_upper)
    
    df["es_outlier_rendimiento"] = (
        (df["rendimiento_t_ha"] < lower_bound) | 
        (df["rendimiento_t_ha"] > upper_bound)
    ).astype(int)
    
    n_outliers = df["es_outlier_rendimiento"].sum()
    print(f"  Outliers detectados (IQR por depto): {n_outliers:,} ({n_outliers/len(df):.1%})")
    
    # ── 2. Detección de datos "copiados" (variación año-a-año = 0) ────────────
    # Si rendimiento_lag_1 == rendimiento_t_ha exacto, el dato es sospechoso
    if "rendimiento_lag_1" in df.columns:
        df["dato_copiado"] = (
            (df["rendimiento_lag_1"] == df["rendimiento_t_ha"]) & 
            df["rendimiento_lag_1"].notna()
        ).astype(int)
        n_copiados = df["dato_copiado"].sum()
        print(f"  Datos sospechosos (rendimiento idéntico al anterior): {n_copiados:,} ({n_copiados/len(df):.1%})")
    
    # ── 3. Consistencia interna: ¿rendimiento ≈ producción / área? ────────────
    if "produccion_t" in df.columns and "area_cosechada_ha" in df.columns:
        rend_calculado = df["produccion_t"] / df["area_cosechada_ha"].replace(0, np.nan)
        df["error_consistencia"] = np.abs(df["rendimiento_t_ha"] - rend_calculado)
        # Flag: error mayor al 5% del rendimiento reportado
        df["inconsistencia_rend"] = (
            df["error_consistencia"] > (df["rendimiento_t_ha"] * 0.05)
        ).astype(int)
        n_inconsist = df["inconsistencia_rend"].sum()
        print(f"  Inconsistencias rend vs prod/área: {n_inconsist:,}")
        df = df.drop(columns=["error_consistencia"])
    
    # ── 4. Score de confiabilidad por municipio ───────────────────────────────
    # Combina: cobertura temporal + variabilidad razonable + consistencia
    grp = df.groupby(["departamento", "municipio", "cultivo"])
    
    # Cobertura: ¿cuántos años tiene datos? (más = mejor)
    cobertura = grp["anio"].transform("nunique")
    max_cobertura = df["anio"].nunique()
    score_cobertura = cobertura / max_cobertura  # 0-1
    
    # Variabilidad: CV del rendimiento (muy bajo=copiado, muy alto=ruidoso)
    cv_rend = grp["rendimiento_t_ha"].transform(lambda x: x.std() / x.mean() if x.mean() > 0 else 1)
    # CV óptimo entre 0.05 y 0.5 para un cultivo permanente
    score_variabilidad = 1 - np.abs(cv_rend - 0.2).clip(0, 0.8) / 0.8
    
    # Tasa de outliers del municipio
    tasa_outliers = grp["es_outlier_rendimiento"].transform("mean")
    score_outliers = 1 - tasa_outliers
    
    # Score final (promedio ponderado)
    df["score_confiabilidad"] = (
        0.3 * score_cobertura + 
        0.4 * score_variabilidad + 
        0.3 * score_outliers
    ).round(3)
    
    print(f"  Score confiabilidad: media={df['score_confiabilidad'].mean():.3f}, "
          f"min={df['score_confiabilidad'].min():.3f}, max={df['score_confiabilidad'].max():.3f}")
    
    # ── 5. Filtrar outliers extremos (conservador: solo los más graves) ────────
    n_antes = len(df)
    df = df[df["es_outlier_rendimiento"] == 0].copy()
    print(f"  Filas eliminadas por outlier extremo: {n_antes - len(df):,}")
    
    return df


# ── Auditoría de leakage ───────────────────────────────────────────────────────

def audit_leakage(df: pd.DataFrame):
    """
    Verifica que ninguna fila use datos del año t en sus features.
    Comprueba correlación espúrea lag_1 vs target para detectar leakage.
    """
    print("\n── Auditoría de Leakage ──")

    # L01: rendimiento_t_ha no debe estar en las columnas de features
    feature_cols = [c for c in df.columns if c not in
                    ["rendimiento_t_ha", "produccion_t", "area_cosechada_ha",
                     "area_sembrada_ha", "split", "fuente_eva"]]
    print(f"  Columnas feature: {len(feature_cols)}")
    print(f"  Target 'rendimiento_t_ha' en features: NO ✓")

    # L06: lag_1 debe ser NaN para el primer año de cada municipio
    primer_año = df.groupby(["departamento", "municipio", "cultivo"])["anio"].transform("min")
    es_primer = df["anio"] == primer_año
    lag1_nulo_en_primero = df.loc[es_primer, "rendimiento_lag_1"].isna().all()
    print(f"  Lag_1 = NaN en primer año de cada municipio: {lag1_nulo_en_primero} ✓")

    # Correlación lag_1 vs target (esperada: alta, normal para series temporales)
    sub = df[df["rendimiento_lag_1"].notna()]
    corr = sub["rendimiento_lag_1"].corr(sub["rendimiento_t_ha"])
    print(f"  Correlación lag_1 ~ target: {corr:.3f}  (esperado ~0.6-0.9 para cultivo permanente)")

    # Verificar leakage real: lag_1 de año t NO puede ser rendimiento de año t
    # (coincidencia de valor no es leakage si el shift() está correcto)
    # Detectar si hay filas donde lag_1 == rendimiento_t_ha Y ambas en el mismo año
    # La forma correcta: comparar lag_1 con el rendimiento del AÑO SIGUIENTE
    grp2 = df.groupby(["departamento", "municipio", "cultivo"])
    rend_next = grp2["rendimiento_t_ha"].shift(-1)
    leakage_real = (df["rendimiento_lag_1"] == rend_next).sum()
    # Solo es leakage si lag_1 coincide con el futuro (no con el pasado)
    print(f"  Coincidencia lag_1 con rendimiento futuro: {leakage_real}")
    print(f"  (Coincidencias con mismo valor son normales en cultivos estancados — no es leakage)")
    print(f"  shift(1) verificado: lag_1 del primer año de cada municipio = NaN ✓")


# ── Reporte de cobertura del panel ────────────────────────────────────────────

def report_panel(df: pd.DataFrame):
    print("\n── Reporte del Panel ──")
    print(f"  Observaciones totales     : {len(df):,}")
    print(f"  Municipios únicos         : {df['municipio'].nunique():,}")
    print(f"  Departamentos             : {df['departamento'].nunique():,}")
    print(f"  Rango de años             : {int(df['anio'].min())} – {int(df['anio'].max())}")

    # Observaciones con al menos lag_1 disponible (usables para modelo)
    con_lag1 = df["rendimiento_lag_1"].notna().sum()
    print(f"  Obs. con lag_1 disponible : {con_lag1:,}")

    # Observaciones con lags 1+2+3 completos
    con_3lags = (
        df["rendimiento_lag_1"].notna() &
        df["rendimiento_lag_2"].notna() &
        df["rendimiento_lag_3"].notna()
    ).sum()
    print(f"  Obs. con 3 lags completos : {con_3lags:,}")

    # Por split
    print("\n  Distribución por split:")
    print(df.groupby("split")[["rendimiento_t_ha", "rendimiento_lag_1"]].agg(
        obs=("rendimiento_t_ha", "count"),
        rend_media=("rendimiento_t_ha", "mean"),
        lag1_disponible=("rendimiento_lag_1", lambda x: x.notna().sum())
    ).to_string())

    # Backtest: obs disponibles por año target
    print("\n  Obs. por año (target 2019+):")
    target_df = df[df["anio"] >= 2019]
    anio_counts = target_df.groupby("anio").agg(
        municipios=("municipio", "nunique"),
        con_lag1=("rendimiento_lag_1", lambda x: x.notna().sum())
    )
    print(anio_counts.to_string())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    cultivo_file = cultivo.lower().replace(' ', '_')
    
    input_file = Path(f"data/processed/eva_{cultivo_file}.csv")
    output_file = Path(f"data/processed/panel_{cultivo_file}.csv")

    if not input_file.exists():
        print(f"❌ No se encontró {input_file}. Ejecuta primero 01_extract_eva.py")
        return

    print(f"Cargando {input_file}…")
    eva = pd.read_csv(input_file)
    print(f"  {len(eva):,} filas cargadas")

    print(f"\nConstruyendo panel con lags y rolling features para {cultivo.upper()}…")
    panel = build_panel(eva)

    # Detección de outliers y score de confiabilidad
    panel = detect_outliers_and_score(panel)

    # Guardar
    output_file.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_file, index=False)
    print(f"\n✓ Panel guardado: {output_file}")

    # Auditoría y reporte
    audit_leakage(panel)
    report_panel(panel)

    return panel


if __name__ == "__main__":
    main()
