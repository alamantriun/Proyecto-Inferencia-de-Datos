"""
Paso 06: Construcción del Model Mart
====================================
Une el panel base (EVA + Lags) con las variables externas:
- Clima IDEAM
- Suelo AGROSAVIA
- Aptitud UPRA
- Crédito FINAGRO
- Precio Internacional del Cacao (FRED)

Regla Leakage (L02): El clima de IDEAM se une con un lag de 1 año.
Es decir, el clima registrado en 2018 explica el rendimiento de 2019.
AGROSAVIA y UPRA se consideran variables estructurales del territorio.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def build_model_mart():
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    cultivo_file = cultivo.lower().replace(' ', '_')

    print(f"=== Construcción del Model Mart ({cultivo.upper()}) ===")
    
    proc_dir = Path("data/processed")
    
    # 1. Cargar panel base (ya tiene splits y lags de EVA)
    print("Cargando panel de EVA...")
    df = pd.read_csv(proc_dir / f"panel_{cultivo_file}.csv")
    
    # 2. Unir Clima IDEAM (Lag 1 año)
    # Fix Error #4: Usar departamento+municipio como llave para evitar
    # que municipios homónimos (ej. SABANALARGA) reciban lluvia equivocada
    if (proc_dir / "ideam_municipio_anio.csv").exists():
        print("Uniendo IDEAM (Clima)...")
        ideam = pd.read_csv(proc_dir / "ideam_municipio_anio.csv")
        # El clima del año t explica el rendimiento del año t+1
        ideam["anio_target"] = ideam["anio"] + 1
        ideam = ideam.drop(columns=["anio"])
        
        # Determinar llaves de merge según columnas disponibles en IDEAM
        if "departamento" in ideam.columns:
            merge_keys_left = ["departamento", "municipio", "anio"]
            merge_keys_right = ["departamento", "municipio", "anio_target"]
        else:
            merge_keys_left = ["municipio", "anio"]
            merge_keys_right = ["municipio", "anio_target"]
        
        df = df.merge(
            ideam,
            left_on=merge_keys_left,
            right_on=merge_keys_right,
            how="left"
        ).drop(columns=["anio_target"])
        
    # 3. Unir Suelos AGROSAVIA
    if (proc_dir / "agrosavia_municipio.csv").exists():
        print("Uniendo AGROSAVIA (Suelos)...")
        suelo = pd.read_csv(proc_dir / "agrosavia_municipio.csv")
        df = df.merge(suelo, on="municipio", how="left")
        
    # 4. Unir Aptitud UPRA
    if (proc_dir / f"upra_aptitud_{cultivo_file}.csv").exists():
        print("Uniendo UPRA (Aptitud)...")
        upra = pd.read_csv(proc_dir / f"upra_aptitud_{cultivo_file}.csv")
        df = df.merge(upra, on="municipio", how="left")
        
    # 4.5 Unir Finagro (Crédito, Lag 1 año)
    if (proc_dir / f"finagro_municipio_anio_{cultivo_file}.csv").exists():
        print("Uniendo FINAGRO (Crédito agropecuario)...")
        finagro = pd.read_csv(proc_dir / f"finagro_municipio_anio_{cultivo_file}.csv")
        # El crédito del año t impacta el rendimiento del año t+1
        finagro["anio_target"] = finagro["anio"] + 1
        finagro = finagro.drop(columns=["anio"])
        
        df = df.merge(
            finagro,
            left_on=["municipio", "anio"],
            right_on=["municipio", "anio_target"],
            how="left"
        ).drop(columns=["anio_target"])
        
        # Llenar nulos de crédito con 0 (asumimos que si no hay registro, no hubo crédito)
        cols_credito = ["credito_total", "colocacion_total", "num_operaciones_credito", "credito_promedio_operacion", "log_credito_total"]
        for c in cols_credito:
            if c in df.columns:
                df[c] = df[c].fillna(0)

    # 4.6 Unir Precio Internacional (FRED, Lag 1 año)
    if (proc_dir / f"precio_internacional_{cultivo_file}.csv").exists():
        print("Uniendo Precio Internacional (FRED)...")
        precio = pd.read_csv(proc_dir / f"precio_internacional_{cultivo_file}.csv")
        # El precio del año t influye en las decisiones de cosecha del año t+1
        precio["anio_target"] = precio["anio"] + 1
        precio = precio.drop(columns=["anio"])
        
        df = df.merge(
            precio,
            left_on="anio",
            right_on="anio_target",
            how="left"
        ).drop(columns=["anio_target"])
        
    # 5. Rellenar valores nulos de features numéricas
    # Usaremos una imputación simple para el model mart, pero 
    # CatBoost manejará nulos nativamente de forma eficiente.
    
    # Guardar
    out_file = proc_dir / f"model_mart_{cultivo_file}.csv"
    df.to_csv(out_file, index=False)
    
    print("\n── Auditoría Final del Model Mart ──")
    print(f"  Filas totales : {len(df):,}")
    print(f"  Columnas      : {len(df.columns)}")
    
    target_df = df[df["split"] == "target"]
    print("\n  Cobertura de variables en el set Target (2019-2024):")
    total_target = len(target_df)
    
    cols_to_check = [
        "rendimiento_lag_1", "precipitacion_acumulada_mm", 
        "ph_media", "aptitud_alta_pct", "credito_total",
        "precio_internacional_usd"
    ]
    
    for c in cols_to_check:
        if c in df.columns:
            n_notna = target_df[c].notna().sum()
            print(f"  {c:<30} {n_notna:>6,} ({n_notna/total_target:.1%})")
            
    print(f"\n✓ Model Mart guardado en: {out_file}")

if __name__ == "__main__":
    build_model_mart()
