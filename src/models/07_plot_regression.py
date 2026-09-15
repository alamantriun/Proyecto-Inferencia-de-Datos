"""
Paso 09: Gráfica de Regresión (Real vs Predicho)
================================================
Genera un scatter plot para evaluar visualmente la efectividad del modelo.
Un modelo perfecto tendría todos sus puntos sobre la línea diagonal (y = x).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from pathlib import Path
import yaml

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    # Mismo target encoding que en 02_train_ml
    df = df.copy()
    historico = df[df["anio"] < 2019]
    muni_mean = historico.groupby("municipio")["rendimiento_t_ha"].mean()
    global_mean = historico["rendimiento_t_ha"].mean()
    muni_count = historico.groupby("municipio")["rendimiento_t_ha"].count()
    smoothing = 10
    df["municipio_rend_historico"] = df["municipio"].map(
        (muni_mean * muni_count + global_mean * smoothing) / (muni_count + smoothing)
    ).fillna(global_mean)
    
    # Cambio de area
    if "area_sembrada_lag_1" in df.columns and "area_cosechada_lag_1" in df.columns:
        grp = df.groupby(["departamento", "municipio", "cultivo"])
        area_lag2 = grp["area_sembrada_lag_1"].shift(1)
        df["cambio_area_pct"] = ((df["area_sembrada_lag_1"] - area_lag2) / area_lag2.replace(0, np.nan)).fillna(0).clip(-1, 5)
    return df

def plot_regression():
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    cultivo_file = cultivo.lower().replace(' ', '_')
    
    mart_path = Path("reports/tablas_entrenamiento/dataset_cafe_ml_ready.csv") if cultivo_file == "cafe" else Path(f"data/processed/model_mart_{cultivo_file}.csv")
    if not mart_path.exists():
        print(f"No se encontró {mart_path}")
        return
        
    df = pd.read_csv(mart_path)
    df = df.dropna(subset=["rendimiento_t_ha"])
    df = enrich_features(df)
    
    # Evaluar en 2024
    target_year = 2024
    train = df[df["anio"] < target_year].dropna(subset=["rendimiento_t_ha", "rendimiento_lag_1"])
    test = df[df["anio"] == target_year].dropna(subset=["rendimiento_t_ha", "rendimiento_lag_1"])
    
    features = [
        "rendimiento_lag_1", "rendimiento_lag_2", "rendimiento_lag_3", 
        "media_rendimiento_3y", "municipio_rend_historico", 
        "precipitacion_acumulada_mm", "precio_internacional_usd", "cambio_area_pct"
    ]
    features = [f for f in features if f in df.columns]
    
    X_train = train[features]
    y_train = train["rendimiento_t_ha"]
    X_test = test[features]
    y_test = test["rendimiento_t_ha"]
    
    # Entrenar (Análisis Profundo)
    model = CatBoostRegressor(
        iterations=2000, 
        learning_rate=0.01, 
        depth=6, 
        l2_leaf_reg=5, 
        min_data_in_leaf=15,
        verbose=0, 
        random_seed=42
    )
    model.fit(X_train, y_train)
    
    # Predecir Híbrido (70% ML, 30% Baseline)
    pred_ml = model.predict(X_test)
    pred_base = test["rendimiento_lag_1"].values
    pred_blend = 0.7 * pred_ml + 0.3 * pred_base
    
    # ── LÓGICA DE GRÁFICA CORREGIDA (Real vs Predicho) ──
    # Un scatter plot tradicional de regresión (X = Real, Y = Predicho)
    plt.figure(figsize=(10, 8))
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Scatter de predicciones vs reales
    plt.scatter(y_test, pred_blend, color='#3498db', alpha=0.6, s=40, label='Predicciones vs Casos Reales')
    
    # Línea de perfección matemática (y = x)
    min_val = min(y_test.min(), pred_blend.min())
    max_val = max(y_test.max(), pred_blend.max())
    # Agregar un pequeño margen
    margin = (max_val - min_val) * 0.05
    min_val, max_val = min_val - margin, max_val + margin
    
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2.5, label='Línea Ideal (Predicción Perfecta)')
    
    # Ajustar límites
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.title(f"Efectividad del Modelo Híbrido: {cultivo.upper()} (Año 2024)\nRegresión: Real vs Predicho", fontsize=14, pad=15)
    plt.xlabel("Rendimiento Real (Toneladas/Hectárea)", fontsize=12)
    plt.ylabel("Rendimiento Predicho (Toneladas/Hectárea)", fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    
    # Añadir texto de métricas
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, pred_blend)
    r2 = r2_score(y_test, pred_blend)
    
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor='gray')
    plt.text(max_val * 0.70, min_val + (max_val - min_val) * 0.1, 
             f"Métricas del Modelo:\nMAE: {mae:.3f} t/ha\nR²: {r2:.3f}", 
             fontsize=12, bbox=bbox_props)
    
    plt.tight_layout()
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_file = fig_dir / f"regression_curve_{cultivo_file}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    print(f"✓ Nueva Gráfica de Regresión guardada en: {out_file}")

if __name__ == "__main__":
    plot_regression()
