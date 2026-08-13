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
    
    mart_path = Path(f"data/processed/model_mart_{cultivo_file}.csv")
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
    
    # Entrenar
    model = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=4, l2_leaf_reg=5, verbose=0, random_seed=42)
    model.fit(X_train, y_train)
    
    # Predecir Híbrido (70% ML, 30% Baseline)
    pred_ml = model.predict(X_test)
    pred_base = test["rendimiento_lag_1"].values
    pred_blend = 0.7 * pred_ml + 0.3 * pred_base
    
    # ── NUEVA LÓGICA DE GRÁFICA (Casos Reales vs Línea Predicha) ──
    # Para que la línea tenga sentido visual, ordenamos los datos
    # de menor a mayor rendimiento real.
    resultados = pd.DataFrame({
        "Real": y_test.values,
        "Predicho": pred_blend
    }).sort_values("Real").reset_index(drop=True)
    
    plt.figure(figsize=(12, 6))
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Eje X será un índice simple (municipio 1, municipio 2...)
    x_index = np.arange(len(resultados))
    
    # Puntos = Casos Reales del dataset
    plt.scatter(x_index, resultados["Real"], color='#3498db', alpha=0.6, s=30, label='Casos Reales (Dataset)')
    
    # Línea Roja = Datos que predice el modelo
    # Aplicamos un ligero suavizado a la línea para que sea visualmente atractiva (media móvil de 5)
    linea_predicha = resultados["Predicho"].rolling(window=5, min_periods=1, center=True).mean()
    plt.plot(x_index, linea_predicha, color='red', linewidth=2.5, label='Curva de Predicción (Modelo)')
    
    # Sombra del margen de error (10% sobre la predicción)
    plt.fill_between(x_index, 
                     linea_predicha * 0.9, 
                     linea_predicha * 1.1, 
                     color='red', alpha=0.15, label='Margen de Error (±10%)')

    plt.title(f"Efectividad del Modelo Híbrido: {cultivo.upper()} (Año 2024)\nCasos Reales vs Línea de Predicción", fontsize=14, pad=15)
    plt.xlabel("Municipios (Ordenados de menor a mayor rendimiento real)", fontsize=12)
    plt.ylabel("Rendimiento (Toneladas/Hectárea)", fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    
    # Añadir texto de métricas
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, pred_blend)
    r2 = r2_score(y_test, pred_blend)
    plt.text(len(x_index)*0.75, resultados["Real"].min(), f"MAE: {mae:.3f} t/ha\nR²: {r2:.3f}", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_file = fig_dir / f"regression_curve_{cultivo_file}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    print(f"✓ Nueva Gráfica de Regresión guardada en: {out_file}")

if __name__ == "__main__":
    plot_regression()
