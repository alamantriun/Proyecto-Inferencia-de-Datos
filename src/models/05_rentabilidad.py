import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import yaml
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import importlib
train_ml = importlib.import_module("src.models.02_train_ml")
get_ablation_features = train_ml.get_ablation_features

def calcular_rentabilidad():
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    cultivo_file = cultivo.lower().replace(' ', '_')
    años_test = config["project"].get("años_backtest", [2019, 2020, 2021, 2022, 2023, 2024])
    trm = config["project"].get("trm_usd_cop", 4000)
    
    ultimo_anio = max(años_test)
    mart_path = Path("reports/tablas_entrenamiento/dataset_cafe_ml_ready.csv") if cultivo_file == "cafe" else Path(f"data/processed/model_mart_{cultivo_file}.csv")
    panel_path = Path(f"data/processed/panel_{cultivo_file}.csv")

    print(f"=== Análisis de Rentabilidad y Negocio ({cultivo.upper()} - {ultimo_anio}) ===")
    
    if not mart_path.exists():
        print(f"❌ No se encontró {mart_path}")
        return
    
    # Cargar datos normalizados (para modelo ML)
    df = pd.read_csv(mart_path)
    df = df.dropna(subset=["rendimiento_t_ha"])
    
    # Cargar datos ORIGINALES sin normalizar (para cálculo de ingresos)
    df_original = None
    if panel_path.exists():
        df_original = pd.read_csv(panel_path)
        print(f"✓ Cargados datos originales sin normalizar desde {panel_path}")
    
    # Entrenar modelo con variables completas (F)
    # Fix Error #5: Usar MISMOS hiperparámetros que 02_train_ml.py
    # y aplicar el blending híbrido para coherencia con las métricas reportadas
    feats = get_ablation_features("F")
    feats = [f for f in feats if f in df.columns]
    
    train = df[df["anio"] < ultimo_anio]
    test = df[df["anio"] == ultimo_anio].copy()
    # Convertir a 2025 (Futuro) asumiendo precio constante
    test["anio"] = ultimo_anio + 1
    
    if len(train) == 0 or len(test) == 0:
        print("❌ No hay suficientes datos para entrenar o evaluar.")
        return
        
    model = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=4,
        l2_leaf_reg=5, min_data_in_leaf=20,
        loss_function='RMSE', verbose=0, random_seed=42
    )
    
    # Peso de muestra basado en confiabilidad
    sample_weight = None
    if "score_confiabilidad" in train.columns:
        sample_weight = train["score_confiabilidad"].fillna(0.5).values
    
    model.fit(train[feats], train["rendimiento_t_ha"], sample_weight=sample_weight)
    
    # Inferir rendimiento con blending híbrido (70% ML + 30% Baseline)
    pred_ml = model.predict(test[feats])
    if "rendimiento_lag_1" in test.columns:
        # Usar pd.Series para que fillna lo acepte correctamente
        pred_base = test["rendimiento_lag_1"].fillna(pd.Series(pred_ml, index=test.index)).values
        test["rendimiento_inferido"] = np.clip(0.7 * pred_ml + 0.3 * pred_base, 0, None)
    else:
        test["rendimiento_inferido"] = np.clip(pred_ml, 0, None)
    
    # Para calcular ingresos reales: necesitamos desnormalizar
    # Cargar datos originales y hacer merge correcto
    print(f"✓ Entrenando modelo y calculando proyecciones...")
    
    if df_original is None:
        print("❌ No se encontraron datos originales.")
        return
    
    # Obtener datos del año a proyectar (original, sin normalizar)
    df_orig_test = df_original[df_original["anio"] == ultimo_anio].copy()
    df_orig_test = df_orig_test[["departamento", "municipio", "area_cosechada_ha", "rendimiento_t_ha"]].copy()
    df_orig_test.rename(columns={"area_cosechada_ha": "area_original", "rendimiento_t_ha": "rendimiento_base_original"}, inplace=True)
    
    test_merged = test.merge(df_orig_test, on=["departamento", "municipio"], how="left")
    
    rendimiento_mean = df_original["rendimiento_t_ha"].mean()
    rendimiento_std = df_original["rendimiento_t_ha"].std()
    
    # Escenarios de precio de Bolsa NY (2025-2029) igual que 06_forecast
    precios = [5500, 4500, 3800, 3500, 3500]
    precio_promedio_5y = sum(precios) / len(precios)
    
    # Desnormalizar rendimiento
    test_merged["rendimiento_inferido_denorm"] = (test_merged["rendimiento_inferido"] * rendimiento_std) + rendimiento_mean
    test_merged["rendimiento_inferido_denorm"] = np.clip(test_merged["rendimiento_inferido_denorm"], 0, None)
    
    test_merged = test_merged.dropna(subset=["area_original", "rendimiento_inferido_denorm", "rendimiento_base_original"])
    
    # Calcular ingreso base 2024 (asumiendo precio base conservador de 4000 USD/t)
    precio_base_2024 = 4000
    test_merged["ingreso_base_cop"] = test_merged["rendimiento_base_original"] * test_merged["area_original"] * precio_base_2024 * trm
    
    # Calcular ingreso promedio a 5 años
    test_merged["produccion_esperada_t"] = test_merged["rendimiento_inferido_denorm"] * test_merged["area_original"]
    test_merged["ingreso_promedio_5y_cop"] = test_merged["produccion_esperada_t"] * precio_promedio_5y * trm
    test_merged["rendimiento_original"] = test_merged["rendimiento_inferido_denorm"]
    
    # Calcular porcentaje de crecimiento
    test_merged["pct_crecimiento"] = ((test_merged["ingreso_promedio_5y_cop"] - test_merged["ingreso_base_cop"]) / test_merged["ingreso_base_cop"]) * 100
    # Evitar divisiones por cero o nulos
    test_merged["pct_crecimiento"] = test_merged["pct_crecimiento"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    test = test_merged.copy()
    top_rentables = test.sort_values("ingreso_promedio_5y_cop", ascending=False).head(15)
    
    print(f"\nTop 15 Municipios: Ingreso Promedio (2025-2029):")
    cols_show = ["departamento", "municipio", "rendimiento_original", "produccion_esperada_t", "ingreso_promedio_5y_cop"]
    
    for _, row in top_rentables.iterrows():
        ingreso_millones = row["ingreso_promedio_5y_cop"] / 1e6
        print(f"{row['departamento'][:15]:<15} | {row['municipio'][:15]:<15} | "
              f"Rend: {row['rendimiento_original']:.2f} t/ha | "
              f"Prod: {row['produccion_esperada_t']:,.1f} t | "
              f"Ingreso: ${ingreso_millones:,.0f} Millones COP")
              
    plt.figure(figsize=(12, 7))
    labels = top_rentables["municipio"] + " (" + top_rentables["departamento"] + ")"
    valores_millones = top_rentables["ingreso_promedio_5y_cop"] / 1e6
    
    bars = plt.barh(labels[::-1], valores_millones[::-1], color="#27ae60")
    
    # Los valores se graficaron al revés ([::-1]) para que el mayor quede arriba.
    # Necesitamos extraer la columna y voltearla para emparejarla con las barras.
    pcts = top_rentables["pct_crecimiento"].values[::-1]
    
    for p, pct in zip(bars, pcts):
        width = p.get_width()
        sign = "+" if pct >= 0 else ""
        plt.text(
            width + (valores_millones.max() * 0.01),
            p.get_y() + p.get_height() / 2,
            f"${width:,.0f}M ({sign}{pct:.1f}%)",
            ha="left",
            va="center",
            fontweight="bold"
        )
        
    plt.title(f"Medición a Futuro: Top 15 Municipios por Ingreso Promedio (2025 - 2029) - {cultivo.upper()}", fontsize=14, pad=20)
    plt.xlabel("Ingreso Bruto Calculado (Millones COP)", fontsize=12)
    plt.ylabel("")
    plt.xlim(0, valores_millones.max() * 1.15)
    plt.tight_layout()
    
    rep_dir = Path("reports/figures")
    rep_dir.mkdir(parents=True, exist_ok=True)
    out_img = rep_dir / f"rentabilidad_top_municipios_{cultivo_file}.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Guardar tabla CSV
    out_csv = Path("reports/tables") / f"proyeccion_negocio_{cultivo_file}.csv"
    test[cols_show].to_csv(out_csv, index=False)
    
    print(f"\n✓ Gráfica guardada en {out_img}")
    print(f"✓ Datos exportados a {out_csv}")

if __name__ == "__main__":
    calcular_rentabilidad()
