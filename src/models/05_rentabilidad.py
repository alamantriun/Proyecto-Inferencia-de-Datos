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
    mart_path = Path(f"data/processed/model_mart_{cultivo_file}.csv")

    print(f"=== Análisis de Rentabilidad y Negocio ({cultivo.upper()} - {ultimo_anio}) ===")
    
    if not mart_path.exists():
        print(f"❌ No se encontró {mart_path}")
        return
        
    df = pd.read_csv(mart_path)
    df = df.dropna(subset=["rendimiento_t_ha", "precio_internacional_usd", "area_cosechada_lag_1"])
    
    # Entrenar modelo con variables completas (F)
    # Fix Error #5: Usar MISMOS hiperparámetros que 02_train_ml.py
    # y aplicar el blending híbrido para coherencia con las métricas reportadas
    feats = get_ablation_features("F")
    feats = [f for f in feats if f in df.columns]
    
    train = df[df["anio"] < ultimo_anio]
    test = df[df["anio"] == ultimo_anio].copy()
    
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
    
    # Cálculo de métrica de negocio: Ingreso Esperado (COP)
    # Ingreso = Rendimiento (t/ha) * Area (ha) * Precio (USD/t) * TRM (COP/USD)
    # Asumimos que el área a cosechar es aproximadamente la del año anterior (ex-ante)
    test["produccion_esperada_t"] = test["rendimiento_inferido"] * test["area_cosechada_lag_1"]
    test["ingreso_esperado_cop"] = test["produccion_esperada_t"] * test["precio_internacional_usd"] * trm
    
    # Ordenar municipios más rentables
    top_rentables = test.sort_values("ingreso_esperado_cop", ascending=False).head(15)
    
    print(f"\nTop 15 Municipios con mayor Ingreso Esperado ({ultimo_anio}):")
    cols_show = ["departamento", "municipio", "rendimiento_inferido", "produccion_esperada_t", "ingreso_esperado_cop"]
    
    # Imprimir bonito
    for _, row in top_rentables.iterrows():
        ingreso_millones = row["ingreso_esperado_cop"] / 1e6
        print(f"{row['departamento'][:15]:<15} | {row['municipio'][:15]:<15} | "
              f"Rend: {row['rendimiento_inferido']:.2f} t/ha | "
              f"Prod: {row['produccion_esperada_t']:,.1f} t | "
              f"Ingreso: ${ingreso_millones:,.0f} Millones COP")
              
    # Generar visualización
    plt.figure(figsize=(12, 7))
    labels = top_rentables["municipio"] + " (" + top_rentables["departamento"] + ")"
    valores_millones = top_rentables["ingreso_esperado_cop"] / 1e6
    
    bars = plt.barh(labels[::-1], valores_millones[::-1], color="#27ae60")
    
    for p in bars:
        width = p.get_width()
        plt.text(
            width + (valores_millones.max() * 0.01),
            p.get_y() + p.get_height() / 2,
            f"${width:,.0f}M",
            ha="left",
            va="center",
            fontweight="bold"
        )
        
    plt.title(f"Top 15 Municipios con Mayor Ingreso Esperado ({ultimo_anio}) - {cultivo.upper()}", fontsize=14, pad=20)
    plt.xlabel("Ingreso Bruto Esperado (Millones COP)", fontsize=12)
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
