"""
Entrenamiento y Evaluación de Baselines (Reglas Duras)
======================================================
Ejecuta el plan de backtesting temporal definido en config.yaml.
Baselines evaluados:
  B1: Rendimiento del año t-1
  B2: Rendimiento promedio de los últimos 3 años (t-1, t-2, t-3)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error con protección para división por cero."""
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_predictions(y_true, y_pred, name):
    mask = y_true.notna() & y_pred.notna()
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    if len(y_t) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan, "N": 0}
        
    res = {
        "MAE": mean_absolute_error(y_t, y_p),
        "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
        "R2": r2_score(y_t, y_p),
        "MAPE": mape(y_t, y_p),
        "N": len(y_t)
    }
    return res

def run_baselines():
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    cultivo_file = cultivo.lower().replace(' ', '_')
    años_test = config["project"].get("años_backtest", [2019, 2020, 2021, 2022, 2023, 2024])
    
    mart_path = Path(f"data/processed/model_mart_{cultivo_file}.csv")

    print(f"=== Evaluación de Baselines (Heurísticas) para {cultivo.upper()} ===\n")
    df = pd.read_csv(mart_path)
    
    resultados_b1 = []
    resultados_b2 = []
    
    for anio in años_test:
        test = df[df["anio"] == anio]
        
        if len(test) == 0:
            continue
            
        y_true = test["rendimiento_t_ha"]
        
        # Predicciones
        y_b1 = test["rendimiento_lag_1"]
        y_b2 = test["media_rendimiento_3y"]
        
        m_b1 = evaluate_predictions(y_true, y_b1, "B1")
        m_b2 = evaluate_predictions(y_true, y_b2, "B2")
        
        resultados_b1.append({"Año": anio, **m_b1})
        resultados_b2.append({"Año": anio, **m_b2})
        
    # Agregados
    print("--- Baseline 1: Rendimiento del año anterior ---")
    df_b1 = pd.DataFrame(resultados_b1).set_index("Año")
    print(df_b1.round(3).to_string())
    print("\nPromedio B1 MAE : {:.3f} t/ha".format(df_b1["MAE"].mean()))
    print("Promedio B1 RMSE: {:.3f} t/ha".format(df_b1["RMSE"].mean()))
    
    print("\n--- Baseline 2: Promedio de los últimos 3 años ---")
    df_b2 = pd.DataFrame(resultados_b2).set_index("Año")
    print(df_b2.round(3).to_string())
    print("\nPromedio B2 MAE : {:.3f} t/ha".format(df_b2["MAE"].mean()))
    print("Promedio B2 RMSE: {:.3f} t/ha".format(df_b2["RMSE"].mean()))
    
    # Guardar reporte
    rep_dir = Path("reports/tables")
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    df_b1["Modelo"] = "Baseline 1 (t-1)"
    df_b2["Modelo"] = "Baseline 2 (media 3y)"
    
    final_report = pd.concat([df_b1, df_b2]).reset_index()
    file_name = f"resultados_baselines_{cultivo_file}.csv"
    final_report.to_csv(rep_dir / file_name, index=False)
    print(f"\n✓ Resultados guardados en {rep_dir / file_name}")

if __name__ == "__main__":
    run_baselines()
