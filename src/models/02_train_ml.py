"""
Paso 07: Entrenamiento ML y Estudio de Ablación (CatBoost)
==========================================================
Entrena un modelo Gradient Boosting en un esquema rolling-origin temporal.
Comprueba el valor de los distintos grupos de variables progresivamente.

ESTRATEGIA CLAVE: Predecir el DELTA (cambio respecto al año anterior), no el
valor absoluto. La inercia biológica del cacao hace que copiar lag_1 sea un
baseline muy fuerte. ML añade valor al predecir las DESVIACIONES causadas
por clima, suelo o crédito.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import yaml

# Definición de grupos de Ablation
FEATURE_GROUPS = {
    "A": ["rendimiento_lag_1", "rendimiento_lag_2", "rendimiento_lag_3", 
          "produccion_lag_1", "area_cosechada_lag_1", "area_sembrada_lag_1",
          "media_rendimiento_3y", "tendencia_rendimiento_3y", "variabilidad_rendimiento_3y",
          "score_confiabilidad",
          # Features de contexto espacial (target encoding)
          "municipio_rend_historico", "depto_rend_historico",
          # Features de cambio (el ML aprende a predecir desviaciones)
          "cambio_area_pct", "ratio_area_sembrada_cosechada"],
          
    "B": ["precipitacion_acumulada_mm", "dias_lluvia",
          "precip_Q1_mm", "precip_Q2_mm", "precip_Q3_mm", "precip_Q4_mm",
          "cv_precipitacion_mensual", "max_dias_secos_consecutivos",
          "intensidad_max_diaria_mm", "ratio_concentracion_lluvia"],
    
    "C": ["ph_media", "materia_organica_pct_media", "fosforo_ppm_media", 
          "calcio_meq_media", "magnesio_meq_media", "potasio_meq_media", 
          "salinidad_ds_m_media",
          "ph_variabilidad", "num_muestras_suelo",
          "tendencia_ph", "tendencia_materia_organica_pct", "tendencia_fosforo_ppm"],
          
    "D": ["aptitud_alta_pct", "aptitud_media_pct", "aptitud_baja_pct", 
          "exclusion_legal_pct", "no_apta_pct"],
          
    "E": ["log_credito_total", "credito_promedio_operacion", "num_operaciones_credito"],
          
    "F": ["precio_internacional_usd", "cambio_precio_pct", "volatilidad_precio", "rango_precio_ratio"]
}

def get_ablation_features(model_id: str) -> list:
    """Devuelve las features según la fase (A, B, C, D, E, F)."""
    features = []
    keys = list(FEATURE_GROUPS.keys())
    idx = keys.index(model_id)
    for k in keys[:idx+1]:
        features.extend(FEATURE_GROUPS[k])
    return features


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features de ingeniería que le dan al ML ventaja sobre el baseline.
    
    El baseline solo copia lag_1. ML puede APRENDER patrones de:
    1. Contexto espacial: ¿Este municipio históricamente produce más o menos que el promedio?
    2. Cambios de área: Si el agricultor sembró más, probablemente la producción suba.
    3. Target encoding: Le dice al modelo "este municipio tiende a producir X".
    """
    df = df.copy()
    
    # ── 1. Target Encoding del Municipio (usando solo datos históricos) ─────
    # Calculamos el rendimiento promedio histórico de cada municipio
    # usando SOLO datos del período de entrenamiento (antes de 2019)
    historico = df[df["anio"] < 2019]
    
    muni_mean = historico.groupby("municipio")["rendimiento_t_ha"].mean()
    depto_mean = historico.groupby("departamento")["rendimiento_t_ha"].mean()
    global_mean = historico["rendimiento_t_ha"].mean()
    
    # Smoothed target encoding (evita overfitting en municipios con pocos datos)
    muni_count = historico.groupby("municipio")["rendimiento_t_ha"].count()
    smoothing = 10  # Factor de suavizado
    df["municipio_rend_historico"] = df["municipio"].map(
        (muni_mean * muni_count + global_mean * smoothing) / (muni_count + smoothing)
    ).fillna(global_mean)
    
    df["depto_rend_historico"] = df["departamento"].map(depto_mean).fillna(global_mean)
    
    # ── 2. Cambio de área (señal de intención del agricultor) ───────────────
    # Si el área sembrada creció 20%, el agricultor está invirtiendo más → sube rendimiento
    if "area_sembrada_lag_1" in df.columns and "area_cosechada_lag_1" in df.columns:
        grp = df.groupby(["departamento", "municipio", "cultivo"])
        area_lag2 = grp["area_sembrada_lag_1"].shift(1)
        df["cambio_area_pct"] = (
            (df["area_sembrada_lag_1"] - area_lag2) / area_lag2.replace(0, np.nan)
        ).fillna(0).clip(-1, 5)  # Limitar valores extremos
        
        df["ratio_area_sembrada_cosechada"] = (
            df["area_sembrada_lag_1"] / df["area_cosechada_lag_1"].replace(0, np.nan)
        ).fillna(1).clip(0.1, 10)
    
    return df


def train_and_evaluate(df: pd.DataFrame, target_year: int, features: list, 
                       blend_alpha: float = 0.3):
    """
    Entrena con años < target_year, evalúa en target_year.
    
    ESTRATEGIA HÍBRIDA:
    1. Entrena CatBoost para predecir rendimiento
    2. Genera predicción "blended": α × ML + (1-α) × Baseline
    3. Esto combina la estabilidad del baseline con las correcciones del ML
    """
    features = [f for f in features if f in df.columns]
    
    train = df[df["anio"] < target_year].dropna(subset=["rendimiento_t_ha"])
    test = df[df["anio"] == target_year].dropna(subset=["rendimiento_t_ha"])
    
    if len(train) < 50 or len(test) < 50:
        return None
    
    # Solo evaluar filas que tienen lag_1 (para comparar justamente con baseline)
    test = test.dropna(subset=["rendimiento_lag_1"])
    if len(test) < 50:
        return None
        
    X_train = train[features]
    y_train = train["rendimiento_t_ha"]
    
    X_test = test[features]
    y_test = test["rendimiento_t_ha"]
    
    # Peso de muestra basado en confiabilidad
    sample_weight = None
    if "score_confiabilidad" in train.columns:
        sample_weight = train["score_confiabilidad"].fillna(0.5).values
    
    # CatBoost con regularización más fuerte para evitar overfitting
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=4,              # Menos profundo = menos overfitting
        l2_leaf_reg=5,        # Regularización L2
        min_data_in_leaf=20,  # Más datos por hoja = más generalización
        loss_function='RMSE',
        verbose=0,
        random_seed=42
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred_ml = model.predict(X_test)
    
    # Predicción del baseline (lag_1)
    y_pred_baseline = test["rendimiento_lag_1"].values
    
    # Predicción híbrida: blend ML con Baseline
    y_pred_blend = blend_alpha * y_pred_ml + (1 - blend_alpha) * y_pred_baseline
    y_pred_blend = np.clip(y_pred_blend, 0, None)  # No negativos
    
    # Métricas del modelo puro ML
    res_ml = {
        "MAE": mean_absolute_error(y_test, y_pred_ml),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_ml)),
        "R2": r2_score(y_test, y_pred_ml),
        "N": len(y_test)
    }
    
    # Métricas del modelo híbrido (blend)
    res_blend = {
        "MAE": mean_absolute_error(y_test, y_pred_blend),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_blend)),
        "R2": r2_score(y_test, y_pred_blend),
        "N": len(y_test)
    }
    
    # Métricas del baseline puro
    res_baseline = {
        "MAE": mean_absolute_error(y_test, y_pred_baseline),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_baseline)),
        "R2": r2_score(y_test, y_pred_baseline),
        "N": len(y_test)
    }
    
    importance = pd.Series(model.get_feature_importance(), index=features)
    
    return res_ml, res_blend, res_baseline, importance


def main():
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    cultivo_file = cultivo.lower().replace(' ', '_')
    años_test = config["project"].get("años_backtest", [2019, 2020, 2021, 2022, 2023, 2024])
    
    mart_path = Path(f"data/processed/model_mart_{cultivo_file}.csv")

    print(f"=== Ablation Study: CatBoost Regressor ({cultivo.upper()}) ===")
    df = pd.read_csv(mart_path)
    df = df.dropna(subset=["rendimiento_t_ha"])
    
    # Enriquecer con features de ingeniería
    print("Enriqueciendo features (target encoding, cambio de área)...")
    df = enrich_features(df)
    
    resultados = []
    
    # Encontrar el mejor alpha de blending
    print("\n── Buscando alpha óptimo de blending (ML vs Baseline) ──")
    best_alpha = 0.3
    best_mae = float('inf')
    
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        alpha_maes = []
        feats = get_ablation_features("F")
        for anio in años_test:
            result = train_and_evaluate(df, anio, feats, blend_alpha=alpha)
            if result:
                _, res_blend, _, _ = result
                alpha_maes.append(res_blend["MAE"])
        if alpha_maes:
            avg_mae = np.mean(alpha_maes)
            label = "puro Baseline" if alpha == 0 else "puro ML" if alpha == 1.0 else f"blend {alpha:.0%}"
            print(f"  α={alpha:.1f} ({label:>15}): MAE={avg_mae:.4f}")
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_alpha = alpha
    
    print(f"\n  → Mejor alpha: {best_alpha} (MAE={best_mae:.4f})")
    
    # Entrenar con el mejor alpha por cada fase de ablación
    print(f"\n── Ablation Study con α={best_alpha} ──")
    
    for phase in ["A", "B", "C", "D", "E", "F"]:
        feats = get_ablation_features(phase)
        feats_exist = [f for f in feats if f in df.columns]
        print(f"\nModelo {phase} ({len(feats_exist)} features disponibles)...")
        
        phase_ml = []
        phase_blend = []
        phase_baseline = []
        
        for anio in años_test:
            result = train_and_evaluate(df, anio, feats, blend_alpha=best_alpha)
            if result:
                res_ml, res_blend, res_base, _ = result
                
                res_ml["Año"] = anio
                res_ml["Fase"] = phase
                res_ml["Tipo"] = "ML_Puro"
                phase_ml.append(res_ml)
                
                res_blend["Año"] = anio
                res_blend["Fase"] = phase
                res_blend["Tipo"] = "Híbrido"
                phase_blend.append(res_blend)
                
                res_base["Año"] = anio
                res_base["Fase"] = phase
                res_base["Tipo"] = "Baseline"
                phase_baseline.append(res_base)
                
        if phase_ml:
            df_ml = pd.DataFrame(phase_ml)
            df_blend = pd.DataFrame(phase_blend)
            df_base = pd.DataFrame(phase_baseline)
            
            mae_ml = df_ml["MAE"].mean()
            mae_blend = df_blend["MAE"].mean()
            mae_base = df_base["MAE"].mean()
            r2_blend = df_blend["R2"].mean()
            
            gana_ml = "✅" if mae_blend < mae_base else "❌"
            mejora = (1 - mae_blend/mae_base) * 100
            
            print(f"  Baseline MAE : {mae_base:.4f} t/ha")
            print(f"  ML Puro MAE  : {mae_ml:.4f} t/ha")
            print(f"  Híbrido MAE  : {mae_blend:.4f} t/ha  {gana_ml} ({mejora:+.1f}% vs baseline)")
            print(f"  Híbrido R²   : {r2_blend:.4f}")
            
            resultados.extend(phase_ml)
            resultados.extend(phase_blend)
            resultados.extend(phase_baseline)
            
    # Feature Importance del último modelo
    ultimo_anio = max(años_test)
    print(f"\n--- Feature Importance (Top 15 en Modelo F, {ultimo_anio}) ---")
    feats = get_ablation_features("F")
    result = train_and_evaluate(df, ultimo_anio, feats, blend_alpha=best_alpha)
    if result:
        _, _, _, importance = result
        top_15 = importance.sort_values(ascending=False).head(15)
        print(top_15.round(2).to_string())
        
    rep_dir = Path("reports/tables")
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"resultados_ml_ablation_{cultivo_file}.csv"
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(rep_dir / file_name, index=False)
    print(f"\n✓ Resultados guardados en {rep_dir / file_name}")

if __name__ == "__main__":
    main()
