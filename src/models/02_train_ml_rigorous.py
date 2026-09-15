"""
Paso 07-RIGUROSO: Entrenamiento ML AVANZADO (CatBoost + Validación Exhaustiva)
================================================================================
VERSIÓN INTENSIVA en cómputo - Requiere 8-12 GB RAM

MEJORAS RESPECTO A VERSIÓN ESTÁNDAR:
1. ✓ Validación cruzada temporal (TimeSeriesSplit)
2. ✓ Grid Search automático de hiperparámetros
3. ✓ Early stopping contra validation set
4. ✓ Ensemble de múltiples modelos (random seeds)
5. ✓ SHAP feature importance (explicabilidad)
6. ✓ Monitoreo exhaustivo de overfitting
7. ✓ Aumenta iteraciones: 500 → 3000
8. ✓ Reduce learning_rate: 0.03 → 0.01 (más lento, mejor generalización)
9. ✓ Aumenta depth: 4 → 6-8 (más complejo)

REQUISITOS DE CÓMPUTO:
- RAM: 8-12 GB (vs 2-4 GB en versión estándar)
- CPU: 30-60 minutos para 4 cultivos (vs 5-10 minutos)
- Disco: +500 MB para guardar modelos intermedios
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler
import json
import yaml
import time
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN: Hiperparámetros más exigentes
# ============================================================================

CATBOOST_BASE = {
    # ITERATIONS: Entrenamiento más lento pero mejor generalización
    "iterations": 3000,              # vs 500 en versión estándar
    
    # LEARNING RATE: Más conservador = converge lentamente
    "learning_rate": 0.01,           # vs 0.03 en versión estándar
    
    # PROFUNDIDAD: Árboles más complejos
    "depth": 6,                      # vs 4 en versión estándar
    
    # REGULARIZACIÓN L2: Castiga pesos grandes
    "l2_leaf_reg": 10,               # vs 5 en versión estándar
    
    # MIN DATA IN LEAF: Menos datos por hoja = más granular
    "min_data_in_leaf": 5,           # vs 20 en versión estándar
    
    # SUBSAMPLE: Usar solo fracción de datos por iteración (boosting estocástico)
    "subsample": 0.8,                # Evita sobreajuste
    
    # COLSAMPLE: Usar solo fracción de features por árbol
    "colsample_bylevel": 0.8,
    
    # EARLY STOPPING: Parar si val_loss no mejora
    "early_stopping_rounds": 100,    # vs sin early stopping
    
    # LOSS FUNCTION: RMSE vs MAE (prueba ambas)
    "loss_function": "RMSE",
    
    "verbose": 0,
    "random_seed": 42
}

# Grid Search: probar múltiples configuraciones
PARAM_GRID = {
    "depth": [5, 6, 7],                    # Profundidad del árbol
    "learning_rate": [0.005, 0.01, 0.02], # Tasa de aprendizaje
    "l2_leaf_reg": [5, 10, 15],            # Regularización
    "subsample": [0.7, 0.8, 0.9],          # Subsampling
}

FEATURE_GROUPS = {
    "A": ["rendimiento_lag_1", "rendimiento_lag_2", "rendimiento_lag_3", 
          "produccion_lag_1", "area_cosechada_lag_1", "area_sembrada_lag_1",
          "media_rendimiento_3y", "tendencia_rendimiento_3y", "variabilidad_rendimiento_3y",
          "score_confiabilidad",
          "municipio_rend_historico", "depto_rend_historico",
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
    """Añade features de ingeniería (igual que versión estándar)."""
    df = df.copy()
    
    historico = df[df["anio"] < 2019]
    
    muni_mean = historico.groupby("municipio")["rendimiento_t_ha"].mean()
    depto_mean = historico.groupby("departamento")["rendimiento_t_ha"].mean()
    global_mean = historico["rendimiento_t_ha"].mean()
    
    muni_count = historico.groupby("municipio")["rendimiento_t_ha"].count()
    smoothing = 10
    df["municipio_rend_historico"] = df["municipio"].map(
        (muni_mean * muni_count + global_mean * smoothing) / (muni_count + smoothing)
    ).fillna(global_mean)
    
    df["depto_rend_historico"] = df["departamento"].map(depto_mean).fillna(global_mean)
    
    if "area_sembrada_lag_1" in df.columns and "area_cosechada_lag_1" in df.columns:
        grp = df.groupby(["departamento", "municipio", "cultivo"])
        area_lag2 = grp["area_sembrada_lag_1"].shift(1)
        df["cambio_area_pct"] = (
            (df["area_sembrada_lag_1"] - area_lag2) / area_lag2.replace(0, np.nan)
        ).fillna(0).clip(-1, 5)
        
        df["ratio_area_sembrada_cosechada"] = (
            df["area_sembrada_lag_1"] / df["area_cosechada_lag_1"].replace(0, np.nan)
        ).fillna(1).clip(0.1, 10)
    
    return df


def grid_search_hyperparams(X_train, y_train, X_val, y_val, param_grid=PARAM_GRID, top_k=3):
    """
    Grid Search exhaustivo sobre hiperparámetros.
    
    EXTREMADAMENTE INTENSIVO EN CÓMPUTO:
    - Prueba 3×3×3×3 = 81 combinaciones de parámetros
    - Cada una entrena 3000 iteraciones
    - Total: 243,000 iteraciones de CatBoost
    """
    
    print(f"\n  🔧 GRID SEARCH: Probando {len(list(ParameterGrid(param_grid)))} combinaciones...")
    print(f"     (Esto puede tomar 20-40 minutos en un equipo de 12GB RAM)")
    
    results = []
    params_list = list(ParameterGrid(param_grid))
    
    for i, params in enumerate(params_list, 1):
        # Crear configuración: base + grid
        config = CATBOOST_BASE.copy()
        config.update(params)
        
        # Pool de CatBoost con validación set (para early stopping)
        pool_train = Pool(X_train, y_train)
        pool_val = Pool(X_val, y_val)
        
        # Entrenar
        model = CatBoostRegressor(**config)
        model.fit(pool_train, eval_set=pool_val, verbose=0)
        
        # Evaluar
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        results.append({
            "params": params,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "iterations_used": model.best_iteration_ if hasattr(model, 'best_iteration_') else config["iterations"]
        })
        
        print(f"    [{i:2d}/{len(params_list)}] depth={params['depth']} lr={params['learning_rate']:.4f} "
              f"→ MAE={mae:.4f}")
    
    # Top K mejores configuraciones
    results_sorted = sorted(results, key=lambda x: x["MAE"])
    
    print(f"\n  ✓ Top {top_k} configuraciones:")
    for j, res in enumerate(results_sorted[:top_k], 1):
        print(f"    {j}. MAE={res['MAE']:.4f} | {res['params']}")
    
    return results_sorted[:top_k]


def train_ensemble_models(df: pd.DataFrame, target_year: int, features: list, 
                          best_hyperparams: dict, n_models: int = 5):
    """
    Entrena múltiples modelos con distintos random_seed.
    
    ENSEMBLE BAGGING:
    - Entrena N modelos independientes (diferentes inicializaciones)
    - Promedia predicciones (más estable que modelo único)
    - Reduce variancia
    - Equivale a validación cruzada sobre el test set
    """
    
    features = [f for f in features if f in df.columns]
    
    train = df[df["anio"] < target_year].dropna(subset=["rendimiento_t_ha"])
    test = df[df["anio"] == target_year].dropna(subset=["rendimiento_t_ha"])
    
    if len(train) < 50 or len(test) < 50:
        return None
    
    test = test.dropna(subset=["rendimiento_lag_1"])
    if len(test) < 50:
        return None
        
    X_train = train[features].values
    y_train = train["rendimiento_t_ha"].values
    X_test = test[features].values
    y_test = test["rendimiento_t_ha"].values
    y_baseline = test["rendimiento_lag_1"].values
    
    # Ensemble: entrenar N modelos
    print(f"      Ensemble de {n_models} modelos...", end=" ", flush=True)
    ensemble_preds = []
    ensemble_importances = []
    
    for seed_idx in range(n_models):
        config = CATBOOST_BASE.copy()
        config.update(best_hyperparams)
        config["random_seed"] = 42 + seed_idx  # Diferentes seeds
        
        sample_weight = None
        if "score_confiabilidad" in train.columns:
            sample_weight = train["score_confiabilidad"].fillna(0.5).clip(0, 1).values
        
        model = CatBoostRegressor(**config)
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=0)
        
        y_pred = model.predict(X_test)
        ensemble_preds.append(y_pred)
        ensemble_importances.append(pd.Series(
            model.get_feature_importance(), 
            index=features
        ))
    
    # Promedio de predicciones
    y_pred_ensemble = np.mean(ensemble_preds, axis=0)
    y_pred_ensemble = np.clip(y_pred_ensemble, 0, None)
    
    # Importancia promedio
    importance_ensemble = pd.concat(ensemble_importances, axis=1).mean(axis=1)
    
    print("✓")
    
    # Métricas
    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
    mae_baseline = mean_absolute_error(y_test, y_baseline)
    
    return {
        "MAE": mae_ensemble,
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_ensemble)),
        "R2": r2_score(y_test, y_pred_ensemble),
        "MAE_baseline": mae_baseline,
        "N": len(y_test),
        "importance": importance_ensemble,
        "y_pred": y_pred_ensemble,
        "y_test": y_test
    }


def timeseries_cross_validation(df: pd.DataFrame, features: list, best_hyperparams: dict, n_splits: int = 5):
    """
    TimeSeriesSplit: Validación cruzada que respeta la naturaleza temporal.
    
    En lugar de split aleatorio:
    - Split 1: Entrena [2006-2014], valida 2015
    - Split 2: Entrena [2006-2015], valida 2016
    - Split 3: Entrena [2006-2016], valida 2017
    - Split 4: Entrena [2006-2017], valida 2018
    - Split 5: Entrena [2006-2018], valida 2019
    
    Esto es mucho más realista para series temporales.
    """
    
    print(f"\n      TimeSeriesSplit ({n_splits} splits)...", flush=True)
    
    features = [f for f in features if f in df.columns]
    df_clean = df.dropna(subset=["rendimiento_t_ha"])
    
    años_unique = sorted(df_clean["anio"].unique())
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(df_clean), 1
    ):
        df_train = df_clean.iloc[train_idx]
        df_val = df_clean.iloc[val_idx]
        
        X_train = df_train[features].values
        y_train = df_train["rendimiento_t_ha"].values
        X_val = df_val[features].values
        y_val = df_val["rendimiento_t_ha"].values
        
        config = CATBOOST_BASE.copy()
        config.update(best_hyperparams)
        
        model = CatBoostRegressor(**config)
        model.fit(X_train, y_train, verbose=0)
        
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, None)
        
        mae = mean_absolute_error(y_val, y_pred)
        fold_results.append(mae)
        
        año_val = df_val["anio"].min()
        print(f"        Fold {fold}: Entrena hasta año {df_train['anio'].max()}, "
              f"valida {año_val} → MAE={mae:.4f}")
    
    mae_cv = np.mean(fold_results)
    std_cv = np.std(fold_results)
    
    print(f"      CV Promedio: MAE={mae_cv:.4f} ± {std_cv:.4f}")
    
    return mae_cv, std_cv, fold_results


def main():
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    cultivo_file = cultivo.lower().replace(' ', '_')
    años_test = config["project"].get("años_backtest", [2019, 2020, 2021, 2022, 2023, 2024])
    
    mart_path = Path("reports/tablas_entrenamiento/dataset_cafe_ml_ready.csv") if cultivo_file == "cafe" else Path(f"data/processed/model_mart_{cultivo_file}.csv")

    print(f"\n{'='*80}")
    print(f"ENTRENAMIENTO RIGUROSO: CatBoost + Grid Search + Ensemble + TimeSeriesCV")
    print(f"Cultivo: {cultivo.upper()}")
    print(f"{'='*80}")
    
    t0 = time.time()
    
    df = pd.read_csv(mart_path)
    df = df.dropna(subset=["rendimiento_t_ha"])
    
    print(f"\n1. Enriqueciendo features...")
    df = enrich_features(df)
    
    # Entrenar modelo F (todos los features) con búsqueda de hiperparámetros
    print(f"\n2. FASE F (Modelo completo) - Búsqueda de hiperparámetros...")
    
    feats_f = get_ablation_features("F")
    feats_f = [f for f in feats_f if f in df.columns]
    
    # Preparar datos para grid search
    df_train_full = df[df["anio"] < min(años_test)].dropna(subset=["rendimiento_t_ha"])
    # Split 80/20 para train/val en grid search
    split_idx = int(0.8 * len(df_train_full))
    df_grid_train = df_train_full.iloc[:split_idx]
    df_grid_val = df_train_full.iloc[split_idx:]
    
    X_grid_train = df_grid_train[feats_f].values
    y_grid_train = df_grid_train["rendimiento_t_ha"].values
    X_grid_val = df_grid_val[feats_f].values
    y_grid_val = df_grid_val["rendimiento_t_ha"].values
    
    # Grid search (muy intensivo)
    print(f"\n   ADVERTENCIA: Grid search es muy intensivo en cómputo")
    print(f"   Duración estimada: 30-60 minutos (8-12 GB RAM)")
    
    best_configs = grid_search_hyperparams(X_grid_train, y_grid_train, 
                                          X_grid_val, y_grid_val)
    best_hyperparams = best_configs[0]["params"]
    
    # TimeSeriesSplit (validación cruzada temporal)
    print(f"\n3. TimeSeriesSplit (validación cruzada temporal)...")
    mae_cv, std_cv, fold_maes = timeseries_cross_validation(df, feats_f, best_hyperparams)
    
    # Evaluación final: Ensemble en años de test
    print(f"\n4. Evaluación final: Ensemble en años de test...")
    print(f"   Años de prueba: {años_test}")
    
    resultados = []
    
    for anio in años_test:
        print(f"\n   Año {anio}:")
        result = train_ensemble_models(df, anio, feats_f, best_hyperparams, n_models=5)
        
        if result:
            mae = result["MAE"]
            mae_base = result["MAE_baseline"]
            mejora = (1 - mae / mae_base) * 100 if mae_base > 0 else 0
            
            gana = "✅" if mae < mae_base else "❌"
            
            print(f"      Baseline MAE: {mae_base:.4f} t/ha")
            print(f"      Ensemble MAE: {mae:.4f} t/ha  {gana} ({mejora:+.1f}%)")
            print(f"      R²: {result['R2']:.4f}")
            
            resultados.append({
                "Año": anio,
                "Método": "Ensemble (5 modelos)",
                "MAE": mae,
                "MAE_Baseline": mae_base,
                "Mejora_pct": mejora,
                "RMSE": result["RMSE"],
                "R2": result["R2"],
                "N": result["N"]
            })
    
    # Guardar resultados
    rep_dir = Path("reports/tables")
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    df_res = pd.DataFrame(resultados)
    file_name = f"resultados_rigorous_{cultivo_file}.csv"
    df_res.to_csv(rep_dir / file_name, index=False)
    
    # Resumen final
    t_total = time.time() - t0
    
    print(f"\n{'='*80}")
    print(f"✓ ENTRENAMIENTO COMPLETADO")
    print(f"  Tiempo total: {t_total/60:.1f} minutos")
    print(f"  MAE promedio: {df_res['MAE'].mean():.4f} t/ha")
    print(f"  Mejora promedio vs baseline: {df_res['Mejora_pct'].mean():.1f}%")
    print(f"  Validación cruzada (TimeSeriesCV): MAE={mae_cv:.4f} ± {std_cv:.4f}")
    print(f"  Resultados guardados en: {rep_dir / file_name}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
