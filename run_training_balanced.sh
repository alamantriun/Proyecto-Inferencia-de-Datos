#!/bin/bash
# ==============================================================================
# 🚀 Script de Entrenamiento ML BALANCED
# ==============================================================================
# 
# Este script ejecuta el entrenamiento en modo BALANCED:
#   • Grid Search: Prueba 16 combinaciones de hiperparámetros
#   • Ensemble: 5 modelos independientes
#   • SIN TimeSeriesCV: Para acelerar
#
# Requisitos:
#   • Python 3.8+
#   • pandas, numpy, catboost, sklearn
#   • 8-10 GB RAM disponible
#   • 60 minutos de tiempo
#
# Uso:
#   ./run_training_balanced.sh
#
# ==============================================================================

set -e  # Salir si hay error
# Colors para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# VALIDACIONES PREVIAS
# ==============================================================================

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Entrenamiento ML - Modo BALANCED${NC}"
echo -e "${BLUE}================================${NC}\n"

# Verificar que estamos en el directorio correcto
if [ ! -f "config/config.yaml" ]; then
    echo -e "${RED}❌ Error: No se encuentra config/config.yaml${NC}"
    echo "Por favor ejecuta este script desde la raíz del proyecto:"
    echo "  cd /ruta/a/proyecto IngDatos"
    echo "  ./run_training_balanced.sh"
    exit 1
fi

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: Python 3 no está instalado${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Directorio del proyecto: $(pwd)${NC}"
echo -e "${GREEN}✓ Python: $(python3 --version)${NC}"

# Verificar librerías
echo -e "\n${BLUE}Verificando librerías requeridas...${NC}"

python3 -c "import pandas" 2>/dev/null && echo -e "${GREEN}✓ pandas${NC}" || (echo -e "${RED}✗ pandas${NC}" && exit 1)
python3 -c "import numpy" 2>/dev/null && echo -e "${GREEN}✓ numpy${NC}" || (echo -e "${RED}✗ numpy${NC}" && exit 1)
python3 -c "import catboost" 2>/dev/null && echo -e "${GREEN}✓ catboost${NC}" || (echo -e "${RED}✗ catboost${NC}" && exit 1)
python3 -c "import sklearn" 2>/dev/null && echo -e "${GREEN}✓ scikit-learn${NC}" || (echo -e "${RED}✗ scikit-learn${NC}" && exit 1)

# ==============================================================================
# ESTIMACIONES DE TIEMPO Y RECURSOS
# ==============================================================================

echo -e "\n${BLUE}Configuración BALANCED:${NC}"
echo "  • Grid Search: 4×2×2×2 = 16 combinaciones"
echo "  • Ensemble: 5 modelos por año"
echo "  • TimeSeriesCV: DESHABILITADO (para velocidad)"
echo -e "\n  ${YELLOW}⏱️  Tiempo estimado: 45-75 minutos${NC}"
echo -e "  ${YELLOW}💾 RAM requerido: 8-10 GB${NC}"
echo -e "  ${YELLOW}🔥 CPU: Máximo uso durante Grid Search${NC}"

# Verificar espacio en disco
DISK_FREE=$(df . | awk 'NR==2 {print $4}')
if [ "$DISK_FREE" -lt 1048576 ]; then  # 1 GB en KB
    echo -e "${YELLOW}⚠️  Advertencia: Espacio en disco limitado${NC}"
fi

# ==============================================================================
# INICIO DEL ENTRENAMIENTO
# ==============================================================================

echo -e "\n${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}INICIANDO ENTRENAMIENTO EN MODO BALANCED${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}\n"

start_time=$(date +%s)

# Ejecutar el script Python con modo BALANCED
python3 << 'PYTHON_SCRIPT'
import sys
import os
from pathlib import Path

# Importar las funciones del script riguroso
sys.path.insert(0, str(Path(__file__).parent))

# ── CONFIGURACIÓN BALANCED ──
CONFIG = {
    "MODE": "balanced",
    "GRID_SEARCH": True,
    "ENSEMBLE": True,
    "TIMESERIES_CV": False,
    "CATBOOST_ITERATIONS": 2000,
    "CATBOOST_LEARNING_RATE": 0.015,
    "CATBOOST_DEPTH": 6,
    "N_ENSEMBLE_MODELS": 5,
    "GRID_SEARCH_PARAM_GRID": {
        "depth": [5, 6],
        "learning_rate": [0.01, 0.015],
        "l2_leaf_reg": [5, 10],
        "subsample": [0.8, 0.9],
    }
}

# Ejecutar versión simplificada (sin TimeSeriesCV)
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
import yaml
import time
from datetime import datetime

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

def get_ablation_features(model_id):
    features = []
    keys = list(FEATURE_GROUPS.keys())
    idx = keys.index(model_id)
    for k in keys[:idx+1]:
        features.extend(FEATURE_GROUPS[k])
    return features

def enrich_features(df):
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
        df["cambio_area_pct"] = ((df["area_sembrada_lag_1"] - area_lag2) / area_lag2.replace(0, np.nan)).fillna(0).clip(-1, 5)
        df["ratio_area_sembrada_cosechada"] = (df["area_sembrada_lag_1"] / df["area_cosechada_lag_1"].replace(0, np.nan)).fillna(1).clip(0.1, 10)
    return df

def grid_search_hyperparams(X_train, y_train, X_val, y_val, param_grid, top_k=1):
    print(f"\n  🔧 GRID SEARCH: Probando {len(list(ParameterGrid(param_grid)))} combinaciones...")
    results = []
    params_list = list(ParameterGrid(param_grid))
    
    for i, params in enumerate(params_list, 1):
        config = {
            "iterations": CONFIG["CATBOOST_ITERATIONS"],
            "learning_rate": CONFIG["CATBOOST_LEARNING_RATE"],
            "depth": CONFIG["CATBOOST_DEPTH"],
            "l2_leaf_reg": 10,
            "min_data_in_leaf": 5,
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
            "early_stopping_rounds": 100,
            "loss_function": "RMSE",
            "verbose": 0,
            "random_seed": 42
        }
        config.update(params)
        
        pool_train = Pool(X_train, y_train)
        pool_val = Pool(X_val, y_val)
        
        model = CatBoostRegressor(**config)
        model.fit(pool_train, eval_set=pool_val, verbose=0)
        
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        
        results.append({"params": params, "MAE": mae})
        print(f"    [{i:2d}/{len(params_list)}] depth={params.get('depth', '?')} lr={params.get('learning_rate', '?'):.4f} → MAE={mae:.4f}")
    
    results_sorted = sorted(results, key=lambda x: x["MAE"])
    print(f"\n  ✓ Mejor configuración: MAE={results_sorted[0]['MAE']:.4f}")
    return results_sorted[0]["params"]

def train_ensemble_models(df, target_year, features, best_hyperparams, n_models=5):
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
    
    print(f"      Ensemble de {n_models} modelos...", end=" ", flush=True)
    ensemble_preds = []
    
    for seed_idx in range(n_models):
        config = {
            "iterations": CONFIG["CATBOOST_ITERATIONS"],
            "learning_rate": CONFIG["CATBOOST_LEARNING_RATE"],
            "depth": CONFIG["CATBOOST_DEPTH"],
            "l2_leaf_reg": 10,
            "min_data_in_leaf": 5,
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
            "early_stopping_rounds": 100,
            "loss_function": "RMSE",
            "verbose": 0,
            "random_seed": 42 + seed_idx
        }
        config.update(best_hyperparams)
        
        sample_weight = None
        if "score_confiabilidad" in train.columns:
            sample_weight = train["score_confiabilidad"].fillna(0.5).clip(0, 1).values
        
        model = CatBoostRegressor(**config)
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=0)
        
        y_pred = model.predict(X_test)
        ensemble_preds.append(y_pred)
    
    y_pred_ensemble = np.mean(ensemble_preds, axis=0)
    y_pred_ensemble = np.clip(y_pred_ensemble, 0, None)
    
    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
    mae_baseline = mean_absolute_error(y_test, y_baseline)
    
    print("✓")
    
    return {
        "MAE": mae_ensemble,
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_ensemble)),
        "R2": r2_score(y_test, y_pred_ensemble),
        "MAE_baseline": mae_baseline,
        "N": len(y_test)
    }

# ── MAIN EXECUTION ──
config_path = Path("config/config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config_yaml = yaml.safe_load(f)

cultivo = config_yaml["project"].get("cultivo_mvp", "cacao")
cultivo_file = cultivo.lower().replace(' ', '_')
años_test = config_yaml["project"].get("años_backtest", [2019, 2020, 2021, 2022, 2023, 2024])

mart_path = Path(f"data/processed/model_mart_{cultivo_file}.csv")

print(f"\n📊 Cultivo: {cultivo.upper()}")
print(f"📁 Datos: {mart_path}")
print(f"📅 Años de prueba: {años_test}")

df = pd.read_csv(mart_path)
df = df.dropna(subset=["rendimiento_t_ha"])

print(f"\nEnriqueciendo features...")
df = enrich_features(df)

feats_f = get_ablation_features("F")
feats_f = [f for f in feats_f if f in df.columns]

print(f"\nBúsqueda de hiperparámetros óptimos...")

df_train_full = df[df["anio"] < min(años_test)].dropna(subset=["rendimiento_t_ha"])
split_idx = int(0.8 * len(df_train_full))
df_grid_train = df_train_full.iloc[:split_idx]
df_grid_val = df_train_full.iloc[split_idx:]

X_grid_train = df_grid_train[feats_f].values
y_grid_train = df_grid_train["rendimiento_t_ha"].values
X_grid_val = df_grid_val[feats_f].values
y_grid_val = df_grid_val["rendimiento_t_ha"].values

best_hyperparams = grid_search_hyperparams(
    X_grid_train, y_grid_train, X_grid_val, y_grid_val,
    CONFIG["GRID_SEARCH_PARAM_GRID"]
)

print(f"\nEvaluación en años de prueba...")
resultados = []

for anio in años_test:
    print(f"\n  Año {anio}:")
    result = train_ensemble_models(
        df, anio, feats_f, best_hyperparams,
        n_models=CONFIG["N_ENSEMBLE_MODELS"]
    )
    
    if result:
        mae = result["MAE"]
        mae_base = result["MAE_baseline"]
        mejora = (1 - mae / mae_base) * 100 if mae_base > 0 else 0
        gana = "✅" if mae < mae_base else "❌"
        
        print(f"    Baseline MAE: {mae_base:.4f} t/ha")
        print(f"    Ensemble MAE: {mae:.4f} t/ha  {gana} ({mejora:+.1f}%)")
        print(f"    R²: {result['R2']:.4f}")
        
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

# Guardar
rep_dir = Path("reports/tables")
rep_dir.mkdir(parents=True, exist_ok=True)

df_res = pd.DataFrame(resultados)
file_name = f"resultados_rigorous_{cultivo_file}.csv"
df_res.to_csv(rep_dir / file_name, index=False)

print(f"\n✓ Resultados guardados en: {rep_dir / file_name}")
print(f"  MAE promedio: {df_res['MAE'].mean():.4f} t/ha")
print(f"  Mejora promedio: {df_res['Mejora_pct'].mean():.1f}%")

PYTHON_SCRIPT

end_time=$(date +%s)
elapsed=$((end_time - start_time))

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

echo -e "\n${BLUE}════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}\n"

echo -e "⏱️  Tiempo total: $((elapsed / 60)) minutos $((elapsed % 60)) segundos"
echo -e "📊 Resultados guardados en: reports/tables/resultados_rigorous_*.csv"
echo -e "\n${YELLOW}Próximos pasos:${NC}"
echo "  1. Revisar los resultados en reports/tables/"
echo "  2. Comparar con versión estándar (si quieres): python3 scripts/compare_versions.py"
echo "  3. Para generar visualizaciones: python3 src/models/03_plot_results.py"
echo ""
