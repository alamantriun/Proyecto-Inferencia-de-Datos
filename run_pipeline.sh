#!/bin/bash

# ==============================================================================
# Pipeline de Datos e Inferencia: Proyecto Rendimiento Agrícola
# ==============================================================================
# Este script ejecuta secuencialmente todas las etapas de extracción,
# procesamiento, feature engineering y entrenamiento del modelo predictivo.
# ==============================================================================

set -e # Detener la ejecución si hay un error

OFFLINE_MODE="${OFFLINE_MODE:-0}"

if [[ "$OFFLINE_MODE" == "1" || "$OFFLINE_MODE" == "true" || "$OFFLINE_MODE" == "TRUE" ]]; then
  echo "Modo offline activado: usando datos locales ya presentes en disco."
  echo "==========================================="

  if [[ -f "reports/tablas_entrenamiento/dataset_cafe_ml_ready.csv" ]]; then
    echo "Copiando dataset_cafe_ml_ready.csv a data/processed/model_mart_cafe.csv"
    cp "reports/tablas_entrenamiento/dataset_cafe_ml_ready.csv" "data/processed/model_mart_cafe.csv"
  elif [[ ! -f "data/processed/model_mart_cafe.csv" ]]; then
    echo "Error: faltan los Model Mart locales. No se puede ejecutar en modo offline."
    exit 1
  fi

  echo -e "\n[8/15] Aplicando ETL v3 sobre Model Mart (normaliza features, preserva IDs)..."
  python3 src/etl/etl_normalizacion.py
  for cultivo in cafe; do
    cp "data/processed/etl_normalized_v3/model_mart_${cultivo}.csv" "data/processed/model_mart_${cultivo}.csv"
    python3 - "$cultivo" <<'PY'
import pandas as pd
import sys
c = sys.argv[1]
path = f'data/processed/model_mart_{c}.csv'
df = pd.read_csv(path)
if 'score_confiabilidad' in df.columns:
    df['score_confiabilidad'] = pd.to_numeric(df['score_confiabilidad'], errors='coerce').clip(0, 1)
    df.to_csv(path, index=False)
print(f'clip score_confiabilidad -> {path}')
PY
  done

  echo -e "\n[9/15] Evaluando heurísticas base (Baselines)..."
  python3 src/models/01_eval_baselines.py

  echo -e "\n[10/15] Entrenando Machine Learning y Ablation Study (CatBoost)..."
  python3 src/models/02_train_ml.py

  echo -e "\n[11/15] Generando reportes gráficos..."
  python3 src/models/03_plot_results.py

  echo -e "\n[12/15] Generando gráfica de regresión (Real vs Predicho)..."
  python3 src/models/07_plot_regression.py

  echo -e "\n[13/15] Proyectando métricas de negocio (Rentabilidad Ex-Ante)..."
  python3 src/models/05_rentabilidad.py

  echo -e "\n[14/15] Simulando Escenarios Futuros (2025-2029)..."
  python3 src/models/06_forecast_futuro.py

  echo -e "\n==========================================="
  echo "Pipeline offline completado con éxito."
  echo "Los resultados están disponibles en reports/"
  exit 0
fi

echo "Iniciando Pipeline de Inferencia Agrícola..."
echo "==========================================="

echo -e "\n[1/15] Extrayendo datos de producción (EVA)..."
python3 src/data/01_extract_eva.py

echo -e "\n[2/15] Construyendo panel temporal y Lags (Reglas Leakage)..."
python3 src/features/02_build_panel.py

echo -e "\n[3/15] Extrayendo variables climáticas (IDEAM)..."
python3 src/data/03_extract_ideam.py

echo -e "\n[4/15] Extrayendo perfiles de suelo (AGROSAVIA)..."
python3 src/data/04_extract_agrosavia.py

echo -e "\n[5/15] Extrayendo mapas de aptitud (UPRA)..."
python3 src/data/05_extract_upra.py

echo -e "\n[6/15] Extrayendo datos económicos/crédito (FINAGRO)..."
python3 src/data/07_extract_finagro.py

echo -e "\n[7/15] Extrayendo precios internacionales (FRED)..."
python3 src/data/08_extract_precio_internacional.py

echo -e "\n[8/15] Ensamblando Model Mart Final..."
python3 src/features/06_build_model_mart.py

echo -e "\n[9/15] Aplicando ETL v3 sobre Model Mart (normaliza features, preserva IDs)..."
python3 src/etl/etl_normalizacion.py
for cultivo in cafe; do
  cp "data/processed/etl_normalized_v3/model_mart_${cultivo}.csv" "data/processed/model_mart_${cultivo}.csv"
done

echo -e "\n[10/15] Evaluando heurísticas base (Baselines)..."
python3 src/models/01_eval_baselines.py

echo -e "\n[11/15] Entrenando Machine Learning y Ablation Study (CatBoost)..."
python3 src/models/02_train_ml.py

echo -e "\n[12/15] Generando reportes gráficos..."
python3 src/models/03_plot_results.py

echo -e "\n[13/15] Generando gráfica de regresión (Real vs Predicho)..."
python3 src/models/07_plot_regression.py

echo -e "\n[14/15] Proyectando métricas de negocio (Rentabilidad Ex-Ante)..."
python3 src/models/05_rentabilidad.py

echo -e "\n[15/15] Simulando Escenarios Futuros (2025-2029)..."
python3 src/models/06_forecast_futuro.py

echo -e "\n==========================================="
echo "Pipeline completado con éxito."
echo "Los resultados están disponibles en reports/"
