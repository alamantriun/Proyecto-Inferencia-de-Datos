#!/bin/bash

# ==============================================================================
# Pipeline de Datos e Inferencia: Proyecto Rendimiento Agrícola
# ==============================================================================
# Este script ejecuta secuencialmente todas las etapas de extracción,
# procesamiento, feature engineering y entrenamiento del modelo predictivo.
# ==============================================================================

set -e # Detener la ejecución si hay un error

echo "Iniciando Pipeline de Inferencia Agrícola..."
echo "==========================================="

echo -e "\n[1/13] Extrayendo datos de producción (EVA)..."
python3 src/data/01_extract_eva.py

echo -e "\n[2/13] Construyendo panel temporal y Lags (Reglas Leakage)..."
python3 src/features/02_build_panel.py

echo -e "\n[3/13] Extrayendo variables climáticas (IDEAM)..."
python3 src/data/03_extract_ideam.py

echo -e "\n[4/13] Extrayendo perfiles de suelo (AGROSAVIA)..."
python3 src/data/04_extract_agrosavia.py

echo -e "\n[5/13] Extrayendo mapas de aptitud (UPRA)..."
python3 src/data/05_extract_upra.py

echo -e "\n[6/13] Extrayendo datos económicos/crédito (FINAGRO)..."
python3 src/data/07_extract_finagro.py

echo -e "\n[7/13] Extrayendo precios internacionales (FRED)..."
python3 src/data/08_extract_precio_internacional.py

echo -e "\n[8/13] Ensamblando Model Mart Final..."
python3 src/features/06_build_model_mart.py

echo -e "\n[9/13] Evaluando heurísticas base (Baselines)..."
python3 src/models/01_eval_baselines.py

echo -e "\n[10/13] Entrenando Machine Learning y Ablation Study (CatBoost)..."
python3 src/models/02_train_ml.py

echo -e "\n[11/13] Generando reportes gráficos..."
python3 src/models/03_plot_results.py

echo -e "\n[12/14] Generando gráfica de regresión (Real vs Predicho)..."
python3 src/models/07_plot_regression.py

echo -e "\n[13/14] Proyectando métricas de negocio (Rentabilidad Ex-Ante)..."
python3 src/models/05_rentabilidad.py

echo -e "\n[14/14] Simulando Escenarios Futuros (2025-2029)..."
python3 src/models/06_forecast_futuro.py

echo -e "\n==========================================="
echo "Pipeline completado con éxito."
echo "Los resultados están disponibles en reports/"
