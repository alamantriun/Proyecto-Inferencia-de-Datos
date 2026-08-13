# Proyecto Predicción de Rendimiento de Cacao (Colombia)

Pipeline de Ingeniería de Datos y Machine Learning para inferir el rendimiento ex-ante (toneladas por hectárea) del cultivo de Cacao a nivel municipal en Colombia.

## Arquitectura de Datos
Este proyecto integra múltiples fuentes de datos abiertos del gobierno colombiano:
1. **EVA (UPRA/Agronet)**: Evaluaciones Agropecuarias Municipales (Área, Producción, Rendimiento).
2. **IDEAM**: Precipitación climática.
3. **AGROSAVIA**: Propiedades químicas del suelo (pH, Fósforo, Calcio, Salinidad).
4. **UPRA**: Mapas de aptitud territorial para el cultivo de cacao.
5. **FINAGRO**: Colocaciones y desembolsos de crédito agropecuario.
6. **FRED (Reserva Federal EE.UU.)**: Precios internacionales de futuros de cacao para simular incentivos de mercado.

## Estructura del Repositorio
- `data/`: Almacenamiento local (ignorado en git excepto diccionarios).
  - `external/`: CSVs de configuración y catálogos fijos.
  - `raw/`: Descargas brutas de las APIs.
  - `processed/`: Tablas limpias, panel temporal y Model Mart final.
- `src/`: Código fuente principal.
  - `data/`: Scripts de extracción web (API SODA).
  - `features/`: Ingeniería de características (Lags) y cruce maestro.
  - `models/`: Entrenamiento de heurísticas (Baselines) y ML (CatBoost).
- `reports/`:
  - `tables/`: CSVs con métricas (MAE, RMSE, MAPE).
  - `figures/`: Gráficos de barras comparativos de desempeño.

## Prevención de Data Leakage
El pipeline respeta estrictamente la temporalidad. Cualquier característica usada para predecir el año $t$ es calculada exclusivamente con información disponible hasta el año $t-1$.

## Ejecución del Pipeline
El repositorio está diseñado de manera secuencial. Puedes ejecutar la tubería completa de extracción, procesamiento, modelado y graficado ejecutando:

```bash
./run_pipeline.sh
```

## Demostración Científica
El repositorio incluye un script especial de control (`src/models/04_demo_transitorios.py`) que comprueba la hipótesis científica del proyecto: al evaluar cultivos de ciclo corto (como el Arroz), los modelos predictivos sí logran modelar la varianza cuando se inyecta clima, demostrando que la dependencia a la inercia (Baseline) es exclusiva de los cultivos permanentes.

## Visualización (Dashboard)
El proyecto incluye un dashboard analítico en **Streamlit**. Para ejecutarlo localmente:

```bash
streamlit run app.py
```

## Requisitos
Se recomienda el uso de un entorno virtual (`venv`):
```bash
pip install -r requirements.txt
```

---
*Proyecto de Ingeniería de Datos para Machine Learning Agrícola - 2024*
