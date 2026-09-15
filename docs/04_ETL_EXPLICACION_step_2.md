# ETL: Preparación del Dataset de Café para Machine Learning

**Documento explicativo del proceso de transformación**

| Campo | Valor |
|-------|-------|
| Dataset original | `dataset_cafe_modelo_sin_leakage.csv` |
| Dataset resultante | `dataset_cafe_ml_ready.csv` |
| Script ETL | `etl_cafe_ml_ready.py` |
| Filas | 10.973 (sin cambios) |
| Columnas originales → finales | 57 → 48 |
| Objetivo | Predecir `rendimiento_t_ha` (t/ha) a nivel municipio-año |

---

## 1. Contexto y objetivo

El dataset original contiene información de producción de café en Colombia a nivel **departamento – municipio – año** (2007–2024), con variables de:

- Área sembrada / cosechada y producción
- Lags y estadísticas móviles de rendimiento
- Análisis de suelo
- Crédito agropecuario
- Precios internacionales
- Flags de calidad y confiabilidad

El objetivo del ETL es producir una versión **lista para entrenamiento de modelos de ML**, eliminando problemas de *data leakage*, features inútiles y añadiendo indicadores de calidad.

El *split* temporal ya venía definido:

| Split | Años | Uso recomendado |
|-------|------|-----------------|
| `historial` | 2007–2018 | Entrenamiento |
| `target` | 2019–2024 | Validación / Test out-of-time |

---

## 2. Pasos del ETL (resumen)

```
1. Carga del CSV original
2. Eliminación de features con leakage (contemporáneas al target)
3. Eliminación de features constantes o de nula utilidad
4. Creación de flags de calidad / missing
5. Imputación mínima de score_confiabilidad
6. Reordenamiento lógico de columnas
7. Validaciones y guardado
```

---

## 3. Detalle de cada paso

### 3.1 Eliminación de features con leakage

**Columnas eliminadas:**

- `area_sembrada_ha`
- `area_cosechada_ha`
- `produccion_t`

**Motivo:**  
Estas variables son del **mismo año** que el target (`rendimiento_t_ha`).  
El rendimiento se define como:

```text
rendimiento_t_ha ≈ produccion_t / area_cosechada_ha
```

Incluirlas en el modelo sería **data leakage**: el modelo aprendería la definición del target en lugar de predecirlo a partir de información disponible *antes* del año de predicción.

Los lags (`produccion_lag_1`, `area_*_lag_1`, etc.) **sí se conservan**, porque corresponden a años anteriores y están disponibles en el momento de predecir.

---

### 3.2 Eliminación de features constantes / inútiles

**Flags constantes (todo 0 en la auditoría):**

- `es_outlier_rendimiento`
- `inconsistencia_rend`

**Precios internacionales (casi constantes):**

- `precio_internacional_usd`
- `precio_internacional_max`
- `precio_internacional_min`
- `volatilidad_precio`
- `cambio_precio_pct`
- `rango_precio_ratio`

**Motivo de los precios:**  
Entre 2008 y 2023 el valor es prácticamente fijo (3000) y la volatilidad es 0. Solo en 2024 aparece 3500. Aportan nula señal predictiva y se eliminan para simplificar el modelo.

---

### 3.3 Creación de flags de calidad

| Flag nuevo | Definición | Utilidad |
|------------|------------|----------|
| `lags_incompletos` | 1 si falta cualquiera de `rendimiento_lag_1/2/3` | Indica series cortas o inicio de municipio |
| `tiene_credito` | 1 si `credito_total > 0` | El crédito solo aparece de forma significativa desde 2022 |

**Flags que ya existían y se conservan:**

- `score_confiabilidad` (0–1)
- `dato_copiado` (posible dato repetido del año anterior)
- `suelo_disponible` (1 = hay datos de suelo)

**Imputación:**  
Los 6 valores nulos de `score_confiabilidad` se rellenaron con la **mediana** del dataset (≈ 0.958).

---

### 3.4 Variables de suelo

No se imputaron las variables de suelo cuando `suelo_disponible = 0` (≈ 23 % de las filas).  

Motivo: LightGBM / XGBoost manejan nativamente los NaN. Forzar una imputación (media/mediana global) podría introducir sesgo. Se deja el flag `suelo_disponible` para que el modelo aprenda a tratar ambos casos.

---

### 3.5 Orden final de columnas

```
1. Identificadores     → departamento, municipio, cultivo, anio, fuente_eva, split
2. Target              → rendimiento_t_ha
3. Calidad             → score_confiabilidad, dato_copiado, lags_incompletos,
                         suelo_disponible, tiene_credito
4. Lags y rolling      → rendimiento_lag_*, produccion_lag_1, area_*_lag_1,
                         media/variabilidad/tendencia_rendimiento_3y
5. Crédito             → credito_total, colocacion_total, num_operaciones_*,
                         credito_promedio_operacion, log_credito_total
6. Suelo (medias)      → ph, materia orgánica, fósforo, calcio, magnesio,
                         potasio, salinidad, num_muestras
7. Suelo (variabilidad)
8. Suelo (tendencias)
```

---

## 4. Resultado del ETL

| Métrica | Valor |
|---------|-------|
| Filas | 10.973 |
| Columnas | 48 |
| Train (`historial`) | 7.206 |
| Test (`target`) | 3.767 |
| Target sin nulls | Sí |
| Leakage de área/producción | Eliminado |
| Precios constantes | Eliminados |

**Archivo de salida:**  
`/home/workdir/artifacts/dataset_cafe_ml_ready.csv`

---

## 5. Cómo usar el dataset limpio

```python
import pandas as pd

df = pd.read_csv("dataset_cafe_ml_ready.csv")

train = df[df["split"] == "historial"].copy()   # 2007-2018
test  = df[df["split"] == "target"].copy()      # 2019-2024

y_train = train["rendimiento_t_ha"]
X_train = train.drop(columns=[
    "rendimiento_t_ha", "split", "fuente_eva", "cultivo"
    # + encoding de departamento y municipio
])
```

**Recomendaciones adicionales al entrenar:**

1. No volver a incluir `area_*` ni `produccion_t`.
2. Usar `score_confiabilidad` como *sample_weight* (opcional).
3. Evaluar también un escenario **sin el año 2019** (año atípico detectado en la auditoría).
4. Codificar `departamento` y `municipio` (Label Encoding o Target Encoding). Tratar municipios no vistos en train como categoría desconocida.

---

## 6. Cómo reproducir el ETL

```bash
# Desde la carpeta de trabajo
python etl_cafe_ml_ready.py \
  --input  /ruta/a/dataset_cafe_modelo_sin_leakage.csv \
  --output /ruta/a/dataset_cafe_ml_ready.csv
```

O desde Python:

```python
from etl_cafe_ml_ready import run_etl

df_limpio = run_etl(
    input_path="dataset_cafe_modelo_sin_leakage.csv",
    output_path="dataset_cafe_ml_ready.csv",
    verbose=True,
)
```

---

## 7. Decisiones de diseño (resumen)

| Decisión | Justificación |
|----------|---------------|
| Quitar área y producción contemporáneas | Evitar leakage (definición del target) |
| Conservar lags de área/producción | Información del pasado, válida para predicción |
| Quitar precios internacionales | Prácticamente constantes → nula varianza |
| No imputar suelo cuando falta | Evitar sesgo; el modelo puede usar el flag |
| Crear `lags_incompletos` y `tiene_credito` | Señales explícitas de calidad / disponibilidad |
| Mantener el split original | Evaluación temporal realista (out-of-time) |
| No eliminar filas | Se prefiere filtrar después según el caso de uso |

---

## 8. Archivos relacionados

| Archivo | Descripción |
|---------|-------------|
| `dataset_cafe_modelo_sin_leakage.csv` | Original (entrada del ETL) |
| `dataset_cafe_ml_ready.csv` | Salida del ETL (listo para ML) |
| `etl_cafe_ml_ready.py` | Script reproducible del ETL |
| `ETL_EXPLICACION.md` | Este documento |

---

*Documento generado como parte de la auditoría y preparación del dataset de café para modelos de predicción de rendimiento.*
