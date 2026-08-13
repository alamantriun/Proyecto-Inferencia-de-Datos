# Trazabilidad de Variables y Modelos del Proyecto Agro-Rank

Este documento sirve como "caja negra" o manual de auditoría de datos. Aquí se explica el origen exacto de cada variable, las transformaciones matemáticas aplicadas antes de que las lea el algoritmo, la justificación de la importancia biológica de los rezagos (Lags), y el inventario de los modelos de Machine Learning desplegados.

---

## 1. Trazabilidad de Variables (Data Lineage)

¿De dónde sale exactamente cada columna que lee el modelo?

### A. Variables de Producción e Historia (El "Panel")
*   **Archivo de Origen:** `data/raw/eva_historico_*.csv` (Extraído vía API SODA).
*   **Transformación en `02_build_panel.py`:**
    *   **`rendimiento_lag_1` a `lag_3`:** Se toma la columna cruda `rendimiento_t_ha`, se agrupa por `municipio` y se aplica la operación matemática `.shift(1)` (rezago temporal). Esto trae el valor del año pasado al renglón del año actual.
    *   **`media_rendimiento_3y`:** Se usa la operación `.rolling(window=3).mean()` sobre el rendimiento histórico.
    *   **`tendencia_rendimiento_3y`:** Se usa una regresión lineal rápida sobre los últimos 3 años para obtener la pendiente (positiva o negativa).
    *   **`score_confiabilidad`:** Fórmula propia que penaliza (resta puntos) si la desviación estándar (`.std()`) del rendimiento de un municipio en los últimos 5 años es cero (lo que indica copia y pega gubernamental).

### B. Variables Climáticas
*   **Archivo de Origen:** `data/raw/ideam_precipitacion_*.csv`.
*   **Transformación en `03_extract_ideam.py`:**
    *   **`max_dias_secos_consecutivos`:** Se toma la columna diaria `Valor` (milímetros de lluvia). Se filtran los días con 0 mm. Luego, se cuentan los tramos consecutivos más largos de ceros en el año para cada estación climática, y se hace un *merge* geográfico por código DIVIPOLA.
    *   **`precipitacion_acumulada_mm`:** Es la sumatoria pura (`.sum()`) de la lluvia diaria en un año.
    *   **`cv_precipitacion_mensual`:** Coeficiente de variación (`desviación estándar / media`) de la lluvia agrupada por meses, para medir si llovió parejo o en trombas destructivas.

### C. Variables Edafológicas (Suelos)
*   **Archivo de Origen:** `data/raw/agrosavia_suelos.csv`.
*   **Transformación en `04_extract_agrosavia.py`:**
    *   **`ph_media`:** Promedio histórico por municipio de la columna cruda `ph_agua_suelo_2_5_1_0`.
    *   **`materia_organica_media`:** Promedio de la columna `materia_organica_mo`.

### D. Variables de Inversión y Mercado
*   **FINAGRO (Origen: `creditos_finagro.csv`):** 
    *   La columna `log_credito_total` es la suma de `valor_aprobado` (pesos COP) agrupada por municipio/año. A ese valor masivo se le aplica la operación matemática $Log_{e}(1 + x)$ (`np.log1p`) para normalizar números multimillonarios y evitar que el modelo se abrume.
*   **FRED (Origen: `precio_internacional_*.csv`):** 
    *   La columna `cambio_precio_pct` toma el `precio_usd` anual, calcula el promedio, y usa la operación `.pct_change()` para saber si creció o cayó en porcentaje frente al año previo.
*   **Target Encoding (En `02_train_ml.py`):**
    *   La variable `municipio_rend_historico` se crea agarrando el nombre de texto (ej. "Arauquita") y reemplazándolo por el promedio (`.mean()`) de su columna real `rendimiento_t_ha` en la fase de entrenamiento.

---

## 2. ¿Por qué el `rendimiento_lag_1` es el factor de mayor influencia en CatBoost?

Al ver la gráfica de "Feature Importance" del modelo Final (CatBoost), la variable `rendimiento_lag_1` domina con cerca del 50% de la influencia en cultivos como Cacao y Café. 

Esto **NO** es un error matemático, es el descubrimiento biológico del algoritmo. Se explica por las siguientes razones:

1. **La Fisiología del Cultivo Permanente:** El Cacao y el Café son árboles que viven más de 20 años. A diferencia del maíz, que se siembra de cero cada año, un árbol de cacao sano en 2023 (`lag_1`) tiene casi la obligación física de producir una cantidad similar en 2024. La *inercia biológica* es el factor más determinante de la producción agrícola a largo plazo.
2. **Matemática del Árbol de Decisión:** CatBoost construye un árbol de preguntas. En su primer nodo, CatBoost se da cuenta que separar a los municipios por su cosecha anterior es la forma más rápida de reducir su error de predicción masivamente. Una vez define la base con el `lag_1` (ej. "Este pueblo produce ~1.2t/ha"), usa las variables de clima, precio y suelo para las ramas pequeñas del árbol, haciendo la *sintonía fina* de los decimales (ej. "-0.2t por sequía de El Niño").

---

## 3. ¿Qué modelos están operativos dentro de este proyecto?

El proyecto no usa un único algoritmo cerrado; utiliza un ecosistema modular de varios enfoques predictivos comparados entre sí:

1. **El Modelo Baseline (El "Modelo Tonto" o de Inercia):**
   * Es una fórmula simple: $Producción_{t} = Producción_{t-1}$.
   * *Uso:* Se utiliza como estándar de la industria. Si el algoritmo avanzado no logra ganarle al Baseline, el Machine Learning se considera un fracaso (en cultivos permanentes, este Baseline es muy fuerte y duro de vencer).
2. **El Modelo CatBoost Regressor (Motor de Inteligencia Artificial):**
   * Algoritmo de "Gradient Boosting" sobre Árboles de Decisión.
   * *Uso:* Es el núcleo del proyecto. Se eligió sobre otros (como Random Forest o XGBoost) porque en Colombia muchas agencias dejan datos vacíos (NaN). CatBoost procesa los vacíos nativamente sin forzarnos a inventar datos para rellenarlos (imputación destructiva). 
3. **El Modelo Híbrido (Blending Adaptativo):**
   * *Uso:* El código de entrenamiento evalúa automáticamente tanto al Baseline como al CatBoost. A través de un parámetro matemático (`alpha`), el script "mezcla" las respuestas de ambos según cuál gane en cada iteración, creando un modelo híbrido ultra-resistente.
4. **Los Modelos Multi-Cultivo (Cross-Crop):**
   * El sistema está diseñado de forma dinámica. No hay un solo modelo. El archivo `run_multicrop.sh` genera, entrena y calibra **4 Modelos Independientes**:
      * Especializado en **Cacao**.
      * Especializado en **Café** (alta volatilidad climática).
      * Especializado en **Plátano** (cultivo de seguridad alimentaria transitoria).
      * Especializado en **Arroz** (cultivo transitorio puro, fuertemente condicionado por distritos de riego e infraestructura).
