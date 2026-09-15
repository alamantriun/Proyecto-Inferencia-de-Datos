# 🤖 CatBoost en AgroRank-Café: Explicación Completa del Modelo

## 1. ¿Qué es CatBoost?

**CatBoost** (Categorical Boosting) es una librería de Machine Learning desarrollada por **Yandex** (empresa rusa, creadora del buscador Yandex). Es un algoritmo de **Gradient Boosting sobre árboles de decisión**, diseñado para:

- Manejar variables categóricas de forma nativa (sin necesidad de convertirlas manualmente a números).
- Ser muy resistente al **overfitting** (cuando el modelo "memoriza" los datos de entrenamiento en vez de aprender patrones reales).
- Dar resultados de alta precisión con poco ajuste manual de hiperparámetros.

### ¿Qué es el Gradient Boosting?

Es una técnica que construye **muchos árboles de decisión pequeños en secuencia**, donde cada árbol nuevo intenta corregir los errores del árbol anterior. Al final, la predicción es la suma ponderada de todos esos árboles.

```
Árbol 1 → comete errores → Árbol 2 los corrige → comete errores → Árbol 3 los corrige → ...
                                                                         ↓
                                                              Predicción Final = Σ(todos los árboles)
```

---

## 2. ¿Qué pregunta responde el modelo en este proyecto?

El modelo **no predice directamente** cuánto café va a producir un municipio. En su lugar, responde:

> *"Dado el historial de producción, las condiciones climáticas, la calidad del suelo, el acceso a crédito y el precio internacional del café, ¿cuál será el **rendimiento en toneladas por hectárea (t/ha)** de este municipio el próximo año?"*

**Variable objetivo (Target):** `rendimiento_t_ha` — toneladas de café producidas por hectárea cosechada.

---

## 3. Los Datos que Recibe el Modelo (Variables de Entrada / Features)

El modelo recibe **6 grupos de variables** organizadas en fases de complejidad creciente (Ablation Study A → F):

### 📊 Grupo A — Memoria Histórica del Cultivo

| Variable | Descripción | Ejemplo |
|---|---|---|
| `rendimiento_lag_1` | Rendimiento del **año pasado** | 1.2 t/ha |
| `rendimiento_lag_2` | Rendimiento de **hace 2 años** | 1.1 t/ha |
| `rendimiento_lag_3` | Rendimiento de **hace 3 años** | 1.3 t/ha |
| `produccion_lag_1` | Producción total del año pasado (toneladas) | 500 t |
| `area_cosechada_lag_1` | Hectáreas cosechadas el año pasado | 420 ha |
| `area_sembrada_lag_1` | Hectáreas sembradas el año pasado | 450 ha |
| `media_rendimiento_3y` | Promedio de rendimiento de los 3 últimos años | 1.2 t/ha |
| `tendencia_rendimiento_3y` | ¿Va subiendo o bajando el rendimiento? | +0.05 t/ha/año |
| `variabilidad_rendimiento_3y` | Qué tan irregular es la producción | 0.12 |
| `score_confiabilidad` | Qué tan confiables son los datos de este municipio | 0.85 |
| `municipio_rend_historico` | Promedio histórico del municipio (Target Encoding) | 1.15 t/ha |
| `depto_rend_historico` | Promedio histórico del departamento | 1.08 t/ha |
| `cambio_area_pct` | ¿El agricultor sembró más o menos? | +10% |
| `ratio_area_sembrada_cosechada` | Eficiencia: área cosechada vs sembrada | 0.93 |

### 🌧️ Grupo B — Variables Climáticas

| Variable | Descripción |
|---|---|
| `precipitacion_acumulada_mm` | Lluvia total acumulada en el año |
| `dias_lluvia` | Número de días con precipitación |
| `precip_Q1_mm` al `precip_Q4_mm` | Lluvia por trimestre (ene-mar, abr-jun, jul-sep, oct-dic) |
| `cv_precipitacion_mensual` | Irregularidad mensual de las lluvias |
| `max_dias_secos_consecutivos` | Sequía más larga del año |
| `intensidad_max_diaria_mm` | Día más lluvioso del año |
| `ratio_concentracion_lluvia` | ¿La lluvia está muy concentrada en pocos días? |

### 🪨 Grupo C — Calidad del Suelo

| Variable | Descripción |
|---|---|
| `ph_media` | Acidez del suelo (ideal para café: 5.5 – 6.5) |
| `materia_organica_pct_media` | % de materia orgánica (nutrientes) |
| `fosforo_ppm_media` | Fósforo disponible para la planta |
| `calcio_meq_media` | Calcio disponible |
| `magnesio_meq_media` | Magnesio disponible |
| `potasio_meq_media` | Potasio disponible |
| `salinidad_ds_m_media` | Salinidad del suelo |
| `tendencia_ph` | ¿El suelo se está volviendo más ácido? |
| `tendencia_materia_organica_pct` | ¿El suelo está mejorando en nutrientes? |

### 🗺️ Grupo D — Aptitud del Territorio

| Variable | Descripción |
|---|---|
| `aptitud_alta_pct` | % de la tierra del municipio muy apta para café |
| `aptitud_media_pct` | % de tierra medianamente apta |
| `aptitud_baja_pct` | % de tierra poco apta |
| `exclusion_legal_pct` | % de tierra en reservas o zonas vedadas |
| `no_apta_pct` | % de tierra no apta para ningún cultivo |

### 💰 Grupo E — Acceso a Financiamiento

| Variable | Descripción |
|---|---|
| `log_credito_total` | Crédito total disponible (escala logarítmica, en COP) |
| `credito_promedio_operacion` | Préstamo promedio por agricultor |
| `num_operaciones_credito` | Cuántos agricultores accedieron a crédito |

### 📈 Grupo F — Precio Internacional del Café (Bolsa de NY)

| Variable | Descripción |
|---|---|
| `precio_internacional_usd` | Precio actual del café en USD/tonelada |
| `cambio_precio_pct` | % de cambio del precio vs el año anterior |
| `volatilidad_precio` | Qué tan inestable fue el precio |
| `rango_precio_ratio` | Diferencia entre precio máximo y mínimo del año |

---

## 4. Parámetros del Modelo CatBoost (Configuración Exacta)

```python
model = CatBoostRegressor(
    iterations    = 2000,   # Número de árboles que construye
    learning_rate = 0.01,   # Qué tanto aprende cada árbol (bajo = más fino y cuidadoso)
    depth         = 6,      # Profundidad de cada árbol (complejidad por árbol)
    l2_leaf_reg   = 5,      # Penalización L2: evita que el modelo se especialice demasiado
    min_data_in_leaf = 15,  # Mínimo de filas que debe tener una "hoja" del árbol
    loss_function = 'RMSE', # Función que minimiza (Error Cuadrático Medio)
    verbose       = 0,      # Sin mensajes en consola durante entrenamiento
    random_seed   = 42      # Semilla aleatoria para reproducibilidad
)
```

### ¿Qué significa cada parámetro?

| Parámetro | Explicación Simple |
|---|---|
| `iterations=2000` | Construye 2000 árboles de decisión pequeños. Más árboles = más preciso pero más lento |
| `learning_rate=0.01` | Cada árbol "avanza" muy poco. Con pasos pequeños se llega más lejos sin pasarse |
| `depth=6` | Cada árbol puede tomar hasta 6 decisiones en cadena (ej: ¿lluvia > 1000mm? → ¿ph > 5.5? → ...) |
| `l2_leaf_reg=5` | Regularización: penaliza modelos muy complejos para evitar que "memorice" los datos |
| `min_data_in_leaf=15` | Cada decisión final del árbol necesita al menos 15 municipios de respaldo |
| `loss_function='RMSE'` | El modelo trata de minimizar el error cuadrático medio entre predicción y realidad |

---

## 5. La Estrategia de Entrenamiento: Rolling-Origin Temporal

El modelo se entrena y evalúa usando una estrategia que **respeta el tiempo**, evitando usar el futuro para predecir el pasado:

```
ITERACIÓN 1:  Entrena con [2012–2018] → Predice 2019 → Mide MAE real
ITERACIÓN 2:  Entrena con [2012–2019] → Predice 2020 → Mide MAE real
ITERACIÓN 3:  Entrena con [2012–2020] → Predice 2021 → Mide MAE real
ITERACIÓN 4:  Entrena con [2012–2021] → Predice 2022 → Mide MAE real
ITERACIÓN 5:  Entrena con [2012–2022] → Predice 2023 → Mide MAE real
ITERACIÓN 6:  Entrena con [2012–2023] → Predice 2024 → Mide MAE real
```

Esto simula exactamente cómo funcionaría el modelo en producción real.

---

## 6. La Predicción Híbrida (ML + Baseline)

El modelo final **no usa CatBoost solo**. La predicción es una combinación ponderada:

```
Predicción Final = α × CatBoost  +  (1-α) × Baseline
```

Donde:
- **CatBoost** detecta desviaciones causadas por clima, precio o suelo.
- **Baseline** es simplemente "el rendimiento del año pasado" (`rendimiento_lag_1`).
- **α** se encuentra automáticamente probando 11 valores (0.0 a 1.0) y escogiendo el que da menor MAE.

### ¿Por qué el híbrido funciona mejor que usar solo ML?

El café tiene **inercia biológica**: una planta establecida tiende a producir de manera similar año a año. Solo eventos extremos (sequías, precios, nuevas variedades) causan desviaciones grandes. Por eso:
- El baseline captura muy bien la tendencia estable.
- CatBoost agrega valor al detectar esas desviaciones extraordinarias.

---

## 7. Las Métricas que Usa para Evaluarse

| Métrica | Fórmula | Interpretación en Café |
|---|---|---|
| **MAE** | Media del error absoluto | "En promedio, me equivoco ±0.18 t/ha" |
| **RMSE** | Raíz del error cuadrático | Penaliza más los errores grandes |
| **R²** | Varianza explicada | 0 = peor que promedio, 1 = perfecto |
| **MAPE** | Error porcentual medio | "Me equivoco ±15% del valor real" |

---

## 8. El Ablation Study: Midiendo el Valor de Cada Grupo

El **Ablation Study** responde: *"¿Cuánto mejora el modelo al agregar cada nuevo grupo de variables?"*

```
Modelo A: Solo historial              → MAE = 0.XXX t/ha
Modelo B: A + Clima                   → MAE = 0.XXX t/ha  (¿bajó el error?)
Modelo C: B + Suelo                   → MAE = 0.XXX t/ha  (¿bajó más?)
Modelo D: C + Territorio              → MAE = 0.XXX t/ha
Modelo E: D + Crédito                 → MAE = 0.XXX t/ha
Modelo F: E + Precio Internacional    → MAE = 0.XXX t/ha  (Modelo completo)
```

Si al agregar un grupo el MAE **baja**, ese grupo de variables es útil. Si no cambia o sube, las variables no aportan información nueva.

---

## 9. El Flujo Completo del Pipeline

```
1. DATOS CRUDOS (datos.gov.co, IDEAM, FRED, AGROSAVIA)
         ↓
2. ETL PASO 1: Limpieza, sin data leakage, creación de lags
         ↓
3. ETL PASO 2: Normalización Z-score de variables numéricas
         ↓
4. ENTRENAMIENTO (02_train_ml.py)
   ├── Target Encoding de municipios
   ├── Búsqueda de alpha óptimo de blending
   ├── Ablation Study A → F
   └── Feature Importance
         ↓
5. EVALUACIÓN (03_plot_results.py, 07_plot_regression.py)
   ├── Gráfica MAE comparativa (Baseline vs Híbrido vs ML puro)
   └── Curva de regresión (predicho vs real)
         ↓
6. RENTABILIDAD (05_rentabilidad.py)
   └── Top 15 municipios por ingreso esperado 2025-2029
         ↓
7. FORECAST FUTURO (06_forecast_futuro.py)
   ├── Simula 2026-2030 con escenarios climáticos y de precio
   └── Calcula probabilidad de éxito (CDF Normal)
```

---

## 10. La Simulación Futura 2026–2030

El modelo entrenado con todos los datos (2012–2024) se usa para proyectar el futuro bajo **escenarios predefinidos**:

| Año | Escenario Climático | Precio Café (USD/t) |
|---|---|---|
| 2026 | Normal | 5,500 |
| 2027 | **Fuerte Sequía (El Niño)** — lluvia -40%, sequía +80% | 4,500 |
| 2028 | Normal | 3,800 |
| 2029 | **Lluvias Fuertes (La Niña)** — lluvia +50% | 3,500 |
| 2030 | Normal | 3,500 |

La **Probabilidad de Éxito** se calcula con la función de distribución normal (CDF):

```
z = (% variación vs línea base 2024) / σ
Probabilidad = Φ(z) × 100%
```

Donde σ = 15% (incertidumbre estimada del modelo). Si el ingreso proyectado supera el de 2024, la probabilidad sube por encima del 50%.

---

## 11. Fuentes de Datos del Proyecto

| Fuente | Dato | URL |
|---|---|---|
| **datos.gov.co (EVA)** | Rendimiento, producción y área histórica | datos.gov.co |
| **datos.gov.co (IDEAM)** | Precipitación y días de lluvia | datos.gov.co |
| **AGROSAVIA** | Análisis de suelos por municipio | datos.gov.co |
| **UPRA** | Aptitud territorial del cultivo | datos.gov.co |
| **FRED (Reserva Federal)** | Precio internacional del café (Bolsa NY) | fred.stlouisfed.org |

---

*Documento generado para el proyecto **AgroRank-Café** — Pipeline de Inferencia de Datos Agrícolas con Machine Learning.*
