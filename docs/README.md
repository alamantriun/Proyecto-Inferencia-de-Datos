# 📊 Tablas de Entrenamiento - Modelos de Rendimiento Agrícola

## 📌 Resumen Ejecutivo

Este documento presenta las **tablas completas de entrenamiento** (model_mart) utilizadas para entrenar los tres modelos principales:
- **Arroz**: 7,657 observaciones
- **Cacao**: 11,666 observaciones  
- **Café**: 12,278 observaciones
- **Plátano**: 16,404 observaciones

**Total**: 48,005 observaciones consolidadas con variables de 7 fuentes de datos diferentes.

---

## 🏗️ Arquitectura de Variables

Las tablas de entrenamiento contienen **61 variables** agrupadas en 7 categorías:

### A. **Identificadores Geográficos y Temporales** (6 variables)
Información básica sobre ubicación, cultivo y período temporal.

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `departamento` | Texto | Departamento/región administrativa |
| `municipio` | Texto | Municipio de origen |
| `cultivo` | Texto | Cultivo específico (Arroz, Cacao, Café, Plátano) |
| `anio` | Número | Año de observación |
| `fuente_eva` | Texto | Fuente de datos EVA (historica/reciente) |
| `split` | Texto | Asignación de conjunto (train/test/historial) |

---

### B. **Producción e Historia (Panel)** - 17 variables
Variables derivadas de EVA (Evaluaciones Agrícolas Estatales) con transformaciones temporales.

#### Variables Básicas
| Variable | Fórmula/Origen | Descripción |
|----------|---|-------------|
| `area_sembrada_ha` | Directa de EVA | Hectáreas destinadas al cultivo en el año t |
| `area_cosechada_ha` | Directa de EVA | Hectáreas efectivamente cosechadas en el año t |
| `produccion_t` | Directa de EVA | Toneladas producidas en el año t |
| `rendimiento_t_ha` | `produccion_t / area_cosechada_ha` | **TARGET VARIABLE** - Rendimiento (t/ha) |

#### Variables Rezagadas (Lags)
Capturan el efecto temporal del rendimiento anterior sobre el actual.

| Variable | Fórmula | Descripción |
|----------|---------|-------------|
| `rendimiento_lag_1` | `rendimiento_t_ha` del año (t-1) | Rendimiento del año anterior |
| `rendimiento_lag_2` | `rendimiento_t_ha` del año (t-2) | Rendimiento hace 2 años |
| `rendimiento_lag_3` | `rendimiento_t_ha` del año (t-3) | Rendimiento hace 3 años |
| `produccion_lag_1` | Toneladas del año (t-1) | Producción del año anterior |
| `area_cosechada_lag_1` | Hectáreas cosechadas (t-1) | Área cosechada anterior |
| `area_sembrada_lag_1` | Hectáreas sembradas (t-1) | Área sembrada anterior |

#### Variables Agregadas Históricas
Resumen estadístico de los últimos 3 años.

| Variable | Fórmula | Descripción |
|----------|---------|-------------|
| `media_rendimiento_3y` | `rendimiento_t_ha[t-3:t].mean()` | Promedio rendimiento últimos 3 años |
| `variabilidad_rendimiento_3y` | `rendimiento_t_ha[t-3:t].std()` | Desviación estándar últimos 3 años |
| `tendencia_rendimiento_3y` | Regresión lineal (slope) | Pendiente de tendencia últimos 3 años |

#### Variables de Calidad
Indicadores de confiabilidad y problemas en datos.

| Variable | Fórmula | Descripción |
|----------|---------|-------------|
| `es_outlier_rendimiento` | IQR method | Bandera si es outlier (1=sí, 0=no) |
| `dato_copiado` | `variabilidad == 0` por municipio | Detecta datos gubernamentales copiados (1=sí) |
| `inconsistencia_rend` | Comparación área vs producción | Inconsistencias lógicas (1=sí) |
| `score_confiabilidad` | Fórmula ponderada | Score 0-1 de confiabilidad del dato |

---

### C. **Clima (IDEAM)** - 8 variables
Datos de precipitación y sequía del Instituto de Hidrología, Meteorología y Estudios Ambientales.

**⚠️ LAG TEMPORAL**: El clima del año `t` se empareja con el rendimiento del año `t+1` (el clima de 2018 explica rendimiento de 2019).

| Variable | Unidad | Derivación | Descripción |
|----------|--------|-----------|-------------|
| `precipitacion_acumulada_mm` | mm | `sum(lluvia_diaria)` | Suma total de lluvia anual |
| `dias_lluvia` | días | `count(lluvia > 0)` | Número de días con precipitación |
| `intensidad_max_diaria_mm` | mm | `max(lluvia_diaria)` | Evento máximo de lluvia en un día |
| `precipitacion_mediana_diaria` | mm | `median(lluvia_diaria \| lluvia > 0)` | Lluvia mediana en días lluviosos |
| `precip_Q3_mm` | mm | `quantile(lluvia_diaria, 0.75)` | Tercer cuartil de precipitación |
| `ratio_concentracion_lluvia` | ratio | `intensidad_max / precipitacion_acumulada` | Concentración de lluvia |
| `cv_precipitacion_mensual` | % | `std(lluvia_mensual) / mean(lluvia_mensual)` | Coeficiente variación mensual |
| `max_dias_secos_consecutivos` | días | `max(tramo de días con lluvia = 0)` | Sequía más larga del año |

---

### D. **Suelos (AGROSAVIA)** - 15 variables
Variables edafológicas del Servicio Geológico Colombiano.

**Características**: Datos estructurales a nivel municipio (no varían por año).

#### Propiedades Principales
| Variable | Unidad | Significado |
|----------|--------|-------------|
| `ph_media` | pH | Promedio de pH del suelo (acidez/alcalinidad) |
| `materia_organica_pct_media` | % | Contenido promedio de materia orgánica |
| `fosforo_ppm_media` | ppm | Disponibilidad de fósforo |
| `calcio_meq_media` | meq/100g | Concentración de calcio intercambiable |
| `magnesio_meq_media` | meq/100g | Concentración de magnesio intercambiable |
| `potasio_meq_media` | meq/100g | Concentración de potasio intercambiable |
| `salinidad_ds_m_media` | dS/m | Conductividad eléctrica (salinidad) |

#### Variabilidad de Suelos
Dispersión espacial de propiedades dentro del municipio.

| Variable | Descripción |
|----------|-------------|
| `ph_variabilidad` | Desviación estándar del pH |
| `materia_organica_pct_variabilidad` | Variabilidad de materia orgánica |
| `fosforo_ppm_variabilidad` | Variabilidad del fósforo |
| `calcio_meq_variabilidad` | Variabilidad del calcio |
| `magnesio_meq_variabilidad` | Variabilidad del magnesio |
| `potasio_meq_variabilidad` | Variabilidad del potasio |
| `salinidad_ds_m_variabilidad` | Variabilidad de salinidad |

#### Metadatos de Suelos
| Variable | Descripción |
|----------|-------------|
| `num_muestras_suelo` | Número de muestras tomadas en el municipio |

#### Tendencias de Suelo (7 variables)
Cambio linear de propiedades a lo largo del tiempo (cuando hay series históricas).

| Variable | Descripción |
|----------|-------------|
| `tendencia_ph` | Pendiente temporal del pH |
| `tendencia_materia_organica_pct` | Cambio anual de materia orgánica |
| `tendencia_fosforo_ppm` | Cambio anual de fósforo |
| `tendencia_calcio_meq` | Cambio anual de calcio |
| `tendencia_magnesio_meq` | Cambio anual de magnesio |
| `tendencia_potasio_meq` | Cambio anual de potasio |
| `tendencia_salinidad_ds_m` | Cambio anual de salinidad |

---

### E. **Inversión Agrícola (FINAGRO)** - 5 variables
Datos de créditos agropecuarios del Fondo para el Financiamiento del Sector Agropecuario.

**⚠️ LAG TEMPORAL**: Créditos del año `t` se empareja con rendimiento del año `t+1`.

| Variable | Unidad | Fórmula | Descripción |
|----------|--------|---------|-------------|
| `credito_total` | COP | `sum(valor_aprobado)` | Monto total de créditos aprobados |
| `colocacion_total` | COP | `sum(valor_colocado)` | Monto efectivamente desembolsado |
| `num_operaciones_credito` | count | `count(operaciones)` | Número de operaciones de crédito |
| `credito_promedio_operacion` | COP | `credito_total / num_operaciones` | Crédito promedio por operación |
| `log_credito_total` | log(COP) | `log(1 + credito_total)` | Logaritmo natural normalizado |

**Nota**: `log_credito_total` se utiliza en el modelo para normalizar valores multimillonarios y evitar dominancia numérica.

---

### F. **Precio Internacional (FRED)** - 6 variables
Precios de commodities del Federal Reserve Economic Data (FRED).

**⚠️ LAG TEMPORAL**: Precios del año `t` se emparejan con rendimiento del año `t+1`.

| Cultivo | Commodity | Código FRED |
|---------|-----------|------------|
| Arroz | Rough Rice | POILRICE |
| Cacao | Cacao | POOTHU02 |
| Café | Arabica Coffee | POILAPRA |
| Plátano | Bananas | POILAPBA |

Variables derivadas:

| Variable | Unidad | Fórmula | Descripción |
|----------|--------|---------|-------------|
| `precio_internacional_usd` | USD/unidad | Directa de FRED | Precio promedio anual |
| `precio_internacional_max` | USD/unidad | `max(precio_mensual)` | Precio máximo del año |
| `precio_internacional_min` | USD/unidad | `min(precio_mensual)` | Precio mínimo del año |
| `volatilidad_precio` | % | `std(precio_mensual)` | Desviación estándar mensual |
| `cambio_precio_pct` | % | `(precio_t - precio_t-1) / precio_t-1 * 100` | Cambio porcentual anual |
| `rango_precio_ratio` | ratio | `(max - min) / mean` | Amplitud relativa de precios |

---

## 📈 Estadísticas Descriptivas por Producto

### ARROZ (7,657 observaciones)
```
Dimensiones: 7,657 filas × 61 columnas
Período: 2006-2021 (algunos datos históricos desde 1990)
Municipios: ~200 municipios productores
Split: 6,132 historial | 1,525 train/test
```

**Rendimiento (t/ha)**
- Media: 5.42 t/ha
- Rango: 0.01 - 23.50 t/ha
- Mediana: 5.28 t/ha

### CACAO (11,666 observaciones)
```
Dimensiones: 11,666 filas × 61 columnas
Período: 2006-2021
Municipios: ~250+ municipios productores
Split: 9,349 historial | 2,317 train/test
```

**Rendimiento (t/ha)**
- Media: 0.68 t/ha
- Rango: 0.01 - 3.50 t/ha
- Mediana: 0.58 t/ha

### CAFÉ (12,278 observaciones)
```
Dimensiones: 12,278 filas × 61 columnas
Período: 2006-2021
Municipios: ~300+ municipios productores
Split: 9,822 historial | 2,456 train/test
```

**Rendimiento (t/ha)**
- Media: 2.15 t/ha
- Rango: 0.01 - 8.00 t/ha
- Mediana: 2.08 t/ha

### PLÁTANO (16,404 observaciones)
```
Dimensiones: 16,404 filas × 61 columnas
Período: 2006-2021
Municipios: ~300+ municipios productores
Split: 13,123 historial | 3,281 train/test
```

**Rendimiento (t/ha)**
- Media: 11.45 t/ha
- Rango: 0.01 - 50.00 t/ha
- Mediana: 11.20 t/ha

---

## 🔗 Lags Temporales Aplicados

La arquitectura temporal de las variables sigue esta lógica:

```
                    Año t-3    Año t-2    Año t-1    Año t (Predicción)
                    ────────   ────────   ────────   ─────────────────
Panel (EVA)                              ✓          ✓ (rendimiento_lag_1/2/3)
Clima (IDEAM)                            ✓          → Se usa para t
Crédito (FINAGRO)                        ✓          → Se usa para t
Precio (FRED)                            ✓          → Se usa para t
Suelo (AGROSAVIA)                        ✓          → Estructura fija
```

**Interpretación**: 
- El rendimiento de `año t` es explicado por:
  - Clima de `año t` (lluvia que cayó ese año)
  - Créditos de `año t` (dinero disponible ese año)
  - Precios de `año t` (incentivos económicos ese año)
  - Rendimientos históricos (inercia/rutina agrícola)

---

## 📊 Relación con el Pipeline ETL

```
raw/ → processed/ → model_mart_*.csv
 ↓
01_EVA_historica.csv  → 02_panel_*.csv
02_IDEAM_precip.csv   → 03_ideam_municipio_anio.csv
03_AGROSAVIA_suelos   → 04_agrosavia_municipio.csv
04_FINAGRO_creditos   → 05_finagro_municipio_anio_*.csv
05_FRED_precios       → 06_precio_internacional_*.csv
                                    ↓
            06_build_model_mart.py (Merge + Validación)
                                    ↓
                    model_mart_*.csv (Tablas finales)
                                    ↓
                    02_train_ml.py (Entrenamiento)
```

---

## ✅ Validaciones Aplicadas

Cada tabla `model_mart_*.csv` ha pasado por:

1. **Validación de Lags**: Verificación que `rendimiento_lag_1` coincida con `rendimiento_t_ha` del año anterior
2. **Detección de Outliers**: Variables derivadas marcan si el rendimiento es anormal (> 3σ)
3. **Detección de Copia**: Identifica municipios con varianza = 0 (indicador de copia governamental)
4. **Validación de Merge**: Verifica que departamento + municipio + año sean únicos
5. **Completitud**: Logging de valores NULL por variable
6. **Consistencia Temporal**: Verifica que los años están en secuencia correcta
7. **Leakage Prevention**: Asegura que variables t+1 no se usan para predecir año t

---

## 📁 Archivos Disponibles

En la carpeta `reports/tablas_entrenamiento/`:

1. **index.html** - Documento interactivo con todas las tablas y variables
2. **model_mart_arroz.csv** - Tabla completa Arroz
3. **model_mart_cacao.csv** - Tabla completa Cacao
4. **model_mart_cafe.csv** - Tabla completa Café
5. **model_mart_platano.csv** - Tabla completa Plátano

---

## 🎯 Cómo Usar Estas Tablas

### Para Reproducir el Modelo
```python
import pandas as pd

# Cargar tabla de entrenamiento
df = pd.read_csv('data/processed/model_mart_cacao.csv')

# Separar features y target
X = df.drop(['rendimiento_t_ha', 'split'], axis=1)
y = df['rendimiento_t_ha']

# Usar split para train/test
X_train = X[df['split'] == 'train']
y_train = y[df['split'] == 'train']
X_test = X[df['split'] == 'test']
y_test = y[df['split'] == 'test']

# Entrenar modelo
modelo = CatBoostRegressor()
modelo.fit(X_train, y_train)
```

### Para Análisis Exploratorio
```python
# Correlaciones con rendimiento
df.corr()['rendimiento_t_ha'].sort_values(ascending=False)

# Municipios con mayor producción
df.groupby('municipio')['produccion_t'].sum().sort_values(ascending=False)

# Tendencia temporal
df.groupby('anio')['rendimiento_t_ha'].mean().plot()
```

---

## 📞 Notas Importantes

1. **Variables NULL**: Algunas filas tienen NULL en variables de clima/crédito/precio porque esos datos no estaban disponibles para el período.
2. **score_confiabilidad**: Usar en análisis para filtrar datos confiables.
3. **Split column**: Usar para reproducir exactamente los conjuntos de train/test del modelo.
4. **Rendimiento como target**: Todas las variables son features EXCEPTO `rendimiento_t_ha` que es la variable a predecir.

---

*Documento generado automáticamente - Última actualización: Agosto 2026*
