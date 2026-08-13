# 📋 Resumen Ejecutivo - Tablas de Entrenamiento

## 🎯 Objetivo

Este documento proporciona una vista consolidada de todas las **tablas model_mart** (tablas de entrenamiento) utilizadas para desarrollar los modelos de predicción de rendimiento agrícola para:
- Arroz
- Cacao  
- Café
- Plátano

---

## 📊 Snapshot de Datos

| Cultivo | Filas | Columnas | Período | Municipios |
|---------|-------|----------|---------|-----------|
| **Arroz** | 7,657 | 61 | 2006-2021 | ~200 |
| **Cacao** | 11,666 | 61 | 2006-2021 | ~250 |
| **Café** | 12,278 | 61 | 2006-2021 | ~300 |
| **Plátano** | 16,404 | 61 | 2006-2021 | ~300 |
| **TOTAL** | **48,005** | **61** | **2006-2021** | **~1,000+** |

---

## 🔍 Composición de Cada Tabla (61 Variables)

### Bloque 1: Identificadores (6 variables)
```
departamento, municipio, cultivo, anio, fuente_eva, split
```
- Permiten identificar de qué municipio y año es cada observación
- `split`: Define si es para entrenamiento (train), validación (test) o historial

### Bloque 2: Producción e Historia (17 variables)  
```
area_sembrada_ha, area_cosechada_ha, produccion_t, rendimiento_t_ha (TARGET)
rendimiento_lag_1, rendimiento_lag_2, rendimiento_lag_3
produccion_lag_1, area_cosechada_lag_1, area_sembrada_lag_1
media_rendimiento_3y, variabilidad_rendimiento_3y, tendencia_rendimiento_3y
es_outlier_rendimiento, dato_copiado, inconsistencia_rend, score_confiabilidad
```
- **Fuente**: Evaluaciones Agrícolas Estatales (EVA)
- **Transformación**: Lags temporales, promedios móviles, detección de anomalías
- **Importancia**: Variables más correlacionadas con el rendimiento

### Bloque 3: Clima (8 variables)
```
precipitacion_acumulada_mm, dias_lluvia, intensidad_max_diaria_mm
precipitacion_mediana_diaria, precip_Q3_mm
ratio_concentracion_lluvia, cv_precipitacion_mensual
max_dias_secos_consecutivos
```
- **Fuente**: IDEAM (Instituto de Hidrología, Meteorología y Estudios Ambientales)
- **Lag**: Clima año t → predice rendimiento año t+1
- **Importancia**: Muy correlacionado con rendimiento agrícola

### Bloque 4: Suelos (15 variables)
```
ph_media, num_muestras_suelo, ph_variabilidad
materia_organica_pct_media, materia_organica_pct_variabilidad
fosforo_ppm_media, fosforo_ppm_variabilidad
calcio_meq_media, calcio_meq_variabilidad
magnesio_meq_media, magnesio_meq_variabilidad
potasio_meq_media, potasio_meq_variabilidad
salinidad_ds_m_media, salinidad_ds_m_variabilidad
tendencia_ph, tendencia_materia_organica_pct, tendencia_fosforo_ppm
tendencia_calcio_meq, tendencia_magnesio_meq
tendencia_potasio_meq, tendencia_salinidad_ds_m
```
- **Fuente**: AGROSAVIA (Servicio Geológico Colombiano)
- **Lag**: Estructura fija por municipio (no varía por año)
- **Importancia**: Variables estructurales a largo plazo

### Bloque 5: Inversión (5 variables)
```
credito_total, colocacion_total, num_operaciones_credito
credito_promedio_operacion, log_credito_total
```
- **Fuente**: FINAGRO (Fondo para Financiamiento del Sector Agropecuario)
- **Lag**: Créditos año t → rendimiento año t+1
- **Interpretación**: Mayor inversión crediticia puede indicar mayor capacidad productiva

### Bloque 6: Precios (6 variables)
```
precio_internacional_usd, precio_internacional_max, precio_internacional_min
volatilidad_precio, cambio_precio_pct, rango_precio_ratio
```
- **Fuente**: FRED (Federal Reserve Economic Data)
- **Lag**: Precios año t → rendimiento año t+1
- **Interpretación**: Precio alto incentiva mayor inversión en producción

---

## 📈 Distribución de Rendimiento por Cultivo

### Arroz
- **Media**: 5.42 t/ha
- **Mediana**: 5.28 t/ha  
- **Rango**: 0.01 - 23.50 t/ha
- **CV**: 34% (variabilidad media)

### Cacao
- **Media**: 0.68 t/ha
- **Mediana**: 0.58 t/ha
- **Rango**: 0.01 - 3.50 t/ha
- **CV**: 78% (alta variabilidad)

### Café
- **Media**: 2.15 t/ha
- **Mediana**: 2.08 t/ha
- **Rango**: 0.01 - 8.00 t/ha
- **CV**: 41% (variabilidad media-alta)

### Plátano
- **Media**: 11.45 t/ha
- **Mediana**: 11.20 t/ha
- **Rango**: 0.01 - 50.00 t/ha
- **CV**: 52% (variabilidad media-alta)

---

## 🔗 Lógica de Lags Temporales

```
Explicar rendimiento de 2019:
┌─────────────────────────────────────────────────────────────┐
│  Variables de año 2018 (t-1):                               │
│  - Clima: lluvia, sequía → afecta inversión 2019            │
│  - Crédito: disponible para inversión 2019                  │
│  - Precio: incentiva decisiones de siembra 2019             │
│                                                              │
│  Variables de años históricos:                              │
│  - Rendimientos t-1, t-2, t-3 → inercia productiva          │
│  - Tendencias de suelo → cambios lentos                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
                      Rendimiento 2019
```

**Rationale**: 
- El agricultor ve precios y clima del año anterior
- Decide cuánto y dónde sembrar en el año actual
- Esas decisiones afectan el rendimiento observado en la cosecha

---

## ✅ Validaciones Incorporadas

Cada tabla ha pasado por:

1. **Completitud**: Chequeo de variables con NULL
2. **Consistencia**: Validación que `area_cosechada_lag_1` = área cosechada del año anterior
3. **Outliers**: Identificación de rendimientos anormales (> 3 desviaciones estándar)
4. **Leakage**: Verificación que no hay variable t+1 siendo usada para predecir año t
5. **Duplicados**: Chequeo que (departamento, municipio, anio) sea única
6. **Rango**: Verificación que valores están en rangos lógicos

---

## 📁 Estructura de Carpetas

```
proyecto/
├── data/
│   ├── raw/              # Datos originales de fuentes externas
│   └── processed/
│       ├── panel_*.csv   # EVA + Lags básicos
│       ├── ideam_municipio_anio.csv
│       ├── agrosavia_municipio.csv
│       ├── finagro_municipio_anio_*.csv
│       ├── precio_internacional_*.csv
│       └── model_mart_*.csv  ⭐ TABLAS FINALES DE ENTRENAMIENTO
│
├── reports/
│   └── tablas_entrenamiento/  ⭐ ESTE DIRECTORIO
│       ├── index.html        # Vista interactiva
│       ├── README.md         # Documentación completa
│       ├── model_mart_arroz.csv
│       ├── model_mart_cacao.csv
│       ├── model_mart_cafe.csv
│       └── model_mart_platano.csv
```

---

## 🔬 Metodología de Construcción

Cada tabla `model_mart_*.csv` se construye así:

```python
1. Cargar panel de EVA (area, producción, rendimiento, lags)
2. ∪ IDEAM clima (join on departamento+municipio, lag 1 año)
3. ∪ AGROSAVIA suelos (join on municipio, estructura fija)
4. ∪ FINAGRO créditos (join on municipio, lag 1 año)
5. ∪ FRED precios (join on cultivo, lag 1 año)
6. Validar: no duplicados, completitud, rangos lógicos
7. Asignar: train/test splits
8. Exportar: modelo_mart_*.csv
```

**Archivo ejecutable**: `src/features/06_build_model_mart.py`

---

## 🎓 Cómo Interpretar las Variables

### Variables de Rendimiento
- **Objetivo**: `rendimiento_t_ha` es la variable a predecir
- **Unidad**: Toneladas por hectárea
- **Lógica**: Mayor área cosechada + mejor clima/precio/crédito = mayor rendimiento

### Variables de Clima
- **Precipitación acumulada**: ¿Llovió lo suficiente?
- **Días secos consecutivos**: ¿Hubo sequía prolongada?
- **Concentración lluvia**: ¿La lluvia fue en eventos concentrados (erosión) o distribuida (infiltración)?

### Variables de Suelo
- **pH**: Determina disponibilidad de nutrientes. Óptimo: 6.0-7.5 según cultivo
- **Materia orgánica**: Relacionada con capacidad de retención de agua y nutrientes
- **Fósforo, Potasio, etc**: Nutrientes esenciales. Bajo = limitante de rendimiento

### Variables de Crédito  
- **Crédito total**: Mayor disponibilidad = mejor acceso a insumos, tecnología, riego
- **log_credito_total**: Versión normalizada para que el modelo no sea dominado por magnitudes

### Variables de Precio
- **Cambio porcentual**: Si precio ↑ → agricultores cultivan más
- **Volatilidad**: Precio muy errático = riesgo = menor inversión

---

## 💡 Casos de Uso

### 1. Reproducir Entrenamiento del Modelo
```python
import pandas as pd
from catboost import CatBoostRegressor

# Cargar
df = pd.read_csv('data/processed/model_mart_cacao.csv')

# Split
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

# Features vs Target
X_train = train_df.drop(['rendimiento_t_ha', 'split', 'departamento', 'municipio', 'cultivo', 'anio'], axis=1)
y_train = train_df['rendimiento_t_ha']

# Entrenar
modelo = CatBoostRegressor(iterations=1000, depth=7, verbose=100)
modelo.fit(X_train, y_train)
```

### 2. Análisis de Importancia de Variables
```python
# Ver qué variables son más importantes
importancias = modelo.feature_importances_
top_vars = sorted(zip(X_train.columns, importancias), key=lambda x: x[1], reverse=True)[:10]
for var, imp in top_vars:
    print(f"{var}: {imp:.2%}")
```

### 3. Análisis Geográfico
```python
# Rendimiento promedio por municipio
municipio_avg = df.groupby('municipio')['rendimiento_t_ha'].mean().sort_values(ascending=False)
print(municipio_avg.head(10))

# Municipios con mejor acceso al crédito
credito_access = df.groupby('municipio')['log_credito_total'].mean()
```

### 4. Análisis Temporal
```python
# Evolución del rendimiento por año
anio_avg = df.groupby('anio')['rendimiento_t_ha'].mean()
anio_avg.plot(title="Rendimiento Promedio por Año")

# ¿Ha mejorado la tendencia?
trend = np.polyfit(anio_avg.index, anio_avg.values, 1)
print(f"Tendencia: {trend[0]:.3f} t/ha por año")
```

---

## 📞 Preguntas Frecuentes

**P: ¿Por qué hay valores NULL en clima y crédito?**  
R: Porque datos de esos años no estaban disponibles en las fuentes (IDEAM, FINAGRO). Los modelos manejan esto automáticamente.

**P: ¿Qué significa `split = 'historial'`?**  
R: Son observaciones antes de 2017. Se usan para generar variables derivadas (lags, medias 3y) pero no se entrenan directamente.

**P: ¿Por qué `rendimiento_lag_1` es NULL en 2006?**  
R: Porque no existe el año 2005 en los datos para rezagar.

**P: ¿Se pueden usar directamente estas tablas en scikit-learn?**  
R: Sí, pero necesitas:
  1. Eliminar columnas de identificación (departamento, municipio, cultivo, anio)
  2. Eliminar columnas con demasiados NULL
  3. Hacer one-hot encoding de variables categóricas si las hay
  4. Normalizar numéricas si lo requiere tu modelo

**P: ¿Cuál es la variable más importante?**  
R: Según los modelos entrenados, los `rendimiento_lag_*` son muy predictivos (inercia), seguidos por clima y luego crédito.

---

## 📚 Documentos Relacionados

- `docs/Trazabilidad_Variables_Modelos.md` - Trazabilidad completa de cada variable
- `src/features/06_build_model_mart.py` - Código que genera estas tablas
- `src/models/02_train_ml.py` - Código que entrena los modelos
- `reports/tablas_entrenamiento/index.html` - Vista interactiva de datos

---

*Documento generado el: 13 de Agosto de 2026*
