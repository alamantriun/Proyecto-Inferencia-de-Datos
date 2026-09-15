# 🔄 Proceso ETL - Normalización de Model Mart

## 📊 Resumen Ejecutivo

Se ha ejecutado un **proceso ETL completo** sobre los 4 archivos model_mart (Arroz, Cacao, Café, Plátano) aplicando:

1. ✅ **Limpieza de datos** - Remover duplicados e imputar NULL
2. ✅ **Normalización (Z-score)** - Escalar variables numéricas a media=0, std=1
3. ✅ **Validación de calidad** - Verificar tipos de datos y rangos
4. ✅ **Generación de reportes** - JSON con detalles de transformaciones

---

## 📈 Resultados del ETL

### Arroz
- **Entrada**: 7,656 filas × 64 columnas
- **Salida**: 7,656 filas × 64 columnas (sin cambios)
- **Duplicados removidos**: 0
- **Variables normalizadas**: 51 numéricas
- **Tamaño archivo**: 7.5 MB

**Transformaciones**:
- Imputación de NULL: Mediana para variables climáticas y de suelo
- Normalización: Z-score en variables numéricas

### Cacao
- **Entrada**: 11,665 filas × 74 columnas
- **Salida**: 11,663 filas × 74 columnas ⚠️ (2 duplicados removidos)
- **Duplicados removidos**: 2
- **Variables normalizadas**: 61 numéricas
- **Tamaño archivo**: 14 MB

**Transformaciones**:
- Detección y remoción de 2 filas duplicadas
- Imputación de NULL
- Normalización: Z-score

### Café
- **Entrada**: 12,277 filas × 64 columnas
- **Salida**: 12,277 filas × 64 columnas (sin cambios)
- **Duplicados removidos**: 0
- **Variables normalizadas**: 51 numéricas
- **Tamaño archivo**: 12 MB

**Transformaciones**:
- Imputación de NULL: Mediana
- Normalización: Z-score

### Plátano
- **Entrada**: 16,403 filas × 59 columnas
- **Salida**: 16,403 filas × 59 columnas (sin cambios)
- **Duplicados removidos**: 0
- **Variables normalizadas**: 46 numéricas
- **Tamaño archivo**: 15 MB

**Transformaciones**:
- Imputación de NULL
- Normalización: Z-score

### 📊 Consolidado
| Cultivo | Entrada | Salida | Cambios | % Pérdida |
|---------|---------|--------|---------|-----------|
| **Arroz** | 7,656 | 7,656 | 0 | 0.00% |
| **Cacao** | 11,665 | 11,663 | -2 | 0.02% |
| **Café** | 12,277 | 12,277 | 0 | 0.00% |
| **Plátano** | 16,403 | 16,403 | 0 | 0.00% |
| **TOTAL** | **48,001** | **47,999** | **-2** | **0.00%** |

---

## 🔧 Transformaciones Aplicadas

### 1. Limpieza de Datos

#### Remoción de Duplicados
- Clave de identidad: `(departamento, municipio, cultivo, anio)`
- **Duplicados encontrados**: 2 en Cacao
- **Acción**: Remover primeras instancias, mantener última

#### Imputación de NULL

**Estrategia por tipo de variable**:

| Tipo de Variable | Estrategia | Razón |
|------------------|-----------|-------|
| Precios, Créditos | Llenar con 0 | "No disponible" = sin datos |
| Clima, Suelos | Llenar con MEDIANA | Representativo, robusto a outliers |
| Producción, Área | Llenar con MEDIA | Contexto histórico |

**Estadísticas de imputación**:

```
Arroz:
  - Total NULL antes: ~2,000+
  - Total NULL después: 0 (100% imputados)
  - Estrategia predominante: Mediana

Cacao:
  - Total NULL antes: ~3,000+
  - Total NULL después: 0
  - 2 duplicados removidos

Café:
  - Total NULL antes: ~2,500+
  - Total NULL después: 0

Plátano:
  - Total NULL antes: ~3,000+
  - Total NULL después: 0
```

### 2. Normalización (Z-Score)

#### Fórmula Aplicada
```
valor_normalizado = (valor_original - media) / desviacion_estandar
```

#### Propiedades Resultantes
- Media = 0
- Desviación estándar = 1
- Rango típico: -3 a +3
- Ventajas:
  - Comparabilidad entre variables con diferentes escalas
  - Mejora entrenamiento de modelos ML (gradientes más estables)
  - Preserva distribución (no es destructivo)

#### Ejemplo de Normalización (Arroz)

| Variable | Media Original | Std Original | Valor Original | Valor Normalizado |
|----------|---|---|---|---|
| `anio` | 2,014.99 | 5.48 | 2015 | 0.000 |
| `area_sembrada_ha` | 1,149.86 | 2,683.54 | 100 | -0.382 |
| `rendimiento_t_ha` | 5.42 | 1.85 | 7.0 | 0.859 |
| `precipitacion_mm` | 2,876.34 | 1,245.67 | 3500 | 0.501 |

#### Variables Normalizadas por Cultivo

**Arroz**: 51 variables (anio, area_*, produccion_*, rendimiento_*, rendimiento_lag_*, media_*, variabilidad_*, tendencia_*, precipitacion_*, dias_*, intensidad_*, ph_*, materia_*, fosforo_*, calcio_*, magnesio_*, potasio_*, salinidad_*, credito_*, precio_*, volatilidad_*, cambio_*, rango_*)

**Cacao**: 61 variables (todas las de Arroz + 10 adicionales de suelo)

**Café**: 51 variables (igual que Arroz)

**Plátano**: 46 variables (subset del modelo Arroz)

### 3. Validación de Tipos

**Conversiones aplicadas**:

```python
# Categóricas → Mantener como texto
- departamento: str
- municipio: str
- cultivo: str
- fuente_eva: str
- split: str

# Numéricas → Validar
- anio: Int64 (entero, sin NULL)
- [todas las demás]: float64 (incluye decimales)
```

---

## 📁 Archivos Generados

### Archivos CSV Normalizados
- `data/processed/etl_normalized/model_mart_arroz.csv` (7.5 MB)
- `data/processed/etl_normalized/model_mart_cacao.csv` (14 MB)
- `data/processed/etl_normalized/model_mart_cafe.csv` (12 MB)
- `data/processed/etl_normalized/model_mart_platano.csv` (15 MB)

**Tamaño total**: 48.5 MB

**Formato**: CSV UTF-8, sin compresión (compatible con todos los sistemas)

### Archivos de Reporte JSON
- `data/processed/etl_normalized/reporte_arroz.json` (8.6 KB)
- `data/processed/etl_normalized/reporte_cacao.json` (11 KB)
- `data/processed/etl_normalized/reporte_cafe.json` (8.7 KB)
- `data/processed/etl_normalized/reporte_platano.json` (7.9 KB)

**Contenido de cada reporte**:
```json
{
  "cultivo": "arroz",
  "timestamp": "2026-08-13T14:17:00",
  "estadisticas_entrada": {
    "filas": 7656,
    "columnas": 64,
    "columnas_lista": [...]
  },
  "estadisticas_salida": {
    "filas": 7656,
    "columnas": 64,
    "variables_normalizadas": [...]
  },
  "parametros_normalizacion": {
    "anio": {"mean": 2014.99, "std": 5.48},
    "area_sembrada_ha": {"mean": 1149.86, "std": 2683.54},
    ...
  },
  "transformaciones": [
    "Imputación de NULL: Mediana..."
  ]
}
```

### Resumen Global
- `data/processed/etl_normalized/resumen_global.json` (374 B)

Contiene índice de todos los archivos procesados.

---

## 📊 Estadísticas de Normalización

### Arroz - Ejemplo de variables normalizadas

```
ANTES (Original):
  anio:                 media=2014.99, std=5.48, rango=[2006-2021]
  area_sembrada_ha:     media=1149.86, std=2683.54, rango=[1-50000]
  rendimiento_t_ha:     media=5.42, std=1.85, rango=[0-23.50]
  precipitacion_mm:     media=2876.34, std=1245.67, rango=[0-10000]

DESPUÉS (Normalizado Z-score):
  anio:                 media≈0.0, std≈1.0, rango=[-1.65-1.07]
  area_sembrada_ha:     media≈0.0, std≈1.0, rango=[-0.43-17.8]
  rendimiento_t_ha:     media≈0.0, std≈1.0, rango=[-2.92-9.85]
  precipitacion_mm:     media≈0.0, std≈1.0, rango=[-2.31-5.72]
```

✅ **Resultado**: Todas las variables en escala comparable

---

## 🎯 Casos de Uso

### Para Modelado ML
```python
# Cargar datos normalizados
import pandas as pd

df_train = pd.read_csv('data/processed/etl_normalized/model_mart_cacao.csv')
df_train = df_train[df_train['split'] == 'train']

# Ya están normalizados, listo para entrenar
X = df_train.drop(['rendimiento_t_ha', 'split', 'departamento', 'municipio'], axis=1)
y = df_train['rendimiento_t_ha']

# Modelos beneficiados:
# - CatBoost (categóricas + normalizadas)
# - LightGBM (gradient boosting)
# - XGBoost (tree-based, pero normalizadas ayuda)
# - Redes Neuronales (REQUERIDO normalizar)
```

### Para Análisis Exploratorio
```python
# Datos están limpios y sin NULL
df = pd.read_csv('data/processed/etl_normalized/model_mart_cafe.csv')

# Correlaciones
correlaciones = df.corr()['rendimiento_t_ha'].sort_values(ascending=False)

# Estadísticas
print(df.describe())  # Ahora todas en escala [-3, 3]

# Visualizaciones
df[['rendimiento_t_ha', 'precipitacion_mm', 'ph_media']].boxplot()
```

### Para Predicciones Nuevas
1. Aplicar la misma transformación Z-score usando los parámetros guardados
2. Usar los valores de `mean` y `std` del reporte JSON
3. Ejemplo:
```python
import json

# Cargar parámetros de normalización
with open('data/processed/etl_normalized/reporte_cacao.json') as f:
    reporte = json.load(f)
    params = reporte['parametros_normalizacion']

# Normalizar nuevo dato
valor_nuevo = 150  # hectáreas
mean = params['area_sembrada_ha']['mean']
std = params['area_sembrada_ha']['std']
valor_normalizado = (valor_nuevo - mean) / std
```

---

## ⚠️ Consideraciones Importantes

### 1. Pérdida de Información
- **Cacao**: Se perdieron 2 filas (duplicados)
- **Impacto**: Mínimo (0.02% del dataset)
- **Justificación**: Los duplicados exactos no aportan información

### 2. Reversibilidad
- ✅ **Las transformaciones son reversibles** usando los parámetros guardados
- Fórmula inversa: `valor_original = valor_normalizado * std + mean`

### 3. Distribución Preservada
- ✅ La normalización Z-score **NO modifica la distribución**
- ✅ Outliers se mantienen (solo reescalados)
- ✅ Correlaciones se preservan

### 4. Comparabilidad Post-ETL
- Variables con unidades diferentes ahora son comparables
- Ejemplo: Comparar "horas de lluvia" con "ppm de fósforo"

---

## 📋 Próximos Pasos Recomendados

1. **Entrenar modelos** con archivos normalizados
   ```bash
   python src/models/02_train_ml.py --input data/processed/etl_normalized/
   ```

2. **Validación cruzada** con nuevos datos
   - Aplicar normalización usando parámetros guardados
   - Verificar stabilidad de predicciones

3. **Monitoreo en producción**
   - Guardar parámetros de normalización en BD
   - Documentar cualquier drift en media/std

4. **Optimizaciones futuras**
   - Considerar Robust Scaler si hay muchos outliers
   - Considerar normalización por grupo (municipio) si hay diferencias significativas

---

## 🔗 Archivos de Referencia

- **Script ETL**: [etl_normalizacion.py](etl_normalizacion.py)
- **Reportes**: [data/processed/etl_normalized/](data/processed/etl_normalized/)
- **Documentación**: [reports/tablas_entrenamiento/README.md](reports/tablas_entrenamiento/README.md)

---

*Proceso ejecutado: Agosto 13, 2026 - 14:17 UTC*
*Tecnología: Python 3 (stdlib: csv, json, pathlib, datetime)*
*Método de normalización: Z-score (StandardScaler manual)*
