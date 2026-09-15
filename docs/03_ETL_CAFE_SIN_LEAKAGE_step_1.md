# ETL del dataset de café: auditoría y eliminación de leakage

## Objetivo

El CSV original contiene una serie temporal por:

`departamento + municipio + cultivo + año`

y el objetivo es predecir:

`rendimiento_t_ha`

La regla central del ETL es:

> Para predecir el rendimiento del año **t**, las variables históricas solo pueden utilizar información disponible antes de **t**.

---

## Problema detectado

Las variables:

- `media_rendimiento_3y`
- `variabilidad_rendimiento_3y`
- `tendencia_rendimiento_3y`

podían incluir el rendimiento del mismo año objetivo.

Ejemplo:

```text
2007 → 0.6098
2008 → 0.5619
2009 → 0.5620  ← target

media original de 2009
≈ media(2007, 2008, 2009)
```

Eso produce **data leakage**, porque el modelo recibe información derivada de la respuesta que debe predecir.

### Corrección

Para predecir 2009:

```text
2007 + 2008 → features
2009        → target
```

Para predecir 2010:

```text
2007 + 2008 + 2009 → features
2010                → target
```

---

# 1. Lags reconstruidos

Se reconstruyen:

```text
rendimiento_lag_1
rendimiento_lag_2
rendimiento_lag_3
produccion_lag_1
area_cosechada_lag_1
area_sembrada_lag_1
```

dentro de cada serie:

```text
departamento + municipio + cultivo
```

y ordenadas por `anio`.

Por tanto:

```text
lag_1(t) = valor de t-1
lag_2(t) = valor de t-2
lag_3(t) = valor de t-3
```

Si el año anterior no existe, se deja `NaN`. No se rellena con un año futuro.

---

# 2. Media histórica sin leakage

El cálculo correcto es:

```python
prior = rendimiento.shift(1)
prior.rolling(3).mean()
```

El `shift(1)` ocurre antes de la ventana.

Así, la fila de 2010 solo puede ver 2009, 2008 y 2007.

---

# 3. Variabilidad histórica

La variable:

`variabilidad_rendimiento_3y`

se calcula como desviación estándar sobre las observaciones anteriores.

Se requieren al menos dos valores históricos para calcularla; de lo contrario queda `NaN`.

---

# 4. Tendencia histórica

`trend_3y` se calcula como la pendiente de una regresión lineal:

```text
rendimiento ~ año
```

usando hasta las tres observaciones anteriores.

Una pendiente positiva indica una tendencia creciente y una negativa una tendencia decreciente.

---

# 5. Producción, área y rendimiento

Se comprueba:

```text
rendimiento_t_ha = produccion_t / area_cosechada_ha
```

cuando las tres variables están disponibles.

La columna:

`inconsistencia_rend`

se marca con:

```text
0 = consistente
1 = inconsistente
```

Esto es una comprobación de calidad y no una imputación.

---

# 6. Filas sin target

Las filas con:

`rendimiento_t_ha = NaN`

no pueden utilizarse para entrenamiento supervisado, porque no existe una respuesta real para calcular MAE, RMSE, R², etc.

El ETL las excluye del archivo destinado al modelado supervisado, pero no las considera automáticamente errores.

---

# 7. Variables de precipitación

En el CSV auditado, las variables de precipitación estaban completamente vacías:

```text
precipitacion_acumulada_mm
dias_lluvia
intensidad_max_diaria_mm
precipitacion_mediana_diaria
precip_Q3_mm
ratio_concentracion_lluvia
cv_precipitacion_mensual
max_dias_secos_consecutivos
```

No se rellenan con cero.

`NaN` significa que el dato no está disponible; no significa que no haya llovido.

Estas variables deben reconstruirse posteriormente desde la fuente climática, manteniendo la regla temporal.

---

# 8. Variables de suelo

Los faltantes de suelo tampoco se transforman automáticamente en cero.

Por ejemplo:

```text
ph = NaN
```

significa “sin medición disponible”, no “pH = 0”.

Para el modelo se puede conservar el `NaN` y/o crear indicadores como:

```text
suelo_disponible
```

Además, hay que verificar que una medición de suelo utilizada para predecir un año no provenga de un periodo posterior.

---

# 9. Datos copiados

El dataset contiene filas marcadas mediante:

`dato_copiado`

El ETL no las elimina automáticamente.

La razón es que copiar un valor de suelo no equivale necesariamente a copiar un target de rendimiento.

La decisión debe depender del origen de cada dato.

---

# 10. Variables que deben excluirse como features

Si el target es:

`rendimiento_t_ha`

no se debe alimentar al modelo con:

```text
rendimiento_t_ha
```

ni con variables del mismo año que permitan reconstruirlo directamente, especialmente:

```text
produccion_t
area_cosechada_ha
```

El ETL conserva estas columnas para trazabilidad y auditoría. La exclusión definitiva debe hacerse en la etapa de selección del conjunto `X`.

---

# 11. División temporal recomendada

Para evaluación realista:

```text
2007 ───────── 2018 | 2019 ───────── 2024
       HISTORIAL    |      TARGET
```

No se debe hacer un `train_test_split` aleatorio que mezcle años futuros y pasados.

---

# 12. Qué genera el script

Ejecutando:

```bash
python3 etl_cafe_sin_leakage.py
```

se generan:

```text
dataset_cafe_modelo_sin_leakage.csv
auditoria_cafe.csv
```

El script parte del archivo:

```text
dataset_original.csv
```

que debe estar en la misma carpeta.

---

# 13. Archivos del proyecto

- `etl_cafe_sin_leakage.py`: ETL reproducible.
- `ETL_CAFE_SIN_LEAKAGE.md`: documentación del proceso.
- `dataset_cafe_modelo_sin_leakage.csv`: resultado del ETL.
- `auditoria_cafe.csv`: auditoría de faltantes y cardinalidad.

---

# 14. Importante

Este ETL corrige el problema temporal de las variables derivadas del rendimiento.

Todavía quedan como trabajo posterior:

1. reconstrucción de precipitación;
2. validación temporal del precio internacional;
3. revisión temporal de las mediciones de suelo;
4. selección final de `X` y `y`;
5. validación temporal del modelo.
