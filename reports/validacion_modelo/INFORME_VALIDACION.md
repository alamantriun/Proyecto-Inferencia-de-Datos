# Validación temporal del modelo de café

## Resumen ejecutivo

- El objetivo nativo es regresión (`rendimiento_t_ha`). La matriz de confusión se calcula para `rendimiento alto`, definido como rendimiento > **0.923337 t/ha**, la mediana histórica 2007-2018 fijada sin mirar 2019-2024.
- Mejor clasificador elegible en 2019-2023: **Regresión logística balanceada**. En el holdout 2024 obtuvo exactitud **0.746**, precisión **0.961**, balanced accuracy **0.802** y MCC **0.546**.
- La ventaja de selección sobre **Regla rendimiento t-1** fue solo **0.003 MCC promedio anual**: es un empate práctico, no evidencia de superioridad concluyente. En 2024, el MCC observado más alto fue el de **Random Forest balanceado** (0.565), pero ese año no se usó para cambiar al ganador.
- Su matriz 2024 (filas reales, columnas predichas; orden bajo/alto) fue: `[[166, 12], [146, 299]]`.
- Mejor regresor elegible en 2019-2023: **Random Forest Regressor**. En 2024 obtuvo MAE **0.270 t/ha**, RMSE **0.355 t/ha** y R² **0.177**.
- En el holdout 2024, el menor MAE observado fue el de **Elastic Net** (0.218 t/ha), otra señal de que el ranking de modelos no es estable entre años.

## Balance de clases

- Entrenamiento histórico utilizable: 3223 bajos y 3290 altos (50.5% altos).
- Selección 2019-2023: 1047 bajos y 2032 altos (66.0% altos).
- Holdout 2024: 178 bajos y 445 altos (71.4% altos).
- No se sobremuestreó la validación. Logistic Regression, Random Forest y CatBoost usaron pesos de clase calculados solo en entrenamiento. MCC fue la métrica primaria.

## Riesgos metodológicos encontrados

1. `score_confiabilidad` se calcula con la variabilidad y los outliers del target de toda la serie; usarlo como predictor o peso filtra información futura.
2. `dato_copiado` compara directamente el target del año con `lag_1`; tampoco puede usarse como feature ex ante.
3. Suelos y crédito se unen solo por `municipio`, aunque existen nombres homónimos entre departamentos. Las variantes con variables externas se reportan como exploratorias y no pueden ganar la selección.
4. La procedencia temporal de las muestras de suelo no está verificada. Por eso el ganador recomendado usa únicamente geografía conocida y variables históricas rezagadas.
5. La fuente EVA cambia de `historica` a `reciente` en 2019 y la proporción de clase alta cambia mucho por año; esto es drift y explica parte de la inestabilidad.
6. El script original elige el alpha de blending mirando 2019-2024 y después informa rendimiento sobre los mismos años; esa cifra es optimista para selección. Aquí 2024 quedó reservado.

## Interpretación

La clase significa **rendimiento alto**, no rentabilidad ni prosperidad financiera. Para llamar al resultado “negocio próspero” faltan costos, margen neto y un umbral económico validado.

## Estado

**Compartible con cautelas.** La evaluación temporal y las métricas son reproducibles, pero el drift de fuente y la trazabilidad incompleta de variables externas impiden afirmar que el modelo esté listo para decisiones de inversión.
