# Diseño del notebook riguroso para café

## Objetivo

Reemplazar `pipeline_interactivo_cafe.ipynb` por un notebook ejecutable de principio a fin que prepare, compare y valide dos tareas relacionadas:

1. clasificación de rendimiento alto o bajo;
2. regresión del rendimiento esperado en toneladas por hectárea.

El notebook prepara un candidato para producción, pero solo lo marca como aprobado cuando supera controles objetivos. No interpreta rendimiento alto como rentabilidad ni presenta probabilidades heurísticas como éxito financiero.

## Alcance y límites

El entregable principal es el notebook. Antes de reemplazarlo se conserva una copia recuperable del archivo actual como `pipeline_interactivo_cafe_original.ipynb`.

El notebook utiliza `reports/tablas_entrenamiento/dataset_cafe_ml_ready.csv`, documenta sus limitaciones y genera artefactos reproducibles bajo `reports/modelo_produccion_cafe/`.

Quedan fuera de alcance:

- despliegue de una API o servicio web;
- predicción de utilidad, ROI o prosperidad financiera;
- incorporación de fuentes externas nuevas;
- corrección retroactiva de datos fuente no trazables.

## Contrato de datos

La unidad analítica es `departamento + municipio + cultivo + anio`.

El notebook verifica antes de entrenar:

- columnas obligatorias;
- unicidad de la llave analítica;
- target no nulo;
- años esperados 2007-2024;
- consistencia de `rendimiento_lag_1` contra el año anterior;
- proporción de nulos por variable;
- distribución del target y de la clase por año;
- cambio de fuente EVA a partir de 2019.

Las variables `score_confiabilidad` y `dato_copiado` se excluyen porque fueron calculadas con información del target. También se excluyen las variables externas cuya temporalidad o llave geográfica no está suficientemente demostrada. La primera versión rigurosa usa únicamente identificadores geográficos conocidos y variables históricas rezagadas.

## Definición de objetivos

### Clasificación

`rendimiento_alto = 1` cuando `rendimiento_t_ha` supera la mediana observada hasta 2018. El umbral se calcula una sola vez sin observar 2019-2024 y queda guardado en el manifiesto del modelo.

### Regresión

El objetivo continuo es `rendimiento_t_ha`. SMOTE no se aplica a regresión porque no es una técnica para targets continuos.

## Separación temporal

La validación nunca usa un `train_test_split` aleatorio.

- 2007-2018: historial inicial;
- 2019-2023: backtest rolling-origin para selección;
- 2024: holdout final, observado una sola vez después de elegir modelo, hiperparámetros y umbral de decisión.

En cada fold, imputadores, codificadores, escaladores y balanceo se ajustan exclusivamente con los años anteriores al año evaluado.

## Balanceo

El notebook muestra el balance original por año y la distribución antes y después del remuestreo.

Como existen variables categóricas, se usa `SMOTENC`, no SMOTE sobre columnas one-hot. El flujo es:

1. imputación aprendida en train;
2. codificación ordinal temporal de `departamento` y `municipio`;
3. escalado de variables numéricas;
4. `SMOTENC` únicamente sobre train;
5. codificación final apropiada para cada estimador;
6. predicción sobre validación sin remuestrear.

Se incluye una comparación explícita contra el mismo algoritmo sin remuestreo para medir si SMOTENC realmente aporta valor.

## Modelos y baselines

### Clasificación

- mayoría del entrenamiento;
- regla de rendimiento del año anterior;
- regresión logística sin remuestreo;
- regresión logística con SMOTENC;
- Random Forest sin remuestreo;
- Random Forest con SMOTENC;
- CatBoost con pesos de clase, sin generar categorías sintéticas.

### Regresión

- rendimiento del año anterior;
- media móvil de tres años;
- Elastic Net;
- Random Forest Regressor;
- CatBoost Regressor.

Los hiperparámetros se mantienen pequeños y explícitos. Cualquier búsqueda se limita al período 2019-2023 y nunca utiliza 2024.

## Métricas

Clasificación:

- MCC como métrica principal;
- precisión, recall, F1, exactitud y balanced accuracy;
- ROC-AUC y PR-AUC;
- Brier score y error de calibración;
- matriz de confusión por año y para 2024.

Regresión:

- MAE como métrica principal;
- RMSE y R²;
- mejora relativa contra `rendimiento_lag_1`;
- error por año y dispersión real contra predicho.

La probabilidad final se calibra con predicciones fuera de muestra de 2019-2023. El calibrador no observa 2024.

El umbral de decisión se elige una sola vez con esas predicciones fuera de muestra de 2019-2023, maximizando el MCC promedio anual. Si hay empate, se prioriza mayor recall y luego el umbral más cercano a 0,50. Tanto el calibrador como el umbral se congelan antes de evaluar 2024.

## Selección y compuerta de producción

El candidato de clasificación se elige por MCC promedio anual 2019-2023, con balanced accuracy como desempate. El candidato de regresión se elige por MAE promedio anual.

El manifiesto solo usa estado `production_candidate` cuando se cumplen todos estos criterios:

- no hay fallos de integridad de datos;
- MCC promedio del challenger supera al baseline `t-1` por al menos 0,02;
- ningún MCC anual del challenger es negativo;
- MCC 2024 es al menos 0,50;
- balanced accuracy 2024 es al menos 0,70;
- el modelo de regresión mejora el MAE del baseline `t-1` al menos 5% en 2024;
- las probabilidades calibradas tienen Brier menor que la predicción por prevalencia de entrenamiento.

Si una condición falla, el artefacto se guarda como `experimental_not_approved` con las razones exactas. No se cambia el ganador después de ver 2024.

Las métricas finales se calculan con el modelo entrenado solamente hasta 2023. Después de cerrar y guardar esa evaluación, el modelo operativo puede reentrenarse con datos hasta 2024 sin volver a calcular ni sustituir las métricas del holdout. El manifiesto diferencia explícitamente el modelo de evaluación del modelo reentrenado e indica la última fecha de datos de cada uno.

## Secuencia visible del notebook

1. Resumen y advertencia de alcance.
2. Configuración, versiones y semilla.
3. Carga del dataset.
4. Auditoría de calidad y leakage.
5. Exploración inicial con tablas y gráficas.
6. Definición del target y balance inicial.
7. Contrato de features y división temporal.
8. Demostración de SMOTENC antes y después.
9. Backtest de clasificadores.
10. Comparación de métricas y estabilidad anual.
11. Selección y calibración sin utilizar 2024.
12. Evaluación final 2024 y matriz de confusión.
13. Backtest de regresores.
14. Evaluación final de regresión.
15. Importancia de variables y análisis de errores.
16. Compuerta de producción, limitaciones y siguientes pasos.
17. Exportación de artefactos y resumen final.

## Gráficas obligatorias

- distribución inicial de `rendimiento_t_ha`;
- boxplot de rendimiento por año;
- mapa de faltantes por variable;
- proporción de clase alta por año;
- conteo de clases antes y después de SMOTENC;
- MCC y balanced accuracy por modelo;
- MCC anual de los mejores modelos;
- matriz de confusión 2024;
- curva ROC y precision-recall 2024;
- calibración de probabilidades;
- MAE por modelo y año;
- real contra predicho para regresión;
- importancia de variables del candidato.

Todas las gráficas incluyen título, ejes, unidades, período y una interpretación adyacente.

## Artefactos

El notebook guarda:

- pipeline clasificador con `joblib`;
- pipeline regresor con `joblib`;
- calibrador de probabilidades;
- `model_manifest.json` con features, umbral, versiones, métricas y estado;
- predicciones fuera de muestra;
- tablas comparativas CSV;
- figuras PNG;
- `MODEL_CARD.md` con uso permitido, uso prohibido y limitaciones.

## Manejo de errores

El notebook se detiene con mensajes descriptivos cuando faltan columnas, hay llaves duplicadas, el target contiene nulos, no existe el archivo de entrada o un fold contiene una sola clase. Los artefactos solo se reemplazan después de completar entrenamiento y validación.

## Pruebas y reproducibilidad

- semilla global 42;
- dependencias declaradas en `requirements.txt`;
- funciones de datos, balanceo, métricas, selección y compuerta cubiertas por pruebas unitarias;
- prueba que demuestra que SMOTENC solo recibe filas de entrenamiento;
- prueba de ejecución completa del notebook;
- validación de estructura `nbformat` y ausencia de salidas de error;
- recomputación independiente de la matriz, MCC y precisión desde las predicciones guardadas.

## Criterios de aceptación

- `pipeline_interactivo_cafe.ipynb` ejecuta de arriba abajo sin intervención manual;
- el notebook conserva la secuencia definida y todas las gráficas obligatorias;
- 2024 no participa en preprocesamiento, balanceo, calibración ni selección;
- la matriz de confusión suma exactamente el total de observaciones evaluadas;
- el estado de producción coincide con las reglas del manifiesto;
- el notebook no afirma rentabilidad o probabilidad de éxito empresarial.
