# Diseño para elevar y validar MCC y Kappa anuales del modelo de café

## Decisión

Se reconstruirá el dataset y el notebook de clasificación para intentar alcanzar, de manera legítima y reproducible, estos dos mínimos simultáneos:

- Matthews Correlation Coefficient (MCC) >= 0,85;
- Cohen's Kappa >= 0,85.

La condición se comprobará por separado en cada año de evaluación. No se considerará suficiente un promedio superior a 0,85 si un año individual queda por debajo. El notebook mostrará una matriz de confusión para cada año y una matriz consolidada.

La meta es un criterio de aceptación, no un resultado garantizado. Si los datos no contienen señal predictiva suficiente, el proceso terminará con estado `target_not_met` y conservará las métricas reales. No se ajustará el umbral con las etiquetas del mismo año evaluado ni se descartarán casos difíciles para fabricar el resultado.

## Estado de partida y razón del rediseño

El modelo de presentación actual obtiene en 2024 MCC 0,6401 y Kappa 0,6219 con umbral 0,50. Incluso seleccionando retrospectivamente el mejor umbral sobre las etiquetas de 2024 —práctica que no sería válida para producción— los máximos aproximados son MCC 0,680 y Kappa 0,680.

La revisión por año mostró que ninguno de los años 2019-2024 puede llegar simultáneamente a 0,85 usando únicamente un cambio de umbral sobre las probabilidades actuales. Por tanto, SMOTE, el ajuste del umbral o el cambio de una sola métrica no bastan: hay que corregir las fuentes, crear variables estrictamente históricas y volver a comparar modelos.

## Pregunta predictiva y momento de decisión

Para cada municipio cafetero y año `t`, el sistema responderá:

> ¿El rendimiento de café del año `t` será alto o bajo respecto a un umbral nacional fijo, usando únicamente información disponible hasta el 31 de diciembre de `t-1`?

La unidad analítica será:

`codigo_dane_municipio + cultivo_cafe + anio_objetivo`.

El umbral de rendimiento alto se calculará una sola vez con el periodo inicial 2007-2018 y se congelará. Se guardarán el valor, la unidad y los registros usados. Esto evita cambiar la definición de la clase después de observar los años evaluados.

El resultado es una clasificación de rendimiento, no una predicción de rentabilidad, ingresos, calidad del grano o éxito del productor.

## Fuentes de datos

### 1. EVA como fuente del objetivo y del historial productivo

- EVA histórica 2007-2018, conjunto `2pnw-mmge`.
- EVA reciente 2019-2025, conjunto `uejq-wxrr`.
- Variables: código DANE de municipio y departamento, área sembrada, área cosechada, producción, rendimiento, cultivo, estado físico y año.

La consulta en vivo del 14 de septiembre de 2026 encontró 651 registros de café para 2025 en la fuente reciente. Ese año se reservará como holdout final. EVA advierte que los años recientes pueden ser revisados; la descarga quedará congelada con fecha, URL, hash y número de filas para que cualquier revisión posterior sea detectable.

Para café se filtrará el código de cultivo `2030300`, descripción `Café` y estado físico `Pergamino o seco de trilla`. Si existen varias filas válidas para una llave, el rendimiento se recalculará como `suma(produccion_t) / suma(area_cosechada_ha)`; no se promediarán rendimientos sin ponderación.

### 2. DIVIPOLA como llave geográfica canónica

- Catálogo DANE `gdxc-w37w`.
- Código municipal normalizado a texto de cinco dígitos.
- Nombre de municipio y departamento usados solo para mostrar y auditar, nunca como llave principal.
- Latitud y longitud utilizadas como punto de referencia para consultar clima cuando no haya una estación IDEAM representativa.

Se generará una tabla de correspondencias explícita para cambios ortográficos y territoriales. Toda unión debe ser muchos-a-uno y reportar coincidencias, no coincidencias y duplicados. Se prohíbe unir solo por `municipio`, porque existen nombres repetidos en departamentos diferentes.

### 3. IDEAM para precipitación observada

- Conjunto oficial `s54a-sgyg`.
- Campos de estación, sensor, fecha de observación, valor observado, departamento, municipio, latitud y longitud.

El extractor actual descarga solamente los 150.000 registros más recientes, por lo cual concentra datos de 2026 y no construye un histórico válido. Se reemplazará por una extracción particionada por año y paginada, con caché local. Primero se agregará por estación-día y se aplicarán controles de unidad, duplicados, valores negativos, continuidad y cobertura; después se agregará por municipio y periodo agrícola.

IDEAM se usará cuando un municipio-año supere la cobertura mínima definida en la auditoría. Los años o municipios con cobertura insuficiente conservarán un indicador de ausencia y podrán usar el respaldo climático descrito abajo; no se imputarán con valores del futuro.

### 4. NASA POWER como respaldo climático reproducible

- API Daily de NASA POWER, con cobertura desde 1981 hasta cerca del tiempo presente.
- Punto consultado mediante las coordenadas DIVIPOLA del municipio.
- Variables candidatas: precipitación corregida, temperatura media, mínima y máxima y humedad relativa.

NASA POWER permitirá completar temperatura y reducir huecos de precipitación. Se almacenará la respuesta original y sus parámetros. Como el centroide municipal no representa necesariamente las zonas cafeteras montañosas, se añadirá un indicador de fuente y se compararán resultados con IDEAM solamente, NASA solamente y fuente combinada.

CHIRPS podrá evaluarse como alternativa de precipitación espacial en una fase experimental, pero no será una dependencia obligatoria del primer modelo porque la descarga y el promedio zonal aumentan considerablemente el costo operativo.

### 5. Suelo AGROSAVIA

- Conjunto `ch4u-f3i5`.
- Propiedades candidatas: pH, materia orgánica, fósforo, azufre, acidez, aluminio, calcio, magnesio, potasio, CIC y conductividad.

Las muestras se unirán por departamento y municipio normalizados a DIVIPOLA y se filtrarán por `fecha_de_analisis < 1 de enero del año objetivo`. Se guardarán mediana robusta, rango intercuartílico, número de muestras y edad de la muestra más reciente.

Por no existir una garantía completa de cuándo cada resultado histórico estuvo publicado, las variables de suelo serán una ampliación auditada. La compuerta principal se calculará también sin suelo. Si el 0,85 depende exclusivamente de esta fuente, el modelo se marcará como candidato condicionado y no como aprobado sin revisar su trazabilidad temporal.

### 6. Aptitud cafetera UPRA

- Capa oficial `Aptitud_Cafe_Jul2022` a escala 1:100.000.
- Porcentaje del municipio en categorías de aptitud y restricciones biofísicas.

Es una fuente estática útil para un modelo operativo actual, pero fue publicada en 2022. No se utilizará para aprobar retrospectivamente los folds 2019-2022, salvo que se demuestre documentalmente que sus componentes son invariantes y no incorporan resultados posteriores. Se presentará como análisis complementario y para el modelo de despliegue posterior a 2022.

### Fuentes excluidas de la compuerta inicial

- Crédito FINAGRO contemporáneo: su cobertura comienza tarde y puede ser consecuencia de la actividad productiva, no una causa disponible antes de predecir.
- Área cosechada, producción o rendimiento del mismo año objetivo: revelan directa o indirectamente el objetivo.
- Variables calculadas con el target, como puntajes de confiabilidad derivados del error observado.
- Predicciones, interpolaciones o imputaciones que hayan usado años futuros.

## Contrato temporal y prevención de leakage

Para predecir el año `t`:

- el entrenamiento contiene únicamente años `< t`;
- las variables EVA usan rezagos `t-1`, `t-2`, `t-3` o resúmenes anteriores;
- clima y suelo tienen fecha estrictamente anterior al 1 de enero de `t`;
- imputadores, escaladores, codificadores, selección de variables, SMOTENC, calibración y umbral se ajustan solo dentro de los datos de entrenamiento;
- el año evaluado nunca participa en la selección de modelo, hiperparámetros o umbral;
- cada artefacto conserva `max_train_year`, fechas máximas por fuente y hashes de entrada.

Una prueba automática fallará si encuentra una fecha de feature posterior al corte o si `max_train_year >= test_year`.

## Construcción de variables

### Historial productivo

- rendimiento, producción, área sembrada y área cosechada con rezagos 1, 2 y 3;
- mediana, media robusta, desviación, mínimo, máximo y pendiente de los últimos 3 y 5 años;
- cambio absoluto y porcentual respecto al año anterior;
- número de años observados y años desde la última observación;
- proporción histórica de años con rendimiento alto;
- promedio histórico de municipio, departamento y país calculado de manera acumulativa, nunca con el año objetivo.

### Clima anterior al año objetivo

- precipitación total, media y extremos mensuales;
- días secos, días de precipitación intensa y rachas secas máximas;
- temperatura media, mínima, máxima y días de estrés térmico;
- anomalías respecto a la climatología histórica disponible antes del corte;
- resúmenes de 12, 24 y 36 meses anteriores;
- cobertura, número de estaciones, fuente y proporción imputada.

### Suelo y geografía

- medianas robustas de propiedades químicas;
- dispersión, cantidad y antigüedad de muestras;
- departamento y municipio como categorías controladas;
- latitud y longitud con transformación explícita;
- aptitud UPRA solo en el análisis permitido por temporalidad.

Las variables con cobertura insuficiente o deriva severa se eliminarán antes de entrenar mediante reglas declaradas, no por observar su rendimiento en 2025.

## Auditoría de calidad y cambio de fuente

El notebook mostrará antes de modelar:

- filas y municipios por año y por fuente;
- duplicados de la llave canónica;
- cobertura y nulos de cada variable por año;
- distribución anual del rendimiento y de las clases;
- comparación 2018 contra 2019 para cuantificar el cambio entre EVA histórica y reciente;
- valores imposibles de área, producción, rendimiento, lluvia, temperatura y suelo;
- tasa de unión de cada fuente externa;
- edad máxima de cada feature usada por fold;
- balance de clases original y después de SMOTENC.

Los registros no se eliminarán solo por ser difíciles de predecir. Cada exclusión requerirá una regla de calidad previa y aparecerá en una tabla de descarte.

## Modelos candidatos

Se mantendrá un conjunto pequeño y explicable:

1. baseline de clase mayoritaria del entrenamiento;
2. baseline de persistencia: clase del rendimiento `t-1`;
3. regresión logística Elastic Net con ponderación de clases;
4. regresión logística Elastic Net con SMOTENC;
5. árbol de gradiente poco profundo con SMOTENC;
6. CatBoost poco profundo con categorías nativas y pesos de clase;
7. clasificador indirecto: regresión robusta del rendimiento y aplicación del umbral fijo.

SMOTENC, y no SMOTE aplicado después de one-hot encoding, se usará porque el dataset contiene categorías. Se ejecutará exclusivamente dentro de cada fold de entrenamiento. Se comparará contra el mismo tipo de modelo sin remuestreo; el modelo ganador no tendrá que usar SMOTENC si empeora MCC o Kappa.

La complejidad se limitará mediante profundidad, regularización, número de árboles y parada temprana. La explicación incluirá coeficientes para la regresión logística, importancia por permutación y contribuciones locales de CatBoost para casos representativos.

## Validación temporal anidada

### Desarrollo y selección

- historial inicial: 2007-2018;
- folds externos de backtesting: 2019, 2020, 2021, 2022, 2023 y 2024;
- cada fold entrena con todos los años anteriores disponibles;
- dentro de cada entrenamiento se usan folds temporales internos para escoger hiperparámetros y umbral.

El objetivo de selección será lexicográfico:

1. maximizar `min_anual(min(MCC, Kappa))` en los folds internos;
2. maximizar el promedio anual de `min(MCC, Kappa)`;
3. maximizar la precisión ponderada estilo WEKA;
4. preferir el modelo más simple si la diferencia es menor de 0,01.

Esta regla prioriza el peor año y evita que un año muy bueno oculte uno malo. El umbral puede cambiar entre folds externos únicamente porque se vuelve a aprender usando años anteriores; jamás se calcula con las etiquetas del año que se está prediciendo.

### Holdout final 2025

Después de cerrar el dataset, la receta, el modelo, los hiperparámetros y el mecanismo de umbral con 2007-2024, se evaluará una sola vez el año 2025. Las etiquetas de 2025 permanecerán separadas hasta esa celda final.

Si la fuente EVA modifica posteriormente 2025, la versión usada en esta evaluación seguirá preservada. Una versión revisada se considerará una evaluación nueva y no reemplazará silenciosamente el resultado original.

## Métricas y formato WEKA

Para cada año 2019-2025 se guardará:

- matriz de confusión `[[TN, FP], [FN, TP]]`;
- MCC;
- Cohen's Kappa;
- accuracy y balanced accuracy;
- TP Rate, FP Rate, Precision, Recall, F-Measure, ROC Area y PRC Area por clase;
- promedio ponderado de las métricas estilo WEKA;
- soporte por clase y prevalencia;
- intervalo de confianza bootstrap por municipio para MCC y Kappa.

Las comparaciones con 0,85 usarán los valores sin redondear. Se mostrarán resultados con cuatro decimales y un semáforo por año.

También se medirá:

- desempeño en municipios cuya clase cambió respecto a `t-1`;
- ganancia frente al baseline de persistencia;
- estabilidad por departamento y por nivel de cobertura;
- calibración de probabilidades mediante Brier score y curva de calibración.

Esto evita considerar útil un modelo que solo memorice que ciertos municipios casi siempre pertenecen a la misma clase.

## Compuerta de aceptación

El estado será `production_candidate` solamente si se cumplen todas estas condiciones:

1. MCC >= 0,85 en cada año 2019-2024;
2. Kappa >= 0,85 en cada año 2019-2024;
3. MCC >= 0,85 y Kappa >= 0,85 en el holdout 2025;
4. ninguna matriz de confusión tiene una clase ausente;
5. mejora el menor `min(MCC, Kappa)` del baseline de persistencia por al menos 0,02;
6. supera las pruebas de leakage, integridad, reproducibilidad y recarga del artefacto;
7. produce probabilidades válidas y no presenta degradación extrema en los casos donde cambia la clase;
8. todas las fuentes usadas tienen trazabilidad y fecha de corte compatible con el año predicho.

Si falla una condición, el estado será `target_not_met`. El notebook señalará exactamente los años, métricas y pruebas que fallaron. No se promoverá un modelo porque su exactitud o precisión ponderada exceda 85% si MCC o Kappa no lo hacen.

Después de la evaluación final podrá entrenarse un artefacto operativo con datos hasta 2025 para predecir 2026, pero ese reentrenamiento tendrá un identificador diferente y no heredará métricas que no fueron calculadas sobre él.

## Secuencia del notebook reemplazado

1. Objetivo, alcance, fecha de predicción y reglas de aceptación.
2. Versiones, semilla y configuración reproducible.
3. Descarga o lectura de caché con manifiesto de fuentes.
4. Normalización DIVIPOLA y armonización EVA 2007-2025.
5. Auditoría de calidad, duplicados, cobertura y cambio de fuente.
6. Gráficas del rendimiento y balance inicial por año.
7. Construcción de rezagos y variables acumulativas sin leakage.
8. Integración temporal de clima, suelo y geografía.
9. Separación explícita del holdout 2025.
10. Demostración visual de SMOTENC antes y después en un fold de entrenamiento.
11. Baselines y backtesting temporal 2019-2024.
12. Selección anidada priorizando el peor MCC/Kappa anual.
13. Tabla WEKA por modelo y año.
14. Matrices de confusión 2019-2024.
15. Análisis de errores, cambios de clase, cobertura y explicabilidad.
16. Congelación de receta, umbral y manifiesto.
17. Evaluación única de 2025 y su matriz de confusión.
18. Cuadrícula final de matrices 2019-2025 y semáforo de criterios.
19. Compuerta de aceptación y conclusión crítica.
20. Exportación del modelo, datos de evaluación, figuras y tarjeta del modelo.

## Gráficas obligatorias

- filas y municipios disponibles por año;
- distribución de rendimiento inicial por año;
- boxplot anual antes y después de armonizar las fuentes;
- proporción y conteo de clases por año;
- mapa de nulos y cobertura de fuentes;
- comparación de clase antes y después de SMOTENC;
- MCC y Kappa por modelo y año, con línea horizontal en 0,85;
- peor valor anual por modelo;
- cuadrícula de matrices de confusión 2019-2025;
- precisión, recall y F1 estilo WEKA por clase;
- curva de calibración y distribución de probabilidades;
- desempeño total frente a desempeño en cambios de clase;
- importancia global de variables y explicación de casos correctos y erróneos.

Cada gráfica tendrá título, ejes, unidades, periodo, tamaño de muestra y una interpretación breve inmediatamente debajo.

## Artefactos

Se escribirán bajo `reports/modelo_mcc_kappa_cafe/`:

- `source_manifest.json` con URL, fecha, filas, esquema y hash por fuente;
- `dataset_modelo.parquet` y diccionario de variables;
- predicciones fuera de muestra 2019-2024;
- predicciones intocadas de 2025;
- métricas anuales y tabla estilo WEKA en CSV;
- matrices de confusión en CSV y PNG;
- figuras de calidad, balance, comparación y explicabilidad;
- pipeline serializado y prueba de recarga;
- `model_manifest.json` con corte temporal, features, umbral, métricas y estado;
- `MODEL_CARD.md` con uso permitido, exclusiones y limitaciones.

Los artefactos se escribirán primero en una carpeta temporal y solo sustituirán una versión previa después de completar todas las pruebas.

## Pruebas automáticas

- unicidad y formato del código DIVIPOLA;
- reconstrucción exacta del rendimiento a partir de producción y área cosechada;
- igualdad de la definición del target en todos los folds;
- ausencia de años futuros en entrenamiento y features;
- SMOTENC recibe solo filas del entrenamiento;
- el holdout 2025 no participa en ningún `fit`, selección o umbral;
- suma de cada matriz igual al número de predicciones del año;
- recomputación independiente de MCC, Kappa y tabla WEKA desde las predicciones guardadas;
- selección basada en el peor año, no en 2025;
- serialización, recarga e igualdad de predicciones;
- ejecución completa del notebook con `nbconvert` y ausencia de salidas de error.

## Riesgos y respuesta

- **La meta 0,85 puede ser inalcanzable:** se reportará el límite observado y no se declarará producción.
- **Cambio metodológico EVA en 2019:** se medirá explícitamente y podrán agregarse indicadores de fuente; no se permitirá aprender del año evaluado.
- **2025 puede ser provisional:** se congelará la versión usada y se informará la fecha.
- **Cobertura IDEAM irregular:** extracción por año, controles por estación y respaldo NASA POWER con indicador de fuente.
- **Municipios con poco historial:** se limitará el uso o se recurrirá a promedios jerárquicos aprendidos del pasado.
- **Memorización geográfica:** comparación contra persistencia y análisis separado de cambios de clase.
- **Datos de suelo no plenamente trazables en publicación:** evaluación principal sin suelo y estado condicionado si resulta determinante.
- **Sobreajuste por buscar muchas combinaciones:** espacio pequeño de modelos, validación anidada y 2025 aislado.

## Criterios de terminado

- el notebook ejecuta de arriba abajo sin intervención manual;
- la descarga es reproducible o funciona desde una caché versionada;
- 2019-2024 tienen predicciones estrictamente walk-forward;
- 2025 permanece intocado hasta la evaluación final;
- aparecen siete matrices de confusión, una por año 2019-2025, más la consolidada;
- MCC y Kappa se calculan y validan independientemente por año;
- el notebook indica con claridad si cada año superó o no 0,85;
- la conclusión distingue entre un modelo útil, un candidato condicionado y una meta no alcanzada;
- ninguna cifra se presenta como superior a 85% si su valor real no cumple el umbral.

## Referencias oficiales consultadas

- EVA reciente: https://www.datos.gov.co/d/uejq-wxrr
- EVA histórica: https://www.datos.gov.co/d/2pnw-mmge
- DIVIPOLA: https://www.datos.gov.co/d/gdxc-w37w
- Precipitación IDEAM: https://www.datos.gov.co/d/s54a-sgyg
- Estaciones IDEAM: https://www.datos.gov.co/d/hp9r-jxuu
- Suelos AGROSAVIA: https://www.datos.gov.co/d/ch4u-f3i5
- Aptitud de café UPRA: https://geoservicios.upra.gov.co/arcgis/rest/services/aptitud_uso_suelo/Aptitud_Cafe_Jul2022/MapServer/0
- NASA POWER Daily API: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- CHIRPS Daily: https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY
