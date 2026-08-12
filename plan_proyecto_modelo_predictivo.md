# Plan de Proyecto: Modelo Predictivo de Éxito de Negocios

## Contexto general

El objetivo es construir un modelo de no nesesariamente de machine learnig(hay que indagar cual puede ser la mejor herramienta) que,-  a partir de datos reales y de calidad, prediga una respuesta **concreta y medible**. El caso guía es: predecir las chances de éxito de un negocio según su localidad y nicho dentro de un sector, usando variables como flujo de personas, ubicación y necesidad del mercado en esa zona.

El proyecto se divide en 8 etapas. Cada una depende de que la anterior esté bien resuelta — por ejemplo, no tiene sentido elegir un modelo (Etapa 5) si todavía no está claro qué variables importan (Etapa 2).

---

## Etapa 1 — Definición del problema

Antes de tocar datos o código, hay que dejar completamente claro **qué va a predecir el modelo** y en qué formato. No es lo mismo predecir "si un negocio va a funcionar" (vago) que predecir "la probabilidad de que un negocio de cierto nicho supere el punto de equilibrio en los primeros 12 meses en una zona X" (concreto y medible).

- La pregunta debe tener una salida clara: un número, una probabilidad, una categoría.
- Evitar el error común de querer resolver "todo el problema de negocios" — se busca una pregunta acotada y respondible con los datos disponibles.
- Esta etapa se revisita al final (Etapa 8) para verificar que el modelo efectivamente contestó lo que aquí se definió.

## Etapa 2 — Identificación de variables

Aquí se cuestiona **qué factores realmente influyen** en el resultado que se quiere predecir, antes de salir a buscar datos a ciegas.

- Ejemplos de variables candidatas: flujo peatonal/vehicular, densidad poblacional, competencia cercana, poder adquisitivo de la zona, tipo de nicho, estacionalidad.
- Cada variable debe tener una justificación (¿por qué creo que esto afecta el éxito de un negocio?), no agregarse solo porque "suena relacionado".
- Esta etapa también define qué columnas necesitará el dataset — es el puente entre la pregunta de negocio y el diseño técnico de los datos.

## Etapa 3 — Dataset

Con las variables ya identificadas, toca resolver de dónde salen los datos y cómo se estructuran.

- **Fuentes**: deben ser confiables y verificables (datos abiertos gubernamentales, APIs de mapas/movilidad, cámaras de comercio, INEGI/DANE u organismos equivalentes según el país, etc.). La calidad del dato es más importante que la cantidad.
- **Arquitectura del dataset**: definir filas (¿cada negocio? ¿cada zona?), columnas (las variables de la Etapa 2) y granularidad temporal si aplica.
- **Recolección**: si se necesita scraping, hay que decidir entre `lxml` y `BeautifulSoup`. Vale la pena comparar ambos antes de decidir — `lxml` suele ser más rápido en el parseo de HTML/XML a gran escala, mientras que `BeautifulSoup` es más flexible y tolerante con HTML mal formado. La elección depende del volumen de datos y de qué tan "sucias" estén las páginas fuente.

## Etapa 4 — Stack tecnológico

Antes de escribir la primera línea de modelo, conviene decidir con qué se va a construir todo el proyecto.

- Definir lenguaje, librerías (pandas, scikit-learn, XGBoost/CatBoost si aplica, etc.) y herramientas de scraping/carga de datos.
- Evaluar cuál es la tecnología **mínima suficiente** para resolver la pregunta — evitar sumar herramientas complejas si no aportan una mejora real.
- Definir cómo se van a mostrar los resultados finales: ¿notebook con gráficos, dashboard, una API simple, un reporte estático? Esto condiciona parte del stack.

## Etapa 5 — Selección del modelo

Aquí se decide qué algoritmo va a aprender de los datos.

- **Punto de partida obligatorio**: probar una regresión lineal (o logística, si la salida es una probabilidad/categoría) como baseline. Sirve para medir si existe una relación razonablemente simple entre variables y resultado, y da un punto de comparación para cualquier modelo más complejo.
- Solo si el baseline no es suficiente, se evalúan modelos más sofisticados como XGBoost o CatBoost — pero siempre cuestionando *por qué* ese modelo es mejor para este problema específico (tipo de datos, tamaño del dataset, variables categóricas vs numéricas, interpretabilidad necesaria, etc.).
- La elección del modelo debe poder defenderse con argumentos, no ser una decisión "porque está de moda".

## Etapa 6 — Entrenamiento y validación

Esta es la etapa de Machine Learning propiamente dicha.

- Entrenar el modelo elegido con el dataset construido en la Etapa 3.
- Validar con datos separados (train/test split, validación cruzada) para asegurar que el modelo generaliza y no memoriza.
- Esta etapa probablemente requiere cómputo considerable, por lo que suele desarrollarse en un notebook (`.ipynb`) donde se pueda iterar y visualizar resultados paso a paso.

## Etapa 7 — Separación entre construcción y ejecución

Es probable que el entrenamiento del modelo exceda la capacidad de una computadora personal, así que conviene separar claramente:

- **Construcción/entrenamiento**: en un entorno con más cómputo (Google Colab, una nube, un servidor con GPU/CPU potente).
- **Ejecución/inferencia**: una vez entrenado, el modelo final (ya liviano) puede ejecutarse localmente o en un entorno más simple para generar predicciones.
- Definir desde ya este límite evita sorpresas a mitad de proyecto.

## Etapa 8 — Entrega de resultados

El cierre del ciclo: el modelo debe devolver una respuesta concreta y medible a la pregunta planteada en la Etapa 1.

- Verificar que la salida sea interpretable para quien la use (por ejemplo, "68% de probabilidad de éxito" y no solo un número sin contexto).
- Mostrar los resultados a través del medio definido en la Etapa 4 (notebook, dashboard, reporte).
- Confirmar que el modelo resuelve el problema original, sin haberse desviado hacia algo más complejo de lo necesario.

---

## Consideraciones transversales

Estas ideas no pertenecen a una sola etapa, sino que atraviesan todo el proyecto:

- **Calidad del dato por encima de todo**: cualquier dato usado debe venir de fuentes confiables y verificables.
- **Justificar cada decisión**: variable, modelo o tecnología elegida debe tener un "por qué" explícito, no ser la opción por defecto.
- **Simplicidad con fundamento**: el objetivo nunca es usar el modelo más sofisticado, sino dar una respuesta concreta, medible y bien argumentada a la pregunta original.
