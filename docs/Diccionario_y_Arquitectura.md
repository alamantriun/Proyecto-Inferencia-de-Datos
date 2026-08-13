# Arquitectura y Diccionario de Datos: Pipeline Agro-Rank

Este documento detalla la anatomía de los datos que alimentan al modelo de Inteligencia Artificial (Agro-Rank) y presenta el diagrama de arquitectura del proceso de principio a fin.

---


## 1. Diccionario de Variables (Fuentes y Datos Proporcionados)

*   **MinAgricultura (EVA):** Producción histórica, toneladas por hectárea, áreas sembradas y cosechadas.
*   **IDEAM:** Precipitación diaria bruta. A partir de ella se derivan: días secos consecutivos, precipitación acumulada y variabilidad de lluvias.
*   **AGROSAVIA:** Análisis físico del suelo, niveles de pH, materia orgánica, magnesio.
*   **UPRA:** Porcentaje de aptitud del suelo para el cultivo.
*   **FINAGRO:** Dinero inyectado en el sector (crédito total bruto) y promedios por operación. El logaritmo (`log_credito_total`) se calcula en la etapa de procesamiento.
*   **FRED (Bolsa NY):** Mercado global, precio internacional en USD y variaciones porcentuales.
*   **Panel Temporal (Ingeniería de Datos):** Rezagos biológicos de 1 a 3 años (lags), tendencias productivas, score de confiabilidad (detector de datos falsos).
*   **Target Encoding (Ingeniería de Datos):** Memoria histórica del desempeño municipal y departamental (el "ADN" geográfico).
    *   *¿Por qué se realizó?* Los algoritmos de Machine Learning no entienden texto (nombres de ciudades). Mediante este proceso, reemplazamos el nombre en texto del municipio por su promedio histórico matemático de toneladas por hectárea. Esto le permite a la IA entender instantáneamente qué tan fértil es una región sin necesidad de enviarle complejas coordenadas GPS espaciales.

---

## 2. El Contexto y la Magia del Panel Temporal

El **Panel Temporal** (ubicado al centro del diagrama en la etapa de Procesamiento) es la pieza de ingeniería de datos más importante de todo el proyecto. Es el motor matemático que le permite a la Inteligencia Artificial predecir el futuro en lugar de solo describir el presente.

En su forma cruda, los datos del Ministerio son solo fotos sueltas (*"En 2024, Arauquita produjo 1 tonelada"*). Si le pasamos eso directamente a un algoritmo, el algoritmo no tiene contexto ni memoria. El Panel Temporal soluciona esto de la siguiente manera:

1. **La Máquina del Tiempo (Los Lags):**
   Transforma las fotos sueltas en una línea de tiempo cronológica por municipio. Si tratamos de adivinar el año 2024, el panel esconde esa respuesta y le inyecta al algoritmo el "pasado": `lag_1` (lo que ocurrió en 2023), `lag_2` (2022), etc. Esto es crítico porque cultivos como el café y el cacao son de ciclo permanente; el árbol es el mismo del año pasado. Si un árbol produjo mucho en el `lag_1`, la *inercia biológica* dicta que producirá algo muy similar este año a menos que haya un shock climático.
2. **Las Tendencias (Rolling Features):**
   El Panel procesa matemáticas avanzadas. Calcula la `tendencia_rendimiento_3y` para enseñarle a la IA si los cultivos en ese municipio vienen en picada (por ej. una plaga incontrolada o árboles muriendo de vejez) o si vienen subiendo (porque los campesinos están inyectando tecnología agrícola).
3. **El Detector de Anomalías (Score de Confiabilidad):**
   Finalmente, en Colombia es común encontrar municipios que reportan exactamente las mismas toneladas durante 5 años seguidos por errores administrativos (copiar y pegar en Excel). El Panel Temporal escanea la base de datos, detecta estas "líneas planas biológicamente imposibles" y les baja el puntaje en la variable `score_confiabilidad`. Así, la Inteligencia Artificial aprende a desconfiar de esos datos y no dejarse engañar.

---

## 3. Tecnologías y Herramientas Utilizadas

*   **Lenguaje Base:** Python 3 (Elegido por su supremacía y ecosistema robusto en Ciencia de Datos).
*   **Orquestación:** Bash Scripting (Linux). Utilizado para el archivo `run_multicrop.sh`, permitiendo correr y alterar los archivos de configuración dinámicamente y ejecutar múltiples entrenamientos masivos de forma automatizada.
*   **Motor Predictivo:** `CatBoost Regressor`. Se eligió por encima de XGBoost y Random Forest debido a su manejo nativo de variables categóricas (como los nombres de los 1000 municipios) y su excepcional resistencia a datos faltantes (nulos) sin necesidad de hacer imputaciones que destruyan la realidad de los datos.
*   **Procesamiento de Datos:** `Pandas` y `NumPy`. Herramientas de vectorización indispensables para cruzar cientos de miles de registros climáticos diarios y condensarlos en paneles anuales en cuestión de segundos.
*   **Extracción (ETL):** `SODA API` (Socrata). Para conectarse en vivo a los servidores del gobierno colombiano en lugar de usar archivos estáticos (CSVs obsoletos).
*   **Visualización:** `Matplotlib` puro. Se removieron librerías de alto nivel (como Seaborn) para evitar problemas de dependencias en servidores y tener el control absoluto sobre los ejes, sombras y las líneas de regresión dibujadas pixel por pixel.



