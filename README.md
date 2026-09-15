# Clasificador anual de rendimiento municipal de café

Proyecto reproducible para estimar, antes de comenzar un año, si el rendimiento municipal de café quedará por encima de un umbral histórico fijo. El objetivo es una **clase de rendimiento**, no rentabilidad, precio, calidad del grano ni éxito de una finca.

## Resultado validado

- Desarrollo walk-forward: 2019-2024.
- Holdout final: 2025, abierto después de congelar algoritmo y umbral.
- Modelo seleccionado: regresión logística L2 con SMOTENC dentro del entrenamiento.
- Umbral de decisión: 0,55.
- Estado: `target_not_met`.

El modelo **no cumple** MCC >= 0,85 y Kappa >= 0,85 en cada año. En 2025 obtuvo precisión ponderada 0,8936, MCC 0,7309 y Kappa 0,7305. Por tanto, el paquete es funcional para presentación, auditoría y experimentación, pero no está autorizado para decisiones automáticas de producción.

## Entregable principal

Abrir y ejecutar [pipeline_interactivo_cafe.ipynb](pipeline_interactivo_cafe.ipynb). Tiene 20 secciones ordenadas con:

- fuentes y hashes;
- auditoría de calidad y leakage;
- distribución inicial, nulos y balance por año;
- demostración de SMOTENC sólo en entrenamiento;
- comparación de siete alternativas, incluidos baselines;
- métricas estilo WEKA, MCC y Kappa;
- matriz de confusión para cada año 2019-2025;
- intervalos bootstrap, calibración e importancia por permutación;
- compuerta estricta y conclusión crítica.

El notebook usa por defecto la corrida congelada para abrir rápido. Para repetir todos los ajustes, cambiar `RETRAIN_MODELS = False` a `True`.

## Datos

- EVA histórica 2007-2018 y EVA reciente 2019-2025.
- DIVIPOLA DANE para códigos municipales y centroides.
- NASA POWER MERRA-2 mensual 2006-2024 para clima rezagado.

La llave canónica es `codigo_dane_municipio + anio`. Para predecir el año `t`, todas las variables provienen como máximo de `t-1`.

## Ejecución

En PowerShell, desde la raíz del proyecto:

```powershell
python -m pip install -r requirements.txt
python -m src.evaluation.annual_pipeline
python scripts/create_mcc_kappa_notebook.py
python -m pytest -q
```

Para regenerar sólo las figuras y tablas desde la corrida guardada:

```powershell
python -m src.evaluation.annual_pipeline --reuse-result-cache
```

## Artefactos

La carpeta `reports/modelo_mcc_kappa_cafe/` incluye predicciones fuera de muestra, métricas anuales, tablas WEKA, matrices, 14 figuras, modelo serializado, receta congelada antes de 2025, manifiestos con SHA-256 y `MODEL_CARD.md`.

## Uso responsable

No afirmar que la precisión ponderada equivale a calidad equilibrada. El cambio de fuente EVA de 2018 a 2019 altera fuertemente la prevalencia de la clase y SMOTENC no puede crear información que no existe. Cualquier intento futuro debe conservar la validación cronológica y el holdout independiente.
