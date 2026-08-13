PROYECTO: INFERENCIA PREDICTIVA DEL RENDIMIENTO AGRICOLA EN COLOMBIA

Pregunta:
¿Podemos predecir el rendimiento futuro de un cultivo en un municipio colombiano usando historial productivo, clima, suelo, aptitud territorial y condiciones económicas, y comprobar la predicción con rendimientos reales posteriores?

MVP recomendado:
Cacao, por ser cultivo permanente y tener:
- EVA: rendimiento histórico municipal.
- AGROSAVIA: 92.7K registros nacionales de análisis de suelo.
- UPRA: zonificación de aptitud de cacao.
- AgroNet: precio de referencia de compra de cacao.
- IDEAM: clima.
- ENA: validación independiente agregada.
- AGROSAVIA/SE-MAPA: evidencia de pertinencia agronómica.

Objetivo:
Predecir rendimiento_t (t/ha) y opcionalmente P(rendimiento_t > umbral).

No usar rendimiento del periodo objetivo ni información posterior a la fecha de decisión.

Validación:
Rolling-origin temporal y validación externa con ENA a nivel agregado.
Comparar contra rendimiento del año anterior y promedio histórico.

Resultado de negocio:
Convertir la predicción en:
rendimiento esperado × área × precio
y comparar el desempeño económico potencial de seleccionar municipios/cultivos de mayor rendimiento frente a baselines.
Esto es una estimación de valor, no beneficio neto observado, salvo que se disponga de costos reales.

Importante:
Cámara de Comercio y DIAN se mantienen fuera del núcleo porque la búsqueda no encontró una base pública nacional, granular y temporalmente compatible que explique rendimiento agrícola mejor que EVA/SIPSA/IDEAM/AGROSAVIA/UPRA.
