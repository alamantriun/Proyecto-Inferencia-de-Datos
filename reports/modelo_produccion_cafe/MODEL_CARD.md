# Model Card — Rendimiento de café

## Estado

`experimental_not_approved`

## Uso permitido

- Priorizar revisión técnica de rendimiento y riesgo productivo.
- Comparar señales históricas municipales antes de una evaluación agronómica humana.

## No usar

- No interpretar la clase alta como rentabilidad, ROI o éxito empresarial.
- No aprobar de forma automática créditos, inversiones, subsidios o intervenciones.
- No usar fuera de Colombia o para cultivos distintos de café sin una nueva validación.

## Modelos

- Clasificación: `catboost_weighted`.
- Regresión: `catboost_regressor`.
- Evaluación congelada hasta 2023 y holdout final 2024.
- Artefacto operativo reentrenado con datos disponibles hasta 2024.

## Fallos de la compuerta

- El MCC promedio no supera al baseline t-1 por al menos 0.02.
- El MCC 2024 es inferior a 0.50.
- La balanced accuracy 2024 es inferior a 0.70.
- La regresión no mejora al baseline t-1 al menos 5% en MAE 2024.

## Limitaciones conocidas

- Existe drift anual y un cambio de fuente EVA a partir de 2019.
- Las variables externas de suelo y crédito fueron excluidas por trazabilidad temporal o geográfica insuficiente.
- `score_confiabilidad` y `dato_copiado` fueron excluidas porque dependen del target.
- Las probabilidades describen rendimiento alto respecto al umbral histórico, no éxito financiero.
