# Tarjeta del modelo anual de rendimiento de café

## Estado

**NO AUTORIZADO PARA PRODUCCIÓN: la meta estadística no se alcanzó.**

- Modelo: `logistic_smotenc`
- Umbral de decisión: `0.5500`
- Meta: MCC >= 0,85 y Kappa >= 0,85 en cada año 2019-2025.
- Años que incumplen: 2019, 2020, 2021, 2022, 2023, 2024, 2025.

## Uso previsto

Clasificar, antes de comenzar un año, si el rendimiento municipal de café superará un umbral histórico fijo. Es una herramienta analítica y explicable; no predice rentabilidad, precio, calidad del grano ni resultados de una finca individual.

## Validación cronológica

| Año | Precisión ponderada | MCC | Kappa |
|---:|---:|---:|---:|
| 2019 | 0.8589 | 0.1382 | 0.0766 |
| 2020 | 0.8036 | 0.2952 | 0.2190 |
| 2021 | 0.6721 | 0.1982 | 0.1763 |
| 2022 | 0.7259 | 0.2980 | 0.1978 |
| 2023 | 0.7498 | 0.4853 | 0.4737 |
| 2024 | 0.8451 | 0.5777 | 0.5383 |
| 2025 | 0.8936 | 0.7309 | 0.7305 |

Cada año se predice con información disponible hasta el 31 de diciembre del año anterior. El periodo 2019-2024 se usó para backtesting walk-forward y 2025 como holdout final con receta congelada.

## Fallas de la compuerta

- MCC o Kappa inferior a 0.85 en años: [2019, 2020, 2021, 2022, 2023, 2024, 2025].
- La mejora del peor año frente a persistencia es 0.0054, inferior a 0.02.

## Limitaciones obligatorias

- El cambio entre las fuentes EVA histórica y reciente en 2019 produce deriva fuerte.
- Una precisión ponderada alta no sustituye MCC o Kappa; puede ocultar una clase minoritaria mal predicha.
- SMOTENC sólo balancea folds de entrenamiento y no crea información nueva sobre cambios estructurales.
- El clima NASA POWER se asigna por la grilla más cercana al centroide municipal; no representa necesariamente cada zona cafetera montañosa.
- Las asociaciones y las importancias de variables no deben interpretarse causalmente.

## Regla de despliegue

No desplegar para decisiones automáticas mientras el estado sea `target_not_met`. Para uso exploratorio, mostrar probabilidad, corte de datos y advertencia, y registrar resultados reales para una reevaluación futura.
