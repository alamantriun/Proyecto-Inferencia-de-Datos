# Solar-Rank

**Sistema de inferencia para priorizar hogares colombianos con mayor viabilidad económica para energía solar.**

> **Naturaleza**: Inferencia predictiva — no causalidad.

---

## Pregunta principal

> ¿Puede un modelo de aprendizaje automático mejorar la priorización de hogares colombianos para proyectos solares, estimando la probabilidad de recuperar la inversión antes de distintos plazos, frente a criterios de selección simples?

## Formulación

```
P(T ≤ X | E, C, G, V, S)
```

| Símbolo | Significado |
|---|---|
| T | Tiempo de recuperación de la inversión |
| X | Horizonte temporal: **3, 5, 7, 10 años** |
| E | Variables energéticas |
| C | Variables económicas |
| G | Variables geográficas y climáticas |
| V | Variables de vivienda |
| S | Características del sistema solar |

## Objetivo central

**No** es simplemente calcular el payback. Es determinar si el ML mejora la **priorización** de hogares candidatos frente a métodos simples.

Escenario: 100.000 hogares candidatos, recursos para intervenir solo 10.000. ¿Qué método selecciona más hogares exitosos?

## Métodos comparados

| # | Método | Tipo |
|---|---|---|
| 1 | Ranking por radiación | Regla simple |
| 2 | Ranking por consumo | Regla simple |
| 3 | Ranking por fórmula económica | Fórmula |
| 4 | Regresión Logística | ML baseline |
| 5 | CatBoost | ML tabular |
| 6 | Transformer compacto | ML temporal |

## Fuentes de datos

Todas colombianas, públicas y documentadas:

| # | Fuente | Entidad | Datos |
|---|---|---|---|
| 1 | SUI Formato 1743 | Superservicios | Consumo, facturación |
| 2 | Tarifas Publicadas | Superservicios | Tarifas, subsidios, estratos |
| 3 | Radiación Solar | UPME | Recurso solar mensual |
| 4 | Atlas Climatológico | IDEAM | Temperatura, precipitación, humedad |
| 5 | LADM-COL | IGAC | Área construida, plantas, año |
| 6 | Sisbén | DNP | Zona, tipo vivienda, cuartos |
| 7 | Indicadores OR | XM/SIMEM | SAIDI, SAIFI |
| 8 | Formato 438 | Superservicios | Vínculo usuario-red (solo joins) |

## Métricas de evaluación

**Priorización**: Precision@K, Recall@K, Lift@K (K = 1%, 5%, 10%, 20%)

**Clasificación**: ROC-AUC, PR-AUC, F1, Brier Score, calibración

**Regresión**: MAE, RMSE, R²

## Salida por hogar

```
Hogar A
Payback esperado: 4.8 años
P(payback ≤ 3)  = 21%
P(payback ≤ 5)  = 68%
P(payback ≤ 7)  = 91%
P(payback ≤ 10) = 98%
Ranking: #137 de 100.000
```

## Estructura del proyecto

```
├── config/           # Configuración YAML
├── data/             # Datos (excluidos de Git)
│   ├── raw/          # Datos crudos descargados
│   ├── interim/      # Tablas intermedias (01-11)
│   ├── processed/    # model_mart listo
│   └── external/     # Datos externos estáticos
├── docs/             # Documentación y artefactos de diseño
├── notebooks/        # Exploración, engineering, modelado
├── src/              # Código fuente del pipeline
│   ├── data/         # Extractores y loaders
│   ├── features/     # Feature engineering
│   ├── models/       # Definición de modelos
│   ├── evaluation/   # Métricas y comparación
│   ├── visualization/
│   └── utils/
├── reports/          # Resultados, figuras, tablas
└── tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Criterio de éxito

El proyecto es exitoso si el ML supera consistentemente los baselines en Precision@K y las probabilidades son razonablemente calibradas.

Si ML ≈ reglas simples, eso también es un **resultado válido**.

## Licencia

Proyecto académico — Ingeniería de Datos.
