# Solar-Rank

**Predicción y priorización de hogares con alta viabilidad económica para energía solar en Colombia.**

---

## Pregunta de investigación

> ¿Puede un modelo de aprendizaje automático identificar y priorizar hogares con mayor probabilidad de alcanzar un payback ≤ 5 años, utilizando simultáneamente información energética, económica, geográfica, climática y de vivienda, superando criterios simples basados únicamente en consumo o radiación?

## Targets

| Target | Tipo | Descripción |
|---|---|---|
| `payback_le_5` | Clasificación binaria | 1 si payback ≤ 5 años, 0 si > 5 años |
| `payback_years` | Regresión | Estimación del periodo de recuperación |

## Fuentes de datos

Todas las fuentes son públicas y colombianas con procedencia documentada:

| # | Fuente | Entidad | Datos |
|---|---|---|---|
| 1 | SUI Formato 1743 | Superservicios | Consumo, facturación |
| 2 | Tarifas Publicadas | Superservicios | Tarifas, subsidios, estratos |
| 3 | Radiación Solar | UPME | Recurso solar mensual |
| 4 | Atlas Climatológico | IDEAM | Temperatura, precipitación, humedad |
| 5 | LADM-COL | IGAC | Área construida, plantas, año |
| 6 | Sisbén Vivienda | DNP | Zona, tipo vivienda, cuartos |
| 7 | Indicadores OR | XM/SIMEM | SAIDI, SAIFI |
| 8 | Formato 438 | Superservicios | Vínculo usuario-red (solo joins) |

## Modelos

1. **Logistic Regression** — Baseline
2. **CatBoost** — Tabular fuerte
3. **Temporal Fusion Transformer** — Avanzado temporal

## Estructura del proyecto

```
├── config/           # Configuración YAML del proyecto
├── data/             # Datos (excluidos de Git)
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── docs/             # Documentación y artefactos de diseño
├── notebooks/        # Notebooks de exploración, engineering, modelado
├── src/              # Código fuente del pipeline
├── reports/          # Resultados, figuras, tablas
└── tests/            # Tests unitarios
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Licencia

Proyecto académico — Ingeniería de Datos.
