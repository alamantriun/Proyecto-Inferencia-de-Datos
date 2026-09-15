"""Crea y ejecuta el notebook de validación a partir de artefactos verificados."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "validacion_modelo"
NOTEBOOK_PATH = OUTPUT / "validacion_modelo_cafe.ipynb"


def main() -> None:
    summary = json.loads((OUTPUT / "resumen_validacion.json").read_text(encoding="utf-8"))
    metrics = summary["classification_holdout_2024"]
    matrix = summary["confusion_matrix_2024"]
    class_holdout = pd.read_csv(OUTPUT / "comparacion_clasificacion_holdout_2024.csv")
    holdout_best = class_holdout[class_holdout["elegible_como_ganador"]].iloc[0]

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (validación)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            f"""# Validación temporal del modelo de café

## tl;dr

- El objetivo original es continuo: `rendimiento_t_ha`. Para la matriz de confusión se define `rendimiento alto` como un valor superior a **{summary['threshold_high_yield_t_ha']:.6f} t/ha**, mediana de 2007-2018 fijada sin mirar el futuro.
- El modelo elegido con 2019-2023 fue **{summary['classification_winner']}**. En el holdout 2024 obtuvo precisión **{metrics['precision']:.3f}**, exactitud **{metrics['accuracy']:.3f}**, balanced accuracy **{metrics['balanced_accuracy']:.3f}** y MCC **{metrics['mcc']:.3f}**.
- En 2024, **{holdout_best['modelo']}** tuvo el MCC observado más alto ({holdout_best['mcc']:.3f}), pero no se usó para cambiar retrospectivamente al ganador.
- Matriz 2024, filas reales y columnas predichas (`bajo`, `alto`): `{matrix}`.
- La validación es **compartible con cautelas**: hay drift entre años, cambio de fuente en 2019 y variables externas con trazabilidad insuficiente.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

### Key Assumptions

1. La unidad es municipio-cultivo-año.
2. El umbral de clase permanece fijo después de calcularse con 2007-2018.
3. Los modelos se seleccionan con backtest rolling-origin 2019-2023.
4. El año 2024 queda reservado como holdout final.
5. MCC promedio anual es el criterio primario; balanced accuracy desempata.
6. `score_confiabilidad` y `dato_copiado` se excluyen porque dependen del target.
7. Las variables externas se muestran solo como sensibilidad exploratoria; no pueden ganar.

La clase es **rendimiento alto**, no beneficio financiero ni prosperidad empresarial.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "validacion_modelo"

# Cambiar a True para recalcular todos los modelos (~5 minutos en la máquina validada).
FORCE_RETRAIN = False
from src.evaluation.validate_models import run_validation
summary = run_validation(output_dir=OUT, force=FORCE_RETRAIN)
summary"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """data = pd.read_csv(ROOT / summary["data_path"])
quality = pd.read_csv(OUT / "calidad_datos.csv")
balance = pd.read_csv(OUT / "balance_clases.csv")
print(f"Filas: {len(data):,} | Columnas: {data.shape[1]} | Años: {data.anio.min()}-{data.anio.max()}")
display(quality)
display(balance)"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(OUT / "balance_clases.png")))"""
        ),
        nbf.v4.new_markdown_cell(
            """El histórico utilizable está prácticamente balanceado, pero las pruebas no: la clase alta representa 89,5% en 2019, 46,4% en 2022 y 71,4% en 2024. Por eso accuracy sola sería engañosa y se prioriza MCC.
"""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            """class_selection = pd.read_csv(OUT / "comparacion_clasificacion_seleccion.csv")
class_holdout = pd.read_csv(OUT / "comparacion_clasificacion_holdout_2024.csv")
display(class_selection.round(4))
display(class_holdout.round(4))
display(Image(filename=str(OUT / "comparacion_modelos_mcc.png")))"""
        ),
        nbf.v4.new_code_cell(
            """matrix = pd.read_csv(OUT / "matriz_confusion_mejor_modelo.csv", index_col=0)
display(matrix)
display(Image(filename=str(OUT / "matriz_confusion_mejor_modelo.png")))"""
        ),
        nbf.v4.new_markdown_cell(
            """La regresión logística gana por un margen mínimo frente a la regla `rendimiento t-1`. La diferencia no es suficiente para afirmar superioridad concluyente: el baseline gana en tres de los cinco años de selección y los resultados cambian con el año.
"""
        ),
        nbf.v4.new_code_cell(
            """reg_selection = pd.read_csv(OUT / "comparacion_regresion_seleccion.csv")
reg_holdout = pd.read_csv(OUT / "comparacion_regresion_holdout_2024.csv")
display(reg_selection.round(4))
display(reg_holdout.round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Takeaways

1. **Clasificación:** {summary['classification_winner']} es el ganador formal del período de selección, con precisión 2024 de {metrics['precision']:.1%} y MCC {metrics['mcc']:.3f}. El margen de selección fue mínimo y {holdout_best['modelo']} alcanzó MCC {holdout_best['mcc']:.3f} en 2024, por lo que no hay superioridad estable.
2. **Balance:** el entrenamiento histórico está balanceado, pero hay desbalance y drift fuerte fuera de tiempo; MCC y balanced accuracy son indispensables.
3. **Regresión:** {summary['regression_winner']} gana en 2019-2023, pero su R² cae a {summary['regression_holdout_2024']['r2']:.3f} en 2024; el desempeño no es estable.
4. **Riesgos:** deben reconstruirse `score_confiabilidad` solo con historia previa y corregirse joins de suelo/crédito con `departamento + municipio` antes de usar variables externas.
5. **Decisión:** el resultado sirve como validación académica con cautelas, no todavía como soporte suficiente para una inversión agrícola.
"""
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
