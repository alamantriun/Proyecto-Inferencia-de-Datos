"""Generate, execute, and safely replace the rigorous coffee notebook."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from nbclient import NotebookClient
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "pipeline_interactivo_cafe.ipynb"
BACKUP_PATH = ROOT / "pipeline_interactivo_cafe_original.ipynb"
TEMP_PATH = ROOT / ".pipeline_interactivo_cafe.executed.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (modelo café riguroso)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "modeling": {"random_state": 42, "holdout_year": 2024},
    }
    notebook["cells"] = [
        markdown(
            """
# Modelo riguroso de rendimiento de café

Notebook reproducible para auditar datos, balancear **solo el entrenamiento** con SMOTENC, comparar modelos mediante backtesting temporal y evaluar una única vez el holdout 2024.

La clasificación estima **rendimiento alto respecto a un umbral histórico**. No estima rentabilidad, ROI ni éxito empresarial.
"""
        ),
        markdown(
            """
## 1. Resumen y alcance

El flujo desarrolla dos tareas complementarias: clasificación de rendimiento alto/bajo y regresión de toneladas por hectárea. La decisión final se somete a siete controles; si alguno falla, los artefactos quedan marcados como experimentales.
"""
        ),
        markdown(
            """
## 2. Configuración reproducible

Se fijan rutas, semilla y versiones. El notebook no instala paquetes, no usa internet y no solicita entradas manuales.
"""
        ),
        code(
            """
from pathlib import Path
import json
import platform
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.production_pipeline import (
    CATEGORICAL_FEATURES,
    CLASS_TARGET,
    DATA_PATH,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    TARGET,
    add_high_yield_target,
    audit_dataset,
    load_dataset,
    run_production_pipeline,
    smotenc_class_counts,
    temporal_splits,
)

np.random.seed(42)
sns.set_theme(style="whitegrid")
print({"python": platform.python_version(), "semilla": 42, "raiz": str(ROOT)})
"""
        ),
        markdown(
            """
## 3. Carga del dataset

La unidad analítica esperada es `departamento + municipio + cultivo + anio`.
"""
        ),
        code(
            """
raw_data = load_dataset(DATA_PATH)
print(f"Filas: {len(raw_data):,} | Columnas: {raw_data.shape[1]} | Años: {raw_data.anio.min()}-{raw_data.anio.max()}")
display(raw_data.head(3))
"""
        ),
        markdown(
            """
## 4. Auditoría de calidad y leakage

La ejecución se detiene ante llaves duplicadas, target nulo, años faltantes o inconsistencias del lag de un año. `score_confiabilidad` y `dato_copiado` se excluyen porque dependen del target.
"""
        ),
        code(
            """
audit = audit_dataset(raw_data)
display(pd.DataFrame({
    "control": ["Filas", "Llaves duplicadas", "Target nulo", "Comparaciones lag", "Coincidencias lag"],
    "resultado": [audit["rows"], audit["duplicate_keys"], audit["target_nulls"], audit["lag_comparisons"], audit["lag_matches"]],
}))
display(pd.Series(audit["missing_percent"], name="faltante_pct").sort_values(ascending=False).head(15).to_frame())
print("Variables excluidas por leakage:", ", ".join(audit["excluded_leakage_columns"]))
"""
        ),
        markdown(
            """
## 5. Exploración inicial

Se observan distribución, evolución anual y faltantes antes de transformar datos. Las unidades del target son toneladas por hectárea.
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
sns.histplot(raw_data[TARGET], bins=35, kde=True, color="#2f6f4e", ax=axes[0])
axes[0].set(title="Rendimiento inicial (2007-2024)", xlabel="t/ha", ylabel="Observaciones")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sns.boxplot(data=raw_data, x="anio", y=TARGET, color="#8eb69b", fliersize=1, ax=axes[1])
axes[1].set(title="Rendimiento por año", xlabel="Año", ylabel="t/ha")
axes[1].tick_params(axis="x", rotation=65)
missing = raw_data.isna().mean().mul(100).sort_values(ascending=False).head(12)
sns.barplot(x=missing.values, y=missing.index, color="#a44a3f", ax=axes[2])
axes[2].set(title="Variables con más faltantes", xlabel="Faltante (%)", ylabel="Variable")
plt.tight_layout()
plt.show()
"""
        ),
        markdown(
            """
## 6. Target y balance inicial

El umbral se calcula con la mediana observada hasta 2018 y queda congelado antes de los años de selección y holdout.
"""
        ),
        code(
            """
data, high_yield_threshold = add_high_yield_target(raw_data)
balance_initial = data.groupby("anio")[CLASS_TARGET].agg(["count", "sum", "mean"])
balance_initial["clase_alta_pct"] = balance_initial["mean"] * 100
print(f"Umbral histórico congelado: {high_yield_threshold:.6f} t/ha")
display(balance_initial[["count", "sum", "clase_alta_pct"]].round(2))
balance_initial["clase_alta_pct"].plot(kind="bar", figsize=(11, 4), color="#4d9078", title="Clase de rendimiento alto por año")
plt.axhline(50, linestyle="--", color="#a44a3f")
plt.ylabel("Clase alta (%)")
plt.xlabel("Año")
plt.tight_layout()
plt.show()
"""
        ),
        markdown(
            """
## 7. Features y separación temporal

Los modelos usan geografía conocida y variables históricas rezagadas. No se usa una división aleatoria.
"""
        ),
        code(
            """
splits = temporal_splits(data)
display(pd.DataFrame({
    "periodo": ["Historia", "Selección", "Holdout"],
    "años": ["2007-2018", "2019-2023", "2024"],
    "filas": [len(splits["history"]), len(splits["selection"]), len(splits["holdout"])],
}))
print("Categóricas:", CATEGORICAL_FEATURES)
print("Numéricas estrictas:", NUMERIC_FEATURES)
"""
        ),
        markdown(
            """
## 8. SMOTENC aplicado únicamente a entrenamiento

SMOTENC conoce cuáles columnas son categóricas. Se ajusta sobre 2007-2018 y las filas sintéticas nunca se mezclan con validación o 2024.
"""
        ),
        code(
            """
smote_snapshot = smotenc_class_counts(splits["history"])
display(smote_snapshot)
smote_snapshot.set_index("class_name")[["before", "after"]].plot(kind="bar", figsize=(8, 4), color=["#a44a3f", "#4d9078"])
plt.title("Clases antes y después de SMOTENC — train 2007-2018")
plt.xlabel("Clase")
plt.ylabel("Filas")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
        ),
        markdown(
            """
## 9. Backtest de clasificadores

Cada año 2019-2023 se predice entrenando exclusivamente con años anteriores. Aquí se ejecuta el pipeline completo y se publican artefactos solo después de terminar todas las validaciones.
"""
        ),
        code(
            """
result = run_production_pipeline(DATA_PATH, OUTPUT_DIR)
manifest = result["manifest"]
display(result["classification_summary"].sort_values("mean_mcc", ascending=False).round(4))
for figure_name in [
    "01_distribucion_rendimiento.png", "02_boxplot_rendimiento_anio.png",
    "03_faltantes_variables.png", "04_balance_clase_anio.png",
    "05_smotenc_antes_despues.png", "06_modelos_mcc_balanced_accuracy.png",
]:
    display(Image(filename=str(OUTPUT_DIR / figure_name)))
"""
        ),
        markdown(
            """
## 10. Comparación y estabilidad anual

MCC es la métrica principal porque considera simultáneamente las cuatro celdas de la matriz de confusión y es más informativa ante desbalance.
"""
        ),
        code(
            """
winner = manifest["classification"]["winner"]
winner_annual = result["classification_annual"].query("model == @winner")
display(winner_annual[["anio", "mcc", "balanced_accuracy", "precision", "recall"]].round(4))
print(f"Challenger seleccionado sin mirar 2024: {winner}")
print(f"Rango MCC anual: {winner_annual.mcc.min():.3f} a {winner_annual.mcc.max():.3f}")
display(Image(filename=str(OUTPUT_DIR / "07_mcc_anual.png")))
"""
        ),
        markdown(
            """
## 11. Selección, calibración y umbral congelado

El calibrador y el umbral usan únicamente predicciones fuera de muestra de 2019-2023. Ambos se congelan antes de evaluar 2024.
"""
        ),
        code(
            """
print({
    "modelo": winner,
    "umbral_decision": manifest["classification"]["decision_threshold"],
    "umbral_rendimiento_t_ha": manifest["targets"]["high_yield_threshold_t_ha"],
})
display(Image(filename=str(OUTPUT_DIR / "10_calibracion_2024.png")))
"""
        ),
        markdown(
            """
## 12. Holdout 2024 y matriz de confusión

2024 no participó en imputación, escalado, SMOTENC, selección, calibración ni elección del umbral. Filas: realidad; columnas: predicción.
"""
        ),
        code(
            """
class_metrics = manifest["classification"]["holdout_2024"]
matrix = pd.DataFrame(
    class_metrics["confusion_matrix"],
    index=["Real bajo", "Real alto"],
    columns=["Predicho bajo", "Predicho alto"],
)
display(matrix)
display(pd.Series({key: class_metrics[key] for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc", "brier", "ece"]}, name="2024").round(4).to_frame())
print(f"Lectura crítica: {class_metrics['fn']} rendimientos altos quedaron clasificados como bajos y {class_metrics['fp']} bajos como altos.")
display(Image(filename=str(OUTPUT_DIR / "08_matriz_confusion_2024.png")))
display(Image(filename=str(OUTPUT_DIR / "09_curvas_roc_pr_2024.png")))
"""
        ),
        markdown(
            """
## 13. Backtest de regresores

SMOTE no se aplica al target continuo. Elastic Net, Random Forest y CatBoost se comparan con rendimiento `t-1` y media móvil de tres años.
"""
        ),
        code(
            """
display(result["regression_summary"].sort_values("mean_mae").round(4))
print("Regresor seleccionado sin mirar 2024:", manifest["regression"]["winner"])
display(Image(filename=str(OUTPUT_DIR / "11_mae_modelo_anio.png")))
"""
        ),
        markdown(
            """
## 14. Evaluación final de regresión

MAE y RMSE están expresados en toneladas por hectárea. La compuerta exige mejorar el MAE del baseline `t-1` al menos 5%.
"""
        ),
        code(
            """
reg_metrics = manifest["regression"]["holdout_2024"]
display(pd.Series(reg_metrics, name="2024").round(4).to_frame())
display(Image(filename=str(OUTPUT_DIR / "12_real_vs_predicho_2024.png")))
"""
        ),
        markdown(
            """
## 15. Importancia de variables y errores

La importancia se calcula por permutación sobre 2024 para mantener un criterio común entre familias de modelos. No implica causalidad.
"""
        ),
        code(
            """
importance = pd.read_csv(OUTPUT_DIR / "importancia_variables.csv")
display(importance.round(4))
display(Image(filename=str(OUTPUT_DIR / "13_importancia_variables.png")))
"""
        ),
        markdown(
            """
## 16. Compuerta de producción y limitaciones

El modelo solo recibe estado `production_candidate` si supera simultáneamente los siete controles predefinidos. Un buen resultado aislado no compensa inestabilidad o falta de mejora frente a baselines.
"""
        ),
        code(
            """
gate_table = pd.DataFrame(manifest["gate"]["checks"])
display(gate_table)
print("ESTADO FINAL:", manifest["status"])
if manifest["gate"]["failed_checks"]:
    print("Razones para no aprobar:")
    for reason in manifest["gate"]["failed_checks"]:
        print("-", reason)
else:
    print("El candidato superó todos los criterios definidos; aún requiere monitoreo de drift y revisión agronómica.")
"""
        ),
        markdown(
            """
## 17. Exportación y resumen final

Se guardan modelos de evaluación, modelos operativos reentrenados hasta 2024, calibrador, manifiesto, model card, predicciones fuera de muestra, tablas y las 13 figuras. El manifiesto conserva las métricas del modelo evaluado hasta 2023; no las sustituye por resultados del reentrenamiento operativo.
"""
        ),
        code(
            """
artifacts = sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file())
print(f"Estado: {manifest['status']}")
print(f"Artefactos generados: {len(artifacts)}")
display(pd.DataFrame({"archivo": artifacts}))
print("Uso válido: priorización de rendimiento y riesgo productivo con revisión humana.")
print("Uso inválido: afirmar rentabilidad o aprobar automáticamente crédito/inversión.")
"""
        ),
    ]
    return notebook


def main() -> None:
    if NOTEBOOK_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(NOTEBOOK_PATH, BACKUP_PATH)

    notebook = build_notebook()
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbf.write(notebook, TEMP_PATH)
    os.replace(TEMP_PATH, NOTEBOOK_PATH)
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Respaldo: {BACKUP_PATH}")
    print(f"Reporte: {ROOT / 'reports' / 'modelo_produccion_cafe'}")


if __name__ == "__main__":
    main()
