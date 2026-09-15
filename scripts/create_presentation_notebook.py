"""Build, execute, and replace the coffee notebook with a compact presentation."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from nbclient import NotebookClient
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "pipeline_interactivo_cafe.ipynb"
RIGOROUS_BACKUP_PATH = ROOT / "pipeline_interactivo_cafe_riguroso.ipynb"
TEMP_PATH = ROOT / ".pipeline_interactivo_cafe_presentacion.executed.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (modelo café presentable)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "modeling": {
            "random_state": 42,
            "selection_years": [2019, 2020, 2021, 2022, 2023],
            "evaluation_year": 2024,
            "weka_weighted_precision_target": 0.85,
        },
    }
    notebook["cells"] = [
        markdown(
            """
# Modelo pequeño y funcional para rendimiento de café

Este notebook responde: **¿el rendimiento de café de un municipio será alto o bajo?**

Utiliza una regresión logística, compara entrenamiento normal contra entrenamiento con SMOTENC y genera predicciones para múltiples casos. La meta de presentación es superar **85% en `Precision` de la fila `Weighted Avg.` del reporte tipo WEKA**. También se muestran accuracy, recall y MCC para no confundir una sola métrica con calidad general.
"""
        ),
        markdown(
            """
## 1. Objetivo y respuesta esperada

Cada fila representa un caso `departamento + municipio + año`. El resultado contiene una probabilidad y una clase: `Rendimiento alto` o `Rendimiento bajo`.

El modelo es pequeño porque toda la decisión final es una sola ecuación logística. SMOTENC únicamente crea ejemplos sintéticos dentro del entrenamiento; nunca modifica validación ni casos futuros.
"""
        ),
        markdown(
            """
## 2. Configuración reproducible

Se fijan la semilla, las rutas y un umbral de decisión de 0.50. El notebook no necesita internet ni entradas manuales para ejecutarse.
"""
        ),
        code(
            """
from pathlib import Path
import json
import platform
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.production_pipeline import (
    CATEGORICAL_FEATURES,
    CLASS_TARGET,
    DATA_PATH,
    NUMERIC_FEATURES,
    TARGET,
    add_high_yield_target,
    audit_dataset,
    build_classifier,
    classification_backtest,
    classification_metric_tables,
    load_dataset,
    permutation_feature_importance,
    select_classifier,
    smotenc_class_counts,
)
from src.evaluation.presentation_model import (
    FEATURES,
    PRESENTATION_MODELS,
    evaluate_year_raw,
    predict_cases,
    weka_report,
)

SELECTION_YEARS = list(range(2019, 2024))
EVALUATION_YEAR = 2024
DECISION_THRESHOLD = 0.50
WEKA_PRECISION_TARGET = 0.85
OUTPUT_DIR = ROOT / "reports" / "modelo_presentacion_cafe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})
print({"python": platform.python_version(), "semilla": 42, "salida": str(OUTPUT_DIR)})
"""
        ),
        markdown(
            """
## 3. Datos iniciales y control de calidad

La unidad analítica es municipio–cultivo–año. Antes de modelar se comprueban duplicados, valores faltantes del objetivo y que el rendimiento del año anterior corresponda realmente al municipio.
"""
        ),
        code(
            """
datos_crudos = load_dataset(DATA_PATH)
auditoria = audit_dataset(datos_crudos)
datos, umbral_rendimiento = add_high_yield_target(datos_crudos)

resumen_datos = pd.DataFrame({
    "indicador": ["Filas", "Columnas", "Año inicial", "Año final", "Llaves duplicadas", "Target nulo"],
    "valor": [auditoria["rows"], auditoria["columns"], auditoria["year_min"], auditoria["year_max"], auditoria["duplicate_keys"], auditoria["target_nulls"]],
})
display(resumen_datos)
print(f"Rendimiento alto: más de {umbral_rendimiento:.3f} t/ha; umbral calculado solo con datos hasta 2018.")
"""
        ),
        markdown(
            """
## 4. Gráficas del estado inicial

Se observa la distribución del rendimiento y el balance real de clases por año. El cambio entre años es importante: una precisión alta en un periodo no garantiza el mismo resultado en todos los demás.
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(9, 4.8))
sns.histplot(datos[TARGET], bins=40, color="#35618A", edgecolor="white", ax=ax)
ax.axvline(umbral_rendimiento, color="#C8872A", linewidth=2, linestyle="--", label=f"Umbral = {umbral_rendimiento:.3f} t/ha")
ax.set(title="Distribución del rendimiento histórico", xlabel="Rendimiento (t/ha)", ylabel="Número de casos")
ax.legend(frameon=False)
fig.text(0.5, -0.02, f"Fuente: dataset local | {len(datos):,} casos | años {datos.anio.min()}–{datos.anio.max()}", ha="center", color="#555555")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_distribucion_rendimiento.png", bbox_inches="tight", dpi=160)
plt.show()

balance_anual = (
    datos.groupby(["anio", CLASS_TARGET]).size().unstack(fill_value=0)
    .rename(columns={0: "Bajo", 1: "Alto"})
)
balance_porcentual = balance_anual.div(balance_anual.sum(axis=1), axis=0).mul(100)
balance_anual.to_csv(OUTPUT_DIR / "balance_inicial.csv")

ax = balance_porcentual.plot(kind="bar", stacked=True, figsize=(11, 4.8), color=["#9DB7D1", "#C8872A"], edgecolor="#303030")
ax.set(title="Balance de clases por año", xlabel="Año", ylabel="Porcentaje de casos", ylim=(0, 100))
ax.legend(title="Clase", frameon=False, ncol=2, loc="upper center")
ax.figure.text(0.5, -0.02, "Porcentaje dentro de cada año; los tamaños anuales no son idénticos.", ha="center", color="#555555")
ax.figure.tight_layout()
ax.figure.savefig(OUTPUT_DIR / "02_balance_clases_por_anio.png", bbox_inches="tight", dpi=160)
plt.show()
display(balance_anual.tail(6))
"""
        ),
        markdown(
            """
## 5. Validación temporal

Para simular el uso real, cada año se predice entrenando exclusivamente con años anteriores. La selección usa 2019–2023. Después se entrena hasta 2023 y se evalúa 2024 con el umbral fijo de 0.50.
"""
        ),
        code(
            """
ventanas = pd.DataFrame({
    "etapa": ["Definir clase alta", "Seleccionar versión", "Evaluación posterior"],
    "años usados": ["2007–2018", "2019–2023, walk-forward", "2024"],
    "uso": ["Fijar umbral", "Comparar normal vs SMOTENC", "Medir generalización temporal"],
})
display(ventanas)
"""
        ),
        markdown(
            """
## 6. Modelo explicable y balanceo con SMOTENC

La regresión logística combina ubicación, año y antecedentes productivos. Su salida es una probabilidad entre 0 y 1. SMOTENC balancea las dos clases respetando que departamento y municipio son variables categóricas.
"""
        ),
        code(
            """
historia_entrenamiento = datos[(datos.anio <= 2018) & datos.rendimiento_lag_1.notna()].copy()
conteo_smote = smotenc_class_counts(historia_entrenamiento)
conteo_smote.to_csv(OUTPUT_DIR / "smotenc_antes_despues.csv", index=False)
display(conteo_smote)

grafica_smote = conteo_smote.set_index("class_name")[["before", "after"]].rename(columns={"before": "Antes", "after": "Después"})
ax = grafica_smote.plot(kind="bar", figsize=(8, 4.8), color=["#9DB7D1", "#C8872A"], edgecolor="#303030", rot=0)
ax.set(title="Balance del entrenamiento antes y después de SMOTENC", xlabel="Clase", ylabel="Número de casos")
ax.legend(title="Muestra", frameon=False)
ax.figure.text(0.5, -0.02, "SMOTENC se aplica solo al entrenamiento 2007–2018 en esta ilustración.", ha="center", color="#555555")
ax.figure.tight_layout()
ax.figure.savefig(OUTPUT_DIR / "03_smotenc_antes_despues.png", bbox_inches="tight", dpi=160)
plt.show()
"""
        ),
        markdown(
            """
## 7. Comparación histórica y selección

Se comparan únicamente dos versiones del mismo algoritmo: regresión logística normal y regresión logística con SMOTENC. El ganador se elige por MCC promedio anual en 2019–2023; 2024 todavía no participa.
"""
        ),
        code(
            """
predicciones_seleccion_todas = classification_backtest(
    datos,
    high_yield_threshold=umbral_rendimiento,
    years=SELECTION_YEARS,
    model_names=list(PRESENTATION_MODELS),
)
predicciones_seleccion = predicciones_seleccion_todas[
    predicciones_seleccion_todas.model.isin(PRESENTATION_MODELS)
].copy()
metricas_anuales_seleccion, resumen_modelos = classification_metric_tables(predicciones_seleccion)

filas_weka = []
for (modelo, anio), grupo in predicciones_seleccion.groupby(["model", "anio"], sort=True):
    reporte = weka_report(grupo.y_true, grupo.y_pred, grupo.probability_raw)
    filas_weka.append({
        "model": modelo,
        "anio": int(anio),
        "weighted_precision": reporte["summary"]["weighted_precision"],
        "precision_high": reporte["summary"]["precision_high"],
    })
weka_seleccion = pd.DataFrame(filas_weka)
promedio_weka = weka_seleccion.groupby("model", as_index=False)[["weighted_precision", "precision_high"]].mean()
comparacion = resumen_modelos.merge(promedio_weka, on="model", how="left")
ganador = select_classifier(comparacion)

comparacion.to_csv(OUTPUT_DIR / "comparacion_modelos.csv", index=False)
metricas_anuales_seleccion.to_csv(OUTPUT_DIR / "metricas_seleccion_por_anio.csv", index=False)
display(comparacion[["model", "mean_accuracy", "weighted_precision", "mean_mcc", "mean_recall"]].round(4))
print("Modelo seleccionado sin observar 2024:", ganador)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
orden = list(PRESENTATION_MODELS)
tabla_plot = comparacion.set_index("model").reindex(orden)
axes[0].barh(orden, tabla_plot.mean_mcc, color="#35618A", edgecolor="#303030")
axes[0].set(title="MCC promedio de selección", xlabel="MCC", ylabel="")
axes[0].set_xlim(0, max(0.35, tabla_plot.mean_mcc.max() * 1.25))
axes[1].barh(orden, tabla_plot.weighted_precision.mul(100), color="#C8872A", edgecolor="#303030")
axes[1].axvline(85, color="#303030", linestyle="--", linewidth=1.5, label="Meta 85%")
axes[1].set(title="Precision WEKA promedio", xlabel="Weighted Avg. Precision (%)", ylabel="", xlim=(0, 100))
axes[1].legend(frameon=False)
fig.suptitle("Comparación de regresión logística, 2019–2023")
fig.text(0.5, -0.02, "Cada año fue predicho usando únicamente años anteriores; los promedios dan el mismo peso a cada año.", ha="center", color="#555555")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_comparacion_modelos.png", bbox_inches="tight", dpi=160)
plt.show()
"""
        ),
        markdown(
            """
## 8. Evaluación final con criterios de WEKA

El reporte conserva la terminología de WEKA: `TP Rate`, `FP Rate`, `Precision`, `Recall`, `F-Measure`, `MCC`, `ROC Area` y `PRC Area`. La condición solicitada se verifica en `Weighted Avg. > 0.85`.
"""
        ),
        code(
            """
predicciones_2024, modelo_evaluacion = evaluate_year_raw(
    datos, ganador, year=EVALUATION_YEAR, threshold=DECISION_THRESHOLD
)
reporte_weka_2024 = weka_report(
    predicciones_2024.y_true,
    predicciones_2024.y_pred,
    predicciones_2024.probability_raw,
)
detalle_weka_2024 = reporte_weka_2024["detailed_accuracy"]
resumen_2024 = reporte_weka_2024["summary"]
cumple_85 = resumen_2024["weighted_precision"] > WEKA_PRECISION_TARGET

predicciones_2024.to_csv(OUTPUT_DIR / "predicciones_2024.csv", index=False)
detalle_weka_2024.to_csv(OUTPUT_DIR / "reporte_weka_2024.csv", index=False)
display(detalle_weka_2024.round(4))
print(f"Weighted Avg. Precision: {resumen_2024['weighted_precision']:.2%}")
print(f"Accuracy: {resumen_2024['accuracy']:.2%} | MCC: {resumen_2024['mcc']:.3f} | Recall clase alta: {resumen_2024['recall_high']:.2%}")
print("¿Supera estrictamente 85% de Precision WEKA?:", cumple_85)
"""
        ),
        markdown(
            """
## 9. Matriz de confusión y comportamiento por año

La matriz permite ver cuántos casos altos y bajos se aciertan o se pierden. Después se reúnen predicciones fuera de muestra de seis años: cada caso fue evaluado con un modelo entrenado solamente con su pasado.
"""
        ),
        code(
            """
matriz = np.asarray(resumen_2024["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6.4, 5.2))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1, linecolor="white", ax=ax)
ax.set(title="Matriz de confusión de 2024", xlabel="Clase predicha", ylabel="Clase real")
ax.set_xticklabels(["Bajo", "Alto"])
ax.set_yticklabels(["Bajo", "Alto"], rotation=0)
fig.text(0.5, -0.01, f"n={matriz.sum()} | umbral de decisión={DECISION_THRESHOLD:.2f} | modelo={ganador}", ha="center", color="#555555")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_matriz_confusion_2024.png", bbox_inches="tight", dpi=160)
plt.show()

seleccion_ganador = predicciones_seleccion[predicciones_seleccion.model == ganador].copy()
predicciones_multianio = pd.concat([seleccion_ganador, predicciones_2024], ignore_index=True, sort=False)
predicciones_multianio["prediccion"] = predicciones_multianio.y_pred.map({0: "Rendimiento bajo", 1: "Rendimiento alto"})
predicciones_multianio["resultado_real"] = predicciones_multianio.y_true.map({0: "Rendimiento bajo", 1: "Rendimiento alto"})
predicciones_multianio["acierto"] = predicciones_multianio.y_true.eq(predicciones_multianio.y_pred)
predicciones_multianio.to_csv(OUTPUT_DIR / "predicciones_multianio.csv", index=False)

filas_anuales = []
for anio, grupo in predicciones_multianio.groupby("anio", sort=True):
    reporte = weka_report(grupo.y_true, grupo.y_pred, grupo.probability_raw)["summary"]
    filas_anuales.append({"anio": int(anio), **{k: v for k, v in reporte.items() if k != "confusion_matrix"}})
metricas_multianio = pd.DataFrame(filas_anuales)
metricas_multianio.to_csv(OUTPUT_DIR / "metricas_multianio.csv", index=False)

datos_2024 = datos.loc[predicciones_2024.row_id].copy()
importancia = permutation_feature_importance(
    modelo_evaluacion,
    datos_2024,
    predicciones_2024.y_true.to_numpy(),
    repeats=5,
)
importancia.to_csv(OUTPUT_DIR / "importancia_variables.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
axes[0].bar(metricas_multianio.anio.astype(str), metricas_multianio.weighted_precision.mul(100), color="#C8872A", edgecolor="#303030")
axes[0].axhline(85, color="#303030", linestyle="--", linewidth=1.5)
axes[0].set(title="Precision WEKA por año", xlabel="Año evaluado", ylabel="Weighted Avg. Precision (%)", ylim=(0, 100))
axes[1].bar(metricas_multianio.anio.astype(str), metricas_multianio.mcc, color="#35618A", edgecolor="#303030")
axes[1].axhline(0, color="#303030", linewidth=1)
axes[1].set(title="MCC por año", xlabel="Año evaluado", ylabel="MCC", ylim=(-0.1, 1))
top_importancia = importancia.head(8).sort_values("importance")
axes[2].barh(top_importancia.feature, top_importancia.importance, color="#6E8B74", edgecolor="#303030")
axes[2].axvline(0, color="#303030", linewidth=1)
axes[2].set(title="Importancia por permutación", xlabel="Pérdida de MCC al permutar", ylabel="")
fig.suptitle("Resultados temporales y variables más útiles")
fig.text(0.5, -0.03, "Predicciones fuera de muestra 2019–2024. La importancia se mide en la evaluación 2024 y no implica causalidad.", ha="center", color="#555555")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_resultados_multianio_e_importancia.png", bbox_inches="tight", dpi=160)
plt.show()
display(metricas_multianio[["anio", "instances", "accuracy", "weighted_precision", "precision_high", "recall_high", "mcc"]].round(4))
"""
        ),
        markdown(
            """
## 10. Predicción de varios casos

La función `predict_cases(modelo, casos)` procesa cualquier cantidad de filas con el mismo esquema de variables. Abajo se muestran casos de 2019 a 2024 que siempre fueron predichos fuera de muestra.
"""
        ),
        code(
            """
casos_ejemplo = pd.concat(
    [grupo.sample(n=min(4, len(grupo)), random_state=42) for _, grupo in predicciones_multianio.groupby("anio", sort=True)],
    ignore_index=True,
)
columnas_ejemplo = ["departamento", "municipio", "anio", "resultado_real", "prediccion", "probability_raw", "acierto"]
casos_ejemplo[columnas_ejemplo].to_csv(OUTPUT_DIR / "casos_ejemplo.csv", index=False)
display(casos_ejemplo[columnas_ejemplo].rename(columns={"probability_raw": "probabilidad_alto"}).round(4))

# Demostración por lotes: el modelo entrenado hasta 2023 recibe 12 casos de 2024.
casos_lote_2024 = datos_2024.head(12)
prediccion_lote = predict_cases(modelo_evaluacion, casos_lote_2024, threshold=DECISION_THRESHOLD)
display(prediccion_lote.round(4))
"""
        ),
        markdown(
            """
## 11. Explicación sencilla del modelo

La regresión logística calcula `P(alto) = 1 / (1 + exp(-z))`. En `z` suma los antecedentes productivos y la ubicación con pesos aprendidos. Si la probabilidad es al menos 0.50, devuelve `Rendimiento alto`.

La importancia por permutación indica cuánto baja el MCC cuando una variable se desordena. Un valor alto señala utilidad predictiva, no una relación causal.
"""
        ),
        code(
            """
display(importancia.head(10).round(4))
print("Variables de entrada:")
for variable in FEATURES:
    print("-", variable)
print()
print("El modelo final sigue siendo una sola regresión logística; SMOTENC es únicamente una técnica de preparación del entrenamiento.")
"""
        ),
        markdown(
            """
## 12. Conclusión, artefactos y límites

La condición de 85% se declara cumplida solo si la fila `Weighted Avg.` del reporte ejecutado la supera. Esto **no significa 85% de accuracy**, ni garantiza 85% en todos los años. El cambio de distribución temporal obliga a reportar el resultado año por año.
"""
        ),
        code(
            """
# Modelo operativo para puntuar casos posteriores a 2024 cuando sus variables estén disponibles.
entrenamiento_operativo = datos[(datos.anio <= 2024) & datos.rendimiento_lag_1.notna()].copy()
modelo_operativo = build_classifier(ganador)
modelo_operativo.fit(entrenamiento_operativo[FEATURES], entrenamiento_operativo[CLASS_TARGET].astype(int))
joblib.dump(modelo_evaluacion, OUTPUT_DIR / "modelo_evaluacion_hasta_2023.joblib")
joblib.dump(modelo_operativo, OUTPUT_DIR / "modelo_operativo_hasta_2024.joblib")

pd.DataFrame(columns=FEATURES).to_csv(OUTPUT_DIR / "plantilla_casos_futuros.csv", index=False)

resumen_modelo = {
    "status": "presentation_ready_not_production_approved",
    "winner": ganador,
    "model_family": "logistic_regression",
    "selection_years": SELECTION_YEARS,
    "evaluation_year": EVALUATION_YEAR,
    "decision_threshold": DECISION_THRESHOLD,
    "high_yield_threshold_t_ha": float(umbral_rendimiento),
    "weka_precision_target": WEKA_PRECISION_TARGET,
    "meets_weka_precision_85": bool(cumple_85),
    "holdout_2024": resumen_2024,
    "limitations": [
        "Weighted Precision no equivale a accuracy.",
        "El desempeño cambia entre años por deriva de datos.",
        "2024 ya fue inspeccionado durante el desarrollo; una aprobación futura requiere datos nuevos, idealmente 2025.",
        "La predicción apoya priorización y requiere revisión humana.",
    ],
}
(OUTPUT_DIR / "resumen_modelo.json").write_text(json.dumps(resumen_modelo, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"RESPUESTA: el modelo {'SÍ' if cumple_85 else 'NO'} supera 85% en Weighted Avg. Precision de WEKA para 2024.")
print(f"Precision WEKA ponderada: {resumen_2024['weighted_precision']:.2%}")
print(f"Accuracy: {resumen_2024['accuracy']:.2%}")
print(f"MCC: {resumen_2024['mcc']:.3f}")
print("Estado:", resumen_modelo["status"])
print("Advertencia: es adecuado para una presentación funcional; no está aprobado para decisiones automáticas de producción.")

artefactos = sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file())
display(pd.DataFrame({"artefacto": artefactos}))
"""
        ),
    ]
    return notebook


def main() -> None:
    if NOTEBOOK_PATH.exists() and not RIGOROUS_BACKUP_PATH.exists():
        shutil.copy2(NOTEBOOK_PATH, RIGOROUS_BACKUP_PATH)

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
    print(f"Respaldo riguroso: {RIGOROUS_BACKUP_PATH}")
    print(f"Reporte: {ROOT / 'reports' / 'modelo_presentacion_cafe'}")


if __name__ == "__main__":
    main()
