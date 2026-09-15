"""Figures and auditable publication bundle for the annual coffee classifier."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import cohen_kappa_score, confusion_matrix, matthews_corrcoef


INK = "#25313C"
BLUE = "#2F6690"
BLUE_LIGHT = "#B8D4E8"
GOLD = "#D6A84B"
ORANGE = "#D97A3A"
OLIVE = "#6F7D3C"
PINK = "#B95C7A"
GRID = "#DDE3E8"
TARGET_LINE = "#4B5563"
FIGURE_NAMES = (
    "01_cobertura_fuentes_anual.png",
    "02_distribucion_rendimiento_inicial.png",
    "03_boxplot_rendimiento_anual.png",
    "04_mapa_nulos_cobertura.png",
    "05_balance_clases_anual.png",
    "06_balance_smotenc.png",
    "07_metricas_mcc_kappa_anuales.png",
    "08_matrices_confusion_2019_2025.png",
    "09_matriz_confusion_consolidada.png",
    "10_metricas_weka_clases.png",
    "11_calibracion_probabilidades.png",
    "12_desempeno_cambios_clase.png",
    "13_importancia_variables.png",
    "14_peor_resultado_anual_modelos.png",
)


def _replace_with_retry(source: Path, target: Path, attempts: int = 8) -> None:
    """Rename a file or directory, tolerating short Windows scanner locks."""
    last_error: PermissionError | None = None
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.25 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _frame(result: Mapping[str, Any], *keys: str) -> pd.DataFrame:
    for key in keys:
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            return value.copy()
    raise ValueError(f"Falta una tabla requerida: {', '.join(keys)}")


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": GRID,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path, caption: str) -> Path:
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=8, color="#58636D")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"No se pudo crear la figura {path.name}")
    return path


def _safe_binary_metric(values: pd.DataFrame, metric: str) -> float:
    if values.empty or values["y_true"].nunique() < 2 or values["y_pred"].nunique() < 2:
        return 0.0
    if metric == "mcc":
        return float(matthews_corrcoef(values["y_true"], values["y_pred"]))
    return float(cohen_kappa_score(values["y_true"], values["y_pred"]))


def _plot_source_coverage(mart: pd.DataFrame, path: Path) -> Path:
    annual = mart.groupby("anio").agg(
        filas=("codigo_dane_municipio", "size"),
        municipios=("codigo_dane_municipio", "nunique"),
    )
    if "climate_available" in mart:
        annual["cobertura_clima_pct"] = (
            mart.groupby("anio")["climate_available"].mean() * 100
        )
    else:
        annual["cobertura_clima_pct"] = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(annual.index, annual["filas"], color=BLUE, marker="o", label="Filas")
    axes[0].plot(
        annual.index,
        annual["municipios"],
        color=GOLD,
        marker="s",
        linestyle="--",
        label="Municipios",
    )
    axes[0].set(title="Cobertura anual de EVA", xlabel="Año", ylabel="Cantidad")
    axes[0].legend(frameon=False)
    axes[1].plot(
        annual.index,
        annual["cobertura_clima_pct"],
        color=OLIVE,
        marker="o",
    )
    axes[1].set(
        title="Cobertura de clima rezagado", xlabel="Año objetivo", ylabel="Cobertura (%)", ylim=(0, 105)
    )
    fig.suptitle("Fuentes disponibles por año", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return _save(fig, path, "Unidad: municipio-año. El clima del año t corresponde exclusivamente a t-1.")


def _plot_initial_distribution(mart: pd.DataFrame, path: Path) -> Path:
    historical = mart.loc[mart["anio"] <= 2018, "rendimiento_t_ha"].dropna()
    recent = mart.loc[mart["anio"] >= 2019, "rendimiento_t_ha"].dropna()
    upper = float(mart["rendimiento_t_ha"].quantile(0.99))
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, max(upper, 0.1), 35)
    ax.hist(historical.clip(upper=upper), bins=bins, density=True, alpha=0.55, color=BLUE, label=f"2007-2018 (n={len(historical):,})")
    ax.hist(recent.clip(upper=upper), bins=bins, density=True, alpha=0.45, color=GOLD, label=f"2019-2025 (n={len(recent):,})")
    ax.set(title="Distribución inicial del rendimiento", xlabel="Rendimiento (t/ha; recorte visual p99)", ylabel="Densidad")
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, path, "El recorte p99 es solamente visual; ninguna fila se elimina del entrenamiento por esta gráfica.")


def _plot_annual_boxplot(mart: pd.DataFrame, path: Path) -> Path:
    plot = mart[["anio", "rendimiento_t_ha"]].dropna().copy()
    upper = float(plot["rendimiento_t_ha"].quantile(0.99))
    plot["rendimiento_visual"] = plot["rendimiento_t_ha"].clip(upper=upper)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    sns.boxplot(data=plot, x="anio", y="rendimiento_visual", color=BLUE_LIGHT, fliersize=0, ax=ax)
    ax.axvline(11.5, color=ORANGE, linestyle="--", linewidth=1.5)
    ax.text(11.6, upper * 0.95, "Cambio de fuente 2018/2019", color=ORANGE, fontsize=9)
    ax.set(title="Rendimiento de café por año", xlabel="Año", ylabel="Rendimiento (t/ha; recorte visual p99)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, path, "Las cajas muestran mediana e intervalo intercuartílico. La línea marca el cambio de fuente EVA.")


def _plot_missingness(mart: pd.DataFrame, path: Path) -> Path:
    candidates = [
        "rendimiento_lag_1",
        "rendimiento_lag_2",
        "rendimiento_media_3y",
        "precipitacion_total_mm_lag_1",
        "temperatura_media_c_lag_1",
        "latitud",
        "longitud",
    ]
    columns = [column for column in candidates if column in mart]
    if not columns:
        columns = [column for column in mart.select_dtypes(include="number").columns if column != "anio"][:7]
    missing = mart.groupby("anio")[columns].apply(lambda data: data.isna().mean() * 100).T
    fig, ax = plt.subplots(figsize=(14, max(4, 0.55 * len(columns))))
    sns.heatmap(missing, cmap=sns.light_palette(BLUE, as_cmap=True), vmin=0, vmax=100, linewidths=0.3, cbar_kws={"label": "Nulos (%)"}, ax=ax)
    ax.set(title="Nulos de variables disponibles antes del año objetivo", xlabel="Año objetivo", ylabel="Variable")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, path, "Los rezagos ausentes reflejan municipios sin historia suficiente; no se completan con datos futuros.")


def _plot_class_balance(mart: pd.DataFrame, path: Path) -> Path:
    balance = mart.groupby(["anio", "rendimiento_alto"]).size().unstack(fill_value=0)
    balance = balance.rename(columns={0: "Bajo", 1: "Alto"})
    for column in ("Bajo", "Alto"):
        if column not in balance:
            balance[column] = 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    balance[["Bajo", "Alto"]].plot(kind="bar", stacked=True, color=[BLUE_LIGHT, BLUE], ax=axes[0])
    proportions = balance.div(balance.sum(axis=1), axis=0) * 100
    proportions[["Bajo", "Alto"]].plot(kind="bar", stacked=True, color=[BLUE_LIGHT, BLUE], ax=axes[1])
    axes[0].set(title="Conteo por clase", xlabel="Año", ylabel="Casos")
    axes[1].set(title="Proporción por clase", xlabel="Año", ylabel="Porcentaje", ylim=(0, 100))
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.legend(frameon=False)
    fig.suptitle("Balance inicial de la variable objetivo", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return _save(fig, path, "Clase alta: rendimiento por encima del umbral fijo aprendido exclusivamente con 2007-2018.")


def _plot_smote_balance(balance: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=balance, x="etapa", y="casos", hue="clase", palette={"Bajo": BLUE_LIGHT, "Alto": BLUE}, ax=ax)
    ax.set(title="Clases antes y después de SMOTENC", xlabel="Fold de entrenamiento", ylabel="Casos")
    ax.legend(title="Clase", frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path, "SMOTENC se aplica sólo al entrenamiento; validación y holdout conservan su distribución real.")


def _plot_model_metrics(metrics: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    models = list(metrics["model"].drop_duplicates())
    palette = sns.color_palette("colorblind", n_colors=max(1, len(models)))
    color_map = dict(zip(models, palette))
    for ax, metric, label in zip(axes, ("mcc", "kappa"), ("MCC", "Kappa de Cohen")):
        for model, group in metrics.groupby("model"):
            ordered = group.sort_values("anio")
            ax.plot(ordered["anio"], ordered[metric], marker="o", linewidth=1.5, color=color_map[model], label=model)
        ax.axhline(0.85, color=TARGET_LINE, linestyle="--", linewidth=1.4, label="Meta 0,85")
        ax.set(title=label, xlabel="Año", ylabel="Puntaje", ylim=(-0.05, 1.02))
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=min(5, len(by_label)), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("MCC y Kappa fuera de muestra por modelo y año", fontsize=15, fontweight="bold", y=1.10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return _save(fig, path, "La meta se evalúa con valores no redondeados. Un promedio alto no compensa un año bajo.")


def _draw_confusion(ax: plt.Axes, matrix: np.ndarray, title: str) -> None:
    sns.heatmap(matrix, annot=True, fmt="d", cmap=sns.light_palette(BLUE, as_cmap=True), cbar=False, square=True, linewidths=0.8, linecolor="white", ax=ax)
    ax.set(title=title, xlabel="Predicción", ylabel="Real")
    ax.set_xticklabels(["Bajo", "Alto"])
    ax.set_yticklabels(["Bajo", "Alto"], rotation=0)


def _plot_annual_confusions(predictions: pd.DataFrame, path: Path) -> Path:
    years = list(range(2019, 2026))
    fig, axes = plt.subplots(2, 4, figsize=(15, 9.5))
    for ax, year in zip(axes.flat, years):
        group = predictions[predictions["anio"] == year]
        matrix = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1])
        _draw_confusion(ax, matrix, f"{year} (n={len(group):,})")
    axes.flat[-1].axis("off")
    fig.suptitle("Matrices de confusión anuales del modelo seleccionado", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.09, hspace=0.55, wspace=0.35)
    return _save(fig, path, "Formato de cada matriz: [[TN, FP], [FN, TP]]. Todos los casos son predicciones fuera de muestra.")


def _plot_consolidated_confusion(predictions: pd.DataFrame, path: Path) -> Path:
    matrix = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _draw_confusion(ax, matrix, f"2019-2025 (n={len(predictions):,})")
    fig.suptitle("Matriz de confusión consolidada", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return _save(fig, path, "La matriz consolidada sirve como resumen; la compuerta de aceptación se decide año por año.")


def _plot_weka(weka: pd.DataFrame, path: Path) -> Path:
    display = weka[weka["Class"] != "Weighted Avg."].copy()
    display["grupo"] = display["anio"].astype(str) + " · " + display["Class"].astype(str)
    long = display.melt(id_vars=["grupo"], value_vars=["Precision", "Recall", "F-Measure"], var_name="Métrica", value_name="Valor")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=long, x="grupo", y="Valor", hue="Métrica", palette=[BLUE, GOLD, OLIVE], ax=ax)
    ax.axhline(0.85, color=TARGET_LINE, linestyle="--", linewidth=1.2)
    ax.set(title="Métricas estilo WEKA por clase y año", xlabel="Año y clase real", ylabel="Puntaje", ylim=(0, 1.02))
    ax.tick_params(axis="x", rotation=55)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path, "Precision, Recall y F-Measure se calculan por clase; el soporte permanece en la tabla CSV detallada.")


def _plot_calibration(predictions: pd.DataFrame, path: Path) -> Path:
    actual = predictions["y_true"].to_numpy(int)
    probability = predictions["probability"].to_numpy(float)
    fraction, mean_probability = calibration_curve(actual, probability, n_bins=10, strategy="quantile")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot([0, 1], [0, 1], color=TARGET_LINE, linestyle="--", label="Calibración ideal")
    axes[0].plot(mean_probability, fraction, marker="o", color=BLUE, label="Modelo")
    axes[0].set(title="Curva de calibración", xlabel="Probabilidad media predicha", ylabel="Proporción observada", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(frameon=False)
    axes[1].hist(probability[actual == 0], bins=20, alpha=0.55, color=BLUE_LIGHT, label="Real bajo")
    axes[1].hist(probability[actual == 1], bins=20, alpha=0.55, color=BLUE, label="Real alto")
    axes[1].set(title="Distribución de probabilidades", xlabel="Probabilidad de rendimiento alto", ylabel="Casos", xlim=(0, 1))
    axes[1].legend(frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path, "La calibración agrupa cuantiles de todas las predicciones fuera de muestra 2019-2025.")


def _plot_class_changes(predictions: pd.DataFrame, path: Path) -> Path:
    change_column = "cambio_clase_vs_lag_1"
    if change_column not in predictions:
        predictions[change_column] = 0
    groups = []
    for label, mask in (
        ("Todos", pd.Series(True, index=predictions.index)),
        ("Clase estable", predictions[change_column].fillna(0).astype(int) == 0),
        ("Cambio de clase", predictions[change_column].fillna(0).astype(int) == 1),
    ):
        data = predictions.loc[mask]
        groups.append({"grupo": label, "MCC": _safe_binary_metric(data, "mcc"), "Kappa": _safe_binary_metric(data, "kappa"), "casos": len(data)})
    summary = pd.DataFrame(groups)
    long = summary.melt(id_vars=["grupo", "casos"], value_vars=["MCC", "Kappa"], var_name="Métrica", value_name="Valor")
    long["etiqueta"] = long["grupo"] + "\n(n=" + long["casos"].astype(str) + ")"
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=long, x="etiqueta", y="Valor", hue="Métrica", palette=[BLUE, GOLD], ax=ax)
    ax.axhline(0.85, color=TARGET_LINE, linestyle="--", linewidth=1.2)
    ax.set(title="Desempeño cuando la clase cambia frente al año anterior", xlabel="Segmento", ylabel="Puntaje", ylim=(-0.05, 1.02))
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path, "Los cambios de clase son el caso difícil para un baseline de persistencia y revelan memorización temporal.")


def _plot_feature_importance(importance: pd.DataFrame, path: Path) -> Path:
    plot = importance.sort_values("importance", ascending=False).head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(plot["feature"], plot["importance"], color=BLUE, edgecolor=INK, linewidth=0.4)
    ax.axvline(0, color=TARGET_LINE, linewidth=1)
    ax.set(title="Importancia global por permutación", xlabel="Disminución media de MCC al permutar", ylabel="Variable")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path, "Importancia calculada sobre el holdout 2025; describe asociación predictiva, no causalidad.")


def _plot_worst_model(model_summary: pd.DataFrame, path: Path) -> Path:
    plot = model_summary.sort_values("worst_joint_score", ascending=True)
    colors = [BLUE if value >= 0.85 else BLUE_LIGHT for value in plot["worst_joint_score"]]
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * len(plot))))
    ax.barh(plot["model"], plot["worst_joint_score"], color=colors, edgecolor=INK, linewidth=0.4)
    ax.axvline(0.85, color=TARGET_LINE, linestyle="--", linewidth=1.4, label="Meta 0,85")
    ax.set(title="Peor min(MCC, Kappa) anual por modelo", xlabel="Peor puntaje anual", ylabel="Modelo", xlim=(min(-0.05, float(plot["worst_joint_score"].min()) - 0.03), 1.0))
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path, "La selección prioriza el peor año para impedir que promedios favorables oculten fallas anuales.")


def create_annual_figures(result: Mapping[str, Any], output_dir: Path) -> list[Path]:
    """Create the complete, labeled static figure inventory."""
    _style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mart = _frame(result, "mart")
    predictions = _frame(result, "winner_predictions", "predictions")
    metrics = _frame(result, "annual_metrics", "winner_metrics")
    weka = _frame(result, "winner_weka_details", "weka_details")
    smote = _frame(result, "smote_balance")
    importance = _frame(result, "feature_importance")
    model_summary = _frame(result, "model_summary")

    paths = [
        _plot_source_coverage(mart, output_dir / FIGURE_NAMES[0]),
        _plot_initial_distribution(mart, output_dir / FIGURE_NAMES[1]),
        _plot_annual_boxplot(mart, output_dir / FIGURE_NAMES[2]),
        _plot_missingness(mart, output_dir / FIGURE_NAMES[3]),
        _plot_class_balance(mart, output_dir / FIGURE_NAMES[4]),
        _plot_smote_balance(smote, output_dir / FIGURE_NAMES[5]),
        _plot_model_metrics(metrics, output_dir / FIGURE_NAMES[6]),
        _plot_annual_confusions(predictions, output_dir / FIGURE_NAMES[7]),
        _plot_consolidated_confusion(predictions, output_dir / FIGURE_NAMES[8]),
        _plot_weka(weka, output_dir / FIGURE_NAMES[9]),
        _plot_calibration(predictions, output_dir / FIGURE_NAMES[10]),
        _plot_class_changes(predictions, output_dir / FIGURE_NAMES[11]),
        _plot_feature_importance(importance, output_dir / FIGURE_NAMES[12]),
        _plot_worst_model(model_summary, output_dir / FIGURE_NAMES[13]),
    ]
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _confusion_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for year, group in predictions.groupby("anio", sort=True):
        tn, fp, fn, tp = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1]).ravel()
        rows.append({"anio": int(year), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "instances": int(len(group))})
    return pd.DataFrame(rows)


def _model_card(result: Mapping[str, Any]) -> str:
    gate = dict(result["gate"])
    recipe = _jsonable(result["recipe"])
    metrics = _frame(result, "winner_metrics")
    failed = ", ".join(map(str, gate.get("failed_years", []))) or "ninguno"
    status_text = (
        "NO AUTORIZADO PARA PRODUCCIÓN: la meta estadística no se alcanzó."
        if gate.get("status") != "production_candidate"
        else "CANDIDATO ESTADÍSTICO: requiere además validación operativa y monitoreo."
    )
    rows = metrics[["anio", "weighted_precision", "mcc", "kappa"]].sort_values("anio")
    table_lines = [
        "| Año | Precisión ponderada | MCC | Kappa |",
        "|---:|---:|---:|---:|",
    ]
    table_lines.extend(
        "| {anio:d} | {precision:.4f} | {mcc:.4f} | {kappa:.4f} |".format(
            anio=int(row.anio),
            precision=float(row.weighted_precision),
            mcc=float(row.mcc),
            kappa=float(row.kappa),
        )
        for row in rows.itertuples(index=False)
    )
    table = "\n".join(table_lines)
    failures = "\n".join(f"- {item}" for item in gate.get("failed_checks", [])) or "- Ninguna."
    return f"""# Tarjeta del modelo anual de rendimiento de café

## Estado

**{status_text}**

- Modelo: `{recipe.get('name', result.get('winner'))}`
- Umbral de decisión: `{float(recipe.get('decision_threshold', 0.5)):.4f}`
- Meta: MCC >= 0,85 y Kappa >= 0,85 en cada año 2019-2025.
- Años que incumplen: {failed}.

## Uso previsto

Clasificar, antes de comenzar un año, si el rendimiento municipal de café superará un umbral histórico fijo. Es una herramienta analítica y explicable; no predice rentabilidad, precio, calidad del grano ni resultados de una finca individual.

## Validación cronológica

{table}

Cada año se predice con información disponible hasta el 31 de diciembre del año anterior. El periodo 2019-2024 se usó para backtesting walk-forward y 2025 como holdout final con receta congelada.

## Fallas de la compuerta

{failures}

## Limitaciones obligatorias

- El cambio entre las fuentes EVA histórica y reciente en 2019 produce deriva fuerte.
- Una precisión ponderada alta no sustituye MCC o Kappa; puede ocultar una clase minoritaria mal predicha.
- SMOTENC sólo balancea folds de entrenamiento y no crea información nueva sobre cambios estructurales.
- El clima NASA POWER se asigna por la grilla más cercana al centroide municipal; no representa necesariamente cada zona cafetera montañosa.
- Las asociaciones y las importancias de variables no deben interpretarse causalmente.

## Regla de despliegue

No desplegar para decisiones automáticas mientras el estado sea `target_not_met`. Para uso exploratorio, mostrar probabilidad, corte de datos y advertencia, y registrar resultados reales para una reevaluación futura.
"""


def export_annual_artifacts(result: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    """Publish a complete bundle through a validated atomic directory swap."""
    output_dir = Path(output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex[:10]}"
    backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex[:10]}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    replaced = False
    try:
        mart = _frame(result, "mart")
        predictions = _frame(result, "predictions")
        winner_predictions = _frame(result, "winner_predictions")
        annual = _frame(result, "annual_metrics")
        winner_metrics = _frame(result, "winner_metrics")
        weka = _frame(result, "weka_details")
        winner_weka = _frame(result, "winner_weka_details")
        model_summary = _frame(result, "model_summary")
        importance = _frame(result, "feature_importance")
        smote = _frame(result, "smote_balance")
        estimator = result.get("estimator")
        if estimator is None:
            raise ValueError("Falta el estimador ajustado para serializar")

        tables = {
            "dataset_modelo.csv": mart,
            "predicciones_todos_modelos_2019_2024.csv": predictions,
            "predicciones_modelo_seleccionado_2019_2025.csv": winner_predictions,
            "metricas_anuales_todos_modelos.csv": annual,
            "metricas_anuales_modelo_seleccionado.csv": winner_metrics,
            "metricas_weka_todos_modelos.csv": weka,
            "metricas_weka_modelo_seleccionado.csv": winner_weka,
            "resumen_modelos.csv": model_summary,
            "matrices_confusion_2019_2025.csv": _confusion_table(winner_predictions),
            "importancia_variables.csv": importance,
            "balance_smotenc.csv": smote,
            "diccionario_variables.csv": pd.DataFrame(
                {"variable": mart.columns, "tipo": [str(dtype) for dtype in mart.dtypes]}
            ),
        }
        for name, table in tables.items():
            table.to_csv(staging / name, index=False, encoding="utf-8-sig")

        source_manifest = _jsonable(result.get("source_manifest", {}))
        (staging / "source_manifest.json").write_text(
            json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        frozen_recipe = {
            "selected_without_2025": True,
            "development_years": list(range(2019, 2025)),
            "winner": str(result.get("winner", "")),
            "recipe": _jsonable(result["recipe"]),
            "selection_rule": "maximizar el peor min(MCC, Kappa) anual; luego promedio, precisión ponderada y simplicidad",
        }
        (staging / "recipe_frozen_before_2025.json").write_text(
            json.dumps(frozen_recipe, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        joblib.dump(estimator, staging / "modelo_evaluado_2025.joblib")
        joblib.dump(dict(result), staging / "analysis_bundle.joblib", compress=3)
        (staging / "MODEL_CARD.md").write_text(_model_card(result), encoding="utf-8")
        figure_paths = create_annual_figures(result, staging)

        required = [
            staging / "dataset_modelo.csv",
            staging / "predicciones_modelo_seleccionado_2019_2025.csv",
            staging / "metricas_anuales_modelo_seleccionado.csv",
            staging / "matrices_confusion_2019_2025.csv",
            staging / "modelo_evaluado_2025.joblib",
            staging / "analysis_bundle.joblib",
            staging / "recipe_frozen_before_2025.json",
            staging / "MODEL_CARD.md",
            *figure_paths,
        ]
        missing = [path.name for path in required if not path.exists() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"Publicación incompleta: {missing}")

        gate = _jsonable(result["gate"])
        manifest = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": gate["status"],
            "winner": str(result.get("winner", "")),
            "recipe": _jsonable(result["recipe"]),
            "target_threshold_t_ha": float(result.get("target_threshold", np.nan)),
            "development_years": list(range(2019, 2025)),
            "final_holdout_year": 2025,
            "gate": gate,
            "artifact_hashes_sha256": {
                path.name: _sha256(path)
                for path in sorted(staging.iterdir())
                if path.is_file() and path.name != "model_manifest.json"
            },
        }
        (staging / "model_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if output_dir.exists():
            _replace_with_retry(output_dir, backup)
            replaced = True
        _replace_with_retry(staging, output_dir)
        if backup.exists():
            shutil.rmtree(backup)
        replaced = False
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if replaced and backup.exists() and not output_dir.exists():
            _replace_with_retry(backup, output_dir)
        raise

    return {
        "manifest": output_dir / "model_manifest.json",
        "model_card": output_dir / "MODEL_CARD.md",
        "model": output_dir / "modelo_evaluado_2025.joblib",
        "analysis_bundle": output_dir / "analysis_bundle.joblib",
        "winner_predictions": output_dir / "predicciones_modelo_seleccionado_2019_2025.csv",
        "winner_metrics": output_dir / "metricas_anuales_modelo_seleccionado.csv",
        "confusions": output_dir / "matrices_confusion_2019_2025.csv",
        "figures": output_dir,
    }
