from __future__ import annotations

import pandas as pd

from src.evaluation.annual_classifier import MODEL_FEATURES
from src.evaluation.annual_pipeline import run_annual_analysis


def _small_temporal_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in range(2012, 2026):
        for index in range(12):
            high = int(index >= 6)
            row: dict[str, object] = {
                "codigo_dane_municipio": f"05{index + 1:03d}",
                "codigo_dane_departamento": "05",
                "departamento": "ANTIOQUIA",
                "municipio": f"M{index}",
                "anio": year,
                "rendimiento_t_ha": 0.7 + 0.7 * high,
                "rendimiento_alto": high,
                "clase_lag_1": high,
                "cambio_clase_vs_lag_1": 0,
                "fuente_eva": "historica" if year <= 2018 else "reciente",
                "feature_cutoff_year": year - 1,
                "max_eva_feature_year": year - 1,
                "max_climate_feature_year": year - 1,
            }
            for feature in MODEL_FEATURES:
                if feature not in row:
                    row[feature] = high + (year - 2012) / 100
            rows.append(row)
    return pd.DataFrame(rows)


def test_run_analysis_keeps_2025_out_of_development_and_returns_seven_years():
    result = run_annual_analysis(
        _small_temporal_panel(),
        model_names=["logistic_plain"],
        bootstrap_samples=20,
    )

    assert set(result["predictions"]["anio"]) == set(range(2019, 2025))
    assert set(result["winner_predictions"]["anio"]) == set(range(2019, 2026))
    assert set(result["winner_metrics"]["anio"]) == set(range(2019, 2026))
    final = result["winner_predictions"].query("anio == 2025")
    assert set(final["train_max_year"]) == {2024}
    assert result["gate"]["status"] == "target_not_met"
    assert {"mcc_ci_low", "mcc_ci_high", "kappa_ci_low", "kappa_ci_high"}.issubset(
        result["winner_metrics"].columns
    )


def test_smote_diagnostic_is_balanced_only_after_resampling():
    panel = _small_temporal_panel()
    panel.loc[(panel["anio"] < 2025) & (panel["codigo_dane_municipio"].isin(["05001", "05002"])), "rendimiento_alto"] = 1
    result = run_annual_analysis(
        panel,
        model_names=["logistic_plain"],
        bootstrap_samples=10,
    )
    balance = result["smote_balance"].pivot(index="etapa", columns="clase", values="casos")

    assert balance.loc["Antes", "Alto"] != balance.loc["Antes", "Bajo"]
    assert balance.loc["Después", "Alto"] == balance.loc["Después", "Bajo"]
