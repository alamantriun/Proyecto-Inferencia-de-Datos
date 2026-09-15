"""End-to-end annual analysis with a sealed 2025 holdout."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, make_scorer, matthews_corrcoef

from src.evaluation.annual_classifier import (
    CLASS_TARGET,
    MODEL_FEATURES,
    ModelRecipe,
    annual_metrics,
    build_estimator,
    evaluate_final_holdout,
    strict_production_gate,
    walk_forward_backtest,
)
from src.features.cafe_annual_features import (
    TARGET,
    audit_point_in_time,
    fixed_high_yield_threshold,
)


PRESENTATION_MODELS = (
    "logistic_plain",
    "logistic_smotenc",
    "hist_plain",
    "random_forest",
    "catboost_plain",
)


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame[TARGET].notna() & frame["rendimiento_lag_1"].notna()
    ].copy()


def bootstrap_annual_intervals(
    predictions: pd.DataFrame,
    samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return deterministic percentile intervals for annual MCC and Kappa."""
    if samples < 1:
        raise ValueError("bootstrap samples debe ser positivo")
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int | str]] = []
    for (model, year), group in predictions.groupby(["model", "anio"], sort=True):
        truth = group["y_true"].to_numpy(int)
        estimate = group["y_pred"].to_numpy(int)
        mcc_values = np.empty(samples, dtype=float)
        kappa_values = np.empty(samples, dtype=float)
        for index in range(samples):
            sample = rng.integers(0, len(group), size=len(group))
            mcc_values[index] = matthews_corrcoef(truth[sample], estimate[sample])
            kappa = cohen_kappa_score(truth[sample], estimate[sample])
            kappa_values[index] = 0.0 if not np.isfinite(kappa) else kappa
        rows.append(
            {
                "model": str(model),
                "anio": int(year),
                "mcc_ci_low": float(np.quantile(mcc_values, 0.025)),
                "mcc_ci_high": float(np.quantile(mcc_values, 0.975)),
                "kappa_ci_low": float(np.quantile(kappa_values, 0.025)),
                "kappa_ci_high": float(np.quantile(kappa_values, 0.975)),
                "bootstrap_samples": int(samples),
            }
        )
    return pd.DataFrame(rows)


def smotenc_balance_diagnostic(
    frame: pd.DataFrame, cutoff_year: int = 2025
) -> pd.DataFrame:
    """Measure class counts before/after train-only SMOTENC for one fold."""
    train = _eligible(frame)
    train = train[train["anio"] < int(cutoff_year)]
    target = train[CLASS_TARGET].astype(int)
    if set(target.unique()) != {0, 1}:
        raise ValueError("El fold para demostrar SMOTENC requiere ambas clases")
    pipeline = build_estimator("logistic_smotenc")
    prepared = pipeline.named_steps["preprocess"].fit_transform(train[MODEL_FEATURES])
    _, resampled = pipeline.named_steps["sampler"].fit_resample(prepared, target)
    before = target.value_counts().reindex([0, 1], fill_value=0)
    after = pd.Series(resampled).value_counts().reindex([0, 1], fill_value=0)
    rows = []
    for stage, counts in (("Antes", before), ("Después", after)):
        for value, label in ((0, "Bajo"), (1, "Alto")):
            rows.append(
                {
                    "etapa": stage,
                    "clase": label,
                    "casos": int(counts.loc[value]),
                    "fold_entrenamiento_hasta": int(cutoff_year) - 1,
                }
            )
    return pd.DataFrame(rows)


def _feature_importance(
    frame: pd.DataFrame,
    recipe: ModelRecipe,
    estimator: Any,
    final_year: int = 2025,
) -> pd.DataFrame:
    if recipe.name == "lag1":
        return pd.DataFrame({"feature": ["clase_lag_1"], "importance": [1.0], "importance_std": [0.0]})
    if recipe.name == "majority":
        return pd.DataFrame({"feature": ["intercepto_mayoritario"], "importance": [0.0], "importance_std": [0.0]})
    holdout = _eligible(frame)
    holdout = holdout[holdout["anio"] == int(final_year)]
    scorer = make_scorer(matthews_corrcoef)
    measured = permutation_importance(
        estimator,
        holdout[MODEL_FEATURES],
        holdout[CLASS_TARGET].astype(int),
        scoring=scorer,
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": measured.importances_mean,
            "importance_std": measured.importances_std,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


def load_source_manifests(cache_dir: Path) -> dict[str, Any]:
    """Load the frozen official-source manifests used by the mart."""
    cache_dir = Path(cache_dir)
    combined: dict[str, Any] = {}
    for path in (cache_dir / "source_manifest.json", cache_dir / "nasa_power_manifest.json"):
        if path.exists():
            combined[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return combined


def run_annual_analysis(
    mart: pd.DataFrame,
    model_names: Sequence[str] = PRESENTATION_MODELS,
    development_years: Sequence[int] = tuple(range(2019, 2025)),
    final_year: int = 2025,
    threshold_history_years: Sequence[int] = (2016, 2017, 2018),
    bootstrap_samples: int = 1000,
    source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select on 2019-2024, then evaluate the frozen winner once on 2025."""
    audit = audit_point_in_time(mart)
    development = walk_forward_backtest(
        mart,
        model_names=model_names,
        years=development_years,
        threshold_history_years=threshold_history_years,
    )
    recipe: ModelRecipe = development["recipe"]
    final_prediction, estimator = evaluate_final_holdout(mart, recipe, year=final_year)

    development_predictions = development["predictions"].copy()
    winner_development = development_predictions[
        development_predictions["model"] == development["winner"]
    ].copy()
    winner_predictions = pd.concat(
        [winner_development, final_prediction], ignore_index=True
    )
    winner_metrics, winner_weka = annual_metrics(winner_predictions)
    intervals = bootstrap_annual_intervals(
        winner_predictions, samples=bootstrap_samples, random_state=42
    )
    winner_metrics = winner_metrics.merge(
        intervals, on=["model", "anio"], how="left", validate="one_to_one"
    )

    final_metrics, final_weka = annual_metrics(final_prediction)
    all_metrics = pd.concat(
        [development["annual_metrics"], final_metrics], ignore_index=True
    )
    all_weka = pd.concat(
        [development["weka_details"], final_weka], ignore_index=True
    )

    baseline_final_prediction, _ = evaluate_final_holdout(
        mart, ModelRecipe("lag1", 0.5), year=final_year
    )
    baseline_final_metrics, _ = annual_metrics(baseline_final_prediction)
    baseline_metrics = pd.concat(
        [
            development["annual_metrics"].query("model == 'lag1'"),
            baseline_final_metrics,
        ],
        ignore_index=True,
    )
    gate = strict_production_gate(
        winner_metrics,
        baseline_metrics,
        threshold=0.85,
        required_years=[*development_years, final_year],
    )

    result: dict[str, Any] = {
        **development,
        "mart": mart.copy(),
        "audit": audit,
        "predictions": development_predictions,
        "winner_predictions": winner_predictions,
        "annual_metrics": all_metrics,
        "winner_metrics": winner_metrics,
        "weka_details": all_weka,
        "winner_weka_details": winner_weka,
        "baseline_metrics": baseline_metrics,
        "final_prediction": final_prediction,
        "estimator": estimator,
        "gate": gate,
        "smote_balance": smotenc_balance_diagnostic(mart, cutoff_year=final_year),
        "feature_importance": _feature_importance(
            mart, recipe, estimator, final_year=final_year
        ),
        "source_manifest": dict(source_manifest or {}),
        "target_threshold": fixed_high_yield_threshold(mart),
    }
    return result


def _read_mart(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "codigo_dane_municipio": str,
            "codigo_dane_departamento": str,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mart", type=Path, default=Path("data/processed/cafe_annual_model_mart.csv"))
    parser.add_argument("--source-cache", type=Path, default=Path("data/external/cafe_annual"))
    parser.add_argument("--output", type=Path, default=Path("reports/modelo_mcc_kappa_cafe"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--result-cache",
        type=Path,
        default=Path("data/processed/cafe_annual_analysis_result.joblib"),
    )
    parser.add_argument("--reuse-result-cache", action="store_true")
    args = parser.parse_args()

    from src.evaluation.annual_artifacts import export_annual_artifacts

    if args.reuse_result_cache and args.result_cache.exists():
        result = joblib.load(args.result_cache)
    else:
        result = run_annual_analysis(
            _read_mart(args.mart),
            bootstrap_samples=args.bootstrap_samples,
            source_manifest=load_source_manifests(args.source_cache),
        )
        args.result_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = args.result_cache.with_name(
            f".{args.result_cache.name}.tmp-{uuid.uuid4().hex[:8]}"
        )
        joblib.dump(result, temporary_cache, compress=3)
        temporary_cache.replace(args.result_cache)
    paths = export_annual_artifacts(result, args.output)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
