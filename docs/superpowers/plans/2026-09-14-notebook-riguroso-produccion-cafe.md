# Rigorous Coffee Production Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar `pipeline_interactivo_cafe.ipynb` por un notebook reproducible que audite los datos, compare modelos con validación temporal, aplique SMOTENC correctamente, evalúe una sola vez en 2024 y exporte artefactos con una decisión objetiva de preparación para producción.

**Architecture:** La lógica estadística vive en `src/evaluation/production_pipeline.py` y el notebook funciona como una narración ejecutable de esa lógica. Un generador crea las 17 secciones, respalda el notebook anterior, ejecuta el reemplazo y conserva resultados visibles. Las pruebas validan el aislamiento temporal, SMOTENC, selección, compuerta y recomputación independiente de métricas.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn, imbalanced-learn, CatBoost, matplotlib, seaborn, joblib, nbformat, nbclient, unittest.

**Spec:** `docs/superpowers/specs/2026-09-14-notebook-riguroso-produccion-cafe-design.md`

## Global Constraints

- El target continuo es `rendimiento_t_ha`; la clase alta usa la mediana de 2007-2018 y no significa rentabilidad.
- Los años 2019-2023 son backtest rolling-origin y 2024 es holdout final.
- `score_confiabilidad`, `dato_copiado` y variables externas con procedencia temporal/geográfica insuficiente quedan excluidas.
- SMOTENC se ajusta solo con filas de entrenamiento y nunca modifica validación o holdout.
- MCC es la métrica primaria de clasificación; MAE es la métrica primaria de regresión.
- La selección de modelos, calibración y umbral se congelan antes de evaluar 2024.
- El notebook debe ejecutar de principio a fin sin intervención y guardar todas las gráficas y artefactos especificados.
- Los commits deben limitarse a los archivos de cada tarea; si Git sigue sin identidad configurada, no se alterará `user.name` ni `user.email` y se continuará con verificación local.

---

## File map

- Create `src/evaluation/production_pipeline.py`: contrato de datos, modelos, backtests, calibración, métricas, gráficas, artefactos y compuerta.
- Create `tests/test_production_pipeline.py`: pruebas unitarias y de aislamiento temporal.
- Create `tests/test_production_notebook.py`: validación estructural, outputs y recomputación de métricas.
- Create `scripts/create_production_notebook.py`: respaldo, generación y ejecución del notebook.
- Create `pipeline_interactivo_cafe_original.ipynb`: copia recuperable del notebook previo; solo se crea si no existe.
- Replace `pipeline_interactivo_cafe.ipynb`: notebook final ejecutado con 17 secciones visibles.
- Modify `requirements.txt`: dependencias explícitas para balanceo, serialización y ejecución del notebook.
- Create `reports/modelo_produccion_cafe/`: manifiesto, model card, pipelines, predicciones, tablas y figuras.

### Task 1: Data contract and target isolation

**Files:**
- Create: `src/evaluation/production_pipeline.py`
- Create: `tests/test_production_pipeline.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `reports/tablas_entrenamiento/dataset_cafe_ml_ready.csv`.
- Produces: `load_dataset(path: Path) -> pd.DataFrame`, `audit_dataset(df: pd.DataFrame) -> dict[str, object]`, `add_high_yield_target(df: pd.DataFrame, cutoff_year: int = 2018) -> tuple[pd.DataFrame, float]`, `temporal_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]`.

- [ ] **Step 1: Declare runtime dependencies**

Append these exact requirements:

```text
imbalanced-learn>=0.12
joblib>=1.3
nbformat>=5.9
nbclient>=0.10
ipykernel>=6.29
```

- [ ] **Step 2: Write failing contract tests**

```python
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.production_pipeline import (
    add_high_yield_target,
    audit_dataset,
    temporal_splits,
)


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "departamento": ["A", "A", "A", "A"],
                "municipio": ["M", "M", "M", "M"],
                "cultivo": ["Café", "Café", "Café", "Café"],
                "anio": [2017, 2018, 2019, 2024],
                "rendimiento_t_ha": [1.0, 3.0, 2.5, 4.0],
                "rendimiento_lag_1": [np.nan, 1.0, 3.0, 2.5],
                "rendimiento_lag_2": [np.nan, np.nan, 1.0, 3.0],
                "rendimiento_lag_3": [np.nan, np.nan, np.nan, 1.0],
                "produccion_lag_1": [np.nan, 10.0, 11.0, 12.0],
                "area_cosechada_lag_1": [np.nan, 4.0, 4.0, 4.0],
                "area_sembrada_lag_1": [np.nan, 5.0, 5.0, 5.0],
                "media_rendimiento_3y": [np.nan, 1.0, 2.0, 2.5],
                "variabilidad_rendimiento_3y": [np.nan, 0.0, 1.0, 0.5],
                "tendencia_rendimiento_3y": [np.nan, 0.0, 2.0, 0.5],
                "lags_incompletos": [1, 1, 0, 0],
                "fuente_eva": ["old", "old", "new", "new"],
            }
        )

    def test_threshold_uses_only_pre_2019_rows(self):
        enriched, threshold = add_high_yield_target(self.df)
        self.assertEqual(threshold, 2.0)
        self.assertEqual(enriched.loc[enriched.anio == 2024, "rendimiento_alto"].item(), 1)

    def test_duplicate_analytic_keys_are_rejected(self):
        duplicate = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "llaves duplicadas"):
            audit_dataset(duplicate)

    def test_holdout_is_disjoint_from_selection(self):
        splits = temporal_splits(self.df)
        self.assertEqual(set(splits["holdout"].anio), {2024})
        self.assertTrue((splits["selection"].anio < 2024).all())
```

- [ ] **Step 3: Run tests and confirm the import fails**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_pipeline.DataContractTests -v`

Expected: FAIL because `src.evaluation.production_pipeline` does not exist.

- [ ] **Step 4: Implement constants and the data contract**

Use these exact constants and validations:

```python
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "reports" / "tablas_entrenamiento" / "dataset_cafe_ml_ready.csv"
OUTPUT_DIR = ROOT / "reports" / "modelo_produccion_cafe"
TARGET = "rendimiento_t_ha"
CLASS_TARGET = "rendimiento_alto"
KEY_COLUMNS = ["departamento", "municipio", "cultivo", "anio"]
CATEGORICAL_FEATURES = ["departamento", "municipio"]
NUMERIC_FEATURES = [
    "anio", "lags_incompletos", "rendimiento_lag_1", "rendimiento_lag_2",
    "rendimiento_lag_3", "produccion_lag_1", "area_cosechada_lag_1",
    "area_sembrada_lag_1", "media_rendimiento_3y",
    "variabilidad_rendimiento_3y", "tendencia_rendimiento_3y",
]
SELECTION_YEARS = [2019, 2020, 2021, 2022, 2023]
HOLDOUT_YEAR = 2024
RANDOM_STATE = 42


def add_high_yield_target(df: pd.DataFrame, cutoff_year: int = 2018) -> tuple[pd.DataFrame, float]:
    reference = df.loc[df["anio"] <= cutoff_year, TARGET].dropna()
    if reference.empty:
        raise ValueError("No existen observaciones hasta 2018 para calcular el umbral.")
    threshold = float(reference.median())
    result = df.copy()
    result[CLASS_TARGET] = (result[TARGET] > threshold).astype("int8")
    return result, threshold


def temporal_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    eligible = df[df[TARGET].notna() & df["rendimiento_lag_1"].notna()].copy()
    return {
        "history": eligible[eligible["anio"] <= 2018].copy(),
        "selection": eligible[eligible["anio"].isin(SELECTION_YEARS)].copy(),
        "holdout": eligible[eligible["anio"] == HOLDOUT_YEAR].copy(),
    }
```

`audit_dataset` must require all key, target, source and strict feature columns; reject duplicate keys, target nulls, absent years in `range(2007, 2025)`, and lag mismatches after joining each group's prior year. It returns row count, column count, year bounds, duplicate count, target-null count, lag comparisons, lag matches, missing percentages, source counts and the two excluded leakage fields.

- [ ] **Step 5: Run the contract tests**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_pipeline.DataContractTests -v`

Expected: 3 tests PASS.

- [ ] **Step 6: Commit the isolated task**

```powershell
git add requirements.txt src/evaluation/production_pipeline.py tests/test_production_pipeline.py
git commit -m "feat: add rigorous coffee data contract"
```

### Task 2: Leakage-safe SMOTENC classifiers

**Files:**
- Modify: `src/evaluation/production_pipeline.py`
- Modify: `tests/test_production_pipeline.py`

**Interfaces:**
- Consumes: `CATEGORICAL_FEATURES`, `NUMERIC_FEATURES`, training `pd.DataFrame`.
- Produces: `build_classifier(name: str) -> BaseEstimator`, `smotenc_class_counts(train: pd.DataFrame) -> pd.DataFrame`, `fit_predict_classifier(name: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, BaseEstimator]`.

- [ ] **Step 1: Write failing SMOTENC tests**

```python
from unittest.mock import patch

from imblearn.over_sampling import SMOTENC

from src.evaluation.production_pipeline import build_classifier, fit_predict_classifier


def make_classification_frame(n: int, last_year: int) -> pd.DataFrame:
    index = np.arange(n)
    frame = pd.DataFrame({
        "departamento": np.where(index % 2 == 0, "A", "B"),
        "municipio": np.where(index % 3 == 0, "M1", "M2"),
        "anio": np.full(n, last_year),
        "lags_incompletos": np.zeros(n),
        "rendimiento_lag_1": 0.7 + (index % 4) * 0.2,
        "rendimiento_lag_2": 0.6 + (index % 4) * 0.2,
        "rendimiento_lag_3": 0.5 + (index % 4) * 0.2,
        "produccion_lag_1": 10.0 + index,
        "area_cosechada_lag_1": 5.0 + index / 10,
        "area_sembrada_lag_1": 6.0 + index / 10,
        "media_rendimiento_3y": 0.6 + (index % 4) * 0.2,
        "variabilidad_rendimiento_3y": 0.1 + (index % 3) * 0.05,
        "tendencia_rendimiento_3y": -0.1 + (index % 5) * 0.05,
        "rendimiento_alto": (index % 2).astype(int),
    })
    return frame


class SmoteIsolationTests(unittest.TestCase):
    def test_smotenc_pipeline_contains_sampler_only_for_smote_candidates(self):
        with_smote = build_classifier("logistic_smotenc")
        without_smote = build_classifier("logistic_plain")
        self.assertIn("sampler", with_smote.named_steps)
        self.assertNotIn("sampler", without_smote.named_steps)

    def test_sampler_receives_train_rows_only(self):
        train = make_classification_frame(24, last_year=2018)
        test = make_classification_frame(7, last_year=2019)
        original = SMOTENC.fit_resample
        with patch(
            "imblearn.over_sampling.SMOTENC.fit_resample",
            autospec=True,
            side_effect=original,
        ) as mocked:
            probabilities, _ = fit_predict_classifier("logistic_smotenc", train, test)
        sampled_rows = mocked.call_args.args[1].shape[0]
        self.assertEqual(sampled_rows, len(train))
        self.assertEqual(len(probabilities), len(test))
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_pipeline.SmoteIsolationTests -v`

Expected: FAIL because classifier builders are missing.

- [ ] **Step 3: Implement preprocessing and classifier factories**

The categorical columns must be the first two transformed columns so `SMOTENC(categorical_features=[0, 1])` addresses them exactly:

```python
def _ordinal_preprocessor() -> ColumnTransformer:
    categorical = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    numeric = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("numeric", numeric, NUMERIC_FEATURES),
    ])


def _one_hot_after_ordinal() -> ColumnTransformer:
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), [0, 1]),
        ("numeric", "passthrough", list(range(2, 2 + len(NUMERIC_FEATURES)))),
    ])
```

Factories must expose the exact names `logistic_plain`, `logistic_smotenc`, `random_forest_plain`, `random_forest_smotenc`, and `catboost_weighted`. Logistic pipelines use `LogisticRegression(max_iter=2000, random_state=42)`. Random forests use 300 trees, `min_samples_leaf=3`, `n_jobs=-1`, and seed 42. CatBoost uses 300 iterations, depth 6, learning rate 0.05, `verbose=False`, `allow_writing_files=False`, and inverse-frequency class weights; it receives categorical strings filled with `__MISSING__` and numeric medians learned from train.

- [ ] **Step 4: Run SMOTENC tests**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_pipeline.SmoteIsolationTests -v`

Expected: 2 tests PASS and test predictions remain exactly 7 rows.

- [ ] **Step 5: Commit the isolated task**

```powershell
git add src/evaluation/production_pipeline.py tests/test_production_pipeline.py
git commit -m "feat: add leakage-safe SMOTENC classifiers"
```

### Task 3: Temporal classification, calibration, and holdout

**Files:**
- Modify: `src/evaluation/production_pipeline.py`
- Modify: `tests/test_production_pipeline.py`

**Interfaces:**
- Consumes: enriched eligible data and classifier factories from Task 2.
- Produces: `classification_metrics(y_true, y_pred, probability) -> dict[str, float]`, `classification_backtest(df: pd.DataFrame, high_yield_threshold: float, years: list[int] = SELECTION_YEARS) -> pd.DataFrame`, `select_classifier(summary: pd.DataFrame) -> str`, `fit_calibrator(oof: pd.DataFrame) -> LogisticRegression`, `choose_threshold(oof: pd.DataFrame, calibrator: LogisticRegression) -> float`, `evaluate_classification_holdout(df: pd.DataFrame, winner: str, calibrator: LogisticRegression, decision_threshold: float) -> tuple[pd.DataFrame, BaseEstimator]`.

- [ ] **Step 1: Write failing metrics and temporal-selection tests**

```python
from src.evaluation.production_pipeline import (
    choose_threshold,
    classification_metrics,
    select_classifier,
)


class ClassificationEvaluationTests(unittest.TestCase):
    def test_mcc_precision_and_matrix_are_reproducible(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        probability = np.array([0.1, 0.7, 0.8, 0.9])
        metrics = classification_metrics(y_true, y_pred, probability)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["mcc"], 1 / np.sqrt(3))
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tp"], 2)

    def test_selection_uses_mean_annual_mcc_then_balanced_accuracy(self):
        summary = pd.DataFrame([
            {"model": "A", "is_challenger": True, "mean_mcc": 0.30, "mean_balanced_accuracy": 0.80},
            {"model": "B", "is_challenger": True, "mean_mcc": 0.31, "mean_balanced_accuracy": 0.70},
            {"model": "lag1", "is_challenger": False, "mean_mcc": 0.90, "mean_balanced_accuracy": 0.95},
        ])
        self.assertEqual(select_classifier(summary), "B")
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_pipeline.ClassificationEvaluationTests -v`

Expected: FAIL because the evaluation functions are missing.

- [ ] **Step 3: Implement strict rolling-origin evaluation**

For each year in `[2019, 2020, 2021, 2022, 2023]`, use `train = df[df.anio < year]` and `test = df[df.anio == year]`. Record row identity, year, true class, raw probability, prediction at 0.5, model name, train maximum year and whether the model is a challenger. Include two reference rows per observation: the train-majority prediction and `rendimiento_lag_1 > high_yield_threshold`.

Compute metrics with:

```python
matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()
return {
    "accuracy": accuracy_score(y_true, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score(y_true, y_pred, zero_division=0),
    "f1": f1_score(y_true, y_pred, zero_division=0),
    "mcc": matthews_corrcoef(y_true, y_pred),
    "roc_auc": roc_auc_score(y_true, probability),
    "pr_auc": average_precision_score(y_true, probability),
    "brier": brier_score_loss(y_true, probability),
    "ece": expected_calibration_error(y_true, probability, bins=10),
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
}
```

- [ ] **Step 4: Implement frozen calibration and threshold selection**

Fit a one-dimensional logistic calibrator to the selected challenger's 2019-2023 raw out-of-fold probabilities. Evaluate thresholds `np.linspace(0.05, 0.95, 91)` and rank by mean annual MCC, mean recall, then negative absolute distance to 0.5. Freeze the winning threshold and calibrator before calling the holdout function.

The holdout function fits the chosen model on years `< 2024`, predicts 2024 exactly once, applies the frozen calibrator and threshold, and returns one prediction row per holdout row plus the fitted evaluation model. It must raise `AssertionError` if training contains 2024 or if prediction count differs from holdout count.

- [ ] **Step 5: Run classification tests and a one-fold smoke test**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_pipeline.ClassificationEvaluationTests -v`

Run: `..\.validation-venv\Scripts\python.exe -c "from src.evaluation.production_pipeline import load_dataset,add_high_yield_target,classification_backtest; d,t=add_high_yield_target(load_dataset()); print(classification_backtest(d, t, years=[2019]).groupby('model').size())"`

Expected: unit tests PASS; every classifier and baseline emits the same positive number of 2019 predictions.

- [ ] **Step 6: Commit the isolated task**

```powershell
git add src/evaluation/production_pipeline.py tests/test_production_pipeline.py
git commit -m "feat: add temporal classification evaluation"
```

### Task 4: Regression comparison and production gate

**Files:**
- Modify: `src/evaluation/production_pipeline.py`
- Modify: `tests/test_production_pipeline.py`

**Interfaces:**
- Consumes: strict features, rolling years, holdout isolation.
- Produces: `build_regressor(name: str) -> BaseEstimator`, `regression_backtest(df: pd.DataFrame, years: list[int] = SELECTION_YEARS) -> pd.DataFrame`, `regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]`, `select_regressor(summary: pd.DataFrame) -> str`, `evaluate_regression_holdout(df: pd.DataFrame, winner: str) -> tuple[pd.DataFrame, BaseEstimator]`, `production_gate(integrity_ok: bool, challenger_mean_mcc: float, lag1_mean_mcc: float, challenger_annual_mcc: list[float], holdout_mcc: float, holdout_balanced_accuracy: float, regression_mae: float, lag1_regression_mae: float, calibrated_brier: float, prevalence_brier: float) -> dict[str, object]`.

- [ ] **Step 1: Write failing gate and regression tests**

```python
from src.evaluation.production_pipeline import production_gate, regression_metrics


class RegressionAndGateTests(unittest.TestCase):
    def test_regression_metrics(self):
        metrics = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        self.assertEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.5))
        self.assertEqual(metrics["r2"], -1.0)

    def test_gate_lists_every_failed_condition(self):
        gate = production_gate(
            integrity_ok=True,
            challenger_mean_mcc=0.31,
            lag1_mean_mcc=0.30,
            challenger_annual_mcc=[0.2, -0.01, 0.3],
            holdout_mcc=0.49,
            holdout_balanced_accuracy=0.69,
            regression_mae=0.96,
            lag1_regression_mae=1.0,
            calibrated_brier=0.21,
            prevalence_brier=0.20,
        )
        self.assertEqual(gate["status"], "experimental_not_approved")
        self.assertEqual(len(gate["failed_checks"]), 6)
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_pipeline.RegressionAndGateTests -v`

Expected: FAIL because regression and gate functions are missing.

- [ ] **Step 3: Implement regressors and temporal evaluation**

Implement `lag1` and `mean3` reference predictions plus these challengers:

```python
def build_regressor(name: str) -> BaseEstimator:
    if name == "elastic_net":
        return ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=42)
    if name == "random_forest_regressor":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
        )
    if name == "catboost_regressor":
        return CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.05, loss_function="MAE",
            random_seed=42, verbose=False, allow_writing_files=False,
        )
    raise ValueError(f"Regresor desconocido: {name}")
```

Elastic Net and Random Forest use strict sklearn preprocessing; CatBoost uses the train-learned categorical fill and numeric medians. Selection ranks challengers by mean annual MAE. The holdout model trains only through 2023.

- [ ] **Step 4: Implement all seven production checks**

`production_gate` returns `{"status": str, "checks": list[dict], "failed_checks": list[str]}`. Checks use exact comparisons: integrity true; MCC improvement at least 0.02; minimum annual MCC at least 0; holdout MCC at least 0.50; holdout balanced accuracy at least 0.70; holdout regression MAE improvement at least 5%; calibrated holdout Brier strictly lower than the holdout Brier of a constant train-prevalence probability.

- [ ] **Step 5: Run regression and gate tests**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_pipeline.RegressionAndGateTests -v`

Expected: tests PASS, including six exact failures in the deliberately failing gate fixture.

- [ ] **Step 6: Commit the isolated task**

```powershell
git add src/evaluation/production_pipeline.py tests/test_production_pipeline.py
git commit -m "feat: add regression validation and production gate"
```

### Task 5: Reports, plots, manifest, and serialized artifacts

**Files:**
- Modify: `src/evaluation/production_pipeline.py`
- Modify: `tests/test_production_pipeline.py`
- Create: `reports/modelo_produccion_cafe/` outputs during execution

**Interfaces:**
- Consumes: classification/regression OOF tables, frozen calibrator and threshold, 2024 holdout predictions, fitted evaluation models.
- Produces: `manifest_status(failed_checks: list[str]) -> str`, `build_manifest(result: dict[str, object]) -> dict[str, object]`, `create_all_plots(result, output_dir) -> list[Path]`, `export_artifacts(result, output_dir) -> dict[str, Path]`, `run_production_pipeline(data_path=DATA_PATH, output_dir=OUTPUT_DIR) -> dict[str, object]`.

- [ ] **Step 1: Write failing artifact tests**

```python
from src.evaluation.production_pipeline import REQUIRED_PLOTS, manifest_status


class ArtifactTests(unittest.TestCase):
    def test_manifest_gate_matches_failed_checks(self):
        self.assertEqual(manifest_status([]), "production_candidate")
        self.assertEqual(
            manifest_status(["MCC holdout inferior a 0.50"]),
            "experimental_not_approved",
        )

    def test_required_plot_names_are_complete(self):
        self.assertEqual(
            set(REQUIRED_PLOTS),
            {
                "01_distribucion_rendimiento.png", "02_boxplot_rendimiento_anio.png",
                "03_faltantes_variables.png", "04_balance_clase_anio.png",
                "05_smotenc_antes_despues.png", "06_modelos_mcc_balanced_accuracy.png",
                "07_mcc_anual.png", "08_matriz_confusion_2024.png",
                "09_curvas_roc_pr_2024.png", "10_calibracion_2024.png",
                "11_mae_modelo_anio.png", "12_real_vs_predicho_2024.png",
                "13_importancia_variables.png",
            },
        )
```

- [ ] **Step 2: Run artifact tests and confirm failure**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_pipeline.ArtifactTests -v`

Expected: FAIL because manifest and plot contract are missing.

- [ ] **Step 3: Implement the orchestrator and atomic exports**

`run_production_pipeline` executes audit, target, SMOTENC snapshot, classification backtest, challenger selection, calibration, threshold choice, classification holdout, regression backtest, regressor selection, regression holdout, gate, plotting and export in that order. It writes to a temporary sibling directory and moves completed files into `reports/modelo_produccion_cafe/` only after every step succeeds.

Export these exact machine-readable files:

```text
model_manifest.json
calidad_datos.json
balance_clases.csv
smotenc_antes_despues.csv
clasificacion_oof_2019_2023.csv
clasificacion_resumen_modelos.csv
clasificacion_metricas_anuales.csv
clasificacion_holdout_2024.csv
regresion_oof_2019_2023.csv
regresion_resumen_modelos.csv
regresion_metricas_anuales.csv
regresion_holdout_2024.csv
classifier_evaluation.joblib
classifier_pipeline.joblib
probability_calibrator.joblib
regressor_evaluation.joblib
regressor_pipeline.joblib
MODEL_CARD.md
```

The two operational pipelines are refitted on all eligible data through 2024 only after the frozen evaluation is stored. The manifest records evaluation max year 2023, operational max year 2024, strict features, excluded features, threshold values, versions, hashes, metrics, checks and status. `MODEL_CARD.md` states permitted use as yield/risk prioritization, prohibited use as profitability or autonomous credit/investment approval, known drift, source switch and geographic join limitations.

- [ ] **Step 4: Implement all required plots**

Use `matplotlib.use("Agg")`, seaborn's `whitegrid`, DPI 150 and Spanish titles. Every plot must include period and units. The confusion matrix uses rows `Real bajo/alto` and columns `Predicho bajo/alto`; ROC and PR share one figure; the calibration plot includes perfect-calibration diagonal; importance uses permutation importance on the 2024 holdout so different model families share one method.

- [ ] **Step 5: Run artifact tests and full unit suite**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_pipeline -v`

Expected: all production pipeline unit tests PASS.

- [ ] **Step 6: Commit the isolated task**

```powershell
git add src/evaluation/production_pipeline.py tests/test_production_pipeline.py
git commit -m "feat: export coffee model evidence and artifacts"
```

### Task 6: Generate, back up, and replace the notebook

**Files:**
- Create: `scripts/create_production_notebook.py`
- Create: `tests/test_production_notebook.py`
- Create once: `pipeline_interactivo_cafe_original.ipynb`
- Replace: `pipeline_interactivo_cafe.ipynb`

**Interfaces:**
- Consumes: `run_production_pipeline`, exported CSV/JSON/PNG files.
- Produces: executed notebook with no error outputs and exactly 17 numbered sections.

- [ ] **Step 1: Write failing notebook structure tests**

```python
import json
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


class ProductionNotebookTests(unittest.TestCase):
    def test_original_backup_and_replacement_exist(self):
        self.assertTrue((ROOT / "pipeline_interactivo_cafe_original.ipynb").is_file())
        self.assertTrue((ROOT / "pipeline_interactivo_cafe.ipynb").is_file())

    def test_notebook_has_ordered_sections_and_no_errors(self):
        notebook = nbformat.read(ROOT / "pipeline_interactivo_cafe.ipynb", as_version=4)
        headings = [cell.source.splitlines()[0] for cell in notebook.cells if cell.cell_type == "markdown" and cell.source.startswith("## ")]
        self.assertEqual([int(h.split(".")[0].removeprefix("## ")) for h in headings], list(range(1, 18)))
        errors = [output for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []) if output.output_type == "error"]
        self.assertEqual(errors, [])
        self.assertTrue(all(cell.get("execution_count") is not None for cell in notebook.cells if cell.cell_type == "code"))

    def test_notebook_reports_manifest_status(self):
        manifest = json.loads((ROOT / "reports/modelo_produccion_cafe/model_manifest.json").read_text(encoding="utf-8"))
        notebook = nbformat.read(ROOT / "pipeline_interactivo_cafe.ipynb", as_version=4)
        rendered = "\n".join(str(output) for cell in notebook.cells for output in cell.get("outputs", []))
        self.assertIn(manifest["status"], rendered)
```

- [ ] **Step 2: Run notebook tests and confirm failure**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_notebook -v`

Expected: FAIL because backup, replacement and report outputs do not yet exist.

- [ ] **Step 3: Implement safe backup and notebook generation**

The generator uses:

```python
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "pipeline_interactivo_cafe.ipynb"
BACKUP_PATH = ROOT / "pipeline_interactivo_cafe_original.ipynb"

if NOTEBOOK_PATH.exists() and not BACKUP_PATH.exists():
    shutil.copy2(NOTEBOOK_PATH, BACKUP_PATH)
```

Generate a title cell followed by numbered Markdown sections 1 through 17 matching the spec. Code cells call `run_production_pipeline`, load each table, display every required figure with `IPython.display.Image`, print interpretations from computed values, display the matriz de confusión 2024, MCC, precision, recall, false negatives, regression metrics, all gate checks and artifact paths. No cell uses shell magics, internet access, manual inputs or hard-coded final metrics.

- [ ] **Step 4: Execute the generated notebook**

Use `NotebookClient(timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})` and save only after successful execution. Set notebook metadata for Python 3.12 and seed 42.

Run: `..\.validation-venv\Scripts\python.exe scripts\create_production_notebook.py`

Expected: prints the replacement notebook path and reports directory; creates all required artifacts without an exception.

- [ ] **Step 5: Run notebook tests**

Run: `..\.validation-venv\Scripts\python.exe -m unittest tests.test_production_notebook -v`

Expected: all notebook structure and output tests PASS.

- [ ] **Step 6: Commit the isolated task**

```powershell
git add scripts/create_production_notebook.py tests/test_production_notebook.py pipeline_interactivo_cafe.ipynb pipeline_interactivo_cafe_original.ipynb
git commit -m "feat: replace coffee notebook with rigorous workflow"
```

### Task 7: Independent final verification

**Files:**
- Verify: `pipeline_interactivo_cafe.ipynb`
- Verify: `reports/modelo_produccion_cafe/`
- Verify: all modified Python and requirement files.

**Interfaces:**
- Consumes: final notebook, predictions and manifest.
- Produces: independent evidence that confusion matrix, MCC, precision, artifact status and temporal boundaries agree.

- [ ] **Step 1: Install declared dependencies in the isolated environment**

Run: `..\.validation-venv\Scripts\python.exe -m pip install -r requirements.txt`

Expected: exit code 0, including `imbalanced-learn`, `nbformat`, `nbclient` and `ipykernel`.

- [ ] **Step 2: Run all tests**

Run: `..\.validation-venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: every test PASS.

- [ ] **Step 3: Recompute final classification evidence independently**

```powershell
@'
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score

root = Path.cwd()
out = root / "reports" / "modelo_produccion_cafe"
pred = pd.read_csv(out / "clasificacion_holdout_2024.csv")
manifest = json.loads((out / "model_manifest.json").read_text(encoding="utf-8"))
matrix = confusion_matrix(pred["y_true"], pred["y_pred"], labels=[0, 1]).tolist()
mcc = matthews_corrcoef(pred["y_true"], pred["y_pred"])
precision = precision_score(pred["y_true"], pred["y_pred"], zero_division=0)
assert matrix == manifest["classification"]["holdout_2024"]["confusion_matrix"]
assert abs(mcc - manifest["classification"]["holdout_2024"]["mcc"]) < 1e-12
assert abs(precision - manifest["classification"]["holdout_2024"]["precision"]) < 1e-12
assert set(pred["anio"]) == {2024}
print({"matrix": matrix, "mcc": mcc, "precision": precision, "status": manifest["status"]})
'@ | ..\.validation-venv\Scripts\python.exe -
```

Expected: all assertions pass and the command prints matrix, MCC, precision and gate status.

- [ ] **Step 4: Verify file inventory and notebook execution state**

Run: `..\.validation-venv\Scripts\python.exe -c "from pathlib import Path; import nbformat; n=nbformat.read('pipeline_interactivo_cafe.ipynb',4); assert not [o for c in n.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']; print(len(n.cells), len(list(Path('reports/modelo_produccion_cafe').glob('*'))))"`

Expected: no assertion error, a nonzero notebook cell count and at least 31 report artifacts including 13 PNG figures.

- [ ] **Step 5: Inspect only task-related changes**

Run: `git status --short -- requirements.txt src/evaluation/production_pipeline.py tests/test_production_pipeline.py tests/test_production_notebook.py scripts/create_production_notebook.py pipeline_interactivo_cafe.ipynb pipeline_interactivo_cafe_original.ipynb docs/superpowers reports/modelo_produccion_cafe`

Expected: only the intended task paths are listed; unrelated pre-existing worktree changes remain untouched.

- [ ] **Step 6: Record the final commit when repository identity permits it**

```powershell
git add requirements.txt src/evaluation/production_pipeline.py tests/test_production_pipeline.py tests/test_production_notebook.py scripts/create_production_notebook.py pipeline_interactivo_cafe.ipynb pipeline_interactivo_cafe_original.ipynb docs/superpowers reports/modelo_produccion_cafe
git commit -m "feat: prepare rigorous coffee model notebook"
```
