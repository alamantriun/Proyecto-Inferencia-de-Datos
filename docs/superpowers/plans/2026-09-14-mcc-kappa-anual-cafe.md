# Annual Coffee MCC and Kappa Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild and replace the coffee notebook with a point-in-time annual classifier that reports MCC, Cohen's Kappa, WEKA metrics, and confusion matrices for 2019-2025 and refuses production status unless every year reaches 0.85 in both priority metrics.

**Architecture:** New focused modules will isolate source normalization, point-in-time feature creation, temporal model selection, and artifact generation. The notebook will be generated from tested functions, will keep 2025 sealed until the final evaluation cell, and will use cached official data so it remains rerunnable without hidden state.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn, imbalanced-learn/SMOTENC, CatBoost, requests, matplotlib, seaborn, joblib, nbformat, nbclient, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-mcc-kappa-anual-cafe-design.md`

## Global Constraints

- The prediction for year `t` may use only information available through December 31 of `t-1`.
- The high-yield target threshold is fixed from 2007-2018 and never recalculated with validation or holdout labels.
- Backtest years are exactly 2019-2024; 2025 is a one-time final holdout.
- MCC and Cohen's Kappa must each be at least 0.85 in every evaluated year; comparisons use unrounded values.
- SMOTENC operates only inside a training fold and never resamples validation or holdout rows.
- Canonical geographic joins use five-character DIVIPOLA municipality codes, never municipality name alone.
- The notebook must show a confusion matrix for every year 2019-2025 and one consolidated matrix.
- A failed requirement produces `target_not_met`; no metric or threshold may be manipulated to claim success.
- Existing unrelated working-tree changes must remain untouched.
- Git commits require repository-local author identity from the user; do not invent or configure an identity.

---

### Task 1: Official EVA and DIVIPOLA source contract

**Files:**
- Create: `src/data/cafe_annual_sources.py`
- Create: `tests/test_cafe_annual_sources.py`
- Modify: `config/config.yaml`

**Interfaces:**
- Produces: `normalize_divipola_code(value: object) -> str`, `normalize_text(value: object) -> str`, `harmonize_eva(historical: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame`, `parse_divipola_features(payload: dict) -> pd.DataFrame`, `download_official_sources(cache_dir: Path, refresh: bool = False) -> dict[str, Path]`.
- The harmonized EVA columns are `codigo_dane_municipio`, `codigo_dane_departamento`, `departamento`, `municipio`, `cultivo`, `anio`, `area_sembrada_ha`, `area_cosechada_ha`, `produccion_t`, `rendimiento_t_ha`, `estado_fisico`, and `fuente_eva`.

- [ ] **Step 1: Write failing normalization and aggregation tests**

```python
def test_harmonize_eva_preserves_five_digit_codes_and_recomputes_weighted_yield():
    historical = make_historical_rows()
    recent = make_recent_rows_with_duplicate_municipality_year()
    result = harmonize_eva(historical, recent)
    row = result.query("codigo_dane_municipio == '05001' and anio == 2025").iloc[0]
    assert row["produccion_t"] == 30.0
    assert row["area_cosechada_ha"] == 20.0
    assert row["rendimiento_t_ha"] == 1.5
    assert result[["codigo_dane_municipio", "anio"]].duplicated().sum() == 0

def test_parse_divipola_uses_official_centroid_and_rejects_duplicate_codes():
    result = parse_divipola_features(make_arcgis_payload())
    assert result.loc[0, "codigo_dane_municipio"] == "05001"
    assert result.loc[0, "latitud"] == pytest.approx(6.25)
    assert result.loc[0, "longitud"] == pytest.approx(-75.56)
```

- [ ] **Step 2: Run source tests and verify RED**

Run: `python -m pytest tests/test_cafe_annual_sources.py -v`

Expected: collection fails because `src.data.cafe_annual_sources` does not exist.

- [ ] **Step 3: Implement deterministic source normalization**

```python
EVA_HISTORICAL_ID = "2pnw-mmge"
EVA_RECENT_ID = "uejq-wxrr"
DIVIPOLA_URL = (
    "https://geoportal.dane.gov.co/mparcgis/rest/services/Divipola/"
    "Serv_DIVIPOLA_MGN_2025/FeatureServer/317/query"
)

def normalize_divipola_code(value: object) -> str:
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        raise ValueError("Código DIVIPOLA vacío")
    return digits.zfill(5)

def _aggregate_municipality_year(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["codigo_dane_municipio", "anio"], as_index=False).agg(
        codigo_dane_departamento=("codigo_dane_departamento", "first"),
        departamento=("departamento", "first"),
        municipio=("municipio", "first"),
        cultivo=("cultivo", "first"),
        area_sembrada_ha=("area_sembrada_ha", "sum"),
        area_cosechada_ha=("area_cosechada_ha", "sum"),
        produccion_t=("produccion_t", "sum"),
        estado_fisico=("estado_fisico", "first"),
        fuente_eva=("fuente_eva", "first"),
    )
    grouped["rendimiento_t_ha"] = grouped["produccion_t"] / grouped["area_cosechada_ha"]
    return grouped
```

Download EVA with its official DANE code fields and filter the recent source with `c_digo_del_cultivo='2030300'`. Query the DANE ArcGIS municipality layer with `returnCentroid=true`, `outSR=4326`, and explicit fields. Write downloads to temporary files, calculate SHA-256, then atomically publish cached CSV/JSON and `source_manifest.json`.

- [ ] **Step 4: Update source years and URLs in configuration**

Set `eva_reciente.años` to `[2019, 2025]`, add the DANE ArcGIS source, add the NASA POWER monthly regional endpoint, and set `project.años_backtest` to 2019-2024 with `holdout_final: 2025`.

- [ ] **Step 5: Run source tests and full existing tests**

Run: `python -m pytest tests/test_cafe_annual_sources.py -v`

Expected: all new source tests pass.

Run: `python -m pytest -q`

Expected: no regression failures.

- [ ] **Step 6: Commit the source contract**

```bash
git add src/data/cafe_annual_sources.py tests/test_cafe_annual_sources.py config/config.yaml
git commit -m "feat: add official annual coffee sources"
```

---

### Task 2: Regional NASA POWER climate cache

**Files:**
- Create: `src/data/nasa_power_climate.py`
- Create: `tests/test_nasa_power_climate.py`

**Interfaces:**
- Consumes: DIVIPOLA output from Task 1.
- Produces: `parse_power_regional(payload: dict, parameter: str) -> pd.DataFrame`, `aggregate_power_yearly(monthly: pd.DataFrame) -> pd.DataFrame`, `nearest_grid_climate(municipalities: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame`, and `download_power_climate(cache_dir: Path, start_year: int = 2006, end_year: int = 2024, refresh: bool = False) -> Path`.
- Final yearly climate columns: `codigo_dane_municipio`, `anio_clima`, `precipitacion_total_mm`, `temperatura_media_c`, `temperatura_min_c`, `temperatura_max_c`, `distancia_grilla_grados`, and `fuente_clima`.

- [ ] **Step 1: Write failing parser and nearest-grid tests**

```python
def test_annual_precipitation_converts_monthly_mm_per_day_to_total_mm():
    monthly = make_monthly_power_frame(value=2.0, year=2020)
    annual = aggregate_power_yearly(monthly)
    assert annual.loc[0, "precipitacion_total_mm"] == pytest.approx(732.0)

def test_nearest_grid_assignment_keeps_one_row_per_municipality_year():
    result = nearest_grid_climate(make_municipalities(), make_grid_climate())
    assert result[["codigo_dane_municipio", "anio_clima"]].duplicated().sum() == 0
    assert set(result["fuente_clima"]) == {"NASA_POWER_MERRA2"}
```

- [ ] **Step 2: Run climate tests and verify RED**

Run: `python -m pytest tests/test_nasa_power_climate.py -v`

Expected: collection fails because `src.data.nasa_power_climate` does not exist.

- [ ] **Step 3: Implement bounded regional extraction and parsing**

```python
POWER_PARAMETERS = ("PRECTOTCORR", "T2M", "T2M_MIN", "T2M_MAX")
COLOMBIA_TILES = (
    (-5.0, 5.0, -80.0, -70.0),
    (-5.0, 5.0, -70.0, -66.0),
    (5.0, 14.0, -80.0, -70.0),
    (5.0, 14.0, -70.0, -66.0),
)

def _power_url(parameter: str, tile: tuple[float, float, float, float], start: int, end: int) -> str:
    return "https://power.larc.nasa.gov/api/temporal/monthly/regional?" + urlencode({
        "latitude-min": tile[0], "latitude-max": tile[1],
        "longitude-min": tile[2], "longitude-max": tile[3],
        "parameters": parameter, "community": "AG",
        "start": start, "end": end, "format": "JSON",
    })
```

Request at most one parameter per regional call, respect the documented ten-degree tile limit, retry HTTP 429/5xx with bounded exponential backoff, reject fill values `<= -900`, and cache every raw response before deriving the municipal file.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_nasa_power_climate.py -v`

Expected: all climate tests pass.

- [ ] **Step 5: Commit climate extraction**

```bash
git add src/data/nasa_power_climate.py tests/test_nasa_power_climate.py
git commit -m "feat: add cached NASA POWER climate features"
```

---

### Task 3: Point-in-time annual feature mart

**Files:**
- Create: `src/features/cafe_annual_features.py`
- Create: `tests/test_cafe_annual_features.py`

**Interfaces:**
- Consumes: harmonized EVA, DIVIPOLA, and yearly climate tables.
- Produces: `build_point_in_time_features(eva: pd.DataFrame, municipalities: pd.DataFrame, climate: pd.DataFrame | None = None) -> pd.DataFrame`, `fixed_high_yield_threshold(frame: pd.DataFrame) -> float`, and `audit_point_in_time(frame: pd.DataFrame) -> dict[str, object]`.
- Required provenance columns: `feature_cutoff_year`, `max_eva_feature_year`, `max_climate_feature_year`, `source_break_2019`, `climate_available`, and `history_years`.

- [ ] **Step 1: Write failing leakage and lag tests**

```python
def test_features_for_2025_never_use_2025_values():
    result = build_point_in_time_features(make_eva_panel(), make_municipalities(), make_climate())
    row = result.query("codigo_dane_municipio == '05001' and anio == 2025").iloc[0]
    assert row["rendimiento_lag_1"] == pytest.approx(1.4)
    assert row["max_eva_feature_year"] == 2024
    assert row["max_climate_feature_year"] <= 2024
    assert row["feature_cutoff_year"] == 2024

def test_lag_requires_consecutive_calendar_year():
    result = build_point_in_time_features(make_panel_with_gap(), make_municipalities())
    row = result.query("anio == 2022").iloc[0]
    assert pd.isna(row["rendimiento_lag_1"])
```

- [ ] **Step 2: Run feature tests and verify RED**

Run: `python -m pytest tests/test_cafe_annual_features.py -v`

Expected: collection fails because `src.features.cafe_annual_features` does not exist.

- [ ] **Step 3: Implement calendar-safe history features**

Create a complete municipality-year index only for observed target rows. Build lags by self-joining on `anio - lag`, not by positional `groupby.shift`, so missing years cannot masquerade as consecutive history. Compute rolling values from rows with `anio < anio_objetivo` and include counts to distinguish short histories.

```python
for lag in (1, 2, 3):
    previous = base[keys + value_columns].copy()
    previous["anio"] = previous["anio"] + lag
    previous = previous.rename(columns={c: f"{c}_lag_{lag}" for c in value_columns})
    mart = mart.merge(previous, on=[*keys, "anio"], how="left", validate="one_to_one")
```

Join climate with `anio_clima = anio - 1` and assert the maximum joined climate year is below the target year. Add expanding municipality/departament/national priors with a one-row shift before aggregation.

- [ ] **Step 4: Implement strict audit and fixed target**

`fixed_high_yield_threshold` must use only rows where `anio <= 2018`. `audit_point_in_time` must raise on duplicate municipality-year keys, invalid codes, impossible target values, or any provenance year greater than or equal to target year.

- [ ] **Step 5: Run feature and full tests**

Run: `python -m pytest tests/test_cafe_annual_features.py -v`

Expected: all feature tests pass.

Run: `python -m pytest -q`

Expected: no regression failures.

- [ ] **Step 6: Commit feature mart**

```bash
git add src/features/cafe_annual_features.py tests/test_cafe_annual_features.py
git commit -m "feat: build point in time coffee features"
```

---

### Task 4: Annual WEKA metrics, nested thresholds, and strict gate

**Files:**
- Create: `src/evaluation/annual_classifier.py`
- Create: `tests/test_annual_classifier.py`

**Interfaces:**
- Consumes: point-in-time mart from Task 3.
- Produces: `annual_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`, `select_threshold_from_prior_years(predictions: pd.DataFrame, evaluation_year: int) -> float`, `walk_forward_backtest(frame: pd.DataFrame, model_names: Sequence[str], years: Sequence[int]) -> dict[str, object]`, `evaluate_final_holdout(frame: pd.DataFrame, recipe: ModelRecipe, year: int = 2025) -> tuple[pd.DataFrame, BaseEstimator]`, and `strict_production_gate(metrics: pd.DataFrame, baseline_metrics: pd.DataFrame, threshold: float = 0.85) -> dict[str, object]`.
- `ModelRecipe` contains model name, hyperparameters, selected raw features, and the rule for learning a threshold from prior predictions.

- [ ] **Step 1: Write failing hand-calculated MCC/Kappa/WEKA tests**

```python
def test_annual_metrics_matches_hand_calculated_confusion_matrix():
    predictions = pd.DataFrame({
        "anio": [2024] * 4, "y_true": [0, 0, 1, 1],
        "y_pred": [0, 1, 1, 1], "probability": [0.1, 0.6, 0.7, 0.8],
        "model": ["demo"] * 4,
    })
    annual, detailed = annual_metrics(predictions)
    row = annual.iloc[0]
    assert [row.tn, row.fp, row.fn, row.tp] == [1, 1, 0, 2]
    assert row.mcc == pytest.approx(1 / np.sqrt(3))
    assert row.kappa == pytest.approx(0.5)
    assert "Weighted Avg." in set(detailed["Class"])
```

- [ ] **Step 2: Write failing temporal selection and gate tests**

```python
def test_threshold_selection_rejects_labels_from_evaluated_or_future_years():
    with pytest.raises(ValueError, match="posteriores"):
        select_threshold_from_prior_years(make_predictions_through_2025(), evaluation_year=2024)

def test_gate_fails_when_one_year_is_below_target():
    metrics = make_metrics(mcc=[.91, .90, .84], kappa=[.90, .89, .88], years=[2023, 2024, 2025])
    gate = strict_production_gate(metrics, make_baseline_metrics(), threshold=.85)
    assert gate["status"] == "target_not_met"
    assert gate["failed_years"] == [2025]
```

- [ ] **Step 3: Run classifier tests and verify RED**

Run: `python -m pytest tests/test_annual_classifier.py -v`

Expected: collection fails because `src.evaluation.annual_classifier` does not exist.

- [ ] **Step 4: Implement model factories and train-only SMOTENC**

Implement majority and lag-1 baselines, Elastic Net logistic regression with and without SMOTENC, shallow HistGradientBoosting with SMOTENC, shallow CatBoost with class weights, and regression-to-threshold classification. Each estimator receives a small fixed grid. All transformations live inside an imbalanced-learn pipeline and fit only on the training slice.

- [ ] **Step 5: Implement nested rolling-origin selection**

For each external year, run inner prior-year folds. Rank candidates lexicographically by worst annual `min(mcc, kappa)`, mean annual `min(mcc, kappa)`, weighted WEKA precision, and model simplicity. Choose the decision threshold on inner out-of-fold predictions only. Persist `train_max_year`, `threshold_source_max_year`, and fitted recipe on every prediction row.

- [ ] **Step 6: Implement strict gate and confusion tables**

```python
def strict_production_gate(metrics, baseline_metrics, threshold=0.85):
    evaluated = metrics.query("anio >= 2019 and anio <= 2025").copy()
    evaluated["passes"] = (evaluated["mcc"] >= threshold) & (evaluated["kappa"] >= threshold)
    failed = evaluated.loc[~evaluated["passes"], "anio"].astype(int).tolist()
    return {
        "status": "production_candidate" if not failed else "target_not_met",
        "threshold": float(threshold),
        "failed_years": failed,
        "minimum_mcc": float(evaluated["mcc"].min()),
        "minimum_kappa": float(evaluated["kappa"].min()),
    }
```

Add the required 0.02 worst-year improvement over persistence and complete failure reasons from the spec.

- [ ] **Step 7: Run classifier and full tests**

Run: `python -m pytest tests/test_annual_classifier.py -v`

Expected: all classifier tests pass.

Run: `python -m pytest -q`

Expected: no regression failures.

- [ ] **Step 8: Commit annual evaluation**

```bash
git add src/evaluation/annual_classifier.py tests/test_annual_classifier.py
git commit -m "feat: validate annual MCC and Kappa"
```

---

### Task 5: Figures and production artifacts

**Files:**
- Create: `src/evaluation/annual_artifacts.py`
- Create: `tests/test_annual_artifacts.py`

**Interfaces:**
- Consumes: source manifest, audited mart, predictions, annual metrics, WEKA detail, fitted evaluation model, fitted operational model, and gate.
- Produces: `create_annual_figures(result: dict[str, object], output_dir: Path) -> list[Path]` and `export_annual_artifacts(result: dict[str, object], output_dir: Path) -> dict[str, Path]`.

- [ ] **Step 1: Write failing figure inventory and atomic export tests**

```python
def test_figure_bundle_contains_all_annual_confusion_matrices(tmp_path):
    paths = create_annual_figures(make_complete_result(), tmp_path)
    names = {p.name for p in paths}
    assert "08_matrices_confusion_2019_2025.png" in names
    assert "09_matriz_confusion_consolidada.png" in names
    assert all(p.stat().st_size > 0 for p in paths)

def test_export_manifest_keeps_real_failed_years(tmp_path):
    paths = export_annual_artifacts(make_failed_result(), tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["status"] == "target_not_met"
    assert manifest["gate"]["failed_years"] == [2019, 2020]
```

- [ ] **Step 2: Run artifact tests and verify RED**

Run: `python -m pytest tests/test_annual_artifacts.py -v`

Expected: collection fails because `src.evaluation.annual_artifacts` does not exist.

- [ ] **Step 3: Implement the required charts**

Generate bounded and labeled plots for source coverage, initial yield, annual boxplots, missingness, annual class balance, before/after SMOTENC, annual MCC/Kappa with a 0.85 line, worst-year score, seven confusion matrices, consolidated confusion matrix, WEKA class metrics, calibration, class changes, and global feature importance.

- [ ] **Step 4: Implement atomic artifact publication and model card**

Write CSV, JSON, joblib, Markdown, and PNG outputs into a temporary sibling folder. Validate every required file and then move files into `reports/modelo_mcc_kappa_cafe/`. Include exact failed years and no production language when the gate fails.

- [ ] **Step 5: Run artifact and full tests**

Run: `python -m pytest tests/test_annual_artifacts.py -v`

Expected: all artifact tests pass.

Run: `python -m pytest -q`

Expected: no regression failures.

- [ ] **Step 6: Commit artifact generation**

```bash
git add src/evaluation/annual_artifacts.py tests/test_annual_artifacts.py
git commit -m "feat: export annual model evidence"
```

---

### Task 6: Replace and execute the presentation notebook

**Files:**
- Create: `scripts/create_mcc_kappa_notebook.py`
- Modify: `pipeline_interactivo_cafe.ipynb`
- Modify: `tests/test_production_notebook.py`
- Create: `tests/test_mcc_kappa_notebook.py`

**Interfaces:**
- Consumes: functions from Tasks 1-5.
- Produces: an executed `pipeline_interactivo_cafe.ipynb` and `reports/modelo_mcc_kappa_cafe/`.

- [ ] **Step 1: Write failing notebook structure tests**

```python
def test_notebook_has_seven_annual_confusion_matrices_and_strict_gate():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "2019-2024: backtesting temporal" in text
    assert "2025: holdout final" in text
    assert "MCC >= 0.85" in text
    assert "Kappa >= 0.85" in text
    assert "08_matrices_confusion_2019_2025.png" in text

def test_notebook_contains_no_execution_errors():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    errors = [o for c in notebook.cells for o in c.get("outputs", []) if o.output_type == "error"]
    assert errors == []
```

- [ ] **Step 2: Run notebook tests and verify RED**

Run: `python -m pytest tests/test_mcc_kappa_notebook.py -v`

Expected: tests fail because the current notebook does not implement the approved annual gate or 2025 holdout.

- [ ] **Step 3: Generate the notebook in reader-facing order**

Use `nbformat` to generate focused markdown and code cells matching all twenty sections from the design. Keep data download separate from cached loading. Seal 2025 before model selection and assert it never appears in inner fitting records.

- [ ] **Step 4: Refresh bounded official data and cache it**

Run: `python -m src.data.cafe_annual_sources --refresh`

Run: `python -m src.data.nasa_power_climate --refresh --start-year 2006 --end-year 2024`

Expected: EVA includes 2007-2025, 2025 has nonzero coffee rows, DIVIPOLA codes are five characters, and climate ends in 2024.

- [ ] **Step 5: Generate and execute notebook top-to-bottom**

Run: `python scripts/create_mcc_kappa_notebook.py`

Run: `python -m jupyter nbconvert --execute --to notebook --inplace pipeline_interactivo_cafe.ipynb --ExecutePreprocessor.timeout=3600`

Expected: exit code 0, no error outputs, and final artifacts exist.

- [ ] **Step 6: Recompute metrics independently**

Run: `python -m pytest tests/test_mcc_kappa_notebook.py tests/test_production_notebook.py -v`

The tests load saved predictions, recompute confusion matrices, MCC, Kappa, annual supports, verify `train_max_year < anio`, and assert 2025 did not influence recipe selection.

- [ ] **Step 7: Inspect every PNG and notebook summary**

Open the generated figures, verify labels and legibility, and compare the notebook `tl;dr` against `metricas_anuales.csv` and `model_manifest.json`. If an observed value differs, fix the generator and re-execute rather than editing outputs manually.

- [ ] **Step 8: Run the complete verification suite**

Run: `python -m pytest -q`

Run: `python -m pip check`

Run: `python -c "import nbformat; n=nbformat.read('pipeline_interactivo_cafe.ipynb',4); nbformat.validate(n); assert not [o for c in n.cells for o in c.get('outputs',[]) if o.output_type=='error']"`

Expected: all tests pass, dependency check reports no broken requirements, and notebook structure validation exits 0.

- [ ] **Step 9: Commit the notebook and evidence**

```bash
git add scripts/create_mcc_kappa_notebook.py pipeline_interactivo_cafe.ipynb tests/test_mcc_kappa_notebook.py tests/test_production_notebook.py reports/modelo_mcc_kappa_cafe
git commit -m "feat: replace notebook with annual MCC Kappa validation"
```

If Git still lacks author identity, leave the changes unstaged and report the exact Git error without configuring an identity.
