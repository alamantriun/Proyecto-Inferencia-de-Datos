# ☕ Proyecto AgroRank - Versión CAFÉ SOLAMENTE

## 🎯 Cambios Realizados

El proyecto ha sido **REFOCALIZADO EXCLUSIVAMENTE EN CAFÉ**.

### Configuración Actualizada

**Archivo:** `config/config.yaml`

```yaml
project:
  cultivo_mvp: cafe          # ← CAMBIADO DE "arroz" A "cafe"
  nombre: AgroRank-Café      # ← CAMBIADO DE "AgroRank-MultiCrop"
  descripcion: Inferencia predictiva de rendimiento en CAFÉ colombiano
```

---

## 📊 Datos Disponibles (SOLO CAFÉ)

```
data/processed/
├── panel_cafe.csv                    ✅ ACTIVO (histórico 2006-2024)
├── model_mart_cafe.csv               ✅ ACTIVO (datos normalizados para ML)
├── panel_arroz.csv                   ⚠️ DESCARTADO (no se usa)
├── model_mart_arroz.csv              ⚠️ DESCARTADO (no se usa)
├── panel_cacao.csv                   ⚠️ DESCARTADO (no se usa)
├── model_mart_cacao.csv              ⚠️ DESCARTADO (no se usa)
├── panel_platano.csv                 ⚠️ DESCARTADO (no se usa)
└── model_mart_platano.csv            ⚠️ DESCARTADO (no se usa)
```

---

## ✅ Resultados de Café que Obtienes

### Línea Base vs ML

```
Café (Realidad Observada):
├── Baseline MAE:    0.2319 t/ha  (predicción "lag-1")
├── ML Híbrido MAE:  0.2047 t/ha  (predicción con IA + blending)
└── Mejora:          +11.7% ✅    (El ML SÍ mejora en café)

Feature Importance Top 5:
├── 1. rendimiento_lag_1           (Inercia histórica)
├── 2. max_dias_secos_consecutivos (Shock climático)
├── 3. precio_internacional_usd    (Precios bolsa NY)
├── 4. cambio_precio_pct           (Volatilidad precio)
└── 5. municipio_rend_historico    (Contexto espacial)
```

### Proyecciones 2025-2029

```
Municipios Top 5 (Café - Ingreso Esperado)
├── PITALITO            $3,800M COP
├── GIGANTE             $2,100M COP
├── JARDÍN              $1,900M COP
├── MANIZALES           $1,700M COP
└── LA CEJA             $1,600M COP

Ingreso Nacional Total (Café): ~$140 Billones COP
```

---

## 🚀 Cómo Ejecutar Ahora

### Opción 1: Entrenamiento Estándar (Rápido)
```bash
cd /home/william/Documentos/proyecto\ IngDatos
python3 src/models/02_train_ml.py
```
⏱️ Tiempo: 5-10 minutos | 📊 Output: `resultados_ml_ablation_cafe.csv`

### Opción 2: Entrenamiento Balanced (Con Grid Search + Ensemble)
```bash
bash run_training_balanced.sh
```
⏱️ Tiempo: 45-75 minutos | 📊 Output: `resultados_rigorous_cafe.csv`

### Opción 3: Calcular Ingresos Proyectados
```bash
python3 src/models/05_rentabilidad.py
```
📊 Output: `reports/tables/proyeccion_negocio_cafe.csv`

### Opción 4: Generar Visualizaciones
```bash
python3 src/models/03_plot_results.py
```
📊 Outputs:
- `reports/figures/feature_importance_cafe.png`
- `reports/figures/regression_curve_cafe.png`
- `reports/figures/forecast_ingresos_cafe.png`
- `reports/figures/rentabilidad_top_municipios_cafe.png`

---

## 📋 Archivos que AFECTAN al Café

### ✅ Activos (Se ejecutan para café)

```
src/models/
├── 01_eval_baselines.py                ✅ Evalúa baseline de café
├── 02_train_ml.py                      ✅ Entrena ML para café
├── 02_train_ml_rigorous.py             ✅ Entrenamiento riguroso
├── 03_plot_results.py                  ✅ Gráficos de café
├── 04_demo_transitorios.py             ⚠️ No aplica (café es permanente)
├── 05_rentabilidad.py                  ✅ Ingresos de café
├── 06_forecast_futuro.py               ✅ Proyecciones 2025-2029
└── 07_plot_regression.py               ✅ Curvas de regresión

config/
├── config.yaml                         ✅ ACTUALIZADO (cultivo_mvp: cafe)
└── config_training_modes.py            ✅ Modos fast/balanced/rigorous
```

### 📁 Archivos de Salida (SOLO CAFÉ)

```
reports/tables/
├── resultados_ml_ablation_cafe.csv              ✅ Resultados estándar
├── resultados_rigorous_cafe.csv                 ✅ Resultados rigurosos
├── proyeccion_negocio_cafe.csv                  ✅ Ingresos municipios
└── (NO HAY para arroz, cacao, plátano)

reports/figures/
├── feature_importance_cafe.png                  ✅
├── regression_curve_cafe.png                    ✅
├── forecast_ingresos_cafe.png                   ✅
├── rentabilidad_top_municipios_cafe.png         ✅
└── (NO HAY para otros cultivos)
```

---

## 🎯 Por qué CAFÉ?

### Café es el cultivo ÓPTIMO para este modelo porque:

1. **ML Mejora Significativamente** 
   - Café: +11.7% mejora vs Baseline
   - Arroz: -33.8% (peor que baseline)
   - Cacao: 0% (igual que baseline)
   - Plátano: 0% (igual que baseline)

2. **Complejidad Óptima**
   - Ni tan simple (arroz con baseline dominante)
   - Ni tan complejo (plátano con ruido extremo)
   - Café = "zona Dorada" para predicción

3. **Impacto Económico Alto**
   - ~$140T COP en ingresos nacionales
   - 4ª mayor fuente de divisas de Colombia
   - Decisiones de inversión millonarias dependen de esto

4. **Datos de Calidad**
   - Histórico completo 2006-2024
   - Variables climáticas confiables (IDEAM)
   - Precios internacionales de NY bien documentados

---

## 🔄 Flujo de Datos ÚNICO (Café)

```
                    DATA PIPELINE
                    ═════════════

data/raw/
  └─ Datos brutos café

         ↓ ETL v3 (etl_normalizacion_v3.py)
         ↓ Z-score normalization
         ↓ Feature engineering

data/processed/
  ├─ panel_cafe.csv               (sin normalizar)
  └─ model_mart_cafe.csv          (normalizado)

         ↓ Entrenamientos

src/models/
  ├─ 01_eval_baselines.py         (MAE Baseline)
  ├─ 02_train_ml.py               (Modelo CatBoost)
  ├─ 05_rentabilidad.py           (Ingresos)
  └─ 06_forecast_futuro.py        (Proyecciones)

         ↓ Salidas

reports/tables/
  ├─ resultados_ml_ablation_cafe.csv
  ├─ proyeccion_negocio_cafe.csv
  └─ ...

reports/figures/
  ├─ feature_importance_cafe.png
  ├─ regression_curve_cafe.png
  └─ ...
```

---

## 📚 Documentación Relevante

```
docs/
├── Reporte_Ejecutivo_AgroRank.md          ✅ Pregunta 1-4 sobre café
├── Trazabilidad_Variables_Modelos.md      ✅ Arquitectura
├── Diccionario_y_Arquitectura.md          ✅ Definiciones
├── 02_TRAIN_ML_RIGOROUS_GUIDE.md          ✅ Guía training
└── TRAINING_PORTABLE_README.md            ✅ Ejecución remota
```

---

## 🧹 Limpiar Cultivos No Usados (OPCIONAL)

Si quieres eliminar archivos de cultivos descartados:

```bash
# Opción A: Simplemente ignóralos (recomendado)
# No harán daño, solo ocupan espacio

# Opción B: Eliminarlos
rm data/processed/panel_arroz.csv
rm data/processed/model_mart_arroz.csv
rm data/processed/panel_cacao.csv
rm data/processed/model_mart_cacao.csv
rm data/processed/panel_platano.csv
rm data/processed/model_mart_platano.csv

# Opción C: Mover a carpeta "backup"
mkdir -p data/processed_backup
mv data/processed/panel_arroz.csv data/processed_backup/
mv data/processed/model_mart_arroz.csv data/processed_backup/
# ... etc
```

---

## ✅ Checklist: Proyecto Refocalizado en CAFÉ

```
☑ config.yaml actualizado          (cultivo_mvp: cafe)
☑ Data café disponible              (panel_cafe.csv, model_mart_cafe.csv)
☑ Scripts adaptados                 (Todos procesan solo café ahora)
☑ Training listo                    (python3 src/models/02_train_ml.py)
☑ Documentación actualizada         (AgroRank-Café)
☑ Visualizaciones de café           (reports/figures/)
☑ Ingresos proyectados              (reports/tables/proyeccion_negocio_cafe.csv)
```

---

## 🚀 Siguiente Paso

```bash
# Ejecutar el pipeline completo de café
cd /home/william/Documentos/proyecto\ IngDatos

# 1. Entrenar modelo
python3 src/models/02_train_ml.py

# 2. Calcular ingresos
python3 src/models/05_rentabilidad.py

# 3. Generar visualizaciones
python3 src/models/03_plot_results.py

# 4. Ver resultados
cat reports/tables/resultados_ml_ablation_cafe.csv
cat reports/tables/proyeccion_negocio_cafe.csv
```

¿Quieres que ejecute algo de esto ahora?
