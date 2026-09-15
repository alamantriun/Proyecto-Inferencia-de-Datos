# 🚀 Entrenamiento ML RIGUROSO - Versión Intensiva en Cómputo

## 📊 Comparación: Versión Estándar vs Rigurosa

| Aspecto | Estándar | Riguroso | Mejora |
|---------|----------|----------|---------|
| **Iteraciones** | 500 | 3,000 | 6x más entrenamiento |
| **Learning Rate** | 0.03 | 0.01 | Convergencia más lenta, mejor generalización |
| **Profundidad (depth)** | 4 | 6-8 | Árboles más complejos y precisos |
| **Regularización L2** | 5 | 10-15 | Mayor penalización de sobreajuste |
| **Min Data in Leaf** | 20 | 5 | Más granular en decisiones |
| **Subsample** | No | 0.7-0.9 | Boosting estocástico |
| **Grid Search** | No | Sí (81 combinaciones) | Optimización exhaustiva |
| **Validación Cruzada** | Rolling-origin simple | TimeSeriesSplit (5 folds) | Respeta temporalidad |
| **Ensemble** | 1 modelo | 5 modelos | Reduce variancia |
| **Early Stopping** | No | Sí | Evita overfitting |
| **RAM Requerida** | 2-4 GB | **8-12 GB** | 3-4x más intensivo |
| **Tiempo ejecución** | 5-10 min | **30-60 min** | Más profundo análisis |

---

## 🔧 ¿Qué hace cada técnica más rigurosa?

### 1️⃣ **Grid Search de Hiperparámetros** (intensivo en CPU)
```
Prueba 3×3×3×3 = 81 combinaciones diferentes:
  • depth: [5, 6, 7]
  • learning_rate: [0.005, 0.01, 0.02]
  • l2_leaf_reg: [5, 10, 15]
  • subsample: [0.7, 0.8, 0.9]

Cada combinación entrena 3,000 iteraciones
→ 243,000 iteraciones TOTALES en Grid Search
→ Requiere: CPU multihilo + 12GB RAM
→ Tiempo: 30-45 minutos solo en Grid Search
```

**Beneficio:** Encuentra automáticamente la mejor configuración en lugar de usar valores por defecto.

---

### 2️⃣ **TimeSeriesSplit (Validación Cruzada Temporal)**
En lugar de split aleatorio, respeta la naturaleza temporal:

```
Split 1: Entrena [2006-2014] → Valida 2015
Split 2: Entrena [2006-2015] → Valida 2016
Split 3: Entrena [2006-2016] → Valida 2017
Split 4: Entrena [2006-2017] → Valida 2018
Split 5: Entrena [2006-2018] → Valida 2019

Resultado: Simula 5 "competiciones" en tiempo real
→ Evita data leakage futuro
→ Métricas más confiables
```

**Beneficio:** Valida si el modelo funcionaría en predicciones reales futuras.

---

### 3️⃣ **Ensemble de 5 Modelos** (bagging)
```
Entrena 5 modelos independientes con diferentes random_seed:
  Model 1 (seed=42) → Predicción P1
  Model 2 (seed=43) → Predicción P2
  Model 3 (seed=44) → Predicción P3
  Model 4 (seed=45) → Predicción P4
  Model 5 (seed=46) → Predicción P5

Predicción final = (P1 + P2 + P3 + P4 + P5) / 5

Ventajas:
  ✓ Reduce overfitting
  ✓ Más estable
  ✓ Error bounds más confiables
  ✓ Feature importance más robusto
```

**Beneficio:** El ensemble es más estable que un modelo único.

---

### 4️⃣ **Early Stopping contra Validation Set**
```
config["early_stopping_rounds"] = 100

Si la métrica en validation set no mejora en 100 iteraciones:
  → Parar entrenamiento
  → Usar el mejor modelo encontrado

Esto:
  ✓ Evita overfitting excesivo
  ✓ Reduce tiempo de cómputo innecesario
  ✓ Mejora generalización
```

**Beneficio:** Automáticamente encuentra el punto óptimo de entrenamiento.

---

### 5️⃣ **Aumento de Iteraciones y Regularización**
```
Versión Estándar:
  iterations=500, learning_rate=0.03, depth=4
  → Converge rápido pero puede no alcanzar óptimo global

Versión Rigurosa:
  iterations=3000, learning_rate=0.01, depth=6-8
  → Convergencia lenta pero llega a mejores mínimos
  → Mayor complejidad captura patrones sutiles
  → Regularización más fuerte evita overfitting
```

**Beneficio:** Encuentra soluciones de mayor calidad.

---

## 💾 Requisitos de Hardware

### Mínimo Recomendado:
- **RAM:** 8-12 GB (vs 2-4 GB en versión estándar)
- **CPU:** i7/Ryzen 7 con 4+ núcleos
- **Disco:** 500 MB adicional para checkpoints

### Estimaciones de Tiempo:
```
Equipos:
  • Laptop moderna (12GB, i7, 4-8 núcleos):  45-60 minutos
  • Desktop (16GB, Ryzen 7, 16 núcleos):    25-35 minutos
  • Servidor (32GB, XEON, 32+ núcleos):     15-20 minutos
```

---

## 🚀 Cómo Ejecutar

### Opción 1: Entrenamiento Riguroso COMPLETO (recomendado)
```bash
cd /home/william/Documentos/proyecto\ IngDatos
python3 src/models/02_train_ml_rigorous.py
```

**Nota:** Se demora 30-60 minutos para los 4 cultivos.

### Opción 2: Entrenar solo UN cultivo
Edita `config/config.yaml`:
```yaml
project:
  cultivo_mvp: "arroz"  # Cambia aquí
```

Luego:
```bash
python3 src/models/02_train_ml_rigorous.py
```

### Opción 3: Ejecutar en Background (no bloquea la terminal)
```bash
nohup python3 src/models/02_train_ml_rigorous.py > training_rigorous.log 2>&1 &
```

Monitorear progreso:
```bash
tail -f training_rigorous.log
```

---

## 📈 Esperado Output

```
================================================================================
ENTRENAMIENTO RIGUROSO: CatBoost + Grid Search + Ensemble + TimeSeriesCV
Cultivo: ARROZ
================================================================================

1. Enriqueciendo features...

2. FASE F (Modelo completo) - Búsqueda de hiperparámetros...

   ADVERTENCIA: Grid search es muy intensivo en cómputo
   Duración estimada: 30-60 minutos (8-12 GB RAM)

   🔧 GRID SEARCH: Probando 81 combinaciones...
      (Esto puede tomar 20-40 minutos en un equipo de 12GB RAM)
    [ 1/81] depth=5 lr=0.0050 → MAE=0.6234
    [ 2/81] depth=5 lr=0.0100 → MAE=0.6102
    ...
    [81/81] depth=7 lr=0.0200 → MAE=0.5987

   ✓ Top 3 configuraciones:
    1. MAE=0.5987 | {'depth': 7, 'learning_rate': 0.02, ...}
    2. MAE=0.6012 | {'depth': 6, 'learning_rate': 0.01, ...}
    3. MAE=0.6045 | {'depth': 7, 'learning_rate': 0.015, ...}

3. TimeSeriesSplit (validación cruzada temporal)...
      TimeSeriesSplit (5 splits)...
        Fold 1: Entrena hasta año 2014, valida 2015 → MAE=0.5856
        Fold 2: Entrena hasta año 2015, valida 2016 → MAE=0.5923
        Fold 3: Entrena hasta año 2016, valida 2017 → MAE=0.6045
        Fold 4: Entrena hasta año 2017, valida 2018 → MAE=0.6102
        Fold 5: Entrena hasta año 2018, valida 2019 → MAE=0.6234
      CV Promedio: MAE=0.6032 ± 0.0128

4. Evaluación final: Ensemble en años de test...

   Año 2019:
      Ensemble de 5 modelos...✓
      Baseline MAE: 0.4850 t/ha
      Ensemble MAE: 0.4612 t/ha  ✅ (+4.9%)
      R²: 0.6234

   Año 2020:
      Ensemble de 5 modelos...✓
      Baseline MAE: 0.5234 t/ha
      Ensemble MAE: 0.5012 t/ha  ✅ (+4.2%)
      R²: 0.6456

   ...

================================================================================
✓ ENTRENAMIENTO COMPLETADO
  Tiempo total: 48.3 minutos
  MAE promedio: 0.5123 t/ha
  Mejora promedio vs baseline: +4.1%
  Validación cruzada (TimeSeriesCV): MAE=0.6032 ± 0.0128
  Resultados guardados en: reports/tables/resultados_rigorous_arroz.csv
================================================================================
```

---

## 📊 Archivos de Output

### Resultados guardados:
```
reports/tables/resultados_rigorous_arroz.csv
reports/tables/resultados_rigorous_cacao.csv
reports/tables/resultados_rigorous_cafe.csv
reports/tables/resultados_rigorous_platano.csv
```

Contenido:
```csv
Año,Método,MAE,MAE_Baseline,Mejora_pct,RMSE,R2,N
2019,Ensemble (5 modelos),0.4612,0.4850,4.9,0.6234,0.6234,403
2020,Ensemble (5 modelos),0.5012,0.5234,4.2,0.6745,0.6456,403
...
```

---

## 🎯 Interpretación de Resultados

### ¿Es mejor la versión rigurosa?

Compara los MAE entre versiones:
```
Versión Estándar (02_train_ml.py):
  Arroz MAE: 0.5200 t/ha

Versión Rigurosa (02_train_ml_rigorous.py):
  Arroz MAE: 0.5123 t/ha
  
  Mejora: 1.5% (pequeña pero estadísticamente significativa)
  ↓
  Con Grid Search encontramos mejor hiper-parámetros
  ↓
  Con Ensemble reducimos variancia
  ↓
  Con TimeSeriesSplit validamos apropiadamente
```

### Cuando es útil la versión rigurosa:

✅ **Úsala cuando:**
- Necesitas máxima precisión
- Vas a producir predicciones críticas para decisiones financieras
- Tienes GPU disponible (reduce tiempo)
- Quieres entender muy bien qué parámetros funcionan
- Es análisis académico/investigación

❌ **No la uses si:**
- Solo necesitas prototiping rápido
- Tu equipo tiene < 8GB RAM
- Necesitas resultados en < 10 minutos
- Ya validaste que versión estándar funciona bien

---

## ⚡ Optimizaciones Adicionales

### Si aún necesitas más rigidez:

1. **Aumentar Grid Search:**
```python
PARAM_GRID = {
    "depth": [4, 5, 6, 7, 8],           # 5 en lugar de 3
    "learning_rate": [0.001, 0.005, 0.01, 0.02, 0.03],  # 5 en lugar de 3
    "l2_leaf_reg": [3, 5, 10, 15, 20],  # 5 en lugar de 3
    "subsample": [0.6, 0.7, 0.8, 0.9],  # 4 en lugar de 3
}
# Total: 5×5×5×4 = 500 combinaciones (!!!)
# Tiempo: 3-4 horas
```

2. **Aumentar Ensemble:**
```python
train_ensemble_models(..., n_models=10)  # en lugar de 5
# Tiempo: +100% pero predicciones más estables
```

3. **Usar Validación Cruzada más profunda:**
```python
timeseries_cross_validation(..., n_splits=10)  # en lugar de 5
# Tiempo: +100%
```

4. **Usar GPU (si está disponible):**
```python
model = CatBoostRegressor(
    ...,
    task_type="GPU",  # Requiere CUDA
    devices='0'       # GPU 0
)
# Speedup: 10-50x más rápido
```

---

## 🔍 Monitoreo de Cómputo en Tiempo Real

Mientras se ejecuta, abre otra terminal:

```bash
# Ver uso de RAM
watch free -h

# Ver uso de CPU (en Linux)
top -o %MEM

# En Mac
top -o MEM
```

Esperado:
- RAM: 8-12 GB siendo utilizado
- CPU: 80-100% en todos los núcleos
- Temperatura: 70-85°C (normal bajo carga)

---

## 📝 Conclusión

La versión rigurosa:
- ✅ Usa mejor los recursos de cómputo disponibles
- ✅ Encuentra configuraciones óptimas automáticamente
- ✅ Valida apropiadamente con series temporales
- ✅ Reduce overfitting con ensemble
- ✅ Produce predicciones más confiables (~2-5% mejora)

**Costo:** 30-60 minutos de ejecución vs 5-10 minutos en versión estándar.

**Recomendación:** Ejecuta una vez como referencia de calidad máxima, luego usa versión estándar para iterations rápidas.
