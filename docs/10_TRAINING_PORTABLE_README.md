# 🚀 Entrenamiento ML BALANCED - Modo Portable

## 📋 Requisitos Previos

Para ejecutar el entrenamiento en otra máquina, necesitas:

### Hardware Mínimo:
- **RAM:** 8-10 GB disponible
- **CPU:** 4+ núcleos
- **Almacenamiento:** 2 GB libres (para datos + modelos)

### Software Requerido:
```bash
Python 3.8 o superior
pip (gestor de paquetes de Python)
```

### Librerías Python:
```
pandas >= 1.3.0
numpy >= 1.20.0
catboost >= 1.0.0
scikit-learn >= 0.24.0
pyyaml >= 5.4
```

---

## ⚡ Instalación Rápida (Todo en Uno)

### Opción A: Script Automático (RECOMENDADO)

```bash
# 1. Coloca la carpeta del proyecto en la otra máquina
# Ejemplo:
#   /home/usuario/proyecto_IngDatos/

# 2. Abre terminal en esa carpeta
cd /ruta/a/proyecto_IngDatos

# 3. Instala dependencias
pip install -r requirements.txt

# 4. Ejecuta el entrenamiento (modo balanced)
bash run_training_balanced.sh
```

**Tiempo estimado:** 45-75 minutos

### Opción B: Instalación Manual

```bash
# 1. Instalar dependencias
pip install pandas numpy catboost scikit-learn pyyaml

# 2. Navegar al proyecto
cd /ruta/a/proyecto_IngDatos

# 3. Ejecutar entrenamiento
python3 run_training_balanced.sh
```

---

## 🎯 Requisitos del Proyecto

Asegúrate de que la carpeta tenga esta estructura:

```
proyecto_IngDatos/
├── config/
│   └── config.yaml           ✅ Necesario
├── data/
│   └── processed/
│       ├── model_mart_arroz.csv
│       ├── model_mart_cacao.csv
│       ├── model_mart_cafe.csv
│       └── model_mart_platano.csv
├── reports/
│   └── tables/               ✅ Se crea automáticamente
├── src/
│   └── models/
│       └── (archivos auxiliares)
├── run_training_balanced.sh  ✅ Este archivo
└── requirements.txt          ✅ Necesario (pip install -r)
```

---

## 📝 Qué Hace "Balanced"

| Característica | Detalle |
|---|---|
| **Grid Search** | Prueba 16 combinaciones de hiperparámetros |
| **Ensemble** | 5 modelos independientes (reduce overfitting) |
| **Validación Cruzada** | Cross-validation en datos de entrenamiento |
| **Salida** | `reports/tables/resultados_rigorous_*.csv` |
| **Tiempo** | 45-75 minutos (depende del hardware) |
| **RAM Usado** | 8-10 GB máximo |
| **Mejora Esperada** | 2-4% vs modelo estándar |

---

## 🚀 Ejecución Paso a Paso

### Paso 1: Verificar que Python está instalado
```bash
python3 --version
# Debe mostrar: Python 3.8+
```

### Paso 2: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 3: Verificar datos
```bash
# Verifica que tienes los archivos de datos
ls -lh data/processed/model_mart_*.csv
# Debe mostrar 4 archivos (arroz, cacao, cafe, platano)
```

### Paso 4: Ejecutar entrenamiento
```bash
# Hacer script ejecutable
chmod +x run_training_balanced.sh

# Ejecutar
./run_training_balanced.sh
```

O en Windows (PowerShell):
```powershell
python3 run_training_balanced.sh
```

---

## 📊 Qué Esperar Durante la Ejecución

```
================================
Entrenamiento ML - Modo BALANCED
================================

✓ Directorio del proyecto: /ruta/proyecto
✓ Python: Python 3.9.x

Verificando librerías requeridas...
✓ pandas
✓ numpy
✓ catboost
✓ scikit-learn

Configuración BALANCED:
  • Grid Search: 4×2×2×2 = 16 combinaciones
  • Ensemble: 5 modelos por año
  • TimeSeriesCV: DESHABILITADO (para velocidad)

  ⏱️  Tiempo estimado: 45-75 minutos
  💾 RAM requerido: 8-10 GB
  🔥 CPU: Máximo uso durante Grid Search

════════════════════════════════════════
INICIANDO ENTRENAMIENTO EN MODO BALANCED
════════════════════════════════════════

📊 Cultivo: CACAO
📁 Datos: data/processed/model_mart_cacao.csv
📅 Años de prueba: [2019, 2020, 2021, 2022, 2023, 2024]

Enriqueciendo features...

Búsqueda de hiperparámetros óptimos...

  🔧 GRID SEARCH: Probando 16 combinaciones...
    [ 1/16] depth=5 lr=0.0100 → MAE=0.0850
    [ 2/16] depth=5 lr=0.0150 → MAE=0.0832
    ...
    [16/16] depth=6 lr=0.0150 → MAE=0.0820

  ✓ Mejor configuración: MAE=0.0820

Evaluación en años de prueba...

  Año 2019:
      Ensemble de 5 modelos...✓
      Baseline MAE: 0.0899 t/ha
      Ensemble MAE: 0.0820 t/ha  ✅ (+8.8%)
      R²: 0.7234

  Año 2020:
      Ensemble de 5 modelos...✓
      Baseline MAE: 0.0912 t/ha
      Ensemble MAE: 0.0841 t/ha  ✅ (+7.8%)
      R²: 0.7156

  ...

════════════════════════════════════════
✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE
════════════════════════════════════════

⏱️  Tiempo total: 62 minutos 45 segundos
📊 Resultados guardados en: reports/tables/resultados_rigorous_*.csv

Próximos pasos:
  1. Revisar los resultados en reports/tables/
  2. Comparar con versión estándar (si quieres): python3 scripts/compare_versions.py
  3. Para generar visualizaciones: python3 src/models/03_plot_results.py
```

---

## 📈 Archivos de Salida

Después de ejecutar, tendrás:

```
reports/tables/
├── resultados_rigorous_arroz.csv
├── resultados_rigorous_cacao.csv
├── resultados_rigorous_cafe.csv
└── resultados_rigorous_platano.csv
```

Cada archivo contiene:
```csv
Año,Método,MAE,MAE_Baseline,Mejora_pct,RMSE,R2,N
2019,Ensemble (5 modelos),0.4612,0.4850,4.9,0.6234,0.6234,403
2020,Ensemble (5 modelos),0.5012,0.5234,4.2,0.6745,0.6456,403
...
```

---

## 🔍 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'catboost'"

**Solución:**
```bash
pip install --upgrade catboost
```

### Error: "No sufficient free memory"

**Significa:** No tienes 8-10 GB de RAM disponible.

**Soluciones:**
1. Cierra otras aplicaciones (navegador, IDE, etc.)
2. Reinicia la máquina
3. Usa una máquina diferente con más RAM
4. Usa la versión estándar (más rápida): `python3 src/models/02_train_ml.py`

### Error: "config.yaml not found"

**Solución:** Ejecuta el script desde la carpeta raíz del proyecto:
```bash
cd /ruta/a/proyecto_IngDatos
./run_training_balanced.sh
```

### Error: "data/processed/model_mart_*.csv not found"

**Solución:** Asegúrate de que tienes los archivos de datos:
```bash
ls -la data/processed/
```

Si faltan archivos, necesitas ejecutar primero el ETL:
```bash
python3 src/data/01_etl_normalizacion_v3.py
```

---

## ⚙️ Personalización Avanzada

### Usar solo UN cultivo

Edita `config/config.yaml`:
```yaml
project:
  cultivo_mvp: "arroz"  # Cambia a: arroz, cacao, cafe, platano
```

Luego ejecuta el script como de costumbre.

### Cambiar años de prueba

En `config/config.yaml`:
```yaml
project:
  años_backtest: [2019, 2020, 2021, 2022, 2023, 2024]
  # Cámbialo a tus años deseados, ej:
  años_backtest: [2020, 2021, 2022, 2023, 2024]
```

### Ejecutar en Background (no bloquea terminal)

En Linux/Mac:
```bash
nohup ./run_training_balanced.sh > training.log 2>&1 &
tail -f training.log  # Ver progreso en tiempo real
```

En Windows PowerShell:
```powershell
Start-Process -NoNewWindow -FilePath python3 -ArgumentList run_training_balanced.sh
```

---

## 📊 Comparar Resultados

Después de ejecutar el training balanced, puedes comparar con la versión estándar:

```bash
python3 scripts/compare_versions.py
```

Mostrará un análisis de:
- MAE por año (Estándar vs Rigurosa)
- Porcentaje de mejora
- Recomendaciones según tu hardware

---

## 💡 Tips para Optimizar

### Optimizar RAM
```bash
# Liberar memoria en Linux
sync; echo 3 > /proc/sys/vm/drop_caches

# En Mac
purge  # Si tienes purge instalado
```

### Monitorear uso de recursos
```bash
# Linux
watch -n 1 free -h
top -o %MEM

# Mac
top -o MEM
```

### Si es muy lento:

Usa la versión estándar en su lugar:
```bash
python3 src/models/02_train_ml.py
```

Es 6-8 veces más rápida pero con resultados casi idénticos.

---

## 🎓 Próximos Pasos

Una vez completado el entrenamiento:

1. **Revisar resultados:**
   ```bash
   cat reports/tables/resultados_rigorous_arroz.csv
   ```

2. **Generar gráficos:**
   ```bash
   python3 src/models/03_plot_results.py
   ```

3. **Calcular ingresos proyectados:**
   ```bash
   python3 src/models/05_rentabilidad.py
   ```

4. **Exportar predicciones:**
   ```bash
   python3 src/models/06_forecast_futuro.py
   ```

---

## 📞 Soporte

Si tienes problemas:

1. **Verifica Python:** `python3 --version` (debe ser 3.8+)
2. **Verifica dependencias:** `pip list` (busca pandas, catboost, etc.)
3. **Revisa los logs:** `tail -f training.log` (si corriste en background)
4. **Comprueba espacio:** `df -h` (necesitas al menos 2 GB libres)
5. **Verifica RAM:** `free -h` (Linux) o `top` (Mac/Windows)

---

## ✅ Checklist Final

Antes de ejecutar:

- [ ] Python 3.8+ instalado
- [ ] `pip install -r requirements.txt` ejecutado
- [ ] 8-10 GB de RAM disponible
- [ ] 2 GB de disco libre
- [ ] Archivos `data/processed/model_mart_*.csv` presentes
- [ ] Archivo `config/config.yaml` presente

¡Listo! Ejecuta:
```bash
bash run_training_balanced.sh
```
