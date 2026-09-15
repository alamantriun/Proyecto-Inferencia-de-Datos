"""
Config Flexible: Elige qué técnicas de rigidez quieres usar
==========================================================

En lugar de todas las técnicas simultáneamente, elige las que más te importan:
  • Grid Search (tiempo: +40 min)
  • Ensemble (tiempo: +15 min)
  • TimeSeriesCV (tiempo: +10 min)
  • Aumentar iteraciones (tiempo: +5 min)

Ejemplo uso:
  CONFIG_MODE = "balanced"  # Grid Search + Ensemble
  python3 src/models/02_train_ml_rigorous.py
"""

# ============================================================================
# ELIGE TU MODO DE ENTRENAMIENTO
# ============================================================================

# Opciones disponibles:
#   "fast"           - Solo aumenta iteraciones (5 min extra)
#   "balanced"       - Grid Search + Ensemble (60 min total)
#   "rigorous"       - TODO: Grid Search + TimeSeriesCV + Ensemble (90 min)
#   "custom"         - Define tu propia configuración abajo

CONFIG_MODE = "balanced"  # ← CAMBIA AQUÍ

# ============================================================================
# CONFIGURACIONES PRE-DEFINIDAS
# ============================================================================

PRESETS = {
    "fast": {
        "description": "Rápido + Seguro (5 min adicionales)",
        "grid_search": False,
        "ensemble": False,
        "timeseries_cv": False,
        "catboost_iterations": 1500,
        "catboost_learning_rate": 0.02,
        "catboost_depth": 5,
        "n_ensemble_models": 1,
    },
    
    "balanced": {
        "description": "Buen balance: Calidad vs Tiempo (60 min total)",
        "grid_search": True,
        "ensemble": True,
        "timeseries_cv": False,
        "catboost_iterations": 2000,
        "catboost_learning_rate": 0.015,
        "catboost_depth": 6,
        "n_ensemble_models": 5,
        "grid_search_param_grid": {
            "depth": [5, 6],
            "learning_rate": [0.01, 0.015],
            "l2_leaf_reg": [5, 10],
            "subsample": [0.8, 0.9],
        }
    },
    
    "rigorous": {
        "description": "Máxima rigidez (90 min total, requiere 12 GB RAM)",
        "grid_search": True,
        "ensemble": True,
        "timeseries_cv": True,
        "catboost_iterations": 3000,
        "catboost_learning_rate": 0.01,
        "catboost_depth": 6,
        "n_ensemble_models": 5,
        "timeseries_n_splits": 5,
        "grid_search_param_grid": {
            "depth": [5, 6, 7],
            "learning_rate": [0.005, 0.01, 0.02],
            "l2_leaf_reg": [5, 10, 15],
            "subsample": [0.7, 0.8, 0.9],
        }
    },
    
    "academic": {
        "description": "Para papers/investigación (extremadamente riguroso, 2-3 horas)",
        "grid_search": True,
        "ensemble": True,
        "timeseries_cv": True,
        "catboost_iterations": 5000,
        "catboost_learning_rate": 0.005,
        "catboost_depth": 7,
        "n_ensemble_models": 10,
        "timeseries_n_splits": 10,
        "grid_search_param_grid": {
            "depth": [4, 5, 6, 7, 8],
            "learning_rate": [0.001, 0.005, 0.01, 0.02, 0.03],
            "l2_leaf_reg": [3, 5, 10, 15, 20],
            "subsample": [0.6, 0.7, 0.8, 0.9],
        }
    }
}

# ============================================================================
# CONFIGURACIÓN PERSONALIZADA (si CONFIG_MODE = "custom")
# ============================================================================

CUSTOM_CONFIG = {
    "description": "Mi configuración personalizada",
    "grid_search": True,
    "ensemble": True,
    "timeseries_cv": True,
    "catboost_iterations": 2000,
    "catboost_learning_rate": 0.01,
    "catboost_depth": 6,
    "n_ensemble_models": 3,
    "timeseries_n_splits": 5,
    "grid_search_param_grid": {
        "depth": [5, 6],
        "learning_rate": [0.01, 0.015],
        "l2_leaf_reg": [10],
        "subsample": [0.8],
    }
}

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_config():
    """Retorna la configuración seleccionada"""
    
    if CONFIG_MODE not in PRESETS and CONFIG_MODE != "custom":
        raise ValueError(f"CONFIG_MODE inválido: {CONFIG_MODE}")
    
    if CONFIG_MODE == "custom":
        config = CUSTOM_CONFIG
    else:
        config = PRESETS[CONFIG_MODE]
    
    return config


def print_config_info():
    """Muestra información sobre la configuración seleccionada"""
    
    config = get_config()
    
    print(f"\n{'='*80}")
    print(f"CONFIGURACIÓN SELECCIONADA: {CONFIG_MODE.upper()}")
    print(f"{'='*80}")
    print(f"\n  {config['description']}\n")
    
    print(f"  Técnicas habilitadas:")
    print(f"    • Grid Search:        {'✅' if config['grid_search'] else '❌'}")
    print(f"    • Ensemble:           {'✅' if config['ensemble'] else '❌'}")
    print(f"    • TimeSeriesCV:       {'✅' if config['timeseries_cv'] else '❌'}")
    
    print(f"\n  Hiperparámetros CatBoost:")
    print(f"    • Iteraciones:        {config['catboost_iterations']}")
    print(f"    • Learning Rate:      {config['catboost_learning_rate']}")
    print(f"    • Profundidad (depth): {config['catboost_depth']}")
    
    if config['ensemble']:
        print(f"    • N° Modelos Ensemble: {config['n_ensemble_models']}")
    
    if config['timeseries_cv']:
        print(f"    • TimeSeriesCV Splits: {config['timeseries_n_splits']}")
    
    if config['grid_search']:
        param_grid = config['grid_search_param_grid']
        n_combinations = 1
        for param_values in param_grid.values():
            n_combinations *= len(param_values)
        print(f"    • Grid Search Combinations: {n_combinations}")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

"""
EJEMPLO 1: Solo aumentar iteraciones (rápido)
──────────────────────────────────────────────
CONFIG_MODE = "fast"
python3 src/models/02_train_ml_rigorous.py
→ Tiempo: 15 minutos
→ RAM: 4 GB
→ Mejora esperada: 1-2%


EJEMPLO 2: Versión balanceada (recomendada)
─────────────────────────────────────────────
CONFIG_MODE = "balanced"
python3 src/models/02_train_ml_rigorous.py
→ Tiempo: 60 minutos
→ RAM: 8-10 GB
→ Mejora esperada: 2-4%


EJEMPLO 3: Máxima rigidez (para investigación)
───────────────────────────────────────────────
CONFIG_MODE = "rigorous"
python3 src/models/02_train_ml_rigorous.py
→ Tiempo: 90 minutos
→ RAM: 12 GB
→ Mejora esperada: 3-5%


EJEMPLO 4: Personalizado (solo Grid Search, sin Ensemble)
──────────────────────────────────────────────────────────
CONFIG_MODE = "custom"
CUSTOM_CONFIG = {
    "description": "Solo Grid Search",
    "grid_search": True,
    "ensemble": False,
    "timeseries_cv": False,
    "catboost_iterations": 2000,
    "catboost_learning_rate": 0.01,
    "catboost_depth": 6,
    "n_ensemble_models": 1,
    "grid_search_param_grid": {
        "depth": [5, 6, 7],
        "learning_rate": [0.01, 0.015],
        "l2_leaf_reg": [10],
        "subsample": [0.8],
    }
}
python3 src/models/02_train_ml_rigorous.py
→ Tiempo: 45 minutos
→ RAM: 8 GB
→ Mejora esperada: 2-3%
"""


if __name__ == "__main__":
    print_config_info()
    
    # Mostrar recomendación
    config = get_config()
    
    print(f"Recomendación según tu equipo:\n")
    
    ram_gb = 8  # Pregunta al usuario o detecta
    print(f"  Tu RAM disponible: {ram_gb} GB\n")
    
    if CONFIG_MODE == "fast" or CONFIG_MODE == "balanced":
        print(f"  ✅ Tu hardware es suficiente para {CONFIG_MODE}")
    elif CONFIG_MODE == "rigorous":
        if ram_gb >= 12:
            print(f"  ✅ Tu hardware es óptimo para {CONFIG_MODE}")
        elif ram_gb >= 8:
            print(f"  ⚠️  {CONFIG_MODE} puede funcionar pero podría ser lento")
        else:
            print(f"  ❌ {CONFIG_MODE} requiere más RAM, considera 'balanced'")
    elif CONFIG_MODE == "academic":
        if ram_gb >= 16:
            print(f"  ✅ Tu hardware es adecuado para {CONFIG_MODE}")
        else:
            print(f"  ⚠️  {CONFIG_MODE} es exigente, considera 'rigorous'")
