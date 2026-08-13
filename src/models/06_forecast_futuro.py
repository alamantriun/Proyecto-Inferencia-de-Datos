"""
Simulador Ex-Ante a 5 Años (2025-2029)
======================================
Usa el modelo Híbrido entrenado para proyectar el rendimiento y rentabilidad futura.
Lee el cultivo desde config.yaml (Fix Error #8).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

def simulate_future():
    # Fix Error #8: Leer cultivo desde config en vez de hardcodear "cacao"
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    cultivo_file = cultivo.lower().replace(' ', '_')
    trm = config["project"].get("trm_usd_cop", 4000)

    print(f"=== Iniciando Simulación de Futuro (2025-2029) para {cultivo.upper()} ===")
    
    # Cargar el Model Mart del cultivo activo
    mart_path = Path(f"data/processed/model_mart_{cultivo_file}.csv")
    if not mart_path.exists():
        print(f"❌ No se encontró {mart_path}. Ejecuta primero el pipeline.")
        return
        
    df = pd.read_csv(mart_path)
    
    # Enriquecer con Target Encoding (replicando 02_train_ml.py)
    historico = df[df["anio"] < 2019]
    muni_mean = historico.groupby("municipio")["rendimiento_t_ha"].mean()
    depto_mean = historico.groupby("departamento")["rendimiento_t_ha"].mean()
    global_mean = historico["rendimiento_t_ha"].mean()
    
    muni_count = historico.groupby("municipio")["rendimiento_t_ha"].count()
    smoothing = 10
    df["municipio_rend_historico"] = df["municipio"].map(
        (muni_mean * muni_count + global_mean * smoothing) / (muni_count + smoothing)
    ).fillna(global_mean)
    df["depto_rend_historico"] = df["departamento"].map(depto_mean).fillna(global_mean)
    
    # Entrenar el modelo con TODOS los datos disponibles (hasta 2024)
    train_df = df.dropna(subset=["rendimiento_t_ha", "rendimiento_lag_1"])
    
    features = [
        "rendimiento_lag_1", "rendimiento_lag_2", "rendimiento_lag_3", 
        "media_rendimiento_3y", "municipio_rend_historico", "depto_rend_historico",
        "precipitacion_acumulada_mm", "max_dias_secos_consecutivos",
        "precio_internacional_usd", "cambio_precio_pct"
    ]
    features = [f for f in features if f in train_df.columns]
    
    X_train = train_df[features]
    y_train = train_df["rendimiento_t_ha"]
    
    print("Entrenando modelo de proyección base...")
    model = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=4, l2_leaf_reg=5, 
        verbose=0, random_seed=42
    )
    model.fit(X_train, y_train)
    
    # ── Preparar Base 2024 para Simular Hacia Adelante ──
    # Tomar los top 5 municipios por producción del último año disponible
    ultimo_anio = int(df["anio"].max())
    df_ultimo = df[df["anio"] == ultimo_anio].dropna(subset=["rendimiento_t_ha"])
    
    # Calcular producción estimada para rankear
    if "area_cosechada_lag_1" in df_ultimo.columns:
        df_ultimo = df_ultimo.copy()
        df_ultimo["_prod_est"] = df_ultimo["rendimiento_t_ha"] * df_ultimo["area_cosechada_lag_1"].fillna(0)
        top_municipios = df_ultimo.nlargest(5, "_prod_est")["municipio"].tolist()
    else:
        top_municipios = df_ultimo.nlargest(5, "rendimiento_t_ha")["municipio"].tolist()
    
    print(f"  Top 5 municipios seleccionados: {top_municipios}")
    
    base = df[(df["anio"] == ultimo_anio) & (df["municipio"].isin(top_municipios))].copy()
    
    if base.empty:
        print("❌ No hay datos para los municipios seleccionados.")
        return
    
    # Fix Error #1: Guardar valores base de clima ANTES del loop
    # para que cada año aplique el multiplicador sobre la base original, no acumulativo
    precip_base = base["precipitacion_acumulada_mm"].copy() if "precipitacion_acumulada_mm" in base.columns else None
    seco_base = base["max_dias_secos_consecutivos"].copy() if "max_dias_secos_consecutivos" in base.columns else None
    
    scenarios = {
        2025: {"precio": 5500, "clima_desc": "Normal", "precip_mod": 1.0, "seco_mod": 1.0},
        2026: {"precio": 4500, "clima_desc": "Fuerte Sequía (El Niño)", "precip_mod": 0.6, "seco_mod": 1.8},
        2027: {"precio": 3800, "clima_desc": "Normal", "precip_mod": 1.0, "seco_mod": 1.0},
        2028: {"precio": 3500, "clima_desc": "Lluvias Fuertes (La Niña)", "precip_mod": 1.5, "seco_mod": 0.5},
        2029: {"precio": 3500, "clima_desc": "Normal", "precip_mod": 1.0, "seco_mod": 1.0},
    }
    
    resultados = []
    current_data = base.copy()
    
    for anio_futuro in range(2025, 2030):
        print(f"  Simulando {anio_futuro}... (Escenario: {scenarios[anio_futuro]['clima_desc']})")
        
        # 1. Actualizar Lags
        current_data["rendimiento_lag_3"] = current_data["rendimiento_lag_2"]
        current_data["rendimiento_lag_2"] = current_data["rendimiento_lag_1"]
        current_data["rendimiento_lag_1"] = current_data["rendimiento_t_ha"] 
        
        current_data["media_rendimiento_3y"] = current_data[["rendimiento_lag_1", "rendimiento_lag_2", "rendimiento_lag_3"]].mean(axis=1)
        
        # 2. Inyectar Escenario
        # Precio
        current_data["cambio_precio_pct"] = (scenarios[anio_futuro]["precio"] - current_data["precio_internacional_usd"]) / current_data["precio_internacional_usd"].replace(0, 1)
        current_data["precio_internacional_usd"] = scenarios[anio_futuro]["precio"]
        
        # Fix Error #1: Aplicar multiplicador sobre la BASE ORIGINAL, no acumulativo
        if precip_base is not None:
            current_data["precipitacion_acumulada_mm"] = precip_base * scenarios[anio_futuro]["precip_mod"]
        if seco_base is not None:
            current_data["max_dias_secos_consecutivos"] = seco_base * scenarios[anio_futuro]["seco_mod"]
            
        current_data["anio"] = anio_futuro
        
        # 3. Predecir (Híbrido: 70% ML, 30% Baseline)
        X_pred = current_data[features]
        pred_ml = model.predict(X_pred)
        pred_base_vals = current_data["rendimiento_lag_1"].values
        pred_blend = 0.7 * pred_ml + 0.3 * pred_base_vals
        pred_blend = np.clip(pred_blend, 0, None)
        
        current_data["rendimiento_t_ha"] = pred_blend
        
        # Guardar resultados
        for _, row in current_data.iterrows():
            # Fix Error #2: Usar "area_cosechada_lag_1" que sí existe, no "area_cosechada"
            area = row.get("area_cosechada_lag_1", np.nan)
            if pd.isna(area):
                area = row.get("area_cosechada_ha", np.nan)
            if pd.isna(area):
                area = 500  # Fallback conservador (500 ha, no 15000)
                
            prod_est = area * row["rendimiento_t_ha"]
            ingreso_millones = (prod_est * row["precio_internacional_usd"] * trm) / 1e6
            
            resultados.append({
                "Año": anio_futuro,
                "Municipio": row["municipio"],
                "Departamento": row["departamento"],
                "Escenario": scenarios[anio_futuro]["clima_desc"],
                "Rendimiento_t_ha": row["rendimiento_t_ha"],
                "Area_ha": area,
                "Precio_USD": row["precio_internacional_usd"],
                "Ingreso_Millones_COP": ingreso_millones
            })
            
    res_df = pd.DataFrame(resultados)
    
    # ── Graficar ──
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Gráfica de Rendimiento
    fig, ax1 = plt.subplots(figsize=(12, 6))
    for muni in top_municipios:
        data_muni = res_df[res_df["Municipio"] == muni].sort_values("Año")
        if not data_muni.empty:
            line, = ax1.plot(data_muni["Año"], data_muni["Rendimiento_t_ha"], marker='o', linewidth=2.5, label=muni)
            color = line.get_color()
            
            # Anotar % de variación respecto a 2025
            val_2025 = data_muni[data_muni["Año"] == 2025]["Rendimiento_t_ha"].values
            if len(val_2025) > 0 and val_2025[0] > 0:
                base_val = val_2025[0]
                for _, row in data_muni.iterrows():
                    anio = row["Año"]
                    val = row["Rendimiento_t_ha"]
                    pct_change = ((val - base_val) / base_val) * 100
                    if anio == 2026: # Anotar especialmente el año de sequía
                        ax1.annotate(f"{pct_change:+.1f}%", 
                                     (anio, val), 
                                     textcoords="offset points", 
                                     xytext=(0, -15), 
                                     ha='center', 
                                     fontsize=8, 
                                     fontweight='bold', 
                                     color=color,
                                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1, alpha=0.8))
    
    ax1.axvline(x=2026, color='red', linestyle='--', alpha=0.5, label='Shock: Sequía (El Niño)')
    ax1.axvline(x=2028, color='blue', linestyle='--', alpha=0.5, label='Shock: Lluvias (La Niña)')
    
    ax1.set_title(f"Proyección de Rendimiento {cultivo.upper()} (2025-2029)\nCon Indicadores de % de Cambio frente a 2025", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Rendimiento (Toneladas / Hectárea)", fontsize=11)
    ax1.set_xlabel("Año de Proyección", fontsize=11)
    ax1.set_xticks([2025, 2026, 2027, 2028, 2029])
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, title="Municipios / Eventos")
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / f"forecast_rendimiento_{cultivo_file}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfica de Ingresos
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Graficar líneas por municipio en ax1
    for muni in top_municipios:
        data_muni = res_df[res_df["Municipio"] == muni].sort_values("Año")
        if not data_muni.empty:
            line, = ax1.plot(data_muni["Año"], data_muni["Ingreso_Millones_COP"], marker='s', linewidth=2.5, label=f"Municipio: {muni}")
            color = line.get_color()
            
            # Anotar % de variación en ingresos respecto a 2025
            val_2025 = data_muni[data_muni["Año"] == 2025]["Ingreso_Millones_COP"].values
            if len(val_2025) > 0 and val_2025[0] > 0:
                base_val = val_2025[0]
                for _, row in data_muni.iterrows():
                    anio = row["Año"]
                    val = row["Ingreso_Millones_COP"]
                    pct_change = ((val - base_val) / base_val) * 100
                    if anio in [2026, 2029]: # Anotar en los años clave de variación
                        ax1.annotate(f"{pct_change:+.1f}%", 
                                     (anio, val), 
                                     textcoords="offset points", 
                                     xytext=(0, 10 if pct_change >= 0 else -15), 
                                     ha='center', 
                                     fontsize=8, 
                                     fontweight='bold', 
                                     color=color,
                                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1, alpha=0.85))

    ax1.set_title(f"Proyección de Ingresos Brutos (2025-2029) - {cultivo.upper()}\nImpacto Combinado: Clima + Precio Bolsa NY (% Variación vs 2025)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Ingreso Bruto Proyectado (Millones COP)", fontsize=11)
    ax1.set_xlabel("Año", fontsize=11)
    ax1.set_xticks([2025, 2026, 2027, 2028, 2029])
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Eje secundario para el precio de la bolsa de NY
    ax2 = ax1.twinx()
    precios = [scenarios[a]["precio"] for a in range(2025, 2030)]
    line_ny, = ax2.plot(range(2025, 2030), precios, color='black', linestyle='--', linewidth=2.5, marker='^', label="Precio Bolsa NY (USD/t)")
    ax2.set_ylabel("Precio Internacional Bolsa NY (USD / Tonelada)", fontsize=11, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Unificar leyenda de ax1 y ax2 con explicaciones claras de color
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, bbox_to_anchor=(1.12, 1), loc='upper left', frameon=True, title="Leyenda de Variables y Colores")
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"forecast_ingresos_{cultivo_file}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Guardar tabla
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(tables_dir / f"forecast_{cultivo_file}.csv", index=False)
    
    print(f"✓ Simulación completada. Gráficas guardadas en reports/figures/")

if __name__ == "__main__":
    simulate_future()
