"""
Paso 08: Visualización de Resultados y Comparativa de Baselines vs ML
=====================================================================
Genera gráficas para comparar el MAE (Error Absoluto Medio) entre:
- Baselines heurísticos (t-1, media móvil)
- Modelos ML (CatBoost Ablation A, B, C, D)
"""

import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

def plot_performance_comparison():
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    cultivo_file = cultivo.lower().replace(' ', '_')

    print(f"=== Generando Gráficos de Rendimiento ({cultivo.upper()}) ===")
    
    rep_dir = Path("reports")
    tables_dir = rep_dir / "tables"
    fig_dir = rep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar resultados
    try:
        base_df = pd.read_csv(tables_dir / f"resultados_baselines_{cultivo_file}.csv")
        ml_df = pd.read_csv(tables_dir / f"resultados_ml_ablation_{cultivo_file}.csv")
    except FileNotFoundError as e:
        print(f"Error cargando resultados: {e}")
        return
        
    # Estandarizar columnas para graficar
    base_agg = base_df.groupby("Modelo")["MAE"].mean().reset_index()
    
    # Filtrar solo el modelo F (completo) para cada tipo
    ml_hibrido = ml_df[ml_df["Tipo"] == "Híbrido"] if "Tipo" in ml_df.columns else pd.DataFrame()
    ml_puro = ml_df[ml_df["Tipo"] == "ML_Puro"] if "Tipo" in ml_df.columns else ml_df
    
    # Si no hay columna Tipo, formato antiguo
    if ml_hibrido.empty and "Fase" in ml_df.columns:
        ml_names = {
            "A": "CatBoost (Historial)",
            "B": "CatBoost (+Clima)",
            "C": "CatBoost (+Suelos)",
            "D": "CatBoost (+Aptitud UPRA)",
            "E": "CatBoost (+Crédito FINAGRO)",
            "F": "CatBoost (+Precio Internacional)"
        }
        ml_df["Modelo"] = ml_df["Fase"].map(ml_names)
        ml_agg = ml_df.groupby("Modelo", sort=False)["MAE"].mean().reset_index()
        perf_df = pd.concat([base_agg, ml_agg], ignore_index=True)
    else:
        # Nuevo formato: mostrar Baseline, ML Puro (F) e Híbrido (F)
        rows = []
        for _, r in base_agg.iterrows():
            rows.append({"Modelo": r["Modelo"], "MAE": r["MAE"], "tipo": "baseline"})
        
        # ML Puro por fase
        for phase in ["A", "F"]:
            sub = ml_puro[ml_puro["Fase"] == phase]
            if len(sub) > 0:
                label = "CatBoost ML Puro (A: Lags)" if phase == "A" else "CatBoost ML Puro (F: Completo)"
                rows.append({"Modelo": label, "MAE": sub["MAE"].mean(), "tipo": "ml"})
        
        # Híbrido por fase
        for phase in ["A", "F"]:
            sub = ml_hibrido[ml_hibrido["Fase"] == phase]
            if len(sub) > 0:
                label = "Híbrido ML+Baseline (A)" if phase == "A" else "⭐ Híbrido ML+Baseline (F)"
                rows.append({"Modelo": label, "MAE": sub["MAE"].mean(), "tipo": "hibrido"})
        
        perf_df = pd.DataFrame(rows)
    
    # Ordenar por MAE (mejor a peor)
    perf_df = perf_df.sort_values("MAE").reset_index(drop=True)
    
    # 2. Configurar estilo y graficar
    plt.figure(figsize=(11, 6))
    
    color_map = {"baseline": "#3498db", "ml": "#e74c3c", "hibrido": "#2ecc71"}
    colors = [color_map.get(t, "#95a5a6") for t in perf_df.get("tipo", ["ml"]*len(perf_df))]
    
    bars = plt.barh(perf_df["Modelo"], perf_df["MAE"], color=colors)
    
    # Etiquetas de valor
    for p in bars:
        width = p.get_width()
        plt.text(
            width + 0.001,
            p.get_y() + p.get_height() / 2,
            f"{width:.4f} t/ha",
            ha="left",
            va="center",
            fontweight="bold"
        )
        
    plt.title(f"Comparativa de Modelos Predictores de Rendimiento ({cultivo.upper()})\n(Error Absoluto Medio - Menor es Mejor)", fontsize=14, pad=20)
    plt.xlabel("MAE (Toneladas / Hectárea)", fontsize=12)
    plt.ylabel("")
    plt.xlim(0, perf_df["MAE"].max() * 1.2)
    plt.tight_layout()
    
    out_file = fig_dir / f"mae_comparative_{cultivo_file}.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfica MAE guardada en: {out_file}")
    
    # 3. Graficar Feature Importance del Modelo D
    try:
        # Extraemos la importancia del modelo F simulando entrenamiento en todo
        from catboost import CatBoostRegressor
        mart_path = f"data/processed/model_mart_{cultivo_file}.csv"
        mart = pd.read_csv(mart_path).dropna(subset=["rendimiento_t_ha"])
        
        # Features F completas
        feats = [
            "rendimiento_lag_1", "rendimiento_lag_2", "rendimiento_lag_3", 
            "produccion_lag_1", "area_cosechada_lag_1", "area_sembrada_lag_1",
            "media_rendimiento_3y", "tendencia_rendimiento_3y", "variabilidad_rendimiento_3y",
            "precipitacion_acumulada_mm", "dias_lluvia",
            "ph_media", "materia_organica_pct_media", "fosforo_ppm_media", 
            "calcio_meq_media", "magnesio_meq_media", "potasio_meq_media", "salinidad_ds_m_media",
            "aptitud_alta_pct", "aptitud_media_pct", "aptitud_baja_pct", "exclusion_legal_pct", "no_apta_pct",
            "log_credito_total", "credito_promedio_operacion", "num_operaciones_credito",
            "precio_internacional_usd", "cambio_precio_pct", "volatilidad_precio", "rango_precio_ratio"
        ]
        feats = [f for f in feats if f in mart.columns]
        
        model = CatBoostRegressor(iterations=100, learning_rate=0.05, depth=5, verbose=0, random_seed=42)
        model.fit(mart[feats], mart["rendimiento_t_ha"])
        
        imp = pd.DataFrame({
            "Variable": feats,
            "Importancia": model.get_feature_importance()
        }).sort_values("Importancia", ascending=False).head(12)
        
        plt.figure(figsize=(10, 6))
        plt.barh(imp["Variable"], imp["Importancia"], color="#2ecc71")
        plt.title(f"Importancia de Variables (CatBoost - Modelo F Completo para {cultivo.upper()})", fontsize=14, pad=20)
        plt.xlabel("Impacto Relativo (%)", fontsize=12)
        plt.ylabel("")
        plt.tight_layout()
        
        imp_file = fig_dir / f"feature_importance_{cultivo_file}.png"
        plt.savefig(imp_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Gráfica de Importancia guardada en: {imp_file}")
        
    except Exception as e:
        print(f"Error generando feature importance: {e}")

if __name__ == "__main__":
    plot_performance_comparison()
